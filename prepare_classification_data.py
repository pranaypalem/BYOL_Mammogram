#!/usr/bin/env python3
"""
prepare_classification_data.py

Generate tiles from mammogram images using the same logic as BYOL training,
and create labeled datasets for classification training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from scipy import ndimage
from typing import List, Tuple, Dict
import json
from tqdm import tqdm
import argparse

# Import tile generation functions from BYOL training
from train_byol_mammo import (
    is_background_tile, compute_frequency_energy, segment_breast_tissue,
    MIN_BREAST_RATIO, MIN_FREQ_ENERGY, MIN_BREAST_FOR_FREQ, 
    MIN_TILE_INTENSITY, MIN_NON_ZERO_PIXELS
)

# Configuration
TILE_SIZE = 512
TILE_STRIDE = 256

# Finding classes from the VinDr dataset (matches train_classification.py)
FINDING_CLASSES = [
    'Architectural_Distortion', 'Asymmetry', 'Focal_Asymmetry', 
    'Global_Asymmetry', 'Mass', 'Nipple_Retraction', 'No_Finding',
    'Skin_Retraction', 'Skin_Thickening', 'Suspicious_Calcification', 
    'Suspicious_Lymph_Node'
]


def extract_tiles_with_info(image_array: np.ndarray, breast_mask: np.ndarray, 
                           tile_size: int, stride: int) -> List[Tuple]:
    """
    Extract tiles with position and quality information.
    Returns list of (x, y, breast_ratio, freq_energy, selected, reason)
    """
    tiles_info = []
    height, width = image_array.shape[:2]
    
    # Generate all possible tile positions
    y_positions = list(range(0, max(1, height - tile_size + 1), stride))
    x_positions = list(range(0, max(1, width - tile_size + 1), stride))
    
    # Add edge positions if needed
    if y_positions[-1] + tile_size < height:
        y_positions.append(height - tile_size)
    if x_positions[-1] + tile_size < width:
        x_positions.append(width - tile_size)
    
    for y in y_positions:
        for x in x_positions:
            # Extract image tile
            tile_image = image_array[y:y+tile_size, x:x+tile_size]
            
            # Initialize tile info
            selected = False
            reason = "rejected"
            
            # STEP 1: Comprehensive background rejection
            if is_background_tile(tile_image):
                reason = "background_tile"
            else:
                # STEP 2: Intensity-based rejection
                mean_intensity = np.mean(tile_image)
                if mean_intensity < MIN_TILE_INTENSITY:
                    reason = "low_intensity"
                else:
                    # STEP 3: Breast tissue ratio check
                    tile_mask = breast_mask[y:y+tile_size, x:x+tile_size]
                    breast_ratio = np.sum(tile_mask) / (tile_size * tile_size)
                    
                    # STEP 4: Enhanced selection logic
                    freq_energy = compute_frequency_energy(tile_image)
                    
                    if breast_ratio >= MIN_BREAST_RATIO:
                        selected = True
                        reason = "breast_tissue"
                    elif (freq_energy >= MIN_FREQ_ENERGY and 
                          breast_ratio >= MIN_BREAST_FOR_FREQ and 
                          mean_intensity >= MIN_TILE_INTENSITY + 10):
                        selected = True
                        reason = "frequency_energy"
                    else:
                        if freq_energy < MIN_FREQ_ENERGY:
                            reason = "low_freq_energy"
                        else:
                            reason = "low_breast_ratio"
            
            # Default values for rejected tiles
            if not selected:
                if 'breast_ratio' not in locals():
                    tile_mask = breast_mask[y:y+tile_size, x:x+tile_size]
                    breast_ratio = np.sum(tile_mask) / (tile_size * tile_size)
                if 'freq_energy' not in locals():
                    freq_energy = compute_frequency_energy(tile_image)
            
            tiles_info.append((x, y, breast_ratio, freq_energy, selected, reason))
    
    return tiles_info


def check_bbox_tile_overlap(bbox: Tuple[float, float, float, float], 
                          tile_x: int, tile_y: int, tile_size: int) -> Tuple[bool, Tuple, float]:
    """
    Check if bounding box overlaps with tile and compute intersection.
    Returns (overlaps, intersection_bbox_in_tile_coords, overlap_ratio)
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Tile boundaries
    tile_xmin, tile_ymin = tile_x, tile_y
    tile_xmax, tile_ymax = tile_x + tile_size, tile_y + tile_size
    
    # Check for intersection
    if (xmax <= tile_xmin or xmin >= tile_xmax or 
        ymax <= tile_ymin or ymin >= tile_ymax):
        return False, None, 0.0
    
    # Calculate intersection
    intersect_xmin = max(xmin, tile_xmin)
    intersect_ymin = max(ymin, tile_ymin)
    intersect_xmax = min(xmax, tile_xmax)
    intersect_ymax = min(ymax, tile_ymax)
    
    # Convert to tile-relative coordinates
    tile_bbox = (
        intersect_xmin - tile_x,
        intersect_ymin - tile_y,
        intersect_xmax - tile_x,
        intersect_ymax - tile_y
    )
    
    # Calculate overlap ratio (intersection area / bbox area)
    intersect_area = (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin)
    bbox_area = (xmax - xmin) * (ymax - ymin)
    overlap_ratio = intersect_area / bbox_area if bbox_area > 0 else 0.0
    
    return True, tile_bbox, overlap_ratio


def get_finding_classes(row: pd.Series) -> List[str]:
    """Extract finding classes from one-hot encoded columns."""
    findings = []
    for class_name in FINDING_CLASSES:
        if row[class_name] == 1:
            findings.append(class_name)
    return findings


def process_image_tiles(image_path: Path, annotations_df: pd.DataFrame, 
                       tiles_output_dir: Path, min_overlap: float = 0.1) -> List[Dict]:
    """
    Process a single image to generate tiles with labels.
    
    Returns:
        List of tile information dictionaries
    """
    # Load image
    try:
        image = np.array(Image.open(image_path))
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return []
    
    # Extract patient_id and image_id from filename
    image_filename = image_path.name
    if '_' in image_filename:
        patient_id_from_file = image_filename.split('_')[0]
        image_id_from_file = '_'.join(image_filename.split('_')[1:])
        
        # Find matching annotations
        image_annotations = annotations_df[
            (annotations_df['patient_id'] == patient_id_from_file) & 
            (annotations_df['image_id'] == image_id_from_file)
        ]
    else:
        return []
    
    if len(image_annotations) == 0:
        return []
    
    # Get all findings for this image
    findings_info = []
    for _, row in image_annotations.iterrows():
        findings = get_finding_classes(row)
        if 'No_Finding' not in findings and len(findings) > 0:
            bbox = (row['resized_xmin'], row['resized_ymin'], row['resized_xmax'], row['resized_ymax'])
            findings_info.append((bbox, findings))
    
    # Generate breast mask and tiles
    breast_mask = segment_breast_tissue(image)
    tiles_info = extract_tiles_with_info(image, breast_mask, TILE_SIZE, TILE_STRIDE)
    
    # Filter for selected tiles only
    selected_tiles = [(x, y, br, fe, sel, reason) for x, y, br, fe, sel, reason in tiles_info if sel]
    
    # Process each selected tile
    tile_data = []
    for tile_idx, (x, y, breast_ratio, freq_energy, _, reason) in enumerate(selected_tiles):
        # Extract tile image
        tile_image = image[y:y+TILE_SIZE, x:x+TILE_SIZE]
        
        # Create tile filename
        tile_filename = f"{Path(image_filename).stem}_tile_{x:04d}_{y:04d}.png"
        tile_path = tiles_output_dir / tile_filename
        
        # Save tile image
        tile_pil = Image.fromarray(tile_image)
        tile_pil.save(tile_path)
        
        # Initialize labels (all zeros)
        labels = {class_name: 0 for class_name in FINDING_CLASSES}
        has_finding = False
        
        # Check overlap with all findings
        for bbox, findings in findings_info:
            overlaps, _, overlap_ratio = check_bbox_tile_overlap(bbox, x, y, TILE_SIZE)
            if overlaps and overlap_ratio >= min_overlap:
                has_finding = True
                for finding in findings:
                    labels[finding] = 1
        
        # If no findings overlap significantly, this is a "normal" tile
        if not has_finding:
            labels['No_Finding'] = 1
        
        # Create tile record
        tile_record = {
            'tile_path': tile_filename,
            'original_image': str(image_path.name),
            'tile_x': x,
            'tile_y': y,
            'breast_ratio': breast_ratio,
            'freq_energy': freq_energy,
            'has_finding': has_finding,
            **labels  # Add all class labels
        }
        
        tile_data.append(tile_record)
    
    return tile_data


def prepare_classification_data(csv_path: str, images_dir: str, output_dir: str,
                              split: str = 'training', max_images: int = None):
    """
    Prepare classification data from mammogram images and annotations.
    
    Args:
        csv_path: Path to VinDr annotations CSV
        images_dir: Directory containing mammogram images
        output_dir: Output directory for tiles and metadata
        split: Dataset split ('training' or 'test')
        max_images: Maximum number of images to process (for testing)
    """
    
    # Load annotations
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(df)} annotations")
    
    # Filter by split
    if 'split' in df.columns:
        df = df[df['split'] == split]
        print(f"📊 Filtered to {len(df)} annotations for {split} split")
    
    # Setup output directories
    output_path = Path(output_dir)
    tiles_dir = output_path / f"tiles_{split}"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    
    # Get unique images with findings (not just No_Finding)
    df_with_findings = df[df['No_Finding'] == 0].copy()
    unique_images = df_with_findings['image_id'].unique()
    
    if max_images:
        unique_images = unique_images[:max_images]
    
    print(f"🖼️  Processing {len(unique_images)} images with findings")
    
    # Process each image
    all_tile_data = []
    images_processed = 0
    tiles_generated = 0
    
    for image_id in tqdm(unique_images, desc=f"Processing {split} images"):
        # Get patient_id for this image
        img_rows = df_with_findings[df_with_findings['image_id'] == image_id]
        if len(img_rows) == 0:
            continue
            
        patient_id = img_rows.iloc[0]['patient_id']
        full_filename = f"{patient_id}_{image_id}"
        
        # Check if image exists
        images_path = Path(images_dir)
        image_path = images_path / full_filename
        
        if not image_path.exists():
            continue
        
        # Process this image
        tile_data = process_image_tiles(image_path, df, tiles_dir)
        
        if tile_data:
            all_tile_data.extend(tile_data)
            images_processed += 1
            tiles_generated += len(tile_data)
    
    print(f"✅ Processed {images_processed} images")
    print(f"🔲 Generated {tiles_generated} tiles")
    
    # Create DataFrame and save
    tiles_df = pd.DataFrame(all_tile_data)
    
    # Add 'No_Finding' as a class if not present
    if 'No_Finding' not in tiles_df.columns:
        tiles_df['No_Finding'] = 0
        # Set No_Finding=1 for tiles with no other findings
        finding_cols = [col for col in FINDING_CLASSES if col in tiles_df.columns]
        tiles_df.loc[tiles_df[finding_cols].sum(axis=1) == 0, 'No_Finding'] = 1
    
    # Save tiles metadata
    output_csv = output_path / f"tiles_{split}_metadata.csv"
    tiles_df.to_csv(output_csv, index=False)
    
    # Print class distribution
    print(f"\n📊 Class Distribution for {split}:")
    all_classes = FINDING_CLASSES + ['No_Finding']
    for class_name in all_classes:
        if class_name in tiles_df.columns:
            count = tiles_df[class_name].sum()
            percentage = (count / len(tiles_df)) * 100
            print(f"  {class_name}: {count:,} ({percentage:.1f}%)")
    
    # Save summary statistics
    stats = {
        'split': split,
        'images_processed': images_processed,
        'tiles_generated': tiles_generated,
        'tiles_with_findings': int(tiles_df['has_finding'].sum()),
        'class_distribution': {class_name: int(tiles_df[class_name].sum()) 
                             for class_name in all_classes if class_name in tiles_df.columns}
    }
    
    stats_path = output_path / f"{split}_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"💾 Saved metadata to: {output_csv}")
    print(f"💾 Saved statistics to: {stats_path}")
    
    return tiles_df


def main():
    parser = argparse.ArgumentParser(description='Prepare classification data from mammogram images')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to VinDr annotations CSV')
    parser.add_argument('--images_dir', type=str, required=True,
                       help='Directory containing mammogram images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for tiles and metadata')
    parser.add_argument('--split', type=str, choices=['training', 'test'], default='training',
                       help='Dataset split to process')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to process (for testing)')
    
    args = parser.parse_args()
    
    print("🔬 Mammogram Classification Data Preparation")
    print("=" * 50)
    print(f"CSV: {args.csv_path}")
    print(f"Images: {args.images_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Split: {args.split}")
    
    prepare_classification_data(
        args.csv_path,
        args.images_dir,
        args.output_dir,
        args.split,
        args.max_images
    )
    
    print("\n🎉 Data preparation completed!")


if __name__ == "__main__":
    main()