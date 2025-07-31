#!/usr/bin/env python3
"""
prepare_dataset.py

Organize VinDr mammography dataset by splitting images into training/test folders
based on the CSV annotations.

Usage:
    python prepare_dataset.py --csv_path /path/to/annotations.csv --images_root /path/to/images --output_dir ./split_images
"""

import argparse
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm


# Expected CSV columns
COLUMNS = [
    "study_id",
    "series_id", 
    "image_id",
    "laterality",
    "view_position",
    "height",
    "width",
    "breast_birads",
    "breast_density",
    "split",
]


def resolve_source_path(images_root: Path, study_id: str, image_id: str) -> Path | None:
    """
    Try multiple filename patterns to find the source image.
    
    Args:
        images_root: Root directory containing images
        study_id: Study ID from CSV
        image_id: Image ID from CSV
    
    Returns:
        Path to the image file if found, None otherwise
    """
    # Try direct path
    p = images_root / study_id / image_id
    if p.exists():
        return p

    # Try with .png extension
    p_png = (images_root / study_id / image_id).with_suffix(".png")
    if p_png.exists():
        return p_png

    # Try any extension with same name
    candidates = list((images_root / study_id).glob(f"{image_id}.*"))
    if candidates:
        return candidates[0]

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Prepare VinDr mammography dataset by splitting into train/test folders"
    )
    parser.add_argument(
        "--csv_path", 
        type=str, 
        required=True,
        help="Path to breast-level_annotations.csv file"
    )
    parser.add_argument(
        "--images_root", 
        type=str, 
        required=True,
        help="Root directory containing the mammogram images"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./split_images",
        help="Output directory for split images (default: ./split_images)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    csv_path = Path(args.csv_path)
    images_root = Path(args.images_root)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if not images_root.exists():
        raise FileNotFoundError(f"Images root directory not found: {images_root}")
    
    print(f"📊 Loading annotations from: {csv_path}")
    print(f"🖼️  Loading images from: {images_root}")
    print(f"📁 Output directory: {output_dir}")
    
    # Load CSV data
    try:
        df = pd.read_csv(csv_path, usecols=COLUMNS)
        print(f"✅ Loaded {len(df)} annotations")
        print(f"   Columns: {list(df.columns)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV: {e}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    total_copied = 0
    total_missing = 0
    
    for split_name, group in df.groupby("split"):
        dest_dir = output_dir / split_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📋 Processing '{split_name}' split: {len(group)} images")
        print(f"   → Destination: {dest_dir}")
        
        copied_count = 0
        missing_count = 0
        
        # Process each image in this split
        for row in tqdm(group.itertuples(index=False), total=len(group), desc=f"Copying {split_name}"):
            study_id = str(row.study_id)
            image_id = str(row.image_id)
            
            # Find source image
            src_path = resolve_source_path(images_root, study_id, image_id)
            if src_path is None:
                tqdm.write(f"⚠️  Missing: study_id={study_id}, image_id={image_id}")
                missing_count += 1
                continue
            
            # Create destination filename
            dst_path = dest_dir / f"{study_id}_{src_path.name}"
            
            # Copy file
            try:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            except Exception as e:
                tqdm.write(f"❌ Error copying {src_path} → {dst_path}: {e}")
                missing_count += 1
        
        print(f"   ✅ Copied: {copied_count} images")
        print(f"   ⚠️  Missing: {missing_count} images")
        
        total_copied += copied_count
        total_missing += missing_count
    
    # Summary
    print(f"\n🎉 Dataset preparation complete!")
    print(f"   📊 Total images copied: {total_copied:,}")
    print(f"   ⚠️  Total missing images: {total_missing:,}")
    print(f"   📁 Output directory: {output_dir}")
    
    # Show split breakdown
    for split_dir in output_dir.iterdir():
        if split_dir.is_dir():
            count = len(list(split_dir.glob("*.png")))
            print(f"   └── {split_dir.name}: {count:,} images")


if __name__ == "__main__":
    main()