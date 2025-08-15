#!/usr/bin/env python3
"""
inference.py

Inference script for mammogram classification.
Takes a single mammogram image, generates tiles, and performs tile-based voting
to predict findings and mark bounding boxes.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional
import argparse
import json

# Import from training scripts
from train_classification import (
    ClassificationModel, create_classification_transforms, FINDING_CLASSES, TILE_SIZE
)
from train_byol_mammo import (
    MammogramBYOL, is_background_tile, compute_frequency_energy, 
    segment_breast_tissue, MIN_BREAST_RATIO, MIN_FREQ_ENERGY, 
    MIN_BREAST_FOR_FREQ, MIN_TILE_INTENSITY, MIN_NON_ZERO_PIXELS
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TILE_STRIDE = 256


def extract_tiles_with_info(image_array: np.ndarray, breast_mask: np.ndarray, 
                           tile_size: int, stride: int) -> List[Tuple]:
    """Extract tiles with position and quality information."""
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
            
            tiles_info.append((x, y, breast_ratio, freq_energy, selected, reason, tile_image))
    
    return tiles_info


def load_trained_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, List[str]]:
    """Load trained classification model from checkpoint."""
    
    print(f"📥 Loading trained model: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get class names from checkpoint
    class_names = checkpoint.get('class_names', FINDING_CLASSES)
    num_classes = len(class_names)
    
    # Create BYOL backbone (same architecture as training)
    from torchvision import models
    resnet = models.resnet50(weights=None)
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    
    byol_model = MammogramBYOL(
        backbone=backbone,
        input_dim=2048,
        hidden_dim=4096, 
        proj_dim=256
    ).to(device)
    
    # Create classification model
    model = ClassificationModel(byol_model, num_classes).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"🎯 Classes: {class_names}")
    
    return model, class_names


def predict_tiles(model: nn.Module, tiles_data: List, transform, 
                 class_names: List[str], device: torch.device, 
                 confidence_threshold: float = 0.5) -> List[Dict]:
    """Predict findings for each tile."""
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for tile_data in tiles_data:
            x, y, breast_ratio, freq_energy, selected, reason, tile_image = tile_data
            
            if not selected:
                # No predictions for rejected tiles
                predictions.append({
                    'x': x, 'y': y, 'selected': False,
                    'reason': reason, 'predictions': {},
                    'probabilities': {}, 'confidence': 0.0
                })
                continue
            
            # Convert to PIL and apply transform
            tile_pil = Image.fromarray(tile_image).convert('RGB')
            tile_tensor = transform(tile_pil).unsqueeze(0).to(device)
            
            # Get predictions
            outputs = model(tile_tensor)
            probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
            
            # Convert to predictions and confidences
            tile_predictions = {}
            tile_probs = {}
            max_confidence = 0.0
            
            for i, class_name in enumerate(class_names):
                prob = float(probabilities[i])
                tile_probs[class_name] = prob
                
                if prob >= confidence_threshold:
                    tile_predictions[class_name] = 1
                    max_confidence = max(max_confidence, prob)
                else:
                    tile_predictions[class_name] = 0
            
            predictions.append({
                'x': x, 'y': y, 'selected': True,
                'reason': reason, 'predictions': tile_predictions,
                'probabilities': tile_probs, 'confidence': max_confidence,
                'breast_ratio': breast_ratio, 'freq_energy': freq_energy
            })
    
    return predictions


def aggregate_predictions(tile_predictions: List[Dict], class_names: List[str],
                         voting_strategy: str = 'probability_weighted') -> Dict:
    """Aggregate tile predictions into final image prediction."""
    
    # Filter selected tiles only
    selected_tiles = [pred for pred in tile_predictions if pred['selected']]
    
    if not selected_tiles:
        return {
            'predictions': {class_name: 0 for class_name in class_names},
            'probabilities': {class_name: 0.0 for class_name in class_names},
            'confidence': 0.0,
            'num_tiles': 0,
            'voting_strategy': voting_strategy
        }
    
    final_predictions = {}
    final_probabilities = {}
    
    if voting_strategy == 'majority_vote':
        # Simple majority voting
        for class_name in class_names:
            votes = sum(1 for tile in selected_tiles if tile['predictions'].get(class_name, 0) == 1)
            final_predictions[class_name] = 1 if votes > len(selected_tiles) / 2 else 0
            final_probabilities[class_name] = votes / len(selected_tiles)
    
    elif voting_strategy == 'probability_weighted':
        # Weight by tile confidence and breast tissue ratio
        for class_name in class_names:
            weighted_prob = 0.0
            total_weight = 0.0
            
            for tile in selected_tiles:
                prob = tile['probabilities'].get(class_name, 0.0)
                weight = tile['breast_ratio'] * (1 + tile['confidence'])  # Weight by tissue and confidence
                weighted_prob += prob * weight
                total_weight += weight
            
            final_prob = weighted_prob / total_weight if total_weight > 0 else 0.0
            final_probabilities[class_name] = final_prob
            final_predictions[class_name] = 1 if final_prob >= 0.3 else 0  # Lower threshold for weighted
    
    elif voting_strategy == 'max_confidence':
        # Use maximum confidence for each class
        for class_name in class_names:
            max_prob = max(tile['probabilities'].get(class_name, 0.0) for tile in selected_tiles)
            final_probabilities[class_name] = max_prob
            final_predictions[class_name] = 1 if max_prob >= 0.5 else 0
    
    # Overall confidence is maximum probability across all positive predictions
    overall_confidence = max(final_probabilities.values()) if final_probabilities else 0.0
    
    return {
        'predictions': final_predictions,
        'probabilities': final_probabilities,
        'confidence': overall_confidence,
        'num_tiles': len(selected_tiles),
        'voting_strategy': voting_strategy
    }


def generate_bounding_boxes(tile_predictions: List[Dict], class_names: List[str],
                           tile_size: int, min_confidence: float = 0.3) -> List[Dict]:
    """Generate bounding boxes for regions with positive predictions."""
    
    bounding_boxes = []
    
    # Group tiles by predicted class
    for class_name in class_names:
        if class_name == 'No_Finding':
            continue
            
        positive_tiles = []
        for tile_pred in tile_predictions:
            if (tile_pred['selected'] and 
                tile_pred['predictions'].get(class_name, 0) == 1 and
                tile_pred['probabilities'].get(class_name, 0) >= min_confidence):
                positive_tiles.append(tile_pred)
        
        if not positive_tiles:
            continue
        
        # Create bounding boxes for connected regions
        # For simplicity, create individual boxes for each tile
        # In practice, you might want to merge overlapping/adjacent tiles
        
        for tile_pred in positive_tiles:
            bbox = {
                'class': class_name,
                'confidence': tile_pred['probabilities'][class_name],
                'x': tile_pred['x'],
                'y': tile_pred['y'],
                'width': tile_size,
                'height': tile_size,
                'tile_info': {
                    'breast_ratio': tile_pred.get('breast_ratio', 0.0),
                    'freq_energy': tile_pred.get('freq_energy', 0.0)
                }
            }
            bounding_boxes.append(bbox)
    
    # Sort by confidence
    bounding_boxes.sort(key=lambda x: x['confidence'], reverse=True)
    
    return bounding_boxes


def visualize_predictions(image_array: np.ndarray, final_prediction: Dict, 
                         bounding_boxes: List[Dict], output_path: str):
    """Create visualization with predictions and bounding boxes."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image
    ax1.imshow(image_array, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Image with predictions
    ax2.imshow(image_array, cmap='gray')
    
    # Color map for classes
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'pink', 'brown']
    class_colors = {cls: colors[i % len(colors)] for i, cls in enumerate(FINDING_CLASSES)}
    
    # Draw bounding boxes
    for bbox in bounding_boxes:
        if bbox['class'] == 'No_Finding':
            continue
            
        color = class_colors.get(bbox['class'], 'red')
        rect = patches.Rectangle(
            (bbox['x'], bbox['y']), bbox['width'], bbox['height'],
            linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
        )
        ax2.add_patch(rect)
        
        # Add label
        ax2.text(bbox['x'], bbox['y'] - 5, 
                f"{bbox['class']}: {bbox['confidence']:.2f}",
                color=color, fontsize=8, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    
    ax2.set_title('Predictions with Bounding Boxes')
    ax2.axis('off')
    
    # Add legend for predicted classes
    predicted_classes = set(bbox['class'] for bbox in bounding_boxes if bbox['class'] != 'No_Finding')
    if predicted_classes:
        legend_elements = [patches.Patch(facecolor=class_colors[cls], label=cls) 
                          for cls in predicted_classes]
        ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualization saved to: {output_path}")


def run_inference(image_path: str, model_checkpoint: str, output_dir: str = None,
                 confidence_threshold: float = 0.5, voting_strategy: str = 'probability_weighted'):
    """Run inference on a single mammogram image."""
    
    # Setup output directory
    if output_dir is None:
        output_dir = f"inference_results_{Path(image_path).stem}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🔬 Running mammogram inference")
    print(f"📸 Image: {image_path}")
    print(f"🎯 Model: {model_checkpoint}")
    print(f"📁 Output: {output_path}")
    print(f"⚙️  Confidence threshold: {confidence_threshold}")
    print(f"🗳️  Voting strategy: {voting_strategy}")
    
    # Load image
    try:
        image = np.array(Image.open(image_path))
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        print(f"📏 Image shape: {image.shape}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    # Load model
    model, class_names = load_trained_model(model_checkpoint, DEVICE)
    
    # Generate breast mask and tiles
    print("🔍 Generating tiles...")
    breast_mask = segment_breast_tissue(image)
    tiles_info = extract_tiles_with_info(image, breast_mask, TILE_SIZE, TILE_STRIDE)
    
    selected_count = sum(1 for t in tiles_info if t[4])  # Count selected tiles
    print(f"🔲 Generated {len(tiles_info)} tiles ({selected_count} selected)")
    
    # Create transforms
    transform = create_classification_transforms(TILE_SIZE, is_training=False)
    
    # Predict on tiles
    print("🧠 Running tile predictions...")
    tile_predictions = predict_tiles(
        model, tiles_info, transform, class_names, DEVICE, confidence_threshold
    )
    
    # Aggregate predictions
    print("🗳️  Aggregating predictions...")
    final_prediction = aggregate_predictions(tile_predictions, class_names, voting_strategy)
    
    # Generate bounding boxes
    print("📦 Generating bounding boxes...")
    bounding_boxes = generate_bounding_boxes(tile_predictions, class_names, TILE_SIZE)
    
    # Print results
    print(f"\n🎯 PREDICTION RESULTS:")
    print(f"   Overall confidence: {final_prediction['confidence']:.3f}")
    print(f"   Tiles analyzed: {final_prediction['num_tiles']}")
    print(f"   Bounding boxes: {len(bounding_boxes)}")
    
    print(f"\n🏷️  PREDICTED FINDINGS:")
    for class_name in class_names:
        if final_prediction['predictions'].get(class_name, 0) == 1:
            prob = final_prediction['probabilities'][class_name]
            print(f"   ✅ {class_name}: {prob:.3f}")
    
    if not any(final_prediction['predictions'].values()):
        print(f"   ❌ No findings detected")
    
    # Save results
    results = {
        'image_path': str(image_path),
        'model_checkpoint': str(model_checkpoint),
        'image_shape': image.shape,
        'final_prediction': final_prediction,
        'bounding_boxes': bounding_boxes,
        'tile_predictions': tile_predictions,
        'parameters': {
            'confidence_threshold': confidence_threshold,
            'voting_strategy': voting_strategy,
            'tile_size': TILE_SIZE,
            'tile_stride': TILE_STRIDE
        }
    }
    
    results_path = output_path / 'inference_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create visualization
    viz_path = output_path / 'prediction_visualization.png'
    visualize_predictions(image, final_prediction, bounding_boxes, str(viz_path))
    
    print(f"\n💾 Results saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Mammogram classification inference')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to mammogram image')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: inference_results_<image_name>)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for predictions')
    parser.add_argument('--voting_strategy', type=str, 
                       choices=['majority_vote', 'probability_weighted', 'max_confidence'],
                       default='probability_weighted',
                       help='Voting strategy for aggregating tile predictions')
    
    args = parser.parse_args()
    
    results = run_inference(
        args.image,
        args.checkpoint,
        args.output_dir,
        args.confidence_threshold,
        args.voting_strategy
    )
    
    print("\n🎉 Inference completed!")


if __name__ == "__main__":
    main()