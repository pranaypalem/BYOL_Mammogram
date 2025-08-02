#!/usr/bin/env python3
"""
inference_classification.py

Inference script for the fine-tuned BYOL classification model.
Demonstrates how to load the trained model and make predictions on new images.
"""

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict
import json

from train_byol_mammo import MammogramBYOL
from train_classification import ClassificationModel


def load_classification_model(checkpoint_path: str, device: torch.device):
    """Load the fine-tuned classification model."""
    
    print(f"📥 Loading classification model: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get configuration
    config = checkpoint.get('config', {})
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    
    # Create BYOL model
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
    
    # Get metrics from checkpoint
    val_metrics = checkpoint.get('val_metrics', {})
    epoch = checkpoint.get('epoch', 'unknown')
    
    print(f"✅ Loaded model from epoch {epoch}")
    print(f"📊 Classes: {class_names}")
    if 'mean_auc' in val_metrics:
        print(f"🎯 Validation AUC: {val_metrics['mean_auc']:.4f}")
    
    return model, class_names, config


def create_inference_transform(tile_size: int = 512):
    """Create transforms for inference (no augmentation)."""
    return T.Compose([
        T.Resize((tile_size, tile_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def predict_single_image(model: nn.Module, image_path: str, transform, 
                        class_names: List[str], device: torch.device,
                        threshold: float = 0.5) -> Dict:
    """Make prediction on a single image."""
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]
    
    # Create results
    results = {
        'image_path': str(image_path),
        'predictions': {},
        'binary_predictions': {},
        'max_class': None,
        'max_probability': 0.0
    }
    
    max_prob = 0.0
    max_class = None
    
    for i, class_name in enumerate(class_names):
        prob = float(probabilities[i])
        binary_pred = prob > threshold
        
        results['predictions'][class_name] = prob
        results['binary_predictions'][class_name] = binary_pred
        
        if prob > max_prob:
            max_prob = prob
            max_class = class_name
    
    results['max_class'] = max_class
    results['max_probability'] = max_prob
    
    return results


def predict_batch(model: nn.Module, image_paths: List[str], transform,
                 class_names: List[str], device: torch.device,
                 batch_size: int = 32, threshold: float = 0.5) -> List[Dict]:
    """Make predictions on a batch of images efficiently."""
    
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        # Load and preprocess batch
        batch_tensors = []
        for path in batch_paths:
            image = Image.open(path).convert('RGB')
            tensor = transform(image)
            batch_tensors.append(tensor)
        
        batch_input = torch.stack(batch_tensors).to(device)
        
        # Make predictions
        with torch.no_grad():
            logits = model(batch_input)
            probabilities = torch.sigmoid(logits).cpu().numpy()
        
        # Process results
        for j, path in enumerate(batch_paths):
            probs = probabilities[j]
            
            result = {
                'image_path': str(path),
                'predictions': {},
                'binary_predictions': {},
                'max_class': None,
                'max_probability': 0.0
            }
            
            max_prob = 0.0
            max_class = None
            
            for k, class_name in enumerate(class_names):
                prob = float(probs[k])
                binary_pred = prob > threshold
                
                result['predictions'][class_name] = prob
                result['binary_predictions'][class_name] = binary_pred
                
                if prob > max_prob:
                    max_prob = prob
                    max_class = class_name
            
            result['max_class'] = max_class
            result['max_probability'] = max_prob
            
            results.append(result)
    
    return results


def print_prediction_results(results: List[Dict], top_k: int = 5):
    """Print prediction results in a nice format."""
    
    for i, result in enumerate(results[:top_k]):
        print(f"\n📸 Image {i+1}: {Path(result['image_path']).name}")
        print(f"🏆 Top prediction: {result['max_class']} ({result['max_probability']:.3f})")
        
        print("📊 All probabilities:")
        # Sort by probability
        sorted_preds = sorted(result['predictions'].items(), 
                            key=lambda x: x[1], reverse=True)
        
        for class_name, prob in sorted_preds:
            binary = "✅" if result['binary_predictions'][class_name] else "❌"
            print(f"   {binary} {class_name:15s}: {prob:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Inference with fine-tuned BYOL classification model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to fine-tuned classification model (.pth file)')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to single image for inference')
    parser.add_argument('--images_dir', type=str, default=None,
                       help='Directory containing images for batch inference')
    parser.add_argument('--output_json', type=str, default=None,
                       help='Save results to JSON file')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--tile_size', type=int, default=512,
                       help='Input tile size')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image_path and not args.images_dir:
        parser.error("Must specify either --image_path or --images_dir")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🔍 BYOL Classification Inference")
    print("=" * 40)
    print(f"Device: {device}")
    print(f"Threshold: {args.threshold}")
    
    # Load model
    model, class_names, config = load_classification_model(args.model_path, device)
    
    # Create transform
    transform = create_inference_transform(args.tile_size)
    
    # Get image paths
    if args.image_path:
        image_paths = [args.image_path]
        print(f"📸 Single image inference: {args.image_path}")
    else:
        images_dir = Path(args.images_dir)
        image_paths = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
        print(f"📁 Batch inference: {len(image_paths)} images from {images_dir}")
    
    if len(image_paths) == 0:
        print("❌ No images found!")
        return
    
    # Make predictions
    if len(image_paths) == 1:
        # Single image
        result = predict_single_image(
            model, image_paths[0], transform, class_names, device, args.threshold
        )
        results = [result]
    else:
        # Batch processing
        print(f"🔄 Processing {len(image_paths)} images in batches of {args.batch_size}...")
        results = predict_batch(
            model, image_paths, transform, class_names, device,
            args.batch_size, args.threshold
        )
    
    # Print results
    print(f"\n🎯 INFERENCE RESULTS")
    print("=" * 40)
    print_prediction_results(results)
    
    # Save to JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output_json}")
    
    # Summary statistics
    print(f"\n📊 SUMMARY")
    print("=" * 40)
    print(f"Total images processed: {len(results)}")
    
    # Count predictions per class
    class_counts = {class_name: 0 for class_name in class_names}
    for result in results:
        for class_name, binary_pred in result['binary_predictions'].items():
            if binary_pred:
                class_counts[class_name] += 1
    
    print("Class distribution (above threshold):")
    for class_name, count in class_counts.items():
        percentage = (count / len(results)) * 100
        print(f"  {class_name:15s}: {count:4d} ({percentage:5.1f}%)")


if __name__ == "__main__":
    main()