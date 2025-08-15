#!/usr/bin/env python3
"""
evaluate_classification.py

Comprehensive evaluation of trained mammogram classification model.
Provides detailed per-class metrics and performance analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    average_precision_score, roc_auc_score, accuracy_score,
    precision_recall_curve, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
from typing import Dict, List, Tuple

# Import from training script
from train_classification import (
    MammogramClassificationDataset, ClassificationModel, 
    load_byol_model, create_classification_transforms, FINDING_CLASSES
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(checkpoint_path: str, num_classes: int, device: torch.device):
    """Load trained classification model from checkpoint."""
    
    print(f"Loading trained model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get BYOL checkpoint path from config or use default
    byol_checkpoint = checkpoint.get('byol_checkpoint', 'mammogram_byol_best.pth')
    
    # Create model architecture
    model = load_byol_model(byol_checkpoint, num_classes, device)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
    print(f"📊 Training validation AUC: {checkpoint['val_metrics']['mean_auc']:.3f}")
    
    return model, checkpoint


def find_optimal_thresholds(y_true: np.ndarray, y_scores: np.ndarray, 
                          class_names: List[str]) -> Dict[str, float]:
    """Find optimal classification thresholds per class using F1 score."""
    
    optimal_thresholds = {}
    
    for i, class_name in enumerate(class_names):
        # Skip if no positive samples
        if np.sum(y_true[:, i]) == 0:
            optimal_thresholds[class_name] = 0.5
            continue
            
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_scores[:, i])
        
        # Calculate F1 scores
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Find threshold with maximum F1
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        optimal_thresholds[class_name] = optimal_threshold
    
    return optimal_thresholds


def calculate_comprehensive_metrics(y_true: np.ndarray, y_scores: np.ndarray, 
                                   class_names: List[str]) -> Dict:
    """Calculate comprehensive metrics for multi-label classification."""
    
    metrics = {}
    
    # Find optimal thresholds
    optimal_thresholds = find_optimal_thresholds(y_true, y_scores, class_names)
    
    # Calculate metrics per class
    for i, class_name in enumerate(class_names):
        class_metrics = {}
        
        y_true_class = y_true[:, i]
        y_scores_class = y_scores[:, i]
        threshold = optimal_thresholds[class_name]
        y_pred_class = (y_scores_class >= threshold).astype(int)
        
        # Basic metrics
        if np.sum(y_true_class) > 0:  # Only if positive samples exist
            class_metrics['auc'] = roc_auc_score(y_true_class, y_scores_class)
            class_metrics['ap'] = average_precision_score(y_true_class, y_scores_class)
        else:
            class_metrics['auc'] = 0.0
            class_metrics['ap'] = 0.0
        
        # Threshold-based metrics
        class_metrics['optimal_threshold'] = threshold
        class_metrics['accuracy'] = accuracy_score(y_true_class, y_pred_class)
        
        # Handle edge cases for precision, recall, f1
        if np.sum(y_pred_class) > 0:
            class_metrics['precision'] = precision_score(y_true_class, y_pred_class, zero_division=0)
        else:
            class_metrics['precision'] = 0.0
            
        if np.sum(y_true_class) > 0:
            class_metrics['recall'] = recall_score(y_true_class, y_pred_class, zero_division=0)
            class_metrics['sensitivity'] = class_metrics['recall']  # Same thing
        else:
            class_metrics['recall'] = 0.0
            class_metrics['sensitivity'] = 0.0
        
        class_metrics['f1'] = f1_score(y_true_class, y_pred_class, zero_division=0)
        
        # Specificity calculation
        tn = np.sum((y_true_class == 0) & (y_pred_class == 0))
        fp = np.sum((y_true_class == 0) & (y_pred_class == 1))
        class_metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        
        # Sample counts
        class_metrics['positive_samples'] = int(np.sum(y_true_class))
        class_metrics['negative_samples'] = int(len(y_true_class) - np.sum(y_true_class))
        class_metrics['predicted_positive'] = int(np.sum(y_pred_class))
        
        metrics[class_name] = class_metrics
    
    # Overall metrics
    valid_aucs = [metrics[cls]['auc'] for cls in class_names if metrics[cls]['auc'] > 0]
    valid_aps = [metrics[cls]['ap'] for cls in class_names if metrics[cls]['ap'] > 0]
    
    metrics['overall'] = {
        'mean_auc': np.mean(valid_aucs) if valid_aucs else 0.0,
        'mean_ap': np.mean(valid_aps) if valid_aps else 0.0,
        'mean_f1': np.mean([metrics[cls]['f1'] for cls in class_names]),
        'mean_accuracy': np.mean([metrics[cls]['accuracy'] for cls in class_names]),
        'classes_evaluated': len(class_names),
        'total_samples': len(y_true)
    }
    
    return metrics


def run_evaluation(model: nn.Module, dataloader: DataLoader, class_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Run model evaluation on test dataset."""
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    print("Running evaluation on test dataset...")
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            with autocast():
                outputs = model(images)
                probs = torch.sigmoid(outputs)
            
            all_predictions.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    
    # Concatenate all results
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    print(f"✅ Evaluated {len(predictions)} samples")
    
    return targets, predictions


def create_results_table(metrics: Dict, class_names: List[str]) -> pd.DataFrame:
    """Create formatted results table."""
    
    rows = []
    for class_name in class_names:
        class_metrics = metrics[class_name]
        rows.append({
            'Class': class_name,
            'AUC': f"{class_metrics['auc']:.3f}",
            'AP': f"{class_metrics['ap']:.3f}", 
            'F1': f"{class_metrics['f1']:.3f}",
            'Precision': f"{class_metrics['precision']:.3f}",
            'Recall': f"{class_metrics['recall']:.3f}",
            'Specificity': f"{class_metrics['specificity']:.3f}",
            'Threshold': f"{class_metrics['optimal_threshold']:.3f}",
            'Pos_Samples': class_metrics['positive_samples'],
            'Neg_Samples': class_metrics['negative_samples']
        })
    
    # Add overall row
    overall = metrics['overall']
    rows.append({
        'Class': 'OVERALL',
        'AUC': f"{overall['mean_auc']:.3f}",
        'AP': f"{overall['mean_ap']:.3f}",
        'F1': f"{overall['mean_f1']:.3f}",
        'Precision': '-',
        'Recall': '-', 
        'Specificity': '-',
        'Threshold': '-',
        'Pos_Samples': '-',
        'Neg_Samples': '-'
    })
    
    return pd.DataFrame(rows)


def print_performance_analysis(metrics: Dict, class_names: List[str]):
    """Print detailed performance analysis."""
    
    print("\n" + "="*80)
    print("🔬 MAMMOGRAM CLASSIFICATION MODEL EVALUATION")
    print("="*80)
    
    overall = metrics['overall']
    print(f"📊 Overall Performance:")
    print(f"   Mean AUC: {overall['mean_auc']:.3f}")
    print(f"   Mean AP:  {overall['mean_ap']:.3f}")
    print(f"   Mean F1:  {overall['mean_f1']:.3f}")
    print(f"   Samples:  {overall['total_samples']:,}")
    
    # Best performing classes
    class_aucs = [(cls, metrics[cls]['auc']) for cls in class_names if metrics[cls]['auc'] > 0]
    class_aucs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Best Performing Classes (by AUC):")
    for i, (cls, auc) in enumerate(class_aucs[:3]):
        print(f"   {i+1}. {cls}: {auc:.3f}")
    
    print(f"\n⚠️  Challenging Classes (by AUC):")
    for i, (cls, auc) in enumerate(class_aucs[-3:]):
        print(f"   {len(class_aucs)-i}. {cls}: {auc:.3f}")
    
    # Sample distribution analysis
    print(f"\n📈 Sample Distribution:")
    sample_counts = [(cls, metrics[cls]['positive_samples']) for cls in class_names]
    sample_counts.sort(key=lambda x: x[1], reverse=True)
    
    for cls, count in sample_counts:
        percentage = (count / overall['total_samples']) * 100
        print(f"   {cls}: {count:,} ({percentage:.1f}%)")


def save_detailed_results(metrics: Dict, results_table: pd.DataFrame, 
                         output_dir: Path, class_names: List[str]):
    """Save detailed results to files."""
    
    # Save metrics as JSON (convert numpy types to Python types)
    def convert_numpy_types(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        return obj
    
    metrics_serializable = convert_numpy_types(metrics)
    metrics_path = output_dir / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    # Save results table as CSV
    table_path = output_dir / 'evaluation_results.csv'
    results_table.to_csv(table_path, index=False)
    
    print(f"\n💾 Results saved:")
    print(f"   Metrics: {metrics_path}")
    print(f"   Table:   {table_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained mammogram classification model')
    parser.add_argument('--checkpoint', type=str, default='classification_models/best_classification_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_csv', type=str, default='classification_data/tiles_test_metadata.csv',
                       help='Path to test CSV file')
    parser.add_argument('--test_tiles_dir', type=str, default='classification_data/tiles_test',
                       help='Directory containing test tile images')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--class_names', type=str, nargs='+', default=None,
                       help='List of class names (defaults to FINDING_CLASSES)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use default classes if not specified
    class_names = args.class_names or FINDING_CLASSES
    
    print("🔬 Mammogram Classification Model Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Test tiles: {args.test_tiles_dir}")
    print(f"Classes: {len(class_names)}")
    print(f"Device: {DEVICE}")
    
    # Load trained model
    model, checkpoint = load_trained_model(args.checkpoint, len(class_names), DEVICE)
    
    # Create test dataset
    test_transform = create_classification_transforms(512, is_training=False)
    test_dataset = MammogramClassificationDataset(
        args.test_csv, args.test_tiles_dir, class_names, test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"📊 Test dataset: {len(test_dataset)} samples")
    
    # Run evaluation
    y_true, y_pred = run_evaluation(model, test_loader, class_names)
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(y_true, y_pred, class_names)
    
    # Create results table
    results_table = create_results_table(metrics, class_names)
    
    # Print results
    print("\n" + "="*100)
    print("📋 DETAILED RESULTS TABLE")
    print("="*100)
    print(results_table.to_string(index=False))
    
    # Print analysis
    print_performance_analysis(metrics, class_names)
    
    # Save results
    save_detailed_results(metrics, results_table, output_dir, class_names)
    
    print(f"\n🎉 Evaluation completed!")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()