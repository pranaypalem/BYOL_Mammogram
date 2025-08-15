#!/usr/bin/env python3
"""
validate_model.py

Validate trained classification model on test images.
Provides comprehensive metrics and analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    average_precision_score, roc_auc_score, 
    accuracy_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import json
from typing import Dict, List, Tuple

# Import from training scripts
from train_classification import (
    MammogramClassificationDataset, create_classification_transforms,
    ClassificationModel, calculate_metrics, FINDING_CLASSES, TILE_SIZE
)
from train_byol_mammo import MammogramBYOL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def validate_model(model: nn.Module, dataloader: DataLoader, 
                  class_names: List[str], device: torch.device) -> Dict:
    """Comprehensive model validation."""
    
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    print("🔍 Running validation...")
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            
            all_probabilities.append(probabilities.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    
    # Concatenate all results
    probabilities = np.concatenate(all_probabilities, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    predictions = (probabilities > 0.5).astype(int)
    
    # Calculate comprehensive metrics
    metrics = calculate_metrics(probabilities, targets, class_names)
    
    # Additional metrics
    metrics['hamming_loss'] = np.mean(predictions != targets)
    metrics['subset_accuracy'] = np.mean(np.all(predictions == targets, axis=1))
    
    # Per-class F1 scores
    for i, class_name in enumerate(class_names):
        try:
            f1 = f1_score(targets[:, i], predictions[:, i], average='binary')
            metrics[f'f1_{class_name}'] = f1
        except:
            metrics[f'f1_{class_name}'] = 0.0
    
    metrics['mean_f1'] = np.mean([metrics[f'f1_{class_name}'] for class_name in class_names])
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'probabilities': probabilities,
        'targets': targets
    }


def plot_confusion_matrices(targets: np.ndarray, predictions: np.ndarray, 
                          class_names: List[str], output_dir: Path):
    """Plot confusion matrices for each class."""
    
    n_classes = len(class_names)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, class_name in enumerate(class_names):
        if i >= len(axes):
            break
            
        cm = confusion_matrix(targets[:, i], predictions[:, i])
        
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], 
                   cmap='Blues', square=True,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        axes[i].set_title(f'{class_name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Hide unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(targets: np.ndarray, probabilities: np.ndarray, 
                   class_names: List[str], output_dir: Path):
    """Plot ROC curves for each class."""
    
    plt.figure(figsize=(12, 8))
    
    for i, class_name in enumerate(class_names):
        try:
            auc = roc_auc_score(targets[:, i], probabilities[:, i])
            
            # Calculate ROC curve points
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(targets[:, i], probabilities[:, i])
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
        except:
            continue
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Classes')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_distribution(targets: np.ndarray, class_names: List[str], 
                          output_dir: Path):
    """Plot class distribution in test set."""
    
    class_counts = targets.sum(axis=0)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), class_counts)
    
    # Add percentage labels
    total_samples = len(targets)
    for i, (bar, count) in enumerate(zip(bars, class_counts)):
        percentage = (count / total_samples) * 100
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Test Set')
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_detailed_report(results: Dict, class_names: List[str], 
                           output_dir: Path) -> str:
    """Generate detailed validation report."""
    
    metrics = results['metrics']
    targets = results['targets']
    predictions = results['predictions']
    
    report = []
    report.append("🔬 MAMMOGRAM CLASSIFICATION MODEL VALIDATION REPORT")
    report.append("=" * 60)
    
    # Overall metrics
    report.append("\n📊 OVERALL METRICS:")
    report.append(f"  Mean AUC-ROC:      {metrics['mean_auc']:.4f}")
    report.append(f"  Mean Average Precision: {metrics['mean_ap']:.4f}")
    report.append(f"  Mean Accuracy:     {metrics['mean_acc']:.4f}")
    report.append(f"  Mean F1-Score:     {metrics['mean_f1']:.4f}")
    report.append(f"  Subset Accuracy:   {metrics['subset_accuracy']:.4f}")
    report.append(f"  Hamming Loss:      {metrics['hamming_loss']:.4f}")
    report.append(f"  Exact Match Acc:   {metrics['exact_match_acc']:.4f}")
    
    # Per-class metrics
    report.append("\n🏷️  PER-CLASS METRICS:")
    report.append("-" * 40)
    report.append(f"{'Class':<25} {'AUC':>6} {'AP':>6} {'F1':>6} {'Acc':>6}")
    report.append("-" * 40)
    
    for class_name in class_names:
        auc = metrics.get(f'auc_{class_name}', 0.0)
        ap = metrics.get(f'ap_{class_name}', 0.0)
        f1 = metrics.get(f'f1_{class_name}', 0.0)
        acc = metrics.get(f'acc_{class_name}', 0.0)
        
        report.append(f"{class_name:<25} {auc:>6.3f} {ap:>6.3f} {f1:>6.3f} {acc:>6.3f}")
    
    # Class distribution
    report.append("\n📈 CLASS DISTRIBUTION:")
    report.append("-" * 40)
    total_samples = len(targets)
    
    for i, class_name in enumerate(class_names):
        count = int(targets[:, i].sum())
        percentage = (count / total_samples) * 100
        report.append(f"  {class_name:<25}: {count:>5,} ({percentage:>5.1f}%)")
    
    # Sample predictions analysis
    report.append("\n🔍 SAMPLE ANALYSIS:")
    report.append("-" * 40)
    
    # Count multi-label samples
    multi_label_count = np.sum(targets.sum(axis=1) > 1)
    no_finding_count = np.sum(targets[:, class_names.index('No_Finding')] == 1)
    
    report.append(f"  Total samples:        {total_samples:,}")
    report.append(f"  Multi-label samples:  {multi_label_count:,} ({multi_label_count/total_samples*100:.1f}%)")
    report.append(f"  No finding samples:   {no_finding_count:,} ({no_finding_count/total_samples*100:.1f}%)")
    
    report_text = "\n".join(report)
    
    # Save report
    with open(output_dir / 'validation_report.txt', 'w') as f:
        f.write(report_text)
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description='Validate mammogram classification model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_csv', type=str, required=True,
                       help='Path to test CSV file with tile metadata')
    parser.add_argument('--tiles_dir', type=str, required=True,
                       help='Directory containing test tile images')
    parser.add_argument('--output_dir', type=str, default='./validation_results',
                       help='Output directory for validation results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for validation')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to validate (for testing)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔬 Mammogram Classification Model Validation")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Tiles directory: {args.tiles_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {DEVICE}")
    
    # Load trained model
    model, class_names = load_trained_model(args.checkpoint, DEVICE)
    
    # Create test dataset
    test_transform = create_classification_transforms(TILE_SIZE, is_training=False)
    test_dataset = MammogramClassificationDataset(
        args.test_csv, args.tiles_dir, class_names, 
        test_transform, max_samples=args.max_samples
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"📊 Test dataset size: {len(test_dataset)}")
    
    # Run validation
    results = validate_model(model, test_loader, class_names, DEVICE)
    
    # Generate visualizations
    print("📈 Generating visualizations...")
    plot_confusion_matrices(
        results['targets'], results['predictions'], 
        class_names, output_dir
    )
    
    plot_roc_curves(
        results['targets'], results['probabilities'], 
        class_names, output_dir
    )
    
    plot_class_distribution(
        results['targets'], class_names, output_dir
    )
    
    # Generate detailed report
    print("📝 Generating validation report...")
    report_text = generate_detailed_report(results, class_names, output_dir)
    print(report_text)
    
    # Save metrics as JSON
    metrics_json = {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                   for k, v in results['metrics'].items()}
    
    with open(output_dir / 'validation_metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"\n🎉 Validation completed!")
    print(f"📊 Results saved to: {output_dir}")
    print(f"📈 Mean AUC: {results['metrics']['mean_auc']:.4f}")
    print(f"📈 Mean F1:  {results['metrics']['mean_f1']:.4f}")


if __name__ == "__main__":
    main()