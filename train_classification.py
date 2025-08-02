#!/usr/bin/env python3
"""
train_classification.py

Fine-tune the BYOL pre-trained model for multi-label classification on mammogram tiles.
This script loads the BYOL checkpoint and trains only the classification head while
optionally fine-tuning the backbone with a lower learning rate.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from tqdm import tqdm
import wandb
import argparse
from typing import Dict, List, Tuple
import json

# Import the BYOL model
from train_byol_mammo import MammogramBYOL

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TILE_SIZE = 512

# Default hyperparameters - can be overridden via command line
DEFAULT_CONFIG = {
    'batch_size': 32,
    'num_workers': 8,
    'epochs': 50,
    'lr_backbone': 1e-5,      # Lower LR for pre-trained backbone
    'lr_head': 1e-3,          # Higher LR for classification head  
    'weight_decay': 1e-4,
    'warmup_epochs': 5,
    'freeze_backbone_epochs': 10,  # Freeze backbone for first N epochs
    'label_smoothing': 0.1,
    'dropout_rate': 0.3,
    'gradient_clip': 1.0,
}


class MammogramClassificationDataset(Dataset):
    """Dataset for mammogram tile classification with multi-label support."""
    
    def __init__(self, csv_path: str, tiles_dir: str, class_names: List[str], 
                 transform=None, max_samples: int = None):
        """
        Args:
            csv_path: Path to CSV with columns ['tile_path', 'class1', 'class2', ...]
            tiles_dir: Directory containing tile images
            class_names: List of class names (e.g., ['mass', 'calcification', 'normal', etc.])
            transform: Image transformations
            max_samples: Limit dataset size for testing
        """
        self.tiles_dir = Path(tiles_dir)
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.transform = transform
        
        # Load data
        self.df = pd.read_csv(csv_path)
        if max_samples:
            self.df = self.df.head(max_samples)
        
        print(f"📊 Loaded {len(self.df)} samples for classification training")
        print(f"🏷️  Classes: {class_names}")
        
        # Validate required columns
        required_cols = ['tile_path'] + class_names
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.tiles_dir / row['tile_path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get multi-label targets
        labels = torch.tensor([row[class_name] for class_name in self.class_names], 
                             dtype=torch.float32)
        
        return image, labels


def create_classification_transforms(tile_size: int, is_training: bool = True):
    """Create transforms for classification training."""
    
    if is_training:
        # Training transforms - moderate augmentation
        transform = T.Compose([
            T.Resize((tile_size, tile_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=10, fill=0),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        # Validation transforms - no augmentation
        transform = T.Compose([
            T.Resize((tile_size, tile_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    return transform


class ClassificationModel(nn.Module):
    """Classification model that wraps BYOL backbone with classification head."""
    
    def __init__(self, byol_model: MammogramBYOL, num_classes: int, hidden_dim: int = 2048):
        super().__init__()
        self.byol_model = byol_model
        
        # Create classification head
        self.classification_head = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """Forward pass for classification."""
        features = self.byol_model.get_features(x)
        return self.classification_head(features)
    
    def get_features(self, x):
        """Get backbone features."""
        return self.byol_model.get_features(x)


def load_byol_model(checkpoint_path: str, num_classes: int, device: torch.device):
    """Load BYOL pre-trained model and prepare for classification."""
    
    print(f"📥 Loading BYOL checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create BYOL model with same architecture as training
    from torchvision import models
    resnet = models.resnet50(weights=None)  # Don't load ImageNet weights
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    
    byol_model = MammogramBYOL(
        backbone=backbone,
        input_dim=2048,
        hidden_dim=4096, 
        proj_dim=256
    ).to(device)
    
    # Load BYOL weights
    byol_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create classification model
    model = ClassificationModel(byol_model, num_classes).to(device)
    
    print(f"✅ Loaded BYOL model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"📊 BYOL training loss: {checkpoint.get('loss', 'unknown'):.4f}")
    print(f"🎯 Added classification head: 2048 → {2048} → {num_classes}")
    
    return model


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray, 
                     class_names: List[str]) -> Dict[str, float]:
    """Calculate comprehensive metrics for multi-label classification."""
    
    metrics = {}
    
    # Convert probabilities to binary predictions
    pred_binary = (predictions > 0.5).astype(int)
    
    # Per-class metrics
    for i, class_name in enumerate(class_names):
        try:
            # AUC-ROC per class
            auc = roc_auc_score(targets[:, i], predictions[:, i])
            metrics[f'auc_{class_name}'] = auc
            
            # Average Precision per class
            ap = average_precision_score(targets[:, i], predictions[:, i])
            metrics[f'ap_{class_name}'] = ap
            
            # Accuracy per class
            acc = accuracy_score(targets[:, i], pred_binary[:, i])
            metrics[f'acc_{class_name}'] = acc
            
        except ValueError:
            # Handle case where all samples are negative for this class
            metrics[f'auc_{class_name}'] = 0.0
            metrics[f'ap_{class_name}'] = 0.0
            metrics[f'acc_{class_name}'] = accuracy_score(targets[:, i], pred_binary[:, i])
    
    # Overall metrics
    metrics['mean_auc'] = np.mean([metrics[f'auc_{class_name}'] for class_name in class_names])
    metrics['mean_ap'] = np.mean([metrics[f'ap_{class_name}'] for class_name in class_names])
    metrics['mean_acc'] = np.mean([metrics[f'acc_{class_name}'] for class_name in class_names])
    
    # Exact match accuracy (all labels correct)
    exact_match = np.all(pred_binary == targets, axis=1).mean()
    metrics['exact_match_acc'] = exact_match
    
    return metrics


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, scaler: GradScaler, epoch: int,
                config: dict, freeze_backbone: bool = False) -> Dict[str, float]:
    """Train for one epoch."""
    
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    # Freeze backbone if specified
    if freeze_backbone:
        for param in model.byol_model.backbone.parameters():
            param.requires_grad = False
        for param in model.byol_model.backbone_momentum.parameters():
            param.requires_grad = False
    else:
        for param in model.byol_model.backbone.parameters():
            param.requires_grad = True
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d}/{config['epochs']} [Train]", 
                ncols=100, leave=False)
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        with autocast():
            # Forward pass through classification model
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg': f'{total_loss/(batch_idx+1):.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    return {'train_loss': total_loss / num_batches}


def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                  class_names: List[str]) -> Dict[str, float]:
    """Validate for one epoch."""
    
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", ncols=100, leave=False)
        
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Convert outputs to probabilities
            probs = torch.sigmoid(outputs)
            
            all_predictions.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # Concatenate all predictions and targets
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, targets, class_names)
    metrics['val_loss'] = total_loss / len(dataloader)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Fine-tune BYOL model for classification')
    parser.add_argument('--byol_checkpoint', type=str, required=True,
                       help='Path to BYOL checkpoint (.pth file)')
    parser.add_argument('--train_csv', type=str, required=True,
                       help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, required=True,
                       help='Path to validation CSV file') 
    parser.add_argument('--tiles_dir', type=str, required=True,
                       help='Directory containing tile images')
    parser.add_argument('--class_names', type=str, nargs='+', required=True,
                       help='List of class names (e.g., mass calcification normal)')
    parser.add_argument('--output_dir', type=str, default='./classification_results',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--config', type=str, default=None,
                       help='JSON config file (overrides defaults)')
    parser.add_argument('--wandb_project', type=str, default='mammogram-classification',
                       help='Weights & Biases project name')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit dataset size for testing')
    
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    try:
        wandb.init(
            project=args.wandb_project,
            config=config,
            name=f"classification_fine_tune_{len(args.class_names)}classes"
        )
        wandb_enabled = True
    except Exception as e:
        print(f"⚠️  WandB not configured: {e}")
        wandb_enabled = False
    
    print("🔬 BYOL Classification Fine-Tuning")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print(f"Classes: {args.class_names}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Output directory: {output_dir}")
    
    # Load model
    model = load_byol_model(args.byol_checkpoint, len(args.class_names), DEVICE)
    
    # Create datasets
    train_transform = create_classification_transforms(TILE_SIZE, is_training=True)
    val_transform = create_classification_transforms(TILE_SIZE, is_training=False)
    
    train_dataset = MammogramClassificationDataset(
        args.train_csv, args.tiles_dir, args.class_names, 
        train_transform, max_samples=args.max_samples
    )
    
    val_dataset = MammogramClassificationDataset(
        args.val_csv, args.tiles_dir, args.class_names,
        val_transform, max_samples=args.max_samples
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"📊 Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    # Setup loss and optimizer
    # Use BCEWithLogitsLoss for multi-label classification
    pos_weight = None  # Could be calculated from class distribution if needed
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
        label_smoothing=config['label_smoothing']
    )
    
    # Different learning rates for backbone and classification head
    backbone_params = list(model.byol_model.backbone.parameters())
    head_params = list(model.classification_head.parameters())
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config['lr_backbone']},
        {'params': head_params, 'lr': config['lr_head']}
    ], weight_decay=config['weight_decay'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training loop
    best_metric = 0.0
    
    for epoch in range(1, config['epochs'] + 1):
        # Decide whether to freeze backbone
        freeze_backbone = epoch <= config['freeze_backbone_epochs']
        if freeze_backbone:
            print(f"🧊 Epoch {epoch}: Backbone frozen (training only classification head)")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, 
            epoch, config, freeze_backbone
        )
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, args.class_names)
        
        # Step scheduler
        scheduler.step()
        
        # Print metrics
        print(f"\nEpoch {epoch:3d}/{config['epochs']}:")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
        print(f"  Mean AUC:   {val_metrics['mean_auc']:.4f}")
        print(f"  Mean AP:    {val_metrics['mean_ap']:.4f}")
        print(f"  Exact Match: {val_metrics['exact_match_acc']:.4f}")
        
        # Log to wandb
        if wandb_enabled:
            log_dict = {**train_metrics, **val_metrics, 'epoch': epoch}
            wandb.log(log_dict)
        
        # Save best model
        current_metric = val_metrics['mean_auc']
        if current_metric > best_metric:
            best_metric = current_metric
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'config': config,
                'class_names': args.class_names
            }
            torch.save(checkpoint, output_dir / 'best_classification_model.pth')
            print(f"  ✅ New best model saved (AUC: {best_metric:.4f})")
        
        # Save periodic checkpoints
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'config': config,
                'class_names': args.class_names
            }
            torch.save(checkpoint, output_dir / f'classification_epoch_{epoch}.pth')
    
    # Save final model
    final_checkpoint = {
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_metrics': val_metrics,
        'config': config,
        'class_names': args.class_names
    }
    torch.save(final_checkpoint, output_dir / 'final_classification_model.pth')
    
    print(f"\n🎉 Classification training completed!")
    print(f"📊 Best validation AUC: {best_metric:.4f}")
    print(f"💾 Models saved to: {output_dir}")
    
    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()