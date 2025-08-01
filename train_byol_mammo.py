#!/usr/bin/env python3
"""
train_byol_mammo.py

Self‑supervised BYOL pre‑training with a ResNet50 backbone on
BREAST TISSUE TILES from mammogram images with intelligent segmentation.
"""

import copy
from pathlib import Path
import time
from typing import List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from torchvision import models
import numpy as np
import cv2
from scipy import ndimage
from tqdm import tqdm
import wandb

# Lightly imports for BYOL
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLProjectionHead, BYOLPredictionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule


# 1) Configuration
DATA_DIR          = "./split_images/training"
BATCH_SIZE        = 8            # Small batch size optimized for A100 memory
NUM_WORKERS       = 8
EPOCHS            = 100
LR                = 3e-4         # Optimized for small batch size (batch=8)
MOMENTUM_BASE     = 0.996
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WANDB_PROJECT     = "mammogram-byol"

# Tile settings - preserve full resolution
TILE_SIZE         = 256          # px - maintain medical detail
TILE_STRIDE       = 128          # px (50% overlap)
MIN_BREAST_RATIO  = 0.1          # Lowered for micro-calcifications in peripheral regions
MIN_FREQ_ENERGY   = 0.01         # Minimum high-frequency energy for calcification detection
MIN_BREAST_FOR_FREQ = 0.05       # Minimum breast tissue required for frequency-based selection

# Model settings for classification readiness
HIDDEN_DIM        = 4096
PROJ_DIM          = 256
INPUT_DIM         = 2048


def compute_frequency_energy(image_patch: np.ndarray) -> float:
    """
    Compute high-frequency energy using Laplacian of Gaussian (LoG) 
    to detect micro-calcifications and other high-frequency structures.
    """
    if len(image_patch.shape) == 3:
        gray = cv2.cvtColor(image_patch, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_patch.copy()
    
    # Apply Laplacian of Gaussian for high-frequency detection
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 1.0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
    
    # Compute energy (normalized variance of high-frequency components)
    energy = np.var(laplacian) / (np.mean(gray) + 1e-8)  # Normalized by intensity
    return float(energy)


def segment_breast_tissue(image_array: np.ndarray) -> np.ndarray:
    """
    Segment breast tissue from mammogram using morphological operations.
    Preserves full resolution for medical accuracy.
    """
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array.copy()
    
    # Gentle blur to preserve medical details
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Otsu thresholding for breast tissue segmentation
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Minimal morphological operations to preserve detail
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Fill holes
    filled = ndimage.binary_fill_holes(opened).astype(np.uint8) * 255
    
    # Keep largest connected component (main breast tissue)
    num_labels, labels = cv2.connectedComponents(filled)
    if num_labels > 1:
        largest_label = 1 + np.argmax([np.sum(labels == i) for i in range(1, num_labels)])
        mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        mask = filled
    
    # Gentle closing to smooth boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask > 0


class BreastTileMammoDataset(Dataset):
    """Produces breast tissue tiles from mammograms with intelligent segmentation."""
    
    def __init__(self, root: str, tile_size: int, stride: int, min_breast_ratio: float = 0.1, min_freq_energy: float = 0.01, min_breast_for_freq: float = 0.05, transform=None):
        self.transform = transform
        self.tile_size = tile_size
        self.stride = stride
        self.min_breast_ratio = min_breast_ratio
        self.min_freq_energy = min_freq_energy
        self.min_breast_for_freq = min_breast_for_freq
        self.tiles = []  # (path, x, y, breast_ratio, freq_energy)
        
        img_paths = list(Path(root).glob("*.png"))
        if len(img_paths) == 0:
            raise RuntimeError(f"No .png files found in {root!r}")
        
        print(f"[Dataset] Processing {len(img_paths)} mammogram images...")
        
        total_tiles = 0
        breast_tiles = 0
        
        for img_path in tqdm(img_paths, desc="Extracting breast tiles"):
            with Image.open(img_path) as img:
                img_array = np.array(img)
            
            # Segment breast tissue
            breast_mask = segment_breast_tissue(img_array)
            
            # Extract tiles from breast regions
            tiles = self._extract_breast_tiles(img_array, breast_mask, img_path)
            self.tiles.extend(tiles)
            
            total_tiles += len(self._get_all_possible_tiles(img_array.shape))
            breast_tiles += len(tiles)
        
        efficiency = (breast_tiles / total_tiles) * 100 if total_tiles > 0 else 0
        print(f"[Dataset] Generated {breast_tiles:,} breast tiles from {total_tiles:,} possible tiles ({efficiency:.1f}% efficiency)")
        print(f"[Dataset] Average breast tissue per tile: {np.mean([t[3] for t in self.tiles]):.1%}")
        print(f"[Dataset] Average frequency energy per tile: {np.mean([t[4] for t in self.tiles]):.4f}")
    
    def _get_all_possible_tiles(self, shape: Tuple) -> List:
        """Get all possible tile positions for efficiency calculation."""
        height, width = shape[:2]
        positions = []
        
        y_positions = list(range(0, max(1, height - self.tile_size + 1), self.stride))
        x_positions = list(range(0, max(1, width - self.tile_size + 1), self.stride))
        
        if y_positions[-1] + self.tile_size < height:
            y_positions.append(height - self.tile_size)
        if x_positions[-1] + self.tile_size < width:
            x_positions.append(width - self.tile_size)
        
        for y in y_positions:
            for x in x_positions:
                positions.append((x, y))
        
        return positions
    
    def _extract_breast_tiles(self, image_array: np.ndarray, breast_mask: np.ndarray, img_path: Path) -> List:
        """Extract tiles containing sufficient breast tissue or high-frequency content (micro-calcifications)."""
        tiles = []
        
        positions = self._get_all_possible_tiles(image_array.shape)
        
        for x, y in positions:
            # Check breast tissue ratio in this tile
            tile_mask = breast_mask[y:y+self.tile_size, x:x+self.tile_size]
            breast_ratio = np.sum(tile_mask) / (self.tile_size * self.tile_size)
            
            # Extract image tile for frequency analysis
            tile_image = image_array[y:y+self.tile_size, x:x+self.tile_size]
            freq_energy = compute_frequency_energy(tile_image)
            
            # Smart tile selection logic:
            # 1. High breast tissue ratio (normal case)
            # 2. High frequency energy BUT only if there's some breast tissue (avoids pure edge artifacts)
            if (breast_ratio >= self.min_breast_ratio or 
                (freq_energy >= self.min_freq_energy and breast_ratio >= self.min_breast_for_freq)):
                tiles.append((img_path, x, y, breast_ratio, freq_energy))
        
        return tiles
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        img_path, x, y, breast_ratio, freq_energy = self.tiles[idx]
        
        with Image.open(img_path) as img:
            # Extract tile while preserving full resolution
            crop = img.crop((x, y, x + self.tile_size, y + self.tile_size))
            
            # Keep as grayscale for medical imaging, convert to RGB by replicating channel
            if crop.mode != 'L':
                crop = crop.convert('L')
            # Convert to RGB by replicating the grayscale channel
            crop = crop.convert('RGB')
        
        # Apply BYOL transformations
        views = self.transform(crop)
        
        return views, breast_ratio  # Return breast ratio for monitoring


class MammogramBYOL(nn.Module):
    """BYOL model adapted for mammogram classification readiness."""
    
    def __init__(self, backbone, input_dim=2048, hidden_dim=4096, proj_dim=256):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(input_dim, hidden_dim, proj_dim)
        self.prediction_head = BYOLPredictionHead(proj_dim, hidden_dim, proj_dim)
        
        # Add classification head for downstream tasks (mass/calcification multi-label)
        self.classification_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2)  # Multi-label classification: [mass, calcification]
        )
        
        # Momentum (target) networks
        self.backbone_momentum = copy.deepcopy(backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)
    
    def forward(self, x):
        """Forward pass for BYOL training."""
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return self.prediction_head(z)
    
    def forward_momentum(self, x):
        """Forward pass through momentum network."""
        h = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(h)
        return z.detach()
    
    def get_features(self, x):
        """Extract features for downstream classification."""
        with torch.no_grad():
            return self.backbone(x).flatten(start_dim=1)
    
    def classify(self, x):
        """Forward pass for classification (after BYOL training)."""
        features = self.get_features(x)
        return self.classification_head(features)


def create_medical_transforms(input_size: int):
    """Create BYOL transforms optimized for medical imaging."""
    import torchvision.transforms as T
    
    # Medical-appropriate transforms for View 1 (lighter augmentations)
    view1_transform = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=7, fill=0),  # Small rotations to preserve anatomy
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),  # Mild brightness/contrast, no color
        T.Resize(input_size, antialias=True),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Grayscale-appropriate normalization for replicated channels
    ])
    
    # Medical-appropriate transforms for View 2 (slightly stronger augmentations)  
    view2_transform = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=7, fill=0),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0, hue=0),  # Slightly stronger
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), fill=0),  # Small translations/scaling
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),  # Very mild blur to preserve details
        T.Resize(input_size, antialias=True),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Grayscale-appropriate normalization for replicated channels
    ])
    
    return BYOLTransform(
        view_1_transform=view1_transform,
        view_2_transform=view2_transform,
    )


def main():
    # Initialize wandb (offline mode if no API key)
    try:
        wandb.init(
            project=WANDB_PROJECT,
            config={
                "tile_size": TILE_SIZE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LR,
                "momentum_base": MOMENTUM_BASE,
                "min_breast_ratio": MIN_BREAST_RATIO,
                "min_freq_energy": MIN_FREQ_ENERGY,
                "min_breast_for_freq": MIN_BREAST_FOR_FREQ,
                "hidden_dim": HIDDEN_DIM,
                "proj_dim": PROJ_DIM,
            }
        )
        wandb_enabled = True
    except Exception as e:
        print(f"⚠️  WandB not configured, running offline. To enable: wandb login")
        wandb_enabled = False
    
    print("🔬 Mammogram BYOL Training")
    print(f"Device: {DEVICE}")
    print(f"Tile size: {TILE_SIZE}x{TILE_SIZE} (medical resolution preserved)")
    print(f"Min breast tissue ratio: {MIN_BREAST_RATIO:.1%}")
    print(f"Min frequency energy: {MIN_FREQ_ENERGY:.3f} (micro-calcification detection)")
    print(f"Min breast for freq selection: {MIN_BREAST_FOR_FREQ:.1%} (avoids edge artifacts)\n")
    
    # Medical-optimized BYOL transforms
    transform = create_medical_transforms(TILE_SIZE)
    
    # Dataset with breast tissue segmentation and micro-calcification detection
    dataset = BreastTileMammoDataset(
        DATA_DIR, TILE_SIZE, TILE_STRIDE, MIN_BREAST_RATIO, MIN_FREQ_ENERGY, MIN_BREAST_FOR_FREQ, transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    
    print(f"📊 Dataset: {len(dataset):,} breast tissue tiles → {len(loader):,} batches")
    
    # Model with classification readiness
    resnet = models.resnet50(weights=None)  # Start from scratch for medical domain
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = MammogramBYOL(backbone, INPUT_DIM, HIDDEN_DIM, PROJ_DIM).to(DEVICE)
    
    criterion = NegativeCosineSimilarity()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)  # Better weight decay for small batches
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()  # Mixed precision training for A100 optimization
    
    print(f"🧠 Model: ResNet50 backbone with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"🎯 Ready for downstream multi-label classification: {INPUT_DIM}→{HIDDEN_DIM//2}→2 [mass, calcification]\n")
    
    # Training loop with progress tracking
    start_time = time.time()
    best_loss = float('inf')
    global_step = 0
    total_steps = EPOCHS * len(loader)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        breast_ratios = []
        
        # Progress bar for epoch
        pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{EPOCHS}", leave=False)
        
        for views, batch_breast_ratios in pbar:
            x0, x1 = views
            x0, x1 = x0.to(DEVICE, non_blocking=True), x1.to(DEVICE, non_blocking=True)
            
            # Per-step momentum update schedule (BYOL best practice)
            momentum = cosine_schedule(global_step, total_steps, MOMENTUM_BASE, 1.0)
            
            # Update momentum networks
            update_momentum(model.backbone, model.backbone_momentum, momentum)
            update_momentum(model.projection_head, model.projection_head_momentum, momentum)
            
            global_step += 1
            
            # Mixed precision forward passes
            with autocast():
                # BYOL forward passes
                p0 = model(x0)
                z1 = model.forward_momentum(x1)
                p1 = model(x1)
                z0 = model.forward_momentum(x0)
                
                # BYOL loss
                loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))
            
            # Mixed precision optimization step
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Unscale before gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Metrics
            epoch_loss += loss.item()
            breast_ratios.extend(batch_breast_ratios.numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Momentum': f'{momentum:.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        scheduler.step()
        
        # Epoch metrics
        avg_loss = epoch_loss / len(loader)
        avg_breast_ratio = np.mean(breast_ratios)
        elapsed = time.time() - start_time
        
        # Log to wandb if enabled
        if wandb_enabled:
            wandb.log({
                "epoch": epoch,
                "loss": avg_loss,
                "momentum": momentum,
                "learning_rate": scheduler.get_last_lr()[0],
                "avg_breast_ratio": avg_breast_ratio,
                "elapsed_hours": elapsed / 3600,
            })
        
        # Console update
        print(f"Epoch {epoch:3d}/{EPOCHS} │ Loss: {avg_loss:.4f} │ Breast: {avg_breast_ratio:.1%} │ Time: {elapsed/60:.1f}min")
        
        # Save best model and periodic checkpoints
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, 'mammogram_byol_best.pth')
        
        if epoch % 10 == 0:
            checkpoint_path = f'mammogram_byol_epoch{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")
    
    # Final save
    final_path = 'mammogram_byol_final.pth'
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_loss,
        'config': wandb.config,
    }, final_path)
    
    total_time = time.time() - start_time
    print(f"\n✅ BYOL pre-training complete!")
    print(f"⏱️  Total time: {total_time/3600:.1f} hours")
    print(f"💾 Final model: {final_path}")
    print(f"🎯 Ready for classification fine-tuning!")
    
    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
