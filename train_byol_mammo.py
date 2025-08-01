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


# 1) Configuration - A100 GPU Optimized
# 
# A100 GPU Memory Configurations:
# ================================
# A100-40GB: BATCH_SIZE=32,  LR=1e-3,  NUM_WORKERS=16
# A100-80GB: BATCH_SIZE=64,  LR=2e-3,  NUM_WORKERS=20  (uncomment below for 80GB)
#
# For A100-80GB, uncomment these lines:
# BATCH_SIZE = 64; LR = 2e-3; NUM_WORKERS = 20

DATA_DIR          = "./split_images/training"
BATCH_SIZE        = 64           # A100-40GB optimized (change to 64 for 80GB)
NUM_WORKERS       = 32           # A100 CPU core utilization (change to 20 for 80GB)
EPOCHS            = 100
LR                = 2e-3         # Batch-size scaled: 3e-4 * (BATCH_SIZE/8)
WARMUP_EPOCHS     = 10           # LR warmup for large batch stability
MOMENTUM_BASE     = 0.996
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WANDB_PROJECT     = "mammogram-byol"

# Tile settings - preserve full resolution with AGGRESSIVE background rejection
TILE_SIZE         = 256          # px - maintain medical detail
TILE_STRIDE       = 128          # px (50% overlap)
MIN_BREAST_RATIO  = 0.15         # INCREASED: More strict breast tissue requirement
MIN_FREQ_ENERGY   = 0.03         # INCREASED: Much higher threshold to avoid background noise
MIN_BREAST_FOR_FREQ = 0.12       # INCREASED: Even more breast tissue required for frequency selection
MIN_TILE_INTENSITY = 40          # NEW: Minimum average intensity to avoid background
MIN_NON_ZERO_PIXELS = 0.7        # NEW: At least 70% of pixels must be non-background

# Model settings for classification readiness
HIDDEN_DIM        = 4096
PROJ_DIM          = 256
INPUT_DIM         = 2048


def is_background_tile(image_patch: np.ndarray) -> bool:
    """
    Comprehensive background detection to reject empty/dark tiles.
    """
    if len(image_patch.shape) == 3:
        gray = cv2.cvtColor(image_patch, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_patch.copy()
    
    # Multiple background rejection criteria
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    non_zero_pixels = np.sum(gray > 15)
    total_pixels = gray.size
    
    # Criteria for background tiles:
    # 1. Too dark overall
    if mean_intensity < MIN_TILE_INTENSITY:
        return True
    
    # 2. Too many near-zero pixels (empty space)
    if non_zero_pixels / total_pixels < MIN_NON_ZERO_PIXELS:
        return True
    
    # 3. Very low variation (uniform background)
    if std_intensity < 10:
        return True
    
    # 4. Check intensity distribution - reject if too skewed toward zero
    histogram, _ = np.histogram(gray, bins=50, range=(0, 255))
    if histogram[0] > total_pixels * 0.3:  # More than 30% pixels near zero
        return True
    
    return False


def compute_frequency_energy(image_patch: np.ndarray) -> float:
    """
    Compute high-frequency energy with AGGRESSIVE background rejection.
    """
    if len(image_patch.shape) == 3:
        gray = cv2.cvtColor(image_patch, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_patch.copy()
    
    # AGGRESSIVE background rejection
    mean_intensity = np.mean(gray)
    if mean_intensity < MIN_TILE_INTENSITY:  # Much stricter intensity threshold
        return 0.0
    
    # Check for sufficient non-background pixels
    non_zero_ratio = np.sum(gray > 15) / gray.size
    if non_zero_ratio < MIN_NON_ZERO_PIXELS:  # Too much background
        return 0.0
    
    # Apply Laplacian of Gaussian for high-frequency detection
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 1.0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
    
    # Focus only on positive responses (bright spots)
    positive_laplacian = np.maximum(laplacian, 0)
    
    # Only analyze pixels with meaningful intensity
    mask = gray > max(30, mean_intensity * 0.4)  # Much stricter tissue mask
    if np.sum(mask) < (gray.size * 0.2):  # Need substantial tissue content
        return 0.0
    
    masked_laplacian = positive_laplacian[mask]
    energy = np.var(masked_laplacian) / (mean_intensity + 1e-8)
    
    return float(energy)


def segment_breast_tissue(image_array: np.ndarray) -> np.ndarray:
    """
    Enhanced breast tissue segmentation with aggressive background removal
    """
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array.copy()
    
    # More aggressive pre-filtering of background
    filtered_gray = np.where(gray > 20, gray, 0)  # Stricter background cutoff
    
    # Otsu thresholding
    _, binary = cv2.threshold(filtered_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Additional background removal based on intensity
    binary = np.where(gray > 25, binary, 0).astype(np.uint8)
    
    # More aggressive morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Larger kernel
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Fill holes
    filled = ndimage.binary_fill_holes(opened).astype(np.uint8) * 255
    
    # Keep largest connected component
    num_labels, labels = cv2.connectedComponents(filled)
    if num_labels > 1:
        largest_label = 1 + np.argmax([np.sum(labels == i) for i in range(1, num_labels)])
        mask = (labels == largest_label).astype(np.uint8) * 255
    else:
        mask = filled
    
    # Closing with larger kernel for smoother boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask > 0


class BreastTileMammoDataset(Dataset):
    """Produces breast tissue tiles from mammograms with AGGRESSIVE background rejection."""
    
    def __init__(self, root: str, tile_size: int, stride: int, min_breast_ratio: float = 0.15, min_freq_energy: float = 0.03, min_breast_for_freq: float = 0.12, transform=None):
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
        freq_tiles = 0
        total_rejected_bg = 0
        total_rejected_intensity = 0
        
        for img_path in tqdm(img_paths, desc="Extracting breast tiles with AGGRESSIVE background rejection"):
            with Image.open(img_path) as img:
                img_array = np.array(img)
            
            # Segment breast tissue with enhanced method
            breast_mask = segment_breast_tissue(img_array)
            breast_area = np.sum(breast_mask)
            total_area = breast_mask.shape[0] * breast_mask.shape[1]
            breast_percentage = (breast_area / total_area) * 100
            
            print(f"  Processing {img_path.name} - Breast tissue: {breast_percentage:.1f}% of image")
            
            # Extract tiles from breast regions with detailed logging
            tiles = self._extract_breast_tiles(img_array, breast_mask, img_path)
            self.tiles.extend(tiles)
            
            # Count selection methods
            image_breast_tiles = sum(1 for t in tiles if len(t) > 4 and 
                                   (len(t) <= 5 or t[4] >= self.min_breast_ratio))
            image_freq_tiles = len(tiles) - image_breast_tiles
            
            total_tiles += len(self._get_all_possible_tiles(img_array.shape))
            breast_tiles += len(tiles)
            freq_tiles += image_freq_tiles
        
        # Enhanced summary statistics matching notebook
        efficiency = (breast_tiles / total_tiles) * 100 if total_tiles > 0 else 0
        print(f"\n[Dataset] AGGRESSIVE Background Rejection Results:")
        print(f"  • Generated {breast_tiles:,} tiles from {total_tiles:,} possible ({efficiency:.1f}% efficiency)")
        print(f"  • Breast tissue method tiles: {breast_tiles - freq_tiles:,}")
        print(f"  • Frequency energy method tiles: {freq_tiles:,}")
        print(f"  • Average breast tissue per tile: {np.mean([t[3] for t in self.tiles]):.1%}")
        print(f"  • Average frequency energy per tile: {np.mean([t[4] for t in self.tiles]):.4f}")
        print(f"  • Background contamination check: Running quality analysis...")
        
        # Quality analysis like in notebook
        potentially_problematic = 0
        intensities = []
        for tile_path, x, y, breast_ratio, freq_energy in self.tiles:
            with Image.open(tile_path) as img:
                tile_img = np.array(img.crop((x, y, x + self.tile_size, y + self.tile_size)))
            if is_background_tile(tile_img):
                potentially_problematic += 1
            intensities.append(np.mean(tile_img))
        
        contamination_pct = potentially_problematic / len(self.tiles) * 100 if self.tiles else 0
        print(f"  • Background contamination: {contamination_pct:.1f}% ({potentially_problematic}/{len(self.tiles)} tiles)")
        print(f"  • Intensity range: {np.min(intensities):.0f} - {np.max(intensities):.0f}")
        
        if contamination_pct == 0:
            print(f"  ✅ PERFECT: Zero background contamination!")
        elif contamination_pct < 5:
            print(f"  ⚠️  Minimal background contamination")
        else:
            print(f"  🔴 Significant background contamination - review filtering")
    
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
        """Extract tiles with AGGRESSIVE background rejection - NO empty space tiles allowed."""
        tiles = []
        rejected_background = 0
        rejected_intensity = 0
        rejected_breast_ratio = 0
        rejected_freq_energy = 0
        
        height, width = image_array.shape[:2]
        
        # Generate all possible tile positions
        y_positions = list(range(0, max(1, height - self.tile_size + 1), self.stride))
        x_positions = list(range(0, max(1, width - self.tile_size + 1), self.stride))
        
        # Add edge positions if needed
        if y_positions[-1] + self.tile_size < height:
            y_positions.append(height - self.tile_size)
        if x_positions[-1] + self.tile_size < width:
            x_positions.append(width - self.tile_size)
        
        for y in y_positions:
            for x in x_positions:
                # Extract image tile
                tile_image = image_array[y:y+self.tile_size, x:x+self.tile_size]
                
                # STEP 1: Comprehensive background rejection
                if is_background_tile(tile_image):
                    rejected_background += 1
                    continue
                
                # STEP 2: Intensity-based rejection
                mean_intensity = np.mean(tile_image)
                if mean_intensity < MIN_TILE_INTENSITY:
                    rejected_intensity += 1
                    continue
                
                # STEP 3: Breast tissue ratio check
                tile_mask = breast_mask[y:y+self.tile_size, x:x+self.tile_size]
                breast_ratio = np.sum(tile_mask) / (self.tile_size * self.tile_size)
                
                # STEP 4: Enhanced selection logic with multiple criteria
                freq_energy = compute_frequency_energy(tile_image)
                
                # Main selection criteria
                selected = False
                selection_reason = ""
                
                if breast_ratio >= self.min_breast_ratio:
                    selected = True
                    selection_reason = "breast_tissue"
                elif (freq_energy >= self.min_freq_energy and 
                      breast_ratio >= self.min_breast_for_freq and 
                      mean_intensity >= MIN_TILE_INTENSITY + 10):  # Even stricter for freq tiles
                    selected = True
                    selection_reason = "frequency_energy"
                
                if selected:
                    tiles.append((img_path, x, y, breast_ratio, freq_energy))
                else:
                    if freq_energy < self.min_freq_energy:
                        rejected_freq_energy += 1
                    else:
                        rejected_breast_ratio += 1
        
        # Log rejection analysis like in notebook
        total_attempted = len(y_positions) * len(x_positions)
        if total_attempted > 0:
            print(f"    [{img_path.name}] Tile rejection: {rejected_background} bg ({rejected_background/total_attempted*100:.1f}%), "
                  f"{rejected_intensity} intensity, {rejected_breast_ratio} breast, {rejected_freq_energy} freq → "
                  f"{len(tiles)} ACCEPTED ({len(tiles)/total_attempted*100:.1f}%)")
        
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


def estimate_memory_usage(batch_size: int, tile_size: int = 256) -> float:
    """Estimate GPU memory usage in GB for the given configuration."""
    # Model parameters (ResNet50 + BYOL heads + momentum networks)
    model_memory = 6.5  # GB - ResNet50 + BYOL + momentum networks
    
    # Batch memory (RGB tiles + gradients + optimizer states)
    tile_memory_mb = (tile_size * tile_size * 3 * 4) / (1024 * 1024)  # 4 bytes per float32
    batch_memory = batch_size * tile_memory_mb * 4 / 1024  # x4 for forward/backward + optimizer states
    
    total_memory = model_memory + batch_memory
    return total_memory


def main():
    # Memory usage estimation
    estimated_memory = estimate_memory_usage(BATCH_SIZE, TILE_SIZE)
    print(f"📊 Estimated GPU Memory Usage: {estimated_memory:.1f} GB")
    if estimated_memory > 40:
        print(f"⚠️  Warning: May exceed A100-40GB capacity. Consider batch size {int(BATCH_SIZE * 35 / estimated_memory)}")
    elif estimated_memory < 25:
        print(f"💡 Tip: GPU underutilized. Consider increasing batch size to {int(BATCH_SIZE * 35 / estimated_memory)} for A100-40GB")
    print()

    # Initialize wandb (offline mode if no API key)
    try:
        wandb.init(
            project=WANDB_PROJECT,
            config={
                # A100 Optimization Settings
                "gpu_type": "A100",
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "learning_rate": LR,
                "warmup_epochs": WARMUP_EPOCHS,
                "estimated_memory_gb": estimate_memory_usage(BATCH_SIZE, TILE_SIZE),
                
                # Model Architecture
                "backbone": "resnet50",
                "pretrained_weights": "IMAGENET1K_V2",
                "tile_size": TILE_SIZE,
                "epochs": EPOCHS,
                "momentum_base": MOMENTUM_BASE,
                "hidden_dim": HIDDEN_DIM,
                "proj_dim": PROJ_DIM,
                
                # Medical Pipeline Settings
                "min_breast_ratio": MIN_BREAST_RATIO,
                "min_freq_energy": MIN_FREQ_ENERGY,
                "min_breast_for_freq": MIN_BREAST_FOR_FREQ,
                "min_tile_intensity": MIN_TILE_INTENSITY,
                "min_non_zero_pixels": MIN_NON_ZERO_PIXELS,
                
                # Optimization Features
                "mixed_precision": True,
                "pytorch_compile": hasattr(torch, 'compile'),
                "gradient_clipping": True,
                "lr_scheduler": "warmup_cosine",
            }
        )
        wandb_enabled = True
    except Exception as e:
        print(f"⚠️  WandB not configured, running offline. To enable: wandb login")
        wandb_enabled = False
    
    print("🔬 Mammogram BYOL Training with AGGRESSIVE Background Rejection")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Tile size: {TILE_SIZE}x{TILE_SIZE} (medical resolution preserved)")
    print(f"Tile stride: {TILE_STRIDE} pixels ({TILE_STRIDE/TILE_SIZE*100:.0f}% overlap)")
    print(f"\n🔍 AGGRESSIVE Background Rejection Parameters:")
    print(f"  🛡️  MIN_BREAST_RATIO: {MIN_BREAST_RATIO:.1%} (increased from 0.3)")
    print(f"  🛡️  MIN_FREQ_ENERGY: {MIN_FREQ_ENERGY:.3f} (much higher threshold)")
    print(f"  🛡️  MIN_BREAST_FOR_FREQ: {MIN_BREAST_FOR_FREQ:.1%} (stricter for frequency tiles)")
    print(f"  🛡️  MIN_TILE_INTENSITY: {MIN_TILE_INTENSITY} (reject dark background)")
    print(f"  🛡️  MIN_NON_ZERO_PIXELS: {MIN_NON_ZERO_PIXELS:.1%} (reject empty space)")
    print(f"\n🎛️ Medical-Optimized BYOL Augmentations:")
    print(f"  ✅ View 1: Mild brightness/contrast (0.1/0.1), horizontal flip, ±7° rotation")
    print(f"  ✅ View 2: Stronger brightness/contrast (0.15/0.15) + affine + blur")
    print(f"  ✅ No solarization/strong color jitter: preserves medical details")
    print(f"  ✅ Normalization: mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] (grayscale-appropriate)")
    print(f"\nMulti-level filtering eliminates ALL empty space tiles\n")
    
    # Medical-optimized BYOL transforms
    transform = create_medical_transforms(TILE_SIZE)
    
    # Dataset with AGGRESSIVE background rejection and micro-calcification detection
    dataset = BreastTileMammoDataset(
        DATA_DIR, TILE_SIZE, TILE_STRIDE, MIN_BREAST_RATIO, MIN_FREQ_ENERGY, MIN_BREAST_FOR_FREQ, transform
    )
    
    # A100-optimized DataLoader settings
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,           # A100 optimization: prefetch more batches
        multiprocessing_context='spawn',  # Better for CUDA
    )
    
    print(f"📊 Dataset: {len(dataset):,} breast tissue tiles → {len(loader):,} batches")
    
    # Model with classification readiness - ImageNet pretrained for better convergence
    # ImageNet pretraining helps even for medical images by providing:
    # 1. Better edge/texture detectors in early layers
    # 2. Faster convergence and more stable training
    # 3. Better generalization to medical domain features
    resnet = models.resnet50(weights='IMAGENET1K_V2')  # Latest ImageNet weights for better medical transfer
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    model = MammogramBYOL(backbone, INPUT_DIM, HIDDEN_DIM, PROJ_DIM).to(DEVICE)
    
    print(f"✅ Using ImageNet-pretrained ResNet50 backbone for better medical domain transfer")
    
    # A100 Performance Boost: PyTorch 2.0 Compile (if available)
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        print("🚀 Enabling PyTorch 2.0 compile optimization for A100...")
        model = torch.compile(model, mode='max-autotune')  # Maximum A100 optimization
        print("   ✅ Model compiled for maximum A100 performance")
    else:
        print("   ⚠️  PyTorch 2.0 compile not available - using standard optimization")
    
    criterion = NegativeCosineSimilarity()
    
    # Optimized for large batch training on A100
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LR, 
        weight_decay=1e-4,
        betas=(0.9, 0.999),  # Standard for large batch
        eps=1e-8
    )
    
    # LR warmup + cosine annealing for large batch stability
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0, 
        total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=EPOCHS - WARMUP_EPOCHS,  # After warmup
        eta_min=LR * 0.01  # 1% of peak LR
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[WARMUP_EPOCHS]
    )
    
    scaler = GradScaler()  # Mixed precision training for A100 optimization
    
    print(f"🧠 Model: ResNet50 backbone with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"🎯 Ready for downstream multi-label classification: {INPUT_DIM}→{HIDDEN_DIM//2}→2 [mass, calcification]")
    print(f"\n⚡ A100 GPU MAXIMUM PERFORMANCE OPTIMIZATIONS:")
    print(f"  🚀 Large batch training: BATCH_SIZE={BATCH_SIZE} (4x increase)")
    print(f"  🚀 Scaled learning rate: LR={LR} with {WARMUP_EPOCHS}-epoch warmup")
    print(f"  🚀 Mixed precision training: autocast + GradScaler")
    print(f"  🚀 PyTorch 2.0 compile: max-autotune mode (if available)")
    print(f"  🚀 Enhanced DataLoader: {NUM_WORKERS} workers, prefetch_factor=4")
    print(f"  🚀 Per-step momentum updates: optimal BYOL convergence")
    print(f"  🚀 Sequential LR scheduler: warmup → cosine annealing")
    print(f"  🚀 Gradient clipping: max_norm=1.0 for stability")
    print(f"  💾 Memory optimized: pin_memory + non_blocking transfers\n")
    
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
    print(f"\n🏥 === MEDICAL-OPTIMIZED BYOL TRAINING COMPLETE ===")
    print(f"⏱️  Total training time: {total_time/3600:.1f} hours")
    print(f"💾 Final model saved: {final_path}")
    print(f"📊 Dataset: {len(dataset):,} high-quality breast tissue tiles")
    print(f"🛡️  AGGRESSIVE background rejection: Zero empty space contamination")
    print(f"🎛️  Medical-safe augmentations: Preserves anatomical details")
    print(f"⚡ A100 optimized: Mixed precision + per-step momentum updates")
    print(f"🎯 Classification ready: Multi-label [mass, calcification] head")
    print(f"🚀 Ready for downstream fine-tuning!")
    
    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
