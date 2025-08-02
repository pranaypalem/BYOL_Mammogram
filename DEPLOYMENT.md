# 🚀 Cloud Deployment Guide

Step-by-step guide for deploying BYOL mammogram training on research computing clusters.

## 📋 Pre-Deployment Checklist

✅ **Dataset Preparation CLI**: `prepare_dataset.py` ready  
✅ **Training Script**: `train_byol_mammo.py` optimized  
✅ **Dependencies**: `requirements.txt` complete  
✅ **Git Repository**: Ready to clone  

## 🔧 Step 1: Create GitHub Repository

1. Go to https://github.com/pranaypalem/BYOL_Mammogram
2. Create the repository (if not exists)
3. Then push the code:

```bash
# From your local directory
git push -u origin main
```

## ☁️ Step 2: Clone on Research Computing

```bash
# On your research computing cluster
git clone https://github.com/pranaypalem/BYOL_Mammogram.git
cd BYOL_Mammogram
```

## 🐍 Step 3: Setup Environment

```bash
# Create conda environment
conda create -n byol_mammogram python=3.10 -y
conda activate byol_mammogram

# Install dependencies
pip install -r requirements.txt

# Optional: Login to WandB for tracking
wandb login
```

## 📊 Step 4: Prepare Dataset

```bash
# Prepare your dataset with actual paths
python prepare_dataset.py \
    --csv_path /path/to/your/breast-level_annotations.csv \
    --images_root /path/to/your/images_png \
    --output_dir ./split_images

# Expected output:
# 📊 Dataset preparation complete!
#    📊 Total images copied: 20,000
#    📁 Output directory: ./split_images
#    └── training: 16,000 images
#    └── test: 4,000 images
```

## 🏋️ Step 5: Start Training

### Option A: Simple Training
```bash
python train_byol_mammo.py
```

### Option B: Background Training with Logging
```bash
nohup python train_byol_mammo.py > training.log 2>&1 &
```

### Option C: SLURM Job (if available)
```bash
# Create job script
cat > train_byol.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=byol_mammogram
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=byol_training_%j.out
#SBATCH --error=byol_training_%j.err

# Load modules (adjust for your cluster)
module load cuda/11.8
module load python/3.10

# Activate environment
conda activate byol_mammogram

# Run training
python train_byol_mammo.py
EOF

# Submit job
sbatch train_byol.slurm
```

## 📈 Step 6: Monitor Training

### Real-time Monitoring
```bash
# Watch log file
tail -f training.log

# Check GPU usage
nvidia-smi -l 1

# Monitor WandB (if enabled)
# Visit: https://wandb.ai/your-username/mammogram-byol
```

### Expected Progress
```
🔬 Mammogram BYOL Training with AGGRESSIVE Background Rejection
Device: cuda  
Tile size: 512x512 (increased for fewer, higher quality tiles)
Tile stride: 256 pixels (50% overlap)

🔍 AGGRESSIVE Background Rejection Parameters:
  🛡️  MIN_BREAST_RATIO: 15.0% (increased from 0.3)
  🛡️  MIN_FREQ_ENERGY: 0.030 (much higher threshold)  
  🛡️  MIN_TILE_INTENSITY: 40 (reject dark background)
  🛡️  MIN_NON_ZERO_PIXELS: 70.0% (reject empty space)

🎛️ Enhanced BYOL Augmentations for Effective Self-Supervised Learning:
  ✅ View 1: Moderate (brightness/contrast 0.3/0.3, ±15° rotation, scale 0.85-1.15)
  ✅ View 2: Strong (brightness/contrast 0.4/0.4, ±25° rotation, perspective, blur)
  ✅ Added: Vertical flips, random perspective, random grayscale for diversity
  ✅ Balanced: Strong enough for BYOL while preserving medical details

[Dataset] Cache miss: Extracting tiles from 16000 mammogram images...
📊 Dataset: ~11,000 breast tissue tiles → 344 batches (4x fewer tiles due to larger size)

🧠 Model: ResNet50 backbone with 28,317,186 parameters
⚡ A100 GPU MAXIMUM PERFORMANCE OPTIMIZATIONS:
  🚀 Large batch training: BATCH_SIZE=32 (increased)
  🚀 Scaled learning rate: LR=2e-3 with 10-epoch warmup
  🚀 Mixed precision training: autocast + GradScaler

Epoch   1/100 │ Loss: -0.0254 │ Breast: 94.4% │ 24.6min                         
Epoch   2/100 │ Loss: -0.9820 │ Breast: 94.4% │ 45.4min                         
Epoch   3/100 │ Loss: -0.9854 │ Breast: 94.4% │ 66.1min
...
```

## 💾 Step 7: Model Checkpoints

Training automatically saves:
- `mammogram_byol_best.pth`: Best model based on loss
- `mammogram_byol_epoch{X}.pth`: Every 10 epochs  
- `mammogram_byol_final.pth`: Final model after training

## 🔍 Step 8: Validate Results

```bash
# Check final model
ls -la *.pth

# Verify model loading
python -c "
import torch
checkpoint = torch.load('mammogram_byol_best.pth', map_location='cpu')
print(f'✅ Best model saved at epoch {checkpoint[\"epoch\"]}')
print(f'📊 Final loss: {checkpoint[\"loss\"]:.4f}')
"
```

## ⚡ Performance Optimization Tips

### For A100 GPUs (Recommended):
- Current config optimized for A100: `BATCH_SIZE = 32`, `NUM_WORKERS = 16`
- Uses `LR = 2e-3` with 10-epoch warmup for large batch stability
- Mixed precision training enabled for maximum performance
- PyTorch 2.0 compile optimization (if available)

### For Smaller GPUs (V100/RTX):
- Reduce `BATCH_SIZE = 16` or `BATCH_SIZE = 8` in script  
- Lower `NUM_WORKERS = 8` to reduce CPU load
- Adjust `LR = 1e-3` proportionally with batch size

### For Long Training:
- Use `nohup` or screen/tmux sessions
- Automatic checkpoint resuming built-in
- Monitor disk space for checkpoints (saved every 10 epochs)
- Tile cache will speed up subsequent runs

### Memory Optimization:
- Larger tiles (512×512) require more GPU memory but train faster
- Enable `pin_memory=True` and `persistent_workers=True` 
- Use `prefetch_factor=4` for A100 optimization

## 🆘 Troubleshooting

### Common Issues:

**CUDA Out of Memory**:
```bash
# Reduce batch size in train_byol_mammo.py
BATCH_SIZE = 16  # For V100/RTX GPUs
BATCH_SIZE = 8   # For smaller GPUs
# Also reduce NUM_WORKERS = 8 to save CPU memory
```

**Dataset Not Found**:
```bash
# Check paths in prepare_dataset.py output
# Ensure split_images/ folder exists with training/ subfolder
```

**WandB Login Issues**:
```bash
# Run offline mode
export WANDB_MODE=offline
python train_byol_mammo.py
```

**Package Installation Errors**:
```bash
# Try with specific CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Expected Training Time

- **A100-80GB**: ~4-5 hours for 100 epochs (optimized config)
- **A100-40GB**: ~5-6 hours for 100 epochs
- **V100**: ~8-10 hours for 100 epochs (with reduced batch size)
- **Tile extraction**: ~57 minutes initially, then cached for future runs
- **Checkpoints**: Every 10 epochs (~25-30 minutes on A100)

## ✅ Success Metrics

Training completed successfully when you see:
```
🏥 === MEDICAL-OPTIMIZED BYOL TRAINING COMPLETE ===
⏱️  Total training time: 4.8 hours
💾 Final model saved: mammogram_byol_final.pth
📊 Dataset: 11,000 high-quality breast tissue tiles
🛡️  AGGRESSIVE background rejection: Zero empty space contamination
🎛️  Medical-safe augmentations: Preserves anatomical details
⚡ A100 optimized: Mixed precision + per-step momentum updates
🎯 Classification ready: Multi-label [mass, calcification] head
🚀 Ready for downstream fine-tuning!
```

Your model is now ready for breast cancer classification tasks! 🎉