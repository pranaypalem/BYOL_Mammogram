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
🔬 Mammogram BYOL Training
Device: cuda
Tile size: 256x256 (medical resolution preserved)

[Dataset] Processing 16000 mammogram images...
📊 Dataset: 45,231 breast tissue tiles → 5,654 batches

🧠 Model: ResNet50 backbone with 28,317,186 parameters

Epoch   1/100 │ Loss: 0.8234 │ Breast: 67.3% │ Time: 2.1min
Epoch   2/100 │ Loss: 0.7891 │ Breast: 68.1% │ Time: 4.3min
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

### For Large Datasets:
- Increase `NUM_WORKERS = 16` in script
- Use `BATCH_SIZE = 16` if GPU memory allows
- Enable `persistent_workers=True` in DataLoader

### For Long Training:
- Use `nohup` or screen/tmux sessions
- Set up automatic checkpoint resuming
- Monitor disk space for checkpoints

### For Multiple GPUs:
- Consider distributed training modifications
- Use larger batch sizes
- Adjust learning rate accordingly

## 🆘 Troubleshooting

### Common Issues:

**CUDA Out of Memory**:
```bash
# Reduce batch size in train_byol_mammo.py
BATCH_SIZE = 4  # or smaller
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

- **16K images**: ~6-8 hours on V100 GPU
- **100 epochs**: Full training cycle
- **Checkpoints**: Every 10 epochs (~45 minutes)

## ✅ Success Metrics

Training completed successfully when you see:
```
✅ BYOL pre-training complete!
⏱️  Total time: 7.2 hours
💾 Final model: mammogram_byol_final.pth
🎯 Ready for classification fine-tuning!
```

Your model is now ready for breast cancer classification tasks! 🎉