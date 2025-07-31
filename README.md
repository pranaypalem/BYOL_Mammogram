# BYOL Mammogram Training

Self-supervised BYOL (Bootstrap Your Own Latent) pre-training on mammogram images with intelligent breast tissue segmentation.

## 🔬 Features

- **Smart Breast Segmentation**: Only creates tiles from breast tissue regions, ignoring background
- **Medical Image Optimized**: Preserves full resolution and uses medical-appropriate augmentations
- **Classification Ready**: Includes classification head for downstream tasks
- **WandB Integration**: Real-time experiment tracking and visualization
- **Efficient Tiling**: Reduces training data by ~70% while maintaining quality

## 📋 Requirements

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Prepare Dataset

```bash
python prepare_dataset.py \
    --csv_path /path/to/breast-level_annotations.csv \
    --images_root /path/to/images_png \
    --output_dir ./split_images
```

### 2. Start Training

```bash
python train_byol_mammo.py
```

### 3. Optional: Enable WandB Tracking

```bash
wandb login
python train_byol_mammo.py
```

## 📊 Expected Results

- **Dataset**: ~45,000 breast tissue tiles from 16,000 mammograms
- **Efficiency**: 30%+ reduction in tiles while preserving breast tissue
- **Model**: ResNet50 backbone with 28M+ parameters
- **Output**: Classification-ready model for breast cancer detection

## 🏗️ Architecture

```
Input Mammogram → Breast Segmentation → Tile Extraction → BYOL Training
                                                       ↓
ResNet50 Backbone → Projection Head → Prediction Head → Loss
                 ↓
          Classification Head (2048→1024→2 classes)
```

## 📁 Project Structure

```
├── train_byol_mammo.py       # Main training script
├── prepare_dataset.py        # Dataset preparation CLI
├── breast_segmentation_tiling.ipynb  # Segmentation demo
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## 🔧 Configuration

Key parameters in `train_byol_mammo.py`:

- `TILE_SIZE = 256`: Tile dimensions (preserves medical detail)
- `MIN_BREAST_RATIO = 0.3`: Minimum breast tissue per tile
- `BATCH_SIZE = 8`: Batch size for training
- `EPOCHS = 100`: Training epochs
- `LR = 0.001`: Learning rate

## 🎯 Usage for Classification

After BYOL pre-training, use the model for classification:

```python
# Load pre-trained BYOL model
model = MammogramBYOL(backbone, INPUT_DIM, HIDDEN_DIM, PROJ_DIM)
model.load_state_dict(torch.load('mammogram_byol_best.pth')['model_state_dict'])

# Extract features or classify
features = model.get_features(image_tensor)
predictions = model.classify(image_tensor)
```

## 📈 Monitoring

Training metrics tracked:
- BYOL loss
- Learning rate schedule
- Breast tissue ratio per tile
- Training time and efficiency
- Model checkpoints

## 🖥️ Research Computing Deployment

1. Clone repository
2. Install dependencies
3. Prepare dataset with your paths
4. Run training

Designed for scalable deployment on HPC systems with GPU support.