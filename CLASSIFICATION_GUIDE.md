# 🎯 Classification Training Guide

Complete guide for fine-tuning the BYOL pre-trained model for multi-label classification.

## 📋 Overview

After BYOL pre-training completes, you can fine-tune the model for classification using the `train_classification.py` script. This approach:

1. **Loads the BYOL checkpoint** with learned representations
2. **Freezes the backbone** initially (optional) to prevent overwriting good features
3. **Fine-tunes the classification head** with a higher learning rate
4. **Gradually unfreezes** the backbone for end-to-end fine-tuning

## 🗂️ Data Preparation

### CSV Format
Create train/validation CSV files with this format:

```csv
tile_path,mass,calcification,architectural_distortion,asymmetry,normal,benign,malignant,birads_2,birads_3,birads_4
patient1_tile_001.png,1,0,0,0,0,1,0,0,1,0
patient1_tile_002.png,0,1,0,0,0,0,1,0,0,1
patient2_tile_001.png,0,0,0,0,1,1,0,1,0,0
...
```

**Requirements:**
- `tile_path`: Relative path to tile image
- **Class columns**: Binary labels (0/1) for each class
- **Multi-label support**: Each image can have multiple classes = 1

### Directory Structure
```
your_project/
├── tiles/                    # Directory containing tile images
│   ├── patient1_tile_001.png
│   ├── patient1_tile_002.png
│   └── ...
├── train_labels.csv         # Training labels
├── val_labels.csv          # Validation labels
└── mammogram_byol_best.pth # BYOL checkpoint
```

## 🚀 Quick Start

### 1. Basic Classification Training

```bash
python train_classification.py \
    --byol_checkpoint ./mammogram_byol_best.pth \
    --train_csv ./train_labels.csv \
    --val_csv ./val_labels.csv \
    --tiles_dir ./tiles \
    --class_names mass calcification architectural_distortion asymmetry normal benign malignant birads_2 birads_3 birads_4 \
    --output_dir ./classification_results
```

### 2. With Custom Configuration

```bash
python train_classification.py \
    --byol_checkpoint ./mammogram_byol_best.pth \
    --train_csv ./train_labels.csv \
    --val_csv ./val_labels.csv \
    --tiles_dir ./tiles \
    --class_names mass calcification normal \
    --config ./classification_config.json \
    --output_dir ./classification_results \
    --wandb_project my-mammogram-classification
```

### 3. Quick Testing (Limited Dataset)

```bash
python train_classification.py \
    --byol_checkpoint ./mammogram_byol_best.pth \
    --train_csv ./train_labels.csv \
    --val_csv ./val_labels.csv \
    --tiles_dir ./tiles \
    --class_names mass calcification normal \
    --max_samples 1000 \
    --output_dir ./test_results
```

## ⚙️ Configuration Options

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Batch size for training |
| `epochs` | 50 | Number of training epochs |
| `lr_backbone` | 1e-5 | Learning rate for pre-trained backbone |
| `lr_head` | 1e-3 | Learning rate for classification head |
| `freeze_backbone_epochs` | 10 | Epochs to freeze backbone (0 = never freeze) |
| `label_smoothing` | 0.1 | Label smoothing for regularization |
| `gradient_clip` | 1.0 | Gradient clipping max norm |

### Custom Configuration File

Create `my_config.json`:
```json
{
  "batch_size": 64,
  "epochs": 100,
  "lr_backbone": 5e-6,
  "lr_head": 2e-3,
  "freeze_backbone_epochs": 20,
  "label_smoothing": 0.2,
  "weight_decay": 1e-3
}
```

## 📊 Expected Training Process

### Phase 1: Backbone Frozen (Epochs 1-10)
```
🧊 Epoch 1: Backbone frozen (training only classification head)
Epoch   1/50:
  Train Loss: 0.6234
  Val Loss:   0.5891
  Mean AUC:   0.7123
  Mean AP:    0.6894
  Exact Match: 0.4512
  ✅ New best model saved (AUC: 0.7123)
```

### Phase 2: End-to-End Fine-tuning (Epochs 11-50)
```
Epoch  15/50:
  Train Loss: 0.3456
  Val Loss:   0.3891
  Mean AUC:   0.8567
  Mean AP:    0.8234
  Exact Match: 0.6789
  ✅ New best model saved (AUC: 0.8567)
```

## 🔍 Making Predictions

### Single Image Inference

```bash
python inference_classification.py \
    --model_path ./classification_results/best_classification_model.pth \
    --image_path ./test_image.png \
    --threshold 0.5
```

**Output:**
```
📸 Image 1: test_image.png
🏆 Top prediction: mass (0.847)
📊 All probabilities:
   ✅ mass              : 0.847
   ❌ calcification     : 0.234
   ❌ normal            : 0.123
   ❌ architectural_distortion: 0.089
```

### Batch Inference

```bash
python inference_classification.py \
    --model_path ./classification_results/best_classification_model.pth \
    --images_dir ./test_images \
    --output_json ./predictions.json \
    --batch_size 64
```

### Programmatic Usage

```python
import torch
from train_byol_mammo import MammogramBYOL
from inference_classification import load_classification_model, create_inference_transform

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, class_names, config = load_classification_model(
    "./classification_results/best_classification_model.pth", device
)

# Make prediction
transform = create_inference_transform()
image = Image.open("test.png").convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model.classify(input_tensor)
    probabilities = torch.sigmoid(logits).cpu().numpy()[0]

# Get results
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {probabilities[i]:.3f}")
```

## 📈 Monitoring Training

### Weights & Biases Integration

The script automatically logs to W&B:
- Training/validation loss curves
- Per-class AUC and Average Precision
- Learning rate schedules
- Model hyperparameters

### Metrics Explained

- **AUC (Area Under Curve)**: Measures ranking quality (0-1, higher better)
- **AP (Average Precision)**: Summarizes precision-recall curve (0-1, higher better)  
- **Exact Match Accuracy**: Percentage where ALL labels are predicted correctly
- **Per-Class Accuracy**: Binary accuracy for each individual class

## 💾 Output Files

Training creates:
```
classification_results/
├── best_classification_model.pth      # Best model by validation AUC
├── final_classification_model.pth     # Final model after all epochs
├── classification_epoch_10.pth        # Periodic checkpoints
├── classification_epoch_20.pth
└── ...
```

Each checkpoint contains:
- Model state dict
- Optimizer state  
- Training configuration
- Class names
- Validation metrics

## 🛠️ Advanced Usage

### Custom Loss Functions

For imbalanced datasets, modify the loss function:

```python
# Calculate positive weights for each class
pos_counts = df[class_names].sum()
neg_counts = len(df) - pos_counts
pos_weight = torch.tensor(neg_counts / pos_counts).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### Transfer Learning Strategies

1. **Conservative**: Freeze backbone for many epochs, low backbone LR
   - `freeze_backbone_epochs = 20`
   - `lr_backbone = 1e-6`

2. **Aggressive**: Unfreeze early, higher backbone LR
   - `freeze_backbone_epochs = 5`  
   - `lr_backbone = 1e-4`

3. **Progressive**: Gradually unfreeze layers (requires code modification)

### Multi-GPU Training

For multiple GPUs, wrap the model:
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

## ⚠️ Troubleshooting

### Common Issues

**Low Validation Performance:**
- Increase `freeze_backbone_epochs` to 15-20
- Reduce `lr_backbone` to 5e-6 or 1e-6
- Check for data leakage between train/val sets

**Overfitting:**
- Increase `label_smoothing` to 0.2-0.3
- Add more dropout (modify model architecture)
- Reduce learning rates
- Use early stopping

**Memory Issues:**
- Reduce `batch_size` to 16 or 8
- Reduce `num_workers` to 4
- Use gradient checkpointing (requires code modification)

**Class Imbalance:**
- Use `pos_weight` in loss function
- Focus on per-class AUC rather than accuracy
- Consider focal loss for extreme imbalance

## 🎯 Best Practices

1. **Start Conservative**: Use default settings first
2. **Monitor Per-Class Metrics**: Some classes may need special attention
3. **Validate Data**: Ensure no train/val overlap
4. **Checkpoint Often**: Training can be interrupted
5. **Use Multiple Runs**: Average results across random seeds
6. **Test Thoroughly**: Use held-out test set for final evaluation

## 📚 Complete Example

Here's a full workflow from BYOL training to classification:

```bash
# 1. Train BYOL (this takes 4-5 hours on A100)
python train_byol_mammo.py

# 2. Prepare classification data (create CSVs with labels)
# ... prepare train_labels.csv and val_labels.csv ...

# 3. Fine-tune for classification (1-2 hours)
python train_classification.py \
    --byol_checkpoint ./mammogram_byol_best.pth \
    --train_csv ./train_labels.csv \
    --val_csv ./val_labels.csv \
    --tiles_dir ./tiles \
    --class_names mass calcification architectural_distortion asymmetry normal \
    --output_dir ./classification_results

# 4. Run inference on new images
python inference_classification.py \
    --model_path ./classification_results/best_classification_model.pth \
    --images_dir ./new_patient_tiles \
    --output_json ./patient_predictions.json
```

This gives you a complete pipeline from self-supervised pre-training to production-ready classification! 🚀