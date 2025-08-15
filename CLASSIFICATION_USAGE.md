# Mammogram Classification Pipeline

This document explains how to use the complete mammogram classification system that builds upon the BYOL pre-trained model.

## Overview

The pipeline consists of 4 main components:

1. **Data Preparation** (`prepare_classification_data.py`): Generates tiles from images with annotations
2. **Training** (`train_classification.py`): Fine-tunes BYOL model for classification  
3. **Validation** (`validate_model.py`): Evaluates model performance on test set
4. **Inference** (`inference.py`): Performs prediction on single images with bounding boxes

## Quick Start

Use the pipeline script for the complete workflow:

```bash
# Run complete pipeline
python run_classification_pipeline.py --mode full_pipeline \
    --byol_checkpoint ./results/best_byol_model.pth \
    --max_samples 100  # Use small number for testing

# Or run individual steps:
python run_classification_pipeline.py --mode prepare_data
python run_classification_pipeline.py --mode train --byol_checkpoint ./results/best_byol_model.pth
python run_classification_pipeline.py --mode validate
python run_classification_pipeline.py --mode inference --inference_image ./split_images/test/example.png
```

## Detailed Usage

### 1. Data Preparation

Generates tiles from mammogram images using the same logic as BYOL training:

```bash
python prepare_classification_data.py \
    --csv_path vindr_detection_v1_folds.csv \
    --images_dir ./split_images/training \
    --output_dir ./classification_data \
    --split training \
    --max_images 1000  # Optional: limit for testing
```

**Outputs:**
- `classification_data/tiles_training/` - Individual tile images (512x512 PNG)
- `classification_data/tiles_training_metadata.csv` - Tile metadata with labels
- `classification_data/training_statistics.json` - Dataset statistics

**Key Features:**
- Uses exact same tile selection logic as BYOL training
- Handles multi-label annotations (Mass, Calcification, etc.)
- Creates "No_Finding" labels for tiles without pathology
- Filters background/low-quality tiles automatically

### 2. Training Classification Model

Fine-tunes the BYOL pre-trained model for multi-label classification:

```bash
python train_classification.py \
    --byol_checkpoint ./results/best_byol_model.pth \
    --train_csv ./classification_data/tiles_training_metadata.csv \
    --val_csv ./classification_data/tiles_test_metadata.csv \
    --tiles_dir ./classification_data/tiles_training \
    --output_dir ./classification_results \
    --epochs 50 \
    --batch_size 32
```

**Key Features:**
- Loads pre-trained BYOL backbone
- Two-stage training: frozen backbone → fine-tuning
- Multi-label BCE loss with label smoothing
- Different learning rates for backbone vs. classification head
- Mixed precision training with gradient clipping

**Outputs:**
- `best_classification_model.pth` - Best model checkpoint
- `final_classification_model.pth` - Final epoch checkpoint
- Training logs and metrics

### 3. Model Validation

Comprehensive evaluation on test set:

```bash
python validate_model.py \
    --checkpoint ./classification_results/best_classification_model.pth \
    --test_csv ./classification_data/tiles_test_metadata.csv \
    --tiles_dir ./classification_data/tiles_test \
    --output_dir ./validation_results \
    --batch_size 64
```

**Outputs:**
- `validation_report.txt` - Detailed text report
- `validation_metrics.json` - Metrics in JSON format
- `confusion_matrices.png` - Per-class confusion matrices
- `roc_curves.png` - ROC curves for all classes
- `class_distribution.png` - Test set class distribution

**Metrics Computed:**
- Per-class: AUC-ROC, Average Precision, F1-Score, Accuracy
- Overall: Mean metrics, Hamming Loss, Subset Accuracy, Exact Match

### 4. Inference on Single Images

Predict findings on new mammogram images:

```bash
python inference.py \
    --image ./path/to/mammogram.png \
    --checkpoint ./classification_results/best_classification_model.pth \
    --output_dir ./inference_results \
    --confidence_threshold 0.5 \
    --voting_strategy probability_weighted
```

**Key Features:**
- Generates tiles from full image using same logic as training
- Multiple voting strategies for aggregating tile predictions:
  - `majority_vote`: Simple majority voting
  - `probability_weighted`: Weight by tissue content and confidence
  - `max_confidence`: Use maximum confidence per class
- Generates bounding boxes for positive findings
- Creates visualization with overlaid predictions

**Outputs:**
- `inference_results.json` - Complete prediction results
- `prediction_visualization.png` - Image with bounding boxes
- Individual tile predictions and metadata

## File Structure

```
classification_data/
├── tiles_training/           # Training tile images
├── tiles_test/              # Test tile images  
├── tiles_training_metadata.csv
├── tiles_test_metadata.csv
├── training_statistics.json
└── test_statistics.json

classification_results/
├── best_classification_model.pth
├── final_classification_model.pth
├── classification_epoch_10.pth
└── ...

validation_results/
├── validation_report.txt
├── validation_metrics.json
├── confusion_matrices.png
├── roc_curves.png
└── class_distribution.png

inference_results/
├── inference_results.json
└── prediction_visualization.png
```

## Understanding the Output

### Classification Classes

The model predicts 11 finding classes from the VinDr dataset:

1. **Architectural_Distortion**
2. **Asymmetry** 
3. **Focal_Asymmetry**
4. **Global_Asymmetry**
5. **Mass**
6. **Nipple_Retraction**
7. **No_Finding** (normal tissue)
8. **Skin_Retraction**
9. **Skin_Thickening** 
10. **Suspicious_Calcification**
11. **Suspicious_Lymph_Node**

### Tile Metadata Format

Each tile has the following metadata:

```csv
tile_path,original_image,tile_x,tile_y,breast_ratio,freq_energy,has_finding,Mass,Calcification,...
tile_001.png,patient_123_image_456.png,0,256,0.85,0.12,1,1,0,...
```

- `tile_x/y`: Position of tile in original image
- `breast_ratio`: Fraction of tile containing breast tissue  
- `freq_energy`: High-frequency energy (tissue texture measure)
- `has_finding`: 1 if tile contains any pathological finding
- Individual class columns: 1 if finding present, 0 otherwise

### Inference Results Format

```json
{
  "final_prediction": {
    "predictions": {"Mass": 1, "Calcification": 0, ...},
    "probabilities": {"Mass": 0.87, "Calcification": 0.23, ...},
    "confidence": 0.87,
    "num_tiles": 45,
    "voting_strategy": "probability_weighted"
  },
  "bounding_boxes": [
    {
      "class": "Mass",
      "confidence": 0.91,
      "x": 512, "y": 768,
      "width": 512, "height": 512
    }
  ]
}
```

## Configuration Options

### Training Configuration

Create `training_config.json`:

```json
{
  "batch_size": 32,
  "epochs": 50,
  "lr_backbone": 1e-5,
  "lr_head": 1e-3,
  "weight_decay": 1e-4,
  "freeze_backbone_epochs": 10,
  "label_smoothing": 0.1,
  "dropout_rate": 0.3
}
```

Use with: `--config training_config.json`

### Tile Generation Parameters

The following parameters control tile quality (defined in `train_byol_mammo.py`):

- `TILE_SIZE = 512`: Size of each tile
- `TILE_STRIDE = 256`: Overlap between tiles (50%)  
- `MIN_BREAST_RATIO = 0.15`: Minimum breast tissue fraction
- `MIN_FREQ_ENERGY = 0.03`: Minimum texture complexity
- `MIN_TILE_INTENSITY = 40`: Minimum average intensity

## Tips for Best Results

### Data Preparation
- Use `--max_images` for quick testing with small datasets
- Check class distribution in generated statistics
- Ensure sufficient positive samples for each class

### Training
- Start with frozen backbone for stability
- Use lower learning rate for pre-trained backbone  
- Monitor validation metrics to avoid overfitting
- Consider class weighting for imbalanced datasets

### Inference
- Use `probability_weighted` voting for best results
- Lower confidence threshold (0.3-0.4) may improve recall
- Check multiple voting strategies for comparison
- Examine individual tile predictions for debugging

## Troubleshooting

### Common Issues

1. **"No annotations found"**: Check image filename format (patient_id_image_id.png)
2. **"Missing columns in CSV"**: Ensure CSV has all required finding class columns
3. **CUDA out of memory**: Reduce batch size or use CPU
4. **Low validation scores**: Check class imbalance, increase training epochs
5. **No bounding boxes in inference**: Lower confidence threshold

### Memory Usage

For large datasets:
- Reduce `batch_size` (default 32 → 16 or 8)
- Use `--max_samples` for testing
- Enable mixed precision (automatically enabled)
- Process test data in smaller batches

### Performance Tips

- Use GPU with sufficient memory (8GB+ recommended)
- Set `num_workers` based on CPU cores
- Use SSD storage for faster tile loading
- Pre-generate all tiles before training (done automatically)

## Example Complete Workflow

```bash
# 1. Prepare data for training and testing
python run_classification_pipeline.py --mode prepare_data --max_samples 500

# 2. Train classification model  
python run_classification_pipeline.py --mode train \
    --byol_checkpoint ./results/best_byol_model.pth \
    --train_epochs 30

# 3. Validate on test set
python run_classification_pipeline.py --mode validate

# 4. Run inference on new image
python run_classification_pipeline.py --mode inference \
    --inference_image ./split_images/test/example_image.png \
    --confidence_threshold 0.4

# Or run everything at once
python run_classification_pipeline.py --mode full_pipeline \
    --byol_checkpoint ./results/best_byol_model.pth \
    --max_samples 200 \
    --train_epochs 20
```

This will create a complete mammogram classification system ready for deployment!