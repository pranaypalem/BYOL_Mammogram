#!/usr/bin/env python3
"""
run_classification_pipeline.py

Complete pipeline script for mammogram classification using BYOL.
This script orchestrates the entire workflow from data preparation to inference.
"""

import subprocess
import argparse
from pathlib import Path
import sys
import json

def run_command(cmd, description, check_output=False):
    """Run a command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        if check_output:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return result.stdout
        else:
            result = subprocess.run(cmd, check=True)
            return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {e}")
        if hasattr(e, 'stdout') and e.stdout:
            print(f"stdout: {e.stdout}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"stderr: {e.stderr}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Complete mammogram classification pipeline')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['prepare_data', 'train', 'validate', 'inference', 'full_pipeline'],
                       help='Pipeline mode to run')
    
    # Data preparation arguments
    parser.add_argument('--csv_path', type=str, default='vindr_detection_v1_folds.csv',
                       help='Path to VinDr annotations CSV')
    parser.add_argument('--images_train_dir', type=str, default='./split_images/training',
                       help='Directory containing training images')
    parser.add_argument('--images_test_dir', type=str, default='./split_images/test', 
                       help='Directory containing test images')
    parser.add_argument('--tiles_output_dir', type=str, default='./classification_data',
                       help='Output directory for tile data')
    
    # Training arguments
    parser.add_argument('--byol_checkpoint', type=str, default='./mammogram_byol_best.pth',
                       help='Path to BYOL checkpoint')
    parser.add_argument('--train_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--classification_output_dir', type=str, default='./classification_results',
                       help='Output directory for classification training')
    
    # Validation arguments
    parser.add_argument('--validation_output_dir', type=str, default='./validation_results',
                       help='Output directory for validation results')
    
    # Inference arguments
    parser.add_argument('--inference_image', type=str,
                       help='Path to image for inference')
    parser.add_argument('--inference_output_dir', type=str, default='./inference_results',
                       help='Output directory for inference results')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for inference')
    
    # General arguments
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples for testing (use small number for quick tests)')
    
    args = parser.parse_args()
    
    print("🔬 MAMMOGRAM CLASSIFICATION PIPELINE")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    
    if args.mode == 'prepare_data' or args.mode == 'full_pipeline':
        print("\n📊 STEP 1: PREPARING CLASSIFICATION DATA")
        print("-" * 40)
        
        # Prepare training data
        cmd_train = [
            'python', 'prepare_classification_data.py',
            '--csv_path', args.csv_path,
            '--images_dir', args.images_train_dir,
            '--output_dir', args.tiles_output_dir,
            '--split', 'training'
        ]
        if args.max_samples:
            cmd_train.extend(['--max_images', str(args.max_samples)])
        
        run_command(cmd_train, "Preparing training tile data")
        
        # Prepare test data
        cmd_test = [
            'python', 'prepare_classification_data.py',
            '--csv_path', args.csv_path,
            '--images_dir', args.images_test_dir,
            '--output_dir', args.tiles_output_dir,
            '--split', 'test'
        ]
        if args.max_samples:
            cmd_test.extend(['--max_images', str(args.max_samples)])
        
        run_command(cmd_test, "Preparing test tile data")
    
    if args.mode == 'train' or args.mode == 'full_pipeline':
        print("\n🧠 STEP 2: TRAINING CLASSIFICATION MODEL")
        print("-" * 40)
        
        # Check if tile data exists
        train_csv = Path(args.tiles_output_dir) / 'tiles_training_metadata.csv'
        test_csv = Path(args.tiles_output_dir) / 'tiles_test_metadata.csv'
        tiles_train_dir = Path(args.tiles_output_dir) / 'tiles_training'
        
        if not train_csv.exists():
            print(f"❌ Training data not found: {train_csv}")
            print("Please run with --mode prepare_data first")
            sys.exit(1)
        
        cmd_train = [
            'python', 'train_classification.py',
            '--byol_checkpoint', args.byol_checkpoint,
            '--train_csv', str(train_csv),
            '--val_csv', str(test_csv),  # Using test as validation for now
            '--tiles_dir', str(tiles_train_dir),
            '--output_dir', args.classification_output_dir,
            '--epochs', str(args.train_epochs),
            '--batch_size', str(args.batch_size)
        ]
        if args.max_samples:
            cmd_train.extend(['--max_samples', str(args.max_samples * 10)])  # More samples for training
        
        run_command(cmd_train, "Training classification model")
    
    if args.mode == 'validate' or args.mode == 'full_pipeline':
        print("\n📈 STEP 3: VALIDATING MODEL")
        print("-" * 40)
        
        # Check if trained model exists
        model_checkpoint = Path(args.classification_output_dir) / 'best_classification_model.pth'
        test_csv = Path(args.tiles_output_dir) / 'tiles_test_metadata.csv'
        tiles_test_dir = Path(args.tiles_output_dir) / 'tiles_test'
        
        if not model_checkpoint.exists():
            print(f"❌ Trained model not found: {model_checkpoint}")
            print("Please run with --mode train first")
            sys.exit(1)
        
        cmd_validate = [
            'python', 'validate_model.py',
            '--checkpoint', str(model_checkpoint),
            '--test_csv', str(test_csv),
            '--tiles_dir', str(tiles_test_dir),
            '--output_dir', args.validation_output_dir,
            '--batch_size', str(args.batch_size)
        ]
        if args.max_samples:
            cmd_validate.extend(['--max_samples', str(args.max_samples)])
        
        run_command(cmd_validate, "Validating model on test set")
    
    if args.mode == 'inference':
        print("\n🎯 RUNNING INFERENCE")
        print("-" * 40)
        
        if not args.inference_image:
            print("❌ --inference_image is required for inference mode")
            sys.exit(1)
        
        # Check if trained model exists
        model_checkpoint = Path(args.classification_output_dir) / 'best_classification_model.pth'
        
        if not model_checkpoint.exists():
            print(f"❌ Trained model not found: {model_checkpoint}")
            print("Please run with --mode train first")
            sys.exit(1)
        
        cmd_inference = [
            'python', 'inference.py',
            '--image', args.inference_image,
            '--checkpoint', str(model_checkpoint),
            '--output_dir', args.inference_output_dir,
            '--confidence_threshold', str(args.confidence_threshold)
        ]
        
        run_command(cmd_inference, "Running inference on single image")
    
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    
    # Print summary of outputs
    print("\n📁 OUTPUT SUMMARY:")
    if args.mode in ['prepare_data', 'full_pipeline']:
        print(f"  📊 Tile data: {args.tiles_output_dir}")
    if args.mode in ['train', 'full_pipeline']:
        print(f"  🧠 Trained model: {args.classification_output_dir}")
    if args.mode in ['validate', 'full_pipeline']:
        print(f"  📈 Validation results: {args.validation_output_dir}")
    if args.mode == 'inference':
        print(f"  🎯 Inference results: {args.inference_output_dir}")


if __name__ == "__main__":
    main()