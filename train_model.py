#!/usr/bin/env python3
"""
Main training script for Invoice Processing Deep Learning Model
Similar to Azure Form Recognizer but more powerful and customizable
"""

import os
import argparse
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_processing import InvoiceDataProcessor
# Use the fixed training module with NaN handling
from src.training_fixes import create_fixed_trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Invoice Processing Deep Learning Model"
    )
    
    # Data arguments
    parser.add_argument(
        "--images_dir", 
        type=str, 
        default="training_data/images",
        help="Directory containing invoice images"
    )
    parser.add_argument(
        "--labels_dir", 
        type=str, 
        default="training_data/labels",
        help="Directory containing JSON annotation files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs",
        help="Directory to save model checkpoints and logs"
    )
    
    # Model arguments
    parser.add_argument(
        "--vision_model", 
        type=str, 
        default="efficientnet_b3",
        help="Vision backbone model name"
    )
    parser.add_argument(
        "--text_model", 
        type=str, 
        default="microsoft/layoutlm-base-uncased",
        help="Text model for layout understanding"
    )
    parser.add_argument(
        "--d_model", 
        type=int, 
        default=768,
        help="Model hidden dimension"
    )
    parser.add_argument(
        "--num_layers", 
        type=int, 
        default=6,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", 
        type=int, 
        default=12,
        help="Number of attention heads"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01,
        help="Weight decay for regularization"
    )
    parser.add_argument(
        "--max_epochs", 
        type=int, 
        default=50,
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4,
        help="Number of data loader workers"
    )
    parser.add_argument(
        "--disable_multiprocessing", 
        action="store_true",
        help="Disable DataLoader multiprocessing (sets num_workers=0)"
    )
    parser.add_argument(
        "--image_size", 
        type=int, 
        nargs=2, 
        default=[512, 512],
        help="Input image size (height width)"
    )
    parser.add_argument(
        "--precision", 
        type=str, 
        default="16-mixed",
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Training precision"
    )
    parser.add_argument(
        "--accumulate_grad_batches", 
        type=int, 
        default=4,
        help="Gradient accumulation steps"
    )
    
    # Experiment arguments
    parser.add_argument(
        "--project_name", 
        type=str, 
        default="invoice-processing",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--experiment_name", 
        type=str, 
        default="multimodal-v1",
        help="Experiment name"
    )
    parser.add_argument(
        "--resume_from", 
        type=str, 
        default=None,
        help="Checkpoint path to resume training from"
    )
    
    # Data split arguments
    parser.add_argument(
        "--test_size", 
        type=float, 
        default=0.2,
        help="Validation set size (fraction)"
    )
    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def setup_directories(output_dir: str):
    """Create necessary directories"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / "checkpoints").mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)
    (output_path / "visualizations").mkdir(exist_ok=True)
    
    logger.info(f"Output directory setup complete: {output_path}")


def process_data(images_dir: str, labels_dir: str, test_size: float, random_seed: int):
    """Process and split the dataset"""
    logger.info("Starting data processing...")
    
    # Initialize data processor
    processor = InvoiceDataProcessor(images_dir, labels_dir)
    
    # Process dataset
    df = processor.process_dataset()
    logger.info(f"Processed {len(df)} invoice samples")
    
    if len(df) == 0:
        raise ValueError("No valid samples found in dataset")
    
    # Split dataset
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_seed,
        stratify=None  # Could stratify by some feature if needed
    )
    
    logger.info(f"Dataset split - Train: {len(train_df)}, Validation: {len(val_df)}")
    
    # Print dataset statistics
    logger.info("Dataset Statistics:")
    logger.info(f"Average words per image: {df['num_words'].mean():.1f}")
    logger.info(f"Average confidence: {df['avg_confidence'].mean():.3f}")
    
    return train_df, val_df


def create_config(args):
    """Create training configuration from arguments"""
    # Handle multiprocessing settings
    num_workers = 0 if args.disable_multiprocessing else args.num_workers
    
    config = {
        # Model config
        'vision_model': args.vision_model,
        'text_model': args.text_model,
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        
        # Training config
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'max_epochs': args.max_epochs,
        'num_workers': num_workers,
        'image_size': tuple(args.image_size),
        'precision': args.precision,
        'accumulate_grad_batches': args.accumulate_grad_batches,
        
        # Experiment config
        'project_name': args.project_name,
        'experiment_name': args.experiment_name,
        'output_dir': args.output_dir
    }
    
    return config


def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    logger.info("=== Invoice Processing Model Training ===")
    logger.info(f"Arguments: {vars(args)}")
    
    # Setup directories
    setup_directories(args.output_dir)
    
    # Process data
    try:
        train_df, val_df = process_data(
            args.images_dir, 
            args.labels_dir, 
            args.test_size, 
            args.random_seed
        )
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        return
    
    # Create training configuration
    config = create_config(args)
    
    # Add stability parameters
    config['gradient_clip_val'] = 1.0
    config['label_smoothing'] = 0.1
    
    # Train model with fixed trainer
    try:
        logger.info("Starting model training with NaN handling...")
        model, trainer = create_fixed_trainer(train_df, val_df, config)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best model saved to: {trainer.checkpoint_callback.best_model_path}")
        
        # Save final statistics
        stats = {
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'total_epochs': trainer.current_epoch,
            'best_checkpoint': trainer.checkpoint_callback.best_model_path
        }
        
        import json
        stats_path = Path(args.output_dir) / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Training statistics saved to: {stats_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main() 