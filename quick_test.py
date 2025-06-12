#!/usr/bin/env python3
"""
Quick test to verify training can start successfully
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run a quick training test"""
    logger.info("🚀 Starting quick training test...")
    
    # Import after setting up logging
    try:
        from src.data_processing import InvoiceDataProcessor
        from src.training import train_invoice_model
        from sklearn.model_selection import train_test_split
        import pandas as pd
        
        # Check if training data exists
        images_dir = "training_data/images"
        labels_dir = "training_data/labels"
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            logger.error(f"Training data not found at {images_dir} and {labels_dir}")
            return 1
        
        # Process minimal data
        logger.info("Processing data...")
        processor = InvoiceDataProcessor(images_dir, labels_dir)
        df = processor.process_dataset()
        
        if len(df) == 0:
            logger.error("No valid samples found in dataset")
            return 1
        
        # Use only first few samples for quick test
        df = df.head(min(10, len(df)))
        logger.info(f"Using {len(df)} samples for quick test")
        
        # Split data
        if len(df) >= 4:
            train_df, val_df = train_test_split(df, test_size=0.25, random_state=42)
        else:
            # If too few samples, use same data for train and val
            train_df = val_df = df
        
        logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
        
        # Minimal configuration for quick test
        config = {
            # Model config - smaller for testing
            'vision_model': 'efficientnet_b0',
            'text_model': 'microsoft/layoutlm-base-uncased',
            'd_model': 256,
            'num_layers': 2,
            'num_heads': 8,
            
            # Training config - minimal resources
            'batch_size': 1,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'max_epochs': 1,
            'num_workers': 0,  # No multiprocessing
            'image_size': (224, 224),  # Smaller images
            'precision': '32',  # No mixed precision for stability
            'accumulate_grad_batches': 1,
            
            # Experiment config
            'project_name': 'invoice-processing-test',
            'experiment_name': 'quick-test',
            'output_dir': './test_output'
        }
        
        logger.info("Starting training with minimal configuration...")
        logger.info(f"Config: {config}")
        
        # Train model
        try:
            model, trainer = train_invoice_model(train_df, val_df, config)
            logger.info("✅ Training completed successfully!")
            logger.info(f"Model saved to: {trainer.checkpoint_callback.best_model_path}")
            return 0
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
            
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    
    if exit_code == 0:
        print("\n" + "="*60)
        print("✅ Quick test passed! The training pipeline is working.")
        print("\nYou can now run full training with:")
        print("  python train_model.py --disable_multiprocessing")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Quick test failed. Please check the errors above.")
        print("\nTry running:")
        print("  python check_shared_memory.py")
        print("  python test_training.py")
        print("="*60)
    
    sys.exit(exit_code) 