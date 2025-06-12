#!/usr/bin/env python3
"""
Test script to verify training setup and fixes
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported"""
    logger.info("Testing imports...")
    try:
        from src.data_processing import InvoiceDataProcessor, create_data_loaders
        from src.model_architecture import InvoiceProcessingModel
        from src.training import InvoiceProcessingLightningModule, train_invoice_model
        logger.info("✅ All imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Import error: {e}")
        return False

def test_model_creation():
    """Test if model can be created"""
    logger.info("\nTesting model creation...")
    try:
        from src.model_architecture import InvoiceProcessingModel
        
        # Create model with smaller dimensions for testing
        model = InvoiceProcessingModel(
            vision_model='efficientnet_b0',  # Smaller model
            d_model=256,  # Smaller dimension
            num_layers=2,  # Fewer layers
            num_heads=8
        )
        
        # Test forward pass with dummy data
        batch_size = 2
        num_boxes = 10
        
        dummy_batch = {
            'images': torch.randn(batch_size, 3, 224, 224),
            'boxes': torch.rand(batch_size, num_boxes, 4),
            'texts': ['dummy'] * batch_size,
            'words': [['word'] * num_boxes] * batch_size
        }
        
        with torch.no_grad():
            outputs = model(dummy_batch)
        
        logger.info(f"✅ Model created successfully")
        logger.info(f"   Output shapes:")
        for key, value in outputs.items():
            logger.info(f"   - {key}: {value.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading with minimal setup"""
    logger.info("\nTesting data loading...")
    try:
        import pandas as pd
        from src.data_processing import create_data_loaders
        
        # Create dummy dataframe
        dummy_df = pd.DataFrame({
            'image_path': ['dummy.jpg'] * 5,
            'json_path': ['dummy.json'] * 5,
            'num_words': [10] * 5,
            'avg_confidence': [0.9] * 5
        })
        
        # Test with no workers to avoid multiprocessing issues
        train_loader, val_loader = create_data_loaders(
            dummy_df, dummy_df,
            batch_size=2,
            num_workers=0,  # No multiprocessing
            image_size=(224, 224)
        )
        
        logger.info(f"✅ Data loaders created successfully")
        logger.info(f"   Train batches: {len(train_loader)}")
        logger.info(f"   Val batches: {len(val_loader)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loading error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_setup():
    """Test if training can be initialized"""
    logger.info("\nTesting training setup...")
    try:
        from src.training import InvoiceProcessingLightningModule
        
        # Create model with test configuration
        model = InvoiceProcessingLightningModule(
            vision_model='efficientnet_b0',
            d_model=256,
            num_layers=2,
            num_heads=8,
            learning_rate=1e-4
        )
        
        # Test optimizer configuration
        optimizers = model.configure_optimizers()
        
        logger.info("✅ Training module created successfully")
        logger.info(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"   Optimizer: {type(optimizers['optimizer']).__name__}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training setup error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_gpu_memory():
    """Check GPU memory availability"""
    logger.info("\nChecking GPU resources...")
    
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA available")
        logger.info(f"   GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            logger.info(f"   GPU {i}: {props.name} ({memory_gb:.1f} GB)")
            
            # Check current memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                logger.info(f"   - Allocated: {allocated:.2f} GB")
                logger.info(f"   - Reserved: {reserved:.2f} GB")
    else:
        logger.info("⚠️  No GPU available, will use CPU")

def main():
    """Run all tests"""
    logger.info("🔍 Running training setup tests...\n")
    
    # Check GPU resources
    check_gpu_memory()
    
    # Run tests
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
        ("Training Setup", test_training_setup)
    ]
    
    results = []
    for test_name, test_func in tests:
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY:")
    logger.info("="*60)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    logger.info("="*60)
    
    if all_passed:
        logger.info("\n✅ All tests passed! The training setup should work.")
        logger.info("\n💡 Recommended training commands:")
        logger.info("   1. With multiprocessing disabled (safest):")
        logger.info("      python train_model.py --disable_multiprocessing")
        logger.info("\n   2. With reduced workers:")
        logger.info("      python train_model.py --num_workers 2")
        logger.info("\n   3. With smaller batch size:")
        logger.info("      python train_model.py --batch_size 4 --num_workers 2")
        logger.info("\n   4. Quick test run:")
        logger.info("      python train_model.py --disable_multiprocessing --batch_size 2 --max_epochs 1")
    else:
        logger.info("\n❌ Some tests failed. Please check the errors above.")
        logger.info("\n💡 Try running:")
        logger.info("   python check_shared_memory.py")
        logger.info("   to diagnose system issues.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 