#!/usr/bin/env python3
"""
Comprehensive test script for the Invoice Processing Deep Learning Model
Tests all components: data processing, model architecture, training, and inference
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import json
import tempfile
from pathlib import Path
import logging
import traceback
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test results storage
test_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def log_test_result(test_name, status, message=""):
    """Log test result"""
    if status == "PASS":
        test_results['passed'].append(test_name)
        logger.info(f"✅ {test_name}: PASSED {message}")
    elif status == "FAIL":
        test_results['failed'].append(test_name)
        logger.error(f"❌ {test_name}: FAILED {message}")
    elif status == "WARN":
        test_results['warnings'].append(test_name)
        logger.warning(f"⚠️  {test_name}: WARNING {message}")

def test_imports():
    """Test all module imports"""
    test_name = "Module Imports"
    try:
        # Test individual imports
        from src.data_processing import InvoiceDataProcessor, InvoiceDataset, create_data_loaders
        from src.model_architecture import InvoiceProcessingModel, MultiModalInvoiceEncoder
        from src.training import InvoiceProcessingLightningModule, train_invoice_model
        from src.inference import InvoiceProcessor, InvoiceAPI
        
        # Test package import
        import src
        
        log_test_result(test_name, "PASS")
        return True
    except Exception as e:
        log_test_result(test_name, "FAIL", f"- {str(e)}")
        traceback.print_exc()
        return False

def test_data_structures():
    """Test data directory structure"""
    test_name = "Data Structure"
    try:
        images_dir = Path("training_data/images")
        labels_dir = Path("training_data/labels")
        
        if not images_dir.exists():
            log_test_result(test_name, "WARN", "- Images directory not found")
            return False
        
        if not labels_dir.exists():
            log_test_result(test_name, "WARN", "- Labels directory not found")
            return False
        
        # Check for files
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        json_files = list(labels_dir.glob("*.json"))
        
        if len(image_files) == 0:
            log_test_result(test_name, "WARN", "- No image files found")
            return False
        
        if len(json_files) == 0:
            log_test_result(test_name, "WARN", "- No JSON files found")
            return False
        
        log_test_result(test_name, "PASS", f"- Found {len(image_files)} images, {len(json_files)} labels")
        return True
        
    except Exception as e:
        log_test_result(test_name, "FAIL", f"- {str(e)}")
        return False

def test_data_processing():
    """Test data processing pipeline"""
    test_name = "Data Processing"
    try:
        from src.data_processing import InvoiceDataProcessor, InvoiceDataset, create_data_loaders
        
        # Create dummy data for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy image
            img_path = Path(tmpdir) / "images" / "test.jpg"
            img_path.parent.mkdir(parents=True)
            dummy_img = Image.new('RGB', (100, 100), color='white')
            dummy_img.save(img_path)
            
            # Create dummy JSON
            json_path = Path(tmpdir) / "labels" / "test.json"
            json_path.parent.mkdir(parents=True)
            dummy_json = {
                "pages": [{
                    "lines": [{
                        "content": "Invoice #12345",
                        "polygon": [{"x": 10, "y": 10}, {"x": 90, "y": 10}, 
                                   {"x": 90, "y": 30}, {"x": 10, "y": 30}],
                        "words": [{
                            "content": "Invoice",
                            "polygon": [{"x": 10, "y": 10}, {"x": 50, "y": 10},
                                       {"x": 50, "y": 30}, {"x": 10, "y": 30}],
                            "confidence": 0.95
                        }]
                    }],
                    "words": [{
                        "content": "Invoice",
                        "polygon": [{"x": 10, "y": 10}, {"x": 50, "y": 10},
                                   {"x": 50, "y": 30}, {"x": 10, "y": 30}],
                        "confidence": 0.95
                    }]
                }]
            }
            
            with open(json_path, 'w') as f:
                json.dump(dummy_json, f)
            
            # Test processor
            processor = InvoiceDataProcessor(str(img_path.parent), str(json_path.parent))
            df = processor.process_dataset()
            
            if len(df) > 0:
                # Test dataset
                dataset = InvoiceDataset(df, max_boxes=100)
                sample = dataset[0]
                
                # Validate sample structure
                required_keys = ['image', 'boxes', 'labels', 'entities', 'texts', 'words']
                for key in required_keys:
                    if key not in sample:
                        log_test_result(test_name, "FAIL", f"- Missing key '{key}' in dataset sample")
                        return False
                
                # Test data loader
                train_loader, val_loader = create_data_loaders(df, df, batch_size=1, num_workers=0)
                batch = next(iter(train_loader))
                
                # Validate batch structure
                batch_keys = ['images', 'boxes', 'labels', 'texts', 'words']
                for key in batch_keys:
                    if key not in batch:
                        log_test_result(test_name, "FAIL", f"- Missing key '{key}' in batch")
                        return False
                
                log_test_result(test_name, "PASS", "- All data processing components working")
                return True
            else:
                log_test_result(test_name, "FAIL", "- No data processed")
                return False
                
    except Exception as e:
        log_test_result(test_name, "FAIL", f"- {str(e)}")
        traceback.print_exc()
        return False

def test_model_architecture():
    """Test model architecture"""
    test_name = "Model Architecture"
    try:
        from src.model_architecture import InvoiceProcessingModel
        
        # Create model with small config for testing
        model = InvoiceProcessingModel(
            vision_model='efficientnet_b0',
            d_model=128,
            num_layers=2,
            num_heads=4
        )
        
        # Test forward pass
        batch_size = 2
        num_boxes = 10
        
        dummy_batch = {
            'images': torch.randn(batch_size, 3, 224, 224),
            'boxes': torch.rand(batch_size, num_boxes, 4),
            'labels': torch.ones(batch_size, num_boxes, dtype=torch.long),
            'texts': [['text'] * num_boxes] * batch_size,
            'words': [[['word'] * 1] * num_boxes] * batch_size
        }
        
        # Run forward pass
        with torch.no_grad():
            outputs = model(dummy_batch)
        
        # Validate outputs
        required_outputs = ['entity_logits', 'key_logits', 'value_logits', 'confidence_scores', 'features']
        for key in required_outputs:
            if key not in outputs:
                log_test_result(test_name, "FAIL", f"- Missing output '{key}'")
                return False
        
        # Check output shapes
        assert outputs['entity_logits'].shape == (batch_size, num_boxes, 11)
        assert outputs['key_logits'].shape == (batch_size, num_boxes, 2)
        assert outputs['value_logits'].shape == (batch_size, num_boxes, 2)
        assert outputs['confidence_scores'].shape == (batch_size, num_boxes, 1)
        
        # Test entity extraction
        entities = model.extract_entities(outputs, dummy_batch)
        assert len(entities) == batch_size
        
        log_test_result(test_name, "PASS", "- Model architecture validated")
        return True
        
    except Exception as e:
        log_test_result(test_name, "FAIL", f"- {str(e)}")
        traceback.print_exc()
        return False

def test_training_module():
    """Test training module"""
    test_name = "Training Module"
    try:
        from src.training import InvoiceProcessingLightningModule
        
        # Create lightning module
        module = InvoiceProcessingLightningModule(
            vision_model='efficientnet_b0',
            d_model=128,
            num_layers=2,
            num_heads=4,
            learning_rate=1e-4
        )
        
        # Test forward pass
        batch_size = 2
        num_boxes = 10
        
        dummy_batch = {
            'images': torch.randn(batch_size, 3, 224, 224),
            'boxes': torch.rand(batch_size, num_boxes, 4),
            'labels': torch.ones(batch_size, num_boxes, dtype=torch.long),
            'texts': [['text'] * num_boxes] * batch_size,
            'words': [[['word'] * 1] * num_boxes] * batch_size
        }
        
        # Test training step
        loss = module.training_step(dummy_batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        
        # Test validation step
        val_loss = module.validation_step(dummy_batch, 0)
        assert isinstance(val_loss, torch.Tensor)
        
        # Test optimizer config
        opt_config = module.configure_optimizers()
        assert 'optimizer' in opt_config
        assert 'lr_scheduler' in opt_config
        
        log_test_result(test_name, "PASS", "- Training module validated")
        return True
        
    except Exception as e:
        log_test_result(test_name, "FAIL", f"- {str(e)}")
        traceback.print_exc()
        return False

def test_inference():
    """Test inference pipeline"""
    test_name = "Inference Pipeline"
    try:
        from src.inference import InvoiceProcessor
        from src.training import InvoiceProcessingLightningModule
        
        # Create and save a dummy model
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy model
            module = InvoiceProcessingLightningModule(
                vision_model='efficientnet_b0',
                d_model=128,
                num_layers=2,
                num_heads=4
            )
            
            # Save checkpoint
            checkpoint_path = Path(tmpdir) / "model.ckpt"
            torch.save({
                'state_dict': module.state_dict(),
                'hyper_parameters': module.hparams
            }, checkpoint_path)
            
            # Test processor initialization
            processor = InvoiceProcessor(str(checkpoint_path), device='cpu')
            
            # Create dummy image
            dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            
            # Test preprocessing
            image_tensor, original_size = processor.preprocess_image(dummy_image)
            assert image_tensor.shape == (1, 3, 512, 512)
            
            # Test full inference (will use dummy OCR)
            result = processor.process_invoice(dummy_image)
            
            # Validate result structure
            assert 'status' in result
            assert 'pages' in result
            assert 'confidence' in result
            
            log_test_result(test_name, "PASS", "- Inference pipeline validated")
            return True
            
    except Exception as e:
        log_test_result(test_name, "FAIL", f"- {str(e)}")
        traceback.print_exc()
        return False

def test_memory_usage():
    """Test memory usage and GPU availability"""
    test_name = "Memory & GPU"
    try:
        import psutil
        
        # Check system memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        if memory_gb < 8:
            log_test_result(test_name, "WARN", f"- Low system memory: {memory_gb:.1f}GB")
        
        # Check GPU
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                logger.info(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
            
            log_test_result(test_name, "PASS", f"- {gpu_count} GPU(s) available")
        else:
            log_test_result(test_name, "WARN", "- No GPU available, will use CPU")
        
        return True
        
    except Exception as e:
        log_test_result(test_name, "FAIL", f"- {str(e)}")
        return False

def run_all_tests():
    """Run all tests"""
    logger.info("🔍 Running comprehensive tests for Invoice Processing Model...\n")
    
    # Define test suite
    tests = [
        ("Imports", test_imports),
        ("Data Structures", test_data_structures),
        ("Data Processing", test_data_processing),
        ("Model Architecture", test_model_architecture),
        ("Training Module", test_training_module),
        ("Inference Pipeline", test_inference),
        ("Memory & GPU", test_memory_usage)
    ]
    
    # Run tests
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        test_func()
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {len(test_results['passed'])}")
    print(f"❌ Failed: {len(test_results['failed'])}")
    print(f"⚠️  Warnings: {len(test_results['warnings'])}")
    print("="*60)
    
    if test_results['failed']:
        print("\nFailed tests:")
        for test in test_results['failed']:
            print(f"  - {test}")
    
    if test_results['warnings']:
        print("\nWarnings:")
        for test in test_results['warnings']:
            print(f"  - {test}")
    
    # Overall result
    if not test_results['failed']:
        print("\n✅ All critical tests passed! The system is ready for use.")
        print("\n💡 Recommended next steps:")
        print("  1. Run quick training test: python quick_test.py")
        print("  2. Start full training: python train_model.py --disable_multiprocessing")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues before proceeding.")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code) 