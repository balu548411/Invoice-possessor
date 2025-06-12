#!/usr/bin/env python3
"""
Diagnostic script to debug NaN issues in entity classification
"""

import torch
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def check_entity_classifier():
    """Check the entity classifier configuration"""
    logger.info("Checking entity classifier setup...")
    
    try:
        from src.model_architecture import InvoiceProcessingModel
        
        # Create a small model
        model = InvoiceProcessingModel(
            vision_model='efficientnet_b0',
            d_model=256,
            num_layers=2,
            num_heads=8
        )
        
        # Check entity classifier
        entity_classifier = model.entity_classifier
        logger.info(f"Entity types: {entity_classifier.entity_types}")
        logger.info(f"Number of entity types: {len(entity_classifier.entity_types)}")
        
        # Check if classifier layers are properly initialized
        for name, param in entity_classifier.named_parameters():
            logger.info(f"Parameter {name}: shape={param.shape}, mean={param.mean():.4f}, std={param.std():.4f}")
            
            # Check for any NaN or Inf values
            if torch.isnan(param).any():
                logger.error(f"NaN detected in {name}!")
            if torch.isinf(param).any():
                logger.error(f"Inf detected in {name}!")
        
        # Test forward pass
        batch_size = 2
        num_boxes = 10
        d_model = 256
        
        # Create dummy input
        dummy_features = torch.randn(batch_size, num_boxes, d_model)
        
        # Forward pass
        with torch.no_grad():
            entity_logits = entity_classifier(dummy_features)
            
        logger.info(f"Entity logits shape: {entity_logits.shape}")
        logger.info(f"Entity logits range: [{entity_logits.min():.4f}, {entity_logits.max():.4f}]")
        
        # Check for NaN in output
        if torch.isnan(entity_logits).any():
            logger.error("NaN detected in entity logits!")
        else:
            logger.info("✅ No NaN in entity logits")
            
        return True
        
    except Exception as e:
        logger.error(f"Error checking entity classifier: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation():
    """Test loss computation with different scenarios"""
    logger.info("\nTesting loss computation...")
    
    try:
        import torch.nn as nn
        
        # Create loss function
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Test scenarios
        scenarios = [
            ("Normal case", torch.randn(10, 5), torch.randint(0, 5, (10,))),
            ("Large logits", torch.randn(10, 5) * 100, torch.randint(0, 5, (10,))),
            ("Very large logits", torch.randn(10, 5) * 1000, torch.randint(0, 5, (10,))),
            ("All zeros", torch.zeros(10, 5), torch.randint(0, 5, (10,))),
            ("Single class", torch.randn(10, 1), torch.zeros(10, dtype=torch.long)),
        ]
        
        for name, logits, targets in scenarios:
            try:
                loss = criterion(logits, targets)
                logger.info(f"{name}: loss={loss.item():.4f}, NaN={torch.isnan(loss).item()}")
            except Exception as e:
                logger.error(f"{name}: Error - {e}")
        
        # Test with label smoothing
        logger.info("\nTesting with label smoothing...")
        criterion_smooth = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        
        for name, logits, targets in scenarios[:3]:
            try:
                loss = criterion_smooth(logits, targets)
                logger.info(f"{name} (smoothed): loss={loss.item():.4f}, NaN={torch.isnan(loss).item()}")
            except Exception as e:
                logger.error(f"{name} (smoothed): Error - {e}")
                
        return True
        
    except Exception as e:
        logger.error(f"Error testing loss computation: {e}")
        return False


def check_data_pipeline():
    """Check if data pipeline produces valid data"""
    logger.info("\nChecking data pipeline...")
    
    try:
        from src.data_processing import InvoiceDataProcessor, create_data_loaders
        
        # Create dummy data
        dummy_df = pd.DataFrame({
            'image_path': ['dummy.jpg'] * 10,
            'json_path': ['dummy.json'] * 10,
            'num_words': [10] * 10,
            'avg_confidence': [0.9] * 10
        })
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            dummy_df, dummy_df,
            batch_size=2,
            num_workers=0,
            image_size=(224, 224)
        )
        
        logger.info("Checking batch structure...")
        
        # Get one batch
        for batch in train_loader:
            logger.info(f"Batch keys: {batch.keys()}")
            
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    
                    # Check for NaN
                    if torch.isnan(value).any():
                        logger.error(f"  NaN detected in {key}!")
                    
                    # Check ranges
                    if value.numel() > 0:
                        logger.info(f"    Range: [{value.min():.4f}, {value.max():.4f}]")
                else:
                    logger.info(f"  {key}: type={type(value)}")
            
            break  # Just check one batch
            
        return True
        
    except Exception as e:
        logger.error(f"Error checking data pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Test gradient flow through the model"""
    logger.info("\nTesting gradient flow...")
    
    try:
        from src.model_architecture import InvoiceProcessingModel
        import torch.nn as nn
        
        # Create small model
        model = InvoiceProcessingModel(
            vision_model='efficientnet_b0',
            d_model=256,
            num_layers=1,
            num_heads=8
        )
        
        # Create dummy batch
        batch = {
            'images': torch.randn(2, 3, 224, 224),
            'boxes': torch.rand(2, 10, 4),
            'texts': ['dummy'] * 2,
            'words': [['word'] * 10] * 2
        }
        
        # Forward pass
        outputs = model(batch)
        
        # Create simple loss
        criterion = nn.CrossEntropyLoss()
        entity_logits = outputs['entity_logits']
        targets = torch.randint(0, entity_logits.size(-1), (2, 10))
        
        loss = criterion(
            entity_logits.view(-1, entity_logits.size(-1)),
            targets.view(-1)
        )
        
        logger.info(f"Loss value: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                grad_stats[name] = {
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'has_nan': torch.isnan(grad).any().item(),
                    'has_inf': torch.isinf(grad).any().item()
                }
        
        # Report problematic gradients
        logger.info("\nGradient statistics:")
        nan_count = 0
        inf_count = 0
        
        for name, stats in grad_stats.items():
            if stats['has_nan'] or stats['has_inf']:
                logger.error(f"{name}: NaN={stats['has_nan']}, Inf={stats['has_inf']}")
                if stats['has_nan']:
                    nan_count += 1
                if stats['has_inf']:
                    inf_count += 1
        
        logger.info(f"\nTotal parameters with NaN gradients: {nan_count}")
        logger.info(f"Total parameters with Inf gradients: {inf_count}")
        
        if nan_count == 0 and inf_count == 0:
            logger.info("✅ No NaN or Inf gradients detected")
            
        return nan_count == 0 and inf_count == 0
        
    except Exception as e:
        logger.error(f"Error testing gradient flow: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all diagnostic tests"""
    logger.info("🔍 Running NaN diagnostic tests...\n")
    
    tests = [
        ("Entity Classifier Check", check_entity_classifier),
        ("Loss Computation Test", test_loss_computation),
        ("Data Pipeline Check", check_data_pipeline),
        ("Gradient Flow Test", test_gradient_flow)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info('='*60)
        
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DIAGNOSTIC SUMMARY:")
    logger.info("="*60)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    logger.info("="*60)
    
    if all_passed:
        logger.info("\n✅ All diagnostic tests passed!")
        logger.info("\n💡 Recommendations:")
        logger.info("1. Try using the fixed training script:")
        logger.info("   python train_model.py --batch_size 8 --learning_rate 1e-4")
        logger.info("\n2. Consider using a smaller model first:")
        logger.info("   python train_model.py --vision_model efficientnet_b0 --batch_size 4")
        logger.info("\n3. Monitor the training closely for the first few epochs")
    else:
        logger.info("\n❌ Some diagnostic tests failed.")
        logger.info("\n💡 Debugging steps:")
        logger.info("1. Check the entity classifier configuration")
        logger.info("2. Verify the number of entity classes matches your data")
        logger.info("3. Consider reducing the learning rate")
        logger.info("4. Use gradient clipping (already enabled in fixed trainer)")


if __name__ == "__main__":
    main() 