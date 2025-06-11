import os
import sys

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
import wandb
from tqdm import tqdm
from pathlib import Path
import time
import gc
from contextlib import nullcontext

# Import project modules
from src.config import TRAIN_CONFIG, MODEL_CONFIG, BATCH_SIZE, NUM_WORKERS, MODEL_DIR, LOG_DIR, LOGGING_CONFIG
from src.data.dataset import get_dataset_splits, collate_fn
from src.model.document_parser import build_model
from src.model.losses import build_criterion
from src.evaluation.metrics import DocumentParsingEvaluator

# For mixed precision training
try:
    from torch.amp import GradScaler, autocast
    MIXED_PRECISION_AVAILABLE = True
except ImportError:
    MIXED_PRECISION_AVAILABLE = False


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, min_lr_factor=0.1, last_epoch=-1
):
    """
    Create a cosine learning rate schedule with warmup.
    
    Args:
        optimizer: The optimizer to use
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        num_cycles: Number of cycles for cosine decay
        min_lr_factor: Minimum learning rate as a factor of max
        last_epoch: Last epoch to resume from
    """
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return min_lr_factor + (1 - min_lr_factor) * 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, scaler=None, lr_scheduler=None):
    """Train the model for one epoch with optional mixed precision."""
    model.train()
    criterion.train()
    
    running_loss = 0.0
    running_loss_classifier = 0.0
    running_loss_bbox = 0.0
    
    start_time = time.time()
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} - Training")
    
    amp_ctx = autocast('cuda') if MIXED_PRECISION_AVAILABLE and TRAIN_CONFIG['mixed_precision'] and device.type == 'cuda' else nullcontext()
    
    for i, (images, targets) in enumerate(pbar):
        # Move data to device
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        # Forward pass with optional mixed precision
        optimizer.zero_grad()
        
        with amp_ctx:
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # Backward pass with optional mixed precision
        if MIXED_PRECISION_AVAILABLE and TRAIN_CONFIG['mixed_precision'] and device.type == 'cuda':
            # AMP: scales loss, backward, and updates scaler for next iteration
            scaler.scale(losses).backward()
            
            if TRAIN_CONFIG['clip_max_norm'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['clip_max_norm'])
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard backward pass
            losses.backward()
            if TRAIN_CONFIG['clip_max_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['clip_max_norm'])
            optimizer.step()
        
        # Update learning rate for schedulers that update per step
        if lr_scheduler is not None and TRAIN_CONFIG.get('lr_scheduler') == 'cosine':
            lr_scheduler.step()
        
        # Update running losses
        running_loss += losses.item()
        running_loss_classifier += loss_dict['loss_ce'].item() if 'loss_ce' in loss_dict else 0
        running_loss_bbox += loss_dict['loss_bbox'].item() if 'loss_bbox' in loss_dict else 0
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses.item():.4f}",
            'cls_loss': f"{loss_dict['loss_ce'].item() if 'loss_ce' in loss_dict else 0:.4f}",
            'box_loss': f"{loss_dict['loss_bbox'].item() if 'loss_bbox' in loss_dict else 0:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
        
        # Log to wandb
        if LOGGING_CONFIG['use_wandb'] and (i % LOGGING_CONFIG['log_interval'] == 0 or i == len(data_loader) - 1):
            wandb.log({
                'train/loss': losses.item(),
                'train/loss_ce': loss_dict['loss_ce'].item() if 'loss_ce' in loss_dict else 0,
                'train/loss_bbox': loss_dict['loss_bbox'].item() if 'loss_bbox' in loss_dict else 0,
                'train/lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch + i / len(data_loader),
                'iteration': epoch * len(data_loader) + i
            })
    
    # Average losses
    avg_loss = running_loss / len(data_loader)
    avg_loss_classifier = running_loss_classifier / len(data_loader)
    avg_loss_bbox = running_loss_bbox / len(data_loader)
    
    # Calculate epoch time
    epoch_time = time.time() - start_time
    logging.info(f"Epoch {epoch} training completed in {epoch_time:.2f} seconds")
    
    return avg_loss, avg_loss_classifier, avg_loss_bbox


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, epoch):
    """Evaluate the model."""
    model.eval()
    criterion.eval()
    
    evaluator = DocumentParsingEvaluator()
    
    running_loss = 0.0
    running_loss_classifier = 0.0
    running_loss_bbox = 0.0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} - Validation")
    
    for images, targets in pbar:
        # Move data to device
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss_dict = criterion(outputs, targets)
        
        # Extract losses
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # Update running losses
        running_loss += losses.item()
        running_loss_classifier += loss_dict['loss_ce'].item() if 'loss_ce' in loss_dict else 0
        running_loss_bbox += loss_dict['loss_bbox'].item() if 'loss_bbox' in loss_dict else 0
        
        # Update evaluator
        evaluator.update(outputs, targets)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': losses.item(),
            'cls_loss': loss_dict['loss_ce'].item() if 'loss_ce' in loss_dict else 0,
            'box_loss': loss_dict['loss_bbox'].item() if 'loss_bbox' in loss_dict else 0
        })
    
    # Average losses
    avg_loss = running_loss / len(data_loader)
    avg_loss_classifier = running_loss_classifier / len(data_loader)
    avg_loss_bbox = running_loss_bbox / len(data_loader)
    
    # Get evaluation metrics
    metrics = evaluator.compute()
    
    # Standardize metric names for compatibility
    metrics['map'] = metrics.get('mAP@0.5', 0.0)  # Use mAP@0.5 as the primary metric
    metrics['map_50'] = metrics.get('mAP@0.5', 0.0)
    metrics['map_75'] = metrics.get('mAP@0.75', 0.0)
    
    # Log to wandb
    if LOGGING_CONFIG['use_wandb']:
        wandb.log({
            'val/loss': avg_loss,
            'val/loss_ce': avg_loss_classifier,
            'val/loss_bbox': avg_loss_bbox,
            'val/mAP': metrics['map'],
            'val/mAP_50': metrics['map_50'],
            'val/mAP_75': metrics['map_75'],
            'epoch': epoch
        })
    
    return avg_loss, metrics


def main():
    """Main training function."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(Path(LOG_DIR) / "training.log"),
            logging.StreamHandler()
        ]
    )
    
    # Initialize wandb if enabled
    if LOGGING_CONFIG['use_wandb']:
        wandb.init(project=LOGGING_CONFIG['wandb_project'], config={
            **TRAIN_CONFIG,
            **MODEL_CONFIG
        })
    
    # Determine device and check for mixed precision support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"Mixed precision available: {MIXED_PRECISION_AVAILABLE}")
        if TRAIN_CONFIG['mixed_precision'] and not MIXED_PRECISION_AVAILABLE:
            logging.warning("Mixed precision requested but not available! Falling back to full precision.")
            TRAIN_CONFIG['mixed_precision'] = False
    else:
        logging.warning("CUDA not available, using CPU. Training will be slow!")
        TRAIN_CONFIG['mixed_precision'] = False
    
    # Get datasets
    train_dataset, val_dataset = get_dataset_splits(train_ratio=0.85)  # Use 85% for training
    logging.info(f"Training set size: {len(train_dataset)}")
    logging.info(f"Validation set size: {len(val_dataset)}")
    
    # Setup data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True  # Drop last batch to ensure consistent batch size
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Build model
    model = build_model()
    model.to(device)
    
    # Log model size
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model has {param_count:,} trainable parameters")
    
    # Build criterion (loss function)
    criterion = build_criterion(MODEL_CONFIG)
    
    # Setup optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                  if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() 
                      if "backbone" in n and p.requires_grad],
            "lr": TRAIN_CONFIG['lr_backbone'],
        },
    ]
    
    optimizer = optim.AdamW(param_dicts, lr=TRAIN_CONFIG['lr'],
                          weight_decay=TRAIN_CONFIG['weight_decay'])
    
    # Set up gradient scaler for mixed precision
    scaler = None
    if MIXED_PRECISION_AVAILABLE and TRAIN_CONFIG['mixed_precision'] and device.type == 'cuda':
        scaler = GradScaler('cuda')  # Specify 'cuda' device
    
    # Learning rate scheduler
    num_epochs = TRAIN_CONFIG['epochs']
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * TRAIN_CONFIG.get('warmup_epochs', 0)
    
    if TRAIN_CONFIG.get('lr_scheduler') == 'cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_factor=TRAIN_CONFIG.get('min_lr') / TRAIN_CONFIG['lr']
        )
    else:
        # Traditional step scheduler
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, TRAIN_CONFIG.get('lr_drop', 50))
    
    # Initialize tracking variables
    best_map = 0.0
    patience_counter = 0
    start_epoch = 0
    
    # Try to load checkpoint if available
    checkpoint_path = Path(MODEL_DIR) / "checkpoint_latest.pth"
    if checkpoint_path.exists():
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        if 'map' in checkpoint:
            best_map = checkpoint['map']
        if 'scaler' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logging.info(f"Resuming from epoch {start_epoch} with best mAP: {best_map:.4f}")
    
    # Training loop
    logging.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(start_epoch, num_epochs):
        logging.info(f"Starting epoch {epoch}/{num_epochs-1}")
        
        # Train for one epoch
        train_loss, train_loss_ce, train_loss_bbox = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, scaler, 
            lr_scheduler if TRAIN_CONFIG.get('lr_scheduler') == 'cosine' else None
        )
        
        logging.info(f"Train loss: {train_loss:.4f}, CE: {train_loss_ce:.4f}, "
                    f"Bbox: {train_loss_bbox:.4f}")
        
        # Evaluate
        val_loss, metrics = evaluate(model, criterion, val_loader, device, epoch)
        
        # Fix metrics mapping - get appropriate values from evaluator results
        if 'mAP@0.5' in metrics:
            metrics['map_50'] = metrics['mAP@0.5']
        if 'mAP@0.75' in metrics:
            metrics['map_75'] = metrics['mAP@0.75']
        # Use mAP@0.5 as the overall map if not present
        if 'map' not in metrics:
            metrics['map'] = metrics.get('mAP@0.5', 0.0)
            
        map_score = metrics['map']
        
        logging.info(f"Validation loss: {val_loss:.4f}, mAP: {map_score:.4f}, "
                    f"mAP_50: {metrics['map_50']:.4f}, mAP_75: {metrics['map_75']:.4f}")
        
        # Update learning rate for non-cosine schedulers
        if TRAIN_CONFIG.get('lr_scheduler') != 'cosine':
            lr_scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'map': map_score,
        }
        
        if scaler is not None:
            checkpoint['scaler'] = scaler.state_dict()
        
        torch.save(checkpoint, Path(MODEL_DIR) / f"checkpoint_latest.pth")
        
        # Save model every 10 epochs
        if epoch % 10 == 0 and epoch > 0:
            torch.save(checkpoint, Path(MODEL_DIR) / f"checkpoint_epoch{epoch}.pth")
        
        # Save best model
        if map_score > best_map:
            best_map = map_score
            torch.save(checkpoint, Path(MODEL_DIR) / "model_best.pth")
            logging.info(f"✓ New best model with mAP: {best_map:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Save model by metric threshold
        if map_score > 0.75:  # Save models that achieve >75% mAP
            torch.save(checkpoint, Path(MODEL_DIR) / f"model_map{map_score:.4f}.pth")
            
        # Early stopping
        if patience_counter >= TRAIN_CONFIG['early_stopping_patience']:
            logging.info(f"Early stopping triggered after {epoch} epochs")
            break
        
        # Clean up memory
        torch.cuda.empty_cache()
        gc.collect()
    
    # Final logging
    logging.info(f"Training completed. Best mAP: {best_map:.4f}")
    
    # Close wandb
    if LOGGING_CONFIG['use_wandb']:
        wandb.finish()


if __name__ == "__main__":
    main() 