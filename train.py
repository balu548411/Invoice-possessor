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

# Import project modules
from src.config import TRAIN_CONFIG, MODEL_CONFIG, BATCH_SIZE, NUM_WORKERS, MODEL_DIR, LOG_DIR, LOGGING_CONFIG
from src.data.dataset import get_dataset_splits, collate_fn
from src.model.document_parser import build_model
from src.model.losses import build_criterion
from src.evaluation.metrics import DocumentParsingEvaluator


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch):
    """Train the model for one epoch."""
    model.train()
    criterion.train()
    
    running_loss = 0.0
    running_loss_classifier = 0.0
    running_loss_bbox = 0.0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} - Training")
    
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
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        if TRAIN_CONFIG['clip_max_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['clip_max_norm'])
        optimizer.step()
        
        # Update running losses
        running_loss += losses.item()
        running_loss_classifier += loss_dict['loss_ce'].item() if 'loss_ce' in loss_dict else 0
        running_loss_bbox += loss_dict['loss_bbox'].item() if 'loss_bbox' in loss_dict else 0
        
        # Update progress bar
        pbar.set_postfix({
            'loss': losses.item(),
            'cls_loss': loss_dict['loss_ce'].item() if 'loss_ce' in loss_dict else 0,
            'box_loss': loss_dict['loss_bbox'].item() if 'loss_bbox' in loss_dict else 0
        })
        
        # Log to wandb
        if LOGGING_CONFIG['use_wandb'] and (pbar.n % LOGGING_CONFIG['log_interval'] == 0 or pbar.n == len(data_loader)):
            wandb.log({
                'train/loss': losses.item(),
                'train/loss_ce': loss_dict['loss_ce'].item() if 'loss_ce' in loss_dict else 0,
                'train/loss_bbox': loss_dict['loss_bbox'].item() if 'loss_bbox' in loss_dict else 0,
                'train/lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'iteration': epoch * len(data_loader) + pbar.n
            })
    
    # Average losses
    avg_loss = running_loss / len(data_loader)
    avg_loss_classifier = running_loss_classifier / len(data_loader)
    avg_loss_bbox = running_loss_bbox / len(data_loader)
    
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
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Get datasets
    train_dataset, val_dataset = get_dataset_splits()
    logging.info(f"Training set size: {len(train_dataset)}")
    logging.info(f"Validation set size: {len(val_dataset)}")
    
    # Setup data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Build model
    model = build_model()
    model.to(device)
    
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
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, TRAIN_CONFIG['lr_drop'])
    
    # Training loop
    num_epochs = TRAIN_CONFIG['epochs']
    best_map = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch}/{num_epochs-1}")
        
        # Train for one epoch
        train_loss, train_loss_ce, train_loss_bbox = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch)
        
        logging.info(f"Train loss: {train_loss:.4f}, CE: {train_loss_ce:.4f}, "
                    f"Bbox: {train_loss_bbox:.4f}")
        
        # Evaluate
        val_loss, metrics = evaluate(model, criterion, val_loader, device, epoch)
        map_score = metrics['map']
        
        logging.info(f"Validation loss: {val_loss:.4f}, mAP: {map_score:.4f}")
        
        # Update learning rate
        lr_scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'map': map_score,
        }
        
        torch.save(checkpoint, Path(MODEL_DIR) / f"checkpoint_latest.pth")
        
        # Save best model
        if map_score > best_map:
            best_map = map_score
            torch.save(checkpoint, Path(MODEL_DIR) / "model_best.pth")
            logging.info(f"Saving best model with mAP: {best_map:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= TRAIN_CONFIG['early_stopping_patience']:
            logging.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Final logging
    logging.info(f"Training completed. Best mAP: {best_map:.4f}")
    
    # Close wandb
    if LOGGING_CONFIG['use_wandb']:
        wandb.finish()


if __name__ == "__main__":
    main() 