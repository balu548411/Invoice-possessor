import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import json
import numpy as np
import random
from datetime import datetime
import gc  # Add garbage collection
import torch.multiprocessing

from config import *
from data_preprocess import prepare_dataset, get_dataloaders, visualize_sample
from model_arch import InvoiceTransformer, count_parameters


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def optimize_memory():
    """Optimize memory usage by clearing cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0


def get_lr_scheduler(optimizer, num_warmup_epochs, num_training_epochs, last_epoch=-1):
    """Get learning rate scheduler"""
    def lr_lambda(current_epoch):
        """Learning rate scheduler lambda function"""
        if current_epoch < num_warmup_epochs:
            # Linear warmup
            return float(current_epoch + 1) / float(max(1, num_warmup_epochs))
        else:
            # Cosine annealing
            progress = float(current_epoch - num_warmup_epochs) / float(max(1, num_training_epochs - num_warmup_epochs))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
            
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def train_epoch(model, dataloader, optimizer, scheduler, epoch, device, tb_writer, gradient_accumulation_steps=2):
    """Train for one epoch with gradient accumulation for memory efficiency"""
    model.train()
    total_loss = 0
    steps = 0
    
    optimizer.zero_grad()  # Zero gradients at the beginning
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits, loss = model(images, labels)
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass with mixed precision if enabled
        if USE_AMP:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            # Clip gradients and optimizer step with mixed precision
            if USE_AMP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()
                
            scheduler.step()
            optimizer.zero_grad()
            
            # Free up memory
            if batch_idx % 5 == 0:
                optimize_memory()
        
        # Update metrics (use the original loss value for reporting)
        total_loss += loss.item() * gradient_accumulation_steps
        steps += 1
        
        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
        
        # Log to tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        tb_writer.add_scalar("train/loss", loss.item() * gradient_accumulation_steps, global_step)
        tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
        
        # Save checkpoint
        if (batch_idx + 1) % SAVE_CHECKPOINT_STEPS == 0:
            checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"checkpoint_epoch{epoch}_step{batch_idx}.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)
    
    avg_loss = total_loss / steps
    return avg_loss


def validate(model, dataloader, device, epoch, tb_writer):
    """Validate the model"""
    model.eval()
    total_loss = 0
    steps = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits, loss = model(images, labels)
            
            # Update metrics
            total_loss += loss.item()
            steps += 1
            
            # Free up memory periodically
            if batch_idx % 5 == 0:
                optimize_memory()
    
    avg_loss = total_loss / steps
    
    # Log to tensorboard
    tb_writer.add_scalar("val/loss", avg_loss, epoch)
    
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, scheduler, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"]
    return epoch


def train_model(args):
    """Main training function"""
    # Set random seed
    set_seed(RANDOM_SEED)
    
    # Create tensorboard writer
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(LOG_DIR, f"run_{timestamp}")
    tb_writer = SummaryWriter(log_dir=log_dir)
    
    # Prepare dataset
    print("Preparing datasets...")
    datasets = prepare_dataset(split=True)
    dataloaders = get_dataloaders(datasets, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    tokenizer = dataloaders["tokenizer"]
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = InvoiceTransformer(vocab_size=tokenizer.vocab_size)
    model.to(device)
    
    # Print model summary
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    tb_writer.add_text("model/parameters", f"{num_params:,}")
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_lr_scheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    # Set gradient accumulation steps for memory efficiency
    # Smaller batch size with more accumulation steps is memory efficient
    gradient_accumulation_steps = 2  # Effective batch size = BATCH_SIZE * gradient_accumulation_steps
    
    # Load checkpoint if resume training
    start_epoch = 0
    if args.resume and os.path.exists(args.checkpoint):
        print(f"Resuming training from {args.checkpoint}")
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.checkpoint)
        start_epoch += 1  # Start from the next epoch
    
    # Print training configuration
    print(f"Training configuration:")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"- Effective batch size: {BATCH_SIZE * gradient_accumulation_steps}")
    print(f"- Learning rate: {LEARNING_RATE}")
    print(f"- Mixed precision: {USE_AMP}")
    
    # Start training
    print("Starting training...")
    best_val_loss = float('inf')
    
    # Pre-optimize memory
    optimize_memory()
    
    # Set multiprocessing sharing strategy
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Train
        train_loss = train_epoch(
            model, 
            dataloaders["train"], 
            optimizer, 
            scheduler, 
            epoch, 
            device, 
            tb_writer,
            gradient_accumulation_steps
        )
        
        # Optimize memory before validation
        optimize_memory()
        
        # Validate
        val_loss = validate(model, dataloaders["val"], device, epoch, tb_writer)
        
        # Print epoch summary
        print(f"Epoch {epoch} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, best_model_path)
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        # Regular epoch checkpoint
        checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"checkpoint_epoch{epoch}.pt")
        save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)
        
        # Check for early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
        # Optimize memory between epochs
        optimize_memory()
    
    # Save final model
    final_model_path = os.path.join(MODEL_SAVE_DIR, "final_model.pt")
    save_checkpoint(model, optimizer, scheduler, NUM_EPOCHS-1, final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}")
    
    # Close tensorboard writer
    tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train invoice processing model")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    train_model(args) 