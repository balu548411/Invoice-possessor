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


def train_epoch(model, dataloader, optimizer, scheduler, epoch, device, tb_writer):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    steps = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, loss = model(images, labels)
        
        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        
        # Update learning rate
        scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        steps += 1
        
        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})
        
        # Log to tensorboard
        global_step = epoch * len(dataloader) + batch_idx
        tb_writer.add_scalar("train/loss", loss.item(), global_step)
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
            
            # Forward pass
            logits, loss = model(images, labels)
            
            # Update metrics
            total_loss += loss.item()
            steps += 1
    
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
    num_training_steps = NUM_EPOCHS * len(dataloaders["train"])
    scheduler = get_lr_scheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    # Load checkpoint if resume training
    start_epoch = 0
    if args.resume and os.path.exists(args.checkpoint):
        print(f"Resuming training from {args.checkpoint}")
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.checkpoint)
        start_epoch += 1  # Start from the next epoch
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    
    # Training loop
    print("Starting training...")
    best_val_loss = float("inf")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Train for one epoch
        train_loss = train_epoch(model, dataloaders["train"], optimizer, scheduler, epoch, device, tb_writer)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, dataloaders["val"], device, epoch, tb_writer)
        print(f"Epoch {epoch} | Val Loss: {val_loss:.4f}")
        
        # Check for early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
            
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, best_model_path)
            
        # Save regular checkpoint
        checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"checkpoint_epoch{epoch}.pt")
        save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(MODEL_SAVE_DIR, "final_model.pt")
    save_checkpoint(model, optimizer, scheduler, NUM_EPOCHS, final_model_path)
    
    # Close tensorboard writer
    tb_writer.close()
    
    print("Training completed!")
    return best_val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train invoice processing model")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    train_model(args) 