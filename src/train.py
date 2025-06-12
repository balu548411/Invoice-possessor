import os
import argparse
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

from data_processor import InvoiceDataProcessor
from dataset import InvoiceDataset
from model import create_invoice_model
from trainer import InvoiceModelTrainer

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def process_data(args):
    """Process and prepare data for training"""
    print("Processing invoice data...")
    
    processor = InvoiceDataProcessor(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_size=(args.image_size, args.image_size),
        max_examples=args.max_examples
    )
    
    # Process and save dataset
    output_path = processor.save_processed_dataset(args.processed_dir)
    
    print(f"Data processing complete. Processed data saved to {output_path}")
    
    return output_path

def train_model(args):
    """Train the model on processed data"""
    print("Starting model training...")
    
    # Paths
    processed_dir = Path(args.processed_dir)
    annotations_path = processed_dir / "annotations.json"
    images_dir = processed_dir / "processed_images"
    
    # Check if processed data exists
    if not annotations_path.exists() or not images_dir.exists():
        raise ValueError(f"Processed data not found at {processed_dir}")
    
    # Create dataset
    full_dataset = InvoiceDataset(
        annotations_file=str(annotations_path),
        images_dir=str(images_dir),
        tokenizer_name=args.tokenizer_name,
        max_seq_len=args.max_seq_len,
        image_size=(args.image_size, args.image_size),
        training=True
    )
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(dataset_size * args.train_ratio)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    # Create dataloaders
    train_dataloader = InvoiceDataset.create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_dataloader = InvoiceDataset.create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create model
    model = create_invoice_model(
        pretrained=True,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        embed_dim=args.embed_dim
    )
    
    # Create trainer
    trainer = InvoiceModelTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoints_dir=args.checkpoints_dir
    )
    
    # Train model
    history = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.patience
    )
    
    print(f"Training complete. Best model saved at: {args.checkpoints_dir}")
    
    return history

def main():
    parser = argparse.ArgumentParser(description="Train an invoice processing model")
    
    # Data processing arguments
    parser.add_argument("--images_dir", type=str, default="training_data/images",
                        help="Directory containing invoice images")
    parser.add_argument("--labels_dir", type=str, default="training_data/labels",
                        help="Directory containing label JSON files")
    parser.add_argument("--processed_dir", type=str, default="processed_data",
                        help="Directory to save processed data")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to process (for debugging)")
    parser.add_argument("--skip_processing", action="store_true",
                        help="Skip data processing step (use existing processed data)")
    
    # Model arguments
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased",
                        help="Name of the tokenizer to use")
    parser.add_argument("--vocab_size", type=int, default=30000,
                        help="Size of token vocabulary")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum sequence length for tokens")
    parser.add_argument("--embed_dim", type=int, default=256,
                        help="Embedding dimension for model")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Size to resize images to (square)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of data to use for training (vs. validation)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience for early stopping")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set default tensor type to float32
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_default_dtype(torch.float32)
    
    # Create output directories
    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    
    # Process data
    if not args.skip_processing:
        process_data(args)
    
    # Train model
    train_model(args)

if __name__ == "__main__":
    main() 