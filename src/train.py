import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import time

# Import our modules
from data_processor import InvoiceDataProcessor
from model import InvoiceProcessor

def main(args):
    # Create directories for outputs
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize data processor
    print("Initializing data processor...")
    data_processor = InvoiceDataProcessor(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        img_size=(args.img_height, args.img_width)
    )
    
    # Create dataset
    print("Creating dataset...")
    (train_images, train_labels), (val_images, val_labels), matched_files = data_processor.create_dataset(
        split_ratio=args.train_split
    )
    
    # Save matched files for reference
    with open(os.path.join(args.log_dir, 'matched_files.json'), 'w') as f:
        json.dump(matched_files, f, indent=2)
    
    # Extract text from labels for the text branch
    print("Preparing text data...")
    train_text = [label.get('document_text', '') for label in train_labels]
    val_text = [label.get('document_text', '') for label in val_labels]
    
    # Initialize model
    print("Building model...")
    model = InvoiceProcessor(
        img_size=(args.img_height, args.img_width),
        max_text_length=args.max_text_length,
        num_fields=args.num_fields
    )
    
    # Create directory for checkpoints
    os.makedirs('model_checkpoints', exist_ok=True)
    
    # Train model
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    
    history = model.train(
        train_images=train_images,
        train_text=train_text,
        val_images=val_images,
        val_text=val_text,
        train_labels=train_labels,
        val_labels=val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save final model
    model.save_model(os.path.join(args.model_dir, 'final_model.h5'))
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot overall accuracy (average of field accuracies)
    accuracies = []
    val_accuracies = []
    
    for field in history.history:
        if field.endswith('accuracy') and not field.startswith('val_'):
            accuracies.append(history.history[field])
        elif field.endswith('accuracy') and field.startswith('val_'):
            val_accuracies.append(history.history[field])
    
    if accuracies:
        avg_acc = np.mean(np.array(accuracies), axis=0)
        avg_val_acc = np.mean(np.array(val_accuracies), axis=0)
        
        plt.subplot(1, 2, 2)
        plt.plot(avg_acc)
        plt.plot(avg_val_acc)
        plt.title('Average Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, 'training_history.png'))
    
    # Save training history
    with open(os.path.join(args.log_dir, 'training_history.json'), 'w') as f:
        json.dump(history.history, f, indent=2)
    
    print(f"Model saved to {os.path.join(args.model_dir, 'final_model.h5')}")
    print(f"Training history saved to {os.path.join(args.log_dir, 'training_history.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train invoice processor model')
    parser.add_argument('--image_dir', type=str, default='../training_data/images', 
                        help='Directory containing image files')
    parser.add_argument('--label_dir', type=str, default='../training_data/labels', 
                        help='Directory containing JSON label files')
    parser.add_argument('--model_dir', type=str, default='../models', 
                        help='Directory to save the trained model')
    parser.add_argument('--log_dir', type=str, default='../logs', 
                        help='Directory to save logs and plots')
    parser.add_argument('--checkpoint_dir', type=str, default='../model_checkpoints', 
                        help='Directory to save model checkpoints')
    parser.add_argument('--img_height', type=int, default=800, 
                        help='Image height for model input')
    parser.add_argument('--img_width', type=int, default=800, 
                        help='Image width for model input')
    parser.add_argument('--max_text_length', type=int, default=512, 
                        help='Maximum text length for BERT input')
    parser.add_argument('--num_fields', type=int, default=10, 
                        help='Number of invoice fields to extract')
    parser.add_argument('--train_split', type=float, default=0.8, 
                        help='Ratio of data to use for training')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Training batch size')
    
    args = parser.parse_args()
    main(args) 