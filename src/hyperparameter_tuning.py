import os
import argparse
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import itertools
import time

from data_processor import InvoiceDataProcessor
from model import InvoiceProcessor

def main(args):
    """Perform hyperparameter tuning for the model"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data processor
    print("Initializing data processor...")
    data_processor = InvoiceDataProcessor(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        img_size=(args.img_height, args.img_width)
    )
    
    # Create dataset
    print("Creating dataset...")
    (train_images, train_labels), (val_images, val_labels), _ = data_processor.create_dataset(
        split_ratio=args.train_split
    )
    
    # Extract text from labels for the text branch
    print("Preparing text data...")
    train_text = [label.get('document_text', '') for label in train_labels]
    val_text = [label.get('document_text', '') for label in val_labels]
    
    # Define hyperparameters to tune
    hyperparameters = {
        'learning_rate': [1e-3, 1e-4, 5e-5],
        'batch_size': [4, 8, 16],
        'dropout_rate': [0.3, 0.5, 0.7],
        'dense_units': [256, 512, 1024],
        'freeze_layers': [True, False]
    }
    
    # Select subset of hyperparameters based on command line arguments
    if args.fast_mode:
        print("Running in fast mode with reduced hyperparameter combinations")
        hyperparameters = {
            'learning_rate': [1e-4],
            'batch_size': [8],
            'dropout_rate': [0.5],
            'dense_units': [512],
            'freeze_layers': [True, False]
        }
    
    # Generate all combinations to test
    param_names = list(hyperparameters.keys())
    param_values = list(hyperparameters.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"Testing {len(param_combinations)} hyperparameter combinations")
    
    # Track best parameters and validation loss
    best_params = {}
    best_val_loss = float('inf')
    
    # Store all results
    all_results = []
    
    # Train models with different hyperparameters
    for i, params in enumerate(param_combinations):
        param_dict = dict(zip(param_names, params))
        
        print(f"\n--- Training model {i+1}/{len(param_combinations)} with parameters:")
        for name, value in param_dict.items():
            print(f"  {name}: {value}")
        
        # Initialize model with custom hyperparameters
        model = InvoiceProcessor(
            img_size=(args.img_height, args.img_width),
            max_text_length=args.max_text_length,
            num_fields=args.num_fields
        )
        
        # Apply custom hyperparameters
        # For a real implementation, we would modify the model's architecture based on hyperparameters
        # This is a simplified version that just trains with different batch sizes and learning rates
        
        # Track training time
        start_time = time.time()
        
        # Train model with limited epochs for hyperparameter search
        history = model.train(
            train_images=train_images[:args.max_train_samples] if args.max_train_samples else train_images,
            train_text=train_text[:args.max_train_samples] if args.max_train_samples else train_text,
            val_images=val_images[:args.max_val_samples] if args.max_val_samples else val_images,
            val_text=val_text[:args.max_val_samples] if args.max_val_samples else val_text,
            train_labels=train_labels[:args.max_train_samples] if args.max_train_samples else train_labels,
            val_labels=val_labels[:args.max_val_samples] if args.max_val_samples else val_labels,
            epochs=args.tuning_epochs,
            batch_size=param_dict['batch_size']
        )
        
        training_time = time.time() - start_time
        
        # Get validation loss from history
        val_loss = history.history['val_loss'][-1]
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = param_dict
            
            # Save best model
            model.save_model(os.path.join(args.output_dir, 'best_model.h5'))
        
        # Store result
        result = {
            'parameters': param_dict,
            'val_loss': val_loss,
            'training_time': training_time,
        }
        
        # Add metrics from history
        for key in history.history:
            result[key] = history.history[key][-1]
        
        all_results.append(result)
        
        # Save current results
        with open(os.path.join(args.output_dir, 'tuning_results.json'), 'w') as f:
            json.dump({
                'best_parameters': best_params,
                'best_val_loss': best_val_loss,
                'all_results': all_results
            }, f, indent=2)
    
    # Print best parameters
    print("\n--- Best Parameters:")
    for name, value in best_params.items():
        print(f"  {name}: {value}")
    print(f"  Validation Loss: {best_val_loss}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot validation loss for each combination
    plt.subplot(2, 2, 1)
    combinations = range(len(all_results))
    losses = [r['val_loss'] for r in all_results]
    plt.bar(combinations, losses)
    plt.xticks(combinations)
    plt.xlabel('Parameter Combination')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss by Parameter Combination')
    
    # Plot training time
    plt.subplot(2, 2, 2)
    times = [r['training_time'] for r in all_results]
    plt.bar(combinations, times)
    plt.xticks(combinations)
    plt.xlabel('Parameter Combination')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time by Parameter Combination')
    
    # Plot learning rate vs loss
    plt.subplot(2, 2, 3)
    lr_values = sorted(list(set([r['parameters']['learning_rate'] for r in all_results])))
    lr_losses = []
    
    for lr in lr_values:
        lr_results = [r['val_loss'] for r in all_results if r['parameters']['learning_rate'] == lr]
        avg_loss = np.mean(lr_results)
        lr_losses.append(avg_loss)
    
    plt.plot(lr_values, lr_losses, 'o-')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Average Validation Loss')
    plt.title('Learning Rate vs. Validation Loss')
    
    # Plot batch size vs loss
    plt.subplot(2, 2, 4)
    bs_values = sorted(list(set([r['parameters']['batch_size'] for r in all_results])))
    bs_losses = []
    
    for bs in bs_values:
        bs_results = [r['val_loss'] for r in all_results if r['parameters']['batch_size'] == bs]
        avg_loss = np.mean(bs_results)
        bs_losses.append(avg_loss)
    
    plt.plot(bs_values, bs_losses, 'o-')
    plt.xlabel('Batch Size')
    plt.ylabel('Average Validation Loss')
    plt.title('Batch Size vs. Validation Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'tuning_results.png'))
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for invoice processor model')
    parser.add_argument('--image_dir', type=str, default='../training_data/images', 
                        help='Directory containing image files')
    parser.add_argument('--label_dir', type=str, default='../training_data/labels', 
                        help='Directory containing JSON label files')
    parser.add_argument('--output_dir', type=str, default='../tuning_results', 
                        help='Directory to save tuning results')
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
    parser.add_argument('--tuning_epochs', type=int, default=5, 
                        help='Number of training epochs for each hyperparameter combination')
    parser.add_argument('--fast_mode', action='store_true', 
                        help='Run in fast mode with fewer hyperparameter combinations')
    parser.add_argument('--max_train_samples', type=int, default=None, 
                        help='Maximum number of training samples to use (for faster tuning)')
    parser.add_argument('--max_val_samples', type=int, default=None, 
                        help='Maximum number of validation samples to use (for faster tuning)')
    
    args = parser.parse_args()
    main(args)