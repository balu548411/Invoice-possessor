import os
import argparse
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

from model import InvoiceProcessor

def main(args):
    """Generate and save a visualization of the model architecture"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    print("Initializing model...")
    model = InvoiceProcessor(
        img_size=(args.img_height, args.img_width),
        max_text_length=args.max_text_length,
        num_fields=args.num_fields
    )
    
    # Generate model plot
    print("Generating model visualization...")
    plot_model(
        model.model,
        to_file=os.path.join(args.output_dir, 'model_architecture.png'),
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',  # Top to bottom layout
        expand_nested=True,
        dpi=96
    )
    
    # Generate summary
    with open(os.path.join(args.output_dir, 'model_summary.txt'), 'w') as f:
        model.model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print(f"Model visualization saved to {os.path.join(args.output_dir, 'model_architecture.png')}")
    print(f"Model summary saved to {os.path.join(args.output_dir, 'model_summary.txt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize the invoice processor model architecture')
    parser.add_argument('--output_dir', type=str, default='../model_visualization',
                        help='Directory to save the model visualization')
    parser.add_argument('--img_height', type=int, default=800,
                        help='Image height for model input')
    parser.add_argument('--img_width', type=int, default=800,
                        help='Image width for model input')
    parser.add_argument('--max_text_length', type=int, default=512,
                        help='Maximum text length for BERT input')
    parser.add_argument('--num_fields', type=int, default=10,
                        help='Number of invoice fields to extract')
    
    args = parser.parse_args()
    main(args) 