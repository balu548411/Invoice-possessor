#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sample script to demonstrate the Invoice Processor pipeline.
This script shows how to:
1. Train the model
2. Evaluate the model
3. Run inference on new invoice images
"""

import os
import sys
import argparse
import torch
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all dependencies are installed"""
    try:
        import torch
        import torchvision
        import transformers
        import albumentations
        import cv2
        import numpy as np
        import tqdm
        import nltk
        import matplotlib
        logger.info("All dependencies are installed.")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Please install all dependencies with: pip install -r requirements.txt")
        return False


def train(args):
    """Train the model"""
    from train import train_model
    
    # Create a dummy args object if needed
    class Args:
        resume = False
        checkpoint = ""
        
    if args is None:
        args = Args()
    
    logger.info("Starting model training...")
    train_model(args)


def evaluate(args):
    """Evaluate the model"""
    from eval import main as eval_main
    
    # Create a dummy args object if needed
    class Args:
        model_path = "./outputs/models/best_model.pt"
        split = "test"
        num_samples = None
        output_dir = "./outputs/evaluation"
        visualize = True
        
    if args is None:
        args = Args()
        
    logger.info(f"Evaluating model from {args.model_path}...")
    eval_main(args)


def inference(args):
    """Run inference on new images"""
    from inference import main as inference_main
    
    # Create a dummy args object if needed
    class Args:
        model_path = "./outputs/models/best_model.pt"
        tokenizer_path = "./outputs/models/tokenizer.json"
        input_path = "../training_data/images"  # Default to using training images
        output_dir = "./outputs/inference"
        temperature = 0.7
        visualize = True
        
    if args is None:
        args = Args()
        
    logger.info(f"Running inference using model from {args.model_path}...")
    results = inference_main(args)
    return results


def demo_usage():
    """Show programmatic usage of the InvoiceProcessor"""
    from inference import InvoiceProcessor
    
    # Path to model and tokenizer
    model_path = "./outputs/models/best_model.pt"
    tokenizer_path = "./outputs/models/tokenizer.json"
    
    # Initialize processor
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        processor = InvoiceProcessor(
            model_path=model_path,
            tokenizer_path=tokenizer_path
        )
        
        # Find a sample image
        sample_dir = "../training_data/images"
        if os.path.exists(sample_dir):
            sample_images = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if sample_images:
                # Process a single image
                sample_image = sample_images[0]
                logger.info(f"Processing sample image: {sample_image}")
                result = processor.process_image(sample_image, visualize=True)
                logger.info(f"Extracted data: {result['json']}")
                
                # Process multiple images (just using the same image multiple times for demo)
                logger.info("Processing multiple images...")
                batch_results = processor.process_batch(
                    [sample_images[0]] * 3,  # Just use the same image 3 times for demo
                    output_dir="./outputs/demo",
                    visualize=True
                )
                logger.info(f"Processed {len(batch_results)} images")
            else:
                logger.warning(f"No images found in {sample_dir}")
        else:
            logger.warning(f"Sample directory {sample_dir} not found")
    else:
        logger.warning("Model or tokenizer not found. Please train a model first.")


def main():
    parser = argparse.ArgumentParser(description="Invoice Processor Demo")
    parser.add_argument("--mode", type=str, default="all", 
                        choices=["train", "evaluate", "inference", "demo", "all"],
                        help="Mode to run (train/evaluate/inference/demo/all)")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint", type=str, default="", 
                        help="Path to checkpoint to resume from")
    parser.add_argument("--model_path", type=str, default="./outputs/models/best_model.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, default="./outputs/models/tokenizer.json",
                       help="Path to tokenizer file")
    parser.add_argument("--input_path", type=str, default="../training_data/images",
                       help="Path to input image file or directory")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Directory to save results")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return 1
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run selected mode
    if args.mode == "train" or args.mode == "all":
        train(args)
        
    if args.mode == "evaluate" or args.mode == "all":
        # Check if model exists
        if os.path.exists(args.model_path):
            evaluate(args)
        else:
            logger.warning(f"Model not found at {args.model_path}. Skipping evaluation.")
        
    if args.mode == "inference" or args.mode == "all":
        # Check if model and tokenizer exist
        if os.path.exists(args.model_path) and os.path.exists(args.tokenizer_path):
            inference(args)
        else:
            logger.warning(f"Model or tokenizer not found. Skipping inference.")
        
    if args.mode == "demo" or args.mode == "all":
        demo_usage()
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 