import os
import sys
import json
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import time
from tqdm import tqdm

from src.config import MODEL_DIR, INFERENCE_CONFIG, IMAGE_SIZE
from src.inference.predictor import DocumentPredictor, format_results_as_json


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with Document Parsing model')
    parser.add_argument('--image', type=str, required=False,
                        help='Path to the input document image')
    parser.add_argument('--dir', type=str, default=None,
                        help='Directory of images to process (batch mode)')
    parser.add_argument('--model', type=str, default=str(Path(MODEL_DIR) / 'model_best.pth'),
                        help='Path to the model weights')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the output visualization or directory in batch mode')
    parser.add_argument('--json_output', type=str, default=None,
                        help='Path to save the JSON output or directory in batch mode')
    parser.add_argument('--threshold', type=float, default=INFERENCE_CONFIG['confidence_threshold'],
                        help='Confidence threshold for predictions')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing multiple images (only used with --dir)')
    return parser.parse_args()


def process_single_image(predictor, image_path, output_path=None, json_output_path=None, show_results=True):
    """Process a single image and save/display results."""
    # Check if image exists
    if not os.path.isfile(image_path):
        print(f"Error: Image file {image_path} does not exist")
        return None
        
    # Run prediction
    start_time = time.time()
    results = predictor.predict(image_path)
    inference_time = time.time() - start_time
    print(f"Found {len(results)} entities in {inference_time:.4f} seconds")
    
    # Visualize results
    visualized = predictor.visualize_prediction(image_path, results)
    
    # Save visualization if requested
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(visualized, cv2.COLOR_RGB2BGR))
        print(f"Visualization saved to {output_path}")
    
    # Show results
    if show_results:
        plt.figure(figsize=(12, 12))
        plt.imshow(visualized)
        plt.axis('off')
        plt.show()
    
    # Format and save JSON output if requested
    if json_output_path:
        # Format results as JSON
        json_output = format_results_as_json(results)
        
        # Save to file
        with open(json_output_path, 'w') as f:
            json.dump(json_output, f, indent=2)
        
        print(f"JSON output saved to {json_output_path}")
    
    # Print detected entities
    print("\nDetected Entities:")
    for res in results:
        print(f"{res['entity_type']}: Confidence={res['confidence']:.4f}, "
              f"Box={res['bounding_box']}")
              
    return results
    

def process_image_directory(predictor, dir_path, output_dir=None, json_output_dir=None, batch_size=1):
    """Process all images in a directory."""
    # Check if directory exists
    if not os.path.isdir(dir_path):
        print(f"Error: Directory {dir_path} does not exist")
        return
        
    # Create output directories if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if json_output_dir:
        os.makedirs(json_output_dir, exist_ok=True)
        
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(Path(dir_path).glob(f'*{ext}')))
    
    print(f"Found {len(image_paths)} images in {dir_path}")
    
    # Process in batches if requested
    if batch_size > 1 and predictor.device.type == 'cuda':
        total_entities = 0
        total_time = 0
        
        # Process images in batches
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            
            # Create batch
            batch_images = []
            for img_path in batch_paths:
                image = predictor.preprocess_image(str(img_path))
                batch_images.append(image)
                
            # Stack images into batch
            if batch_images:
                start_time = time.time()
                batch_results = predictor.predict_batch(batch_images)
                batch_time = time.time() - start_time
                total_time += batch_time
                
                # Process each result
                for j, (img_path, results) in enumerate(zip(batch_paths, batch_results)):
                    if output_dir:
                        output_path = Path(output_dir) / f"{img_path.stem}_result{img_path.suffix}"
                        # Visualize and save
                        visualized = predictor.visualize_prediction(str(img_path), results)
                        cv2.imwrite(str(output_path), cv2.cvtColor(visualized, cv2.COLOR_RGB2BGR))
                        
                    if json_output_dir:
                        json_path = Path(json_output_dir) / f"{img_path.stem}_result.json"
                        # Format and save JSON
                        json_output = format_results_as_json(results)
                        with open(json_path, 'w') as f:
                            json.dump(json_output, f, indent=2)
                            
                    total_entities += len(results)
        
        # Print summary
        print(f"\nProcessed {len(image_paths)} images in {total_time:.2f} seconds")
        print(f"Average processing time: {total_time/len(image_paths):.4f} seconds per image")
        print(f"Total entities found: {total_entities}")
        
    else:
        # Process images one by one
        total_time = 0
        total_entities = 0
        
        for img_path in tqdm(image_paths):
            if output_dir:
                output_path = Path(output_dir) / f"{img_path.stem}_result{img_path.suffix}"
            else:
                output_path = None
                
            if json_output_dir:
                json_path = Path(json_output_dir) / f"{img_path.stem}_result.json"
            else:
                json_path = None
                
            # Process image
            start_time = time.time()
            results = predictor.predict(str(img_path))
            processing_time = time.time() - start_time
            total_time += processing_time
            total_entities += len(results)
            
            # Visualize and save if requested
            if output_path:
                visualized = predictor.visualize_prediction(str(img_path), results)
                cv2.imwrite(str(output_path), cv2.cvtColor(visualized, cv2.COLOR_RGB2BGR))
                
            # Save JSON if requested
            if json_path:
                json_output = format_results_as_json(results)
                with open(json_path, 'w') as f:
                    json.dump(json_output, f, indent=2)
        
        # Print summary
        print(f"\nProcessed {len(image_paths)} images in {total_time:.2f} seconds")
        print(f"Average processing time: {total_time/len(image_paths):.4f} seconds per image")
        print(f"Total entities found: {total_entities}")


def main():
    """Main inference function."""
    # Parse arguments
    args = parse_args()
    
    # Check if batch processing or single image
    if not args.image and not args.dir:
        print("Error: Either --image or --dir must be specified")
        return
    
    # Check if model exists
    if not os.path.isfile(args.model):
        print(f"Error: Model file {args.model} does not exist")
        return
    
    # Initialize predictor
    print(f"Loading model from {args.model}")
    predictor = DocumentPredictor(
        args.model,
        confidence_threshold=args.threshold,
        image_size=IMAGE_SIZE
    )
    
    # Check if running on GPU
    device_name = "GPU" if predictor.device.type == "cuda" else "CPU"
    print(f"Using {device_name} for inference")
    
    # Process single image or directory
    if args.image:
        # Single image mode
        process_single_image(
            predictor,
            args.image,
            output_path=args.output,
            json_output_path=args.json_output
        )
    else:
        # Batch processing mode
        process_image_directory(
            predictor,
            args.dir,
            output_dir=args.output,
            json_output_dir=args.json_output,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main() 