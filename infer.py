import os
import sys
import json
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import MODEL_DIR, INFERENCE_CONFIG
from src.inference.predictor import DocumentPredictor, format_results_as_json


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with Document Parsing model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the input document image')
    parser.add_argument('--model', type=str, default=str(Path(MODEL_DIR) / 'model_best.pth'),
                        help='Path to the model weights')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the output visualization')
    parser.add_argument('--json_output', type=str, default=None,
                        help='Path to save the JSON output')
    parser.add_argument('--threshold', type=float, default=INFERENCE_CONFIG['confidence_threshold'],
                        help='Confidence threshold for predictions')
    return parser.parse_args()


def main():
    """Main inference function."""
    # Parse arguments
    args = parse_args()
    
    # Check if image exists
    if not os.path.isfile(args.image):
        print(f"Error: Image file {args.image} does not exist")
        return
    
    # Check if model exists
    if not os.path.isfile(args.model):
        print(f"Error: Model file {args.model} does not exist")
        return
    
    # Initialize predictor
    print(f"Loading model from {args.model}")
    predictor = DocumentPredictor(args.model, confidence_threshold=args.threshold)
    
    # Run prediction
    print(f"Processing image: {args.image}")
    results = predictor.predict(args.image)
    print(f"Found {len(results)} entities")
    
    # Display and save results
    visualized = predictor.visualize_prediction(args.image, results)
    
    # Convert to RGB for matplotlib
    if visualized is not None:
        # Show results
        plt.figure(figsize=(12, 12))
        plt.imshow(visualized)
        plt.axis('off')
        
        # Save visualization if requested
        if args.output:
            plt.savefig(args.output, bbox_inches='tight', pad_inches=0.0)
            print(f"Visualization saved to {args.output}")
        else:
            plt.show()
    
    # Format and save JSON output if requested
    if args.json_output:
        # Format results as JSON
        json_output = format_results_as_json(results)
        
        # Save to file
        with open(args.json_output, 'w') as f:
            json.dump(json_output, f, indent=2)
        
        print(f"JSON output saved to {args.json_output}")
    
    # Print detected entities
    print("\nDetected Entities:")
    for res in results:
        print(f"{res['entity_type']}: Confidence={res['confidence']:.4f}, "
              f"Box={res['bounding_box']}")


if __name__ == "__main__":
    main() 