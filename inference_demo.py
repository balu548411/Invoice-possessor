#!/usr/bin/env python3
"""
Demo script for invoice processing inference
Demonstrates how to use the trained model for invoice processing
"""

import argparse
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

from src.inference import InvoiceProcessor, process_invoice_file

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Invoice Processing Inference Demo"
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--image_path", 
        type=str, 
        required=True,
        help="Path to invoice image to process"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="inference_results",
        help="Directory to save inference results"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Create visualization of detected entities"
    )
    parser.add_argument(
        "--confidence_threshold", 
        type=float, 
        default=0.5,
        help="Confidence threshold for entity extraction"
    )
    
    return parser.parse_args()


def visualize_results(image_path: str, results: dict, output_path: str):
    """Create visualization of detected entities on the invoice image"""
    
    # Load image
    image = Image.open(image_path)
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    ax.imshow(image)
    
    # Define colors for different entity types
    colors = {
        'INVOICE_NUM': 'red',
        'DATE': 'blue', 
        'TOTAL': 'green',
        'VENDOR': 'orange',
        'CUSTOMER': 'purple'
    }
    
    # Draw bounding boxes for detected entities
    if results['status'] == 'succeeded' and results['pages']:
        page = results['pages'][0]
        extracted_fields = page.get('extracted_fields', {})
        
        for entity_type, field_data in extracted_fields.items():
            color = colors.get(entity_type, 'yellow')
            confidence = field_data.get('confidence', 0.0)
            value = field_data.get('value', '')
            bboxes = field_data.get('bounding_boxes', [])
            
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                # Draw bounding box
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                label = f"{entity_type}: {value[:20]}..." if len(value) > 20 else f"{entity_type}: {value}"
                label += f" ({confidence:.2f})"
                
                ax.text(
                    x1, y1 - 5, label,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                    fontsize=8, color='white', weight='bold'
                )
    
    ax.set_title(f"Invoice Processing Results - {Path(image_path).name}", fontsize=14)
    ax.axis('off')
    
    # Create legend
    legend_elements = [patches.Patch(color=color, label=entity_type) 
                      for entity_type, color in colors.items()]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization saved to: {output_path}")


def print_results(results: dict):
    """Print formatted results to console"""
    
    print("\n" + "="*60)
    print("INVOICE PROCESSING RESULTS")
    print("="*60)
    
    if results['status'] != 'succeeded':
        print(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
        return
    
    print(f"✅ Status: {results['status']}")
    print(f"📊 Overall Confidence: {results['confidence']:.3f}")
    
    if results['pages']:
        page = results['pages'][0]
        print(f"📄 Page Size: {page['width']} x {page['height']} {page['unit']}")
        
        extracted_fields = page.get('extracted_fields', {})
        
        if extracted_fields:
            print("\n📋 EXTRACTED ENTITIES:")
            print("-" * 40)
            
            for entity_type, field_data in extracted_fields.items():
                value = field_data.get('value', 'N/A')
                confidence = field_data.get('confidence', 0.0)
                num_boxes = len(field_data.get('bounding_boxes', []))
                
                print(f"🏷️  {entity_type}")
                print(f"   Value: {value}")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Bounding Boxes: {num_boxes}")
                print()
        else:
            print("\n⚠️  No entities extracted")
    
    print("="*60)


def save_results(results: dict, output_path: str):
    """Save results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_path}")


def main():
    """Main inference demo function"""
    args = parse_arguments()
    
    logger.info("=== Invoice Processing Inference Demo ===")
    
    # Validate inputs
    model_path = Path(args.model_path)
    image_path = Path(args.image_path)
    
    if not model_path.exists():
        logger.error(f"Model checkpoint not found: {model_path}")
        return
    
    if not image_path.exists():
        logger.error(f"Image file not found: {image_path}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Process invoice
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Processing image: {image_path}")
        
        processor = InvoiceProcessor(
            str(model_path),
            confidence_threshold=args.confidence_threshold
        )
        
        results = processor.process_invoice(str(image_path))
        
        # Print results to console
        print_results(results)
        
        # Save results to JSON
        json_output = output_dir / f"{image_path.stem}_results.json"
        save_results(results, json_output)
        
        # Create visualization if requested
        if args.visualize:
            viz_output = output_dir / f"{image_path.stem}_visualization.png"
            visualize_results(str(image_path), results, viz_output)
        
        logger.info("✅ Inference completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        raise


def batch_demo():
    """Demo function for batch processing multiple invoices"""
    
    # Example usage for batch processing
    model_path = "path/to/your/model.ckpt"
    image_dir = "path/to/invoice/images"
    output_dir = "batch_results"
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(Path(image_dir).glob(f"*{ext}"))
    
    if not image_paths:
        logger.warning(f"No images found in {image_dir}")
        return
    
    # Initialize processor
    processor = InvoiceProcessor(model_path)
    
    # Process batch
    results = processor.batch_process(
        [str(p) for p in image_paths],
        output_dir=output_dir
    )
    
    # Print summary statistics
    successful = sum(1 for r in results if r['status'] == 'succeeded')
    total = len(results)
    avg_confidence = np.mean([r['confidence'] for r in results if r['confidence'] > 0])
    
    print(f"\n📊 BATCH PROCESSING SUMMARY")
    print(f"Total images: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Success rate: {successful/total*100:.1f}%")
    print(f"Average confidence: {avg_confidence:.3f}")


if __name__ == "__main__":
    main() 