import os
import argparse
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import random

def visualize_invoice_predictions(image_path, results_json, output_path=None):
    """Visualize predictions on an invoice image"""
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error loading image: {image_path}")
        return False
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load results
    with open(results_json, 'r') as f:
        results = json.load(f)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 15))
    
    # Display image
    ax.imshow(img)
    ax.axis('off')
    
    # Add title
    filename = os.path.basename(image_path)
    ax.set_title(f"Invoice: {filename}", fontsize=16)
    
    # Add text box with extraction results
    text_content = []
    fields_to_display = [
        'invoice_number', 'date', 'due_date', 'total_amount',
        'vendor_name', 'customer_name', 'tax_amount', 'subtotal'
    ]
    
    for field in fields_to_display:
        if field in results:
            value = results[field]
            if value:  # Only add non-empty values
                text_content.append(f"{field.replace('_', ' ').title()}: {value}")
    
    # Create text box
    textstr = '\n'.join(text_content)
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=props)
    
    # Save figure if output path is specified
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close(fig)
    return True

def main(args):
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get inference results
    results_dir = Path(args.results_dir)
    json_files = list(results_dir.glob('*.json'))
    
    # Filter out all_results.json
    json_files = [f for f in json_files if f.name != 'all_results.json']
    
    if not json_files:
        print(f"No JSON result files found in {args.results_dir}")
        return
    
    # Create PDF document to save all visualizations if specified
    if args.create_pdf:
        pdf_path = output_dir / 'extraction_results.pdf'
        pdf = PdfPages(str(pdf_path))
    else:
        pdf = None
    
    # Randomly select samples if requested
    if args.num_samples and len(json_files) > args.num_samples:
        json_files = random.sample(json_files, args.num_samples)
    
    # Process each result file
    successful_visualizations = 0
    for json_file in json_files:
        # Get base filename without extension
        base_name = json_file.stem
        
        # Try to find corresponding image
        image_path = None
        possible_extensions = ['.jpg', '.jpeg', '.png', '.pdf']
        for ext in possible_extensions:
            img_path = Path(args.image_dir) / f"{base_name}{ext}"
            if img_path.exists():
                image_path = img_path
                break
        
        if not image_path:
            print(f"Could not find image for {json_file}")
            continue
        
        # Output path for individual visualization
        output_path = output_dir / f"{base_name}_visualization.png"
        
        # Visualize
        success = visualize_invoice_predictions(image_path, json_file, output_path)
        
        if success:
            successful_visualizations += 1
            
            # Add to PDF if requested
            if pdf:
                pdf.savefig(plt.figure(figsize=(12, 15)))
                plt.close()
    
    # Close PDF if open
    if pdf:
        pdf.close()
        print(f"All visualizations saved to {pdf_path}")
    
    print(f"Created {successful_visualizations} visualizations")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize invoice extraction results')
    parser.add_argument('--image_dir', type=str, default='../training_data/images',
                        help='Directory containing invoice images')
    parser.add_argument('--results_dir', type=str, default='../inference_results',
                        help='Directory containing inference result JSON files')
    parser.add_argument('--output_dir', type=str, default='../visualizations',
                        help='Directory to save visualization images')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to visualize (default: all)')
    parser.add_argument('--create_pdf', action='store_true',
                        help='Create a PDF with all visualizations')
    
    args = parser.parse_args()
    main(args) 