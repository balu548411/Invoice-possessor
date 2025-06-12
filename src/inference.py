import os
import argparse
import json
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path

from data_processor import InvoiceDataProcessor 
from model import InvoiceProcessor

class InvoiceExtractor:
    def __init__(self, model_path, img_size=(800, 800), max_text_length=512, num_fields=10):
        """Initialize the invoice extractor with a trained model"""
        self.img_size = img_size
        self.processor = InvoiceProcessor(
            img_size=img_size,
            max_text_length=max_text_length,
            num_fields=num_fields
        )
        
        # Load model weights
        self.processor.load_model(model_path)
        
    def extract_text_from_image(self, image_path):
        """Extract text from image using OCR as fallback"""
        # Use pytesseract for OCR
        img = cv2.imread(str(image_path))
        if img is None:
            return "Error loading image"
        
        # Convert to grayscale for better OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get a binary image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Extract text
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(thresh, config=custom_config)
        
        return text
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess an image for the model"""
        # Check if file is PDF
        if str(image_path).lower().endswith('.pdf'):
            # Convert first page of PDF to image
            images = convert_from_path(image_path, first_page=1, last_page=1)
            if not images:
                raise ValueError(f"Could not convert PDF to image: {image_path}")
            # Convert PIL Image to numpy array
            img = np.array(images[0])
            # Convert RGB to BGR for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            # Load image directly
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
        
        # Resize and normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def process_invoice(self, file_path):
        """Process an invoice file (image or PDF) and extract information"""
        try:
            # Load and preprocess image
            img = self.load_and_preprocess_image(file_path)
            
            # Extract text using OCR as fallback for text branch
            text = self.extract_text_from_image(file_path)
            
            # Run inference with the model
            results = self.processor.predict(img, text)
            
            # Add filename to results
            results["filename"] = os.path.basename(file_path)
            
            return results
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return {"error": str(e), "filename": os.path.basename(file_path)}

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize invoice extractor
    extractor = InvoiceExtractor(
        model_path=args.model_path,
        img_size=(args.img_height, args.img_width),
        max_text_length=args.max_text_length,
        num_fields=args.num_fields
    )
    
    # Get list of files to process
    input_path = Path(args.input_path)
    if input_path.is_dir():
        # Process all files in directory
        files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.jpeg')) + \
                list(input_path.glob('*.png')) + list(input_path.glob('*.pdf'))
    else:
        # Process single file
        files = [input_path]
    
    # Process each file
    results = []
    for file_path in files:
        print(f"Processing {file_path}...")
        result = extractor.process_invoice(file_path)
        results.append(result)
        
        # Save individual result
        output_file = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    # Save all results to a single file if multiple files were processed
    if len(files) > 1:
        all_results_file = os.path.join(args.output_dir, "all_results.json")
        with open(all_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"All results saved to {all_results_file}")
    
    print(f"Processing complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on invoice images/PDFs')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input file or directory containing files')
    parser.add_argument('--output_dir', type=str, default='../inference_results',
                        help='Directory to save output JSON files')
    parser.add_argument('--model_path', type=str, default='../models/final_model.h5',
                        help='Path to trained model weights')
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