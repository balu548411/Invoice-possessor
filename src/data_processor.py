import os
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

class InvoiceDataProcessor:
    """
    Process invoice images and their associated JSON annotations.
    This class handles loading, preprocessing, and preparing the data for model training.
    """
    def __init__(self, 
                images_dir: str, 
                labels_dir: str,
                output_size: Tuple[int, int] = (800, 800),
                max_examples: int = None):
        """
        Initialize the data processor.
        
        Args:
            images_dir: Directory containing invoice images
            labels_dir: Directory containing JSON label files
            output_size: Resized image dimensions (height, width)
            max_examples: Maximum number of examples to process (for debugging)
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_size = output_size
        self.max_examples = max_examples
        
        # Validate directories
        if not self.images_dir.exists():
            raise ValueError(f"Images directory {images_dir} does not exist")
        if not self.labels_dir.exists():
            raise ValueError(f"Labels directory {labels_dir} does not exist")
            
        self.image_files = list(self.images_dir.glob("*.jpg"))
        if max_examples:
            self.image_files = self.image_files[:max_examples]
        
        print(f"Found {len(self.image_files)} image files")
        
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess an image"""
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image {image_path}")
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (self.output_size[1], self.output_size[0]))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def _find_matching_json(self, image_path: Path) -> Optional[Path]:
        """Find matching JSON file for an image"""
        image_name = image_path.stem
        
        # Try direct match
        direct_match = self.labels_dir / f"{image_name}.json"
        if direct_match.exists():
            return direct_match
        
        # Try removing trailing numbers
        base_name = ''.join([c for c in image_name if not c.isdigit()])
        for json_file in self.labels_dir.glob("*.json"):
            if base_name in json_file.stem:
                return json_file
        
        return None
    
    def _process_json_labels(self, json_path: Path, image_shape: Tuple[int, int]) -> Dict:
        """
        Process JSON labels and normalize bounding box coordinates
        """
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract fields and bounding boxes
        processed_data = {
            'fields': [],
            'boxes': [],
            'values': []
        }
        
        if 'pages' in data and len(data['pages']) > 0:
            page = data['pages'][0]  # For now, we process only the first page
            
            orig_height = page.get('height', 1)
            orig_width = page.get('width', 1)
            
            # Process each line (text extraction)
            for line in page.get('lines', []):
                text = line.get('content', '')
                polygon = line.get('polygon', [])
                
                if not polygon or not text:
                    continue
                
                # Extract and normalize coordinates
                x_coords = [p['x'] / orig_width for p in polygon]
                y_coords = [p['y'] / orig_height for p in polygon]
                
                # Calculate bounding box
                x_min, y_min = min(x_coords), min(y_coords)
                x_max, y_max = max(x_coords), max(y_coords)
                
                # Convert to [x_min, y_min, width, height] format, normalized
                box = [
                    x_min,
                    y_min,
                    x_max - x_min,
                    y_max - y_min
                ]
                
                # Add to processed data
                processed_data['fields'].append("text")
                processed_data['boxes'].append(box)
                processed_data['values'].append(text)
        
        return processed_data
    
    def extract_key_fields(self, processed_data: Dict) -> Dict:
        """
        Extract key fields from processed data dynamically.
        Uses heuristic rules to identify important fields, but also retains all keys from the original data.
        """
        # Initialize with common invoice fields for compatibility
        key_fields = {
            'invoice_number': None,
            'date': None,
            'due_date': None,
            'total_amount': None,
            'vendor_name': None,
            'vendor_address': None,
            'customer_name': None,
            'customer_address': None,
        }
        
        # Extract all available fields from raw data if present
        if 'raw_data' in processed_data and isinstance(processed_data['raw_data'], dict):
            # Copy all keys from the raw data
            for key, value in processed_data['raw_data'].items():
                key_fields[key] = value
        
        # Apply heuristic rules for common fields if not already extracted
        for field, box, value in zip(processed_data['fields'], processed_data['boxes'], processed_data['values']):
            value_lower = value.lower()
            
            # Invoice number detection
            if ('invoice' in value_lower and '#' in value_lower) or ('invoice' in value_lower and 'no' in value_lower):
                words = value.split()
                for i, word in enumerate(words):
                    if i < len(words) - 1 and ('invoice' in word.lower() and ('#' in words[i+1] or 'no' in word.lower())):
                        key_fields['invoice_number'] = words[i+1].replace('#', '')
            
            # Date detection
            if 'date' in value_lower and not 'due' in value_lower:
                parts = value.split()
                for part in parts:
                    if '/' in part or '-' in part:
                        key_fields['date'] = part
            
            # Due date
            if 'due' in value_lower and 'date' in value_lower:
                parts = value.split()
                for part in parts:
                    if '/' in part or '-' in part:
                        key_fields['due_date'] = part
            
            # Total amount
            if ('total' in value_lower or 'amount' in value_lower or 'balance' in value_lower) and ('$' in value or '.' in value):
                for word in value.split():
                    if ('$' in word or '.' in word) and any(c.isdigit() for c in word):
                        key_fields['total_amount'] = word.replace('$', '')
            
            # Vendor name detection
            if 'vendor' in value_lower or 'seller' in value_lower or 'from' in value_lower:
                if not key_fields.get('vendor_name'):
                    key_fields['vendor_name'] = value
            
            # Customer name detection
            if 'customer' in value_lower or 'bill to' in value_lower or 'buyer' in value_lower:
                if not key_fields.get('customer_name'):
                    key_fields['customer_name'] = value
            
            # Tax amount
            if 'tax' in value_lower and ('$' in value or '.' in value):
                for word in value.split():
                    if ('$' in word or '.' in word) and any(c.isdigit() for c in word):
                        key_fields['tax_amount'] = word.replace('$', '')
            
            # Subtotal
            if 'subtotal' in value_lower and ('$' in value or '.' in value):
                for word in value.split():
                    if ('$' in word or '.' in word) and any(c.isdigit() for c in word):
                        key_fields['subtotal'] = word.replace('$', '')
            
            # Payment terms
            if 'terms' in value_lower or 'payment' in value_lower and 'terms' in value_lower:
                key_fields['payment_terms'] = value
            
            # PO number
            if 'po' in value_lower and 'number' in value_lower:
                words = value.split()
                for i, word in enumerate(words):
                    if i < len(words) - 1 and ('po' in word.lower() and '#' in words[i+1]):
                        key_fields['po_number'] = words[i+1].replace('#', '')
        
        # Include any other field we can identify
        for field, value in zip(processed_data['fields'], processed_data['values']):
            field_lower = field.lower().replace(' ', '_')
            if field_lower not in key_fields and value:
                key_fields[field_lower] = value
        
        return key_fields
    
    def create_dataset(self):
        """
        Create a dataset of images and processed labels
        """
        dataset = []
        
        for i, image_path in enumerate(self.image_files):
            try:
                print(f"Processing image {i+1}/{len(self.image_files)}: {image_path.name}")
                
                # Find matching JSON
                json_path = self._find_matching_json(image_path)
                if json_path is None:
                    print(f"Warning: No matching JSON found for {image_path}")
                    continue
                
                # Load and process image
                image = self._load_image(image_path)
                
                # Process JSON
                processed_data = self._process_json_labels(json_path, image.shape[:2])
                
                # Extract key fields
                key_fields = self.extract_key_fields(processed_data)
                
                # Add to dataset
                dataset.append({
                    'image': image,
                    'image_path': str(image_path),
                    'json_path': str(json_path),
                    'processed_data': processed_data,
                    'key_fields': key_fields
                })
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        print(f"Successfully created dataset with {len(dataset)} examples")
        return dataset
    
    def save_processed_dataset(self, output_dir: str):
        """
        Save processed dataset to disk
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dataset = self.create_dataset()
        
        # Save images
        images_dir = output_path / "processed_images"
        images_dir.mkdir(exist_ok=True)
        
        # Save annotations
        annotations_path = output_path / "annotations.json"
        
        annotations = []
        for i, item in enumerate(dataset):
            # Save image
            img_filename = f"image_{i:05d}.jpg"
            img = (item['image'] * 255).astype(np.uint8)
            img_pil = Image.fromarray(img)
            img_pil.save(images_dir / img_filename)
            
            # Add annotation
            annotation = {
                'image_id': i,
                'image_filename': img_filename,
                'original_image': item['image_path'],
                'original_json': item['json_path'],
                'boxes': item['processed_data']['boxes'],
                'fields': item['processed_data']['fields'],
                'values': item['processed_data']['values'],
                'key_fields': item['key_fields']
            }
            annotations.append(annotation)
        
        # Save annotations
        with open(annotations_path, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2)
            
        print(f"Saved {len(annotations)} processed examples to {output_path}")
        return output_path

if __name__ == "__main__":
    # Example usage
    processor = InvoiceDataProcessor(
        images_dir="training_data/images",
        labels_dir="training_data/labels", 
        output_size=(800, 800),
        max_examples=10  # For testing
    )
    
    processor.save_processed_dataset("processed_data")