import os
import json
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from pdf2image import convert_from_path
import tensorflow as tf

class InvoiceDataProcessor:
    def __init__(self, image_dir, label_dir, img_size=(800, 800)):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
        
    def load_image(self, image_path):
        """Load and preprocess an image file"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, self.img_size)
        
        # Normalize image
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def load_pdf(self, pdf_path):
        """Convert PDF to images"""
        images = convert_from_path(pdf_path)
        return [np.array(img) for img in images]
    
    def parse_label(self, label_path):
        """Parse JSON label file and extract key information"""
        with open(label_path, 'r') as f:
            data = json.load(f)
        
        # Extract relevant fields
        fields = {}
        
        if 'pages' in data and len(data['pages']) > 0:
            page = data['pages'][0]
            
            # Process lines and words to extract field values
            lines = page.get('lines', [])
            words = page.get('words', [])
            
            # Extract all text for document-level feature
            document_text = " ".join([line.get('content', '') for line in lines])
            fields['document_text'] = document_text
            
            # Create word coordinates and text for layout understanding
            word_info = []
            for word in words:
                if 'polygon' in word and 'content' in word:
                    # Calculate normalized bounding box coordinates
                    polygon = word['polygon']
                    x_coords = [point.get('x', 0) for point in polygon]
                    y_coords = [point.get('y', 0) for point in polygon]
                    
                    if x_coords and y_coords:  # Make sure coordinates exist
                        x_min, y_min = min(x_coords), min(y_coords)
                        x_max, y_max = max(x_coords), max(y_coords)
                        
                        # Get page dimensions
                        page_width = page.get('width', 1)
                        page_height = page.get('height', 1)
                        
                        # Normalize coordinates
                        x_min_norm = x_min / page_width
                        y_min_norm = y_min / page_height
                        x_max_norm = x_max / page_width
                        y_max_norm = y_max / page_height
                        
                        word_info.append({
                            'text': word.get('content', ''),
                            'bbox': [x_min_norm, y_min_norm, x_max_norm, y_max_norm],
                            'confidence': word.get('confidence', 1.0)
                        })
            
            fields['words'] = word_info
            
            # Try to extract common invoice fields
            # This is a simplified approach - in a real application we would use NER or more sophisticated extraction
            fields['invoice_number'] = self.extract_field(document_text, words, ['invoice #', 'invoice no', 'invoice number'])
            fields['date'] = self.extract_field(document_text, words, ['date', 'invoice date'])
            fields['due_date'] = self.extract_field(document_text, words, ['due date'])
            fields['total_amount'] = self.extract_field(document_text, words, ['total', 'amount', 'balance', 'due'])
            fields['vendor_name'] = self.extract_field(document_text, words, ['vendor', 'company', 'from'])
            fields['customer_name'] = self.extract_field(document_text, words, ['bill to', 'customer', 'client'])
            
            
        return fields
    
    def extract_field(self, text, words, keywords):
        """Extract field value based on keyword context"""
        # Simple extraction logic - can be enhanced with regex or ML approaches
        text_lower = text.lower()
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                # Find the word that follows the keyword
                index = text_lower.find(keyword.lower())
                if index != -1:
                    # Extract the next few words after the keyword
                    segment = text[index + len(keyword):index + len(keyword) + 50].strip()
                    return segment
        
        return ""
    
    def create_dataset(self, split_ratio=0.8):
        """Create TF dataset from images and labels"""
        images = []
        labels = []
        matched_files = []
        
        for img_file in tqdm(self.image_files, desc="Processing files"):
            # Find corresponding JSON file (handle different naming conventions)
            img_name = os.path.splitext(img_file)[0]
            json_candidates = [
                f"{img_name}.json",
                f"{img_name.split('_')[0]}.json"  # Try base name
            ]
            
            json_file = None
            for candidate in json_candidates:
                if candidate in self.label_files:
                    json_file = candidate
                    break
            
            if json_file is None:
                print(f"No matching JSON found for {img_file}, skipping...")
                continue
                
            try:
                # Load image and label
                img_path = self.image_dir / img_file
                label_path = self.label_dir / json_file
                
                img = self.load_image(img_path)
                label_data = self.parse_label(label_path)
                
                # For now, we'll use the document text as our label
                # In a real application, we would create a more structured label format
                images.append(img)
                labels.append(label_data)
                matched_files.append((img_file, json_file))
            except Exception as e:
                print(f"Error processing {img_file} and {json_file}: {e}")
        
        # Split into training and validation sets
        n_samples = len(images)
        n_train = int(n_samples * split_ratio)
        
        # Create TF datasets
        train_images = images[:n_train]
        train_labels = labels[:n_train]
        val_images = images[n_train:]
        val_labels = labels[n_train:]
        
        print(f"Created dataset with {len(train_images)} training and {len(val_images)} validation samples")
        
        return (train_images, train_labels), (val_images, val_labels), matched_files
    
    def prepare_batch_data(self, images, labels):
        """Convert raw data into model-ready format"""
        # This would be expanded based on the model architecture
        # For example, converting label dictionaries to tensors
        return np.array(images), labels 