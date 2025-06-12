import json
import os
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvoiceDataProcessor:
    """
    Comprehensive data processor for invoice OCR data similar to Azure Form Recognizer format
    """
    
    def __init__(self, images_dir: str, labels_dir: str):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_files = list(self.images_dir.glob("*.jpg"))
        self.label_files = list(self.labels_dir.glob("*.json"))
        
    def load_annotation(self, json_path: str) -> Dict:
        """Load and parse JSON annotation file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading {json_path}: {e}")
            return None
    
    def extract_entities(self, annotation: Dict) -> Dict[str, List]:
        """
        Extract key invoice entities from annotation data
        Similar to Azure Form Recognizer's entity extraction
        """
        entities = {
            'invoice_number': [],
            'invoice_date': [],
            'due_date': [],
            'vendor_name': [],
            'vendor_address': [],
            'customer_name': [],
            'customer_address': [],
            'total_amount': [],
            'subtotal': [],
            'tax_amount': [],
            'line_items': [],
            'payment_terms': [],
            'all_text': []
        }
        
        if not annotation or 'pages' not in annotation:
            return entities
            
        for page in annotation['pages']:
            if 'lines' not in page:
                continue
                
            for line in page['lines']:
                content = line.get('content', '').strip()
                content_lower = content.lower()
                
                # Store all text
                entities['all_text'].append({
                    'text': content,
                    'bbox': line.get('polygon', []),
                    'confidence': self._get_avg_confidence(line)
                })
                
                # Extract specific entities using pattern matching
                self._extract_invoice_patterns(content, content_lower, line, entities)
        
        return entities
    
    def _extract_invoice_patterns(self, content: str, content_lower: str, line: Dict, entities: Dict):
        """Extract specific invoice patterns"""
        import re
        
        # Invoice Number patterns
        if any(keyword in content_lower for keyword in ['invoice', 'inv', '#']):
            # Look for invoice number patterns
            inv_patterns = [
                r'invoice\s*#?\s*(\w+)',
                r'inv\s*#?\s*(\w+)',
                r'#\s*(\w+)'
            ]
            for pattern in inv_patterns:
                match = re.search(pattern, content_lower)
                if match:
                    entities['invoice_number'].append({
                        'text': match.group(1),
                        'bbox': line.get('polygon', []),
                        'confidence': self._get_avg_confidence(line)
                    })
        
        # Date patterns
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{1,2}\s+[a-zA-Z]+\s+\d{2,4}',
            r'[a-zA-Z]+\s+\d{1,2},?\s+\d{2,4}'
        ]
        for pattern in date_patterns:
            match = re.search(pattern, content)
            if match:
                if 'due' in content_lower:
                    entities['due_date'].append({
                        'text': match.group(0),
                        'bbox': line.get('polygon', []),
                        'confidence': self._get_avg_confidence(line)
                    })
                elif any(word in content_lower for word in ['date', 'invoice']):
                    entities['invoice_date'].append({
                        'text': match.group(0),
                        'bbox': line.get('polygon', []),
                        'confidence': self._get_avg_confidence(line)
                    })
        
        # Amount patterns
        amount_patterns = [
            r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
            r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
        ]
        for pattern in amount_patterns:
            match = re.search(pattern, content)
            if match and any(char.isdigit() for char in content):
                if any(keyword in content_lower for keyword in ['total', 'amount']):
                    entities['total_amount'].append({
                        'text': match.group(0),
                        'bbox': line.get('polygon', []),
                        'confidence': self._get_avg_confidence(line)
                    })
                elif 'tax' in content_lower or 'gst' in content_lower:
                    entities['tax_amount'].append({
                        'text': match.group(0),
                        'bbox': line.get('polygon', []),
                        'confidence': self._get_avg_confidence(line)
                    })
        
        # Company/Vendor name patterns (usually at the top)
        if any(keyword in content_lower for keyword in ['inc', 'ltd', 'llc', 'corp', 'company', 'technologies', 'pvt']):
            entities['vendor_name'].append({
                'text': content,
                'bbox': line.get('polygon', []),
                'confidence': self._get_avg_confidence(line)
            })
    
    def _get_avg_confidence(self, line: Dict) -> float:
        """Calculate average confidence for a line"""
        if 'words' in line:
            confidences = [word.get('confidence', 0.0) for word in line['words']]
            return sum(confidences) / len(confidences) if confidences else 0.0
        return 0.0
    
    def create_bounding_box_labels(self, annotation: Dict, img_width: int, img_height: int) -> Dict:
        """Create bounding box labels for object detection"""
        boxes = []
        labels = []
        
        if not annotation or 'pages' not in annotation:
            return {'boxes': [], 'labels': []}
            
        for page in annotation['pages']:
            if 'words' not in page:
                continue
                
            for word in page['words']:
                polygon = word.get('polygon', [])
                if len(polygon) >= 4:
                    # Convert polygon to bounding box
                    x_coords = [p['x'] for p in polygon]
                    y_coords = [p['y'] for p in polygon]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Normalize coordinates
                    x_min_norm = x_min / img_width
                    y_min_norm = y_min / img_height
                    x_max_norm = x_max / img_width
                    y_max_norm = y_max / img_height
                    
                    boxes.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm])
                    labels.append(1)  # All text elements get label 1
        
        return {'boxes': boxes, 'labels': labels}
    
    def process_dataset(self) -> pd.DataFrame:
        """Process entire dataset and create structured DataFrame"""
        data_records = []
        
        for img_path in self.image_files:
            # Find corresponding JSON file
            json_name = img_path.stem + '.json'
            json_path = self.labels_dir / json_name
            
            if not json_path.exists():
                logger.warning(f"No annotation found for {img_path.name}")
                continue
            
            # Load image to get dimensions
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                h, w = image.shape[:2]
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                continue
            
            # Load annotation
            annotation = self.load_annotation(json_path)
            if annotation is None:
                continue
            
            # Extract entities
            entities = self.extract_entities(annotation)
            
            # Create bounding boxes
            bbox_data = self.create_bounding_box_labels(annotation, w, h)
            
            record = {
                'image_path': str(img_path),
                'json_path': str(json_path),
                'image_width': w,
                'image_height': h,
                'entities': entities,
                'bboxes': bbox_data,
                'num_words': len(bbox_data['boxes']),
                'avg_confidence': self._calculate_avg_confidence(annotation)
            }
            
            data_records.append(record)
        
        return pd.DataFrame(data_records)
    
    def _calculate_avg_confidence(self, annotation: Dict) -> float:
        """Calculate average confidence across all words in annotation"""
        confidences = []
        
        if not annotation or 'pages' not in annotation:
            return 0.0
            
        for page in annotation['pages']:
            if 'words' in page:
                for word in page['words']:
                    conf = word.get('confidence', 0.0)
                    confidences.append(conf)
        
        return sum(confidences) / len(confidences) if confidences else 0.0


class InvoiceDataset(Dataset):
    """
    PyTorch Dataset for invoice processing
    """
    
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 transform=None, 
                 max_boxes: int = 500,
                 image_size: Tuple[int, int] = (512, 512)):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.max_boxes = max_boxes
        self.image_size = image_size
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        image = cv2.imread(row['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get bounding boxes and labels
        boxes = row['bboxes']['boxes']
        labels = row['bboxes']['labels']
        
        # Pad or truncate boxes to max_boxes
        if len(boxes) > self.max_boxes:
            boxes = boxes[:self.max_boxes]
            labels = labels[:self.max_boxes]
        else:
            # Pad with zeros
            pad_length = self.max_boxes - len(boxes)
            boxes.extend([[0, 0, 0, 0]] * pad_length)
            labels.extend([0] * pad_length)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'entities': row['entities'],
            'image_path': row['image_path']
        }


def get_train_transforms(image_size: Tuple[int, int] = (512, 512)):
    """Get training transforms with augmentation"""
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.Blur(blur_limit=3, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms(image_size: Tuple[int, int] = (512, 512)):
    """Get validation transforms without augmentation"""
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def create_data_loaders(train_df: pd.DataFrame, 
                       val_df: pd.DataFrame,
                       batch_size: int = 8,
                       num_workers: int = 4,
                       image_size: Tuple[int, int] = (512, 512)):
    """Create training and validation data loaders"""
    
    # Check if we should use multiprocessing
    import multiprocessing as mp
    import psutil
    
    # Get available shared memory
    try:
        shm_stats = psutil.disk_usage('/dev/shm')
        available_shm_gb = shm_stats.free / (1024**3)
        
        # If shared memory is low, reduce workers or disable multiprocessing
        if available_shm_gb < 1.0:  # Less than 1GB
            logger.warning(f"Low shared memory detected ({available_shm_gb:.2f}GB). Reducing workers.")
            num_workers = min(num_workers, 2)
        if available_shm_gb < 0.5:  # Less than 500MB
            logger.warning("Very low shared memory. Disabling multiprocessing.")
            num_workers = 0
    except:
        logger.warning("Could not check shared memory. Using conservative settings.")
        num_workers = min(num_workers, 2)
    
    train_dataset = InvoiceDataset(
        train_df, 
        transform=get_train_transforms(image_size),
        image_size=image_size
    )
    
    val_dataset = InvoiceDataset(
        val_df,
        transform=get_val_transforms(image_size),
        image_size=image_size
    )
    
    # DataLoader configuration to avoid shared memory issues
    dataloader_kwargs = {
        'batch_size': batch_size,
        'pin_memory': True,
        'collate_fn': collate_fn,
        'persistent_workers': num_workers > 0,  # Keep workers alive between epochs
        'prefetch_factor': 2 if num_workers > 0 else None,  # Reduce memory usage
    }
    
    # Add multiprocessing settings only if using workers
    if num_workers > 0:
        dataloader_kwargs.update({
            'num_workers': num_workers,
            'multiprocessing_context': mp.get_context('spawn'),  # Use spawn instead of fork
        })
    else:
        dataloader_kwargs['num_workers'] = 0
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs
    )
    
    logger.info(f"Created DataLoaders with {num_workers} workers")
    return train_loader, val_loader


def collate_fn(batch):
    """Custom collate function for batching"""
    images = torch.stack([item['image'] for item in batch])
    boxes = torch.stack([item['boxes'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    entities = [item['entities'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'images': images,
        'boxes': boxes,
        'labels': labels,
        'entities': entities,
        'image_paths': image_paths
    } 