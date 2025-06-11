import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGE_SIZE, IMAGES_DIR, LABELS_DIR, AUG_CONFIG


class DocumentDataset(Dataset):
    """Dataset for document parsing with key-value extraction."""
    
    def __init__(self, image_paths, label_paths, image_size=IMAGE_SIZE, is_training=True):
        """
        Args:
            image_paths: List of paths to document images
            label_paths: List of paths to JSON label files
            image_size: Target image size (height, width)
            is_training: Whether to apply training augmentations
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.image_size = image_size
        self.is_training = is_training
        
        # Set up transforms
        if is_training:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                # Mild geometric transforms for document images
                A.RandomRotate90(p=0.1),
                A.Rotate(limit=AUG_CONFIG["rotate_limit"], p=0.3),
                A.RandomScale(scale_limit=AUG_CONFIG["scale_limit"], p=0.3),
                A.Affine(translate_percent=AUG_CONFIG["shift_limit"], p=0.3),
                
                # Document-specific augmentations
                A.OneOf([
                    A.ElasticTransform(
                        alpha=120, sigma=120 * 0.05, p=0.5
                    ),  # Simulate paper wrinkles
                    A.GridDistortion(p=0.5),  # Simulate perspective distortion
                    A.OpticalDistortion(distort_limit=0.05, p=0.5),  # Simulate lens effects
                ], p=AUG_CONFIG.get("elastic_transform_prob", 0.2)),
                
                # Color and noise transforms
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=AUG_CONFIG["brightness_contrast_limit"],
                        contrast_limit=AUG_CONFIG["brightness_contrast_limit"],
                        p=0.8
                    ),
                    A.RandomGamma(p=0.5),
                    A.GaussNoise(p=0.5),  # Use default var limit
                ], p=0.5),
                
                # Document artifacts simulation
                A.OneOf([
                    A.GaussianBlur(blur_limit=AUG_CONFIG.get("blur_limit", 3), p=0.3),  # Blur
                    A.MotionBlur(blur_limit=3, p=0.3),  # Scanner motion blur
                    A.MedianBlur(blur_limit=3, p=0.3),  # Reduce noise
                ], p=0.2),
                
                # Shadow and light simulation
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=AUG_CONFIG.get("random_shadow_prob", 0.2)),
                
                # Random dropout augmentation
                A.OneOf([
                    A.RandomGridShuffle(grid=(3, 3), p=0.3),
                    A.GridDropout(ratio=0.2, p=0.3)
                ], p=AUG_CONFIG.get("cutout_prob", 0.1)),
                
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load label
        label_path = self.label_paths[idx]
        with open(label_path, 'r') as f:
            label_data = json.load(f)
        
        # Extract entity bounding boxes and classes
        boxes = []
        classes = []
        texts = []
        
        # Process the extracted entities from the label
        label_fields = label_data.get('data', {})
        
        for field_name, field_info in label_fields.items():
            if field_name == 'Items':  # Special handling for item lists
                continue  # Items are complex and handled separately
                
            value = field_info.get('value', '')
            confidence = field_info.get('confidence', 0.0)
            
            # Skip low confidence or empty values
            if confidence is None or confidence < 0.5 or not value:
                continue
                
            # Find bounding regions if available
            bounding_regions = []
            if 'bounding_regions' in field_info:
                bounding_regions = field_info['bounding_regions']
            
            for region in bounding_regions:
                # Extract polygon points for this region
                polygon = region.get('polygon', [])
                if not polygon:
                    continue
                    
                # Convert polygon to bounding box [x_min, y_min, x_max, y_max]
                x_coords = [p['x'] for p in polygon]
                y_coords = [p['y'] for p in polygon]
                
                x_min = min(x_coords)
                y_min = min(y_coords)
                x_max = max(x_coords)
                y_max = max(y_coords)
                
                # Normalize box coordinates based on image dimensions
                h, w = image.shape[:2]
                x_min, x_max = x_min / w, x_max / w
                y_min, y_max = y_min / h, y_max / h
                
                # Create box with normalized coordinates [x_min, y_min, x_max, y_max]
                box = [x_min, y_min, x_max, y_max]
                boxes.append(box)
                
                # Map field name to class id
                from config import MODEL_CONFIG
                class_id = MODEL_CONFIG["entity_classes"].get(field_name, -1)
                if class_id >= 0:
                    classes.append(class_id)
                    texts.append(str(value))
        
        # Apply transformations to the image
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        # Prepare the target
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(classes, dtype=torch.int64) if classes else torch.zeros((0,), dtype=torch.int64),
            "texts": texts,
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([image.shape[1], image.shape[2]]),
        }
        
        return image, target


def get_dataset_splits(train_ratio=0.8, shuffle=True, seed=42):
    """
    Split the dataset into training and validation sets.
    
    Args:
        train_ratio: Fraction of data to use for training
        shuffle: Whether to shuffle before splitting
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset
    """
    # Get all image paths
    image_paths = list(IMAGES_DIR.glob('*.jpg'))
    
    # Find corresponding label paths
    all_data = []
    for img_path in image_paths:
        label_path = LABELS_DIR / f"{img_path.stem}.json"
        if label_path.exists():
            all_data.append((img_path, label_path))
    
    # Shuffle if needed
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_data)
    
    # Split into train and validation
    split_idx = int(len(all_data) * train_ratio)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    # Create datasets
    train_img_paths, train_label_paths = zip(*train_data) if train_data else ([], [])
    val_img_paths, val_label_paths = zip(*val_data) if val_data else ([], [])
    
    train_dataset = DocumentDataset(train_img_paths, train_label_paths, is_training=True)
    val_dataset = DocumentDataset(val_img_paths, val_label_paths, is_training=False)
    
    return train_dataset, val_dataset


def collate_fn(batch):
    """
    Custom collate function for batching data with variable number of targets.
    """
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, targets 