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
        
        # Print dataset size information
        print(f"Dataset initialized with {len(self.image_paths)} images and {len(self.label_paths)} labels")
        
        # Set up transforms - always include resize at the beginning to ensure consistent sizes
        # Remove always_apply parameter from Resize transform
        if is_training:
            self.transform = A.Compose([
                # Always resize first to ensure consistent tensor shapes
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
                
                # Final resize to ensure consistent size after augmentations
                A.Resize(height=image_size[0], width=image_size[1]),
                
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        
        # Preload and validate labels
        self.valid_indices = []
        for i, (img_path, label_path) in enumerate(zip(self.image_paths, self.label_paths)):
            # Check if both files exist
            if not os.path.exists(img_path) or not os.path.exists(label_path):
                continue
                
            try:
                # Validate that we can load the label file
                with open(label_path, 'r') as f:
                    label_data = json.load(f)
                # Check if label contains data field
                if 'data' not in label_data:
                    print(f"Warning: No 'data' field in label {label_path}")
                    continue
                    
                # Add to valid indices
                self.valid_indices.append(i)
            except Exception as e:
                print(f"Error loading {label_path}: {e}")
        
        print(f"Found {len(self.valid_indices)} valid samples out of {len(self.image_paths)}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Map to valid index
        idx = self.valid_indices[idx]
        
        # Load image
        image_path = self.image_paths[idx]
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            # Return a dummy sample
            return self._create_dummy_sample()
        
        # Get original image dimensions
        orig_h, orig_w = image.shape[:2]
        
        # Load label
        label_path = self.label_paths[idx]
        try:
            with open(label_path, 'r') as f:
                label_data = json.load(f)
        except Exception as e:
            print(f"Error reading label {label_path}: {e}")
            # Return a dummy sample
            return self._create_dummy_sample()
        
        # Extract entity bounding boxes and classes
        boxes = []
        classes = []
        texts = []
        
        # Process the extracted entities from the label
        label_fields = label_data.get('data', {})
        
        # Process each field in the label data
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
            
            # Get class id for this field
            from config import MODEL_CONFIG
            class_id = MODEL_CONFIG["entity_classes"].get(field_name, -1)
            if class_id < 0:
                continue  # Skip fields that don't have a class mapping
            
            for region in bounding_regions:
                # Extract polygon points for this region
                polygon = region.get('polygon', [])
                if not polygon:
                    continue
                    
                # Convert polygon to bounding box [x_min, y_min, x_max, y_max]
                x_coords = [p['x'] for p in polygon]
                y_coords = [p['y'] for p in polygon]
                
                try:
                    x_min = min(x_coords)
                    y_min = min(y_coords)
                    x_max = max(x_coords)
                    y_max = max(y_coords)
                    
                    # Skip invalid boxes
                    if x_min >= x_max or y_min >= y_max:
                        continue
                        
                    # Normalize box coordinates based on image dimensions
                    x_min, x_max = x_min / orig_w, x_max / orig_w
                    y_min, y_max = y_min / orig_h, y_max / orig_h
                    
                    # Clamp values to [0, 1]
                    x_min = max(0, min(1, x_min))
                    y_min = max(0, min(1, y_min))
                    x_max = max(0, min(1, x_max))
                    y_max = max(0, min(1, y_max))
                    
                    # Convert to center-width-height format
                    x_c = (x_min + x_max) / 2
                    y_c = (y_min + y_max) / 2
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    # Skip boxes that are too small
                    if w <= 0.01 or h <= 0.01:
                        continue
                    
                    # Create box with normalized coordinates [cx, cy, w, h]
                    box = [x_c, y_c, w, h]
                    boxes.append(box)
                    classes.append(class_id)
                    texts.append(str(value))
                except Exception as e:
                    print(f"Error processing box in {label_path}: {e}")
        
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
            "orig_size": torch.tensor([orig_h, orig_w]),
        }
        
        return image, target
    
    def _create_dummy_sample(self):
        """Create a dummy sample for error cases."""
        # Create a blank image of the target size
        dummy_image = torch.zeros((3, self.image_size[0], self.image_size[1]), dtype=torch.float32)
        
        # Create an empty target
        dummy_target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "texts": [],
            "image_id": torch.tensor([0]),
            "orig_size": torch.tensor([self.image_size[0], self.image_size[1]]),
        }
        
        return dummy_image, dummy_target


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
    image_paths = list(IMAGES_DIR.glob('*.jpg')) + list(IMAGES_DIR.glob('*.jpeg')) + list(IMAGES_DIR.glob('*.png'))
    print(f"Found {len(image_paths)} images")
    
    # Find corresponding label paths
    all_data = []
    for img_path in image_paths:
        label_path = LABELS_DIR / f"{img_path.stem}.json"
        if label_path.exists():
            all_data.append((img_path, label_path))
    
    print(f"Found {len(all_data)} image-label pairs")
    
    # Shuffle if needed
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_data)
    
    # Split into train and validation
    split_idx = int(len(all_data) * train_ratio)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"Training set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    
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