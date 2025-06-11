import os
import json
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import random
from sklearn.model_selection import train_test_split
import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvoiceTokenizer:
    """
    Simple tokenizer for handling invoice JSON data to sequence format.
    """
    def __init__(self, max_length=512):
        self.max_length = max_length
        self.special_tokens = {
            "pad": PAD_TOKEN,
            "start": START_TOKEN,
            "end": END_TOKEN,
            "unk": UNK_TOKEN,
        }
        self.token2idx = {
            PAD_TOKEN: 0,
            START_TOKEN: 1,
            END_TOKEN: 2,
            UNK_TOKEN: 3,
        }
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self.vocab_size = len(self.token2idx)
        
    def build_vocab(self, json_files, min_freq=2):
        """Build vocabulary from list of JSON files"""
        logger.info("Building vocabulary...")
        token_freqs = {}
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                # Extract all values from the JSON structure
                values = []
                self._extract_values(data, values)
                
                # Count token frequencies
                for value in values:
                    if isinstance(value, str):
                        tokens = value.strip().split()
                        for token in tokens:
                            if token not in token_freqs:
                                token_freqs[token] = 1
                            else:
                                token_freqs[token] += 1
            except Exception as e:
                logger.warning(f"Error processing {json_file}: {e}")
                
        # Add tokens that meet minimum frequency to vocabulary
        idx = len(self.token2idx)
        for token, freq in token_freqs.items():
            if freq >= min_freq and token not in self.token2idx:
                self.token2idx[token] = idx
                self.idx2token[idx] = token
                idx += 1
                
        self.vocab_size = len(self.token2idx)
        logger.info(f"Vocabulary built with {self.vocab_size} tokens")
        
    def _extract_values(self, obj, values):
        """Recursive function to extract all values from a nested structure"""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "value" and isinstance(v, (str, int, float)):
                    values.append(str(v))
                else:
                    self._extract_values(v, values)
        elif isinstance(obj, list):
            for item in obj:
                self._extract_values(item, values)
                
    def encode_json(self, json_obj) -> List[int]:
        """Convert JSON to a sequence of tokens"""
        flat_data = self._flatten_json(json_obj)
        text_representation = self._json_to_text(flat_data)
        
        # Tokenize
        tokens = [self.special_tokens["start"]]
        tokens.extend(text_representation.split())
        tokens.append(self.special_tokens["end"])
        
        # Convert to indices
        token_ids = []
        for token in tokens:
            token_ids.append(self.token2idx.get(token, self.token2idx[UNK_TOKEN]))
            
        # Pad or truncate
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length-1] + [self.token2idx[END_TOKEN]]
        else:
            token_ids = token_ids + [self.token2idx[PAD_TOKEN]] * (self.max_length - len(token_ids))
            
        return token_ids
    
    def decode_ids(self, token_ids) -> str:
        """Convert sequence of token IDs back to text"""
        tokens = [self.idx2token[idx] for idx in token_ids if idx in self.idx2token]
        
        # Remove special tokens
        if START_TOKEN in tokens:
            tokens = tokens[tokens.index(START_TOKEN)+1:]
        if END_TOKEN in tokens:
            tokens = tokens[:tokens.index(END_TOKEN)]
        
        # Remove padding
        tokens = [t for t in tokens if t != PAD_TOKEN]
        
        return " ".join(tokens)
    
    def _flatten_json(self, json_obj, prefix=""):
        """Flatten nested JSON to key-value pairs"""
        items = {}
        for k, v in json_obj.items():
            new_key = k if prefix == "" else prefix + "." + k
            
            if isinstance(v, dict) and "value" in v and "confidence" in v:
                # Handle Azure OCR format
                items[new_key] = v["value"]
            elif isinstance(v, dict):
                items.update(self._flatten_json(v, new_key))
            elif isinstance(v, list):
                # Convert list to string representation
                items[new_key] = str(v)
            else:
                items[new_key] = v
                
        return items
    
    def _json_to_text(self, flat_json):
        """Convert flattened JSON to text sequence"""
        text_parts = []
        for k, v in flat_json.items():
            if v is not None and v != "":
                text_parts.append(f"{k} = {v}")
        
        return " ; ".join(text_parts)
    
    def save(self, path):
        """Save tokenizer vocabulary"""
        data = {
            "token2idx": self.token2idx,
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)
        logger.info(f"Tokenizer saved to {path}")
        
    def load(self, path):
        """Load tokenizer vocabulary"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.token2idx = data["token2idx"]
        self.idx2token = {int(k): v for k, v in 
                        {v: k for k, v in self.token2idx.items()}.items()}
        self.vocab_size = data["vocab_size"]
        self.max_length = data["max_length"]
        logger.info(f"Tokenizer loaded from {path} with {self.vocab_size} tokens")


class InvoiceDataset(Dataset):
    """Dataset for invoice images and their corresponding labels."""
    
    def __init__(self, image_paths, label_paths, tokenizer, transform=None, max_seq_length=512):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to invoice images
            label_paths: List of paths to corresponding JSON labels
            tokenizer: Tokenizer for encoding/decoding JSON data
            transform: Image transformations to apply
            max_seq_length: Maximum sequence length for tokenized output
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_length = max_seq_length
        
        assert len(self.image_paths) == len(self.label_paths), \
            "Number of images and labels must be equal"
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Load and process label
        try:
            with open(label_path, 'r') as f:
                label_data = json.load(f)
                
            # Extract data field which contains the key-value pairs
            if "data" in label_data:
                label_data = label_data["data"]

            # Tokenize the label
            encoded_label = self.tokenizer.encode_json(label_data)
            encoded_label = torch.tensor(encoded_label, dtype=torch.long)
            
            return {
                "image": image,
                "label": encoded_label,
                "image_path": image_path,
                "label_path": label_path
            }
            
        except Exception as e:
            logger.error(f"Error processing {label_path}: {e}")
            # Return a default empty label in case of error
            empty_label = torch.zeros(self.max_seq_length, dtype=torch.long)
            return {
                "image": image,
                "label": empty_label,
                "image_path": image_path,
                "label_path": label_path
            }


def get_transforms(mode="train", max_image_size=(800, 800)):
    """
    Get image transforms for training or validation/testing
    
    Args:
        mode: Either 'train' or 'val' or 'test'
        max_image_size: Maximum image dimensions (height, width)
    """
    if mode == "train":
        return A.Compose([
            A.Resize(height=max_image_size[0], width=max_image_size[1]),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=max_image_size[0], width=max_image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def prepare_dataset(split=True):
    """
    Prepare dataset for training, validation and testing
    
    Args:
        split: Whether to split the data into train/val/test
        
    Returns:
        Tuple of datasets or a single dataset
    """
    # Get all image and label file paths
    image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, '*.jpg')))
    label_files = sorted(glob.glob(os.path.join(LABEL_DIR, '*.json')))
    
    logger.info(f"Found {len(image_files)} images and {len(label_files)} labels")
    
    # Match image and label files by name
    matched_pairs = []
    for img_path in image_files:
        img_name = os.path.basename(img_path).split('.')[0]
        json_path = os.path.join(LABEL_DIR, f"{img_name}.json")
        
        if os.path.exists(json_path):
            matched_pairs.append((img_path, json_path))
    
    logger.info(f"Matched {len(matched_pairs)} image-label pairs")
    
    if len(matched_pairs) == 0:
        raise ValueError("No matched image-label pairs found!")
        
    # Split images and labels
    image_paths = [p[0] for p in matched_pairs]
    label_paths = [p[1] for p in matched_pairs]
    
    # Create tokenizer
    tokenizer = InvoiceTokenizer(max_length=MAX_SEQ_LENGTH)
    tokenizer.build_vocab(label_paths)
    
    # Save tokenizer for inference
    tokenizer.save(os.path.join(MODEL_SAVE_DIR, "tokenizer.json"))
    
    if split:
        # Split into train, validation and test sets
        train_val_image_paths, test_image_paths, train_val_label_paths, test_label_paths = train_test_split(
            image_paths, label_paths, test_size=TRAIN_VAL_TEST_SPLIT[2], random_state=RANDOM_SEED
        )
        
        train_image_paths, val_image_paths, train_label_paths, val_label_paths = train_test_split(
            train_val_image_paths, train_val_label_paths, 
            test_size=TRAIN_VAL_TEST_SPLIT[1]/(TRAIN_VAL_TEST_SPLIT[0]+TRAIN_VAL_TEST_SPLIT[1]),
            random_state=RANDOM_SEED
        )
        
        logger.info(f"Split data into {len(train_image_paths)} train, {len(val_image_paths)} val, {len(test_image_paths)} test samples")
        
        # Create datasets
        train_dataset = InvoiceDataset(
            train_image_paths, train_label_paths, tokenizer,
            transform=get_transforms("train", MAX_IMAGE_SIZE),
            max_seq_length=MAX_SEQ_LENGTH
        )
        
        val_dataset = InvoiceDataset(
            val_image_paths, val_label_paths, tokenizer,
            transform=get_transforms("val", MAX_IMAGE_SIZE),
            max_seq_length=MAX_SEQ_LENGTH
        )
        
        test_dataset = InvoiceDataset(
            test_image_paths, test_label_paths, tokenizer,
            transform=get_transforms("test", MAX_IMAGE_SIZE),
            max_seq_length=MAX_SEQ_LENGTH
        )
        
        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
            "tokenizer": tokenizer
        }
    else:
        # Create a single dataset without splitting
        dataset = InvoiceDataset(
            image_paths, label_paths, tokenizer,
            transform=get_transforms("test", MAX_IMAGE_SIZE),
            max_seq_length=MAX_SEQ_LENGTH
        )
        return {
            "full": dataset,
            "tokenizer": tokenizer
        }


def get_dataloaders(datasets, batch_size=8, num_workers=4):
    """
    Create DataLoader objects for the datasets
    
    Args:
        datasets: Dictionary of datasets
        batch_size: Batch size for training and evaluation
        num_workers: Number of worker processes for data loading
        
    Returns:
        Dictionary of DataLoader objects
    """
    dataloaders = {}
    
    for split, dataset in datasets.items():
        if split == "tokenizer":
            dataloaders[split] = dataset
            continue
            
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=(split == "train"),
            pin_memory=True
        )
        dataloaders[split] = dataloader
    
    return dataloaders


def visualize_sample(dataset, idx=None):
    """
    Visualize a sample from the dataset
    
    Args:
        dataset: The dataset to visualize from
        idx: Index of sample to visualize, if None, a random sample is chosen
    """
    import matplotlib.pyplot as plt
    
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)
        
    sample = dataset[idx]
    
    image = sample['image']
    label = sample['label']
    
    # Convert from tensor to numpy for visualization
    if isinstance(image, torch.Tensor):
        # Denormalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image.permute(1, 2, 0).numpy()
        image = std * image + mean
        image = np.clip(image, 0, 1)
    
    # Decode the label
    tokenizer = dataset.tokenizer
    decoded_label = tokenizer.decode_ids(label.tolist())
    
    # Display the image and label
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(f"Sample {idx}")
    plt.axis('off')
    plt.show()
    
    print("Decoded Label:")
    print(decoded_label)


if __name__ == "__main__":
    # Test the dataset preparation
    datasets = prepare_dataset(split=True)
    
    # Display dataset statistics
    for split, dataset in datasets.items():
        if split != "tokenizer":
            print(f"{split} set: {len(dataset)} samples")
    
    # Visualize a random sample
    if "train" in datasets:
        print("\nVisualizing a random training sample:")
        visualize_sample(datasets["train"])
        
    # Create dataloaders
    dataloaders = get_dataloaders(datasets, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    print("\nDataloaders created with batch size:", BATCH_SIZE)
    for split, dataloader in dataloaders.items():
        if split != "tokenizer":
            print(f"{split} dataloader: {len(dataloader)} batches") 