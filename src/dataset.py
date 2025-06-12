import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from PIL import Image
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

class InvoiceDataset(Dataset):
    """
    Dataset for training the invoice processor model
    Takes processed data from the InvoiceDataProcessor
    """
    def __init__(self, 
                 annotations_file: str, 
                 images_dir: str,
                 tokenizer_name: str = "bert-base-uncased",
                 max_seq_len: int = 512,
                 image_size: Tuple[int, int] = (224, 224),
                 training: bool = True):
        """
        Initialize the dataset
        
        Args:
            annotations_file: Path to JSON file containing annotations
            images_dir: Directory containing processed images
            tokenizer_name: Name of the tokenizer to use
            max_seq_len: Maximum sequence length for tokens
            image_size: Size of input images (height, width)
            training: Whether this dataset is for training
        """
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        self.training = training
        
        # Load annotations
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        print(f"Loaded {len(self.annotations)} samples from {annotations_file}")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        annotation = self.annotations[idx]
        
        # Load image
        img_path = self.images_dir / annotation['image_filename']
        image = Image.open(img_path).convert('RGB')
        
        # Convert to tensor
        image = self._preprocess_image(image)
        
        # Get text and boxes
        boxes = np.array(annotation['boxes'], dtype=np.float32)
        text_values = annotation['values']
        
        # Tokenize text
        tokens_info = self._tokenize_text(text_values)
        
        # Adjust boxes for tokenization
        tokenized_boxes = self._align_boxes_with_tokens(boxes, tokens_info)
        
        # Get ground truth fields if available
        key_fields = annotation.get('key_fields', {})
        
        sample = {
            'image': image,
            'tokens': tokens_info['input_ids'],
            'attention_mask': tokens_info['attention_mask'],
            'boxes': tokenized_boxes,
            'text': text_values,
            'image_id': annotation['image_id'],
            'key_fields': key_fields
        }
        
        return sample
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess an image for the model"""
        # Resize
        image = image.resize(self.image_size, Image.BILINEAR)
        
        # Convert to array
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std
        
        # Convert to tensor [C, H, W]
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()  # Explicitly cast to float32
        
        return image_tensor
    
    def _tokenize_text(self, texts: List[str]) -> Dict:
        """Tokenize text values"""
        # Combine all texts with a separator
        combined_text = " [SEP] ".join(texts)
        
        # Tokenize
        tokenized = self.tokenizer(
            combined_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        for key in tokenized:
            if torch.is_tensor(tokenized[key]):
                tokenized[key] = tokenized[key].squeeze(0)
        
        # Create token to text mapping
        tokenized['token_to_text_mapping'] = self._create_token_to_text_mapping(
            combined_text,
            tokenized['input_ids'],
            texts
        )
        
        return tokenized
    
    def _create_token_to_text_mapping(self, combined_text: str, input_ids: torch.Tensor, 
                                      original_texts: List[str]) -> List[int]:
        """
        Create a mapping from token indices to original text indices
        
        Args:
            combined_text: Combined text string
            input_ids: Token IDs from tokenizer
            original_texts: List of original text strings
            
        Returns:
            List mapping each token to its original text index
        """
        # Decode tokens back to text to find alignments
        token_to_text = []
        
        # Start with special tokens (e.g., CLS) - map to -1
        token_to_text.append(-1)
        
        # Keep track of which original text we're in
        text_idx = 0
        text_offset = 0
        
        # Process each token (skipping first special token)
        for i in range(1, len(input_ids)):
            if i < len(input_ids) and input_ids[i] == self.tokenizer.sep_token_id:
                # SEP token - move to next text
                text_idx += 1
                text_offset = 0
                token_to_text.append(-1)  # Special token
            elif i < len(input_ids) and input_ids[i] in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]:
                # Padding or EOS token
                token_to_text.append(-1)
            else:
                # Regular token - map to current text
                if text_idx < len(original_texts):
                    token_to_text.append(text_idx)
                else:
                    # Beyond our original texts (e.g., padding)
                    token_to_text.append(-1)
        
        return token_to_text

    def _align_boxes_with_tokens(self, boxes: np.ndarray, tokens_info: Dict) -> np.ndarray:
        """
        Align bounding boxes with tokenized text
        
        Args:
            boxes: Original boxes corresponding to text lines [N, 4]
            tokens_info: Tokenization information
            
        Returns:
            Boxes aligned with tokens [max_seq_len, 4]
        """
        token_to_text = tokens_info['token_to_text_mapping']
        seq_length = len(token_to_text)
        
        # Create output array with zeros for special tokens
        aligned_boxes = np.zeros((seq_length, 4), dtype=np.float32)
        
        # For each token, assign the corresponding box
        for i, text_idx in enumerate(token_to_text):
            if text_idx >= 0 and text_idx < len(boxes):
                aligned_boxes[i] = boxes[text_idx]
        
        return aligned_boxes

    @staticmethod
    def create_dataloader(dataset, batch_size: int = 8, shuffle: bool = True, 
                          num_workers: int = 4) -> DataLoader:
        """
        Create a DataLoader for the dataset
        
        Args:
            dataset: The dataset to create a loader for
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            
        Returns:
            DataLoader instance
        """
        def collate_fn(batch):
            """Custom collate function to handle variable-length data"""
            # Extract batch elements
            images = torch.stack([item['image'] for item in batch]).float()  # Ensure float32
            tokens = torch.stack([item['tokens'] for item in batch])
            attention_masks = torch.stack([item['attention_mask'] for item in batch])
            boxes = torch.from_numpy(np.stack([item['boxes'] for item in batch])).float()  # Ensure float32
            image_ids = [item['image_id'] for item in batch]
            texts = [item['text'] for item in batch]
            key_fields = [item['key_fields'] for item in batch]
            
            # Create batch dictionary
            batch_dict = {
                'image': images,
                'tokens': tokens,
                'attention_mask': attention_masks,
                'boxes': boxes,
                'image_id': image_ids,
                'text': texts,
                'key_fields': key_fields
            }
            
            return batch_dict
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        ) 