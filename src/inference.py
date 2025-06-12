import os
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import warnings
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional, Any

# Import doctr first to configure TensorFlow environment
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Import doctr components at the top level
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from model import create_invoice_model

class InvoiceProcessor:
    """
    Class for performing inference with the trained invoice model
    """
    def __init__(self, model_path: str, 
                 tokenizer_name: str = "bert-base-uncased",
                 device: str = None,
                 image_size: Tuple[int, int] = (224, 224),
                 max_seq_len: int = 512):
        """
        Initialize the invoice processor
        
        Args:
            model_path: Path to the trained model checkpoint
            tokenizer_name: Name of the tokenizer to use
            device: Device to run inference on (default: auto-detect)
            image_size: Size to resize input images to
            max_seq_len: Maximum sequence length for text tokens
        """
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set parameters
        self.image_size = image_size
        self.max_seq_len = max_seq_len
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Load model
        self.model = self._load_model(model_path)
        
        print(f"Initialized InvoiceProcessor with model from {model_path} on {self.device}")
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load the trained model"""
        # Create model
        model = create_invoice_model(
            pretrained=False,  # No need for pretrained weights when loading checkpoint
            vocab_size=30000,
            max_seq_len=self.max_seq_len,
            embed_dim=256
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device and set to eval mode
        model.to(self.device)
        model.eval()
        
        return model
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess an image for the model"""
        # Resize
        image = image.resize(self.image_size)
        
        # Convert to array
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std
        
        # Convert to tensor [C, H, W]
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def _perform_ocr(self, image: np.ndarray) -> Tuple[List[str], List[List[float]]]:
        """
        Perform OCR on an image to extract text and bounding boxes
        
        Uses doctr OCR to extract text and bounding boxes from the image
        """
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Initialize doctr model (uses pretrained models)
        # Use a lightweight model for speed, can change to 'db_resnet50' for higher accuracy
        model = ocr_predictor(pretrained=True)
        
        # Prepare the image and run inference
        # doctr expects RGB images
        if len(image.shape) == 2:  # Grayscale
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = image
            
        # Create a temporary file to use with DocumentFile
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_file:
            # Save the image to the temporary file
            cv2.imwrite(temp_file.name, image_rgb)
            
            # Create a DocumentFile from the temporary file
            doc = DocumentFile.from_images([temp_file.name])
            
            # Run inference
            result = model(doc)
        
        # Extract text and boxes
        texts = []
        boxes = []
        
        # Process predictions
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        # Get text
                        text = word.value
                        
                        # Get bounding box coordinates (doctr returns boxes as relative coordinates)
                        # Format is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] 
                        rel_coords = word.geometry
                        
                        # Convert to [x, y, width, height] format
                        x = rel_coords[0][0]
                        y = rel_coords[0][1]
                        width = rel_coords[1][0] - rel_coords[0][0]
                        height = rel_coords[2][1] - rel_coords[1][1]
                        
                        # Add to results
                        texts.append(text)
                        boxes.append([x, y, width, height])
        
        # If no text was found, add a default empty box
        if not texts:
            texts = ["No text detected"]
            boxes = [[0.1, 0.1, 0.8, 0.1]]  # Default box in the middle
            
        return texts, boxes
    
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
        
        # Create token to text mapping
        token_to_text = []
        
        # Start with special tokens (e.g., CLS) - map to -1
        token_to_text.append(-1)
        
        # Keep track of which original text we're in
        text_idx = 0
        
        # Process each token (skipping first special token)
        for i in range(1, len(tokenized['input_ids'][0])):
            if tokenized['input_ids'][0][i].item() == self.tokenizer.sep_token_id:
                # SEP token - move to next text
                text_idx += 1
                token_to_text.append(-1)  # Special token
            elif tokenized['input_ids'][0][i].item() in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]:
                # Padding or EOS token
                token_to_text.append(-1)
            else:
                # Regular token - map to current text
                if text_idx < len(texts):
                    token_to_text.append(text_idx)
                else:
                    # Beyond our original texts (e.g., padding)
                    token_to_text.append(-1)
        
        tokenized['token_to_text_mapping'] = token_to_text
        
        return tokenized
    
    def _align_boxes_with_tokens(self, boxes: List[List[float]], tokens_info: Dict) -> np.ndarray:
        """Align bounding boxes with tokenized text"""
        token_to_text = tokens_info['token_to_text_mapping']
        seq_length = len(token_to_text)
        
        # Create output array with zeros for special tokens
        boxes = np.array(boxes, dtype=np.float32)
        aligned_boxes = np.zeros((seq_length, 4), dtype=np.float32)
        
        # For each token, assign the corresponding box
        for i, text_idx in enumerate(token_to_text):
            if text_idx >= 0 and text_idx < len(boxes):
                aligned_boxes[i] = boxes[text_idx]
        
        return aligned_boxes
    
    def process_image(self, image_path: str) -> Dict:
        """
        Process an invoice image and extract fields
        
        Args:
            image_path: Path to the invoice image
            
        Returns:
            Dictionary of extracted fields
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self._preprocess_image(image)
        
        # Convert to numpy for OCR
        image_np = np.array(image)
        
        # Perform OCR
        texts, boxes = self._perform_ocr(image_np)
        
        # Tokenize text
        tokens_info = self._tokenize_text(texts)
        
        # Align boxes with tokens
        aligned_boxes = self._align_boxes_with_tokens(boxes, tokens_info)
        
        # Prepare inputs for model
        image_tensor = image_tensor.to(self.device).float()
        tokens = tokens_info['input_ids'].to(self.device)
        attention_mask = tokens_info['attention_mask'].to(self.device)
        boxes_tensor = torch.from_numpy(aligned_boxes).unsqueeze(0).to(self.device).float()
        
        # Compute attention mask for tokens
        token_mask = attention_mask.eq(0)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(image_tensor, tokens, boxes_tensor, token_mask)
        
        # Extract readable field values
        extracted_fields = self.model.extract_fields_from_predictions(outputs, [texts])
        
        # Return the first (and only) item in the batch
        return extracted_fields[0]
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Process a PDF invoice and extract fields from each page
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries with extracted fields for each page
        """
        # In a real implementation, use pdf2image to convert PDF to images
        # For this demo, we'll just pretend the PDF is a single image
        print(f"Processing PDF: {pdf_path}")
        print("Note: In this demo, PDFs are processed as single page images")
        
        # Just call process_image for the PDF
        return [self.process_image(pdf_path)]
    
    def process_document(self, document_path: str) -> Dict:
        """
        Process a document (image or PDF) and extract fields
        
        Args:
            document_path: Path to the document
            
        Returns:
            Dictionary of extracted fields
        """
        path = Path(document_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")
        
        if path.suffix.lower() in ['.pdf']:
            # Process PDF
            results = self.process_pdf(document_path)
            # For simplicity, return results from the first page
            return results[0]
        elif path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
            # Process image
            return self.process_image(document_path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    
def format_results(results: Dict) -> Dict:
    """Format results for display"""
    formatted = {}
    
    for field_name, field_info in results.items():
        value = field_info['text']
        confidence = field_info['confidence']
        
        # Format the field value
        if field_name == 'total_amount':
            # Format currency values
            if value:
                try:
                    value = f"${float(value.replace('$', '').replace(',', '')):.2f}"
                except ValueError:
                    pass
        
        formatted[field_name] = {
            'value': value,
            'confidence': f"{confidence:.2f}"
        }
    
    return formatted

def main():
    parser = argparse.ArgumentParser(description="Process invoices using the trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--document_path", type=str, required=True, help="Path to the invoice document (image or PDF)")
    parser.add_argument("--output_path", type=str, help="Path to save the output JSON (optional)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Device to run inference on")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length for tokens")
    parser.add_argument("--image_size", type=int, default=224, help="Size to resize images to (square)")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = InvoiceProcessor(
        model_path=args.model_path,
        device=args.device,
        max_seq_len=args.max_seq_len,
        image_size=(args.image_size, args.image_size)
    )
    
    # Process document
    results = processor.process_document(args.document_path)
    
    # Format results
    formatted_results = format_results(results)
    
    # Print results
    print("\nExtracted Fields:")
    for field_name, field_info in formatted_results.items():
        print(f"{field_name}: {field_info['value']} (confidence: {field_info['confidence']})")
    
    # Save results if output path specified
    if args.output_path:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_results, f, indent=2)
        print(f"\nResults saved to {args.output_path}")

if __name__ == "__main__":
    main() 