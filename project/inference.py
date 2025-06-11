import os
import torch
import json
import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import re

from config import *
from model_arch import InvoiceTransformer
from data_preprocess import InvoiceTokenizer


def load_tokenizer(tokenizer_path):
    """Load tokenizer from saved file"""
    tokenizer = InvoiceTokenizer()
    tokenizer.load(tokenizer_path)
    return tokenizer


def load_model(model_path, vocab_size, device):
    """Load a trained model from checkpoint"""
    model = InvoiceTransformer(vocab_size=vocab_size)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, max_size=(800, 800)):
    """Preprocess image for model input"""
    # Load image
    if isinstance(image_path, str):
        # Load from file
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # Assume it's already a numpy array
        image = image_path
        
    # Apply transformations
    transform = A.Compose([
        A.Resize(height=max_size[0], width=max_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    transformed = transform(image=image)
    tensor_image = transformed["image"]
    
    # Add batch dimension
    tensor_image = tensor_image.unsqueeze(0)
    
    return tensor_image


def text_to_json(text):
    """Convert the model output text to structured JSON"""
    # Clean up the text
    text = text.strip()
    
    # Find key-value pairs
    pairs = re.findall(r'([^;=]+)=([^;]+)(?:;|$)', text)
    
    # Create JSON structure
    result = {}
    for key, value in pairs:
        key = key.strip()
        value = value.strip()
        
        # Handle nested keys (with dot notation)
        if '.' in key:
            parts = key.split('.')
            current = result
            for part in parts[:-1]:
                part = part.strip()
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1].strip()] = value
        else:
            result[key] = value
    
    return result


def run_inference(model, tokenizer, image_path, device, temperature=0.7):
    """Run inference on a single image"""
    # Preprocess image
    processed_image = preprocess_image(image_path)
    processed_image = processed_image.to(device)
    
    # Generate prediction
    with torch.no_grad():
        generated_tokens = model.generate(processed_image, max_length=MAX_SEQ_LENGTH, temperature=temperature)
    
    # Decode prediction
    predicted_text = tokenizer.decode_ids(generated_tokens)
    
    # Convert to JSON
    predicted_json = text_to_json(predicted_text)
    
    return {
        "text": predicted_text,
        "json": predicted_json
    }


def visualize_inference(image_path, result, save_path=None):
    """Visualize inference result"""
    # Display image
    plt.figure(figsize=(16, 10))
    plt.subplot(1, 2, 1)
    img = Image.open(image_path) if isinstance(image_path, str) else Image.fromarray(image_path)
    plt.imshow(img)
    plt.title("Input Invoice")
    plt.axis('off')
    
    # Display JSON result
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.text(0.05, 0.95, json.dumps(result["json"], indent=2), fontsize=9, 
             va='top', ha='left', wrap=True, family='monospace')
    plt.title("Extracted Data")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    plt.show()


def process_batch(model, tokenizer, image_paths, device, temperature=0.7, output_dir=None, visualize=False):
    """Process a batch of images"""
    results = []
    
    for i, image_path in enumerate(tqdm(image_paths, desc="Processing")):
        # Run inference
        result = run_inference(model, tokenizer, image_path, device, temperature)
        
        # Create result object
        output = {
            "image_path": image_path,
            "result": result
        }
        results.append(output)
        
        # Save JSON result
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            json_path = os.path.join(output_dir, f"{base_name}_result.json")
            
            with open(json_path, "w") as f:
                json.dump(result["json"], f, indent=2)
                
            # Save visualization
            if visualize:
                vis_path = os.path.join(output_dir, f"{base_name}_visualization.png")
                visualize_inference(image_path, result, vis_path)
    
    return results


def main(args):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = load_tokenizer(args.tokenizer_path)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, tokenizer.vocab_size, device)
    
    # Get input images
    if os.path.isdir(args.input_path):
        # Process all images in directory
        image_paths = [
            os.path.join(args.input_path, f) for f in os.listdir(args.input_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
        ]
        print(f"Found {len(image_paths)} images in {args.input_path}")
    else:
        # Process single image
        image_paths = [args.input_path]
    
    # Process images
    results = process_batch(
        model, tokenizer, image_paths, device, 
        temperature=args.temperature, 
        output_dir=args.output_dir,
        visualize=args.visualize
    )
    
    print(f"Processed {len(results)} images. Results saved to {args.output_dir}")
    
    # Return the first result for single image case
    return results[0] if len(results) == 1 else results


class InvoiceProcessor:
    """Wrapper class for invoice processing functionality"""
    
    def __init__(self, model_path, tokenizer_path, device=None):
        """
        Initialize the invoice processor.
        
        Args:
            model_path: Path to the trained model checkpoint
            tokenizer_path: Path to the tokenizer json file
            device: Device to run inference on (cuda or cpu)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Load tokenizer
        self.tokenizer = load_tokenizer(tokenizer_path)
        
        # Load model
        self.model = load_model(model_path, self.tokenizer.vocab_size, self.device)
    
    def process_image(self, image, temperature=0.7, visualize=False):
        """
        Process a single image and extract information.
        
        Args:
            image: Path to image file or numpy array
            temperature: Sampling temperature for generation
            visualize: Whether to visualize the result
            
        Returns:
            Dictionary with extracted information
        """
        result = run_inference(self.model, self.tokenizer, image, self.device, temperature)
        
        if visualize:
            visualize_inference(image, result)
            
        return result
    
    def process_batch(self, image_paths, temperature=0.7, output_dir=None, visualize=False):
        """
        Process a batch of images.
        
        Args:
            image_paths: List of paths to image files
            temperature: Sampling temperature for generation
            output_dir: Directory to save results
            visualize: Whether to visualize the results
            
        Returns:
            List of dictionaries with extracted information
        """
        return process_batch(
            self.model, self.tokenizer, image_paths, self.device,
            temperature=temperature, output_dir=output_dir, visualize=visualize
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with invoice processing model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer file")
    parser.add_argument("--input_path", type=str, required=True, 
                        help="Path to input image file or directory")
    parser.add_argument("--output_dir", type=str, default="./outputs/inference", 
                        help="Directory to save results")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Sampling temperature (higher = more diverse)")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    
    args = parser.parse_args()
    
    main(args) 