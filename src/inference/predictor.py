import os
import torch
import numpy as np
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG, IMAGE_SIZE, INFERENCE_CONFIG
from model.document_parser import build_model


class DocumentPredictor:
    """
    Class to handle document parsing prediction with a trained model.
    """
    def __init__(self, model_path, device=None, confidence_threshold=None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the saved model weights
            device: Device to run inference on. If None, will use CUDA if available
            confidence_threshold: Confidence threshold for predictions
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Set confidence threshold
        if confidence_threshold is None:
            self.confidence_threshold = INFERENCE_CONFIG['confidence_threshold']
        else:
            self.confidence_threshold = confidence_threshold
            
        # Load model
        self.model = self._load_model(model_path)
        
        # Set up transforms for preprocessing
        self.transform = A.Compose([
            A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # Create class id to name mapping
        self.id_to_class = {v: k for k, v in MODEL_CONFIG["entity_classes"].items()}
    
    def _load_model(self, model_path):
        """Load model from checkpoint."""
        # Build model
        model = build_model()
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load weights
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
            
        # Move to device and set to eval mode
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for inference.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """
        # Read image
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Assume it's already a numpy array
            image = image_path
            
        # Get original dimensions
        height, width = image.shape[:2]
        
        # Apply transforms
        transformed = self.transform(image=image)
        img_tensor = transformed['image']
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor, (height, width)
    
    def postprocess_outputs(self, outputs, orig_size):
        """
        Post-process model outputs to get readable predictions.
        
        Args:
            outputs: Model output dictionary
            orig_size: Original image dimensions (height, width)
            
        Returns:
            List of detected entities with class names, bounding boxes, and scores
        """
        # Get predictions
        pred_logits = outputs['pred_logits'][0]  # Shape: [num_queries, num_classes+1]
        pred_boxes = outputs['pred_boxes'][0]    # Shape: [num_queries, 4]
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(pred_logits, dim=-1)
        
        # Get scores (probability of being non-background)
        scores = probs[:, :-1].max(dim=1)[0]
        
        # Get classes (excluding background)
        classes = probs[:, :-1].argmax(dim=1)
        
        # Filter by confidence threshold
        keep = scores > self.confidence_threshold
        scores = scores[keep]
        classes = classes[keep]
        pred_boxes = pred_boxes[keep]
        
        # Convert normalized boxes to pixel coordinates
        orig_h, orig_w = orig_size
        boxes = []
        for box in pred_boxes:
            x_min = box[0].item() * orig_w
            y_min = box[1].item() * orig_h
            x_max = box[2].item() * orig_w
            y_max = box[3].item() * orig_h
            boxes.append([x_min, y_min, x_max, y_max])
            
        # Convert class ids to names
        class_names = [self.id_to_class[class_id.item()] for class_id in classes]
        
        # Create results list
        results = []
        for i in range(len(class_names)):
            results.append({
                'entity_type': class_names[i],
                'confidence': scores[i].item(),
                'bounding_box': {
                    'x_min': boxes[i][0],
                    'y_min': boxes[i][1],
                    'x_max': boxes[i][2],
                    'y_max': boxes[i][3],
                }
            })
            
        return results
    
    @torch.no_grad()
    def predict(self, image_path):
        """
        Run inference on an image.
        
        Args:
            image_path: Path to the image file or numpy array
            
        Returns:
            List of detected entities with class names, bounding boxes, and scores
        """
        # Preprocess image
        img_tensor, orig_size = self.preprocess_image(image_path)
        
        # Move to device
        img_tensor = img_tensor.to(self.device)
        
        # Run inference
        outputs = self.model(img_tensor)
        
        # Post-process outputs
        results = self.postprocess_outputs(outputs, orig_size)
        
        return results
    
    def visualize_prediction(self, image_path, results=None):
        """
        Visualize prediction results on the image.
        
        Args:
            image_path: Path to the image file
            results: Prediction results. If None, will run inference
            
        Returns:
            Image with visualized predictions
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Assume it's already a numpy array
            image = image_path.copy()
            
        # Run prediction if results not provided
        if results is None:
            results = self.predict(image_path)
            
        # Draw bounding boxes and labels
        for res in results:
            # Get box coordinates
            x_min = int(res['bounding_box']['x_min'])
            y_min = int(res['bounding_box']['y_min'])
            x_max = int(res['bounding_box']['x_max'])
            y_max = int(res['bounding_box']['y_max'])
            
            # Generate random color for this class
            np.random.seed(hash(res['entity_type']) % 10000)
            color = tuple(map(int, np.random.randint(0, 255, 3).tolist()))
            
            # Draw rectangle
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Prepare label text
            label = f"{res['entity_type']}: {res['confidence']:.2f}"
            
            # Draw label background
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x_min, y_min - text_size[1] - 5), (x_min + text_size[0], y_min), color, -1)
            
            # Draw label text
            cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
        return image


def format_results_as_json(results, text_recognition=None):
    """
    Format prediction results as a structured JSON output.
    
    Args:
        results: Prediction results from DocumentPredictor
        text_recognition: Optional text recognition results
        
    Returns:
        Structured JSON output
    """
    # Initialize output structure
    output = {
        "data": {},
        "meta": {
            "model_version": "1.0.0"
        }
    }
    
    # Process each detected entity
    for res in results:
        entity_type = res['entity_type']
        confidence = res['confidence']
        bbox = res['bounding_box']
        
        # Create polygon points from bounding box
        polygon = [
            {"x": bbox['x_min'], "y": bbox['y_min']},
            {"x": bbox['x_max'], "y": bbox['y_min']},
            {"x": bbox['x_max'], "y": bbox['y_max']},
            {"x": bbox['x_min'], "y": bbox['y_max']}
        ]
        
        # Get text if available
        text = ""
        if text_recognition is not None and entity_type in text_recognition:
            text = text_recognition[entity_type]
        
        # Add to output
        output["data"][entity_type] = {
            "value": text,
            "confidence": confidence,
            "bounding_regions": [
                {
                    "polygon": polygon
                }
            ],
            "content": text
        }
    
    return output 