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
    def __init__(self, model_path, device=None, confidence_threshold=None, image_size=None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the saved model weights
            device: Device to run inference on. If None, will use CUDA if available
            confidence_threshold: Confidence threshold for predictions
            image_size: Image size to use for inference. If None, will use config value
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
        
        # Set image size
        self.image_size = image_size if image_size is not None else IMAGE_SIZE
            
        # Load model
        self.model = self._load_model(model_path)
        
        # Set up transforms for preprocessing
        self.transform = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
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
            image_path: Path to the image file or a numpy array
            
        Returns:
            Preprocessed image tensor and original dimensions
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
    
    def postprocess_batch_outputs(self, outputs, orig_sizes):
        """
        Post-process batch model outputs.
        
        Args:
            outputs: Model output dictionary
            orig_sizes: List of original image dimensions (height, width)
            
        Returns:
            List of lists of detected entities for each image
        """
        batch_results = []
        
        # Process each image in the batch
        for i in range(outputs['pred_logits'].shape[0]):
            # Extract predictions for this image
            pred_logits = outputs['pred_logits'][i]  # [num_queries, num_classes+1]
            pred_boxes = outputs['pred_boxes'][i]    # [num_queries, 4]
            orig_h, orig_w = orig_sizes[i]
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(pred_logits, dim=-1)
            
            # Get scores and classes
            scores = probs[:, :-1].max(dim=1)[0]
            classes = probs[:, :-1].argmax(dim=1)
            
            # Filter by confidence threshold
            keep = scores > self.confidence_threshold
            scores = scores[keep]
            classes = classes[keep]
            pred_boxes = pred_boxes[keep]
            
            # Convert normalized boxes to pixel coordinates
            boxes = []
            for box in pred_boxes:
                x_min = box[0].item() * orig_w
                y_min = box[1].item() * orig_h
                x_max = box[2].item() * orig_w
                y_max = box[3].item() * orig_h
                boxes.append([x_min, y_min, x_max, y_max])
                
            # Convert class ids to names
            class_names = [self.id_to_class[class_id.item()] for class_id in classes]
            
            # Create results list for this image
            img_results = []
            for j in range(len(class_names)):
                img_results.append({
                    'entity_type': class_names[j],
                    'confidence': scores[j].item(),
                    'bounding_box': {
                        'x_min': boxes[j][0],
                        'y_min': boxes[j][1],
                        'x_max': boxes[j][2],
                        'y_max': boxes[j][3],
                    }
                })
                
            batch_results.append(img_results)
            
        return batch_results
    
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
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        img_tensor = img_tensor.to(self.device)
        
        # Run inference
        outputs = self.model(img_tensor)
        
        # Post-process outputs
        results = self.postprocess_outputs(outputs, orig_size)
        
        return results
    
    @torch.no_grad()
    def predict_batch(self, images):
        """
        Run inference on a batch of images.
        
        Args:
            images: List of preprocessed image tensors and their original sizes
            
        Returns:
            List of lists of detected entities for each image
        """
        # Prepare batch
        batch_tensors = []
        orig_sizes = []
        
        for image in images:
            if isinstance(image, tuple) and len(image) == 2:
                # Image is already preprocessed (tensor, orig_size)
                img_tensor, orig_size = image
                batch_tensors.append(img_tensor)
                orig_sizes.append(orig_size)
            else:
                # Image needs preprocessing
                img_tensor, orig_size = self.preprocess_image(image)
                batch_tensors.append(img_tensor)
                orig_sizes.append(orig_size)
        
        # Stack tensors into a batch
        if batch_tensors:
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Run inference
            outputs = self.model(batch_tensor)
            
            # Post-process outputs
            results = self.postprocess_batch_outputs(outputs, orig_sizes)
            
            return results
        
        return []
    
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
        "document": {
            "entities": {},
            "confidence": 0.0
        }
    }
    
    # Group results by entity type
    entities_by_type = {}
    total_confidence = 0.0
    
    for res in results:
        entity_type = res['entity_type']
        confidence = res['confidence']
        bbox = res['bounding_box']
        
        # Format bounding box
        polygon = [
            {"x": bbox['x_min'], "y": bbox['y_min']},
            {"x": bbox['x_max'], "y": bbox['y_min']},
            {"x": bbox['x_max'], "y": bbox['y_max']},
            {"x": bbox['x_min'], "y": bbox['y_max']}
        ]
        
        # Create entity data
        entity_data = {
            "value": text_recognition[entity_type] if text_recognition and entity_type in text_recognition else "",
            "confidence": confidence,
            "bounding_regions": [
                {
                    "polygon": polygon,
                    "confidence": confidence
                }
            ]
        }
        
        # Add to entities by type
        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = entity_data
        else:
            # For duplicate types, keep the one with higher confidence
            if confidence > entities_by_type[entity_type]['confidence']:
                entities_by_type[entity_type] = entity_data
        
        # Track total confidence
        total_confidence += confidence
    
    # Add entities to output
    output['document']['entities'] = entities_by_type
    
    # Calculate average confidence
    if results:
        output['document']['confidence'] = total_confidence / len(results)
    
    return output 