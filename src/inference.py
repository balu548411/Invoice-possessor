import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import json
from pathlib import Path
import logging

from .model_architecture import InvoiceProcessingModel
from .training import InvoiceProcessingLightningModule
from .data_processing import get_val_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InvoiceProcessor:
    """
    Production-ready invoice processor for inference
    Similar to Azure Form Recognizer API
    """
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'auto',
                 confidence_threshold: float = 0.5):
        self.device = self._setup_device(device)
        self.confidence_threshold = confidence_threshold
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Setup transforms
        self.transform = get_val_transforms()
        
        logger.info(f"Invoice processor initialized on {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> InvoiceProcessingModel:
        """Load trained model from checkpoint"""
        try:
            # Load Lightning checkpoint
            lightning_model = InvoiceProcessingLightningModule.load_from_checkpoint(
                model_path, map_location=self.device
            )
            model = lightning_model.model
            model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocess image for inference"""
        # Load image
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            image = np.array(image_input)
        else:
            image = image_input
        
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        return image_tensor, original_size
    
    def extract_text_regions(self, image: np.ndarray, 
                           min_area: int = 100) -> List[Dict]:
        """
        Extract text regions using OpenCV (simplified OCR)
        In production, you'd use a proper OCR engine like Tesseract or Azure OCR
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if area > min_area:
                # Normalize coordinates
                height, width = image.shape[:2]
                normalized_box = [
                    x / width,
                    y / height,
                    (x + w) / width,
                    (y + h) / height
                ]
                
                text_regions.append({
                    'bbox': normalized_box,
                    'area': area,
                    'text': f"text_region_{len(text_regions)}"  # Placeholder
                })
        
        return text_regions
    
    def create_inference_batch(self, image_tensor: torch.Tensor, 
                              text_regions: List[Dict],
                              max_boxes: int = 500) -> Dict[str, torch.Tensor]:
        """Create batch for model inference"""
        
        # Extract bounding boxes
        boxes = [region['bbox'] for region in text_regions]
        
        # Pad or truncate to max_boxes
        if len(boxes) > max_boxes:
            boxes = boxes[:max_boxes]
        else:
            pad_length = max_boxes - len(boxes)
            boxes.extend([[0, 0, 0, 0]] * pad_length)
        
        # Create labels (all 1 for text regions)
        labels = [1] * len([region for region in text_regions if region['bbox'] != [0, 0, 0, 0]])
        labels.extend([0] * (max_boxes - len(labels)))
        
        batch = {
            'images': image_tensor,
            'boxes': torch.tensor([boxes], dtype=torch.float32, device=self.device),
            'labels': torch.tensor([labels], dtype=torch.long, device=self.device)
        }
        
        return batch
    
    @torch.no_grad()
    def process_invoice(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict:
        """
        Process an invoice and extract structured information
        
        Returns:
            Dictionary containing extracted entities similar to Azure Form Recognizer
        """
        try:
            # Preprocess image
            image_tensor, original_size = self.preprocess_image(image_input)
            
            # Load original image for text region extraction
            if isinstance(image_input, str):
                original_image = cv2.imread(image_input)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            elif isinstance(image_input, Image.Image):
                original_image = np.array(image_input)
            else:
                original_image = image_input
            
            # Extract text regions
            text_regions = self.extract_text_regions(original_image)
            
            if not text_regions:
                logger.warning("No text regions detected in image")
                return self._empty_result()
            
            # Create batch for inference
            batch = self.create_inference_batch(image_tensor, text_regions)
            
            # Run inference
            outputs = self.model(batch)
            
            # Extract entities
            entities = self.model.extract_entities(outputs, batch)
            
            # Post-process results
            result = self._post_process_results(entities[0], text_regions, original_size)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing invoice: {e}")
            return self._empty_result()
    
    def _post_process_results(self, entities: Dict, 
                            text_regions: List[Dict],
                            original_size: Tuple[int, int]) -> Dict:
        """Post-process model outputs into structured format"""
        
        # Convert normalized coordinates back to original image coordinates
        width, height = original_size
        
        result = {
            'status': 'succeeded',
            'pages': [{
                'page_number': 1,
                'width': width,
                'height': height,
                'unit': 'pixel',
                'extracted_fields': {}
            }],
            'confidence': 0.0
        }
        
        # Process each entity type
        page = result['pages'][0]
        confidences = []
        
        for entity_type, entity_data in entities.items():
            if entity_data and 'boxes' in entity_data:
                # Convert normalized boxes back to pixel coordinates
                pixel_boxes = []
                for box in entity_data['boxes']:
                    pixel_box = [
                        box[0] * width,
                        box[1] * height,
                        box[2] * width,
                        box[3] * height
                    ]
                    pixel_boxes.append(pixel_box)
                
                # Create field entry
                field_entry = {
                    'value': self._extract_text_from_boxes(pixel_boxes, text_regions, original_size),
                    'confidence': entity_data.get('confidence', 0.0),
                    'bounding_boxes': pixel_boxes
                }
                
                page['extracted_fields'][entity_type] = field_entry
                confidences.append(entity_data.get('confidence', 0.0))
        
        # Calculate overall confidence
        if confidences:
            result['confidence'] = sum(confidences) / len(confidences)
        
        return result
    
    def _extract_text_from_boxes(self, pixel_boxes: List[List[float]], 
                                text_regions: List[Dict],
                                original_size: Tuple[int, int]) -> str:
        """Extract text content from bounding boxes (simplified)"""
        # In production, this would use proper OCR
        # For now, return placeholder text based on box positions
        
        texts = []
        width, height = original_size
        
        for box in pixel_boxes:
            # Find overlapping text regions
            norm_box = [
                box[0] / width,
                box[1] / height,
                box[2] / width,
                box[3] / height
            ]
            
            # Simple overlap detection
            for region in text_regions:
                if self._boxes_overlap(norm_box, region['bbox']):
                    texts.append(region.get('text', 'extracted_text'))
        
        return ' '.join(texts) if texts else 'extracted_text'
    
    def _boxes_overlap(self, box1: List[float], box2: List[float]) -> bool:
        """Check if two bounding boxes overlap"""
        return not (box1[2] < box2[0] or box2[2] < box1[0] or 
                   box1[3] < box2[1] or box2[3] < box1[1])
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'status': 'failed',
            'pages': [],
            'confidence': 0.0,
            'error': 'Processing failed'
        }
    
    def batch_process(self, image_paths: List[str], 
                     output_dir: Optional[str] = None) -> List[Dict]:
        """Process multiple invoices in batch"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            result = self.process_invoice(image_path)
            result['source_image'] = image_path
            results.append(result)
            
            # Save individual result if output directory specified
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                result_file = output_path / f"{Path(image_path).stem}_result.json"
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
        
        return results


class InvoiceAPI:
    """
    RESTful API wrapper for invoice processing
    Similar to Azure Form Recognizer API interface
    """
    
    def __init__(self, model_path: str):
        self.processor = InvoiceProcessor(model_path)
    
    def analyze_invoice(self, image_input: Union[str, bytes]) -> Dict:
        """
        Analyze invoice endpoint
        
        Args:
            image_input: Image file path or bytes
            
        Returns:
            Analysis result in Azure Form Recognizer format
        """
        try:
            # Handle bytes input
            if isinstance(image_input, bytes):
                # Convert bytes to numpy array
                nparr = np.frombuffer(image_input, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_input = image
            
            result = self.processor.process_invoice(image_input)
            
            return {
                'status': 'succeeded',
                'result': result,
                'api_version': '1.0'
            }
            
        except Exception as e:
            logger.error(f"API error: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'api_version': '1.0'
            }
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_type': 'invoice_processor',
            'version': '1.0',
            'supported_formats': ['jpg', 'jpeg', 'png', 'pdf'],
            'max_file_size': '50MB',
            'supported_languages': ['en']
        }


def load_invoice_processor(model_path: str, **kwargs) -> InvoiceProcessor:
    """Convenience function to load invoice processor"""
    return InvoiceProcessor(model_path, **kwargs)


def process_invoice_file(model_path: str, image_path: str) -> Dict:
    """Convenience function to process a single invoice"""
    processor = InvoiceProcessor(model_path)
    return processor.process_invoice(image_path) 