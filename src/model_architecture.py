import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import timm
from typing import Dict, List, Tuple, Optional
import numpy as np

class PositionalEncoding2D(nn.Module):
    """2D Positional encoding for spatial relationships"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            boxes: [batch_size, num_boxes, 4] normalized coordinates
        Returns:
            pos_emb: [batch_size, num_boxes, d_model]
        """
        batch_size, num_boxes, _ = boxes.shape
        
        # Normalize coordinates to [0, 1]
        x_center = (boxes[:, :, 0] + boxes[:, :, 2]) / 2
        y_center = (boxes[:, :, 1] + boxes[:, :, 3]) / 2
        
        # Scale to positional encoding range
        x_pos = (x_center * (self.pe.size(0) - 1)).long()
        y_pos = (y_center * (self.pe.size(0) - 1)).long()
        
        # Get positional embeddings
        x_emb = self.pe[x_pos]  # [batch_size, num_boxes, d_model]
        y_emb = self.pe[y_pos]  # [batch_size, num_boxes, d_model]
        
        # Combine x and y embeddings
        pos_emb = x_emb + y_emb
        
        return pos_emb


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for understanding layout relationships"""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.spatial_bias = nn.Parameter(torch.randn(num_heads, 1, 1))
        
    def forward(self, x: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Add spatial bias based on bounding box relationships
        spatial_bias = self._compute_spatial_bias(boxes)
        scores = scores + spatial_bias  # spatial_bias is already batch-compatible
        
        # Apply attention
        attention_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads and project
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.out_linear(attended)
    
    def _compute_spatial_bias(self, boxes: torch.Tensor) -> torch.Tensor:
        """Compute spatial bias based on bounding box relationships"""
        batch_size, num_boxes = boxes.size(0), boxes.size(1)
        
        # Compute center points
        centers = torch.stack([
            (boxes[:, :, 0] + boxes[:, :, 2]) / 2,
            (boxes[:, :, 1] + boxes[:, :, 3]) / 2
        ], dim=-1)  # [batch_size, num_boxes, 2]
        
        # Compute pairwise distances
        centers_expanded = centers.unsqueeze(2)  # [batch_size, num_boxes, 1, 2]
        centers_transposed = centers.unsqueeze(1)  # [batch_size, 1, num_boxes, 2]
        
        distances = torch.norm(centers_expanded - centers_transposed, dim=-1)
        # distances shape: [batch_size, num_boxes, num_boxes]
        
        # Convert distances to bias (closer boxes have higher bias)
        # Apply spatial bias scaling per head and expand for batch
        spatial_bias_scale = self.spatial_bias.view(self.num_heads, 1, 1, 1)  # [num_heads, 1, 1, 1]
        distances_expanded = distances.unsqueeze(0)  # [1, batch_size, num_boxes, num_boxes]
        
        # Compute bias: [num_heads, batch_size, num_boxes, num_boxes]
        spatial_bias = -distances_expanded * spatial_bias_scale
        
        # Transpose to match attention scores format: [batch_size, num_heads, num_boxes, num_boxes]
        return spatial_bias.transpose(0, 1)


class MultiModalInvoiceEncoder(nn.Module):
    """Multi-modal encoder combining visual and textual features"""
    
    def __init__(self, 
                 vision_model: str = 'efficientnet_b3',
                 text_model: str = 'microsoft/layoutlm-base-uncased',
                 d_model: int = 768,
                 num_layers: int = 6,
                 num_heads: int = 12):
        super().__init__()
        
        self.d_model = d_model
        
        # Vision backbone
        self.vision_backbone = timm.create_model(
            vision_model, 
            pretrained=True, 
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        
        # Get vision feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 512, 512)
            vision_features = self.vision_backbone(dummy_input)
            vision_dim = vision_features.shape[1]
        
        # Text backbone (LayoutLM or similar)
        self.text_backbone = AutoModel.from_pretrained(text_model)
        text_dim = self.text_backbone.config.hidden_size
        
        # Projection layers to common dimension
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(d_model)
        
        # Transformer layers for multi-modal fusion
        self.fusion_layers = nn.ModuleList([
            SpatialAttention(d_model, num_heads) for _ in range(num_layers)
        ])
        
        # Layer normalization and dropout
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, images: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        batch_size = images.size(0)
        
        # Extract visual features
        vision_features = self.vision_backbone(images)  # [batch_size, vision_dim, H, W]
        
        # Global average pooling for image-level features
        vision_global = F.adaptive_avg_pool2d(vision_features, (1, 1))
        vision_global = vision_global.flatten(2).transpose(1, 2)  # [batch_size, 1, vision_dim]
        vision_global = self.vision_proj(vision_global)  # [batch_size, 1, d_model]
        
        # Process each bounding box region
        box_features = []
        for i in range(batch_size):
            img_features = vision_features[i]  # [vision_dim, H, W]
            img_boxes = boxes[i]  # [num_boxes, 4]
            
            # Extract ROI features for each box
            roi_features = self._extract_roi_features(img_features, img_boxes)
            box_features.append(roi_features)
        
        box_features = torch.stack(box_features)  # [batch_size, num_boxes, vision_dim]
        box_features = self.vision_proj(box_features)  # [batch_size, num_boxes, d_model]
        
        # Add positional encoding
        pos_emb = self.pos_encoding(boxes)
        box_features = box_features + pos_emb
        
        # Combine global and local features
        features = torch.cat([vision_global, box_features], dim=1)
        
        # Apply fusion layers
        for fusion_layer, layer_norm in zip(self.fusion_layers, self.layer_norms):
            # Create extended boxes for global feature (dummy box)
            global_box = torch.zeros(batch_size, 1, 4, device=boxes.device)
            extended_boxes = torch.cat([global_box, boxes], dim=1)
            
            residual = features
            features = fusion_layer(features, extended_boxes)
            features = layer_norm(features + residual)
            features = self.dropout(features)
        
        return features
    
    def _extract_roi_features(self, img_features: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """Extract ROI features from image features using bounding boxes"""
        vision_dim, H, W = img_features.shape
        num_boxes = boxes.size(0)
        
        roi_features = []
        
        for box in boxes:
            if box.sum() == 0:  # Skip padding boxes
                roi_features.append(torch.zeros(vision_dim, device=img_features.device))
                continue
            
            # Convert normalized coordinates to pixel coordinates
            x1 = int(box[0] * W)
            y1 = int(box[1] * H)
            x2 = int(box[2] * W)
            y2 = int(box[3] * H)
            
            # Ensure valid coordinates
            x1, x2 = max(0, min(x1, x2)), min(W, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(H, max(y1, y2))
            
            if x2 > x1 and y2 > y1:
                # Extract region and pool
                roi = img_features[:, y1:y2, x1:x2]
                pooled = F.adaptive_avg_pool2d(roi.unsqueeze(0), (1, 1))
                roi_features.append(pooled.squeeze())
            else:
                roi_features.append(torch.zeros(vision_dim, device=img_features.device))
        
        return torch.stack(roi_features)


class InvoiceEntityClassifier(nn.Module):
    """Entity classification head for invoice fields"""
    
    def __init__(self, d_model: int, num_entity_types: int = 10):
        super().__init__()
        
        self.entity_types = [
            'O',  # Outside any entity
            'B-INVOICE_NUM', 'I-INVOICE_NUM',
            'B-DATE', 'I-DATE',
            'B-TOTAL', 'I-TOTAL',
            'B-VENDOR', 'I-VENDOR',
            'B-CUSTOMER', 'I-CUSTOMER'
        ]
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, len(self.entity_types))
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class InvoiceProcessingModel(nn.Module):
    """
    Complete invoice processing model combining all components
    Similar to Azure Form Recognizer but more powerful
    """
    
    def __init__(self, 
                 vision_model: str = 'efficientnet_b3',
                 text_model: str = 'microsoft/layoutlm-base-uncased',
                 d_model: int = 768,
                 num_layers: int = 6,
                 num_heads: int = 12,
                 num_entity_types: int = 11):
        super().__init__()
        
        # Multi-modal encoder
        self.encoder = MultiModalInvoiceEncoder(
            vision_model=vision_model,
            text_model=text_model,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads
        )
        
        # Task-specific heads
        self.entity_classifier = InvoiceEntityClassifier(d_model, num_entity_types)
        
        # Additional heads for key-value extraction
        self.key_classifier = nn.Linear(d_model, 2)  # Is this a key?
        self.value_classifier = nn.Linear(d_model, 2)  # Is this a value?
        
        # Regression head for confidence scores
        self.confidence_regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        images = batch['images']
        boxes = batch['boxes']
        
        # Encode multi-modal features
        features = self.encoder(images, boxes)
        
        # Skip global feature for entity classification (first token)
        box_features = features[:, 1:, :]  # [batch_size, num_boxes, d_model]
        
        # Entity classification
        entity_logits = self.entity_classifier(box_features)
        
        # Key-value classification
        key_logits = self.key_classifier(box_features)
        value_logits = self.value_classifier(box_features)
        
        # Confidence prediction
        confidence_scores = self.confidence_regressor(box_features)
        
        return {
            'entity_logits': entity_logits,
            'key_logits': key_logits,
            'value_logits': value_logits,
            'confidence_scores': confidence_scores,
            'features': features
        }
    
    def extract_entities(self, outputs: Dict[str, torch.Tensor], 
                        batch: Dict[str, torch.Tensor]) -> List[Dict]:
        """Extract structured entities from model outputs"""
        entity_logits = outputs['entity_logits']
        confidence_scores = outputs['confidence_scores']
        
        batch_size = entity_logits.size(0)
        results = []
        
        for i in range(batch_size):
            # Get predictions for this sample
            entity_preds = torch.argmax(entity_logits[i], dim=-1)
            confidences = confidence_scores[i].squeeze(-1)
            boxes = batch['boxes'][i]
            
            # Extract entities using BIO tagging
            entities = self._extract_bio_entities(
                entity_preds, confidences, boxes, 
                self.entity_classifier.entity_types
            )
            
            results.append(entities)
        
        return results
    
    def _extract_bio_entities(self, predictions: torch.Tensor, 
                             confidences: torch.Tensor,
                             boxes: torch.Tensor,
                             entity_types: List[str]) -> Dict:
        """Extract entities using BIO tagging scheme"""
        entities = {}
        current_entity = None
        current_text = []
        current_boxes = []
        current_confidences = []
        
        for i, (pred, conf, box) in enumerate(zip(predictions, confidences, boxes)):
            if box.sum() == 0:  # Skip padding
                continue
                
            pred_label = entity_types[pred.item()]
            
            if pred_label.startswith('B-'):
                # Save previous entity if exists
                if current_entity:
                    entities[current_entity] = {
                        'boxes': current_boxes,
                        'confidence': sum(current_confidences) / len(current_confidences)
                    }
                
                # Start new entity
                current_entity = pred_label[2:]
                current_boxes = [box.tolist()]
                current_confidences = [conf.item()]
                
            elif pred_label.startswith('I-') and current_entity == pred_label[2:]:
                # Continue current entity
                current_boxes.append(box.tolist())
                current_confidences.append(conf.item())
                
            else:
                # End current entity
                if current_entity:
                    entities[current_entity] = {
                        'boxes': current_boxes,
                        'confidence': sum(current_confidences) / len(current_confidences)
                    }
                current_entity = None
                current_boxes = []
                current_confidences = []
        
        # Save final entity if exists
        if current_entity:
            entities[current_entity] = {
                'boxes': current_boxes,
                'confidence': sum(current_confidences) / len(current_confidences)
            }
        
        return entities 