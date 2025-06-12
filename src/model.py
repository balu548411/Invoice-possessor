import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, List, Tuple, Optional

class InvoiceTextEncoder(nn.Module):
    """Text encoder for invoice words/lines"""
    def __init__(self, vocab_size: int = 30000, embed_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=8, 
                dim_feedforward=1024,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of token indices [batch_size, seq_len]
            mask: Optional attention mask
        Returns:
            Encoded text features
        """
        x = self.embedding(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.output_proj(x)
        return x

class LayoutPositionalEncoding(nn.Module):
    """Positional encoding for 2D layout information"""
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.pos_encoder_x = nn.Linear(1, embed_dim // 4)
        self.pos_encoder_y = nn.Linear(1, embed_dim // 4)
        self.pos_encoder_w = nn.Linear(1, embed_dim // 4)
        self.pos_encoder_h = nn.Linear(1, embed_dim // 4)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, boxes):
        """
        Args:
            boxes: Normalized bounding boxes [batch_size, num_boxes, 4]
                  Format: [x_min, y_min, width, height]
        Returns:
            Position encodings
        """
        # Split into components
        x_min = boxes[:, :, 0:1]
        y_min = boxes[:, :, 1:2]
        width = boxes[:, :, 2:3]
        height = boxes[:, :, 3:4]
        
        # Encode each component
        x_enc = self.pos_encoder_x(x_min)
        y_enc = self.pos_encoder_y(y_min)
        w_enc = self.pos_encoder_w(width)
        h_enc = self.pos_encoder_h(height)
        
        # Concatenate
        pos_enc = torch.cat([x_enc, y_enc, w_enc, h_enc], dim=-1)
        pos_enc = self.output_proj(pos_enc)
        
        return pos_enc

class InvoiceVisionModel(nn.Module):
    """Vision component of the invoice model using a pre-trained ViT"""
    def __init__(self, pretrained: bool = True, embed_dim: int = 768):
        super().__init__()
        # Use ViT as the visual backbone
        self.backbone = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=pretrained,
            num_classes=0  # Remove classifier head
        )
        
        # Project backbone features to common embedding space
        self.projection = nn.Linear(self.backbone.embed_dim, embed_dim)
    
    def forward(self, images):
        """
        Args:
            images: Input images [batch_size, channels, height, width]
        Returns:
            Visual features
        """
        # Get features from backbone
        features = self.backbone.forward_features(images)  # [B, num_patches, embed_dim]
        
        # Project to common embedding space
        features = self.projection(features)
        
        return features

class KeyFieldExtractor(nn.Module):
    """Extract key invoice fields from encoded document features"""
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.fields = [
            'invoice_number', 'date', 'due_date', 'total_amount',
            'vendor_name', 'customer_name'
        ]
        
        # Shared encoding
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Field-specific heads
        self.field_extractors = nn.ModuleDict()
        for field in self.fields:
            self.field_extractors[field] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)  # Output confidence score
            )
    
    def forward(self, features, word_features, boxes):
        """
        Args:
            features: Document-level features [batch_size, embed_dim]
            word_features: Word-level features [batch_size, num_words, embed_dim]
            boxes: Word bounding boxes [batch_size, num_words, 4]
        Returns:
            Dictionary of predicted field values and locations
        """
        batch_size, num_words, _ = word_features.shape
        
        # Process document features
        doc_encoding = self.shared_encoder(features)
        
        # For each field, predict which text element contains the value
        results = {}
        for field in self.fields:
            # Expand document features to match words
            doc_features_expanded = doc_encoding.unsqueeze(1).expand(-1, num_words, -1)
            
            # Predict confidence score for each word
            field_scores = self.field_extractors[field](doc_features_expanded).squeeze(-1)
            
            # Get most likely word for this field
            confidence, indices = torch.max(field_scores, dim=1)
            
            # Store predictions
            results[field] = {
                'indices': indices,               # Indices of most likely words
                'confidence': confidence          # Confidence scores
            }
        
        return results

class InvoiceProcessorModel(nn.Module):
    """Complete invoice processing model"""
    def __init__(self,
                 vocab_size: int = 30000,
                 max_seq_length: int = 512,
                 embed_dim: int = 256):
        super().__init__()
        
        # Visual backbone
        self.vision_model = InvoiceVisionModel(pretrained=True, embed_dim=embed_dim)
        
        # Text encoding for OCR tokens
        self.text_encoder = InvoiceTextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=3
        )
        
        # Layout position encoding
        self.layout_encoder = LayoutPositionalEncoding(embed_dim=embed_dim)
        
        # Multimodal fusion transformer
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=1024,
                batch_first=True
            ),
            num_layers=4
        )
        
        # Document-level pooling
        self.doc_pooler = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Field extractor
        self.field_extractor = KeyFieldExtractor(
            input_dim=embed_dim,
            hidden_dim=512
        )
    
    def forward(self, images, tokens, boxes, token_mask=None):
        """
        Args:
            images: Batch of invoice images [batch_size, channels, height, width]
            tokens: OCR token ids [batch_size, seq_length]
            boxes: Normalized bounding boxes for OCR tokens [batch_size, seq_length, 4]
            token_mask: Mask for padding tokens (1 for pad, 0 for valid)
        Returns:
            Dictionary of extracted fields with values and confidence scores
        """
        # Get visual features
        visual_features = self.vision_model(images)
        
        # Get text features
        text_features = self.text_encoder(tokens, mask=token_mask)
        
        # Get position encoding
        pos_encoding = self.layout_encoder(boxes)
        
        # Combine text features and position encoding
        text_pos_features = text_features + pos_encoding
        
        # Concatenate visual and text features
        # We use the visual tokens (patch embeddings) and OCR token embeddings
        combined_features = torch.cat([visual_features, text_pos_features], dim=1)
        
        # Create attention mask for the combined features
        # Visual tokens can attend to all other tokens
        seq_len = visual_features.size(1) + text_pos_features.size(1)
        visual_len = visual_features.size(1)
        
        # If token_mask is provided, extend it to include visual tokens
        if token_mask is not None:
            # token_mask has 1 for pad tokens, 0 for valid tokens
            # We need to add zeros for visual tokens (all valid)
            batch_size = token_mask.size(0)
            extended_mask = torch.zeros((batch_size, seq_len), device=token_mask.device)
            extended_mask[:, visual_len:] = token_mask
        else:
            extended_mask = None
        
        # Process through fusion transformer
        fused_features = self.fusion_transformer(combined_features, src_key_padding_mask=extended_mask)
        
        # Pool document features for global representation
        # Take CLS token (first token from ViT)
        doc_features = fused_features[:, 0]
        doc_features = self.doc_pooler(doc_features)
        
        # Extract OCR word features (skip visual tokens)
        word_features = fused_features[:, visual_len:]
        
        # Extract fields
        field_predictions = self.field_extractor(doc_features, word_features, boxes)
        
        return {
            'doc_features': doc_features,
            'word_features': word_features,
            'field_predictions': field_predictions
        }
    
    def extract_fields_from_predictions(self, predictions, ocr_texts):
        """
        Extract readable field values from model predictions
        
        Args:
            predictions: Model predictions dict
            ocr_texts: Original OCR text for each token [batch_size, seq_length]
        Returns:
            Dictionary mapping field names to extracted text values
        """
        batch_size = len(ocr_texts)
        field_preds = predictions['field_predictions']
        
        results = []
        for b in range(batch_size):
            sample_results = {}
            for field_name, field_data in field_preds.items():
                index = field_data['indices'][b].item()
                confidence = field_data['confidence'][b].item()
                
                # Only include predictions with reasonable confidence
                if confidence > 0.5 and index < len(ocr_texts[b]):
                    sample_results[field_name] = {
                        'text': ocr_texts[b][index],
                        'confidence': confidence
                    }
                else:
                    sample_results[field_name] = {
                        'text': "",
                        'confidence': confidence
                    }
            
            results.append(sample_results)
        
        return results


def create_invoice_model(
    pretrained: bool = True, 
    vocab_size: int = 30000,
    max_seq_len: int = 512, 
    embed_dim: int = 256
) -> InvoiceProcessorModel:
    """
    Create an instance of the InvoiceProcessorModel
    
    Args:
        pretrained: Whether to use pretrained vision backbone
        vocab_size: Size of token vocabulary
        max_seq_len: Maximum sequence length for tokens
        embed_dim: Embedding dimension
    Returns:
        Initialized model
    """
    model = InvoiceProcessorModel(
        vocab_size=vocab_size,
        max_seq_length=max_seq_len,
        embed_dim=embed_dim
    )
    
    return model 