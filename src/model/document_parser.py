import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG
from model.backbones import build_backbone
from model.transformer import Transformer


class PositionEmbeddingSine(nn.Module):
    """
    2D sine position embedding for image features.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        # mask: [batch_size, h, w]
        not_mask = ~mask  # Invert mask: True where we have feature vectors
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # [batch_size, h, w]
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # [batch_size, h, w]
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # Create position embeddings
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # [batch_size, h, w, num_pos_feats]
        pos_y = y_embed[:, :, :, None] / dim_t  # [batch_size, h, w, num_pos_feats]
        
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # [batch_size, 2*num_pos_feats, h, w]
        return pos


class DocumentParser(nn.Module):
    """
    Main document parsing model combining visual backbone and transformer.
    """
    def __init__(self, config=None):
        super().__init__()
        
        if config is None:
            config = MODEL_CONFIG
        
        # Build backbone
        self.backbone = build_backbone(config)
        
        # Feature dimensions from backbone
        hidden_dim = config["hidden_dim"]
        
        # Project backbone features to transformer dimension
        self.input_proj = nn.Conv2d(self.backbone.out_channels, hidden_dim, kernel_size=1)
        
        # Build transformer
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=config["encoder_heads"],
            num_encoder_layers=config["encoder_layers"],
            num_decoder_layers=config["decoder_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            activation=config["activation"],
        )
        
        # Define position embedding
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        
        # Query embeddings for the decoder (learnable)
        self.query_embed = nn.Embedding(config["num_queries"], hidden_dim)
        
        # Prediction heads
        num_classes = len(config["entity_classes"])
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background class
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 4 for [x1, y1, x2, y2]
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights of prediction heads."""
        # Init class embedding
        nn.init.constant_(self.class_embed.bias, 0)
        nn.init.orthogonal_(self.class_embed.weight)
        
        # Init bbox MLP
        for layer in self.bbox_embed.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, images):
        """
        Forward pass for document parsing.
        
        Args:
            images: Batch of document images [batch_size, 3, H, W]
            
        Returns:
            Dict with predictions (class logits, bboxes)
        """
        # Extract features from backbone
        features = self.backbone(images)
        features = features["feat"]
        
        # Create attention mask (None for now since we process full images)
        mask = torch.zeros((features.shape[0], features.shape[2], features.shape[3]), 
                          dtype=torch.bool, device=features.device)
        
        # Create positional embeddings
        pos_embed = self.position_embedding(mask)
        
        # Project features to transformer dimension
        features_proj = self.input_proj(features)
        
        # Run transformer encoder-decoder
        hs, memory = self.transformer(features_proj, mask, self.query_embed.weight, pos_embed)
        
        # Apply prediction heads to each decoder layer output
        outputs_class = self.class_embed(hs)  # [batch_size, num_queries, num_classes+1]
        outputs_coord = self.bbox_embed(hs).sigmoid()  # [batch_size, num_queries, 4]
        
        # Return only the last decoder layer outputs
        out = {
            'pred_logits': outputs_class[-1],  # [batch_size, num_queries, num_classes+1]
            'pred_boxes': outputs_coord[-1],   # [batch_size, num_queries, 4]
        }
        
        # Include intermediate outputs during training
        if self.training:
            out['aux_outputs'] = [
                {'pred_logits': outputs_class[i], 'pred_boxes': outputs_coord[i]}
                for i in range(len(outputs_class) - 1)
            ]
            
        return out


class MLP(nn.Module):
    """
    Multi-layer perceptron with specified number of layers.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_model(config=None):
    """Build document parsing model."""
    if config is None:
        config = MODEL_CONFIG
    model = DocumentParser(config)
    return model 