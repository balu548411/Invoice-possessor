import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import BertModel, BertConfig
import timm
from typing import Dict, Tuple, Optional, List

from config import MODEL_CONFIG


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but should be saved and loaded with model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        Returns:
            x + positional encodings
        """
        return x + self.pe[:, :x.size(1)]


class VisionEncoder(nn.Module):
    """
    Vision encoder that processes document images.
    """
    def __init__(self, model_name="resnet50", pretrained=True, output_dim=768):
        super().__init__()
        
        self.model_name = model_name
        
        if model_name == "resnet50":
            # Load pretrained ResNet and remove final FC layer
            base_model = models.resnet50(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
            self.feature_dim = 2048
            
        elif model_name == "efficientnet_b3":
            # Load EfficientNet from timm
            base_model = timm.create_model('efficientnet_b3', pretrained=pretrained, features_only=True)
            self.feature_extractor = base_model
            self.feature_dim = 1536  # EfficientNet-B3's last feature map channels
            
        elif model_name == "vit_base_patch16_224":
            # Load Vision Transformer from timm
            base_model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_dim = 768
            
        else:
            raise ValueError(f"Unsupported vision encoder: {model_name}")
            
        # Add projection if needed
        if self.feature_dim != output_dim:
            self.projection = nn.Linear(self.feature_dim, output_dim)
        else:
            self.projection = nn.Identity()
            
    def forward(self, x):
        """
        Args:
            x: Input images of shape [batch_size, channels, height, width]
            
        Returns:
            features: Image features of shape [batch_size, num_features, output_dim]
                     where num_features depends on the model and input size
        """
        if self.model_name == "vit_base_patch16_224":
            # For ViT, return all patch embeddings
            features = self.feature_extractor(x)
            features = features.reshape(features.size(0), -1, self.feature_dim)
        else:
            # For CNN architectures
            features = self.feature_extractor(x)  # [batch_size, channels, h, w]
            
            # Reshape to [batch_size, num_features, channels]
            batch_size, channels, height, width = features.size()
            features = features.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
            
        # Project to output dimension
        features = self.projection(features)
        
        return features


class InvoiceTransformer(nn.Module):
    """
    Transformer model for invoice understanding, converting document images to structured JSON.
    Uses a vision encoder (CNN/ViT) followed by a transformer decoder.
    """
    def __init__(self, vocab_size, config=None):
        super().__init__()
        
        if config is None:
            config = MODEL_CONFIG
            
        self.config = config
        self.vocab_size = vocab_size
        
        # Vision Encoder
        self.vision_encoder = VisionEncoder(
            model_name=config['vision_encoder'],
            pretrained=config['vision_encoder_pretrained'],
            output_dim=config['hidden_dim']
        )
        
        # Text token embeddings
        self.token_embedding = nn.Embedding(vocab_size, config['hidden_dim'])
        self.positional_encoding = PositionalEncoding(config['hidden_dim'])
        
        # Transformer modules
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['hidden_dim'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config['num_encoder_layers']
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config['hidden_dim'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config['num_decoder_layers']
        )
        
        # Output projection layer
        self.output_projection = nn.Linear(config['hidden_dim'], vocab_size)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token (index 0)
        
    def _create_mask(self, target):
        """
        Create masking for transformer decoder.
        
        Args:
            target: Target tensor of shape [batch_size, seq_len]
            
        Returns:
            target_mask: Mask of shape [batch_size, seq_len, seq_len] with triangular mask
            target_padding_mask: Padding mask of shape [batch_size, seq_len] for padding tokens
        """
        batch_size, seq_len = target.shape
        
        # Create causal attention mask (can't look ahead)
        target_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        target_mask = target_mask.to(target.device)
        
        # Create padding mask (1 for pad tokens, 0 for others)
        target_padding_mask = (target == 0)
        
        return target_mask, target_padding_mask
        
    def forward(self, images, target=None, teacher_forcing_ratio=1.0):
        """
        Forward pass through the model.
        
        Args:
            images: Images of shape [batch_size, channels, height, width]
            target: Target token indices of shape [batch_size, seq_len]
            teacher_forcing_ratio: Probability of using teacher forcing (0-1)
            
        Returns:
            output: Output logits of shape [batch_size, seq_len, vocab_size]
            loss: Loss value if target is provided, None otherwise
        """
        batch_size = images.size(0)
        device = images.device
        
        # Get encoder features
        memory = self.vision_encoder(images)
        
        if self.training and target is not None:
            # Training mode with teacher forcing
            target_seq_len = target.size(1)
            
            # Shift target for teacher forcing (input = target shifted right)
            # We remove the last token from target and prepend a start token (1)
            decoder_input = torch.cat([
                torch.ones(batch_size, 1, dtype=torch.long, device=device),
                target[:, :-1]
            ], dim=1)
            
            # Embed tokens and add positional encoding
            embedded_tokens = self.token_embedding(decoder_input)
            embedded_tokens = self.positional_encoding(embedded_tokens)
            
            # Create attention masks
            target_mask, target_padding_mask = self._create_mask(decoder_input)
            
            # Apply transformer decoder
            output = self.transformer_decoder(
                embedded_tokens, 
                memory,
                tgt_mask=target_mask,
                tgt_key_padding_mask=target_padding_mask
            )
            
            # Project to vocabulary
            logits = self.output_projection(output)
            
            # Compute loss if target is provided
            loss = self.criterion(logits.reshape(-1, self.vocab_size), target.reshape(-1))
            
            return logits, loss
            
        else:
            # Inference mode with autoregressive generation
            # Start with batch of start tokens
            current_token = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            max_length = 512 if target is None else target.size(1)
            outputs = []
            
            # Autoregressive generation
            for i in range(max_length):
                # Embed tokens and add positional encoding
                embedded_tokens = self.token_embedding(current_token)
                embedded_tokens = self.positional_encoding(embedded_tokens)
                
                # Create attention masks
                target_mask, target_padding_mask = self._create_mask(current_token)
                
                # Apply transformer decoder
                output = self.transformer_decoder(
                    embedded_tokens, 
                    memory,
                    tgt_mask=target_mask,
                    tgt_key_padding_mask=target_padding_mask
                )
                
                # Project to vocabulary
                step_logits = self.output_projection(output[:, -1:])
                outputs.append(step_logits)
                
                # Sample next token
                next_token = step_logits.argmax(dim=-1)
                
                # Check if we need teacher forcing for this step
                if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                    # Use ground truth token at this step if available
                    if i < target.size(1) - 1:  # Ensure we don't go out of bounds
                        next_token = target[:, i+1:i+2]
                        
                # Append next token to current sequence
                current_token = torch.cat([current_token, next_token], dim=1)
                
                # Check if all sequences have generated end token
                if (next_token == 2).all():  # 2 = end token
                    break
                    
            # Concatenate all step outputs
            logits = torch.cat(outputs, dim=1)
            
            # Compute loss if target is provided
            loss = None
            if target is not None:
                # We need to pad/truncate logits to match target length
                seq_len = min(logits.size(1), target.size(1) - 1)  # -1 because we didn't predict the first token
                loss = self.criterion(
                    logits[:, :seq_len].reshape(-1, self.vocab_size), 
                    target[:, 1:seq_len+1].reshape(-1)
                )
            
            return logits, loss
    
    def generate(self, image, max_length=512, temperature=1.0):
        """
        Generate output sequence for a single image.
        
        Args:
            image: Input image tensor of shape [1, channels, height, width]
            max_length: Maximum length of generated sequence
            temperature: Temperature for sampling (higher = more random)
            
        Returns:
            tokens: Generated token indices
        """
        device = image.device
        
        # Get encoder features
        memory = self.vision_encoder(image)
        
        # Start with start token
        current_token = torch.ones(1, 1, dtype=torch.long, device=device)
        
        # Storage for generated tokens
        generated_tokens = [current_token.item()]
        
        # Autoregressive generation
        for i in range(max_length - 1):
            # Embed tokens and add positional encoding
            embedded_tokens = self.token_embedding(current_token)
            embedded_tokens = self.positional_encoding(embedded_tokens)
            
            # Create attention masks
            target_mask, target_padding_mask = self._create_mask(current_token)
            
            # Apply transformer decoder
            output = self.transformer_decoder(
                embedded_tokens, 
                memory,
                tgt_mask=target_mask,
                tgt_key_padding_mask=target_padding_mask
            )
            
            # Project to vocabulary
            logits = self.output_projection(output[:, -1:])
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
                
            # Sample next token
            probs = F.softmax(logits.squeeze(), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to generated tokens
            generated_tokens.append(next_token.item())
            
            # Append next token to current sequence
            current_token = torch.cat([current_token, next_token.view(1, 1)], dim=1)
            
            # Stop if we generated end token
            if next_token.item() == 2:  # 2 = end token
                break
                
        return generated_tokens


# Utility function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model architecture
    from config import MAX_SEQ_LENGTH
    
    # Create a dummy batch
    batch_size = 2
    channels = 3
    height, width = 800, 800
    seq_len = 256
    vocab_size = 10000
    
    images = torch.randn(batch_size, channels, height, width)
    target = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create model
    model = InvoiceTransformer(vocab_size)
    
    # Forward pass
    logits, loss = model(images, target)
    
    # Print model statistics
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Output shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation
    test_image = torch.randn(1, channels, height, width)
    tokens = model.generate(test_image, max_length=32)
    print(f"Generated tokens: {tokens}") 