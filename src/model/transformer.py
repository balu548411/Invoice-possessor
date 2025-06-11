import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Optional, List


class Transformer(nn.Module):
    """
    Enhanced Transformer for document parsing with stochastic depth.
    
    Args:
        d_model: Feature dimension
        nhead: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        dim_feedforward: Hidden dimension of FFN
        dropout: Dropout probability
        activation: Activation function
        stochastic_depth_prob: Stochastic depth probability
    """
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        stochastic_depth_prob=0.0,
    ):
        super().__init__()

        # Build encoder
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm, stochastic_depth_prob
        )

        # Build decoder
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm, stochastic_depth_prob
        )

        self.d_model = d_model
        self.nhead = nhead
        
        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        """
        Forward pass for the transformer.
        
        Args:
            src: Image features [batch_size, channels, height, width]
            mask: Attention mask [batch_size, height, width]
            query_embed: Object query embeddings [num_queries, d_model]
            pos_embed: Position embeddings [batch_size, channels, height, width]
            
        Returns:
            decoder_output: Decoder output [num_layers, batch_size, num_queries, d_model]
            encoder_output: Encoder output [batch_size, height*width, d_model]
        """
        # Flatten feature map
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # [h*w, batch_size, channels]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # [h*w, batch_size, channels]
        mask = mask.flatten(1) if mask is not None else None  # [batch_size, h*w]
        
        # Process query embeddings
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # [num_queries, batch_size, d_model]
        
        # Initialize target with zeros
        target = torch.zeros_like(query_embed)
        
        # Run transformer encoder
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        
        # Run transformer decoder
        hs = self.decoder(target, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        
        # Reshape outputs
        memory = memory.permute(1, 0, 2)  # [batch_size, h*w, channels]
        hs = hs.permute(0, 2, 1, 3)  # [num_layers, batch_size, num_queries, channels]
        
        return hs, memory


class TransformerEncoder(nn.Module):
    """
    TransformerEncoder with stochastic depth.
    
    Args:
        encoder_layer: Single encoder layer
        num_layers: Number of layers
        norm: Normalization layer
        stochastic_depth_prob: Probability of dropping a layer
    """
    def __init__(self, encoder_layer, num_layers, norm=None, stochastic_depth_prob=0.0):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        
        # Set up stochastic depth
        self.stochastic_depth_prob = stochastic_depth_prob
        if stochastic_depth_prob > 0:
            # Linear decay of drop probability from 0 to stochastic_depth_prob
            self.drop_probs = [i / (num_layers - 1) * stochastic_depth_prob for i in range(num_layers)]
        else:
            self.drop_probs = [0.0] * num_layers

    def forward(
        self,
        src,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through encoder.
        
        Args:
            src: Input features
            mask: Attention mask
            src_key_padding_mask: Source padding mask
            pos: Position embeddings
        """
        output = src
        
        # Process each layer
        for idx, layer in enumerate(self.layers):
            # Apply stochastic depth during training
            if self.training and self.drop_probs[idx] > 0 and torch.rand(1).item() < self.drop_probs[idx]:
                continue
                
            output = layer(output, src_mask=mask,
                          src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """
    TransformerDecoder with stochastic depth.
    
    Args:
        decoder_layer: Single decoder layer
        num_layers: Number of layers
        norm: Normalization layer
        stochastic_depth_prob: Probability of dropping a layer
    """
    def __init__(self, decoder_layer, num_layers, norm=None, stochastic_depth_prob=0.0):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        
        # Set up stochastic depth
        self.stochastic_depth_prob = stochastic_depth_prob
        if stochastic_depth_prob > 0:
            # Linear decay of drop probability from 0 to stochastic_depth_prob
            self.drop_probs = [i / (num_layers - 1) * stochastic_depth_prob for i in range(num_layers)]
        else:
            self.drop_probs = [0.0] * num_layers
        
        # Track outputs of each layer for auxiliary losses
        self.return_intermediate = True

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through decoder.
        
        Args:
            tgt: Target embeddings
            memory: Memory from encoder
            tgt_mask: Target attention mask
            memory_mask: Memory attention mask
            tgt_key_padding_mask: Target padding mask
            memory_key_padding_mask: Memory padding mask
            pos: Position embeddings
            query_pos: Query position embeddings
        """
        output = tgt

        intermediate = []

        for idx, layer in enumerate(self.layers):
            # Apply stochastic depth during training
            if self.training and self.drop_probs[idx] > 0 and torch.rand(1).item() < self.drop_probs[idx]:
                intermediate.append(output)
                continue
                
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    """
    Enhanced transformer encoder layer with pre-norm architecture.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation
        self.activation = _get_activation_fn(activation)
        
    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for encoder layer.
        """
        # Pre-norm architecture
        src2 = self.norm1(src)
        
        # Add position embeddings to queries and keys
        q = k = self.with_pos_embed(src2, pos)
        
        # Self-attention
        src2, _ = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)
        
        # Residual connection and dropout
        src = src + self.dropout1(src2)
        
        # Pre-norm for FFN
        src2 = self.norm2(src)
        
        # Feed-forward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        
        # Residual connection and dropout
        src = src + self.dropout2(src2)
        
        return src


class TransformerDecoderLayer(nn.Module):
    """
    Enhanced transformer decoder layer with pre-norm architecture.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        
        # Cross-attention
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Activation
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for decoder layer.
        """
        # Pre-norm for self-attention
        tgt2 = self.norm1(tgt)
        
        # Add position embeddings to queries and keys
        q = k = self.with_pos_embed(tgt2, query_pos)
        
        # Self-attention
        tgt2, _ = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                               key_padding_mask=tgt_key_padding_mask)
        
        # Residual connection and dropout
        tgt = tgt + self.dropout1(tgt2)
        
        # Pre-norm for cross attention
        tgt2 = self.norm2(tgt)
        
        # Cross-attention with memory
        tgt2, _ = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        
        # Residual connection and dropout
        tgt = tgt + self.dropout2(tgt2)
        
        # Pre-norm for FFN
        tgt2 = self.norm3(tgt)
        
        # Feed-forward network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        
        # Residual connection and dropout
        tgt = tgt + self.dropout3(tgt2)
        
        return tgt


def _get_clones(module, N):
    """Create N copies of the module."""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return the activation function."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}") 