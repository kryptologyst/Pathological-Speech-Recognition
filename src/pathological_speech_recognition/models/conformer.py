"""Conformer model for pathological speech recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from omegaconf import DictConfig

from .base import BasePathologicalSpeechModel


class ConvolutionModule(nn.Module):
    """Convolution module for Conformer."""
    
    def __init__(
        self,
        encoder_dim: int,
        conv_kernel_size: int = 31,
        dropout_p: float = 0.1,
        conv_expansion_factor: int = 2,
    ):
        """Initialize convolution module.
        
        Args:
            encoder_dim: Encoder dimension.
            conv_kernel_size: Convolution kernel size.
            dropout_p: Dropout probability.
            conv_expansion_factor: Convolution expansion factor.
        """
        super().__init__()
        
        self.pointwise_conv1 = nn.Conv1d(encoder_dim, encoder_dim * conv_expansion_factor, 1)
        self.pointwise_conv2 = nn.Conv1d(encoder_dim * conv_expansion_factor, encoder_dim, 1)
        
        self.depthwise_conv = nn.Conv1d(
            encoder_dim * conv_expansion_factor,
            encoder_dim * conv_expansion_factor,
            conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2,
            groups=encoder_dim * conv_expansion_factor,
        )
        
        self.layer_norm = nn.LayerNorm(encoder_dim * conv_expansion_factor)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolution module.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        # x: (batch, time, dim)
        x = x.transpose(1, 2)  # (batch, dim, time)
        
        x = self.pointwise_conv1(x)
        x = F.gelu(x)
        
        x = self.depthwise_conv(x)
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(x)
        
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        return x.transpose(1, 2)  # (batch, time, dim)


class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""
    
    def __init__(
        self,
        encoder_dim: int,
        num_attention_heads: int = 4,
        dropout_p: float = 0.1,
    ):
        """Initialize multi-head attention.
        
        Args:
            encoder_dim: Encoder dimension.
            num_attention_heads: Number of attention heads.
            dropout_p: Dropout probability.
        """
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            encoder_dim,
            num_attention_heads,
            dropout=dropout_p,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through attention module.
        
        Args:
            x: Input tensor.
            attention_mask: Attention mask.
            
        Returns:
            Output tensor.
        """
        attn_output, _ = self.attention(x, x, x, key_padding_mask=attention_mask)
        return self.dropout(attn_output)


class FeedForwardModule(nn.Module):
    """Feed-forward module for Conformer."""
    
    def __init__(
        self,
        encoder_dim: int,
        feed_forward_expansion_factor: int = 4,
        dropout_p: float = 0.1,
    ):
        """Initialize feed-forward module.
        
        Args:
            encoder_dim: Encoder dimension.
            feed_forward_expansion_factor: Feed-forward expansion factor.
            dropout_p: Dropout probability.
        """
        super().__init__()
        
        self.linear1 = nn.Linear(encoder_dim, encoder_dim * feed_forward_expansion_factor)
        self.linear2 = nn.Linear(encoder_dim * feed_forward_expansion_factor, encoder_dim)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feed-forward module.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)


class ConformerBlock(nn.Module):
    """Conformer block."""
    
    def __init__(
        self,
        encoder_dim: int,
        num_attention_heads: int = 4,
        conv_kernel_size: int = 31,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        dropout_p: float = 0.1,
        half_step_residual: bool = True,
    ):
        """Initialize Conformer block.
        
        Args:
            encoder_dim: Encoder dimension.
            num_attention_heads: Number of attention heads.
            conv_kernel_size: Convolution kernel size.
            feed_forward_expansion_factor: Feed-forward expansion factor.
            conv_expansion_factor: Convolution expansion factor.
            dropout_p: Dropout probability.
            half_step_residual: Whether to use half-step residual.
        """
        super().__init__()
        
        self.half_step_residual = half_step_residual
        self.feed_forward_residual_factor = 0.5 if half_step_residual else 1.0
        
        self.feed_forward1 = FeedForwardModule(
            encoder_dim,
            feed_forward_expansion_factor,
            dropout_p,
        )
        self.self_attention = MultiHeadAttention(
            encoder_dim,
            num_attention_heads,
            dropout_p,
        )
        self.convolution = ConvolutionModule(
            encoder_dim,
            conv_kernel_size,
            dropout_p,
            conv_expansion_factor,
        )
        self.feed_forward2 = FeedForwardModule(
            encoder_dim,
            feed_forward_expansion_factor,
            dropout_p,
        )
        
        self.layer_norm = nn.LayerNorm(encoder_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through Conformer block.
        
        Args:
            x: Input tensor.
            attention_mask: Attention mask.
            
        Returns:
            Output tensor.
        """
        residual = x
        x = self.feed_forward1(x)
        x = residual + x * self.feed_forward_residual_factor
        
        residual = x
        x = self.layer_norm(x)
        x = self.self_attention(x, attention_mask)
        x = residual + x
        
        residual = x
        x = self.layer_norm(x)
        x = self.convolution(x)
        x = residual + x
        
        residual = x
        x = self.feed_forward2(x)
        x = residual + x * self.feed_forward_residual_factor
        
        return x


class ConformerPathologicalSpeechModel(BasePathologicalSpeechModel):
    """Conformer model for pathological speech recognition."""
    
    def __init__(self, config: DictConfig):
        """Initialize Conformer model.
        
        Args:
            config: Model configuration.
        """
        super().__init__(config)
        
        # Model parameters
        self.encoder_dim = config.get("encoder_dim", 144)
        self.num_encoder_layers = config.get("num_encoder_layers", 16)
        self.num_attention_heads = config.get("num_attention_heads", 4)
        self.feed_forward_expansion_factor = config.get("feed_forward_expansion_factor", 4)
        self.conv_expansion_factor = config.get("conv_expansion_factor", 2)
        self.conv_kernel_size = config.get("conv_kernel_size", 31)
        self.half_step_residual = config.get("half_step_residual", True)
        
        # Dropout probabilities
        self.input_dropout_p = config.get("input_dropout_p", 0.1)
        self.feed_forward_dropout_p = config.get("feed_forward_dropout_p", 0.1)
        self.attention_dropout_p = config.get("attention_dropout_p", 0.1)
        self.conv_dropout_p = config.get("conv_dropout_p", 0.1)
        
        # Input projection
        self.input_projection = nn.Linear(80, self.encoder_dim)  # Assuming 80-dim features
        self.input_dropout = nn.Dropout(self.input_dropout_p)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(5000, self.encoder_dim) * 0.1)
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                self.encoder_dim,
                self.num_attention_heads,
                self.conv_kernel_size,
                self.feed_forward_expansion_factor,
                self.conv_expansion_factor,
                self.attention_dropout_p,
                self.half_step_residual,
            )
            for _ in range(self.num_encoder_layers)
        ])
        
        # CTC decoder
        self.ctc_decoder = nn.Linear(self.encoder_dim, self.vocab_size)
        
        # Special tokens
        self.blank_id = config.get("blank_id", 0)
        self.sos_id = config.get("sos_id", 1)
        self.eos_id = config.get("eos_id", 2)
        self.pad_id = config.get("pad_id", 3)
    
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            input_values: Input audio features.
            attention_mask: Attention mask for input.
            labels: Target labels for training.
            
        Returns:
            Dict containing model outputs and loss if labels provided.
        """
        # Input projection
        x = self.input_projection(input_values)
        x = self.input_dropout(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        if seq_len <= self.pos_encoding.size(0):
            x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Pass through Conformer blocks
        for block in self.conformer_blocks:
            x = block(x, attention_mask)
        
        # CTC decoder
        logits = self.ctc_decoder(x)
        
        # Compute CTC loss if labels provided
        loss = None
        if labels is not None:
            # Flatten for CTC loss
            logits_flat = logits.view(-1, self.vocab_size)
            labels_flat = labels.view(-1)
            
            # CTC loss
            input_lengths = torch.full(
                (logits.size(0),), logits.size(1), dtype=torch.long
            )
            label_lengths = torch.full(
                (labels.size(0),), labels.size(1), dtype=torch.long
            )
            
            loss = F.ctc_loss(
                logits_flat,
                labels_flat,
                input_lengths,
                label_lengths,
                blank=self.blank_id,
                reduction="mean",
                zero_infinity=True,
            )
        
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": x,
        }
    
    def generate(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        num_beams: int = 1,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> torch.Tensor:
        """Generate predictions from input.
        
        Args:
            input_values: Input audio features.
            attention_mask: Attention mask for input.
            max_length: Maximum sequence length.
            num_beams: Number of beams for beam search.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling.
            
        Returns:
            Generated token sequences.
        """
        with torch.no_grad():
            outputs = self.forward(input_values, attention_mask)
            logits = outputs["logits"]
            
            # Apply temperature if sampling
            if do_sample and temperature != 1.0:
                logits = logits / temperature
            
            # Greedy decoding (CTC)
            predicted_ids = torch.argmax(logits, dim=-1)
            
            return predicted_ids
