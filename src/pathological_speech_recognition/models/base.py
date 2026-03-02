"""Base model class for pathological speech recognition."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from omegaconf import DictConfig


class BasePathologicalSpeechModel(nn.Module, ABC):
    """Base class for pathological speech recognition models."""
    
    def __init__(self, config: DictConfig):
        """Initialize base model.
        
        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        self.vocab_size = config.get("vocab_size", 32)
        
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dict containing model information.
        """
        from ..utils.common import get_model_size
        
        info = get_model_size(self)
        info.update({
            "model_type": self.__class__.__name__,
            "vocab_size": self.vocab_size,
            "config": self.config,
        })
        
        return info
    
    def save_model(self, path: str) -> None:
        """Save model to file.
        
        Args:
            path: Path to save the model.
        """
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "model_info": self.get_model_info(),
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: Optional[str] = None) -> "BasePathologicalSpeechModel":
        """Load model from file.
        
        Args:
            path: Path to load the model from.
            device: Device to load the model on.
            
        Returns:
            Loaded model instance.
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if device:
            model = model.to(device)
            
        return model
