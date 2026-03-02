"""Wav2Vec2 model for pathological speech recognition."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from omegaconf import DictConfig

from .base import BasePathologicalSpeechModel


class Wav2Vec2PathologicalSpeechModel(BasePathologicalSpeechModel):
    """Wav2Vec2 model adapted for pathological speech recognition."""
    
    def __init__(self, config: DictConfig):
        """Initialize Wav2Vec2 model.
        
        Args:
            config: Model configuration.
        """
        super().__init__(config)
        
        # Load pre-trained model
        self.pretrained_model_name = config.get("pretrained_model", "facebook/wav2vec2-large-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.pretrained_model_name,
            vocab_size=self.vocab_size,
            ctc_loss_reduction=config.get("ctc_loss_reduction", "mean"),
            pad_token_id=config.get("pad_token_id", 0),
            mask_time_prob=config.get("mask_time_prob", 0.05),
            mask_time_length=config.get("mask_time_length", 10),
            mask_feature_prob=config.get("mask_feature_prob", 0.0),
            mask_feature_length=config.get("mask_feature_length", 10),
            apply_spec_augment=config.get("apply_spec_augment", True),
        )
        
        # Load processor
        self.processor = Wav2Vec2Processor.from_pretrained(self.pretrained_model_name)
        
        # Freeze feature extractor if specified
        if config.get("freeze_feature_extractor", False):
            self.model.freeze_feature_extractor()
    
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
        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return {
            "logits": outputs.logits,
            "loss": outputs.loss if labels is not None else None,
            "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
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
            outputs = self.model(input_values=input_values, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Apply temperature if sampling
            if do_sample and temperature != 1.0:
                logits = logits / temperature
            
            # Greedy decoding (CTC)
            predicted_ids = torch.argmax(logits, dim=-1)
            
            return predicted_ids
    
    def transcribe(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> str:
        """Transcribe audio to text.
        
        Args:
            input_values: Input audio features.
            attention_mask: Attention mask for input.
            
        Returns:
            Transcribed text.
        """
        predicted_ids = self.generate(input_values, attention_mask)
        transcription = self.processor.decode(predicted_ids[0])
        return transcription
    
    def get_feature_extractor(self):
        """Get the feature extractor component."""
        return self.model.wav2vec2.feature_extractor
    
    def get_encoder(self):
        """Get the encoder component."""
        return self.model.wav2vec2.encoder
    
    def get_ctc_head(self):
        """Get the CTC head component."""
        return self.model.lm_head
