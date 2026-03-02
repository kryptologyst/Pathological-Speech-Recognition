"""Tests for pathological speech recognition package."""

import pytest
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.pathological_speech_recognition.models import Wav2Vec2PathologicalSpeechModel, ConformerPathologicalSpeechModel
from src.pathological_speech_recognition.data import AudioPreprocessor, PathologicalSpeechAugmentation
from src.pathological_speech_recognition.metrics import PathologicalSpeechMetrics
from src.pathological_speech_recognition.utils.common import get_device, set_seed


class TestModels:
    """Test model implementations."""
    
    def test_wav2vec2_model_initialization(self):
        """Test Wav2Vec2 model initialization."""
        config = DictConfig({
            "pretrained_model": "facebook/wav2vec2-base-960h",
            "vocab_size": 32,
            "freeze_feature_extractor": False,
            "ctc_loss_reduction": "mean"
        })
        
        model = Wav2Vec2PathologicalSpeechModel(config)
        assert model is not None
        assert model.vocab_size == 32
    
    def test_conformer_model_initialization(self):
        """Test Conformer model initialization."""
        config = DictConfig({
            "encoder_dim": 144,
            "num_encoder_layers": 4,
            "num_attention_heads": 4,
            "vocab_size": 32,
            "blank_id": 0,
            "sos_id": 1,
            "eos_id": 2,
            "pad_id": 3
        })
        
        model = ConformerPathologicalSpeechModel(config)
        assert model is not None
        assert model.vocab_size == 32
        assert model.encoder_dim == 144
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        config = DictConfig({
            "encoder_dim": 144,
            "num_encoder_layers": 2,
            "num_attention_heads": 4,
            "vocab_size": 32,
            "blank_id": 0,
            "sos_id": 1,
            "eos_id": 2,
            "pad_id": 3
        })
        
        model = ConformerPathologicalSpeechModel(config)
        
        # Create dummy input
        batch_size = 2
        seq_len = 100
        input_dim = 80  # log-mel features
        
        input_values = torch.randn(batch_size, seq_len, input_dim)
        labels = torch.randint(0, 32, (batch_size, 20))  # Dummy labels
        
        # Forward pass
        outputs = model(input_values=input_values, labels=labels)
        
        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, 32)
        assert outputs["loss"] is not None


class TestDataProcessing:
    """Test data processing components."""
    
    def test_audio_preprocessor(self):
        """Test audio preprocessor."""
        config = DictConfig({
            "sample_rate": 16000,
            "normalize": True,
            "preemphasis": 0.97,
            "features": {
                "feature_type": "wav2vec2",
                "n_fft": 512,
                "hop_length": 160,
                "n_mels": 80
            }
        })
        
        preprocessor = AudioPreprocessor(config)
        
        # Create dummy audio
        waveform = torch.randn(1, 16000)  # 1 second of audio
        sample_rate = 16000
        
        # Process audio
        processed = preprocessor(waveform, sample_rate)
        
        assert processed is not None
        assert isinstance(processed, torch.Tensor)
    
    def test_augmentation(self):
        """Test pathological speech augmentation."""
        config = DictConfig({
            "sample_rate": 16000,
            "speed_perturb_prob": 0.5,
            "pitch_shift_prob": 0.3,
            "add_noise_prob": 0.4,
            "tremor_simulation_prob": 0.3,
            "slur_simulation_prob": 0.2,
            "volume_reduction_prob": 0.3,
            "speed_range": [0.9, 1.1],
            "pitch_range": [-2, 2],
            "noise_snr_range": [10, 30],
            "volume_range": [0.7, 1.3]
        })
        
        augmentation = PathologicalSpeechAugmentation(config)
        
        # Create dummy audio
        waveform = torch.randn(1, 16000)
        
        # Apply augmentation
        augmented = augmentation(waveform)
        
        assert augmented is not None
        assert isinstance(augmented, torch.Tensor)
        assert augmented.shape == waveform.shape


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_pathological_speech_metrics(self):
        """Test pathological speech metrics."""
        metrics = PathologicalSpeechMetrics(vocab_size=32)
        
        # Test basic metrics
        wer = metrics.compute_wer("hello world", "hello world")
        assert wer == 0.0
        
        wer = metrics.compute_wer("hello world", "hello there")
        assert wer == 0.5  # 1 word error out of 2 words
        
        cer = metrics.compute_cer("hello", "hello")
        assert cer == 0.0
        
        cer = metrics.compute_cer("hello", "helo")
        assert cer == 0.2  # 1 char error out of 5 chars
    
    def test_metrics_update(self):
        """Test metrics update functionality."""
        metrics = PathologicalSpeechMetrics(vocab_size=32)
        
        references = ["hello world", "how are you"]
        hypotheses = ["hello world", "how are you"]
        conditions = ["parkinson", "stroke"]
        
        metrics.update(references, hypotheses, conditions)
        
        final_metrics = metrics.compute()
        
        assert "wer" in final_metrics
        assert "cer" in final_metrics
        assert "intelligibility" in final_metrics
        assert "fluency" in final_metrics
        assert "articulation" in final_metrics


class TestUtils:
    """Test utility functions."""
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("cpu")
        assert device.type == "cpu"
        
        device = get_device("auto")
        assert device is not None
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        torch_rand = torch.rand(1)
        np_rand = np.random.rand(1)
        
        # Reset seed and generate again
        set_seed(42)
        torch_rand2 = torch.rand(1)
        np_rand2 = np.random.rand(1)
        
        # Should be the same
        assert torch.allclose(torch_rand, torch_rand2)
        assert np.allclose(np_rand, np_rand2)


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_conformer(self):
        """Test end-to-end Conformer pipeline."""
        # Set seed for reproducibility
        set_seed(42)
        
        # Model config
        config = DictConfig({
            "encoder_dim": 144,
            "num_encoder_layers": 2,
            "num_attention_heads": 4,
            "vocab_size": 32,
            "blank_id": 0,
            "sos_id": 1,
            "eos_id": 2,
            "pad_id": 3
        })
        
        # Initialize model
        model = ConformerPathologicalSpeechModel(config)
        model.eval()
        
        # Create dummy input
        batch_size = 1
        seq_len = 100
        input_dim = 80
        
        input_values = torch.randn(batch_size, seq_len, input_dim)
        
        # Generate prediction
        with torch.no_grad():
            predicted_ids = model.generate(input_values)
        
        assert predicted_ids is not None
        assert predicted_ids.shape[0] == batch_size
        assert predicted_ids.shape[1] == seq_len
    
    def test_model_info(self):
        """Test model information retrieval."""
        config = DictConfig({
            "encoder_dim": 144,
            "num_encoder_layers": 2,
            "num_attention_heads": 4,
            "vocab_size": 32,
            "blank_id": 0,
            "sos_id": 1,
            "eos_id": 2,
            "pad_id": 3
        })
        
        model = ConformerPathologicalSpeechModel(config)
        info = model.get_model_info()
        
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "model_size_mb" in info
        assert "model_type" in info
        assert "vocab_size" in info
        
        assert info["model_type"] == "ConformerPathologicalSpeechModel"
        assert info["vocab_size"] == 32


if __name__ == "__main__":
    pytest.main([__file__])
