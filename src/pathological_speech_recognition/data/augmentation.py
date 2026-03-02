"""Data processing and augmentation for pathological speech recognition."""

import torch
import torchaudio
import librosa
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from omegaconf import DictConfig


@dataclass
class AudioSample:
    """Audio sample data structure."""
    waveform: torch.Tensor
    sample_rate: int
    duration: float
    text: str
    speaker_id: Optional[str] = None
    condition: Optional[str] = None  # e.g., "parkinson", "stroke", "als"
    metadata: Optional[Dict] = None


class PathologicalSpeechAugmentation:
    """Augmentation techniques specific to pathological speech."""
    
    def __init__(self, config: DictConfig):
        """Initialize augmentation.
        
        Args:
            config: Augmentation configuration.
        """
        self.config = config
        self.sample_rate = config.get("sample_rate", 16000)
        
        # Augmentation probabilities
        self.speed_perturb_prob = config.get("speed_perturb_prob", 0.5)
        self.pitch_shift_prob = config.get("pitch_shift_prob", 0.3)
        self.add_noise_prob = config.get("add_noise_prob", 0.4)
        self.add_reverb_prob = config.get("add_reverb_prob", 0.2)
        self.time_stretch_prob = config.get("time_stretch_prob", 0.3)
        self.volume_scale_prob = config.get("volume_scale_prob", 0.4)
        
        # Pathological speech specific
        self.tremor_simulation_prob = config.get("tremor_simulation_prob", 0.3)
        self.slur_simulation_prob = config.get("slur_simulation_prob", 0.2)
        self.volume_reduction_prob = config.get("volume_reduction_prob", 0.3)
        
        # Parameters
        self.speed_range = config.get("speed_range", [0.9, 1.1])
        self.pitch_range = config.get("pitch_range", [-2, 2])
        self.noise_snr_range = config.get("noise_snr_range", [10, 30])
        self.volume_range = config.get("volume_range", [0.7, 1.3])
        self.tremor_freq_range = config.get("tremor_freq_range", [3, 8])
        self.slur_factor_range = config.get("slur_factor_range", [0.8, 1.2])
    
    def speed_perturbation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply speed perturbation.
        
        Args:
            waveform: Input waveform.
            
        Returns:
            Augmented waveform.
        """
        if torch.rand(1) > self.speed_perturb_prob:
            return waveform
        
        speed_factor = torch.uniform(self.speed_range[0], self.speed_range[1])
        
        # Resample to change speed
        new_length = int(waveform.size(-1) / speed_factor)
        waveform_resampled = torchaudio.functional.resample(
            waveform, self.sample_rate, int(self.sample_rate * speed_factor)
        )
        
        # Truncate or pad to original length
        if waveform_resampled.size(-1) > waveform.size(-1):
            waveform_resampled = waveform_resampled[..., :waveform.size(-1)]
        else:
            pad_length = waveform.size(-1) - waveform_resampled.size(-1)
            waveform_resampled = torch.nn.functional.pad(
                waveform_resampled, (0, pad_length)
            )
        
        return waveform_resampled
    
    def pitch_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply pitch shifting.
        
        Args:
            waveform: Input waveform.
            
        Returns:
            Augmented waveform.
        """
        if torch.rand(1) > self.pitch_shift_prob:
            return waveform
        
        pitch_shift = torch.uniform(self.pitch_range[0], self.pitch_range[1])
        
        # Convert to numpy for librosa
        waveform_np = waveform.squeeze().numpy()
        
        # Apply pitch shift
        waveform_shifted = librosa.effects.pitch_shift(
            waveform_np, sr=self.sample_rate, n_steps=pitch_shift
        )
        
        return torch.from_numpy(waveform_shifted).unsqueeze(0)
    
    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add background noise.
        
        Args:
            waveform: Input waveform.
            
        Returns:
            Augmented waveform.
        """
        if torch.rand(1) > self.add_noise_prob:
            return waveform
        
        snr_db = torch.uniform(self.noise_snr_range[0], self.noise_snr_range[1])
        
        # Generate white noise
        noise = torch.randn_like(waveform)
        
        # Calculate noise power
        signal_power = torch.mean(waveform ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Scale noise
        noise = noise * torch.sqrt(noise_power)
        
        return waveform + noise
    
    def add_reverb(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add reverberation.
        
        Args:
            waveform: Input waveform.
            
        Returns:
            Augmented waveform.
        """
        if torch.rand(1) > self.add_reverb_prob:
            return waveform
        
        # Simple reverb simulation using convolution
        reverb_length = int(self.sample_rate * 0.5)  # 0.5 second reverb
        reverb = torch.exp(-torch.linspace(0, 5, reverb_length))
        reverb = reverb * torch.randn(reverb_length) * 0.1
        
        # Apply convolution
        waveform_reverb = torch.nn.functional.conv1d(
            waveform.unsqueeze(0),
            reverb.unsqueeze(0).unsqueeze(0),
            padding=reverb_length // 2
        ).squeeze(0)
        
        return waveform_reverb
    
    def time_stretch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply time stretching.
        
        Args:
            waveform: Input waveform.
            
        Returns:
            Augmented waveform.
        """
        if torch.rand(1) > self.time_stretch_prob:
            return waveform
        
        stretch_factor = torch.uniform(0.9, 1.1)
        
        # Convert to numpy for librosa
        waveform_np = waveform.squeeze().numpy()
        
        # Apply time stretch
        waveform_stretched = librosa.effects.time_stretch(
            waveform_np, rate=stretch_factor
        )
        
        # Resize to original length
        if len(waveform_stretched) > waveform.size(-1):
            waveform_stretched = waveform_stretched[:waveform.size(-1)]
        else:
            pad_length = waveform.size(-1) - len(waveform_stretched)
            waveform_stretched = np.pad(waveform_stretched, (0, pad_length))
        
        return torch.from_numpy(waveform_stretched).unsqueeze(0)
    
    def volume_scale(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply volume scaling.
        
        Args:
            waveform: Input waveform.
            
        Returns:
            Augmented waveform.
        """
        if torch.rand(1) > self.volume_scale_prob:
            return waveform
        
        scale_factor = torch.uniform(self.volume_range[0], self.volume_range[1])
        return waveform * scale_factor
    
    def tremor_simulation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Simulate tremor in speech.
        
        Args:
            waveform: Input waveform.
            
        Returns:
            Augmented waveform.
        """
        if torch.rand(1) > self.tremor_simulation_prob:
            return waveform
        
        # Generate tremor frequency
        tremor_freq = torch.uniform(self.tremor_freq_range[0], self.tremor_freq_range[1])
        
        # Create tremor modulation
        t = torch.linspace(0, waveform.size(-1) / self.sample_rate, waveform.size(-1))
        tremor = 1 + 0.1 * torch.sin(2 * np.pi * tremor_freq * t)
        
        # Apply tremor
        waveform_tremor = waveform * tremor.unsqueeze(0)
        
        return waveform_tremor
    
    def slur_simulation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Simulate slurred speech.
        
        Args:
            waveform: Input waveform.
            
        Returns:
            Augmented waveform.
        """
        if torch.rand(1) > self.slur_simulation_prob:
            return waveform
        
        # Apply low-pass filtering to simulate slurred speech
        slur_factor = torch.uniform(self.slur_factor_range[0], self.slur_factor_range[1])
        cutoff_freq = int(8000 * slur_factor)
        
        # Simple low-pass filter
        waveform_np = waveform.squeeze().numpy()
        waveform_slurred = librosa.effects.preemphasis(waveform_np, coef=0.97)
        
        return torch.from_numpy(waveform_slurred).unsqueeze(0)
    
    def volume_reduction(self, waveform: torch.Tensor) -> torch.Tensor:
        """Simulate volume reduction common in pathological speech.
        
        Args:
            waveform: Input waveform.
            
        Returns:
            Augmented waveform.
        """
        if torch.rand(1) > self.volume_reduction_prob:
            return waveform
        
        # Reduce volume by 20-40%
        reduction_factor = torch.uniform(0.6, 0.8)
        return waveform * reduction_factor
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply all augmentations.
        
        Args:
            waveform: Input waveform.
            
        Returns:
            Augmented waveform.
        """
        # Apply standard augmentations
        waveform = self.speed_perturbation(waveform)
        waveform = self.pitch_shift(waveform)
        waveform = self.add_noise(waveform)
        waveform = self.add_reverb(waveform)
        waveform = self.time_stretch(waveform)
        waveform = self.volume_scale(waveform)
        
        # Apply pathological speech specific augmentations
        waveform = self.tremor_simulation(waveform)
        waveform = self.slur_simulation(waveform)
        waveform = self.volume_reduction(waveform)
        
        return waveform


class AudioPreprocessor:
    """Audio preprocessing for pathological speech recognition."""
    
    def __init__(self, config: DictConfig):
        """Initialize preprocessor.
        
        Args:
            config: Preprocessing configuration.
        """
        self.config = config
        self.sample_rate = config.get("sample_rate", 16000)
        self.normalize = config.get("normalize", True)
        self.preemphasis = config.get("preemphasis", 0.97)
        
        # Feature extraction parameters
        self.feature_config = config.get("features", {})
        self.feature_type = self.feature_config.get("feature_type", "wav2vec2")
        
        if self.feature_type == "log_mel":
            self.n_fft = self.feature_config.get("n_fft", 512)
            self.hop_length = self.feature_config.get("hop_length", 160)
            self.win_length = self.feature_config.get("win_length", 400)
            self.n_mels = self.feature_config.get("n_mels", 80)
            self.f_min = self.feature_config.get("f_min", 0)
            self.f_max = self.feature_config.get("f_max", 8000)
    
    def preprocess(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Preprocess audio waveform.
        
        Args:
            waveform: Input waveform.
            sample_rate: Input sample rate.
            
        Returns:
            Preprocessed waveform.
        """
        # Resample if necessary
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.sample_rate
            )
        
        # Pre-emphasis
        if self.preemphasis > 0:
            waveform = torchaudio.functional.preemphasis(waveform, self.preemphasis)
        
        # Normalize
        if self.normalize:
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        return waveform
    
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features from waveform.
        
        Args:
            waveform: Input waveform.
            
        Returns:
            Extracted features.
        """
        if self.feature_type == "wav2vec2":
            # For Wav2Vec2, return raw waveform
            return waveform
        
        elif self.feature_type == "log_mel":
            # Extract log-mel spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                f_min=self.f_min,
                f_max=self.f_max,
            )
            
            mel_spec = mel_transform(waveform)
            log_mel = torch.log(mel_spec + 1e-8)
            
            return log_mel
        
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
    
    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Preprocess and extract features.
        
        Args:
            waveform: Input waveform.
            sample_rate: Input sample rate.
            
        Returns:
            Processed features.
        """
        waveform = self.preprocess(waveform, sample_rate)
        features = self.extract_features(waveform)
        return features
