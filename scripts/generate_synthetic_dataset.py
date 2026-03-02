#!/usr/bin/env python3
"""Generate synthetic pathological speech dataset for demonstration."""

import os
import json
import pandas as pd
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import List, Dict, Any
import argparse
from tqdm import tqdm

from src.pathological_speech_recognition.data import PathologicalSpeechAugmentation
from src.pathological_speech_recognition.utils.common import set_seed


class SyntheticDatasetGenerator:
    """Generate synthetic pathological speech dataset."""
    
    def __init__(self, output_dir: str, sample_rate: int = 16000):
        """Initialize generator.
        
        Args:
            output_dir: Output directory for generated data.
            sample_rate: Audio sample rate.
        """
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "audio").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        # Common phrases for pathological speech
        self.phrases = [
            "Hello, how are you today?",
            "I am feeling better now.",
            "Can you help me with this?",
            "Thank you very much.",
            "I need to see the doctor.",
            "My name is John Smith.",
            "I live in New York City.",
            "The weather is nice today.",
            "I like to read books.",
            "What time is it now?",
            "I have an appointment tomorrow.",
            "Please speak more slowly.",
            "I don't understand you.",
            "Can you repeat that?",
            "I am having trouble speaking.",
            "My voice sounds different.",
            "I need more time to think.",
            "This is difficult for me.",
            "I am trying my best.",
            "Thank you for your patience."
        ]
        
        # Pathological conditions
        self.conditions = ["parkinson", "stroke", "als", "dysarthria", "normal"]
        
        # Speaker IDs
        self.speaker_ids = [f"speaker_{i:03d}" for i in range(1, 21)]
    
    def generate_sine_wave(self, frequency: float, duration: float, amplitude: float = 0.3) -> torch.Tensor:
        """Generate sine wave audio.
        
        Args:
            frequency: Frequency in Hz.
            duration: Duration in seconds.
            amplitude: Amplitude of the wave.
            
        Returns:
            Generated audio tensor.
        """
        t = torch.linspace(0, duration, int(self.sample_rate * duration))
        waveform = amplitude * torch.sin(2 * np.pi * frequency * t)
        return waveform.unsqueeze(0)
    
    def generate_noise(self, duration: float, noise_type: str = "white") -> torch.Tensor:
        """Generate noise audio.
        
        Args:
            duration: Duration in seconds.
            noise_type: Type of noise.
            
        Returns:
            Generated noise tensor.
        """
        length = int(self.sample_rate * duration)
        
        if noise_type == "white":
            noise = torch.randn(1, length) * 0.1
        elif noise_type == "pink":
            # Simple pink noise approximation
            noise = torch.randn(1, length) * 0.1
            # Apply simple filtering
            for i in range(1, length):
                noise[0, i] = 0.9 * noise[0, i-1] + noise[0, i]
        else:
            noise = torch.randn(1, length) * 0.1
        
        return noise
    
    def generate_speech_like_audio(self, duration: float, condition: str) -> torch.Tensor:
        """Generate speech-like audio with pathological characteristics.
        
        Args:
            duration: Duration in seconds.
            condition: Pathological condition.
            
        Returns:
            Generated audio tensor.
        """
        # Base speech-like signal (combination of sine waves)
        base_freqs = [200, 400, 600, 800]  # Formant-like frequencies
        waveform = torch.zeros(1, int(self.sample_rate * duration))
        
        for freq in base_freqs:
            sine_wave = self.generate_sine_wave(freq, duration, amplitude=0.1)
            waveform += sine_wave
        
        # Add modulation for speech-like characteristics
        modulation_freq = 5  # Hz
        t = torch.linspace(0, duration, waveform.shape[1])
        modulation = 1 + 0.3 * torch.sin(2 * np.pi * modulation_freq * t)
        waveform = waveform * modulation
        
        # Apply condition-specific modifications
        if condition == "parkinson":
            # Add tremor
            tremor_freq = 4  # Hz
            tremor = 1 + 0.1 * torch.sin(2 * np.pi * tremor_freq * t)
            waveform = waveform * tremor
            
        elif condition == "stroke":
            # Add irregular patterns
            irregular = 1 + 0.2 * torch.randn(waveform.shape[1])
            waveform = waveform * irregular
            
        elif condition == "als":
            # Reduce amplitude and add noise
            waveform = waveform * 0.7
            noise = self.generate_noise(duration, "white")
            waveform = waveform + noise * 0.3
            
        elif condition == "dysarthria":
            # Add slurring effect (low-pass filtering simulation)
            waveform = waveform * 0.8
        
        # Normalize
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        return waveform
    
    def generate_dataset(self, num_samples: int = 1000, train_split: float = 0.7, val_split: float = 0.15) -> None:
        """Generate the complete dataset.
        
        Args:
            num_samples: Total number of samples to generate.
            train_split: Fraction for training set.
            val_split: Fraction for validation set.
        """
        set_seed(42)
        
        # Calculate splits
        train_samples = int(num_samples * train_split)
        val_samples = int(num_samples * val_split)
        test_samples = num_samples - train_samples - val_samples
        
        splits = ["train"] * train_samples + ["val"] * val_samples + ["test"] * test_samples
        np.random.shuffle(splits)
        
        metadata = []
        
        print(f"Generating {num_samples} samples...")
        
        for i in tqdm(range(num_samples)):
            # Randomly select phrase, condition, and speaker
            phrase = np.random.choice(self.phrases)
            condition = np.random.choice(self.conditions)
            speaker_id = np.random.choice(self.speaker_ids)
            split = splits[i]
            
            # Generate duration (1-5 seconds)
            duration = np.random.uniform(1.0, 5.0)
            
            # Generate audio
            audio = self.generate_speech_like_audio(duration, condition)
            
            # Save audio file
            audio_filename = f"sample_{i:06d}.wav"
            audio_path = self.output_dir / "audio" / audio_filename
            
            torchaudio.save(
                str(audio_path),
                audio,
                self.sample_rate,
                format="wav"
            )
            
            # Add to metadata
            metadata.append({
                "id": f"sample_{i:06d}",
                "path": str(audio_path.relative_to(self.output_dir)),
                "text": phrase,
                "speaker_id": speaker_id,
                "condition": condition,
                "duration": duration,
                "split": split,
                "sample_rate": self.sample_rate
            })
        
        # Save metadata
        df = pd.DataFrame(metadata)
        df.to_csv(self.output_dir / "metadata" / "metadata.csv", index=False)
        
        # Save dataset info
        dataset_info = {
            "name": "synthetic_pathological_speech",
            "description": "Synthetic dataset for pathological speech recognition",
            "total_samples": num_samples,
            "train_samples": train_samples,
            "val_samples": val_samples,
            "test_samples": test_samples,
            "sample_rate": self.sample_rate,
            "conditions": self.conditions,
            "num_speakers": len(self.speaker_ids),
            "num_phrases": len(self.phrases),
            "generated_by": "SyntheticDatasetGenerator",
            "note": "This is a synthetic dataset for demonstration purposes only"
        }
        
        with open(self.output_dir / "metadata" / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"Dataset generated successfully!")
        print(f"Output directory: {self.output_dir}")
        print(f"Train samples: {train_samples}")
        print(f"Validation samples: {val_samples}")
        print(f"Test samples: {test_samples}")
        
        # Print condition distribution
        condition_counts = df["condition"].value_counts()
        print("\nCondition distribution:")
        for condition, count in condition_counts.items():
            print(f"  {condition}: {count}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate synthetic pathological speech dataset")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--train-split", type=float, default=0.7, help="Training split fraction")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split fraction")
    
    args = parser.parse_args()
    
    # Generate dataset
    generator = SyntheticDatasetGenerator(args.output_dir, args.sample_rate)
    generator.generate_dataset(
        num_samples=args.num_samples,
        train_split=args.train_split,
        val_split=args.val_split
    )


if __name__ == "__main__":
    main()
