#!/usr/bin/env python3
"""Quick start script to test the pathological speech recognition system."""

import os
import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pathological_speech_recognition.models import ConformerPathologicalSpeechModel
from pathological_speech_recognition.data import AudioPreprocessor, PathologicalSpeechAugmentation
from pathological_speech_recognition.metrics import PathologicalSpeechMetrics
from pathological_speech_recognition.utils.common import get_device, set_seed
from omegaconf import DictConfig


def test_basic_functionality():
    """Test basic functionality of the system."""
    print("🧪 Testing Pathological Speech Recognition System")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Test 1: Model initialization
    print("\n1. Testing model initialization...")
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
    device = get_device()
    model = model.to(device)
    print(f"✅ Conformer model initialized on {device}")
    
    # Test 2: Audio preprocessing
    print("\n2. Testing audio preprocessing...")
    preprocessor_config = DictConfig({
        "sample_rate": 16000,
        "normalize": True,
        "preemphasis": 0.97,
        "features": {
            "feature_type": "log_mel",
            "n_fft": 512,
            "hop_length": 160,
            "n_mels": 80
        }
    })
    
    preprocessor = AudioPreprocessor(preprocessor_config)
    
    # Generate dummy audio
    duration = 2.0  # seconds
    sample_rate = 16000
    waveform = torch.randn(1, int(sample_rate * duration))
    
    processed_audio = preprocessor(waveform, sample_rate)
    print(f"✅ Audio preprocessing successful: {processed_audio.shape}")
    
    # Test 3: Pathological speech augmentation
    print("\n3. Testing pathological speech augmentation...")
    augmentation_config = DictConfig({
        "sample_rate": 16000,
        "tremor_simulation_prob": 0.5,
        "slur_simulation_prob": 0.3,
        "volume_reduction_prob": 0.4,
        "tremor_freq_range": [3, 8],
        "slur_factor_range": [0.8, 1.2]
    })
    
    augmentation = PathologicalSpeechAugmentation(augmentation_config)
    augmented_audio = augmentation(waveform)
    print(f"✅ Pathological speech augmentation successful: {augmented_audio.shape}")
    
    # Test 4: Model inference
    print("\n4. Testing model inference...")
    model.eval()
    
    # Prepare input (log-mel features for Conformer)
    input_features = preprocessor.extract_features(waveform)
    input_features = input_features.to(device)
    
    with torch.no_grad():
        outputs = model(input_features)
        logits = outputs["logits"]
        print(f"✅ Model inference successful: {logits.shape}")
    
    # Test 5: Metrics
    print("\n5. Testing evaluation metrics...")
    metrics = PathologicalSpeechMetrics(vocab_size=32)
    
    references = ["hello world", "how are you"]
    hypotheses = ["hello world", "how are you"]
    conditions = ["parkinson", "stroke"]
    
    metrics.update(references, hypotheses, conditions)
    final_metrics = metrics.compute()
    
    print("✅ Metrics computed successfully:")
    for metric, value in final_metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # Test 6: Model information
    print("\n6. Testing model information...")
    model_info = model.get_model_info()
    print(f"✅ Model info retrieved:")
    print(f"   Model type: {model_info['model_type']}")
    print(f"   Total parameters: {model_info['total_parameters']:,}")
    print(f"   Trainable parameters: {model_info['trainable_parameters']:,}")
    print(f"   Model size: {model_info['model_size_mb']:.1f} MB")
    
    print("\n🎉 All tests passed! The system is working correctly.")
    return True


def generate_sample_data():
    """Generate sample data for testing."""
    print("\n📊 Generating sample data...")
    
    # Create data directory
    data_dir = Path("data/sample")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a few sample audio files
    sample_rate = 16000
    duration = 3.0
    
    for i in range(3):
        # Generate different types of audio
        if i == 0:
            # Normal speech-like
            waveform = torch.randn(1, int(sample_rate * duration)) * 0.3
        elif i == 1:
            # Tremor-like (Parkinson's)
            t = torch.linspace(0, duration, int(sample_rate * duration))
            tremor = 1 + 0.1 * torch.sin(2 * np.pi * 4 * t)  # 4 Hz tremor
            waveform = torch.randn(1, int(sample_rate * duration)) * 0.3 * tremor
        else:
            # Reduced volume (ALS-like)
            waveform = torch.randn(1, int(sample_rate * duration)) * 0.1
        
        # Save audio file
        audio_path = data_dir / f"sample_{i:02d}.wav"
        torchaudio.save(str(audio_path), waveform, sample_rate)
        print(f"   Generated: {audio_path}")
    
    print(f"✅ Sample data generated in {data_dir}")


def main():
    """Main function."""
    print("🚀 Pathological Speech Recognition - Quick Start")
    print("=" * 60)
    
    try:
        # Test basic functionality
        success = test_basic_functionality()
        
        if success:
            # Generate sample data
            generate_sample_data()
            
            print("\n" + "=" * 60)
            print("🎯 Next Steps:")
            print("1. Run the Streamlit demo: streamlit run demo/streamlit_demo.py")
            print("2. Generate synthetic dataset: python scripts/generate_synthetic_dataset.py")
            print("3. Run tests: pytest tests/")
            print("4. Check the README.md for detailed usage instructions")
            print("\n⚠️  Remember: This is for research and educational purposes only!")
            
        else:
            print("❌ Some tests failed. Please check the error messages above.")
            return 1
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
