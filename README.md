# Pathological Speech Recognition

Research-focused system for automatic speech recognition of pathological speech, including speech affected by conditions such as Parkinson's disease, stroke, or ALS.

## ⚠️ PRIVACY DISCLAIMER

**This is a research demonstration tool only.** This system is designed for educational and research purposes to help understand pathological speech recognition challenges. 

- Audio data is processed locally and not stored or transmitted
- This system is NOT intended for biometric identification
- Do NOT use in production systems without proper privacy safeguards
- Misuse for voice cloning or unauthorized biometric identification is prohibited

## Features

- **Multiple Model Architectures**: Wav2Vec2 and Conformer models optimized for pathological speech
- **Pathological Speech Augmentation**: Specialized data augmentation techniques including tremor simulation, slur simulation, and volume reduction
- **Comprehensive Evaluation**: WER/CER metrics plus pathological speech-specific metrics (intelligibility, fluency, articulation)
- **Interactive Demo**: Streamlit-based web interface for real-time testing
- **Privacy-First Design**: Local processing with anonymization options
- **Modern Architecture**: PyTorch 2.x, Python 3.10+, with proper type hints and documentation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Pathological-Speech-Recognition.git
cd Pathological-Speech-Recognition

# Install dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from pathological_speech_recognition import Wav2Vec2PathologicalSpeechModel
from pathological_speech_recognition.data import AudioPreprocessor
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("configs/config.yaml")

# Initialize model
model = Wav2Vec2PathologicalSpeechModel(config.model)

# Load and preprocess audio
preprocessor = AudioPreprocessor(config.data)
audio_features = preprocessor(audio_waveform, sample_rate)

# Transcribe
transcription = model.transcribe(audio_features)
print(f"Transcription: {transcription}")
```

### Running the Demo

```bash
# Start the Streamlit demo
streamlit run demo/streamlit_demo.py
```

## Project Structure

```
pathological-speech-recognition/
├── src/pathological_speech_recognition/
│   ├── models/              # Model implementations
│   │   ├── base.py         # Base model class
│   │   ├── wav2vec2.py     # Wav2Vec2 implementation
│   │   └── conformer.py     # Conformer implementation
│   ├── data/               # Data processing
│   │   └── augmentation.py # Pathological speech augmentation
│   ├── metrics/            # Evaluation metrics
│   │   └── pathological_metrics.py
│   ├── utils/              # Utility functions
│   │   └── common.py       # Common utilities
│   └── __init__.py
├── configs/                 # Configuration files
│   ├── config.yaml         # Main configuration
│   └── conformer.yaml      # Conformer-specific config
├── demo/                   # Demo applications
│   └── streamlit_demo.py   # Streamlit demo
├── tests/                  # Test suite
├── data/                   # Data directory
├── assets/                 # Output artifacts
├── scripts/                # Training/evaluation scripts
├── notebooks/              # Jupyter notebooks
├── pyproject.toml          # Project configuration
└── README.md
```

## Configuration

The system uses Hydra/OmegaConf for configuration management. Key configuration sections:

### Model Configuration
- Model architecture selection (Wav2Vec2, Conformer)
- Pre-trained model paths
- Model-specific hyperparameters

### Data Configuration
- Audio preprocessing parameters
- Augmentation settings
- Pathological speech-specific augmentations

### Training Configuration
- Optimizer settings
- Learning rate schedules
- Early stopping criteria

## Models

### Wav2Vec2 Model
- Pre-trained on large-scale speech data
- Fine-tuned for pathological speech
- CTC-based decoding
- Robust to various speech impairments

### Conformer Model
- State-of-the-art architecture for speech recognition
- Self-attention + convolution modules
- Optimized for pathological speech characteristics
- Better handling of temporal dependencies

## Data Augmentation

Specialized augmentation techniques for pathological speech:

- **Tremor Simulation**: Modulates audio with tremor-like patterns
- **Slur Simulation**: Applies filtering to simulate slurred speech
- **Volume Reduction**: Simulates reduced volume common in pathological speech
- **Standard Augmentations**: Speed perturbation, pitch shifting, noise addition, reverb

## Evaluation Metrics

### Standard Metrics
- **WER (Word Error Rate)**: Primary metric for speech recognition
- **CER (Character Error Rate)**: Character-level accuracy

### Pathological Speech Specific Metrics
- **Intelligibility Score**: Measures speech clarity and comprehensibility
- **Fluency Score**: Evaluates speech flow and rhythm
- **Articulation Score**: Assesses pronunciation accuracy

### Confidence Calibration
- **Expected Calibration Error (ECE)**: Measures prediction confidence reliability
- **Reliability Diagrams**: Visualize confidence vs. accuracy relationship

## Training

### Data Preparation
1. Organize audio files in `data/raw/` directory
2. Create metadata CSV with columns: `id`, `path`, `text`, `speaker_id`, `condition`
3. Update configuration with dataset paths

### Training Commands
```bash
# Train Wav2Vec2 model
python scripts/train.py --config-name config

# Train Conformer model
python scripts/train.py --config-name conformer

# Resume training from checkpoint
python scripts/train.py --config-name config checkpoint=path/to/checkpoint.pt
```

### Evaluation
```bash
# Evaluate on test set
python scripts/evaluate.py --config-name config checkpoint=path/to/model.pt

# Generate detailed metrics report
python scripts/evaluate.py --config-name config checkpoint=path/to/model.pt --detailed
```

## API Usage

### FastAPI Server
```bash
# Start API server
python scripts/serve.py --config-name config checkpoint=path/to/model.pt

# API endpoints:
# POST /transcribe - Transcribe audio file
# GET /health - Health check
# GET /model_info - Model information
```

### Python API
```python
from pathological_speech_recognition import PathologicalSpeechRecognizer

# Initialize recognizer
recognizer = PathologicalSpeechRecognizer.from_pretrained("path/to/model")

# Transcribe audio
result = recognizer.transcribe("path/to/audio.wav")
print(f"Text: {result.text}")
print(f"Confidence: {result.confidence}")
print(f"Intelligibility: {result.intelligibility_score}")
```

## Dataset Schema

### Required Format
```csv
id,path,text,speaker_id,condition,split
sample_001,audio/sample_001.wav,"Hello world",speaker_01,parkinson,train
sample_002,audio/sample_002.wav,"How are you",speaker_02,stroke,val
```

### Optional Fields
- `duration`: Audio duration in seconds
- `age`: Speaker age
- `gender`: Speaker gender
- `severity`: Condition severity level
- `metadata`: JSON string with additional information

## Limitations and Considerations

### Current Limitations
- Limited to English language
- Requires sufficient training data for each pathological condition
- Performance may vary significantly across different speech impairments
- Real-time processing capabilities depend on hardware

### Ethical Considerations
- This system is for research and educational purposes only
- Do not use for unauthorized biometric identification
- Respect privacy and obtain proper consent for any data collection
- Consider bias and fairness implications in model deployment

### Technical Considerations
- Model performance depends on training data quality and diversity
- Pathological speech characteristics vary significantly between individuals
- Regular model updates may be necessary as new data becomes available
- Consider domain adaptation for different recording conditions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
ruff check src/ tests/

# Run pre-commit hooks
pre-commit run --all-files
```

## Citation

If you use this work in your research, please cite:

```bibtex
@software{pathological_speech_recognition,
  title={Pathological Speech Recognition},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Pathological-Speech-Recognition}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face Transformers for Wav2Vec2 implementation
- PyTorch team for the deep learning framework
- The speech recognition research community
- Contributors to pathological speech datasets

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review the example notebooks

---

**Remember**: This is a research tool. Always prioritize privacy, ethics, and responsible use of speech recognition technology.
# Pathological-Speech-Recognition
