"""Streamlit demo for pathological speech recognition."""

import streamlit as st
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any
import io
import base64
import logging

from ..models import Wav2Vec2PathologicalSpeechModel, ConformerPathologicalSpeechModel
from ..data import AudioPreprocessor, PathologicalSpeechAugmentation
from ..metrics import PathologicalSpeechMetrics
from ..utils.common import get_device, load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Pathological Speech Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Privacy disclaimer
st.markdown("""
<div style="background-color: #ffebee; padding: 15px; border-radius: 5px; border-left: 5px solid #f44336;">
    <h4 style="color: #c62828; margin-top: 0;">⚠️ PRIVACY DISCLAIMER</h4>
    <p style="margin-bottom: 0;">
        <strong>This is a research demonstration tool only.</strong> 
        Audio data is processed locally and not stored or transmitted. 
        This system is designed for educational and research purposes to help understand 
        pathological speech recognition challenges. 
        <strong>Do not use for biometric identification or production systems.</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Title and description
st.title("🎤 Pathological Speech Recognition Demo")
st.markdown("""
This demo showcases automatic speech recognition capabilities for pathological speech, 
including speech affected by conditions such as Parkinson's disease, stroke, or ALS.
""")

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model",
    ["Wav2Vec2", "Conformer"],
    help="Choose the speech recognition model to use"
)

# Load model configuration
@st.cache_resource
def load_model_config():
    """Load model configuration."""
    try:
        config = load_config("configs/config.yaml")
        return config
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return None

config = load_model_config()
if config is None:
    st.error("Failed to load configuration. Please check the config file.")
    st.stop()

# Model loading
@st.cache_resource
def load_model(_model_type: str, _config):
    """Load the selected model."""
    try:
        device = get_device()
        
        if _model_type == "Wav2Vec2":
            model = Wav2Vec2PathologicalSpeechModel(_config.model)
        elif _model_type == "Conformer":
            model = ConformerPathologicalSpeechModel(_config.model)
        else:
            raise ValueError(f"Unknown model type: {_model_type}")
        
        model = model.to(device)
        model.eval()
        
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, device = load_model(model_type, config)
if model is None:
    st.error("Failed to load model. Please check the model files.")
    st.stop()

# Initialize components
preprocessor = AudioPreprocessor(config.data)
augmentation = PathologicalSpeechAugmentation(config.data.augmentation)
metrics = PathologicalSpeechMetrics()

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Audio Input")
    
    # Audio input options
    input_method = st.radio(
        "Choose input method:",
        ["Upload Audio File", "Record Audio"],
        horizontal=True
    )
    
    audio_data = None
    sample_rate = None
    
    if input_method == "Upload Audio File":
        uploaded_file = st.file_uploader(
            "Upload an audio file",
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Upload an audio file containing speech to transcribe"
        )
        
        if uploaded_file is not None:
            try:
                # Load audio
                audio_data, sample_rate = torchaudio.load(io.BytesIO(uploaded_file.read()))
                
                # Convert to mono if stereo
                if audio_data.shape[0] > 1:
                    audio_data = audio_data.mean(dim=0, keepdim=True)
                
                st.success(f"Audio loaded successfully! Duration: {audio_data.shape[1] / sample_rate:.2f}s")
                
            except Exception as e:
                st.error(f"Error loading audio file: {e}")
    
    elif input_method == "Record Audio":
        st.info("Audio recording functionality would be implemented here using streamlit-audio-recorder")
        # Note: This would require additional dependencies and implementation
    
    # Audio visualization
    if audio_data is not None:
        st.subheader("Audio Waveform")
        
        # Create waveform plot
        fig, ax = plt.subplots(figsize=(10, 4))
        time_axis = np.linspace(0, audio_data.shape[1] / sample_rate, audio_data.shape[1])
        ax.plot(time_axis, audio_data[0].numpy())
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Audio Waveform")
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Audio playback
        st.audio(audio_data[0].numpy(), sample_rate=sample_rate)

with col2:
    st.header("Processing Options")
    
    # Augmentation options
    st.subheader("Augmentation")
    apply_augmentation = st.checkbox("Apply Pathological Speech Augmentation", value=False)
    
    if apply_augmentation:
        st.markdown("**Augmentation Settings:**")
        tremor_sim = st.checkbox("Tremor Simulation", value=True)
        slur_sim = st.checkbox("Slur Simulation", value=True)
        volume_reduction = st.checkbox("Volume Reduction", value=True)
        noise_addition = st.checkbox("Noise Addition", value=True)
    
    # Processing button
    if st.button("🚀 Process Audio", disabled=audio_data is None):
        if audio_data is not None:
            with st.spinner("Processing audio..."):
                try:
                    # Preprocess audio
                    processed_audio = preprocessor(audio_data, sample_rate)
                    
                    # Apply augmentation if requested
                    if apply_augmentation:
                        processed_audio = augmentation(processed_audio)
                    
                    # Prepare input for model
                    if model_type == "Wav2Vec2":
                        # For Wav2Vec2, use raw waveform
                        model_input = processed_audio
                    else:
                        # For Conformer, extract features
                        model_input = preprocessor.extract_features(processed_audio)
                    
                    # Move to device
                    model_input = model_input.to(device)
                    
                    # Generate transcription
                    with torch.no_grad():
                        if hasattr(model, 'transcribe'):
                            transcription = model.transcribe(model_input)
                        else:
                            predicted_ids = model.generate(model_input)
                            # Decode using processor if available
                            if hasattr(model, 'processor'):
                                transcription = model.processor.decode(predicted_ids[0])
                            else:
                                transcription = "Transcription not available"
                    
                    # Store results
                    st.session_state['transcription'] = transcription
                    st.session_state['processed_audio'] = processed_audio
                    st.session_state['model_input'] = model_input
                    
                    st.success("Processing completed!")
                    
                except Exception as e:
                    st.error(f"Error processing audio: {e}")
                    logger.error(f"Processing error: {e}")

# Results section
if 'transcription' in st.session_state:
    st.header("Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Transcription")
        st.text_area(
            "Recognized Text:",
            value=st.session_state['transcription'],
            height=100,
            disabled=True
        )
        
        # Confidence analysis
        if 'model_input' in st.session_state:
            st.subheader("Confidence Analysis")
            
            # Simple confidence based on model output
            with torch.no_grad():
                outputs = model(st.session_state['model_input'])
                if 'logits' in outputs:
                    logits = outputs['logits']
                    probs = torch.softmax(logits, dim=-1)
                    max_probs = torch.max(probs, dim=-1)[0]
                    avg_confidence = torch.mean(max_probs).item()
                    
                    st.metric("Average Confidence", f"{avg_confidence:.3f}")
                    
                    # Confidence distribution
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(max_probs.cpu().numpy(), bins=20, alpha=0.7)
                    ax.set_xlabel("Confidence Score")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Confidence Distribution")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
    
    with col2:
        st.subheader("Audio Analysis")
        
        if 'processed_audio' in st.session_state:
            processed_audio = st.session_state['processed_audio']
            
            # Audio statistics
            duration = processed_audio.shape[1] / config.data.sample_rate
            rms_energy = torch.sqrt(torch.mean(processed_audio ** 2)).item()
            zero_crossing_rate = torch.mean(torch.abs(torch.diff(torch.sign(processed_audio)))).item()
            
            st.metric("Duration", f"{duration:.2f}s")
            st.metric("RMS Energy", f"{rms_energy:.4f}")
            st.metric("Zero Crossing Rate", f"{zero_crossing_rate:.4f}")
            
            # Spectrogram
            st.subheader("Spectrogram")
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Compute spectrogram
            spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=160)(processed_audio)
            log_spec = torch.log(spectrogram + 1e-8)
            
            im = ax.imshow(log_spec[0].cpu().numpy(), aspect='auto', origin='lower')
            ax.set_xlabel("Time Frames")
            ax.set_ylabel("Frequency Bins")
            ax.set_title("Log Spectrogram")
            plt.colorbar(im, ax=ax)
            
            st.pyplot(fig)

# Model information
st.header("Model Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Type", model_type)
    st.metric("Device", str(device))

with col2:
    model_info = model.get_model_info()
    st.metric("Total Parameters", f"{model_info['total_parameters']:,}")
    st.metric("Trainable Parameters", f"{model_info['trainable_parameters']:,}")

with col3:
    st.metric("Model Size", f"{model_info['model_size_mb']:.1f} MB")
    st.metric("Vocabulary Size", model_info['vocab_size'])

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>
        <strong>Pathological Speech Recognition Demo</strong><br>
        Research and Educational Use Only<br>
        Built with PyTorch, Transformers, and Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
