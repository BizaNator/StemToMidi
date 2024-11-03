import torch
import torchaudio
import logging
import os
from demucs.pretrained import get_model
from demucs.apply import apply_model
from typing import Tuple

logger = logging.getLogger(__name__)

class DemucsProcessor:
    def __init__(self, model_name="htdemucs"):
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            
            self.model = get_model(model_name)
            print(f"Model name: {model_name}")
            print(f"Model sources: {self.model.sources}")  # This will show available stems
            print(f"Model sample rate: {self.model.samplerate}")
            
            self.model.to(self.device)
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise

    def separate_stems(self, audio_path: str, progress=None) -> Tuple[torch.Tensor, int]:
        try:
            if progress:
                progress(0.1, "Loading audio file...")
            
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            print(f"Audio loaded - Shape: {waveform.shape}")
            
            if progress:
                progress(0.3, "Processing stems...")
            
            # Input validation and logging: Check waveform dimensions
            if waveform.dim() not in (1, 2):
                raise ValueError(f"Invalid waveform dimensions: Expected 1D or 2D, got {waveform.dim()}")

            # Handle mono input by duplicating to stereo
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
                print("Converted mono to stereo by duplication")
            
            # Ensure 3D tensor for apply_model (batch, channels, time)
            waveform = waveform.unsqueeze(0)
            print(f"Waveform shape before apply_model: {waveform.shape}")

            # Process
            with torch.no_grad():
                sources = apply_model(self.model, waveform.to(self.device))
                print(f"Sources shape after processing: {sources.shape}")
                print(f"Available stems: {self.model.sources}")
            
            if progress:
                progress(0.8, "Finalizing separation...")
            
            return sources, sample_rate
            
        except Exception as e:
            print(f"Error in stem separation: {str(e)}")
            raise

    def save_stem(self, stem: torch.Tensor, stem_name: str, output_path: str, sample_rate: int):
        try:
            torchaudio.save(
                f"{output_path}/{stem_name}.wav",
                stem.cpu(),
                sample_rate
            )
        except Exception as e:
            print(f"Error saving stem: {str(e)}")
            raise