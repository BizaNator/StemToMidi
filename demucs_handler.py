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
            if self.device == "cuda":
                logger.info(f"GPU Memory before model load: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            
            self.model = get_model(model_name)
            self.model.to(self.device)
            
            if self.device == "cuda":
                logger.info(f"GPU Memory after model load: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
                
            logger.info(f"Demucs model {model_name} loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing Demucs model: {str(e)}")
            raise

    def separate_stems(self, audio_path: str, progress=None) -> Tuple[torch.Tensor, int]:
        try:
            if progress:
                progress(0.1, "Loading audio file...")
            
            # Log file info
            file_size = os.path.getsize(audio_path) / (1024 * 1024)  # Size in MB
            logger.info(f"Processing file: {audio_path} (Size: {file_size:.2f}MB)")
            
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            logger.info(f"Audio loaded - Sample rate: {sample_rate}, Shape: {waveform.shape}")
            
            if self.device == "cuda":
                logger.info(f"GPU Memory before processing: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            
            if progress:
                progress(0.3, "Processing stems...")
            
            # Input validation and logging: Check waveform dimensions
            logger.info(f"Waveform shape before processing: {waveform.shape}")
            if waveform.dim() not in (1, 2):
                raise ValueError(f"Invalid waveform dimensions: Expected 1D or 2D, got {waveform.dim()}")

            # Handle mono input by duplicating to stereo
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.shape[0] == 1:  # If mono, duplicate to stereo
                waveform = waveform.repeat(2, 1)
                logger.info("Converted mono to stereo by duplication")
            
            # Ensure 3D tensor for apply_model (batch, channels, time)
            waveform = waveform.unsqueeze(0)  # Add batch dimension
            logger.info(f"Waveform shape before apply_model: {waveform.shape}")

            # Process
            with torch.no_grad():
                sources = apply_model(self.model, waveform.to(self.device))
            
            if self.device == "cuda":
                logger.info(f"GPU Memory after processing: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            
            if progress:
                progress(0.8, "Finalizing separation...")
            
            return sources, sample_rate
            
        except Exception as e:
            logger.error(f"Error in stem separation: {str(e)}", exc_info=True)
            raise

    def save_stem(self, stem: torch.Tensor, stem_name: str, output_path: str, sample_rate: int):
        try:
            torchaudio.save(
                f"{output_path}/{stem_name}.wav",
                stem.cpu(),
                sample_rate
            )
        except Exception as e:
            logger.error(f"Error saving stem: {str(e)}")
            raise