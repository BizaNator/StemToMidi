import torch
import torchaudio
import logging
from demucs.pretrained import get_model
from demucs.apply import apply_model
from typing import Tuple

logger = logging.getLogger(__name__)

class DemucsProcessor:
    def __init__(self, model_name="htdemucs"):
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = get_model(model_name)
            self.model.to(self.device)
            logger.info(f"Demucs model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing Demucs model: {str(e)}")
            raise

    def separate_stems(self, audio_path: str, progress=None) -> Tuple[torch.Tensor, int]:
        try:
            if progress:
                progress(0.1, "Loading audio file...")
            
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if progress:
                progress(0.3, "Processing stems...")
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Adjust channels for model input
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # Process
            with torch.no_grad():
                sources = apply_model(self.model, waveform.to(self.device), progress=progress)
            
            if progress:
                progress(0.8, "Finalizing separation...")
            
            return sources, sample_rate
            
        except Exception as e:
            logger.error(f"Error in stem separation: {str(e)}")
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