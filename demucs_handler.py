import torch
import logging
from functools import lru_cache
from typing import Tuple
from transformers import AutoModel
import torchaudio

logger = logging.getLogger(__name__)

class DemucsProcessor:
    def __init__(self):
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # Load model from Hugging Face
            self.model = AutoModel.from_pretrained("facebook/demucs")
            self.model.to(self.device)
            logger.info("Demucs model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing Demucs model: {str(e)}")
            raise
    
    @lru_cache(maxsize=32)
    def separate_stems(self, audio_path: str, progress=None) -> Tuple[torch.Tensor, int]:
        try:
            if progress:
                progress(0.1, "Loading audio file...")
            
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if progress:
                progress(0.3, "Processing stems...")
                
            # Process in segments for memory efficiency
            sources = self.model.separate(waveform.to(self.device))
            
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