import logging
from transformers import AutoModel, AutoProcessor
import torch
from typing import Optional

logger = logging.getLogger(__name__)

class BasicPitchConverter:
    def __init__(self):
        try:
            self.processor = AutoProcessor.from_pretrained("spotify/basic-pitch")
            self.model = AutoModel.from_pretrained("spotify/basic-pitch")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info("Basic Pitch model loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing Basic Pitch model: {str(e)}")
            raise

    def convert_to_midi(self, audio_path: str, output_path: str, progress: Optional[callable] = None) -> str:
        try:
            if progress:
                progress(0.1, "Loading audio for MIDI conversion...")
                
            # Process audio with the Spotify model
            audio_input = self.processor(audio_path, return_tensors="pt")
            
            if progress:
                progress(0.4, "Converting to MIDI...")
                
            outputs = self.model(**audio_input)
            
            if progress:
                progress(0.7, "Saving MIDI file...")
                
            # Convert model outputs to MIDI
            self.save_midi(outputs, output_path)
            
            return output_path
        except Exception as e:
            logger.error(f"Error in MIDI conversion: {str(e)}")
            raise