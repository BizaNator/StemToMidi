import logging
from basic_pitch import note_creation
from basic_pitch.inference import predict
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

class BasicPitchConverter:
    def __init__(self):
        self.process_options = {
            'tempo': 120,
            'min_note_duration': 0.125,
            'min_frequency': 32.7,  # C1
            'max_frequency': 2093,  # C7
            'onset_threshold': 0.5
        }
        logger.info("Basic Pitch converter initialized")

    def convert_to_midi(self, audio_path: str, output_path: str, progress: Optional[callable] = None) -> str:
        try:
            if progress:
                progress(0.1, "Loading audio for MIDI conversion...")
            
            # Get model output
            model_output = predict(audio_path)
            
            if progress:
                progress(0.4, "Converting to MIDI...")
            
            # Create notes from the model output
            notes = note_creation.create_notes_from_model_output(
                model_output['pitch_outputs'],
                model_output['onset_outputs'],
                model_output['offset_outputs'],
                model_output['velocity_outputs'],
                minimum_duration=self.process_options['min_note_duration'],
                minimum_frequency=self.process_options['min_frequency'],
                maximum_frequency=self.process_options['max_frequency'],
                onset_threshold=self.process_options['onset_threshold']
            )
            
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
            
            # Rest of your existing code...
            
            if self.device == "cuda":
                logger.info(f"GPU Memory after processing: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
                
            return sources, sample_rate
            
        except Exception as e:
            logger.error(f"Error in stem separation: {str(e)}", exc_info=True)  # Added exc_info=True
            raise

    def set_process_options(self, **kwargs):
        """Update processing options"""
        self.process_options.update(kwargs)