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
            
            if progress:
                progress(0.7, "Saving MIDI file...")
            
            # Save as MIDI
            note_creation.save_midi(
                notes,
                output_path,
                tempo=self.process_options['tempo']
            )
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error in MIDI conversion: {str(e)}")
            raise

    def set_process_options(self, **kwargs):
        """Update processing options"""
        self.process_options.update(kwargs)