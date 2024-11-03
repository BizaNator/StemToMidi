import logging
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import pretty_midi
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class BasicPitchConverter:
    def __init__(self):
        self.process_options = {
            'onset_threshold': 0.5,
            'frame_threshold': 0.3,
            'minimum_note_length': 127.70,  # in milliseconds
            'minimum_frequency': 32.7,  # C1
            'maximum_frequency': 2093,  # C7
            'multiple_pitch_bends': True,
            'melodia_trick': True,
            'midi_tempo': 120.0
        }
        print("Basic Pitch converter initialized")  # Keep using print for consistency

    def convert_to_midi(self, audio_path: str, output_path: str, progress: Optional[callable] = None) -> str:
        """
        Convert audio to MIDI using Basic Pitch.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to save MIDI file
            progress: Optional callback function for progress updates
        
        Returns:
            str: Path to saved MIDI file
        """
        try:
            print(f"Converting to MIDI: {audio_path}")  # Keep debugging output
            if progress:
                progress(0.1, "Loading audio for MIDI conversion...")
            
            # Predict using Basic Pitch with correct parameters
            model_output, midi_data, note_events = predict(
                audio_path=audio_path,
                onset_threshold=self.process_options['onset_threshold'],
                frame_threshold=self.process_options['frame_threshold'],
                minimum_note_length=self.process_options['minimum_note_length'],
                minimum_frequency=self.process_options['minimum_frequency'],
                maximum_frequency=self.process_options['maximum_frequency'],
                multiple_pitch_bends=self.process_options['multiple_pitch_bends'],
                melodia_trick=self.process_options['melodia_trick'],
                midi_tempo=self.process_options['midi_tempo']
            )
            
            if progress:
                progress(0.7, "Saving MIDI file...")
            
            print(f"Saving MIDI to: {output_path}")  # Keep debugging output
            
            # Save MIDI file with validation
            if isinstance(midi_data, pretty_midi.PrettyMIDI):
                midi_data.write(output_path)
                print(f"Successfully saved MIDI to {output_path}")  # Keep using print
                return output_path
            else:
                raise ValueError("MIDI conversion failed: Invalid MIDI data")
            
        except Exception as e:
            print(f"Error in MIDI conversion: {str(e)}")  # Keep using print
            raise

    def set_process_options(self, **kwargs):
        """Update processing options"""
        self.process_options.update(kwargs)