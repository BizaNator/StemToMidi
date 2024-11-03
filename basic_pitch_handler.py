from basic_pitch.inference import predict
from basic_pitch import note_creation
from basic_pitch import BasicPitch
from basic_pitch.inference import predict
from basic_pitch.note_creation import create_notes_from_model_output
from basic_pitch.inference import BasicPitch
import numpy as np

class BasicPitchConverter:
    def __init__(self):
        pass
    
        self.model = BasicPitch()
    
    def convert_to_midi(self, audio_path, output_path):
        # Load audio and convert to MIDI
        model_output = predict(audio_path)
        
        # Create notes from the model output
        notes = note_creation.create_notes_from_model_output(
            model_output['pitch_outputs'],
            model_output['onset_outputs'],
            model_output['offset_outputs'],
            model_output['velocity_outputs'],
            minimum_duration=self.process_options()['min_note_duration'],
            minimum_frequency=self.process_options()['min_frequency'],
            maximum_frequency=self.process_options()['max_frequency'],
            onset_threshold=self.process_options()['onset_threshold']
        )
        
        # Save as MIDI
        note_creation.save_midi(notes, output_path, tempo=self.process_options()['tempo'])
        
        return output_path

    def process_options(self):
        return {
            'tempo': 120,
            'min_note_duration': 0.125,
            'min_frequency': 32.7,  # C1
            'max_frequency': 2093,  # C7
            'onset_threshold': 0.5
        }