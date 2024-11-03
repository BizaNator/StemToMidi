import gradio as gr
import os
import tempfile
from demucs_handler import DemucsProcessor
from basic_pitch_handler import BasicPitchConverter

def create_interface():
    processor = DemucsProcessor()
    converter = BasicPitchConverter()

    def process_audio(audio_file, stem_type, convert_midi=True):
        try:
            if audio_file is None:
                return None, "Please upload an audio file."

            # Process audio through Demucs
            sources, sample_rate = processor.separate_stems(audio_file)
            
            # Get the selected stem
            stem_index = ["drums", "bass", "other", "vocals"].index(stem_type)
            selected_stem = sources[stem_index]
            
            # Create a temporary directory for output files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save the selected stem
                stem_path = os.path.join(temp_dir, f"{stem_type}.wav")
                processor.save_stem(selected_stem, stem_type, temp_dir, sample_rate)
                
                midi_path = None
                if convert_midi:
                    # Convert to MIDI
                    midi_path = os.path.join(temp_dir, f"{stem_type}.mid")
                    converter.convert_to_midi(stem_path, midi_path)
                
                return stem_path, midi_path
        except Exception as e:
            return None, str(e)

    interface = gr.Interface(
        fn=process_audio,
        inputs=[
            gr.Audio(type="filepath", label="Upload Audio", source="upload"),
            gr.Dropdown(
                choices=["vocals", "drums", "bass", "other"],
                label="Select Stem",
                value="vocals"
            ),
            gr.Checkbox(label="Convert to MIDI", value=True)
        ],
        outputs=[
            gr.Audio(label="Separated Stem"),
            gr.File(label="MIDI File")
        ],
        title="Audio Stem Separator & MIDI Converter",
        description="Upload audio to separate stems and convert to MIDI"
    )
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()

