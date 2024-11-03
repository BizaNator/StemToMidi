import gradio as gr
import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import logging
from validators import AudioValidator
from demucs_handler import DemucsProcessor
from basic_pitch_handler import BasicPitchConverter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='audio_processor.log'
)
logger = logging.getLogger(__name__)

def batch_process_audio(file_paths: List[str], stem_type: str, convert_midi: bool = True) -> List[Tuple[str, Optional[str]]]:
    """
    Process multiple audio files in parallel
    """
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            lambda x: process_single_audio(x, stem_type, convert_midi),
            file_paths
        ))
    return results

def process_single_audio(audio_path: str, stem_type: str, convert_midi: bool) -> Tuple[str, Optional[str]]:
    with tempfile.TemporaryDirectory() as temp_dir:
        processor = DemucsProcessor()
        converter = BasicPitchConverter()
        
        # Process stems
        sources, sample_rate = processor.separate_stems(audio_path)
        stem_index = ["drums", "bass", "other", "vocals"].index(stem_type)
        selected_stem = sources[stem_index]
        
        # Save stem
        stem_path = os.path.join(temp_dir, f"{stem_type}.wav")
        processor.save_stem(selected_stem, stem_type, temp_dir, sample_rate)
        
        # Convert to MIDI if requested
        midi_path = None
        if convert_midi:
            midi_path = os.path.join(temp_dir, f"{stem_type}.mid")
            converter.convert_to_midi(stem_path, midi_path)
        
        return stem_path, midi_path

def create_interface():
    processor = DemucsProcessor()
    converter = BasicPitchConverter()
    validator = AudioValidator()

    def process_audio(
        audio_files: List[str],
        stem_type: str,
        convert_midi: bool = True,
        progress=gr.Progress()
    ) -> Tuple[List[str], List[Optional[str]]]:
        try:
            # Validate all files
            for audio_file in audio_files:
                is_valid, message = validator.validate_audio_file(audio_file)
                if not is_valid:
                    raise ValueError(f"Invalid audio file: {message}")

            # Process files in batch
            results = batch_process_audio(audio_files, stem_type, convert_midi)
            
            # Unzip results
            stem_files, midi_files = zip(*results)
            
            return list(stem_files), list(midi_files)
            
        except Exception as e:
            logger.error(f"Error in audio processing: {str(e)}")
            raise gr.Error(str(e))

    interface = gr.Interface(
        fn=process_audio,
        inputs=[
            gr.File(
                file_count="multiple",
                file_types=AudioValidator.SUPPORTED_FORMATS,
                label="Upload Audio Files"
            ),
            gr.Dropdown(
                choices=["vocals", "drums", "bass", "other"],
                label="Select Stem",
                value="vocals"
            ),
            gr.Checkbox(label="Convert to MIDI", value=True)
        ],
        outputs=[
            gr.Audio(label="Separated Stems", type="filepath"),
            gr.File(label="MIDI Files")
        ],
        title="Audio Stem Separator & MIDI Converter",
        description="Upload audio files to separate stems and convert to MIDI",
        examples=[
            ["example1.mp3", "vocals", True],
            ["example2.wav", "drums", False]
        ],
        cache_examples=True,
        allow_flagging="never"
    )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        share=False,  # Set to True for public access
        server_name="0.0.0.0",
        server_port=7860,
        auth=None,  # Add authentication if needed
        ssl_keyfile=None,  # Add SSL if needed
        ssl_certfile=None
    )