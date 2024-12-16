import gradio as gr
import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import logging
import soundfile as sf
import numpy as np
import shutil
from validators import AudioValidator
from demucs_handler import DemucsProcessor
from basic_pitch_handler import BasicPitchConverter

# Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Create a persistent directory for outputs
OUTPUT_DIR = Path("F:/AI_Tools/Audio/Audio2Stem2Midi/Aud2Stm2Mdi/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def process_single_audio(audio_path: str, stem_type: str, convert_midi: bool) -> Tuple[Tuple[int, np.ndarray], Optional[str]]:
    try:
        # Create unique subdirectory for this processing
        process_dir = OUTPUT_DIR / str(hash(audio_path))
        process_dir.mkdir(parents=True, exist_ok=True)
        
        processor = DemucsProcessor()
        converter = BasicPitchConverter()
        
        print(f"Starting processing of file: {audio_path}")
        
        # Process stems
        sources, sample_rate = processor.separate_stems(audio_path)
        print(f"Number of sources returned: {sources.shape}")
        print(f"Stem type requested: {stem_type}")
        
        # Get the requested stem
        stem_index = ["drums", "bass", "other", "vocals"].index(stem_type)
        selected_stem = sources[0, stem_index]
        
        # Save stem
        stem_path = process_dir / f"{stem_type}.wav"
        processor.save_stem(selected_stem, stem_type, str(process_dir), sample_rate)
        print(f"Saved stem to: {stem_path}")
        
        # Load the saved audio file for Gradio
        audio_data, sr = sf.read(str(stem_path))
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)  # Convert to mono if stereo
        
        # Convert to int16 format
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Convert to MIDI if requested
        midi_path = None
        if convert_midi:
            midi_path = process_dir / f"{stem_type}.mid"
            converter.convert_to_midi(str(stem_path), str(midi_path))
            print(f"Saved MIDI to: {midi_path}")
                
        return (sr, audio_data), str(midi_path) if midi_path else None
    except Exception as e:
        print(f"Error in process_single_audio: {str(e)}")
        raise

def create_interface():
    processor = DemucsProcessor()
    converter = BasicPitchConverter()
    validator = AudioValidator()

    def process_audio(
        audio_files: List[str],
        stem_type: str,
        convert_midi: bool = True,
        progress=gr.Progress()
    ) -> Tuple[Tuple[int, np.ndarray], Optional[str]]:
        try:
            print(f"Starting processing of {len(audio_files)} files")
            print(f"Selected stem type: {stem_type}")
            
            # Process single file for now
            if len(audio_files) > 0:
                audio_path = audio_files[0]  # Take first file
                print(f"Processing file: {audio_path}")
                return process_single_audio(audio_path, stem_type, convert_midi)
            else:
                raise ValueError("No audio files provided")
            
        except Exception as e:
            print(f"Error in audio processing: {str(e)}")
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
            gr.Audio(label="Separated Stems", type="numpy"),
            gr.File(label="MIDI Files")
        ],
        title="Audio Stem Separator & MIDI Converter",
        description="Upload audio files to separate stems and convert to MIDI\n\n" +
                   "Created by Ever Olivares - Looking for Summer 2025 Internship Opportunities\n" +
                   "Connect with me: [LinkedIn](https://www.linkedin.com/in/everolivares/)",
        cache_examples=True,
        allow_flagging="never"
    )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        auth=None,
        ssl_keyfile=None,
        ssl_certfile=None
    )