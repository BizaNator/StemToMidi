import torch
import torchaudio
from demucs import pretrained

class DemucsProcessor:
    def __init__(self, model_name="htdemucs"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = pretrained.get_model(model_name)
        self.model.to(self.device)
    
    def separate_stems(self, audio_path):
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Process stems
        sources = self.model.separate(waveform.to(self.device))
        
        # Return separated stems
        return sources, sample_rate

    def configure_processing(self):
        gpu_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        if gpu_memory < 2e9:  # Less than 2GB
            return {"device": "cpu"}
        elif gpu_memory < 7e9:  # Less than 7GB
            return {
                "device": "cuda",
                "segment_size": 8,
                "overlap": 0.1
            }
        return {"device": "cuda"}

    def save_stem(self, stem, stem_name, output_path, sample_rate):
        torchaudio.save(
            f"{output_path}/{stem_name}.wav",
            stem.cpu(),
            sample_rate
        )
