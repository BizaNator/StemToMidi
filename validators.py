import os
import logging
import torchaudio
from typing import Tuple

logger = logging.getLogger(__name__)

class AudioValidator:
    SUPPORTED_FORMATS = ['.mp3', '.wav', '.flac']
    MAX_FILE_SIZE = 125 * 1024 * 1024  # Max Upload File Size Increased to 125MB
    
    @staticmethod
    def validate_audio_file(file_path: str) -> Tuple[bool, str]:
        try:
            if not os.path.exists(file_path):
                return False, "File does not exist"
                
            file_size = os.path.getsize(file_path)
            if file_size > AudioValidator.MAX_FILE_SIZE:
                return False, f"File too large. Maximum size: {AudioValidator.MAX_FILE_SIZE // 1024 // 1024}MB"
                
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in AudioValidator.SUPPORTED_FORMATS:
                return False, f"Unsupported format. Supported formats: {', '.join(AudioValidator.SUPPORTED_FORMATS)}"
                
            # Validate audio file integrity
            try:
                waveform, sample_rate = torchaudio.load(file_path)
                if sample_rate < 8000 or sample_rate > 48000:
                    return False, "Invalid sample rate"
            except Exception as e:
                return False, f"Invalid audio file: {str(e)}"
                
            return True, "Valid audio file"
        except Exception as e:
            logger.error(f"Error validating audio file: {str(e)}")
            return False, str(e)