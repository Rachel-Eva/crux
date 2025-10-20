"""
audio_input.py

Responsibilities:
- Validate uploaded audio files (format, sample rate, channels, max duration)
- Convert / normalize audio to a canonical format (WAV, 16 kHz, mono)
- Provide helpers to read audio and export normalized temp file path

Dependencies:
- pydub (requires ffmpeg on PATH)
- soundfile
- numpy

Exports:
- validate_audio_file(path) -> dict
- normalize_audio_to_wav(src_path, dst_path=None, target_sr=16000) -> dst_path
- load_audio_to_numpy(wav_path) -> (numpy_array, sample_rate)

Usage (CLI):
python audio_input.py tests/data/sample_short.mp3
"""
import os
import tempfile
from typing import Tuple, Optional, Dict

import numpy as np
from pydub import AudioSegment
import soundfile as sf

MAX_DURATION_SECONDS = 3 * 60 * 60  # 3 hours upper bound


class AudioProcessingError(Exception):
    pass


def validate_audio_file(path: str) -> Dict:
    """
    Validate audio exists and return basic metadata.

    Returns: {path, channels, sample_rate, duration_seconds, frame_width}
    Raises AudioProcessingError on problems.
    """
    if not os.path.isfile(path):
        raise AudioProcessingError(f"File not found: {path}")

    try:
        audio = AudioSegment.from_file(path)
    except Exception as e:
        raise AudioProcessingError(f"Could not open audio file: {e}")

    duration_s = len(audio) / 1000.0
    if duration_s <= 0:
        raise AudioProcessingError("Audio duration is zero.")
    if duration_s > MAX_DURATION_SECONDS:
        raise AudioProcessingError(f"Audio too long: {duration_s:.1f}s")

    return {
        "path": path,
        "channels": audio.channels,
        "sample_rate": audio.frame_rate,
        "duration_seconds": duration_s,
        "frame_width": audio.sample_width,
    }


def normalize_audio_to_wav(src_path: str, dst_path: Optional[str] = None, target_sr: int = 16000) -> str:
    """
    Convert audio to WAV, target sample rate, mono.
    Returns path to created WAV file.
    """
    if dst_path is None:
        fd, dst_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    try:
        audio = AudioSegment.from_file(src_path)
        audio = audio.set_frame_rate(target_sr).set_channels(1)
        audio.export(dst_path, format="wav")
    except Exception as e:
        raise AudioProcessingError(f"Failed to normalize audio: {e}")

    return dst_path


def load_audio_to_numpy(wav_path: str) -> Tuple[np.ndarray, int]:
    """
    Load a WAV to numpy (float32, -1..1) and return (samples, sr).
    """
    try:
        data, sr = sf.read(wav_path, dtype="float32")
    except Exception as e:
        raise AudioProcessingError(f"soundfile read failed: {e}")

    # If stereo, convert to mono by averaging channels
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    return data, sr


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quick test for audio_input module")
    parser.add_argument("file", help="Path to audio file to test")
    args = parser.parse_args()

    print("Validating:", args.file)
    try:
        info = validate_audio_file(args.file)
        print("Info:", info)
        wav = normalize_audio_to_wav(args.file)
        print("Normalized to:", wav)
        data, sr = load_audio_to_numpy(wav)
        print(f"Loaded numpy shape: {data.shape}, sr={sr}")
    except AudioProcessingError as e:
        print("ERROR:", e)
