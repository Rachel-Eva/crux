"""
audio_input.py

Responsibilities:
- Validate uploaded audio files (format, sample rate, channels, max duration)
- Convert / normalize audio to a canonical format (WAV, 16 kHz, mono)
- Provide helpers to read audio and export normalized temp file path

Dependencies:
- ffmpeg and ffprobe (must be on system PATH)
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
import subprocess
from typing import Tuple, Optional, Dict

import numpy as np
import soundfile as sf

MAX_DURATION_SECONDS = 3 * 60 * 60  # 3 hours upper bound


class AudioProcessingError(Exception):
    pass


def validate_audio_file(path: str) -> Dict:
    """
    Validate audio exists and return basic metadata using ffprobe.

    Returns: {path, channels, sample_rate, duration_seconds}
    Raises AudioProcessingError on problems.
    """
    if not os.path.isfile(path):
        raise AudioProcessingError(f"File not found: {path}")

    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=channels,sample_rate,duration",
            "-of", "default=noprint_wrappers=1:nokey=0",
            path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = result.stdout.strip().splitlines()
        info = {}
        for line in lines:
            if "=" in line:
                key, val = line.split("=", 1)
                info[key.strip()] = val.strip()

        duration = float(info.get("duration", 0))
        if duration <= 0:
            raise AudioProcessingError("Audio duration is zero.")
        if duration > MAX_DURATION_SECONDS:
            raise AudioProcessingError(f"Audio too long: {duration:.1f}s")

        return {
            "path": path,
            "channels": int(info.get("channels", 1)),
            "sample_rate": int(info.get("sample_rate", 16000)),
            "duration_seconds": duration,
        }

    except Exception as e:
        raise AudioProcessingError(f"Failed to validate audio: {e}")


def normalize_audio_to_wav(src_path: str, dst_path: Optional[str] = None, target_sr: int = 16000) -> str:
    """
    Convert audio to WAV, target sample rate, mono using ffmpeg.
    Returns path to created WAV file.
    """
    if not os.path.exists(src_path):
        raise AudioProcessingError(f"Audio file not found: {src_path}")

    if dst_path is None:
        fd, dst_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-ar", str(target_sr),
        "-ac", "1",
        dst_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise AudioProcessingError(f"ffmpeg failed to normalize audio: {e.stderr.decode()}")

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
