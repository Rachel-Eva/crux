"""
speaker_id.py

Responsibilities:
- Run speaker diarization and return speaker-labeled segments.
- Preferred backend: pyannote.audio pipeline (requires HF_TOKEN).
- Fallback: simple fixed-window diarizer for quick demos.

Functions:
- diarize_with_pyannote(wav_path, hf_token_env="HF_TOKEN") -> list[dict]
- simple_window_diarization(wav_path, window_sec=30.0) -> list[dict]

Segment format:
[ {"start": float, "end": float, "speaker": "SPEAKER_XX"} ]
"""
import os
from typing import List, Dict

from audio_input import validate_audio_file

try:
    from pyannote.audio import Pipeline
except Exception:
    Pipeline = None

class DiarizationError(Exception):
    pass


def diarize_with_pyannote(wav_path: str, hf_token_env: str = "HF_TOKEN") -> List[Dict]:
    """
    Use pyannote.audio pretrained speaker-diarization pipeline.
    Requires HF_TOKEN env var with a Hugging Face access token that can use the model.
    Returns list of {"start": float, "end": float, "speaker": str}
    """
    if Pipeline is None:
        raise DiarizationError("pyannote.audio not installed. pip install pyannote.audio")

    hf_token = os.getenv(hf_token_env)
    if not hf_token:
        raise DiarizationError(f"{hf_token_env} env var not set; create a Hugging Face token and export it.")

    validate_audio_file(wav_path)

    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)
        diarization = pipeline(wav_path)  # returns pyannote.core.Annotation
    except Exception as e:
        raise DiarizationError(f"pyannote diarization failed: {e}")

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)})
    return segments


def simple_window_diarization(wav_path: str, window_sec: float = 30.0) -> List[Dict]:
    """
    Naive fallback splitting audio into fixed windows and round-robin speakers.
    Useful for demo/testing only.
    """
    import soundfile as sf
    data, sr = sf.read(wav_path)
    duration = len(data) / sr
    segments = []
    start = 0.0
    idx = 0
    while start < duration:
        end = min(start + window_sec, duration)
        segments.append({"start": float(start), "end": float(end), "speaker": f"SPEAKER_{(idx % 4):02d}"})
        start = end
        idx += 1
    return segments


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quick diarization demo")
    parser.add_argument("wav", help="Normalized WAV file path")
    parser.add_argument("--backend", choices=["pyannote", "simple"], default="simple")
    args = parser.parse_args()

    try:
        if args.backend == "pyannote":
            segs = diarize_with_pyannote(args.wav)
        else:
            segs = simple_window_diarization(args.wav, window_sec=30.0)
        print("Found segments:", len(segs))
        for s in segs[:10]:
            print(s)
    except DiarizationError as e:
        print("ERROR:", e)
