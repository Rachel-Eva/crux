"""
speech_to_text.py

Responsibilities:
- Transcribe normalized WAV audio into text
- Provide two interchangeable backends:
    - OpenAI transcription (via openai package)
    - Local whisper (openai-whisper) fallback

Functions:
- transcribe_with_openai(wav_path, model=None, language="en") -> dict
- transcribe_with_local_whisper(wav_path, model="small") -> dict

Output shape:
{
  "text": "full transcript",
  "segments": [ {"start": float, "end": float, "text": str}, ... ],
  "language": "en"
}
"""
import os
from typing import Dict, Optional

from audio_input import validate_audio_file

# optional imports
try:
    import openai
except Exception:
    openai = None

try:
    import whisper
except Exception:
    whisper = None


DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")  # adjust as needed


class STTError(Exception):
    pass


def transcribe_with_openai(wav_path: str, model: Optional[str] = None, language: str = "en") -> Dict:
    """
    Transcribe audio via OpenAI Speech-to-Text API (Audio.transcriptions).
    Requires OPENAI_API_KEY env var. Returns dict {text, segments, language}
    """
    if openai is None:
        raise STTError("openai package is not installed. pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise STTError("OPENAI_API_KEY not set in environment")
    openai.api_key = api_key

    validate_audio_file(wav_path)

    try:
        with open(wav_path, "rb") as fh:
            # Note: SDK endpoints may vary by openai version. Adapt call if needed.
            response = openai.Audio.transcriptions.create(
                model=model or DEFAULT_OPENAI_MODEL,
                file=fh,
                language=language
            )
    except Exception as e:
        raise STTError(f"OpenAI transcription failed: {e}")

    # Response parsing: be defensive
    text = response.get("text") or response.get("transcript") or ""
    segments = response.get("segments", [])
    return {"text": text, "segments": segments, "language": response.get("language", language)}


def transcribe_with_local_whisper(wav_path: str, model: str = "small") -> Dict:
    """
    Use local whisper (openai-whisper) for offline transcription.
    Returns same dict shape as transcribe_with_openai.
    """
    if whisper is None:
        raise STTError("whisper package not installed. pip install -U openai-whisper")

    validate_audio_file(wav_path)

    try:
        model_obj = whisper.load_model(model)
        result = model_obj.transcribe(wav_path, language="en")
    except Exception as e:
        raise STTError(f"Local Whisper transcription failed: {e}")

    text = result.get("text", "")
    segments = result.get("segments", [])
    return {"text": text, "segments": segments, "language": result.get("language", "en")}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quick transcription CLI")
    parser.add_argument("wav", help="Normalized WAV file path")
    parser.add_argument("--backend", choices=["openai", "local"], default="local")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    try:
        if args.backend == "openai":
            out = transcribe_with_openai(args.wav, model=args.model)
        else:
            out = transcribe_with_local_whisper(args.wav, model=args.model or "small")
        print("Full transcript:")
        print(out["text"][:2000])
        if out["segments"]:
            print("First segment:", out["segments"][0])
    except STTError as e:
        print("ERROR:", e)
