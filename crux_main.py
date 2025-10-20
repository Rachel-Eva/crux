
---

# `crux_main.py`
```python
"""
crux_main.py

Orchestrator for the CRUX pipeline (audio -> stt -> diarize -> merge -> nlp -> output)

Responsibilities
----------------
- Provide a simple CLI and programmatic interface:
    - process_audio_file(audio_path, stt_backend="local"|"openai", diarize_backend="simple"|"pyannote", use_llm=False)
- Steps:
    1. Normalize audio (audio_input.normalize_audio_to_wav)
    2. Transcribe (speech_to_text.*)
    3. Diarize (speaker_id.*)
    4. Merge speaker labels with transcript segments into labeled blocks
    5. Run NLP extraction (nlp_extraction.extract_tasks_from_labeled_transcript)
    6. Output JSON meeting object (save to disk or return)

Usage (CLI)
-----------
python crux_main.py --wav path/to/norm.wav --stt local --diarize simple --out out.json
"""

from typing import Dict, Any, List
import os
import json
import logging

# Local imports (assumes files are in same package/folder)
from audio_input import normalize_audio_to_wav, validate_audio_file, load_audio_to_numpy
from speech_to_text import transcribe_with_local_whisper, transcribe_with_openai, STTError
from speaker_id import simple_window_diarization, diarize_with_pyannote, DiarizationError
from nlp_extraction import extract_tasks_from_labeled_transcript

logger = logging.getLogger("crux_main")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)


class CRUXError(Exception):
    pass


def merge_transcript_and_diarization(transcript: Dict[str, Any], diarization: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Align transcript segments with diarization segments and return speaker-labeled blocks.
    transcript: dict with 'segments' (list) or plain 'text'
    diarization: list of {'start','end','speaker'}
    """
    t_segments = transcript.get("segments", [])
    if not t_segments:
        # Fallback: single segment spanning diarization
        total_end = max((d["end"] for d in diarization), default=0.0)
        t_segments = [{"start": 0.0, "end": total_end, "text": transcript.get("text", "")}]

    labeled = []
    for t in t_segments:
        tstart, tend, ttext = t.get("start", 0.0), t.get("end", 0.0), t.get("text", "")
        overlapping = [d for d in diarization if not (d["end"] <= tstart or d["start"] >= tend)]
        if overlapping:
            # choose speaker with largest overlap
            best = max(overlapping, key=lambda d: min(d["end"], tend) - max(d["start"], tstart))
            labeled.append({"start": tstart, "end": tend, "speaker": best["speaker"], "text": ttext})
        else:
            labeled.append({"start": tstart, "end": tend, "speaker": "UNKNOWN", "text": ttext})
    return labeled


def process_audio_file(
    src_path: str,
    work_dir: str = ".",
    target_sr: int = 16000,
    stt_backend: str = "local",
    diarize_backend: str = "simple",
    use_llm_for_nlp: bool = False,
) -> Dict[str, Any]:
    """
    Full pipeline. Returns a meeting JSON dict.
    """
    if not os.path.exists(src_path):
        raise CRUXError(f"Audio file not found: {src_path}")

    # 1) Normalize audio
    logger.info("Validating audio")
    validate_audio_file(src_path)
    norm_path = normalize_audio_to_wav(src_path, dst_path=None, target_sr=target_sr)
    logger.info("Normalized audio to %s", norm_path)

    # 2) Transcription
    logger.info("Transcribing audio with backend=%s", stt_backend)
    try:
        if stt_backend == "openai":
            transcript = transcribe_with_openai(norm_path)
        else:
            transcript = transcribe_with_local_whisper(norm_path, model="tiny")
    except STTError as e:
        raise CRUXError(f"Transcription error: {e}")

    logger.info("Transcript length %d chars", len(transcript.get("text", "")))

    # 3) Diarization
    logger.info("Diarizing audio with backend=%s", diarize_backend)
    try:
        if diarize_backend == "pyannote":
            diarization = diarize_with_pyannote(norm_path)
        else:
            diarization = simple_window_diarization(norm_path, window_sec=30.0)
    except DiarizationError as e:
        raise CRUXError(f"Diarization error: {e}")

    logger.info("Found %d diarization segments", len(diarization))

    # 4) Merge
    labeled_blocks = merge_transcript_and_diarization(transcript, diarization)

    # 5) NLP extraction
    tasks = extract_tasks_from_labeled_transcript(labeled_blocks, use_llm_fallback=use_llm_for_nlp)

    # 6) Compose meeting JSON
    meeting = {
        "MeetingTitle": os.path.splitext(os.path.basename(src_path))[0],
        "Date": "",  # could be set via metadata or file timestamp
        "AudioPath": src_path,
        "NormalizedAudio": norm_path,
        "Transcript": transcript.get("text", ""),
        "LabeledBlocks": labeled_blocks,
        "Tasks": tasks,
    }

    return meeting


# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CRUX main pipeline (audio -> stt -> diarize -> nlp)")
    parser.add_argument("--src", required=True, help="Source audio file")
    parser.add_argument("--stt", choices=["local", "openai"], default="local")
    parser.add_argument("--diarize", choices=["simple", "pyannote"], default="simple")
    parser.add_argument("--use-llm-nlp", action="store_true", help="Use OpenAI LLM as NLP fallback (requires OPENAI_API_KEY)")
    parser.add_argument("--out", default="meeting_output.json", help="Where to write JSON output")
    args = parser.parse_args()

    meeting = process_audio_file(args.src, stt_backend=args.stt, diarize_backend=args.diarize, use_llm_for_nlp=args.use_llm_nlp)
    with open(args.out, "w", encoding="utf8") as fh:
        json.dump(meeting, fh, indent=2)
    print("Wrote", args.out)
