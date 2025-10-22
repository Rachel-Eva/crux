"""
crux_main.py

Orchestrator for the CRUX pipeline (audio -> stt -> diarize -> merge -> nlp -> output)

Responsibilities
----------------
- Provide a simple CLI and programmatic interface:
    - process_audio_file(audio_path, stt_backend="local"|"openai", diarize_backend="simple"|"pyannote", use_llm=False, dry_run=False)
- Steps:
    1. Normalize audio (audio_input.normalize_audio_to_wav)
    2. Transcribe (speech_to_text.*)
    3. Diarize (speaker_id.*)
    4. Merge speaker labels with transcript segments into labeled blocks
    5. Run NLP extraction (nlp_extraction.extract_tasks_from_labeled_transcript)
    6. Output JSON meeting object (save to disk or return)
"""

import os
import json
import logging
from typing import Dict, Any, List

# Local imports (assumes files are in same package/folder)
from audio_input import normalize_audio_to_wav, validate_audio_file
from speech_to_text import transcribe_with_local_whisper, transcribe_with_openai, STTError
from speaker_id import simple_window_diarization, diarize_with_pyannote, DiarizationError
from nlp_extraction import extract_tasks_from_labeled_transcript

# Optional Notion integration (updated)
try:
    from notion_client import Client as NotionClient
    from notion_sync import push_task_to_notion, extract_deadline
    NOTION_AVAILABLE = True
except ImportError:
    NotionClient = None
    push_task_to_notion = None
    extract_deadline = None
    NOTION_AVAILABLE = False

# -------------------- Logging --------------------
logger = logging.getLogger("crux_main")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)


# -------------------- Exceptions --------------------
class CRUXError(Exception):
    pass


# -------------------- Helpers --------------------
def merge_transcript_and_diarization(transcript: Dict[str, Any], diarization: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Align transcript segments with diarization segments and return speaker-labeled blocks.
    """
    t_segments = transcript.get("segments", [])
    if not t_segments:
        total_end = max((d["end"] for d in diarization), default=0.0)
        t_segments = [{"start": 0.0, "end": total_end, "text": transcript.get("text", "")}]

    labeled = []
    for t in t_segments:
        tstart, tend, ttext = t.get("start", 0.0), t.get("end", 0.0), t.get("text", "")
        overlapping = [d for d in diarization if not (d["end"] <= tstart or d["start"] >= tend)]
        if overlapping:
            best = max(overlapping, key=lambda d: min(d["end"], tend) - max(d["start"], tstart))
            labeled.append({"start": tstart, "end": tend, "speaker": best["speaker"], "text": ttext})
        else:
            labeled.append({"start": tstart, "end": tend, "speaker": "UNKNOWN", "text": ttext})
    return labeled


# -------------------- Main Pipeline --------------------
def process_audio_file(
    src_path: str,
    work_dir: str = ".",
    target_sr: int = 16000,
    stt_backend: str = "local",
    diarize_backend: str = "simple",
    use_llm_for_nlp: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Full CRUX pipeline. Returns a meeting JSON dict.
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
    logger.debug("Transcript preview: %s", transcript.get("text", "")[:200])

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

    # 4) Merge transcript & diarization
    labeled_blocks = merge_transcript_and_diarization(transcript, diarization)

    # 5) NLP extraction
    logger.info("Extracting tasks from transcript blocks...")
    tasks = extract_tasks_from_labeled_transcript(labeled_blocks, use_llm_fallback=use_llm_for_nlp)
    logger.info("Detected %d tasks", len(tasks))
    for t in tasks:
        logger.info("📌 Task: %s (Speaker: %s, Deadline: %s)", 
                    t.get("sentence", "N/A"), t.get("speaker", "N/A"), t.get("deadline", "N/A"))

    # 6) Optional Notion sync (updated)
    if tasks and not dry_run and NOTION_AVAILABLE:
        logger.info("📤 Pushing tasks to Notion...")
        
        # Create Notion client (mirroring notion_sync.py)
        notion_token = os.environ.get("NOTION_TOKEN")
        notion_db = os.environ.get("NOTION_DATABASE_ID")
        if not notion_token or not notion_db:
            logger.warning("⚠️ NOTION_TOKEN or NOTION_DATABASE_ID not set — skipping sync.")
        else:
            notion = NotionClient(auth=notion_token)
            for task in tasks:
                task_text = task.get("sentence") or task.get("description") or task.get("task") or str(task)  # Prioritize 'sentence'
                confidence = task.get("confidence", 0.0)
                deadline = extract_deadline(task_text) if extract_deadline else None  # Optional: extract deadline
                description = f"Speaker: {task.get('speaker', 'N/A')}\nConfidence: {confidence:.2f}\nOriginal: {task_text}"
                
                try:
                    push_task_to_notion(
                        notion=notion,
                        database_id=notion_db,
                        task_text=task_text,
                        confidence=confidence,
                        deadline=deadline,
                        description=description,
                        dry_run=False  # Already checked dry_run earlier
                    )
                except Exception as e:
                    logger.error(f"Failed to push task '{task_text}': {e}")
            logger.info("✅ Notion sync complete.")
    elif dry_run:
        logger.info("🧪 Dry run mode — skipping Notion push.")
    elif not NOTION_AVAILABLE:
        logger.warning("⚠️ Notion integration unavailable — install notion-client and notion_sync.py")

    # 7) Compose meeting JSON
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


# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CRUX main pipeline (audio -> stt -> diarize -> nlp)")
    parser.add_argument("--src", required=True, help="Source audio file")
    parser.add_argument("--stt", choices=["local", "openai"], default="local")
    parser.add_argument("--diarize", choices=["simple", "pyannote"], default="simple")
    parser.add_argument("--use-llm-nlp", action="store_true", help="Use OpenAI LLM as NLP fallback")
    parser.add_argument("--dry-run", action="store_true", help="Run without pushing tasks (for debugging)")
    parser.add_argument("--out", default="meeting_output.json", help="Where to write JSON output")
    args = parser.parse_args()

    meeting = process_audio_file(
        args.src,
        stt_backend=args.stt,
        diarize_backend=args.diarize,
        use_llm_for_nlp=args.use_llm_nlp,
        dry_run=args.dry_run
    )

    with open(args.out, "w", encoding="utf8") as fh:
        json.dump(meeting, fh, indent=2)

    print(f"🏁 Wrote output JSON to {args.out}")