from __future__ import annotations
import whisper
import argparse
import logging
import os
from typing import List, Optional, Dict
from datetime import datetime
import audio_input, speech_to_text, nlp_extraction
from audio_input import normalize_audio_to_wav, validate_audio_file
from dotenv import load_dotenv

load_dotenv()

# Import Notion client at runtime (fix for NameError)
try:
    from notion_client import Client as NotionClient
except ImportError:
    NotionClient = None  # notion-client not installed or unavailable

# crux imports fallback
try:
    from audio_input import normalize_audio_to_wav, validate_audio_file
except Exception:
    normalize_audio_to_wav = None
    validate_audio_file = None

try:
    from speech_to_text import transcribe_with_local_whisper, transcribe_with_openai
except Exception:
    transcribe_with_local_whisper = None
    transcribe_with_openai = None

try:
    from nlp_extraction import extract_tasks_from_labeled_transcript, convert_text_to_labeled_blocks
except Exception:
    extract_tasks_from_labeled_transcript = None
    convert_text_to_labeled_blocks = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("crux.notion_sync")


def extract_deadline(text: str) -> Optional[str]:
    from dateparser.search import search_dates
    if not text:
        return None
    try:
        results = search_dates(text, settings={"PREFER_DATES_FROM": "future"})
    except Exception:
        return None
    if not results:
        return None
    _, dt = results[0]
    if dt.time() != datetime.min.time():
        return dt.replace(microsecond=0).isoformat()
    else:
        return dt.date().isoformat()


def push_task_to_notion(
    notion: Optional[NotionClient],
    database_id: str,
    task_text: str,
    confidence: float = None,
    deadline: Optional[str] = None,
    status: str = "Not started",
    priority: str = "Medium",
    description: Optional[str] = None,
    dry_run: bool = False,
):
    properties = {
        "Task name": {"title": [{"text": {"content": task_text}}]},
        "Status": {"status": {"name": status}},
        "Priority": {"select": {"name": priority}},
    }
    if deadline:
        properties["Due date"] = {"date": {"start": deadline}}

    desc_text = (
        description
        or (f"Extracted (confidence {confidence:.2f})" if confidence is not None else "Extracted from transcript")
    )
    properties["Description"] = {"rich_text": [{"text": {"content": desc_text}}]}

    if dry_run:
        logger.info("DRY RUN - would push to Notion: %s", properties)
        return

    if NotionClient is None or notion is None:
        raise RuntimeError("Notion client not available. Install notion-client and set NOTION_TOKEN/NOTION_DATABASE_ID")

    notion.pages.create(parent={"database_id": database_id}, properties=properties)
    logger.info("Pushed task to Notion: %s", task_text)


def run(
    audio_path: Optional[str] = None,
    backend: str = "whisper",
    model: Optional[str] = None,
    dry_run: bool = False,
    test_mode: bool = False,
    use_llm: bool = False,
):
    transcript_text = ""
    tasks: List[Dict] = []  # always initialize

    if test_mode:
        transcript_text = """
        Please finish the Q3 report by Friday.
        Assign John to review the design document.
        We need to implement the new feature for the mobile app.
        """
        logger.info("Running in TEST MODE with sample transcript")

        labeled_blocks = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 8.0, "text": "Please finish the Q3 report by Friday."},
            {"speaker": "SPEAKER_01", "start": 8.0, "end": 15.0, "text": "Assign John to review the design document."},
            {"speaker": "SPEAKER_02", "start": 15.0, "end": 22.0, "text": "We need to implement the new feature for the mobile app."},
        ]

        if extract_tasks_from_labeled_transcript:
            try:
                tasks = extract_tasks_from_labeled_transcript(labeled_blocks, use_llm_fallback=use_llm)
            except Exception as e:
                logger.warning("Hybrid extractor failed: %s", e)

        if not tasks:
            logger.info("Using improved fallback NLP extractor in test mode.")
            if convert_text_to_labeled_blocks and extract_tasks_from_labeled_transcript:
                blocks = convert_text_to_labeled_blocks(transcript_text)
                tasks = extract_tasks_from_labeled_transcript(blocks, confidence_floor=0.8)
                logger.info("Improved fallback detected %d tasks", len(tasks))
            else:
                logger.warning("NLP fallback extractor unavailable.")

    else:
        if not audio_path or not os.path.exists(audio_path):
            logger.error("Audio file not found: %s", audio_path)
            raise FileNotFoundError(audio_path)

        if validate_audio_file:
            try:
                info = validate_audio_file(audio_path)
                logger.info("Audio validation: %s", info)
            except Exception as e:
                logger.warning("validate_audio_file failed: %s", e)

        norm_path = audio_path
        if normalize_audio_to_wav:
            try:
                norm_path = normalize_audio_to_wav(audio_path)
                logger.info("Normalized audio written to %s", norm_path)
            except Exception as e:
                logger.warning("Audio normalization failed, will try original file: %s", e)
                norm_path = audio_path

        try:
            if backend == "openai" and transcribe_with_openai:
                transcript_obj = transcribe_with_openai(norm_path, model=model)
            elif transcribe_with_local_whisper:
                transcript_obj = transcribe_with_local_whisper(norm_path, model=(model or "small"))
            else:
                import whisper as _wh
                m = _wh.load_model(model or "small")
                transcript_obj = m.transcribe(norm_path)
        except Exception as e:
            logger.exception("Transcription failed: %s", e)
            raise

        transcript_text = transcript_obj.get("text") if isinstance(transcript_obj, dict) else str(transcript_obj)
        logger.info("Transcript length: %d chars", len(transcript_text or ""))

        if convert_text_to_labeled_blocks:
            labeled_blocks = convert_text_to_labeled_blocks(transcript_text)
        else:
            labeled_blocks = [{"text": transcript_text.strip(), "speaker": "unknown", "start": 0.0, "end": 0.0}]

        if extract_tasks_from_labeled_transcript:
            try:
                tasks = extract_tasks_from_labeled_transcript(labeled_blocks, use_llm_fallback=use_llm)
            except Exception as e:
                logger.warning("Hybrid extractor failed: %s", e)
                tasks = []

        if not tasks:
            logger.info("Using improved fallback NLP extractor.")
            if convert_text_to_labeled_blocks and extract_tasks_from_labeled_transcript:
                blocks = convert_text_to_labeled_blocks(transcript_text)
                tasks = extract_tasks_from_labeled_transcript(blocks, confidence_floor=0.8)
                logger.info("Improved fallback detected %d tasks", len(tasks))
            else:
                logger.warning("NLP fallback extractor unavailable.")

    logger.info("Found %d tasks", len(tasks))

    notion_token = os.environ.get("NOTION_TOKEN")
    notion_db = os.environ.get("NOTION_DATABASE_ID")
    if not notion_token or not notion_db:
        logger.warning("NOTION_TOKEN or NOTION_DATABASE_ID not set. Running in dry-run mode.")
        dry_run = True

    notion = None
    if not dry_run and NotionClient:
        notion = NotionClient(auth=notion_token)

    for t in tasks:
        sent = t.get("description") or t.get("sentence") or str(t)
        conf = t.get("confidence") or 0.0
        deadline = extract_deadline(sent)
        desc = f"Extracted with confidence {conf:.2f}\nOriginal: {sent}"
        push_task_to_notion(notion, notion_db, sent, confidence=conf, deadline=deadline, description=desc, dry_run=dry_run)


def cli(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(prog="crux.notion_sync", description="Transcribe audio, extract tasks, push to Notion")
    p.add_argument("--audio", "-a", required=False, help="path to audio/video file")
    p.add_argument("--backend", choices=["whisper", "openai"], default="whisper")
    p.add_argument("--model", default=None)
    p.add_argument("--dry-run", action="store_true", help="do everything but don't push to Notion")
    p.add_argument("--test", action="store_true", help="run hardcoded test transcript")
    p.add_argument("--llm", action="store_true", help="enable LLM fallback")
    args = p.parse_args(argv)
    run(audio_path=args.audio, backend=args.backend, model=args.model, dry_run=args.dry_run, test_mode=args.test, use_llm=args.llm)


if __name__ == "__main__":
    cli()
