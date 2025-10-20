"""
nlp_extraction.py

Responsibilities
----------------
- Extract structured tasks from meeting transcripts
  (task description, assignee, priority, deadline, confidence)
- Use spaCy for NER (PERSON, DATE, TIME) and rule-based extraction
- Use dateparser for parsing natural-language deadlines
- Provide an LLM fallback (OpenAI) to parse ambiguous sentences into structured tasks
- Export functions for integration:
    - extract_tasks_from_labeled_transcript(labeled_blocks, use_llm_fallback=False)
    - extract_entities_from_text(text)

Inputs
------
- labeled_blocks: list of dicts: {"start": float, "end": float, "speaker": str, "text": str}
- use_llm_fallback: bool — if true, use OpenAI to parse unclear sentences (requires OPENAI_API_KEY)

Outputs
-------
- tasks: list of dicts:
    {
      "description": str,
      "assignee": str | None,
      "priority": "critical"|"non-critical"|"follow-up"|None,
      "deadline": "YYYY-MM-DD" | None,
      "source": { "speaker": str, "start": float, "end": float },
      "confidence": float  # 0..1 heuristic
    }

Dependencies
------------
- spacy (en_core_web_sm or en_core_web_trf)
- dateparser
- python-dateutil
- openai (optional, for LLM fallback)

Quick usage (CLI)
-----------------
python nlp_extraction.py --demo
"""
from __future__ import annotations
import os
import re
from typing import List, Dict, Optional, Any
import logging
import json
from datetime import datetime

# External libs
try:
    import spacy
except Exception:
    spacy = None

try:
    import dateparser
except Exception:
    dateparser = None

# Optional LLM fallback
try:
    import openai
except Exception:
    openai = None

# Initialize logging
logger = logging.getLogger("nlp_extraction")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)


class NLPExtractionError(Exception):
    pass


# Load spaCy model lazily
_SPACY_MODEL = None


def _get_spacy_model(model_name: str = "en_core_web_sm"):
    global _SPACY_MODEL
    if _SPACY_MODEL is None:
        if spacy is None:
            raise NLPExtractionError("spaCy is not installed. pip install spacy and a model such as en_core_web_sm")
        try:
            _SPACY_MODEL = spacy.load(model_name)
        except Exception as e:
            raise NLPExtractionError(f"Failed to load spaCy model '{model_name}': {e}")
    return _SPACY_MODEL


def extract_entities_from_text(text: str) -> Dict[str, List[str]]:
    """
    Run spaCy NER and return basic entities:
    { "PERSON": [...], "DATE": [...], "TIME": [...], "ORG": [...], "GPE": [...] }
    """
    nlp = _get_spacy_model()
    doc = nlp(text)
    out = {}
    for ent in doc.ents:
        out.setdefault(ent.label_, []).append(ent.text)
    return out


PRIORITY_KEYWORDS = {
    "critical": ["urgent", "critical", "asap", "immediately", "priority"],
    "non-critical": ["sometime", "whenever", "nice to have", "low priority"],
    "follow-up": ["follow-up", "follow up", "check in", "remind", "next"]
}


def _detect_priority(text: str) -> Optional[str]:
    tl = text.lower()
    for p, kws in PRIORITY_KEYWORDS.items():
        for kw in kws:
            if kw in tl:
                return p
    return None


def _parse_deadline(text: str) -> Optional[str]:
    """
    Try to parse a deadline using dateparser. Return ISO date string (YYYY-MM-DD) or None.
    """
    if dateparser is None:
        raise NLPExtractionError("dateparser not installed. pip install dateparser")
    settings = {"PREFER_DATES_FROM": "future", "RELATIVE_BASE": datetime.now()}
    parsed = dateparser.parse(text, settings=settings)
    if not parsed:
        return None
    return parsed.date().isoformat()


# Heuristic to find assignees from a block using NER PERSON or patterns like 'Bob will' / 'Alice to'
ASSIGNEE_PATTERNS = [
    r"(?P<person>[A-Z][a-z]+)\s+will\s+(?P<rest>.+)",
    r"(?P<person>[A-Z][a-z]+)\s+to\s+(?P<rest>.+)",
    r"assign to\s+(?P<person>[A-Z][a-z]+)",
    r"(?P<person>[A-Z][a-z]+)\s+can\s+(?P<rest>.+)",
    r"(?P<person>[A-Z][a-z]+)\s+is\s+going\s+to\s+(?P<rest>.+)",
]


def _heuristic_assignee(text: str, entities: Dict[str, List[str]]) -> Optional[str]:
    # Prefer PERSON entities
    persons = entities.get("PERSON", [])
    if persons:
        # choose first capitalized single-word person-like token
        for p in persons:
            # simple filter
            if len(p.split()) <= 3:
                return p.split(",")[0]
        return persons[0]
    # Try regex patterns
    for pat in ASSIGNEE_PATTERNS:
        m = re.search(pat, text)
        if m and m.groupdict().get("person"):
            return m.group("person")
    return None


def _llm_parse_task(block_text: str, openai_model: str = "gpt-4o-mini") -> List[Dict[str, Any]]:
    """
    Use an LLM (OpenAI) to parse one block into structured tasks.
    Returns list of tasks (may be empty). Requires OPENAI_API_KEY in env.
    """
    if openai is None:
        raise NLPExtractionError("OpenAI SDK not installed for LLM fallback. pip install openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise NLPExtractionError("OPENAI_API_KEY not set for LLM fallback")
    openai.api_key = api_key

    prompt = (
        "Extract actionable tasks from the following meeting transcript block.\n"
        "Output JSON list of objects with fields: description, assignee (or null), priority (critical/non-critical/follow-up/null), "
        "deadline (YYYY-MM-DD or null). Be conservative and only include true tasks.\n\n"
        "Transcript block:\n\"\"\"\n" + block_text + "\n\"\"\"\n\nOutput only JSON."
    )
    try:
        resp = openai.ChatCompletion.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        # Defensive parsing: check structure
        content = resp["choices"][0]["message"]["content"]
        # Try to find first JSON structure in content
        import json, re as _re
        m = _re.search(r"(\[.*\])", content, flags=_re.DOTALL)
        json_text = m.group(1) if m else content
        tasks = json.loads(json_text)
        return tasks
    except Exception as e:
        logger.warning("LLM fallback failed: %s", e)
        return []


def extract_tasks_from_labeled_transcript(
    labeled_blocks: List[Dict[str, Any]],
    use_llm_fallback: bool = False,
    llm_model: str = "gpt-4o-mini"
) -> List[Dict[str, Any]]:
    """
    Main entrypoint: turn speaker-labeled transcript blocks into a list of tasks.

    Strategy:
    - For each block, run NER + heuristics:
        - find sentences that look like tasks (verb phrases, 'we need to', 'please', 'action', etc.)
        - extract assignee via PERSON entities or regex heuristics
        - detect priority via keywords
        - detect deadline via dateparser on the same sentence or nearby words
    - If use_llm_fallback is True and heuristics fail, call _llm_parse_task for that block.

    Returns list of task dicts (may be empty).
    """
    tasks = []
    nlp = _get_spacy_model()
    for block in labeled_blocks:
        text = block.get("text", "").strip()
        if not text:
            continue
        # run spaCy once
        doc = nlp(text)
        entities = {}
        for ent in doc.ents:
            entities.setdefault(ent.label_, []).append(ent.text)

        # naive sentence-level scanning: find sentences likely to contain tasks
        for sent in doc.sents:
            s_text = sent.text.strip()
            if len(s_text) < 3:
                continue
            # heuristic: sentences with verbs + actionable keywords
            if re.search(r"\b(need to|please|shall we|let's|should|assign|action|deadline|due|deliver|finish|complete|implement|create|build|prepare)\b", s_text, flags=re.I) \
               or re.search(r"\b(will|to do|to-?do)\b", s_text, flags=re.I):
                # Extract assignee heuristic
                sent_entities = {}
                for ent in sent.ents:
                    sent_entities.setdefault(ent.label_, []).append(ent.text)
                assignee = _heuristic_assignee(s_text, sent_entities or entities)
                priority = _detect_priority(s_text)
                deadline = _parse_deadline(s_text) or _parse_deadline(text)  # check whole block if not in sentence
                description = s_text
                confidence = 0.7
                tasks.append({
                    "description": description,
                    "assignee": assignee,
                    "priority": priority,
                    "deadline": deadline,
                    "source": {"speaker": block.get("speaker"), "start": block.get("start"), "end": block.get("end")},
                    "confidence": confidence
                })
        # If no tasks detected heuristically and fallback enabled, try LLM
        if use_llm_fallback:
            heur_count = sum(1 for t in tasks if t["source"]["start"] == block.get("start") and t["source"]["end"] == block.get("end"))
            if heur_count == 0:
                try:
                    llm_tasks = _llm_parse_task(text, openai_model=llm_model)
                    for lt in llm_tasks:
                        # Normalize and validate fields; be defensive
                        desc = lt.get("description") or lt.get("task") or ""
                        assignee = lt.get("assignee")
                        priority = lt.get("priority")
                        dl = lt.get("deadline")
                        # try parse deadline to ISO
                        try:
                            if dl:
                                dl_iso = _parse_deadline(dl) or dl
                            else:
                                dl_iso = None
                        except Exception:
                            dl_iso = None
                        tasks.append({
                            "description": desc,
                            "assignee": assignee,
                            "priority": priority,
                            "deadline": dl_iso,
                            "source": {"speaker": block.get("speaker"), "start": block.get("start"), "end": block.get("end")},
                            "confidence": 0.6
                        })
                except NLPExtractionError as e:
                    logger.warning("LLM parse skipped: %s", e)
    # basic deduplication (by description)
    seen = set()
    deduped = []
    for t in tasks:
        key = (t["description"].strip().lower(), (t.get("assignee") or "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)
    return deduped


# Simple CLI demo
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Demo NLP extraction on a sample labeled transcript (JSON).")
    parser.add_argument("--demo", action="store_true", help="Run a small demo")
    parser.add_argument("--input", type=str, help="Path to JSON file containing labeled blocks")
    parser.add_argument("--llm", action="store_true", help="Use LLM fallback (requires OPENAI_API_KEY)")
    args = parser.parse_args()

    if args.demo:
        sample = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 8.0, "text": "Vibhu will finalize the API endpoints by next Tuesday. Also we need to update the README."},
            {"speaker": "SPEAKER_01", "start": 8.0, "end": 15.0, "text": "Can someone prepare the demo slides? It's low priority."},
        ]
        out = extract_tasks_from_labeled_transcript(sample, use_llm_fallback=args.llm)
        print(json.dumps(out, indent=2))
    elif args.input:
        with open(args.input, "r", encoding="utf8") as fh:
            labeled = json.load(fh)
        out = extract_tasks_from_labeled_transcript(labeled, use_llm_fallback=args.llm)
        print(json.dumps(out, indent=2))
    else:
        parser.print_help()
