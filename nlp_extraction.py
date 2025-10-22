import spacy
from transformers import pipeline
from typing import List, Dict, Optional
import os  # For potential OpenAI key

# Load spaCy English model once
nlp = spacy.load("en_core_web_sm")

# Load zero-shot classification pipeline for task detection
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["work task", "personal/irrelevant", "neutral/info"]

# Optional: LLM fallback pipeline (text generation for classification)
# llm_fallback = pipeline("text-generation", model="gpt2")  # Local, no API needed; or use OpenAI below
# For OpenAI (uncomment if you have OPENAI_API_KEY in .env):
# from openai import OpenAI
# llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

def convert_text_to_labeled_blocks(transcript_text: str) -> List[Dict]:
    """
    Use spaCy to split transcript text into speaker-agnostic labeled blocks.
    """
    doc = nlp(transcript_text)
    blocks = []
    start_char = 0
    for sent in doc.sents:
        text = sent.text.strip()
        if not text:
            continue
        end_char = start_char + len(text)
        blocks.append({
            "start": start_char,
            "end": end_char,
            "speaker": "unknown",  # No speaker diarization here
            "text": text
        })
        start_char = end_char + 1  # Include space/newline
    return blocks

def extract_tasks_from_labeled_transcript(
    labeled_blocks: List[Dict], 
    confidence_floor: float = 0.8,
    use_llm_fallback: bool = False  # <-- Added: Accepts the arg, defaults to False
) -> List[Dict]:
    """
    Use zero-shot classifier on each block to detect 'work task' sentences.
    If use_llm_fallback=True and no tasks found, retry with lower threshold or LLM.
    Returns list of detected tasks with confidence.
    """
    tasks = []
    for block in labeled_blocks:
        text = block.get("text", "")
        if not text:
            continue
        result = classifier(text, candidate_labels)
        label = result["labels"][0]
        score = result["scores"][0]
        if label == "work task" and score >= confidence_floor:
            tasks.append({
                "sentence": text,
                "confidence": score,
                "speaker": block.get("speaker", "unknown")
            })
    
    # Fallback logic if no tasks and use_llm_fallback enabled
    if not tasks and use_llm_fallback:
        print("Applying fallback: Retrying with lower confidence_floor=0.5")  # Use logger in prod
        for block in labeled_blocks:
            text = block.get("text", "")
            if not text:
                continue
            result = classifier(text, candidate_labels)
            label = result["labels"][0]
            score = result["scores"][0]
            if label == "work task" and score >= 0.5:  # Lowered threshold
                tasks.append({
                    "sentence": text,
                    "confidence": score,
                    "speaker": block.get("speaker", "unknown")
                })
        
        # Optional: True LLM retry for remaining low-conf (uncomment if desired)
        # if not tasks and llm_client:  # Or use local llm_fallback
        #     for block in labeled_blocks:
        #         text = block.get("text", "")
        #         if not text:
        #             continue
        #         response = llm_client.chat.completions.create(
        #             model="gpt-3.5-turbo",
        #             messages=[{"role": "user", "content": f"Is '{text}' a work task? Respond 'yes' or 'no'."}]
        #         )
        #         if "yes" in response.choices[0].message.content.lower():
        #             tasks.append({
        #                 "sentence": text,
        #                 "confidence": 0.75,  # Arbitrary for LLM
        #                 "speaker": block.get("speaker", "unknown")
        #             })
        # elif not tasks:
        #     print("LLM fallback unavailable (no API key).")
    
    return tasks