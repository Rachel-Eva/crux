# `dashboard_ui.py` (Streamlit)
"""python

dashboard_ui.py

Streamlit UI for CRUX:
- Upload audio file or pick an example
- Run pipeline (calls crux_main.process_audio_file)
- Display transcript, speaker-labeled blocks, and extracted tasks
- "Sync to Notion" button (calls a stub function or notion_sync when available)

How to run:
streamlit run dashboard_ui.py
"""
import streamlit as st
import os
import json
from pathlib import Path
from typing import Any

# Try to import the pipeline
try:
    from crux_main import process_audio_file
except Exception:
    # If you're running this file directly inside main/ folder, adapt import:
    try:
        from main.crux_main import process_audio_file  # optional path
    except Exception:
        process_audio_file = None

st.set_page_config(page_title="CRUX Dashboard", layout="wide")

st.title("CRUX — Meeting Minutes → Notion")

st.sidebar.header("Settings")
stt_backend = st.sidebar.selectbox("STT backend", options=["local", "openai"], index=0)
diarize_backend = st.sidebar.selectbox("Diarization backend", options=["simple", "pyannote"], index=0)
use_llm_nlp = st.sidebar.checkbox("Use LLM fallback for NLP", value=False)

st.markdown("Upload an audio file (mp3/wav) or choose an example in `assets/sample_audio/`.")

uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"], accept_multiple_files=False)

col1, col2 = st.columns([1, 1])

if uploaded:
    # save uploaded file to a temp path
    temp_dir = Path(".crux_tmp")
    temp_dir.mkdir(exist_ok=True)
    src_path = str(temp_dir / uploaded.name)
    with open(src_path, "wb") as fh:
        fh.write(uploaded.getbuffer())
    st.success(f"Saved to {src_path}")
elif Path("assets/sample_audio").exists():
    examples = list(Path("assets/sample_audio").glob("*.*"))
    if examples:
        sel = st.selectbox("Or pick an example file", options=[str(x) for x in examples])
        src_path = sel
    else:
        src_path = None
else:
    src_path = None

if src_path and st.button("Run CRUX pipeline"):
    if process_audio_file is None:
        st.error("Pipeline not available — ensure crux_main.py is importable from this file's location.")
    else:
        with st.spinner("Processing audio..."):
            try:
                meeting = process_audio_file(src_path, stt_backend=stt_backend, diarize_backend=diarize_backend, use_llm_for_nlp=use_llm_nlp)
                st.success("Processing complete.")
                # Show outputs
                st.subheader("Transcript")
                st.text_area("Full transcript", value=meeting.get("Transcript",""), height=200)

                st.subheader("Speaker-labeled blocks")
                blocks = meeting.get("LabeledBlocks", [])
                for b in blocks:
                    st.markdown(f"**{b.get('speaker','UNKNOWN')}** [{b.get('start'):.1f}s - {b.get('end'):.1f}s]")
                    st.write(b.get("text",""))

                st.subheader("Extracted tasks")
                tasks = meeting.get("Tasks", [])
                if tasks:
                    for t in tasks:
                        st.write(f"- **{t.get('description')}** — assignee: {t.get('assignee') or 'Unassigned'} — priority: {t.get('priority') or 'N/A'} — deadline: {t.get('deadline') or 'N/A'}")
                else:
                    st.info("No tasks extracted.")

                # Save JSON
                out_path = Path(".crux_outputs")
                out_path.mkdir(exist_ok=True)
                fname = out_path / f"{Path(src_path).stem}_meeting.json"
                with open(fname, "w", encoding="utf8") as fh:
                    json.dump(meeting, fh, indent=2)
                st.write(f"Saved meeting JSON to `{fname}`")

                # Sync button (stub)
                if st.button("Sync to Notion"):
                    st.info("Notion sync not implemented in this demo. Implement notion_sync.py and call it here.")

            except Exception as e:
                st.error(f"Processing failed: {e}")
