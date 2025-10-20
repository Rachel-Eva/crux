from speech_to_text import transcribe_with_local_whisper

def test_local_whisper_basic():
    # Optional test: requires local whisper and model availability.
    wav = "tests/data/sample_short.wav"
    out = transcribe_with_local_whisper(wav, model="tiny")
    assert "text" in out
    assert isinstance(out["text"], str)
