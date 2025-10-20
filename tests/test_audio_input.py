import os
from audio_input import validate_audio_file, normalize_audio_to_wav, load_audio_to_numpy

SAMPLE = "tests/data/sample_short.mp3"

def test_validate_sample():
    info = validate_audio_file(SAMPLE)
    assert "duration_seconds" in info
    assert info["duration_seconds"] > 0

def test_normalize_and_load(tmp_path):
    out = tmp_path / "norm.wav"
    norm = normalize_audio_to_wav(SAMPLE, str(out), target_sr=16000)
    data, sr = load_audio_to_numpy(norm)
    assert sr == 16000
    assert data.size > 0
