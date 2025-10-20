from speaker_id import simple_window_diarization

def test_simple_window():
    wav = "tests/data/sample_short.wav"
    segs = simple_window_diarization(wav, window_sec=5.0)
    assert len(segs) >= 1
    assert "start" in segs[0] and "end" in segs[0] and "speaker" in segs[0]
