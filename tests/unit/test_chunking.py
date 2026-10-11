from app.pipeline.chunking import plan_chunks


def test_short_audio_is_one_chunk():
    chunks = plan_chunks(duration=200, silences=[], target_sec=300, max_sec=450)
    assert [(c.start, c.end) for c in chunks] == [(0.0, 200)]


def test_cuts_fall_inside_silences():
    # 20 minutes of audio with a 2-second silence every 60 seconds
    silences = [(t, t + 2.0) for t in range(60, 1200, 60)]
    chunks = plan_chunks(duration=1200, silences=silences, target_sec=300, max_sec=450)

    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert any(s <= chunk.end <= e for s, e in silences), f"cut at {chunk.end} is not in a silence"


def test_chunks_cover_the_whole_file_without_gaps_or_overlap():
    silences = [(t, t + 1.0) for t in range(50, 2000, 50)]
    chunks = plan_chunks(duration=2000, silences=silences, target_sec=300, max_sec=450)

    assert chunks[0].start == 0.0
    assert chunks[-1].end == 2000
    for previous, current in zip(chunks, chunks[1:]):
        assert previous.end == current.start
    assert all(c.end - c.start <= 450 for c in chunks)


def test_hard_cut_when_there_is_no_silence():
    chunks = plan_chunks(duration=1000, silences=[], target_sec=300, max_sec=450)
    assert [(c.start, c.end) for c in chunks] == [(0.0, 450), (450, 900), (900, 1000)]
