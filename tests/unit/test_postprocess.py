from app.core.models import Segment
from app.pipeline.postprocess import full_text, postprocess


def seg(text: str, start: float = 0.0, end: float = 1.0, confidence: float = 0.9, no_speech: float = 0.0) -> Segment:
    return Segment(start=start, end=end, text=text, confidence=confidence, no_speech_prob=no_speech)


def test_trims_whitespace_and_drops_empty_segments():
    out = postprocess([seg("  hello "), seg("   "), seg("")], confidence_threshold=0.5)
    assert [s.text for s in out] == ["hello"]


def test_collapses_repetition_loops():
    segments = [seg("Thank you.", 0, 1), seg("thank you", 1, 2), seg("Thank you!", 2, 3), seg("Bye", 3, 4)]
    out = postprocess(segments, confidence_threshold=0.5)
    assert [s.text for s in out] == ["Thank you.", "Bye"]
    assert (out[0].start, out[0].end) == (0, 3)  # the kept copy covers the whole repeated span


def test_drops_likely_hallucinations():
    out = postprocess([seg("real speech"), seg("ghost words", confidence=0.2, no_speech=0.9)], 0.5)
    assert [s.text for s in out] == ["real speech"]


def test_flags_low_confidence_and_renumbers_ids():
    out = postprocess([seg("", 0, 1), seg("sure", 1, 2, confidence=0.9), seg("maybe", 2, 3, confidence=0.3)], 0.5)
    assert [(s.id, s.text, s.low_confidence) for s in out] == [(0, "sure", False), (1, "maybe", True)]


def test_full_text_joins_segments():
    assert full_text([seg("Hello there."), seg("General Kenobi.")]) == "Hello there. General Kenobi."
