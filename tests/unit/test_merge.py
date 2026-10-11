from app.core.models import Segment, TranscriptionResult, Word
from app.pipeline.chunking import Chunk
from app.pipeline.merge import merge


def result(*segments: Segment) -> TranscriptionResult:
    return TranscriptionResult(language="en", language_probability=1.0, segments=list(segments))


def test_offsets_are_added_to_segment_and_word_timestamps():
    first = result(Segment(start=0.0, end=2.0, text="hello", confidence=0.9))
    second = result(
        Segment(
            start=1.0,
            end=3.0,
            text="world",
            confidence=0.9,
            words=[Word(start=1.0, end=3.0, text="world", probability=0.9)],
        )
    )
    merged = merge([(Chunk(0, 0.0, 300.0), first), (Chunk(1, 300.0, 600.0), second)])

    assert [(s.start, s.end) for s in merged] == [(0.0, 2.0), (301.0, 303.0)]
    assert merged[1].words[0].start == 301.0
    assert merged[1].words[0].end == 303.0


def test_chunks_are_merged_in_order_even_if_given_out_of_order():
    a = result(Segment(start=0.0, end=1.0, text="a", confidence=0.9))
    b = result(Segment(start=0.0, end=1.0, text="b", confidence=0.9))
    merged = merge([(Chunk(1, 100.0, 200.0), b), (Chunk(0, 0.0, 100.0), a)])
    assert [s.text for s in merged] == ["a", "b"]
