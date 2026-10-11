"""Stitch per-chunk results back into one timeline."""

from app.core.models import Segment, TranscriptionResult
from app.pipeline.chunking import Chunk


def merge(results: list[tuple[Chunk, TranscriptionResult]]) -> list[Segment]:
    """Shift each chunk's timestamps by the chunk's start offset and concatenate in order."""
    merged: list[Segment] = []
    for chunk, result in sorted(results, key=lambda pair: pair[0].index):
        offset = chunk.start
        for seg in result.segments:
            words = None
            if seg.words is not None:
                words = [
                    w.model_copy(update={"start": w.start + offset, "end": w.end + offset}) for w in seg.words
                ]
            merged.append(seg.model_copy(update={"start": seg.start + offset, "end": seg.end + offset, "words": words}))
    return merged
