"""Split long audio into chunks, always cutting inside a silence.

Because every cut falls in silence, no word is split in half. That means chunks do not need
to overlap, and we never have to de-duplicate text at the boundaries.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float
    end: float


def plan_chunks(
    duration: float,
    silences: list[tuple[float, float]],
    target_sec: float,
    max_sec: float,
) -> list[Chunk]:
    bounds: list[tuple[float, float]] = []
    start = 0.0
    while duration - start > max_sec:
        ideal = start + target_sec
        # Candidate cut points: the middle of each silence that is not too close to the
        # chunk start (avoid tiny chunks) and not past the hard limit.
        candidates = [
            (s + e) / 2
            for s, e in silences
            if start + target_sec / 2 < (s + e) / 2 <= start + max_sec
        ]
        if candidates:
            cut = min(candidates, key=lambda c: abs(c - ideal))
        else:
            cut = start + max_sec  # no silence found: hard cut (rare, e.g. continuous music)
        bounds.append((start, cut))
        start = cut
    bounds.append((start, duration))
    return [Chunk(index=i, start=s, end=e) for i, (s, e) in enumerate(bounds)]
