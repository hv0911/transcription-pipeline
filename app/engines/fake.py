"""Deterministic fake engine. Used by tests and to prove engines can be swapped with config only."""

import wave
from pathlib import Path

from app.engines.base import EngineInfo
from app.core.models import Segment, TranscriptionOptions, TranscriptionResult, Word

SEGMENT_SEC = 5.0


class FakeTranscriber:
    name = "fake"

    def load(self) -> None:
        pass  # nothing to load

    def info(self) -> EngineInfo:
        return EngineInfo(engine="fake", model="fake", device="cpu", compute_type="none")

    def transcribe(self, audio: Path, options: TranscriptionOptions) -> TranscriptionResult:
        with wave.open(str(audio)) as f:
            duration = f.getnframes() / f.getframerate()

        segments = []
        start = 0.0
        while start < duration:
            end = min(start + SEGMENT_SEC, duration)
            text = f"fake segment {len(segments) + 1}"
            words = [Word(start=start, end=end, text=text, probability=0.9)] if options.word_timestamps else None
            segments.append(Segment(start=start, end=end, text=text, confidence=0.9, words=words))
            start = end
        return TranscriptionResult(language=options.language or "en", language_probability=1.0, segments=segments)
