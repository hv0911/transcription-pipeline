"""The one interface every speech-to-text engine implements."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.core.models import TranscriptionOptions, TranscriptionResult


@dataclass
class EngineInfo:
    engine: str
    model: str
    device: str
    compute_type: str


class Transcriber(Protocol):
    name: str

    def load(self) -> None:
        """Load the model (or create an API client). Called once per worker process."""

    def info(self) -> EngineInfo:
        """Which engine, model and device are actually in use."""

    def transcribe(self, audio: Path, options: TranscriptionOptions) -> TranscriptionResult:
        """Transcribe a 16 kHz mono WAV file. Timestamps are relative to this file."""
