"""Domain models: plain data, no database or framework code.

Every engine converts its own output into these types, so the rest of the system
never depends on a specific speech-to-text library.
"""

from enum import Enum

from pydantic import BaseModel


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Word(BaseModel):
    start: float
    end: float
    text: str
    probability: float


class Segment(BaseModel):
    id: int = 0
    start: float
    end: float
    text: str
    confidence: float  # 0..1, exp(avg_logprob) for Whisper
    no_speech_prob: float = 0.0
    low_confidence: bool = False
    words: list[Word] | None = None


class TranscriptionOptions(BaseModel):
    language: str | None = None  # None = auto-detect
    word_timestamps: bool = False
    beam_size: int = 5


class TranscriptionResult(BaseModel):
    """What any engine returns. Timestamps are relative to the audio it was given."""

    language: str
    language_probability: float
    segments: list[Segment]


class Transcript(BaseModel):
    """The final result stored for a job."""

    language: str
    duration_sec: float
    text: str
    segments: list[Segment]
    engine: str
    model: str
