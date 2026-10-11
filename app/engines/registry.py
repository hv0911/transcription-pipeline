"""Pick an engine by name (env TRANSCRIBER_ENGINE). Adding an engine = one new file + one line here."""

from typing import Callable

from app.core.config import Settings
from app.engines.base import Transcriber
from app.engines.fake import FakeTranscriber
from app.engines.faster_whisper import FasterWhisperTranscriber

ENGINES: dict[str, Callable[[Settings], Transcriber]] = {
    "faster_whisper": lambda s: FasterWhisperTranscriber(
        s.whisper_model, s.whisper_device, s.whisper_compute_type, s.model_dir
    ),
    "fake": lambda s: FakeTranscriber(),
}


def build_transcriber(name: str, settings: Settings) -> Transcriber:
    if name not in ENGINES:
        raise ValueError(f"Unknown TRANSCRIBER_ENGINE '{name}'. Available: {', '.join(sorted(ENGINES))}")
    return ENGINES[name](settings)


def model_name_for(name: str, settings: Settings) -> str:
    """The model label stored on jobs, used to decide whether a cached result can be reused."""
    return settings.whisper_model if name == "faster_whisper" else name
