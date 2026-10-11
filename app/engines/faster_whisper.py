"""Real engine: Whisper via faster-whisper (CTranslate2). Runs on CPU or GPU."""

import logging
import math
from pathlib import Path

from app.core import errors
from app.core.errors import PermanentError
from app.core.languages import LANGUAGE_CODES
from app.engines.base import EngineInfo
from app.engines.device import cuda_device_count, resolve_compute_type, resolve_device
from app.core.models import Segment, TranscriptionOptions, TranscriptionResult, Word

log = logging.getLogger(__name__)


class FasterWhisperTranscriber:
    name = "faster_whisper"

    def __init__(self, model: str, device: str, compute_type: str, model_dir: str) -> None:
        self._model_name = model
        self._requested_device = device
        self._requested_compute_type = compute_type
        self._model_dir = model_dir
        self._model = None
        self._device = ""
        self._compute_type = ""

    def load(self) -> None:
        from faster_whisper import WhisperModel  # heavy import, only needed in the worker

        device = resolve_device(self._requested_device, cuda_device_count())
        compute_type = resolve_compute_type(self._requested_compute_type, device)
        try:
            self._model = WhisperModel(
                self._model_name, device=device, compute_type=compute_type, download_root=self._model_dir
            )
        except Exception:
            # "auto" means "use the GPU if it works". If the user forced cuda, fail loudly instead.
            if device != "cuda" or self._requested_device != "auto":
                raise
            log.warning("GPU model load failed, falling back to CPU", exc_info=True)
            device, compute_type = "cpu", resolve_compute_type("auto", "cpu")
            self._model = WhisperModel(
                self._model_name, device=device, compute_type=compute_type, download_root=self._model_dir
            )
        self._device, self._compute_type = device, compute_type

    def info(self) -> EngineInfo:
        return EngineInfo(self.name, self._model_name, self._device, self._compute_type)

    def transcribe(self, audio: Path, options: TranscriptionOptions) -> TranscriptionResult:
        if self._model is None:
            raise RuntimeError("load() must be called before transcribe()")
        if options.language and options.language not in LANGUAGE_CODES:
            # A bad option will fail on every attempt, so mark it permanent: no pointless retries.
            raise PermanentError(errors.INVALID_OPTIONS, f"'{options.language}' is not a supported language code")
        segments, info = self._model.transcribe(
            str(audio),
            language=options.language,
            beam_size=options.beam_size,
            word_timestamps=options.word_timestamps,
            vad_filter=True,  # skip silence inside the chunk; reduces hallucinations
        )
        # faster-whisper returns a lazy generator: the real work happens while we iterate.
        # Its types never leave this file; we convert to our own domain types here.
        return TranscriptionResult(
            language=info.language,
            language_probability=info.language_probability,
            segments=[_to_segment(s) for s in segments],
        )


def _to_segment(s) -> Segment:
    words = None
    if s.words:
        words = [Word(start=w.start, end=w.end, text=w.word.strip(), probability=w.probability) for w in s.words]
    return Segment(
        start=s.start,
        end=s.end,
        text=s.text,
        confidence=math.exp(s.avg_logprob),
        no_speech_prob=s.no_speech_prob,
        words=words,
    )
