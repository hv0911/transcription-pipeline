"""The transcription pipeline: validate -> normalize -> chunk -> transcribe -> merge -> postprocess.

It knows nothing about HTTP, databases or queues. It takes a file and returns a Transcript,
and reports progress through a callback. That is why the same code runs from the worker
(with a DB) and from the CLI (with nothing at all).
"""

import logging
from pathlib import Path
from typing import Callable

from app.core import errors
from app.pipeline import audio
from app.core.config import Settings
from app.engines.base import Transcriber
from app.core.errors import PermanentError
from app.core.models import Transcript, TranscriptionOptions, TranscriptionResult
from app.pipeline.chunking import Chunk, plan_chunks
from app.pipeline.merge import merge
from app.pipeline.postprocess import full_text, postprocess

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int], None]  # (chunks_done, chunks_total)


def run_pipeline(
    source: Path,
    workdir: Path,
    transcriber: Transcriber,
    options: TranscriptionOptions,
    settings: Settings,
    progress: ProgressFn = lambda done, total: None,
) -> Transcript:
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. validate: is this real audio, and is it within limits?
    duration = validate(source, settings.max_duration_sec)

    # 2. normalize: any format -> 16 kHz mono WAV
    wav = workdir / "audio.wav"
    audio.to_wav16k_mono(source, wav)

    # 3. chunk: long files are split at silences
    silences = audio.find_silences(wav)
    if sum(e - s for s, e in silences) >= duration * 0.98:
        raise PermanentError(errors.NO_SPEECH, "The file contains only silence")
    if duration > settings.chunk_threshold_sec:
        chunks = plan_chunks(duration, silences, settings.chunk_target_sec, settings.chunk_max_sec)
    else:
        chunks = [Chunk(index=0, start=0.0, end=duration)]
    log.info("pipeline source=%s duration=%.1fs chunks=%d", source.name, duration, len(chunks))

    # 4. transcribe each chunk, saving a checkpoint after each one
    results: list[tuple[Chunk, TranscriptionResult]] = []
    for chunk in chunks:
        results.append((chunk, _transcribe_chunk(chunk, wav, workdir, transcriber, options, len(chunks))))
        progress(len(results), len(chunks))

    # 5. merge chunk results into one timeline, 6. clean them up
    segments = postprocess(merge(results), settings.confidence_threshold)

    info = transcriber.info()
    return Transcript(
        language=options.language or results[0][1].language,
        duration_sec=round(duration, 3),
        text=full_text(segments),
        segments=segments,
        engine=info.engine,
        model=info.model,
    )


def validate(source: Path, max_duration_sec: int) -> float:
    info = audio.probe(source)
    if not info.has_audio:
        raise PermanentError(errors.NO_AUDIO_STREAM, "The file has no audio stream")
    if info.duration_sec <= 0:
        raise PermanentError(errors.EMPTY_AUDIO, "The audio is empty")
    if info.duration_sec > max_duration_sec:
        raise PermanentError(errors.TOO_LONG, f"Audio is longer than the {max_duration_sec}s limit")
    return info.duration_sec


def _transcribe_chunk(
    chunk: Chunk,
    wav: Path,
    workdir: Path,
    transcriber: Transcriber,
    options: TranscriptionOptions,
    total: int,
) -> TranscriptionResult:
    checkpoint = workdir / "chunks" / f"{chunk.index:04d}.json"
    if checkpoint.exists():
        # A previous attempt already did this chunk (the worker crashed or the job was retried).
        log.info("chunk %d/%d loaded from checkpoint", chunk.index + 1, total)
        return TranscriptionResult.model_validate_json(checkpoint.read_text())

    if total == 1:
        chunk_wav = wav  # no need to copy the audio for a single chunk
    else:
        chunk_wav = workdir / f"chunk_{chunk.index:04d}.wav"
        audio.cut(wav, chunk_wav, chunk.start, chunk.end)

    result = transcriber.transcribe(chunk_wav, options)

    checkpoint.parent.mkdir(exist_ok=True)
    tmp = checkpoint.with_suffix(".tmp")
    tmp.write_text(result.model_dump_json())
    tmp.replace(checkpoint)  # atomic: a crash never leaves a half-written checkpoint
    if chunk_wav != wav:
        chunk_wav.unlink(missing_ok=True)
    log.info("chunk %d/%d transcribed (%.0fs-%.0fs)", chunk.index + 1, total, chunk.start, chunk.end)
    return result
