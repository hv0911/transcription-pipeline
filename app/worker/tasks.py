"""The job the worker runs for each upload."""

import logging
from functools import lru_cache
from pathlib import Path

from rq import get_current_job

from app.core import errors
from app.core.config import get_settings
from app.engines.base import Transcriber
from app.engines.registry import build_transcriber
from app.core.errors import PermanentError
from app.core.models import JobStatus, TranscriptionOptions
from app.pipeline.runner import run_pipeline
from app.infra.repository import JobRepository
from app.infra.storage import LocalStorage

log = logging.getLogger(__name__)


@lru_cache
def get_transcriber() -> Transcriber:
    """Load the model once per worker process, not once per job (loading takes seconds)."""
    settings = get_settings()
    transcriber = build_transcriber(settings.transcriber_engine, settings)
    transcriber.load()
    return transcriber


def process_job(job_id: str) -> None:
    settings = get_settings()
    repo = JobRepository()
    storage = LocalStorage(settings.data_dir)

    job = repo.get(job_id)
    if job is None or job.status == JobStatus.COMPLETED.value:
        log.info("job=%s skipped (deleted or already completed)", job_id)  # safe to run twice
        return

    transcriber = get_transcriber()
    repo.mark_processing(job_id, transcriber.info())
    log.info("job=%s started attempt=%d", job_id, job.attempts + 1)
    options = TranscriptionOptions(
        language=job.language_requested, word_timestamps=job.word_timestamps, beam_size=settings.beam_size
    )

    try:
        transcript = run_pipeline(
            Path(job.audio_path),
            storage.workdir(job_id),
            transcriber,
            options,
            settings,
            progress=lambda done, total: repo.update_progress(job_id, done, total),
        )
    except PermanentError as exc:
        # Bad input: fail now, do NOT raise, so RQ does not retry.
        log.warning("job=%s failed permanently code=%s", job_id, exc.code)
        repo.fail(job_id, exc.code, exc.message)
        storage.clean_workdir(job_id)
        return
    except Exception as exc:
        # Transient: re-raise so RQ retries. The chunk checkpoints stay on disk,
        # so the next attempt resumes where this one stopped.
        if _is_last_attempt():
            log.exception("job=%s failed after all retries", job_id)
            repo.fail(job_id, errors.TRANSCRIPTION_FAILED, str(exc))
        else:
            log.warning("job=%s attempt failed, will retry: %s", job_id, exc)
            repo.requeue(job_id, reason=str(exc))
        raise

    repo.complete(job_id, transcript)
    storage.clean_workdir(job_id)
    log.info("job=%s completed segments=%d", job_id, len(transcript.segments))


def _is_last_attempt() -> bool:
    rq_job = get_current_job()
    return rq_job is None or not rq_job.retries_left
