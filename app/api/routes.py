"""HTTP endpoints. Each route is thin: validate input, call storage/repository/queue, shape the response.

The API never transcribes. It stores the file, creates a job and returns 202 immediately;
the worker does the slow part.
"""

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile

from app.api.dependencies import get_queue, get_repository, get_storage
from app.api.errors import ApiError
from app.api.schemas import ErrorInfo, JobResponse, Progress, SubmitResponse
from app.core.config import Settings, get_settings
from app.core.languages import LANGUAGE_CODES
from app.infra.db import Job
from app.engines.registry import model_name_for
from app.pipeline.exporters import EXPORTERS
from app.infra.jobqueue import JobQueue
from app.core.models import JobStatus, Transcript
from app.infra.repository import JobRepository
from app.infra.storage import FileTooLargeError, LocalStorage

router = APIRouter(prefix="/v1/transcriptions", tags=["transcriptions"])

# A quick pre-check only. The real check is ffprobe in the worker: if ffmpeg can decode it, we accept it.
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".oga", ".opus", ".flac", ".webm", ".aac", ".wma", ".mkv"}


@router.post("", status_code=202, response_model=SubmitResponse)
def submit(
    response: Response,
    file: UploadFile = File(...),
    language: str | None = Form(None, description="e.g. 'en'. Leave empty to auto-detect."),
    word_timestamps: bool = Form(False),
    settings: Settings = Depends(get_settings),
    storage: LocalStorage = Depends(get_storage),
    repo: JobRepository = Depends(get_repository),
    queue: JobQueue = Depends(get_queue),
) -> SubmitResponse:
    extension = Path(file.filename or "").suffix.lower()
    if extension not in AUDIO_EXTENSIONS and not (file.content_type or "").startswith(("audio/", "video/")):
        raise ApiError(415, "unsupported_media_type", "Upload an audio file such as WAV, MP3, M4A or OGG")
    language = (language or "").strip().lower() or None
    if language is not None and language not in LANGUAGE_CODES:
        # Reject bad input here, before storing anything, instead of letting the job fail in the worker.
        raise ApiError(422, "invalid_language", f"'{language}' is not a supported language code. Use e.g. 'en', or leave it empty to auto-detect.")
    if queue.depth() >= settings.max_queue_depth:
        # Backpressure: better to tell the client to come back later than to accept work we cannot do.
        raise ApiError(503, "queue_full", "Too many jobs waiting. Please retry later.")

    job_id = str(uuid4())
    try:
        stored = storage.save_upload(file.file, job_id, extension, settings.max_upload_bytes)
    except FileTooLargeError as exc:
        raise ApiError(413, "file_too_large", str(exc))

    engine = settings.transcriber_engine
    model = model_name_for(engine, settings)
    existing = repo.find_completed_duplicate(stored.sha256, engine, model, language, word_timestamps)
    if existing is not None:
        storage.delete(job_id)
        response.status_code = 200
        return SubmitResponse(job_id=existing.id, status=existing.status, deduplicated=True)

    repo.create(
        Job(
            id=job_id,
            status=JobStatus.QUEUED.value,
            original_filename=file.filename or "upload",
            audio_path=str(stored.path),
            file_hash=stored.sha256,
            file_size_bytes=stored.size,
            language_requested=language,
            word_timestamps=word_timestamps,
            engine=engine,
            model_name=model,
        )
    )
    queue.enqueue(job_id)
    return SubmitResponse(job_id=job_id, status=JobStatus.QUEUED.value)


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, repo: JobRepository = Depends(get_repository)) -> JobResponse:
    return to_response(_get_or_404(repo, job_id))


@router.get("/{job_id}/export")
def export(
    job_id: str,
    format: str = Query("txt", description="txt, srt, vtt or json"),
    repo: JobRepository = Depends(get_repository),
) -> Response:
    if format not in EXPORTERS:
        raise ApiError(400, "unsupported_format", f"format must be one of: {', '.join(EXPORTERS)}")
    job = _get_or_404(repo, job_id)
    if job.status != JobStatus.COMPLETED.value or job.transcript is None:
        raise ApiError(409, "not_completed", f"Job is {job.status}; export is available once it is completed")
    render, media_type = EXPORTERS[format]
    return Response(content=render(Transcript.model_validate(job.transcript)), media_type=media_type)


@router.post("/{job_id}/retry", status_code=202, response_model=SubmitResponse)
def retry(
    job_id: str, repo: JobRepository = Depends(get_repository), queue: JobQueue = Depends(get_queue)
) -> SubmitResponse:
    job = _get_or_404(repo, job_id)
    if job.status != JobStatus.FAILED.value:
        raise ApiError(409, "not_failed", f"Only failed jobs can be retried (job is {job.status})")
    repo.reset_attempts(job_id)
    repo.requeue(job_id, reason="Manual retry")
    queue.enqueue(job_id)  # resumes from any chunk checkpoints left on disk
    return SubmitResponse(job_id=job_id, status=JobStatus.QUEUED.value)


@router.delete("/{job_id}", status_code=204)
def delete(
    job_id: str, repo: JobRepository = Depends(get_repository), storage: LocalStorage = Depends(get_storage)
) -> Response:
    _get_or_404(repo, job_id)
    storage.delete(job_id)
    repo.delete(job_id)
    return Response(status_code=204)


def _get_or_404(repo: JobRepository, job_id: str) -> Job:
    job = repo.get(job_id)
    if job is None:
        raise ApiError(404, "not_found", f"No job with id {job_id}")
    return job


def to_response(job: Job) -> JobResponse:
    return JobResponse(
        job_id=job.id,
        status=job.status,
        original_filename=job.original_filename,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        attempts=job.attempts,
        engine=job.engine,
        model=job.model_name,
        device=job.device,
        progress=Progress(chunks_done=job.chunks_done, chunks_total=job.chunks_total),
        error=ErrorInfo(code=job.error_code, message=job.error_message) if job.error_message else None,
        transcript=Transcript.model_validate(job.transcript) if job.transcript else None,
    )
