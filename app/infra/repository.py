"""All reads and writes of job records. Each method uses its own short transaction."""

from datetime import datetime

from sqlalchemy import select

from app.infra.db import Job, SessionLocal, utcnow
from app.engines.base import EngineInfo
from app.core.models import JobStatus, Transcript


class JobRepository:
    def create(self, job: Job) -> Job:
        with SessionLocal.begin() as session:
            session.add(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with SessionLocal() as session:
            return session.get(Job, job_id)

    def find_completed_duplicate(
        self, file_hash: str, engine: str, model_name: str, language: str | None, word_timestamps: bool
    ) -> Job | None:
        """Idempotency: the same file with the same engine, model and options gives the same result."""
        with SessionLocal() as session:
            return session.scalars(
                select(Job)
                .where(Job.file_hash == file_hash, Job.engine == engine, Job.model_name == model_name)
                .where(Job.language_requested.is_(language) if language is None else Job.language_requested == language)
                .where(Job.word_timestamps == word_timestamps, Job.status == JobStatus.COMPLETED.value)
                .order_by(Job.created_at.desc())
                .limit(1)
            ).first()

    def find_stale_processing(self, older_than: datetime) -> list[Job]:
        """Jobs stuck in 'processing' with no progress: their worker most likely died."""
        with SessionLocal() as session:
            return list(
                session.scalars(
                    select(Job).where(Job.status == JobStatus.PROCESSING.value, Job.updated_at < older_than)
                )
            )

    def mark_processing(self, job_id: str, info: EngineInfo) -> None:
        self._update(
            job_id,
            status=JobStatus.PROCESSING.value,
            started_at=utcnow(),
            engine=info.engine,
            model_name=info.model,
            device=info.device,
            compute_type=info.compute_type,
            error_code=None,
            error_message=None,
            increment_attempts=True,
        )

    def update_progress(self, job_id: str, done: int, total: int) -> None:
        self._update(job_id, chunks_done=done, chunks_total=total)

    def complete(self, job_id: str, transcript: Transcript) -> None:
        self._update(
            job_id,
            status=JobStatus.COMPLETED.value,
            finished_at=utcnow(),
            duration_sec=transcript.duration_sec,
            language_detected=transcript.language,
            transcript=transcript.model_dump(),
        )

    def fail(self, job_id: str, code: str, message: str) -> None:
        self._update(
            job_id, status=JobStatus.FAILED.value, finished_at=utcnow(), error_code=code, error_message=message[:2000]
        )

    def requeue(self, job_id: str, reason: str | None = None) -> None:
        """Back to 'queued' (for a retry). The last error is kept so users can see why."""
        self._update(job_id, status=JobStatus.QUEUED.value, finished_at=None, error_message=reason)

    def reset_attempts(self, job_id: str) -> None:
        self._update(job_id, attempts=0, error_code=None)

    def delete(self, job_id: str) -> None:
        with SessionLocal.begin() as session:
            job = session.get(Job, job_id)
            if job is not None:
                session.delete(job)

    def _update(self, job_id: str, increment_attempts: bool = False, **fields: object) -> None:
        with SessionLocal.begin() as session:
            job = session.get(Job, job_id)
            if job is None:
                return
            for name, value in fields.items():
                setattr(job, name, value)
            if increment_attempts:
                job.attempts += 1
            job.updated_at = utcnow()
