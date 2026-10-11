"""Request/response shapes for the HTTP API (also drive the OpenAPI docs at /docs)."""

from datetime import datetime

from pydantic import BaseModel

from app.core.models import Transcript


class SubmitResponse(BaseModel):
    job_id: str
    status: str
    deduplicated: bool = False  # true when an identical file was already transcribed


class Progress(BaseModel):
    chunks_done: int
    chunks_total: int


class ErrorInfo(BaseModel):
    code: str | None
    message: str | None


class JobResponse(BaseModel):
    job_id: str
    status: str
    original_filename: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    attempts: int
    engine: str
    model: str
    device: str | None
    progress: Progress
    error: ErrorInfo | None = None
    transcript: Transcript | None = None


class HealthResponse(BaseModel):
    status: str
    database: bool
    redis: bool
    queue_depth: int | None
    worker: dict | None
