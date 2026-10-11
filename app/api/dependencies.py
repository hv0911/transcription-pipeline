"""Wiring: which concrete storage, repository and queue the API uses. Tests override these."""

from functools import lru_cache

from app.core.config import get_settings
from app.infra.jobqueue import JobQueue
from app.infra.repository import JobRepository
from app.infra.storage import LocalStorage


@lru_cache
def get_storage() -> LocalStorage:
    return LocalStorage(get_settings().data_dir)


@lru_cache
def get_queue() -> JobQueue:
    s = get_settings()
    return JobQueue(s.redis_url, s.max_retries, s.job_timeout_sec)


def get_repository() -> JobRepository:
    return JobRepository()
