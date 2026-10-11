"""All configuration comes from environment variables (see .env.example)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Speech-to-text engine
    transcriber_engine: str = "faster_whisper"  # or "fake" (no model download)
    whisper_model: str = "base"
    whisper_device: str = "auto"  # auto | cpu | cuda
    whisper_compute_type: str = "auto"  # auto -> float16 on GPU, int8 on CPU
    beam_size: int = 5
    model_dir: str = "/models"

    # Input limits
    max_upload_mb: int = 500
    max_duration_sec: int = 10800  # 3 hours

    # Long audio handling
    chunk_threshold_sec: int = 600  # files longer than this are chunked
    chunk_target_sec: int = 300  # aim for ~5 minute chunks
    chunk_max_sec: int = 450  # hard cut if no silence is found

    # Post-processing
    confidence_threshold: float = 0.5

    # Jobs and retries
    max_queue_depth: int = 100
    max_retries: int = 3
    job_timeout_sec: int = 3600
    stale_job_sec: int = 600  # a "processing" job with no progress for this long is recovered

    # Infrastructure
    database_url: str = "postgresql+psycopg://transcribe:transcribe@postgres:5432/transcribe"
    redis_url: str = "redis://redis:6379/0"
    data_dir: str = "/data"
    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
