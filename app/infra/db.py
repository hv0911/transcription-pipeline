"""Database connection and the `jobs` table (SQLAlchemy)."""

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

# JSONB on Postgres (queryable, compact); plain JSON on SQLite, which the tests use.
JsonColumn = JSON().with_variant(JSONB(), "postgresql")

SessionLocal = sessionmaker(expire_on_commit=False)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Input
    original_filename: Mapped[str] = mapped_column(String(255))
    audio_path: Mapped[str] = mapped_column(String(512))
    file_hash: Mapped[str] = mapped_column(String(64), index=True)
    file_size_bytes: Mapped[int] = mapped_column(Integer)
    duration_sec: Mapped[float | None] = mapped_column(Float)
    language_requested: Mapped[str | None] = mapped_column(String(8))
    word_timestamps: Mapped[bool] = mapped_column(default=False)

    # Engine actually used
    engine: Mapped[str] = mapped_column(String(32))
    model_name: Mapped[str] = mapped_column(String(64))
    device: Mapped[str | None] = mapped_column(String(16))
    compute_type: Mapped[str | None] = mapped_column(String(32))

    # Progress, retries, errors
    chunks_total: Mapped[int] = mapped_column(Integer, default=0)
    chunks_done: Mapped[int] = mapped_column(Integer, default=0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)

    # Output
    language_detected: Mapped[str | None] = mapped_column(String(8))
    transcript: Mapped[dict | None] = mapped_column(JsonColumn)


def init_db(database_url: str) -> None:
    """Connect and create tables. (In production, schema changes would go through Alembic.)"""
    engine = create_engine(database_url, pool_pre_ping=True)
    SessionLocal.configure(bind=engine)
    Base.metadata.create_all(engine)
