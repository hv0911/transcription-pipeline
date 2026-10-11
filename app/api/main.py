"""API entry point: `uvicorn app.api.main:app`."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, routes
from app.api.errors import register_error_handlers
from app.core.config import get_settings
from app.infra.db import init_db
from app.core.logs import setup_logging


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    init_db(settings.database_url)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Transcription Pipeline",
        description="Upload audio, get a transcript with timestamps per segment.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(routes.router)
    app.include_router(health.router)
    register_error_handlers(app)
    return app


app = create_app()
