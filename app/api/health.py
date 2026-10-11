from typing import Callable

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.dependencies import get_queue
from app.api.schemas import HealthResponse
from app.infra.db import SessionLocal
from app.infra.jobqueue import JobQueue

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(queue: JobQueue = Depends(get_queue)) -> JSONResponse:
    database = _check(_ping_database)
    redis = _check(queue.ping)
    body = HealthResponse(
        status="ok" if database and redis else "degraded",
        database=database,
        redis=redis,
        queue_depth=queue.depth() if redis else None,
        worker=queue.get_worker_info() if redis else None,  # engine/model/device the worker loaded
    )
    return JSONResponse(status_code=200 if body.status == "ok" else 503, content=body.model_dump())


def _ping_database() -> None:
    with SessionLocal() as session:
        session.execute(text("SELECT 1"))


def _check(fn: Callable[[], object]) -> bool:
    try:
        fn()
        return True
    except Exception:
        return False
