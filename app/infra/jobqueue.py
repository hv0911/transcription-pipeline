"""Job queue backed by Redis + RQ. (Named jobqueue so it does not shadow Python's `queue` module.)"""

import json

from redis import Redis
from rq import Queue, Retry

QUEUE_NAME = "transcriptions"
WORKER_INFO_KEY = "worker:engine_info"
RETRY_INTERVALS_SEC = [10, 30, 60]  # wait longer before each new attempt


class JobQueue:
    def __init__(self, redis_url: str, max_retries: int, job_timeout_sec: int) -> None:
        self.redis = Redis.from_url(redis_url)
        self.queue = Queue(QUEUE_NAME, connection=self.redis)
        self._max_retries = max_retries
        self._job_timeout_sec = job_timeout_sec

    def enqueue(self, job_id: str) -> None:
        # Only the job id goes on the queue; the worker reads everything else from the database.
        self.queue.enqueue(
            "app.worker.tasks.process_job",
            job_id,
            retry=Retry(max=self._max_retries, interval=RETRY_INTERVALS_SEC),
            job_timeout=self._job_timeout_sec,
        )

    def depth(self) -> int:
        return len(self.queue)

    def ping(self) -> bool:
        return bool(self.redis.ping())

    # The worker publishes which engine/device it actually loaded, so /health can show it.
    def set_worker_info(self, info: dict) -> None:
        self.redis.set(WORKER_INFO_KEY, json.dumps(info))

    def get_worker_info(self) -> dict | None:
        raw = self.redis.get(WORKER_INFO_KEY)
        return json.loads(raw) if raw else None
