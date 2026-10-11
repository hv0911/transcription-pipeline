"""Worker entry point: load the model once, recover stuck jobs, then process the queue forever."""

import logging
from dataclasses import asdict
from datetime import timedelta

from rq import SimpleWorker

from app.core import errors
from app.core.config import Settings, get_settings
from app.infra.db import init_db, utcnow
from app.infra.jobqueue import JobQueue
from app.core.logs import setup_logging
from app.infra.repository import JobRepository
from app.worker.tasks import get_transcriber

log = logging.getLogger("app.worker")


def recover_stale_jobs(repo: JobRepository, queue: JobQueue, settings: Settings) -> None:
    """If a worker was killed mid-job, the job is stuck in 'processing'. Put it back on the queue.

    Its chunk checkpoints are still on disk, so it resumes instead of starting over.
    """
    cutoff = utcnow() - timedelta(seconds=settings.stale_job_sec)
    for job in repo.find_stale_processing(cutoff):
        if job.attempts > settings.max_retries:
            repo.fail(job.id, errors.TRANSCRIPTION_FAILED, "Worker stopped too many times while processing")
            log.warning("job=%s stale, out of attempts, marked failed", job.id)
        else:
            repo.requeue(job.id, reason="Recovered after worker stopped")
            queue.enqueue(job.id)
            log.warning("job=%s stale, re-enqueued", job.id)


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    init_db(settings.database_url)
    queue = JobQueue(settings.redis_url, settings.max_retries, settings.job_timeout_sec)

    info = get_transcriber().info()  # loads the model now, before taking any job
    log.info("engine=%s model=%s device=%s compute_type=%s", info.engine, info.model, info.device, info.compute_type)
    queue.set_worker_info(asdict(info))

    recover_stale_jobs(JobRepository(), queue, settings)

    # SimpleWorker runs jobs in this process (no fork per job), so the loaded model is reused
    # and GPU memory is held once. Scale out by running more worker containers.
    worker = SimpleWorker([queue.queue], connection=queue.redis)
    worker.work(with_scheduler=True)  # the scheduler is what fires delayed retries


if __name__ == "__main__":
    main()
