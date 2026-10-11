"""API tests: real FastAPI app, real pipeline (ffmpeg), fake engine, SQLite, and an in-memory queue.

The in-memory queue runs the worker task right away, so a full upload -> transcript flow
is tested without Redis, Postgres or a model download.
"""

import io
import math
import struct
import wave

import pytest
from fastapi.testclient import TestClient

from app.api import dependencies
from app.api.main import create_app
from app.core.config import Settings, get_settings
from app.infra.storage import LocalStorage
from app.worker import tasks


class InMemoryQueue:
    def __init__(self, run_jobs: bool) -> None:
        self.run_jobs = run_jobs
        self.enqueued: list[str] = []

    def enqueue(self, job_id: str) -> None:
        self.enqueued.append(job_id)
        if self.run_jobs:
            tasks.process_job(job_id)

    def depth(self) -> int:
        return 0 if self.run_jobs else len(self.enqueued)

    def ping(self) -> bool:
        return True

    def get_worker_info(self) -> dict:
        return {"engine": "fake"}


def make_wav(seconds: float, silent: bool = False) -> bytes:
    rate = 16000
    frames = b"".join(
        struct.pack("<h", 0 if silent else int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
        for i in range(int(seconds * rate))
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(frames)
    return buffer.getvalue()


def build_client(tmp_path, monkeypatch, run_jobs: bool = True, **settings_overrides) -> TestClient:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TRANSCRIBER_ENGINE", "fake")
    get_settings.cache_clear()
    tasks.get_transcriber.cache_clear()

    app = create_app()
    queue = InMemoryQueue(run_jobs)
    app.dependency_overrides[dependencies.get_queue] = lambda: queue
    app.dependency_overrides[dependencies.get_storage] = lambda: LocalStorage(tmp_path / "data")
    if settings_overrides:
        app.dependency_overrides[get_settings] = lambda: Settings(**settings_overrides)
    return TestClient(app)


@pytest.fixture
def client(tmp_path, monkeypatch):
    with build_client(tmp_path, monkeypatch) as c:
        yield c


def upload(client: TestClient, data: bytes, name: str = "speech.wav"):
    return client.post("/v1/transcriptions", files={"file": (name, data, "audio/wav")})


def test_upload_returns_202_and_job_completes_with_timestamps(client):
    response = upload(client, make_wav(12))
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    job = client.get(f"/v1/transcriptions/{job_id}").json()
    assert job["status"] == "completed"
    segments = job["transcript"]["segments"]
    assert [(s["start"], s["end"]) for s in segments] == [(0.0, 5.0), (5.0, 10.0), (10.0, 12.0)]
    assert job["progress"] == {"chunks_done": 1, "chunks_total": 1}


def test_unknown_job_returns_404_with_error_body(client):
    response = client.get("/v1/transcriptions/does-not-exist")
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found", "message": "No job with id does-not-exist"}}


def test_non_audio_upload_returns_415(client):
    response = client.post("/v1/transcriptions", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_invalid_language_returns_422_and_creates_no_job(tmp_path, monkeypatch):
    with build_client(tmp_path, monkeypatch, run_jobs=False) as c:
        # "string" is the placeholder Swagger UI fills in for optional text fields.
        response = c.post(
            "/v1/transcriptions",
            files={"file": ("speech.wav", make_wav(1), "audio/wav")},
            data={"language": "string"},
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_language"
        assert c.app.dependency_overrides[dependencies.get_queue]().enqueued == []


def test_valid_language_is_accepted(client):
    response = client.post(
        "/v1/transcriptions", files={"file": ("speech.wav", make_wav(2), "audio/wav")}, data={"language": "EN"}
    )
    assert response.status_code == 202


def test_oversized_upload_returns_413(tmp_path, monkeypatch):
    with build_client(tmp_path, monkeypatch, max_upload_mb=0) as c:
        assert upload(c, make_wav(1)).status_code == 413


def test_export_before_completion_returns_409(tmp_path, monkeypatch):
    with build_client(tmp_path, monkeypatch, run_jobs=False) as c:
        job_id = upload(c, make_wav(3)).json()["job_id"]
        assert c.get(f"/v1/transcriptions/{job_id}/export?format=srt").status_code == 409


def test_export_srt_after_completion(client):
    job_id = upload(client, make_wav(3)).json()["job_id"]
    response = client.get(f"/v1/transcriptions/{job_id}/export?format=srt")
    assert response.status_code == 200
    assert response.text.startswith("1\n00:00:00,000 --> 00:00:03,000\nfake segment 1\n")


def test_same_file_twice_returns_the_same_completed_job(client):
    data = make_wav(4)
    first = upload(client, data).json()
    second = upload(client, data)
    assert second.status_code == 200
    assert second.json() == {"job_id": first["job_id"], "status": "completed", "deduplicated": True}


def test_corrupt_file_fails_permanently_without_retry(client):
    job_id = upload(client, b"this is not really audio", name="broken.mp3").json()["job_id"]
    job = client.get(f"/v1/transcriptions/{job_id}").json()
    assert job["status"] == "failed"
    assert job["error"]["code"] == "invalid_audio"
    assert job["attempts"] == 1


def test_silent_file_fails_with_no_speech(client):
    job_id = upload(client, make_wav(5, silent=True)).json()["job_id"]
    job = client.get(f"/v1/transcriptions/{job_id}").json()
    assert job["status"] == "failed"
    assert job["error"]["code"] == "no_speech"


def test_delete_removes_the_job(client):
    job_id = upload(client, make_wav(2)).json()["job_id"]
    assert client.delete(f"/v1/transcriptions/{job_id}").status_code == 204
    assert client.get(f"/v1/transcriptions/{job_id}").status_code == 404
