"""Pipeline tests with real ffmpeg and the fake engine: chunking and checkpoint resume."""

import subprocess
from pathlib import Path

from app.core.config import Settings
from app.engines.fake import FakeTranscriber
from app.core.models import TranscriptionOptions
from app.pipeline.runner import run_pipeline


def make_tone_with_gaps(path: Path, blocks: int, tone_sec: int = 50, gap_sec: int = 2) -> None:
    """Tone, silence, tone, silence ... so the chunker has silences to cut at."""
    parts = []
    for _ in range(blocks):
        parts += [f"sine=frequency=440:duration={tone_sec}", f"anullsrc=r=16000:cl=mono,atrim=0:{gap_sec}"]
    inputs = sum((["-f", "lavfi", "-i", p] for p in parts), [])
    concat = "".join(f"[{i}:a]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=0:a=1"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", concat, "-ar", "16000", "-ac", "1", str(path)],
        check=True,
    )


class CountingTranscriber(FakeTranscriber):
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio, options):
        self.calls += 1
        return super().transcribe(audio, options)


def test_long_file_is_chunked_and_timestamps_continue_across_chunks(tmp_path):
    source = tmp_path / "long.wav"
    make_tone_with_gaps(source, blocks=14)  # 14 * 52s = 728s, above the 600s threshold
    settings = Settings(chunk_threshold_sec=600, chunk_target_sec=300, chunk_max_sec=450)
    progress = []

    transcript = run_pipeline(
        source, tmp_path / "work", FakeTranscriber(), TranscriptionOptions(), settings,
        progress=lambda done, total: progress.append((done, total)),
    )

    assert progress[-1][0] == progress[-1][1] > 1  # more than one chunk, all done
    starts = [s.start for s in transcript.segments]
    assert starts == sorted(starts)
    assert abs(transcript.segments[-1].end - transcript.duration_sec) < 0.1  # covers the whole file


def test_rerun_resumes_from_checkpoints(tmp_path):
    source = tmp_path / "long.wav"
    make_tone_with_gaps(source, blocks=14)
    settings = Settings(chunk_threshold_sec=600, chunk_target_sec=300, chunk_max_sec=450)
    workdir = tmp_path / "work"

    first = CountingTranscriber()
    run_pipeline(source, workdir, first, TranscriptionOptions(), settings)

    # Simulate a crash after the first chunk: keep only that chunk's checkpoint.
    checkpoints = sorted((workdir / "chunks").glob("*.json"))
    for extra in checkpoints[1:]:
        extra.unlink()

    second = CountingTranscriber()
    run_pipeline(source, workdir, second, TranscriptionOptions(), settings)
    assert second.calls == first.calls - 1  # chunk 0 was not transcribed again
