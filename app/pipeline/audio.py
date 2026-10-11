"""Thin wrappers around ffprobe / ffmpeg.

Format handling lives here and nowhere else: whatever ffmpeg can decode (WAV, MP3, M4A, OGG,
FLAC, WebM, even video files) is converted once to 16 kHz mono 16-bit WAV, which is what
Whisper expects. Everything after this point only ever sees that one format.
"""

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.core import errors
from app.core.errors import PermanentError


@dataclass
class AudioInfo:
    duration_sec: float
    has_audio: bool
    format_name: str


def probe(path: Path) -> AudioInfo:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise PermanentError(errors.INVALID_AUDIO, "File could not be decoded as audio")
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    return AudioInfo(
        duration_sec=float(fmt.get("duration") or 0.0),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        format_name=fmt.get("format_name", "unknown"),
    )


def to_wav16k_mono(src: Path, dst: Path) -> None:
    _ffmpeg(["-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)])


def cut(src: Path, dst: Path, start: float, end: float) -> None:
    """Copy [start, end) of a WAV file into a new WAV file."""
    _ffmpeg(["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(src), "-c:a", "pcm_s16le", str(dst)])


def find_silences(path: Path, noise_db: int = -35, min_silence_sec: float = 0.5) -> list[tuple[float, float]]:
    """Return (start, end) of every silent stretch. Streams the file, so memory use is constant."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_sec}", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg silencedetect failed: {result.stderr[-500:]}")
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", result.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", result.stderr)]
    if len(ends) < len(starts):  # file ended while still silent
        ends.append(probe(path).duration_sec)
    return list(zip(starts, ends))


def _ffmpeg(args: list[str]) -> None:
    result = subprocess.run(["ffmpeg", "-y", "-v", "error", *args], capture_output=True, text=True)
    if result.returncode != 0:
        # The input already passed ffprobe, so a decode failure here means the file is corrupt.
        raise PermanentError(errors.INVALID_AUDIO, f"ffmpeg could not process the file: {result.stderr[-300:]}")
