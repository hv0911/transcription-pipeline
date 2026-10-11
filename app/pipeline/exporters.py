"""Render a transcript in different formats. Adding a format = one function + one line in EXPORTERS."""

from typing import Callable

from app.core.models import Transcript


def to_txt(t: Transcript) -> str:
    return "\n".join(seg.text for seg in t.segments) + "\n"


def to_json(t: Transcript) -> str:
    return t.model_dump_json(indent=2)


def to_srt(t: Transcript) -> str:
    blocks = [
        f"{i}\n{_timestamp(seg.start, ',')} --> {_timestamp(seg.end, ',')}\n{seg.text}\n"
        for i, seg in enumerate(t.segments, start=1)
    ]
    return "\n".join(blocks)


def to_vtt(t: Transcript) -> str:
    blocks = [f"{_timestamp(seg.start, '.')} --> {_timestamp(seg.end, '.')}\n{seg.text}\n" for seg in t.segments]
    return "WEBVTT\n\n" + "\n".join(blocks)


def _timestamp(seconds: float, ms_separator: str) -> str:
    """SRT uses 00:01:02,500 and WebVTT uses 00:01:02.500."""
    total_ms = round(seconds * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{ms_separator}{ms:03d}"


# format -> (render function, HTTP media type)
EXPORTERS: dict[str, tuple[Callable[[Transcript], str], str]] = {
    "txt": (to_txt, "text/plain"),
    "json": (to_json, "application/json"),
    "srt": (to_srt, "application/x-subrip"),
    "vtt": (to_vtt, "text/vtt"),
}
