from app.pipeline.exporters import EXPORTERS, to_srt, to_txt, to_vtt
from app.core.models import Segment, Transcript


def transcript() -> Transcript:
    return Transcript(
        language="en",
        duration_sec=3725.5,
        text="Hello. World.",
        segments=[
            Segment(id=0, start=0.0, end=1.5, text="Hello.", confidence=0.9),
            Segment(id=1, start=3661.25, end=3662.0, text="World.", confidence=0.9),
        ],
        engine="fake",
        model="fake",
    )


def test_srt_format():
    assert to_srt(transcript()) == (
        "1\n00:00:00,000 --> 00:00:01,500\nHello.\n"
        "\n"
        "2\n01:01:01,250 --> 01:01:02,000\nWorld.\n"
    )


def test_vtt_format():
    assert to_vtt(transcript()) == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.500\nHello.\n"
        "\n"
        "01:01:01.250 --> 01:01:02.000\nWorld.\n"
    )


def test_txt_has_one_line_per_segment():
    assert to_txt(transcript()) == "Hello.\nWorld.\n"


def test_every_exporter_renders():
    for name, (render, media_type) in EXPORTERS.items():
        assert render(transcript()), name
        assert "/" in media_type
