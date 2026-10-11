import pytest

from app.core.config import Settings
from app.engines.device import resolve_compute_type, resolve_device
from app.engines.fake import FakeTranscriber
from app.engines.registry import build_transcriber


def test_registry_builds_known_engines():
    assert isinstance(build_transcriber("fake", Settings()), FakeTranscriber)
    assert build_transcriber("faster_whisper", Settings()).name == "faster_whisper"


def test_registry_rejects_unknown_engine_with_clear_message():
    with pytest.raises(ValueError, match="Unknown TRANSCRIBER_ENGINE 'nope'. Available: fake, faster_whisper"):
        build_transcriber("nope", Settings())


@pytest.mark.parametrize(
    ("requested", "cuda_count", "expected"),
    [("auto", 1, "cuda"), ("auto", 0, "cpu"), ("cpu", 1, "cpu"), ("cuda", 0, "cuda")],
)
def test_resolve_device(requested, cuda_count, expected):
    assert resolve_device(requested, cuda_count) == expected


def test_resolve_device_rejects_invalid_value():
    with pytest.raises(ValueError):
        resolve_device("tpu", 0)


def test_resolve_compute_type():
    assert resolve_compute_type("auto", "cuda") == "float16"
    assert resolve_compute_type("auto", "cpu") == "int8"
    assert resolve_compute_type("int8_float16", "cuda") == "int8_float16"
