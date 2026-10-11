"""Decide whether to run on GPU or CPU. The GPU is always optional."""


def cuda_device_count() -> int:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count()
    except Exception:  # no CUDA libraries, no driver, or ctranslate2 not installed
        return 0


def resolve_device(requested: str, cuda_count: int) -> str:
    if requested == "auto":
        return "cuda" if cuda_count > 0 else "cpu"
    if requested in ("cpu", "cuda"):
        return requested
    raise ValueError(f"WHISPER_DEVICE must be auto, cpu or cuda, got '{requested}'")


def resolve_compute_type(requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    # float16 is fast on GPUs; int8 is the fastest accurate option on CPUs.
    return "float16" if device == "cuda" else "int8"
