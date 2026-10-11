"""Error types shared by the pipeline, worker and API."""


class PermanentError(Exception):
    """The input itself is bad. Retrying can never succeed, so the job fails immediately.

    Every other exception is treated as transient (network blip, out of memory, worker crash)
    and is retried.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# Error codes stored on failed jobs and returned by the API
INVALID_AUDIO = "invalid_audio"
NO_AUDIO_STREAM = "no_audio_stream"
EMPTY_AUDIO = "empty_audio"
TOO_LONG = "too_long"
NO_SPEECH = "no_speech"
INVALID_OPTIONS = "invalid_options"
TRANSCRIPTION_FAILED = "transcription_failed"
