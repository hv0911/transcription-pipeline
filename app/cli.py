"""Run the pipeline on one file, in-process. No API, database or queue needed.

    python -m app.cli samples/hello.mp3 --format srt
    python -m app.cli samples/hello.mp3 --out result.json --engine fake
"""

import argparse
import sys
import tempfile
from pathlib import Path

from app.core.config import get_settings
from app.engines.registry import build_transcriber
from app.pipeline.exporters import EXPORTERS
from app.core.logs import setup_logging
from app.core.models import TranscriptionOptions
from app.pipeline.runner import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe an audio file.")
    parser.add_argument("audio", type=Path, help="Path to an audio file (WAV, MP3, M4A, OGG, ...)")
    parser.add_argument("--out", type=Path, help="Write the result here instead of printing it")
    parser.add_argument("--format", default="json", choices=sorted(EXPORTERS))
    parser.add_argument("--engine", help="Override TRANSCRIBER_ENGINE (e.g. fake)")
    parser.add_argument("--language", help="Language code such as 'en'; default is auto-detect")
    parser.add_argument("--word-timestamps", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    transcriber = build_transcriber(args.engine or settings.transcriber_engine, settings)
    transcriber.load()
    options = TranscriptionOptions(
        language=args.language, word_timestamps=args.word_timestamps, beam_size=settings.beam_size
    )

    with tempfile.TemporaryDirectory() as workdir:
        transcript = run_pipeline(
            args.audio,
            Path(workdir),
            transcriber,
            options,
            settings,
            progress=lambda done, total: print(f"progress: {done}/{total} chunks", file=sys.stderr),
        )

    render, _ = EXPORTERS[args.format]
    output = render(transcript)
    if args.out:
        args.out.write_text(output, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
