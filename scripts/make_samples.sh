#!/usr/bin/env bash
# Generate test audio offline with text-to-speech (espeak-ng) inside a throwaway container.
#
#   samples/hello.{wav,mp3,m4a,ogg}  same speech in four formats
#   samples/silent.wav               10 seconds of silence  -> should fail with no_speech
#   samples/corrupt.mp3              not audio at all       -> should fail with invalid_audio
#   samples/long_meeting.wav         ~16 minutes of speech with pauses (not committed) -> gets chunked
set -euo pipefail

cd "$(dirname "$0")/.."
# Docker Desktop on Windows needs a C:/... path; `pwd -W` gives that in Git Bash.
SAMPLES="$(pwd -W 2>/dev/null || pwd)/samples"
export MSYS_NO_PATHCONV=1

# The script runs with `bash -c` (not piped on stdin) because ffmpeg would read stdin.
SCRIPT=$(cat <<'EOF'
set -euo pipefail
apt-get update -qq && apt-get install -y -qq espeak-ng ffmpeg >/dev/null 2>&1
cd /out

say() { espeak-ng -s 150 -w "$1" "$2"; }

# 1. Short sample in several formats
say hello.wav "Hello, and welcome to the transcription pipeline demo. This audio file was generated with text to speech. The pipeline converts any audio format to sixteen kilohertz mono, then transcribes it with Whisper and returns timestamps for every segment."
ffmpeg -y -v error -i hello.wav -codec:a libmp3lame -b:a 64k hello.mp3
ffmpeg -y -v error -i hello.wav -codec:a aac -b:a 64k hello.m4a
ffmpeg -y -v error -i hello.wav -codec:a libvorbis hello.ogg

# 2. Edge cases
ffmpeg -y -v error -f lavfi -i anullsrc=r=16000:cl=mono -t 10 silent.wav
echo "this is definitely not an mp3 file" > corrupt.mp3

# 3. Long file: numbered paragraphs separated by 1.5 s pauses (pauses are where the chunker cuts)
mkdir -p /tmp/long && : > /tmp/long/list.txt
ffmpeg -y -v error -f lavfi -i anullsrc=r=22050:cl=mono -t 1.5 /tmp/long/pause.wav
for i in $(seq 1 55); do
  say /tmp/long/p$i.wav "This is part $i of the long meeting recording. We are discussing the project plan, the budget for the next quarter, and the hiring timeline. Each part is separated by a short pause, so the pipeline can split the file at a silence instead of in the middle of a word."
  echo "file '/tmp/long/p$i.wav'" >> /tmp/long/list.txt
  echo "file '/tmp/long/pause.wav'" >> /tmp/long/list.txt
done
ffmpeg -y -v error -f concat -safe 0 -i /tmp/long/list.txt -ar 16000 -ac 1 long_meeting.wav

chown -R "$(stat -c %u:%g /out)" /out
ls -la /out
EOF
)
docker run --rm -v "$SAMPLES:/out" debian:bookworm-slim bash -c "$SCRIPT"
