#!/usr/bin/env bash
# End-to-end check against a running stack: upload a file, poll until done, print the segments.
#   ./scripts/e2e.sh samples/hello.mp3
set -euo pipefail

FILE="${1:-samples/hello.mp3}"
API="${API:-http://localhost:8000}"

echo "Uploading $FILE ..."
RESPONSE=$(curl -s -F "file=@$FILE" "$API/v1/transcriptions")
echo "$RESPONSE"
JOB_ID=$(echo "$RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin)['job_id'])")

while true; do
  JOB=$(curl -s "$API/v1/transcriptions/$JOB_ID")
  STATUS=$(echo "$JOB" | python -c "import sys, json; j = json.load(sys.stdin); print(j['status'], j['progress']['chunks_done'], '/', j['progress']['chunks_total'])")
  echo "status: $STATUS"
  case "$STATUS" in
    completed*|failed*) break ;;
  esac
  sleep 2
done

echo "$JOB" | python -c "
import sys, json
job = json.load(sys.stdin)
if job['status'] == 'failed':
    print('FAILED:', job['error'])
    sys.exit(1)
t = job['transcript']
print(f\"language={t['language']} duration={t['duration_sec']}s segments={len(t['segments'])}\")
for s in t['segments']:
    flag = ' (low confidence)' if s['low_confidence'] else ''
    print(f\"[{s['start']:8.2f} -> {s['end']:8.2f}] {s['text']}{flag}\")
"
