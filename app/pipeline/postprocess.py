"""Clean up raw segments so the output is ready for downstream use."""

from app.core.models import Segment

# Whisper sometimes "hears" words in silence. Those segments have a high no-speech
# probability AND low confidence at the same time.
HALLUCINATION_NO_SPEECH_PROB = 0.6
HALLUCINATION_MAX_CONFIDENCE = 0.4


def postprocess(segments: list[Segment], confidence_threshold: float) -> list[Segment]:
    cleaned: list[Segment] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue  # empty segment
        if seg.no_speech_prob > HALLUCINATION_NO_SPEECH_PROB and seg.confidence < HALLUCINATION_MAX_CONFIDENCE:
            continue  # likely hallucination
        if cleaned and _normalize(cleaned[-1].text) == _normalize(text):
            # Repetition loop: the same sentence again and again. Keep one copy, extend its end time.
            cleaned[-1] = cleaned[-1].model_copy(update={"end": seg.end})
            continue
        cleaned.append(seg.model_copy(update={"text": text}))

    return [
        seg.model_copy(update={"id": i, "low_confidence": seg.confidence < confidence_threshold})
        for i, seg in enumerate(cleaned)
    ]


def full_text(segments: list[Segment]) -> str:
    return " ".join(seg.text for seg in segments)


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())
