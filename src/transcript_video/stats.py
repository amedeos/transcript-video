"""Per-speaker statistics and suspect-segment flagging.

Torch-free module. Imported by both :mod:`pipeline` (when writing the canonical
JSON) and :mod:`cli_to_md` (for the ``--list-speakers`` overview).
"""

from __future__ import annotations

from typing import Any

_FIRST_TEXT_MAX_CHARS = 80

# Default thresholds for marking a Whisper segment as "suspect" (likely
# unreliable transcription). Tuned for the typical avg_logprob distribution
# of large-v3: well-recognized segments sit between -0.2 and -0.5, hard
# segments hit -0.8 to -1.0, hallucinations frequently land below -1.5.
DEFAULT_AVG_LOGPROB_THRESHOLD = -1.0
DEFAULT_NO_SPEECH_PROB_THRESHOLD = 0.6


def mark_suspect_segments(
    segments: list[dict[str, Any]],
    *,
    avg_logprob_threshold: float = DEFAULT_AVG_LOGPROB_THRESHOLD,
    no_speech_prob_threshold: float = DEFAULT_NO_SPEECH_PROB_THRESHOLD,
) -> int:
    """Annotate segments with ``suspect: true`` and ``suspect_reasons: [...]``.

    Mutates ``segments`` in place. Reasons currently considered:

    - ``low_logprob``: ``avg_logprob`` (when present) is below
      ``avg_logprob_threshold``; the model is unsure of the tokens it picked.
    - ``high_no_speech_prob``: ``no_speech_prob`` (when present) is above
      ``no_speech_prob_threshold``; Whisper believed the segment was silence
      but emitted text anyway — a classic hallucination shape.

    Segments without the relevant fields are left untouched (we don't
    fabricate suspicion). Returns the number of segments flagged.
    """
    flagged = 0
    for seg in segments:
        reasons: list[str] = []

        avg_logprob = seg.get("avg_logprob")
        if isinstance(avg_logprob, int | float) and avg_logprob < avg_logprob_threshold:
            reasons.append("low_logprob")

        no_speech_prob = seg.get("no_speech_prob")
        if isinstance(no_speech_prob, int | float) and no_speech_prob > no_speech_prob_threshold:
            reasons.append("high_no_speech_prob")

        if reasons:
            seg["suspect"] = True
            seg["suspect_reasons"] = reasons
            flagged += 1
    return flagged


def count_suspect_segments(segments: list[dict[str, Any]]) -> int:
    """Return the count of segments already flagged with ``suspect: true``."""
    return sum(1 for s in segments if s.get("suspect"))


def count_unique_speakers(segments: list[dict[str, Any]]) -> int:
    """Count distinct non-empty speaker labels across ``segments``."""
    seen: set[str] = set()
    for seg in segments:
        label = seg.get("speaker")
        if label:
            seen.add(label)
    return len(seen)


def compute_speaker_stats(segments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate per-speaker stats from a list of segments.

    Returns a dict keyed by speaker label, in first-appearance order, with:

    - ``duration_seconds``: total speech time for that speaker (sum of ``end - start``).
    - ``num_turns``: number of distinct consecutive runs of that speaker
      (e.g. A→B→A counts as 2 turns for A).
    - ``num_suspect``: count of segments flagged ``suspect: true`` for this speaker.
    - ``percentage``: duration as a fraction of the total diarized time, in [0, 100].
    - ``first_text``: the first segment's text trimmed to ~80 chars (with ellipsis
      if truncated). Useful for identifying who is who at a glance.

    Segments without a ``speaker`` field are skipped.
    """
    out: dict[str, dict[str, Any]] = {}
    last_speaker: str | None = None

    for seg in segments:
        speaker = seg.get("speaker")
        if not speaker:
            last_speaker = None
            continue

        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        duration = max(0.0, end - start)
        text = (seg.get("text") or "").strip()

        bucket = out.get(speaker)
        if bucket is None:
            preview = text[:_FIRST_TEXT_MAX_CHARS]
            if len(text) > _FIRST_TEXT_MAX_CHARS:
                preview = preview.rstrip() + "..."
            bucket = {
                "duration_seconds": 0.0,
                "num_turns": 0,
                "num_suspect": 0,
                "first_text": preview,
            }
            out[speaker] = bucket

        bucket["duration_seconds"] += duration
        if speaker != last_speaker:
            bucket["num_turns"] += 1
        if seg.get("suspect"):
            bucket["num_suspect"] += 1
        last_speaker = speaker

    total = sum(b["duration_seconds"] for b in out.values())
    for bucket in out.values():
        if total > 0:
            bucket["percentage"] = round(bucket["duration_seconds"] / total * 100, 1)
        else:
            bucket["percentage"] = 0.0

    return out
