"""Per-speaker statistics derived from a list of segments.

Torch-free module. Imported by both :mod:`pipeline` (when writing the canonical
JSON) and :mod:`cli_to_md` (for the ``--list-speakers`` overview).
"""

from __future__ import annotations

from typing import Any

_FIRST_TEXT_MAX_CHARS = 80


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
                "first_text": preview,
            }
            out[speaker] = bucket

        bucket["duration_seconds"] += duration
        if speaker != last_speaker:
            bucket["num_turns"] += 1
        last_speaker = speaker

    total = sum(b["duration_seconds"] for b in out.values())
    for bucket in out.values():
        if total > 0:
            bucket["percentage"] = round(bucket["duration_seconds"] / total * 100, 1)
        else:
            bucket["percentage"] = 0.0

    return out
