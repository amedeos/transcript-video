"""Output writers for JSON, SRT, and TXT.

Kept torch-free so it can be imported by ``transcript-to-md`` if needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import format_timestamp_srt


def write_json(data: dict[str, Any], output_path: Path) -> None:
    """Write the canonical JSON artifact (UTF-8, indented, non-ASCII preserved)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_srt(segments: list[dict[str, Any]], output_path: Path) -> None:
    """Write a SubRip subtitle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            text = (seg.get("text") or "").strip()
            f.write(f"{i}\n")
            f.write(f"{format_timestamp_srt(start)} --> {format_timestamp_srt(end)}\n")
            f.write(f"{text}\n\n")


def write_txt(segments: list[dict[str, Any]], output_path: Path) -> None:
    """Write a plain-text transcript, one segment per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            text = (seg.get("text") or "").strip()
            f.write(f"{text}\n")
