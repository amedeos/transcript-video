"""Output writers for JSON, SRT, VTT, and TXT.

Kept torch-free so it can be imported by ``transcript-to-md`` if needed. The
optional speaker names for subtitle cues are resolved by the caller (pipeline
or ``transcript-to-md``) via ``markdown.build_effective_speaker_map`` and passed
in as a plain ``{label: name}`` dict, so no heavy imports leak into this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import format_timestamp_srt, format_timestamp_vtt


def write_json(data: dict[str, Any], output_path: Path) -> None:
    """Write the canonical JSON artifact (UTF-8, indented, non-ASCII preserved)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _cue_text(seg: dict[str, Any], speaker_names: dict[str, str] | None) -> str:
    """Build a subtitle cue's text, optionally prefixed with the speaker name.

    ``speaker_names`` is the effective ``{label: name}`` map. When ``None`` the
    text is returned unchanged (default — matches the historical SRT output).
    When a map is given and the segment carries a ``speaker`` label, the cue is
    prefixed with ``"<name>: "``; an unmapped label falls back to the raw label.
    """
    text = (seg.get("text") or "").strip()
    if speaker_names is not None:
        label = seg.get("speaker")
        if label:
            name = speaker_names.get(label, label)
            text = f"{name}: {text}"
    return text


def write_srt(
    segments: list[dict[str, Any]],
    output_path: Path,
    *,
    speaker_names: dict[str, str] | None = None,
) -> None:
    """Write a SubRip subtitle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            text = _cue_text(seg, speaker_names)
            f.write(f"{i}\n")
            f.write(f"{format_timestamp_srt(start)} --> {format_timestamp_srt(end)}\n")
            f.write(f"{text}\n\n")


def write_vtt(
    segments: list[dict[str, Any]],
    output_path: Path,
    *,
    speaker_names: dict[str, str] | None = None,
) -> None:
    """Write a WebVTT subtitle file (``.vtt``), readable by VLC and HTML5 video."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, 1):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            text = _cue_text(seg, speaker_names)
            f.write(f"{i}\n")
            f.write(f"{format_timestamp_vtt(start)} --> {format_timestamp_vtt(end)}\n")
            f.write(f"{text}\n\n")


def write_txt(segments: list[dict[str, Any]], output_path: Path) -> None:
    """Write a plain-text transcript, one segment per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            text = (seg.get("text") or "").strip()
            f.write(f"{text}\n")
