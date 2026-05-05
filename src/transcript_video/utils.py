"""Shared utilities: timestamp formatting and small file helpers."""

from __future__ import annotations

import sys
from pathlib import Path


def format_timestamp_srt(seconds: float) -> str:
    """Convert seconds to ``HH:MM:SS,mmm`` (SRT spec)."""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        millis = 0
        secs += 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_hms(seconds: float) -> str:
    """Convert seconds to ``HH:MM:SS`` (used in Markdown headings and frontmatter ``duration``)."""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_timestamp_short(seconds: float) -> str:
    """Convert seconds to ``[MM:SS]`` for live console progress output."""
    if seconds < 0:
        seconds = 0.0
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"[{minutes:02d}:{secs:02d}]"


def read_text_file(path: str | Path, label: str) -> str:
    """Read a UTF-8 file and return its stripped content. Exit on error."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: {label} file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error reading {label} file '{path}': {e}", file=sys.stderr)
        sys.exit(1)
