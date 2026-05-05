"""Shared utilities: timestamp formatting, file helpers, and logging setup."""

from __future__ import annotations

import logging
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


def setup_logging(*, quiet: bool = False, verbose: bool = False) -> None:
    """Configure the root logger based on CLI verbosity flags.

    - ``quiet=True`` → WARNING (errors and warnings only).
    - ``verbose=True`` → DEBUG (per-segment progress visible).
    - default → INFO (status messages but no per-segment spam).

    Output goes to stderr so it doesn't pollute stdout-redirected outputs.
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(level)


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
