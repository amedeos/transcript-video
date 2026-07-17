"""Shared utilities: timestamp formatting, file helpers, logging, warning filters."""

from __future__ import annotations

import logging
import sys
import warnings
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


def format_timestamp_vtt(seconds: float) -> str:
    """Convert seconds to ``HH:MM:SS.mmm`` (WebVTT spec).

    Identical to :func:`format_timestamp_srt` except WebVTT uses a dot as the
    decimal separator for milliseconds instead of SRT's comma.
    """
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        millis = 0
        secs += 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


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


def silence_known_noisy_warnings() -> None:
    """Hide third-party warnings that are not actionable for this tool.

    These all originate from the pyannote / torchcodec / lightning stack and
    do not indicate problems with our pipeline:

    - **torchcodec / ffmpeg version mismatch**: torchcodec supports ffmpeg 4-7
      and warns loudly on ffmpeg ≥ 8. We don't use torchcodec — whisperX loads
      audio through ffmpeg directly via subprocess, which is exactly the
      fallback path the warning describes. Hiding it removes a 20-line
      paragraph that confuses users every run.
    - **TF32 disabled**: pyannote turns off TensorFloat-32 for reproducibility.
      Informational; no quality impact for transcription/diarization.

    Lightning's checkpoint-upgrade *print* (not a warning) is left visible
    because it can't be suppressed without redirecting stderr globally and
    isn't actually noisy enough to justify that.
    """
    # Both pyannote warnings are multi-line and start with a leading newline
    # (they're built from triple-quoted f-strings), so the regex needs the
    # DOTALL flag (?s) to let `.` match newlines. Without it, the filter
    # silently never fires — this exact bug shipped on the first attempt.
    warnings.filterwarnings(
        "ignore",
        message=r"(?s).*torchcodec is not installed correctly.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"(?s).*TensorFloat-32.*has been disabled.*",
    )


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
