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


class _LightningUpgradeFilter(logging.Filter):
    """Drop Lightning's "automatically upgraded your loaded checkpoint" log records.

    whisperX ships a VAD checkpoint saved with Lightning v1.5.4; Lightning ≥2.x
    emits an INFO record on every load offering the one-shot upgrade command.
    The record is emitted via ``logging`` (logger
    ``lightning.pytorch.utilities.migration.utils``), not via ``warnings.warn``,
    so the ``warnings.filterwarnings`` calls in this module never touched it.

    Two complementary remedies exist and both ship together:

    - **Runtime filter (this class)**: durable across reinstalls and portable
      to fresh installs. Always active after :func:`silence_known_noisy_warnings`.
    - **One-shot checkpoint upgrade**: ``python -m
      lightning.pytorch.utilities.upgrade_checkpoint --map-to-cpu
      <site-packages>/whisperx/assets/pytorch_model.bin`` rewrites the file in
      place so Lightning stops emitting the record. Lost on every
      ``uv sync`` / whisperX reinstall and CPU-only hosts need ``--map-to-cpu``
      (the checkpoint stores CUDA tensors). Not reproducible across machines,
      so the runtime filter remains the real guarantee.
    """

    _MARKER = "Lightning automatically upgraded your loaded checkpoint"

    def filter(self, record: logging.LogRecord) -> bool:
        # Return False to drop the record. Written as `not in` (ruff E713)
        # so a future marker change only touches one string.
        return self._MARKER not in record.getMessage()


_LIGHTNING_UPGRADE_LOGGERS = (
    "lightning.pytorch.utilities.migration.utils",
    # Legacy name kept for installs that still re-export from pytorch_lightning.
    "pytorch_lightning.utilities.migration.utils",
)


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
    - **Lightning checkpoint-upgrade notice**: Lightning ≥2.x prints a
      "automatically upgraded your loaded checkpoint" message on every load
      of the whisperX VAD checkpoint (saved with Lightning v1.5.4). It's an
      INFO log record, not a warning, so it bypasses
      ``warnings.filterwarnings``. Suppressed here via a
      :class:`logging.Filter` on the emitting logger. See
      :class:`_LightningUpgradeFilter` for why the one-shot file upgrade is
      kept as a secondary, non-durable remedy.
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

    # Lightning upgrade notice: attach the filter idempotently. The filter
    # object must be the same instance to detect re-registration —
    # `logging.Filter.__eq__` is identity, so a fresh `_LightningUpgradeFilter()`
    # on every call would silently stack N copies and break the idempotency
    # contract documented in the test suite. Cache it on the function.
    if not hasattr(silence_known_noisy_warnings, "_lightning_filter"):
        silence_known_noisy_warnings._lightning_filter = _LightningUpgradeFilter()  # type: ignore[attr-defined]
    filt = silence_known_noisy_warnings._lightning_filter  # type: ignore[attr-defined]
    for logger_name in _LIGHTNING_UPGRADE_LOGGERS:
        logger = logging.getLogger(logger_name)
        if filt not in logger.filters:
            logger.addFilter(filt)


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
