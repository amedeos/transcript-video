"""Speaker diarization via whisperX's pyannote integration.

Resolves the HuggingFace token from (in order):

1. The ``--hf-token`` CLI flag.
2. The ``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN`` environment variable.
3. The ``~/.cache/huggingface/token`` file (where ``huggingface-cli login`` writes it).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


def resolve_hf_token(cli_token: str | None) -> str | None:
    """Return the first available HF token source, or ``None`` if none found."""
    if cli_token:
        return cli_token

    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(var)
        if value:
            return value

    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.is_file():
        try:
            text = token_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if text:
            return text
    return None


def diarize_and_assign(
    aligned: dict[str, Any],
    audio,
    *,
    hf_token: str | None,
    device: str,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict[str, Any]:
    """Run pyannote diarization and assign speakers to the aligned segments.

    Returns the updated ``aligned`` dict (segments now carry a ``speaker`` field
    where assignment was possible).
    """
    if not hf_token:
        print(
            "Error: a HuggingFace token is required for pyannote diarization. "
            "Pass --hf-token, set HF_TOKEN, or run `huggingface-cli login`.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import whisperx
    except ImportError:
        print(
            "Error: whisperx is not installed. Diarization requires the full "
            "pipeline; install with `uv pip install -e .`.",
            file=sys.stderr,
        )
        sys.exit(1)

    diarize_pipeline = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)

    diarize_kwargs: dict[str, Any] = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarize_kwargs["max_speakers"] = max_speakers

    diarize_segments = diarize_pipeline(audio, **diarize_kwargs)
    return whisperx.assign_word_speakers(diarize_segments, aligned)


def count_unique_speakers(segments: list[dict[str, Any]]) -> int:
    """Count distinct non-empty speaker labels across ``segments``."""
    seen: set[str] = set()
    for seg in segments:
        label = seg.get("speaker")
        if label:
            seen.add(label)
    return len(seen)
