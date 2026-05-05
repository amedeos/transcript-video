"""Speaker diarization via whisperX's pyannote integration.

Resolves the HuggingFace token from (in order):

1. The ``--hf-token`` CLI flag.
2. The ``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN`` environment variable.
3. The ``~/.cache/huggingface/token`` file (where ``huggingface-cli login`` writes it).
"""

from __future__ import annotations

import inspect
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


def _resolve_diarization_api():
    """Resolve ``DiarizationPipeline`` and ``assign_word_speakers`` across whisperX versions.

    whisperX moved the diarization pipeline between top-level and ``whisperx.diarize``
    over its 3.x history; we accept both layouts so the package works against the
    pinned ``>=3.1`` range without a hard version lock. Note: the bare
    ``whisperx.diarize.Pipeline`` symbol (re-exported pyannote class) is not a
    valid fallback — its ``__init__`` signature is incompatible.
    """
    try:
        import whisperx
    except ImportError:
        print(
            "Error: whisperx is not installed. Diarization requires the full "
            "pipeline; install with `uv pip install -e .`.",
            file=sys.stderr,
        )
        sys.exit(1)

    pipeline_cls = getattr(getattr(whisperx, "diarize", None), "DiarizationPipeline", None)
    if pipeline_cls is None:
        pipeline_cls = getattr(whisperx, "DiarizationPipeline", None)
    if pipeline_cls is None:
        try:
            from whisperx.diarize import DiarizationPipeline as pipeline_cls  # type: ignore
        except ImportError:
            print(
                "Error: could not locate DiarizationPipeline in the installed whisperx package. "
                "Try `uv pip install --upgrade whisperx` (or pin to a version that exposes it).",
                file=sys.stderr,
            )
            sys.exit(1)

    assign_fn = getattr(whisperx, "assign_word_speakers", None) or getattr(
        getattr(whisperx, "diarize", None), "assign_word_speakers", None
    )
    if assign_fn is None:
        try:
            from whisperx.diarize import assign_word_speakers as assign_fn  # type: ignore
        except ImportError:
            print(
                "Error: could not locate assign_word_speakers in the installed whisperx package. "
                "Try `uv pip install --upgrade whisperx`.",
                file=sys.stderr,
            )
            sys.exit(1)

    return pipeline_cls, assign_fn


def _build_pipeline_init_kwargs(pipeline_cls, hf_token: str, device: str) -> dict[str, Any]:
    """Build the ``__init__`` kwargs for ``DiarizationPipeline`` adapting to its signature.

    whisperX renamed ``use_auth_token`` to ``token`` to match the newer
    ``huggingface_hub`` convention; we accept either by inspecting the
    constructor signature.
    """
    try:
        params = inspect.signature(pipeline_cls.__init__).parameters
    except (TypeError, ValueError):
        params = {}

    init_kwargs: dict[str, Any] = {}
    for token_arg in ("token", "use_auth_token", "auth_token"):
        if token_arg in params:
            init_kwargs[token_arg] = hf_token
            break
    else:
        # Fall back: surface the token via env var so pyannote/HF can pick it up.
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)

    if "device" in params:
        init_kwargs["device"] = device

    return init_kwargs


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

    pipeline_cls, assign_fn = _resolve_diarization_api()
    init_kwargs = _build_pipeline_init_kwargs(pipeline_cls, hf_token, device)
    diarize_pipeline = pipeline_cls(**init_kwargs)

    diarize_kwargs: dict[str, Any] = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarize_kwargs["max_speakers"] = max_speakers

    diarize_segments = diarize_pipeline(audio, **diarize_kwargs)
    return assign_fn(diarize_segments, aligned)


def count_unique_speakers(segments: list[dict[str, Any]]) -> int:
    """Count distinct non-empty speaker labels across ``segments``."""
    seen: set[str] = set()
    for seg in segments:
        label = seg.get("speaker")
        if label:
            seen.add(label)
    return len(seen)
