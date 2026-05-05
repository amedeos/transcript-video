"""whisperX-based ASR wrapper.

Builds the kwargs dictionary, probes CUDA capabilities, and provides a thin
loader for the whisperX model. All references to :mod:`whisperx` are local to
the functions that need them so importing this module without the heavy ML
stack installed still works (e.g. for unit tests of helpers).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def check_cuda_available() -> tuple[bool, str | None]:
    """Return ``(cuda_available, compute_type_hint)``.

    The hint is the preferred compute type to request from ctranslate2 when CUDA
    is usable; ``None`` otherwise.
    """
    try:
        import ctranslate2
    except ImportError:
        return False, None
    try:
        types = ctranslate2.get_supported_compute_types("cuda")
    except Exception:
        return False, None
    if not types:
        return False, None
    if "float16" in types:
        return True, "float16"
    if "int8_float16" in types:
        return True, "int8_float16"
    return True, "int8"


def resolve_device_and_compute(
    requested_device: str | None, requested_compute: str | None
) -> tuple[str, str]:
    """Resolve the actual device + compute type to use.

    - ``requested_device``: ``None`` (auto), ``"cuda"``, or ``"cpu"``.
    - ``requested_compute``: explicit override, otherwise picked automatically.
    """
    if requested_device == "cuda":
        device = "cuda"
    elif requested_device == "cpu":
        device = "cpu"
    else:
        cuda_ok, _ = check_cuda_available()
        device = "cuda" if cuda_ok else "cpu"

    if requested_compute:
        return device, requested_compute

    if device == "cuda":
        _, hint = check_cuda_available()
        return device, hint or "float16"
    return device, "int8"


def build_transcribe_kwargs(
    *,
    language: str | None,
    initial_prompt: str | None,
    hotwords: str | None,
    anti_loop: bool,
) -> dict[str, Any]:
    """Build the kwargs dict passed to whisperX's ``model.transcribe``.

    - ``language=None`` → autodetect (whisperX default).
    - ``initial_prompt`` / ``hotwords``: pass-through. ``None`` and ``""`` both
      omit the kwarg; an empty string explicitly disables it (we still skip the
      kwarg since an empty value would do nothing useful).
    - ``anti_loop=True`` applies the same three mitigations as the reference
      project's ``--anti-loop`` flag.
    """
    kwargs: dict[str, Any] = {}
    if language:
        kwargs["language"] = language
    if anti_loop:
        kwargs["condition_on_previous_text"] = False
        kwargs["compression_ratio_threshold"] = 2.0
        kwargs["no_speech_threshold"] = 0.5
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    if hotwords:
        kwargs["hotwords"] = hotwords
    return kwargs


def load_whisperx_model(model_name: str, device: str, compute_type: str, beam_size: int):
    """Load a whisperX model. Imports whisperx lazily so failures are clearer."""
    try:
        import whisperx
    except ImportError:
        print(
            "Error: whisperx is not installed. Install with `uv pip install -e .` "
            "or `pip install whisperx`.",
            file=sys.stderr,
        )
        sys.exit(1)

    asr_options = {"beam_size": beam_size}
    return whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        asr_options=asr_options,
    )


def load_audio(input_path: Path):
    """Load audio with whisperX's ffmpeg-backed loader."""
    import whisperx

    return whisperx.load_audio(str(input_path))


def align_segments(
    segments: list[dict[str, Any]],
    language_code: str,
    audio,
    device: str,
) -> dict[str, Any]:
    """Run whisperX forced alignment to produce word-level timestamps.

    Falls back to returning the unaligned segments if no alignment model is
    available for the detected language (whisperX raises in that case).
    """
    import whisperx

    try:
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    except Exception as e:
        print(
            f"Warning: alignment model unavailable for language '{language_code}': {e}. "
            "Continuing without word-level alignment.",
            file=sys.stderr,
        )
        return {"segments": segments, "word_segments": []}

    aligned = whisperx.align(
        segments,
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    return aligned
