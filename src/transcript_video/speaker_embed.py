"""Compute per-cluster speaker embeddings via ``pyannote.audio.Inference``.

This is the heavy companion of :mod:`speaker_db`. It loads a pretrained
embedding model (``pyannote/wespeaker-voxceleb-resnet34-LM`` by default —
the same one ``pyannote/speaker-diarization-3.1`` uses internally),
extracts an embedding per diarized segment, and aggregates per cluster
using the **median** of long-enough segments.

The median is the anti-impure-cluster mitigation: when pyannote
occasionally merges two voices into the same ``SPEAKER_XX`` label, a few
outlier segments don't drag the cluster centroid the way a mean would.

This module is **not** torch-free. To keep ``pipeline.py`` importable in
the torch-free CI (which doesn't install pyannote), the heavy imports
are deferred to function-call time — same pattern as :mod:`diarize`.
"""

from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("transcript_video.speaker_embed")

DEFAULT_EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
DEFAULT_MIN_SEGMENT_SECONDS = 1.5


def _build_from_pretrained_kwargs(model_cls, hf_token: str) -> dict[str, Any]:
    """Pick the token kwarg accepted by ``Model.from_pretrained``.

    pyannote.audio renamed ``use_auth_token`` to ``token`` across releases;
    introspect and pick whichever the installed version accepts.
    """
    try:
        params = inspect.signature(model_cls.from_pretrained).parameters
    except (TypeError, ValueError):
        params = {}

    kwargs: dict[str, Any] = {}
    for token_arg in ("token", "use_auth_token", "auth_token"):
        if token_arg in params:
            kwargs[token_arg] = hf_token
            break
    return kwargs


def _build_inference_init_kwargs(
    inference_cls, device: str, window: str
) -> dict[str, Any]:
    """Build ``Inference.__init__`` kwargs adapting to its signature.

    ``Inference`` no longer accepts a token in current pyannote.audio (the
    token is consumed by ``Model.from_pretrained``); we only forward
    ``window`` and ``device`` if the signature accepts them.
    """
    try:
        params = inspect.signature(inference_cls.__init__).parameters
    except (TypeError, ValueError):
        params = {}

    kwargs: dict[str, Any] = {}
    if "window" in params:
        kwargs["window"] = window
    if "device" in params:
        kwargs["device"] = device
    return kwargs


def extract_cluster_embeddings(
    audio_path: str | Path,
    segments: list[dict[str, Any]],
    *,
    hf_token: str,
    device: str = "cpu",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    min_segment_seconds: float = DEFAULT_MIN_SEGMENT_SECONDS,
) -> dict[str, dict[str, Any]]:
    """Compute one aggregated embedding per diarized speaker label.

    For each unique ``speaker`` label in ``segments``:

    1. Pick segments at least ``min_segment_seconds`` long (fallback: keep all
       if no segment qualifies — better some signal than dropping the cluster).
    2. Extract an embedding per chosen segment via ``pyannote.audio.Inference``.
    3. Aggregate by per-dimension **median** across segment embeddings.

    Returns ``{speaker_label: {"embedding": list[float], "duration_s": float,
    "n_segments": int, "embedding_model": str}}``. Clusters for which no
    segment could be embedded are omitted from the result (with a warning).

    Imports pyannote at call time; raises whatever the underlying library
    raises if pyannote is not installed or the model is inaccessible.
    """
    import numpy as np
    import torch
    import whisperx
    from pyannote.audio import Inference, Model

    by_speaker: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for seg in segments:
        speaker = seg.get("speaker")
        if not speaker:
            continue
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        if end > start:
            by_speaker[speaker].append((start, end))

    if not by_speaker:
        return {}

    logger.info(
        "Loading embedding model '%s' (device=%s)...", model_name, device
    )
    # Current pyannote.audio requires a pre-loaded Model; passing the bare
    # model_name to Inference fails with `'str' object has no attribute 'eval'`.
    pretrained_kwargs = _build_from_pretrained_kwargs(Model, hf_token)
    model = Model.from_pretrained(model_name, **pretrained_kwargs)
    # pyannote.audio's Inference stores `self.device` as-is; some of its
    # internals (e.g. fix_reproducibility) then access `device.type`, which
    # raises AttributeError if we passed a bare string. Convert up front.
    torch_device = torch.device(device) if isinstance(device, str) else device
    init_kwargs = _build_inference_init_kwargs(Inference, torch_device, window="whole")
    inference = Inference(model, **init_kwargs)

    # Load the audio via whisperx — the loader the rest of the pipeline uses
    # for ASR/diarization. ffmpeg-based via subprocess, robust across formats
    # and not subject to the torchaudio backend churn that has produced
    # segfaults in some pyannote.audio builds when passing audio in directly.
    # Always returns mono float32 at 16 kHz, which is what wespeaker expects.
    audio_path = str(audio_path)
    audio_array = whisperx.load_audio(audio_path)
    waveform = torch.from_numpy(audio_array).unsqueeze(0)  # shape (1, n_samples)
    sample_rate = 16000

    # Inference is fed each segment as a ProtocolFile dict ({"waveform",
    # "sample_rate"}); passing a path string to Inference.crop (the documented
    # short form) hits `'str' object has no attribute 'type'` in some
    # pyannote.audio builds. The dict-based path bypasses that code entirely.
    min_chunk_samples = int(0.1 * sample_rate)  # require ≥100ms of audio

    result: dict[str, dict[str, Any]] = {}

    for speaker, spans in by_speaker.items():
        long_spans = [(s, e) for (s, e) in spans if (e - s) >= min_segment_seconds]
        chosen = long_spans if long_spans else spans

        per_segment: list[Any] = []
        for start, end in chosen:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            chunk = waveform[:, start_sample:end_sample].contiguous()
            if chunk.shape[1] < min_chunk_samples:
                continue
            try:
                emb = inference({"waveform": chunk, "sample_rate": sample_rate})
            except Exception as e:
                logger.warning(
                    "Failed to embed segment %.2f-%.2f for %s: %s",
                    start, end, speaker, e,
                    exc_info=True,
                )
                continue
            per_segment.append(np.asarray(emb).reshape(-1))

        if not per_segment:
            logger.warning(
                "No embeddings extracted for cluster %s; omitting from result.",
                speaker,
            )
            continue

        stacked = np.stack(per_segment, axis=0)
        median = np.median(stacked, axis=0)

        total_duration = sum(e - s for (s, e) in chosen)
        result[speaker] = {
            "embedding": median.astype(float).tolist(),
            "duration_s": float(total_duration),
            "n_segments": len(chosen),
            "embedding_model": model_name,
        }

    return result
