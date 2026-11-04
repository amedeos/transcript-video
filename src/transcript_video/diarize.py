"""Speaker diarization via whisperX's pyannote integration.

Resolves the HuggingFace token from (in order):

1. The ``--hf-token`` CLI flag.
2. The ``HF_TOKEN`` or ``HUGGING_FACE_HUB_TOKEN`` environment variable.
3. The ``~/.cache/huggingface/token`` file (where ``huggingface-cli login`` writes it).
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger("transcript_video.diarize")


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


def _build_pipeline_init_kwargs(
    pipeline_cls, hf_token: str, device: str, model_name: str | None
) -> dict[str, Any]:
    """Build the ``__init__`` kwargs for ``DiarizationPipeline`` adapting to its signature.

    whisperX renamed ``use_auth_token`` to ``token`` to match the newer
    ``huggingface_hub`` convention; we accept either by inspecting the
    constructor signature. ``model_name`` is forwarded only when set and only
    when the constructor exposes a recognized parameter for it.
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

    if model_name:
        for model_arg in ("model_name", "model"):
            if model_arg in params:
                init_kwargs[model_arg] = model_name
                break

    return init_kwargs


def _gated_repo_message(model_name: str | None, error: Exception) -> str:
    """Build a clear hint for HF gated-repo / 403 errors during model download."""
    msg = str(error)
    repo_hint = ""
    # Try to extract the gated repo id from the error message.
    if "speaker-diarization-community-1" in msg:
        repo_hint = "pyannote/speaker-diarization-community-1"
    elif "speaker-diarization-3.1" in msg:
        repo_hint = "pyannote/speaker-diarization-3.1"
    elif model_name:
        repo_hint = model_name

    lines = [
        "Error: HuggingFace denied access to the diarization model (HTTP 403 / GatedRepo).",
        "",
        "Two distinct gating modes exist:",
        "  - 'community-1' uses MANUAL approval — accepting the terms only files a request;",
        "    the pyannote team must approve it (can take hours or days).",
        "  - 'speaker-diarization-3.1' uses AUTO approval — clicking 'Agree' grants access",
        "    immediately. Recommended if you need to start now.",
        "",
        "Steps:",
    ]
    if repo_hint:
        lines.append(f"  1. Visit https://huggingface.co/{repo_hint} and click 'Agree and access repository'.")
    else:
        lines.append("  1. Visit the model page on huggingface.co and accept its terms.")
    lines.append("  2. Confirm your token has 'read' scope at https://hf.co/settings/tokens .")
    lines.append("  3. For 'speaker-diarization-3.1' also accept https://huggingface.co/pyannote/segmentation-3.0 .")
    lines.append("  4. Re-run; the model is downloaded once and cached locally.")
    lines.append("")
    lines.append("Switch model with: --diarize-model pyannote/speaker-diarization-3.1")
    return "\n".join(lines)


def diarize_and_assign(
    aligned: dict[str, Any],
    audio,
    *,
    hf_token: str | None,
    device: str,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Run pyannote diarization and assign speakers to the aligned segments.

    Returns the updated ``aligned`` dict (segments now carry a ``speaker`` field
    where assignment was possible). ``model_name`` overrides whisperX's default
    diarization model (currently ``pyannote/speaker-diarization-community-1``);
    pass ``None`` to keep the upstream default.
    """
    if not hf_token:
        print(
            "Error: a HuggingFace token is required for pyannote diarization. "
            "Pass --hf-token, set HF_TOKEN, or run `huggingface-cli login`.",
            file=sys.stderr,
        )
        sys.exit(1)

    pipeline_cls, assign_fn = _resolve_diarization_api()
    init_kwargs = _build_pipeline_init_kwargs(pipeline_cls, hf_token, device, model_name)

    try:
        diarize_pipeline = pipeline_cls(**init_kwargs)
    except Exception as e:
        # Surface gated-repo failures with actionable instructions; re-raise others.
        msg = str(e)
        if "GatedRepo" in type(e).__name__ or "403" in msg or "Forbidden" in msg or "gated" in msg.lower():
            print(_gated_repo_message(model_name, e), file=sys.stderr)
            sys.exit(1)
        raise

    try:
        _share_embedding_backbone(diarize_pipeline)
    except Exception as e:  # a speed-up must never stop diarization
        logger.warning("Diarization: stock pyannote embeddings (%s: %s).", type(e).__name__, e)

    diarize_kwargs: dict[str, Any] = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarize_kwargs["max_speakers"] = max_speakers

    diarize_segments = diarize_pipeline(audio, **diarize_kwargs)
    return assign_fn(diarize_segments, aligned)


def _share_embedding_backbone(diarize_pipeline: Any) -> bool:
    """Make pyannote run the speaker-embedding network once per window.

    pyannote's ``SpeakerDiarization.get_embeddings`` feeds the embedding
    model one row per (window, local speaker): the same 10 s of audio, three
    times, each with that speaker's mask. Only the final statistics pooling
    reads the mask, so the network (fbank + ResNet) computes the same frame
    features three times. On CPU that is nearly all of the diarization time
    (230 of 237 s on a 2-minute clip, measured on a laptop).

    The WeSpeaker ResNet models already split the forward pass in two
    (``forward_frames`` and ``forward_embedding``), and the pipeline never
    calls the halves. This swaps the class of the pipeline's embedding
    wrapper for a subclass whose ``__call__`` runs ``forward_frames`` once
    per distinct window in the batch (the rows of one window are consecutive
    and carry the same audio) and ``forward_embedding`` once per row with
    its mask: the operations of ``forward()``. The row batch is left as it
    is, so each pooling and linear layer sees the same rows as in stock
    pyannote, and the network sees fewer windows per call (less memory).

    It relies on pyannote internals (``model._embedding`` and its
    ``model_``), so nothing is trusted blindly:

    - the pieces are feature-detected; a pipeline without them (no
      embedding model, another pyannote version) stays stock;
    - a self-check compares the shared and stock calls on a small random
      batch, and any difference or error restores the stock class;
    - an error inside a shared call during the run restores the stock
      class and redoes that batch the stock way, so diarization never
      fails because of this.

    Returns whether sharing is on.
    """
    pyannote_pipeline = getattr(diarize_pipeline, "model", None)
    embedding = getattr(pyannote_pipeline, "_embedding", None)
    if embedding is None:
        logger.info("Diarization: this pipeline has no pyannote embedding model; nothing to share.")
        return False
    network = getattr(embedding, "model_", None)
    if not (
        callable(getattr(network, "forward_frames", None))
        and callable(getattr(network, "forward_embedding", None))
    ):
        logger.info(
            "Diarization: stock pyannote embeddings (the embedding network has no "
            "forward_frames/forward_embedding split)."
        )
        return False
    base = type(embedding)
    if getattr(base, "_shares_backbone", False):
        return True

    def __call__(self, waveforms, masks=None):
        if masks is None:
            return base.__call__(self, waveforms, masks=masks)
        try:
            return _shared_backbone_embeddings(self, waveforms, masks)
        except Exception as e:  # never let the shortcut break diarization
            logger.warning(
                "Diarization: shared embedding network failed (%s: %s); "
                "falling back to stock pyannote embeddings.", type(e).__name__, e,
            )
            self.__class__ = base
            return base.__call__(self, waveforms, masks=masks)

    try:
        embedding.__class__ = type(
            f"SharedBackbone{base.__name__}", (base,), {"__call__": __call__, "_shares_backbone": True}
        )
        _self_check(embedding, base)
    except Exception as e:
        with contextlib.suppress(Exception):
            embedding.__class__ = base
        logger.warning(
            "Diarization: stock pyannote embeddings (the shared embedding network is not "
            "usable here: %s: %s).", type(e).__name__, e,
        )
        return False
    logger.info("Diarization: embedding network run once per window, shared by its local speakers.")
    return True


def _shared_backbone_embeddings(embedding: Any, waveforms, masks):
    """Embeddings for (window, speaker) rows, one ``forward_frames`` per distinct window."""
    import torch

    # Rows of the same window are consecutive and carry the same audio.
    keep, index = [0], [0]
    for i in range(1, len(waveforms)):
        if not torch.equal(waveforms[i], waveforms[keep[-1]]):
            keep.append(i)
        index.append(len(keep) - 1)
    network = embedding.model_
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frames = network.forward_frames(waveforms[keep].to(embedding.device))
        if len(keep) < len(waveforms):
            frames = frames[torch.tensor(index, device=frames.device)]
        embeddings = network.forward_embedding(frames, weights=masks.to(embedding.device))
    return embeddings.cpu().numpy()


# Self-check batch: 2 windows x 3 local speakers of 2 s noise, the pipeline's row layout.
_CHECK_WINDOWS, _CHECK_SPEAKERS, _CHECK_SECONDS, _CHECK_FRAMES = 2, 3, 2.0, 100


def _self_check(embedding: Any, base: type) -> None:
    """Raise unless the shared call reproduces the stock call on a small random batch."""
    import numpy as np
    import torch

    generator = torch.Generator().manual_seed(0)
    samples = int(_CHECK_SECONDS * getattr(embedding, "sample_rate", 16000))
    windows = 0.1 * torch.randn(_CHECK_WINDOWS, 1, 1, samples, generator=generator)
    waveforms = windows.repeat_interleave(_CHECK_SPEAKERS, dim=0).reshape(-1, 1, samples)
    masks = (torch.rand(len(waveforms), _CHECK_FRAMES, generator=generator) > 0.5).float()
    stock = base.__call__(embedding, waveforms, masks=masks)
    shared = _shared_backbone_embeddings(embedding, waveforms, masks)
    if stock.shape != shared.shape:
        raise ValueError(f"shape {shared.shape} != stock {stock.shape}")
    if not np.array_equal(np.isnan(stock), np.isnan(shared)):
        raise ValueError("NaN pattern differs from stock")
    if not np.allclose(stock, shared, rtol=1e-4, atol=1e-5, equal_nan=True):
        raise ValueError("embeddings differ from stock")


from .stats import count_unique_speakers  # noqa: E402,F401  (re-export for backward compat)
