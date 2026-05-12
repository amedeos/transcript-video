"""Lightweight pre-flight checks that fail in seconds instead of minutes.

The pipeline takes minutes (ASR + alignment) before reaching the diarization
step, where most user-facing failures used to surface (missing HF token,
gated repo, whisperX API drift). The checks here detect those failures
upfront so the user is told to fix them BEFORE paying the GPU bill.

This module is intentionally non-blocking on individual import failures:
each check returns a :class:`CheckResult`; the caller decides what to do
when something is not OK.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass

from . import asr as asr_mod
from . import diarize as diarize_mod
from . import speaker_embed as speaker_embed_mod
from .config import RunConfig

logger = logging.getLogger("transcript_video.preflight")

DEFAULT_DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str = ""


def check_ffmpeg() -> CheckResult:
    """ffmpeg must be on $PATH for whisperX to load audio (and pyannote / torchaudio)."""
    if shutil.which("ffmpeg"):
        return CheckResult("ffmpeg", True, "found in PATH")
    return CheckResult(
        "ffmpeg",
        False,
        "not found in PATH; install ffmpeg (e.g. `apt install ffmpeg` / `brew install ffmpeg`)",
    )


def check_cuda(device_request: str | None) -> CheckResult:
    """Verify CUDA is usable when the user requested it (or wants auto-detect)."""
    requested = (device_request or "auto").lower()
    if requested == "cpu":
        return CheckResult("cuda", True, "skipped (--device cpu)")

    cuda_ok, hint = asr_mod.check_cuda_available()
    if cuda_ok:
        return CheckResult("cuda", True, f"available (compute_type hint: {hint})")

    if requested == "cuda":
        return CheckResult(
            "cuda", False, "--device cuda requested but no CUDA device is available"
        )
    # auto: not blocking, will silently fall back to CPU.
    return CheckResult("cuda", True, "not available; pipeline will fall back to CPU (slower)")


def _resolve_pipeline_class():
    """Best-effort lookup of ``DiarizationPipeline`` mirroring :func:`diarize._resolve_diarization_api`.

    Returns the class or ``None``. Tries (1) attribute on ``whisperx.diarize``,
    (2) attribute on ``whisperx`` (older API), (3) explicit submodule import
    (newer whisperX often does **not** eagerly load the ``diarize`` submodule
    on ``import whisperx``, so the attribute is missing until forced).
    """
    try:
        import whisperx
    except ImportError:
        return None

    pipeline_cls = getattr(
        getattr(whisperx, "diarize", None), "DiarizationPipeline", None
    ) or getattr(whisperx, "DiarizationPipeline", None)
    if pipeline_cls is not None:
        return pipeline_cls

    try:
        from whisperx.diarize import DiarizationPipeline  # noqa: PLC0415
    except ImportError:
        return None
    return DiarizationPipeline


def _resolve_assign_fn():
    """Companion of :func:`_resolve_pipeline_class` for ``assign_word_speakers``."""
    try:
        import whisperx
    except ImportError:
        return None

    fn = getattr(whisperx, "assign_word_speakers", None) or getattr(
        getattr(whisperx, "diarize", None), "assign_word_speakers", None
    )
    if fn is not None:
        return fn

    try:
        from whisperx.diarize import assign_word_speakers
    except ImportError:
        return None
    return assign_word_speakers


def check_whisperx_api() -> CheckResult:
    """Confirm whisperX is importable and the diarization API can be resolved."""
    try:
        import whisperx  # noqa: F401
    except ImportError as e:
        return CheckResult(
            "whisperx_api",
            False,
            f"whisperx not importable: {e}. Install with `uv pip install -e .`.",
        )

    pipeline_cls = _resolve_pipeline_class()
    if pipeline_cls is None:
        return CheckResult(
            "whisperx_api",
            False,
            "whisperx is installed but DiarizationPipeline cannot be resolved; "
            "try `uv pip install --upgrade whisperx`",
        )

    assign_fn = _resolve_assign_fn()
    if assign_fn is None:
        return CheckResult(
            "whisperx_api",
            False,
            "whisperx exposes DiarizationPipeline but not assign_word_speakers; "
            "try `uv pip install --upgrade whisperx`",
        )

    return CheckResult("whisperx_api", True, "DiarizationPipeline + assign_word_speakers resolved")


def check_hf_token(token: str | None, *, network: bool) -> CheckResult:
    """Verify a HuggingFace token is present, and (when ``network=True``) that it works."""
    if not token:
        return CheckResult(
            "hf_token",
            False,
            "no HuggingFace token; pass --hf-token, set HF_TOKEN, or run `huggingface-cli login`",
        )
    if not network:
        return CheckResult("hf_token", True, "present (validity not checked offline)")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        return CheckResult("hf_token", True, "present (huggingface_hub not installed; cannot verify)")

    try:
        info = HfApi().whoami(token=token)
    except Exception as e:
        return CheckResult("hf_token", False, f"token rejected by HuggingFace: {e}")

    name = info.get("name") if isinstance(info, dict) else None
    return CheckResult("hf_token", True, f"valid (user: {name or 'unknown'})")


def check_diarize_model_access(token: str | None, model_id: str | None) -> CheckResult:
    """Probe whether the gated diarization model is accessible to ``token``."""
    if not token:
        return CheckResult("diarize_model_access", True, "skipped (no token)")

    target = model_id or DEFAULT_DIARIZE_MODEL
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return CheckResult(
            "diarize_model_access",
            True,
            "skipped (huggingface_hub not installed; cannot verify)",
        )

    try:
        HfApi().model_info(target, token=token)
    except Exception as e:
        msg = str(e)
        gated = "GatedRepo" in type(e).__name__ or "403" in msg or "gated" in msg.lower()
        if gated:
            return CheckResult(
                "diarize_model_access",
                False,
                f"access denied to {target}; visit https://huggingface.co/{target} and click 'Agree'",
            )
        return CheckResult("diarize_model_access", False, f"could not verify {target}: {e}")
    return CheckResult("diarize_model_access", True, f"accessible: {target}")


def check_embedding_model_access(token: str | None, model_id: str | None) -> CheckResult:
    """Probe whether the speaker-embedding model is accessible to ``token``.

    Failure here is non-fatal: the pipeline can still produce a transcript
    without cluster embeddings (auto-identification just becomes unavailable
    for this run). The check is informational; the user decides whether to
    fix the access issue before running.
    """
    if not token:
        return CheckResult("embedding_model_access", True, "skipped (no token)")

    target = model_id or speaker_embed_mod.DEFAULT_EMBEDDING_MODEL
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return CheckResult(
            "embedding_model_access",
            True,
            "skipped (huggingface_hub not installed; cannot verify)",
        )

    try:
        HfApi().model_info(target, token=token)
    except Exception as e:
        msg = str(e)
        gated = "GatedRepo" in type(e).__name__ or "403" in msg or "gated" in msg.lower()
        if gated:
            return CheckResult(
                "embedding_model_access",
                False,
                f"access denied to {target}; visit https://huggingface.co/{target} and click 'Agree'",
            )
        return CheckResult("embedding_model_access", False, f"could not verify {target}: {e}")
    return CheckResult("embedding_model_access", True, f"accessible: {target}")


def run_preflight(config: RunConfig, *, network: bool = True) -> list[CheckResult]:
    """Run all checks relevant to ``config``. Cheap checks first, network last.

    ``network=False`` skips the HF API calls (validity + model access) so the
    pre-flight stays offline-friendly. ``--check`` toggles this on.
    """
    results: list[CheckResult] = []
    results.append(check_ffmpeg())
    results.append(check_cuda(config.device))
    results.append(check_whisperx_api())

    if not config.diarization.enabled:
        return results

    token = diarize_mod.resolve_hf_token(config.diarization.hf_token)
    results.append(check_hf_token(token, network=network))

    if network and token:
        results.append(check_diarize_model_access(token, config.diarization.model_name))
        results.append(check_embedding_model_access(token, None))

    return results


def report_results(results: list[CheckResult]) -> bool:
    """Print results via the logger; return True iff every check is OK."""
    all_ok = True
    for r in results:
        marker = "OK  " if r.ok else "FAIL"
        msg = f"[{marker}] {r.name:<22}  {r.message}"
        (logger.info if r.ok else logger.error)(msg)
        if not r.ok:
            all_ok = False
    return all_ok
