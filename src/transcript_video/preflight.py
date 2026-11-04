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

import importlib.metadata
import logging
import shutil
from dataclasses import dataclass

from . import asr as asr_mod
from . import diarize as diarize_mod
from . import parakeet as parakeet_mod
from . import speaker_db as speaker_db_mod
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


def check_cuda(
    device_request: str | None,
    *,
    run_asr: bool = True,
    run_torch: bool = True,
    engine: str = asr_mod.ENGINE_WHISPERX,
) -> CheckResult:
    """Verify the requested device can be honored for the stages that will run.

    Uses :func:`asr.plan_devices`, the same rules the pipeline applies, so the
    pre-flight verdict and the actual run can't disagree. An explicit ``cuda``
    fails unless every library that will run can use it. ``auto`` never fails:
    whichever library lacks CUDA runs on CPU. ``run_asr=False`` (resume) skips
    the ctranslate2 requirement; with ``run_torch=False`` too (no
    diarization), nothing needs a device. ``engine`` is the requested ASR
    engine (:func:`asr.requested_engine`): Parakeet needs no ctranslate2.
    """
    if (device_request or "auto").lower() == "cpu":
        return CheckResult("cuda", True, "skipped (--device cpu)")
    plan = asr_mod.plan_devices(device_request, run_asr=run_asr, run_torch=run_torch, engine=engine)
    return _cuda_result(plan, device_request)


def _cuda_result(plan: asr_mod.DevicePlan, device_request: str | None) -> CheckResult:
    """The ``cuda`` check's verdict for a plan (see :func:`check_cuda`)."""
    if (device_request or "auto").lower() == "cpu":
        return CheckResult("cuda", True, "skipped (--device cpu)")
    if plan.error:
        return CheckResult("cuda", False, plan.error)
    if plan.engine == asr_mod.ENGINE_PARAKEET:
        # Parakeet is on CPU by design, not for lack of a GPU: only torch counts.
        notes = " ".join(plan.notes)
        if plan.torch == "cuda":
            message = "available: diarization on cuda (Parakeet ASR runs on cpu by design)"
        elif plan.torch == "cpu":
            message = "not available for diarization (cpu, slower); Parakeet ASR runs on cpu by design"
        else:
            message = "not needed: Parakeet runs on cpu and there is no diarization"
        return CheckResult("cuda", True, f"{message}. {notes}".strip() if notes else message)

    used = {d for d in (plan.asr, plan.torch, plan.vad) if d}
    if not used:
        return CheckResult("cuda", True, "skipped (no stage needs a device)")
    if used == {"cuda"}:
        hint = asr_mod.check_cuda_available()[1] if plan.asr else None
        detail = f"compute_type hint: {hint}" if hint else "torch"
        return CheckResult("cuda", True, f"available ({detail})")
    if used == {"cpu"}:
        message = "not available; pipeline will fall back to CPU (slower)"
        return CheckResult("cuda", True, " ".join((message, *plan.notes)))

    stages = []
    if plan.asr:
        stages.append(f"ASR on {plan.asr}")
    if plan.vad and plan.vad != plan.asr:
        stages.append(f"VAD on {plan.vad}")
    if plan.torch:
        stages.append(f"alignment/diarization on {plan.torch}")
    return CheckResult("cuda", True, f"partial: {', '.join(stages)}. {' '.join(plan.notes)}".strip())


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


def check_identify_db(voice_db_path) -> CheckResult:
    """Verify the voice DB is loadable and uses a compatible embedding model.

    Only runs when ``--identify-speakers`` is set. A *missing* file is not an
    error — it just means the DB is empty (cold start). A *malformed* file or
    an *incompatible* embedding model is an error: the auto-identification
    step would fail at pipeline time anyway, better surface it now.
    """
    db_path = speaker_db_mod.resolve_db_path(voice_db_path)
    try:
        db = speaker_db_mod.load_db(db_path)
    except Exception as e:
        return CheckResult(
            "identify_db", False, f"could not load voice DB at {db_path}: {e}"
        )

    model = speaker_embed_mod.DEFAULT_EMBEDDING_MODEL
    if not speaker_db_mod.embedding_model_compatible(db, model):
        return CheckResult(
            "identify_db",
            False,
            f"voice DB at {db_path} uses embedding_model {db.get('embedding_model')!r}; "
            f"clusters produced by this pipeline use {model!r}. Use a fresh "
            "--voice-db or remove the DB to start over.",
        )

    n = len(db.get("speakers") or {})
    return CheckResult(
        "identify_db", True,
        f"{db_path} ({n} speaker(s) enrolled)" if n else f"{db_path} (empty, will populate on first transcript-learn)",
    )


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


def check_asr_engine(config: RunConfig, plan: asr_mod.DevicePlan) -> CheckResult:
    """Report the ASR engine that will run, and why; FAIL on options Parakeet can't apply."""
    how = f" (--model auto: {plan.auto_reason})" if plan.auto_reason else ""
    if plan.engine == asr_mod.ENGINE_PARAKEET:
        conflict = asr_mod.parakeet_conflicts(config, auto_reason=plan.auto_reason)
        if conflict:
            return CheckResult("asr_engine", False, conflict)
        return CheckResult("asr_engine", True, f"Parakeet {parakeet_mod.MODEL_NAME} on cpu{how}")
    model = asr_mod.resolve_model_name(config.model, asr_mod.ENGINE_WHISPERX)
    return CheckResult("asr_engine", True, f"whisperX {model} on {plan.asr}{how}")


def check_parakeet(compute_type: str | None, *, network: bool) -> CheckResult:
    """onnx-asr importable at a supported version, ONNX Runtime's CPU provider, model files.

    Missing model files are not a failure (the first run downloads them),
    but with ``network`` each one is looked up at its pinned revision: an
    offline machine learns it now, not after audio decoding. Any error,
    an import included (a broken native wheel raises more than
    ImportError), becomes a FAIL line instead of a traceback.
    """
    name = "parakeet"
    try:
        import onnx_asr  # noqa: F401
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except Exception as e:
        return CheckResult(
            name, False,
            f"onnx-asr / onnxruntime not usable: {type(e).__name__}: {e}. "
            "Reinstall with `uv pip install -e .`.",
        )
    try:
        version = importlib.metadata.version("onnx-asr")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version is not None and not version.startswith(parakeet_mod.ONNX_ASR_SERIES):
        return CheckResult(
            name, False, f"onnx-asr {version} installed; this transcript-video needs {parakeet_mod.ONNX_ASR_SERIES}x"
        )
    if "CPUExecutionProvider" not in providers:
        return CheckResult(name, False, "ONNX Runtime has no CPUExecutionProvider")
    weights, _ = parakeet_mod.resolve_weights(compute_type)
    label = f"onnx-asr {version or '(version unknown)'}, {weights} weights"
    try:
        missing = parakeet_mod.missing_files(weights)
    except Exception as e:
        return CheckResult(name, False, f"{label}: can't read the model cache: {type(e).__name__}: {e}")
    if not missing:
        return CheckResult(name, True, f"{label} ready")
    download = (
        f"the first run downloads {len(missing)} file(s) "
        f"(~{parakeet_mod.download_estimate(weights, missing)}) from huggingface.co"
    )
    if network:
        try:
            from huggingface_hub import file_exists

            absent = [f"{repo}/{file}" for repo, revision, file in missing
                      if not file_exists(repo, file, revision=revision)]
        except Exception as e:
            return CheckResult(name, False, f"{download}, but it is not reachable: {type(e).__name__}: {e}")
        if absent:
            return CheckResult(name, False, f"not found at the pinned revision: {', '.join(absent)}")
    return CheckResult(name, True, f"{label}: {download}")


def run_preflight(config: RunConfig, *, network: bool = True) -> list[CheckResult]:
    """Run all checks relevant to ``config``. Cheap checks first, network last.

    ``network=False`` skips the HF API calls (validity + model access) so the
    pre-flight stays offline-friendly. ``--check`` toggles this on.
    """
    results: list[CheckResult] = []
    results.append(check_ffmpeg())
    # One plan for the device and engine checks: the same rules as the pipeline.
    plan = asr_mod.plan_devices(
        config.device,
        run_asr=config.runs_asr,
        run_torch=config.diarization.enabled,
        engine=asr_mod.requested_engine(config.model),
    )
    results.append(_cuda_result(plan, config.device))
    if plan.engine is not None:  # an ASR runs (not a resume) and the plan holds
        results.append(check_asr_engine(config, plan))
        if plan.engine == asr_mod.ENGINE_PARAKEET:
            results.append(check_parakeet(config.compute_type, network=network))
    results.append(check_whisperx_api())

    if not config.diarization.enabled:
        return results

    token = diarize_mod.resolve_hf_token(config.diarization.hf_token)
    results.append(check_hf_token(token, network=network))

    if network and token:
        results.append(check_diarize_model_access(token, config.diarization.model_name))
        results.append(check_embedding_model_access(token, None))

    if config.identify.enabled:
        results.append(check_identify_db(config.identify.voice_db))

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
