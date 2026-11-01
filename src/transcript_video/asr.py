"""whisperX-based ASR wrapper.

Builds the kwargs dictionary, probes CUDA capabilities, and provides a thin
loader for the whisperX model. All references to :mod:`whisperx` are local to
the functions that need them so importing this module without the heavy ML
stack installed still works (e.g. for unit tests of helpers).
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import inspect
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("transcript_video.asr")

# CTranslate2 has no float16 kernels on CPU. float32 is the lossless choice
# (precision >= the float16 used on CUDA). This deliberately diverges from the
# reference project, which used int8 on CPU and gave up a little accuracy for speed.
CPU_DEFAULT_COMPUTE_TYPE = "float32"

# Compute types that ask ctranslate2 to choose; never listed by
# get_supported_compute_types, so they bypass validation.
_SELF_RESOLVING_COMPUTE_TYPES = frozenset({"default", "auto"})

# Preferred CUDA compute types, best first (float16 first: reference-project parity).
_CUDA_PREFERENCE = ("float16", "int8_float16", "int8", "float32")


@functools.lru_cache(maxsize=1)
def _probe_cuda() -> tuple[bool, str | None, frozenset[str]]:
    """Probe CUDA once per process: ``(usable, compute_type_hint, supported_types)``.

    Cached because the answer must not change mid-run. On a machine without an
    NVIDIA driver, ctranslate2's ``get_supported_compute_types("cuda")`` raises
    on the first call but returns ``{"float32"}`` on later calls. So we also
    require ``get_cuda_device_count() > 0``, which is stable.
    """
    try:
        import ctranslate2
    except ImportError:
        return False, None, frozenset()
    try:
        if ctranslate2.get_cuda_device_count() < 1:
            return False, None, frozenset()
        types = frozenset(ctranslate2.get_supported_compute_types("cuda"))
    except Exception:
        return False, None, frozenset()
    hint = next((t for t in _CUDA_PREFERENCE if t in types), None)
    if hint is None:
        return False, None, frozenset()
    return True, hint, types


def check_cuda_available() -> tuple[bool, str | None]:
    """Return ``(cuda_available, compute_type_hint)``.

    The hint is the preferred compute type to request from ctranslate2 when CUDA
    is usable (``float16`` if supported); ``None`` otherwise.
    """
    usable, hint, _ = _probe_cuda()
    return usable, hint


def _supported_compute_types(device: str) -> frozenset[str] | None:
    """Compute types ctranslate2 supports on ``device``; ``None`` if unknown."""
    if device == "cuda":
        usable, _, types = _probe_cuda()
        return types if usable else None
    try:
        import ctranslate2
    except ImportError:
        return None
    try:
        return frozenset(ctranslate2.get_supported_compute_types(device))
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def torch_cuda_available() -> bool:
    """Whether torch can use CUDA (alignment, diarization, embeddings).

    Probed separately from ctranslate2: the two can disagree, e.g. a CPU-only
    torch wheel on a GPU machine. Cached so the answer is stable within a run.
    """
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@functools.cache
def _warn_once(message: str) -> None:
    """Log a warning once per process (resolution can run more than once per run)."""
    logger.warning(message)


# Which library runs each stage: ctranslate2 for ASR, torch for the rest.
BACKEND_CTRANSLATE2 = "ctranslate2"
BACKEND_TORCH = "torch"
_BACKEND_STAGES = {
    BACKEND_CTRANSLATE2: "ASR (ctranslate2)",
    BACKEND_TORCH: "alignment/diarization (torch)",
}


def _cuda_available(backend: str) -> bool:
    if backend == BACKEND_TORCH:
        return torch_cuda_available()
    return check_cuda_available()[0]


def _exit_cuda_required(backend: str) -> None:
    """Explicit ``cuda`` that can't be honored: stop instead of silently using CPU."""
    print(
        f"Error: device 'cuda' was requested (--device or config) but {_BACKEND_STAGES[backend]} "
        "cannot use a CUDA device. Fix the GPU setup, or pass --device auto "
        "(use CPU when there is no GPU) or --device cpu.",
        file=sys.stderr,
    )
    sys.exit(1)


def resolve_device(requested_device: str | None, backend: str = BACKEND_CTRANSLATE2) -> str:
    """Resolve the device one library (``backend``) runs on.

    - ``"cpu"`` → ``"cpu"``.
    - ``None`` (auto) → ``"cuda"`` when that library can use CUDA, otherwise
      ``"cpu"``.
    - ``"cuda"`` → ``"cuda"``. If that library can't use CUDA, print an error
      and exit: an explicit GPU request never silently degrades to hours of CPU.
    """
    if requested_device == "cpu":
        return "cpu"
    if _cuda_available(backend):
        return "cuda"
    if requested_device == "cuda":
        _exit_cuda_required(backend)
    return "cpu"


def resolve_device_and_compute(
    requested_device: str | None, requested_compute: str | None
) -> tuple[str, str]:
    """Resolve the actual device + compute type to use.

    - ``requested_device``: ``None`` (auto), ``"cuda"``, or ``"cpu"``. See
      :func:`resolve_device` for the rules (auto falls back to CPU; explicit
      ``cuda`` without a GPU exits).
    - ``requested_compute``: explicit override, otherwise picked automatically:
      the best CUDA type (``float16`` preferred) on GPU, ``float32`` on CPU.
      ``default`` / ``auto`` are passed through (ctranslate2 picks). An override
      the resolved device does not support is replaced, with a warning: an
      ``int8*`` type by the closest supported ``int8*`` type (so a quantized
      choice stays quantized, e.g. ``int8_float16`` → ``int8_float32`` on CPU),
      anything else by the device default (e.g. ``float16`` → ``float32`` on CPU).
    """
    device = resolve_device(requested_device, BACKEND_CTRANSLATE2)
    hint = check_cuda_available()[1] if device == "cuda" else None
    default = (hint or "float16") if device == "cuda" else CPU_DEFAULT_COMPUTE_TYPE

    if not requested_compute:
        return device, default
    if requested_compute in _SELF_RESOLVING_COMPUTE_TYPES:
        return device, requested_compute

    supported = _supported_compute_types(device)
    if supported is None or requested_compute in supported:
        return device, requested_compute

    replacement = default
    if requested_compute.startswith("int8"):
        replacement = next(
            (t for t in ("int8_float16", "int8_float32", "int8") if t in supported), default
        )
    _warn_once(
        f"compute_type '{requested_compute}' is not supported on {device} "
        f"(supported: {', '.join(sorted(supported))}); using '{replacement}' instead."
    )
    return device, replacement


def _logical_cpu_count() -> int:
    """Logical CPUs this process may be scheduled on (affinity-aware)."""
    if hasattr(os, "process_cpu_count"):
        count = os.process_cpu_count()
        if count:
            return count
    if hasattr(os, "sched_getaffinity"):
        try:
            count = len(os.sched_getaffinity(0))
        except OSError:
            count = 0
        if count:
            return count
    return os.cpu_count() or 1


def _quota_cpus(quota: float, period: float) -> int | None:
    if quota <= 0 or period <= 0:
        return None
    return max(1, math.ceil(quota / period))


def _read_cgroup_v2_limit(directory: Path) -> int | None:
    """``cpu.max`` holds ``"<quota> <period>"`` or ``"max <period>"`` (no limit)."""
    try:
        fields = (directory / "cpu.max").read_text().split()
    except OSError:
        return None
    if len(fields) != 2 or fields[0] == "max":
        return None
    try:
        return _quota_cpus(float(fields[0]), float(fields[1]))
    except ValueError:
        return None


def _read_cgroup_v1_limit(directory: Path) -> int | None:
    """``cpu.cfs_quota_us`` is ``-1`` when there is no limit."""
    try:
        quota = float((directory / "cpu.cfs_quota_us").read_text().strip())
        period = float((directory / "cpu.cfs_period_us").read_text().strip())
    except (OSError, ValueError):
        return None
    return _quota_cpus(quota, period)


def _cgroup_dirs(mount: Path, cgroup_path: str) -> list[Path]:
    """The process's cgroup directory and its ancestors, up to the mount point.

    Inside a container the cgroup namespace usually maps the container's own
    cgroup to the mount point, so the mount point itself is always included.
    """
    dirs: list[Path] = []
    rel = Path(cgroup_path.lstrip("/"))
    while True:
        candidate = mount / rel
        if candidate.is_dir() and candidate not in dirs:
            dirs.append(candidate)
        if rel == rel.parent:
            break
        rel = rel.parent
    if mount.is_dir() and mount not in dirs:
        dirs.append(mount)
    return dirs


def _cgroup_cpu_limit(
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> int | None:
    """CPU limit from cgroup quotas (e.g. ``podman run --cpus=4``), or ``None``.

    Affinity-based counts don't see quotas: a 4-CPU container on a 24-CPU host
    still reports 24. Supports cgroup v2 (``cpu.max``) and v1 (CFS quota).
    The tightest limit along the cgroup's ancestors wins.
    """
    try:
        lines = proc_cgroup.read_text().splitlines()
    except OSError:
        return None
    limits: list[int] = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, cgroup_path = parts
        if controllers == "":  # cgroup v2 unified hierarchy
            for d in _cgroup_dirs(cgroup_root, cgroup_path):
                limit = _read_cgroup_v2_limit(d)
                if limit:
                    limits.append(limit)
        elif "cpu" in controllers.split(","):  # cgroup v1 cpu controller
            for mount in (cgroup_root / controllers, cgroup_root / "cpu"):
                for d in _cgroup_dirs(mount, cgroup_path):
                    limit = _read_cgroup_v1_limit(d)
                    if limit:
                        limits.append(limit)
    return min(limits) if limits else None


def available_cpu_count() -> int:
    """CPUs this process can actually use: affinity-aware, capped by cgroup quotas.

    Logical CPUs come from ``os.process_cpu_count`` (3.13+), then
    ``os.sched_getaffinity`` (Linux; respects taskset), then ``os.cpu_count``.
    A container CPU quota (``--cpus``) lowers the count further.
    """
    count = _logical_cpu_count()
    limit = _cgroup_cpu_limit()
    return min(count, limit) if limit else count


def resolve_cpu_threads(requested: int | None) -> int:
    """CPU threads to use: the explicit value, else logical CPUs minus one (min 1)."""
    if requested is not None and requested > 0:
        return requested
    return max(1, available_cpu_count() - 1)


def apply_torch_threads(threads: int) -> None:
    """Set torch's intra-op thread count (alignment, diarization, embeddings)."""
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(threads)


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


def load_whisperx_model(
    model_name: str,
    device: str,
    compute_type: str,
    beam_size: int,
    threads: int | None = None,
):
    """Load a whisperX model. Imports whisperx lazily so failures are clearer.

    ``threads`` sets ctranslate2's CPU thread count. It is forwarded only when
    the installed whisperX's ``load_model`` accepts a ``threads`` kwarg.
    """
    try:
        import whisperx
    except ImportError:
        print(
            "Error: whisperx is not installed. Install with `uv pip install -e .` "
            "or `pip install whisperx`.",
            file=sys.stderr,
        )
        sys.exit(1)

    kwargs: dict[str, Any] = {
        "device": device,
        "compute_type": compute_type,
        "asr_options": {"beam_size": beam_size},
    }
    if threads is not None and _load_model_accepts(whisperx, "threads"):
        kwargs["threads"] = threads
    return whisperx.load_model(model_name, **kwargs)


def _load_model_accepts(whisperx_mod, kwarg: str) -> bool:
    """Whether whisperX's real ``load_model`` takes ``kwarg``.

    Recent whisperX exposes a lazy top-level wrapper ``load_model(*args, **kwargs)``,
    which hides the real signature. So we inspect ``whisperx.asr.load_model``
    first and the top-level function only as a fallback.
    """
    candidates = []
    with contextlib.suppress(Exception):
        candidates.append(importlib.import_module(f"{whisperx_mod.__name__}.asr").load_model)
    candidates.append(getattr(whisperx_mod, "load_model", None))
    for fn in candidates:
        try:
            if kwarg in inspect.signature(fn).parameters:
                return True
        except (TypeError, ValueError):
            continue
    return False


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
