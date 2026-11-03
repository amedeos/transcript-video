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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("transcript_video.asr")

# CTranslate2 has no float16 kernels on CPU. float32 is the lossless choice
# (precision >= the float16 used on CUDA). This deliberately diverges from the
# reference project, which used int8 on CPU and gave up a little accuracy for speed.
CPU_DEFAULT_COMPUTE_TYPE = "float32"

# Every compute type ctranslate2 knows, plus our "auto"/"default" aliases for
# the lossless default (see resolve_compute_type). The CLI accepts only these,
# so a typo or an unknown name stops the run instead of silently becoming
# float32 (and losing a quantization asked for to save RAM).
COMPUTE_TYPES = (
    "auto", "default",
    "float32", "float16", "bfloat16",
    "int8", "int8_float32", "int8_float16", "int8_bfloat16", "int16",
)

# Preferred CUDA compute types, best first. float16 first, as in the reference
# project. Without float16 (compute capability < 7.0, e.g. GTX 10xx), float32,
# never int8: the default must stay lossless (upstream fell back to int8).
_CUDA_PREFERENCE = ("float16", "float32")

# Minimum GPU for the ASR: ctranslate2 must support float16 on it (compute
# capability >= 7.0: GTX 16xx / RTX 20xx and newer). Below it, --device auto
# runs the ASR on CPU (float32, lossless); an explicit --device cuda still
# uses the GPU, in float32. torch stages are unaffected.
MIN_GPU_COMPUTE_TYPE = "float16"

# ASR batch size: how many 30 s VAD chunks whisperX decodes at once. Each chunk
# is padded to 30 s and decoded on its own, so the batch changes speed and
# memory, not what gets decoded (results can differ only by floating-point
# noise; recorded in parameters.batch_size). ASR RAM grows with the batch: each chunk holds
# its own encoder activations and cross-attention cache, about 0.65 GB per
# chunk for large-v3 in float32 (extrapolated from measurements on tiny and small).
# beam_size barely matters: the cross-attention cache is not copied per beam
# hypothesis. On CUDA we keep whisperX's usual 16. On CPU, 4 keeps large-v3
# float32 at roughly 10 GB peak, against about 18 GB at 16, at a modest speed cost.
GPU_DEFAULT_BATCH_SIZE = 16
CPU_DEFAULT_BATCH_SIZE = 4


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
        # Not just ImportError: a broken CUDA torch wheel (missing libcudnn,
        # libcublas, ...) raises OSError/ValueError from the import itself.
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@functools.cache
def _warn_once(message: str) -> None:
    """Log a warning once per process (resolution can run more than once per run)."""
    logger.warning(message)


_VALID_DEVICES = ("auto", "cuda", "cpu")


@dataclass(frozen=True)
class DevicePlan:
    """Where each library runs for one run, decided once before any heavy work.

    - ``asr``: the ctranslate2 device; ``None`` when no ASR runs (resume).
    - ``torch``: the device for alignment, diarization and embeddings; ``None``
      when none of them runs (resume without diarization).
    - ``vad``: where whisperX's pyannote VAD runs. It is a torch model loaded
      together with the ASR, on the ASR device, except when ctranslate2 can
      use CUDA and torch can't: then it runs on ``cpu``.
    - ``error``: set when an explicit ``cuda`` request can't be honored.
    - ``notes``: warnings worth logging (mixed devices, fallbacks).
    - ``compute_type``: the ctranslate2 compute type, filled in by the
      pipeline once ``asr`` is known (see :func:`resolve_compute_type`).
    """

    asr: str | None
    torch: str | None
    vad: str | None = None
    error: str | None = None
    notes: tuple[str, ...] = ()
    compute_type: str | None = None

    @property
    def torch_on_cpu(self) -> bool:
        """Some torch work (VAD, alignment, diarization) runs on the CPU."""
        return "cpu" in (self.torch, self.vad)

    @property
    def uses_device(self) -> bool:
        """At least one stage runs (on any device). ``vad`` implies ``asr``."""
        return any((self.asr, self.torch))


def plan_devices(requested: str | None, *, run_asr: bool, run_torch: bool) -> DevicePlan:
    """Decide the devices for the stages that will run. Never exits.

    The single source of the device rules, shared by the pre-flight check and
    the pipeline so they can't disagree:

    - ``cpu``: everything on CPU.
    - ``auto`` / ``None``: each library uses CUDA when it can, otherwise CPU.
    - ``cuda``: every library that will run must be able to use CUDA, otherwise
      ``error`` is set. An explicit GPU request never silently becomes CPU work.

    ``run_asr`` implies torch work too (whisperX's VAD and the alignment).
    """
    req = (requested or "auto").lower()
    if req not in _VALID_DEVICES:
        return DevicePlan(None, None, error=f"unknown device {requested!r}; use auto, cuda or cpu.")
    run_torch = run_torch or run_asr
    if req == "cpu":
        cpu_if = lambda runs: "cpu" if runs else None  # noqa: E731
        return DevicePlan(asr=cpu_if(run_asr), torch=cpu_if(run_torch), vad=cpu_if(run_asr))

    ct2_cuda, ct2_hint = check_cuda_available() if run_asr else (False, None)
    # A GPU below the minimum (no float16) is used only on an explicit request.
    ct2_ok = ct2_cuda and ct2_hint == MIN_GPU_COMPUTE_TYPE
    below_minimum = ct2_cuda and not ct2_ok
    torch_ok = run_torch and torch_cuda_available()

    if req == "cuda":
        missing = []
        if run_asr and not ct2_cuda:
            missing.append("ASR (ctranslate2)")
        if run_torch and not torch_ok:
            missing.append("alignment/diarization (torch)")
        if missing:
            return DevicePlan(None, None, error=(
                f"device 'cuda' was requested (--device or config) but {' and '.join(missing)} "
                "cannot use a CUDA device. Fix the GPU setup, or pass --device auto "
                "(use CPU when there is no GPU) or --device cpu."
            ))
        cuda_if = lambda runs: "cuda" if runs else None  # noqa: E731
        notes = (
            ("GPU below the float16 minimum (compute capability < 7.0): the ASR runs on "
             "cuda in float32 because --device cuda was requested.",)
            if below_minimum else ()
        )
        return DevicePlan(
            asr=cuda_if(run_asr), torch=cuda_if(run_torch), vad=cuda_if(run_asr), notes=notes
        )

    asr = ("cuda" if ct2_ok else "cpu") if run_asr else None
    torch_dev = ("cuda" if torch_ok else "cpu") if run_torch else None
    vad = asr
    notes: list[str] = []
    if below_minimum:
        notes.append(
            "GPU below the float16 minimum (compute capability < 7.0, e.g. GTX 10xx): "
            "the ASR runs on cpu in float32 (lossless, slower). Pass --device cuda to "
            "run it on this GPU in float32 anyway."
        )
    if asr == "cuda" and torch_dev == "cpu":
        if cpu_vad_supported():
            vad = "cpu"
            notes.append(
                "Mixed devices: ASR on cuda; VAD, alignment and diarization on cpu "
                "(torch cannot use CUDA on this machine)."
            )
        else:
            asr = vad = "cpu"
            notes.append(
                "torch cannot use CUDA, and this whisperX can't run its VAD apart from "
                "the ASR: the ASR runs on cpu too."
            )
    elif asr == "cpu" and torch_dev == "cuda" and not below_minimum:
        notes.append(
            "Mixed devices: ASR on cpu (ctranslate2 cannot use CUDA), "
            "alignment/diarization on cuda."
        )
    return DevicePlan(asr=asr, torch=torch_dev, vad=vad, notes=tuple(notes))


def resolve_compute_type(device: str, requested_compute: str | None) -> str:
    """The ctranslate2 compute type for ``device``.

    - Unset, ``auto`` or ``default``: ``float16`` on GPU (``float32`` on a
      GPU below the float16 minimum), ``float32`` on CPU.
      Our ``auto`` is this lossless default, **not** ctranslate2's ``auto``,
      which picks the fastest type (int8 on CPU). ``default`` resolves here too:
      whisperX would rewrite it to float16/float32 anyway, without checking the
      GPU, and the JSON should record the type that actually ran.
    - An override the device does not support is replaced, with a warning: an
      ``int8*`` type by the closest supported ``int8*`` type (so a quantized
      choice stays quantized, e.g. ``int8_float16`` → ``int8_float32`` on CPU),
      anything else by the device default (e.g. ``float16`` → ``float32`` on CPU).
    """
    hint = check_cuda_available()[1] if device == "cuda" else None
    default = (hint or "float16") if device == "cuda" else CPU_DEFAULT_COMPUTE_TYPE

    requested_compute = (requested_compute or "").strip().lower()
    if not requested_compute or requested_compute in ("auto", "default"):
        return default

    supported = _supported_compute_types(device)
    if supported is None or requested_compute in supported:
        return requested_compute

    replacement = default
    if requested_compute.startswith("int8"):
        replacement = next(
            (t for t in ("int8_float16", "int8_float32", "int8") if t in supported), default
        )
    _warn_once(
        f"compute_type '{requested_compute}' is not supported on {device} "
        f"(supported: {', '.join(sorted(supported))}); using '{replacement}' instead."
    )
    return replacement


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


# Standard thread-count variables, honored like an explicit --threads when
# --threads / config `threads` is not given (first one set wins).
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")


def env_thread_limit() -> tuple[str, int] | None:
    """``(variable, threads)`` from OMP_NUM_THREADS / MKL_NUM_THREADS, or ``None``.

    OpenMP allows a comma-separated list for nested levels ("4,2"); the first
    level is the one that counts. Invalid or non-positive values are ignored
    with a warning.
    """
    for var in THREAD_ENV_VARS:
        raw = os.environ.get(var, "").strip()
        if not raw:
            continue
        try:
            value = int(raw.split(",")[0].strip())
        except ValueError:
            value = 0
        if value > 0:
            return var, value
        _warn_once(f"Ignoring {var}={raw!r}: not a positive integer.")
    return None


def resolve_cpu_threads(requested: int | None) -> int:
    """CPU threads to use: the explicit value, else logical CPUs minus one (min 1)."""
    if requested is not None and requested > 0:
        return requested
    return max(1, available_cpu_count() - 1)


def resolve_batch_size(requested: int | None, device: str) -> int:
    """ASR batch size: the explicit value, else :data:`GPU_DEFAULT_BATCH_SIZE`
    on CUDA and :data:`CPU_DEFAULT_BATCH_SIZE` on CPU."""
    if requested is not None and requested > 0:
        return requested
    return GPU_DEFAULT_BATCH_SIZE if device == "cuda" else CPU_DEFAULT_BATCH_SIZE


def apply_torch_threads(threads: int) -> None:
    """Set torch's intra-op thread count (alignment, diarization, embeddings)."""
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(threads)


def build_asr_options(
    *,
    beam_size: int,
    initial_prompt: str | None,
    hotwords: str | None,
    anti_loop: bool,
) -> dict[str, Any]:
    """Decoding options for whisperX's ``load_model(asr_options=...)``.

    whisperX's batched ``transcribe()`` takes no decoding options (only audio,
    batch size, language, task, ...). Prompt, hotwords and the anti-loop
    thresholds must be set when the model is loaded, where they override
    whisperX's ``default_asr_options``. Passing them to ``transcribe()`` raised
    ``TypeError``.

    - ``initial_prompt`` / ``hotwords``: ``None`` and ``""`` both keep
      whisperX's default (none).
    - ``anti_loop=True`` sets the same three values as the reference project's
      ``--anti-loop``. whisperX's batched decoding (3.8) reads none of them:
      each 30 s chunk is decoded on its own, without previous-text
      conditioning or temperature fallback. They are kept for parity and in
      case whisperX starts honoring them.
    """
    options: dict[str, Any] = {"beam_size": beam_size}
    if anti_loop:
        options["condition_on_previous_text"] = False
        options["compression_ratio_threshold"] = 2.0
        options["no_speech_threshold"] = 0.5
    if initial_prompt:
        options["initial_prompt"] = initial_prompt
    if hotwords:
        options["hotwords"] = hotwords
    return options


# whisperX's default VAD onset (``default_vad_options`` in ``whisperx.asr.load_model``).
# Only the onset reaches the Pyannote VAD constructor; offset and chunk size are
# still applied by load_model, to a manually assigned VAD as well.
_WHISPERX_VAD_ONSET = 0.500


def cpu_vad_supported() -> bool:
    """Whether whisperX can take a VAD built on another device than the ASR.

    Needs ``whisperx.vads.Pyannote`` and a ``load_model`` with a ``vad_model``
    parameter (whisperX >= 3.4). Without them, whisperX always builds its
    pyannote VAD on the ASR device.
    """
    try:
        import whisperx

        vads = importlib.import_module("whisperx.vads")
    except Exception:
        return False
    return hasattr(vads, "Pyannote") and _load_model_accepts(whisperx, "vad_model")


def _build_vad(device: str):
    """whisperX's pyannote VAD, built on ``device`` instead of the ASR device."""
    import torch
    from whisperx.vads import Pyannote

    return Pyannote(torch.device(device), token=None, vad_onset=_WHISPERX_VAD_ONSET)


def load_whisperx_model(
    model_name: str,
    device: str,
    compute_type: str,
    asr_options: dict[str, Any],
    threads: int | None = None,
    vad_device: str | None = None,
):
    """Load a whisperX model. Imports whisperx lazily so failures are clearer.

    ``asr_options`` come from :func:`build_asr_options` (beam size, prompt,
    hotwords, anti-loop thresholds).

    ``threads`` sets ctranslate2's CPU thread count. It is forwarded only when
    the installed whisperX's ``load_model`` accepts a ``threads`` kwarg.

    ``vad_device``: where whisperX's pyannote VAD (a torch model) runs. When it
    differs from ``device`` (ASR on CUDA, CPU-only torch), the VAD is built
    here and passed in; otherwise whisperX builds it on ``device``.
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
        "asr_options": dict(asr_options),
    }
    if threads is not None and _load_model_accepts(whisperx, "threads"):
        kwargs["threads"] = threads
    if vad_device is not None and vad_device != device:
        kwargs["vad_model"] = _build_vad(vad_device)
        # whisperX print()s "Use manually assigned vad_model..." here; keep stdout
        # for our "Files written:" summary, which scripts may parse.
        with contextlib.redirect_stdout(sys.stderr):
            return whisperx.load_model(model_name, **kwargs)
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
