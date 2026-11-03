"""Unit tests for device / compute-type / thread resolution in :mod:`asr`.

ctranslate2 and whisperX are mocked: these tests run without a GPU and
without the heavy ML stack.
"""

from __future__ import annotations

import logging
import sys
import types

import pytest

from transcript_video import asr

CUDA_TYPES = {"float16", "int8_float16", "int8", "float32", "int8_float32"}
CPU_TYPES = {"int8", "int8_float32", "float32"}


def _clear_all():
    asr._probe_cuda.cache_clear()
    asr.torch_cuda_available.cache_clear()
    asr._warn_once.cache_clear()


@pytest.fixture(autouse=True)
def _clear_caches():
    _clear_all()
    yield
    _clear_all()


def _install_fake_ctranslate2(monkeypatch, *, device_count, compute_types):
    """``compute_types``: a set, or a callable for stateful behavior."""
    fake = types.ModuleType("ctranslate2")
    fake.get_cuda_device_count = lambda: device_count

    def get_supported_compute_types(device):
        if device != "cuda":
            return CPU_TYPES
        return compute_types() if callable(compute_types) else compute_types

    fake.get_supported_compute_types = get_supported_compute_types
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)


class TestCudaProbe:
    def test_flaky_probe_without_driver_is_consistently_false(self, monkeypatch):
        # Real ctranslate2 with no NVIDIA driver: the first cuda probe raises,
        # later ones return {"float32"}, while the device count stays at 0.
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("CUDA driver version is insufficient")
            return {"float32"}

        _install_fake_ctranslate2(monkeypatch, device_count=0, compute_types=flaky)
        results = {asr.check_cuda_available() for _ in range(3)}
        assert results == {(False, None)}
        plan = asr.plan_devices(None, run_asr=True, run_torch=True)
        assert plan.asr == "cpu"
        assert asr.resolve_compute_type(plan.asr, None) == "float32"

    def test_probe_is_cached(self, monkeypatch):
        calls = {"n": 0}

        def counting():
            calls["n"] += 1
            return CUDA_TYPES

        _install_fake_ctranslate2(monkeypatch, device_count=1, compute_types=counting)
        for _ in range(3):
            asr.check_cuda_available()
        assert calls["n"] == 1

    def test_hint_prefers_float16(self, monkeypatch):
        _install_fake_ctranslate2(monkeypatch, device_count=1, compute_types=CUDA_TYPES)
        assert asr.check_cuda_available() == (True, "float16")

    def test_hint_is_always_a_supported_type(self, monkeypatch):
        # Old GPU (< compute capability 6.1): no float16/int8 kernels.
        _install_fake_ctranslate2(monkeypatch, device_count=1, compute_types={"float32"})
        assert asr.check_cuda_available() == (True, "float32")

    def test_no_ctranslate2(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ctranslate2", None)  # import raises ImportError
        assert asr.check_cuda_available() == (False, None)


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(asr, "check_cuda_available", lambda: (True, "float16"))
    monkeypatch.setattr(asr, "torch_cuda_available", lambda: True)
    monkeypatch.setattr(
        asr, "_supported_compute_types", lambda d: CUDA_TYPES if d == "cuda" else CPU_TYPES
    )


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(asr, "check_cuda_available", lambda: (False, None))
    monkeypatch.setattr(asr, "torch_cuda_available", lambda: False)
    monkeypatch.setattr(asr, "_supported_compute_types", lambda d: CPU_TYPES)


class TestTorchCudaProbe:
    def _install_fake_torch(self, monkeypatch, is_available):
        fake = types.ModuleType("torch")
        fake.cuda = types.SimpleNamespace(is_available=is_available)
        monkeypatch.setitem(sys.modules, "torch", fake)

    def test_available(self, monkeypatch):
        self._install_fake_torch(monkeypatch, lambda: True)
        assert asr.torch_cuda_available() is True

    def test_not_available(self, monkeypatch):
        self._install_fake_torch(monkeypatch, lambda: False)
        assert asr.torch_cuda_available() is False

    def test_probe_error_means_unavailable(self, monkeypatch):
        def boom():
            raise RuntimeError("driver")

        self._install_fake_torch(monkeypatch, boom)
        assert asr.torch_cuda_available() is False

    def test_no_torch(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        assert asr.torch_cuda_available() is False

    def test_broken_cuda_wheel_means_unavailable(self, monkeypatch):
        # A CUDA torch wheel with missing libcudnn raises OSError on import.
        import builtins

        real_import = builtins.__import__

        def _import(name, *a, **kw):
            if name == "torch":
                raise OSError("libcudnn.so.9: cannot open shared object file")
            return real_import(name, *a, **kw)

        monkeypatch.delitem(sys.modules, "torch", raising=False)
        monkeypatch.setattr(builtins, "__import__", _import)
        assert asr.torch_cuda_available() is False


class TestResolveComputeType:
    def test_cuda_default(self, gpu):
        assert asr.resolve_compute_type("cuda", None) == "float16"

    def test_cpu_default_is_lossless_float32(self, no_gpu):
        assert asr.resolve_compute_type("cpu", None) == "float32"

    @pytest.mark.parametrize(("device", "expected"), [("cpu", "float32"), ("cuda", "float16")])
    def test_auto_is_our_lossless_default(self, gpu, device, expected):
        # Not ctranslate2's "auto" (fastest type, i.e. int8 on CPU).
        assert asr.resolve_compute_type(device, "auto") == expected

    def test_default_resolves_to_the_real_type(self, monkeypatch):
        # whisperX would turn "default" into float16 on any GPU, even one
        # without float16 kernels; resolve it with the probe instead.
        monkeypatch.setattr(asr, "check_cuda_available", lambda: (True, "int8_float16"))
        assert asr.resolve_compute_type("cuda", "default") == "int8_float16"
        assert asr.resolve_compute_type("cpu", "default") == "float32"

    def test_explicit_int8_on_cpu_is_kept(self, no_gpu):
        assert asr.resolve_compute_type("cpu", "int8") == "int8"

    def test_float16_on_cpu_becomes_float32(self, no_gpu, caplog):
        with caplog.at_level(logging.WARNING, logger="transcript_video.asr"):
            assert asr.resolve_compute_type("cpu", "float16") == "float32"
        assert "not supported on cpu" in caplog.text

    def test_supported_override_on_cuda_is_kept(self, gpu):
        assert asr.resolve_compute_type("cuda", "int8_float16") == "int8_float16"

    def test_unsupported_override_on_cuda_uses_cuda_default(self, gpu, caplog):
        with caplog.at_level(logging.WARNING, logger="transcript_video.asr"):
            assert asr.resolve_compute_type("cuda", "bfloat16") == "float16"
        assert "not supported on cuda" in caplog.text

    def test_quantized_override_stays_quantized_on_cpu(self, no_gpu, caplog):
        # A GPU config with int8_float16 used on a laptop: keep int8, don't jump to float32.
        with caplog.at_level(logging.WARNING, logger="transcript_video.asr"):
            assert asr.resolve_compute_type("cpu", "int8_float16") == "int8_float32"
        assert "using 'int8_float32'" in caplog.text

    def test_compute_type_warning_logged_once(self, no_gpu, caplog):
        with caplog.at_level(logging.WARNING, logger="transcript_video.asr"):
            asr.resolve_compute_type("cpu", "float16")
            asr.resolve_compute_type("cpu", "float16")
        assert caplog.text.count("not supported on cpu") == 1

    def test_override_kept_when_supported_types_unknown(self, monkeypatch):
        monkeypatch.setattr(asr, "check_cuda_available", lambda: (False, None))
        monkeypatch.setattr(asr, "_supported_compute_types", lambda d: None)
        assert asr.resolve_compute_type("cpu", "int8") == "int8"

    def test_cuda_hint_is_used(self, monkeypatch):
        monkeypatch.setattr(asr, "check_cuda_available", lambda: (True, "int8_float16"))
        assert asr.resolve_compute_type("cuda", None) == "int8_float16"


def _plan(requested, *, run_asr=True, run_torch=True):
    return asr.plan_devices(requested, run_asr=run_asr, run_torch=run_torch)


class TestPlanDevices:
    @pytest.fixture
    def ct2_only(self, monkeypatch):
        """GPU visible to ctranslate2, but a CPU-only torch wheel is installed."""
        monkeypatch.setattr(asr, "check_cuda_available", lambda: (True, "float16"))
        monkeypatch.setattr(asr, "torch_cuda_available", lambda: False)

    def test_cpu(self, gpu):
        plan = _plan("cpu")
        assert (plan.asr, plan.torch, plan.vad, plan.error) == ("cpu", "cpu", "cpu", None)

    @pytest.mark.parametrize("requested", [None, "auto", "AUTO"])
    def test_auto_gpu(self, gpu, requested):
        plan = _plan(requested)
        assert (plan.asr, plan.torch, plan.vad, plan.notes) == ("cuda", "cuda", "cuda", ())

    def test_auto_without_gpu(self, no_gpu):
        plan = _plan(None)
        assert (plan.asr, plan.torch, plan.vad, plan.error) == ("cpu", "cpu", "cpu", None)

    def test_cuda_without_gpu_is_an_error(self, no_gpu):
        plan = _plan("cuda")
        assert plan.error and "ASR (ctranslate2) and alignment/diarization (torch)" in plan.error
        assert "--device auto" in plan.error

    def test_explicit_cuda_needs_torch_too(self, ct2_only):
        plan = _plan("cuda")
        assert plan.error and "alignment/diarization (torch)" in plan.error
        assert "ctranslate2" not in plan.error

    def test_ct2_only_runs_vad_on_cpu(self, ct2_only, monkeypatch):
        monkeypatch.setattr(asr, "cpu_vad_supported", lambda: True)
        plan = _plan(None)
        assert (plan.asr, plan.vad, plan.torch) == ("cuda", "cpu", "cpu")
        assert "VAD, alignment and diarization on cpu" in plan.notes[0]

    def test_ct2_only_without_cpu_vad_support_puts_asr_on_cpu(self, ct2_only, monkeypatch):
        monkeypatch.setattr(asr, "cpu_vad_supported", lambda: False)
        plan = _plan(None)
        assert (plan.asr, plan.vad, plan.torch) == ("cpu", "cpu", "cpu")
        assert "ASR runs on cpu too" in plan.notes[0]

    def test_torch_only_keeps_vad_with_asr_on_cpu(self, monkeypatch):
        monkeypatch.setattr(asr, "check_cuda_available", lambda: (False, None))
        monkeypatch.setattr(asr, "torch_cuda_available", lambda: True)
        plan = _plan(None)
        assert (plan.asr, plan.vad, plan.torch) == ("cpu", "cpu", "cuda")
        assert "Mixed devices" in plan.notes[0]

    def test_resume_with_cuda_needs_only_torch(self, monkeypatch):
        # ctranslate2 lacks CUDA, but a resume runs no ASR.
        monkeypatch.setattr(asr, "check_cuda_available", lambda: (False, None))
        monkeypatch.setattr(asr, "torch_cuda_available", lambda: True)
        plan = _plan("cuda", run_asr=False)
        assert (plan.asr, plan.vad, plan.torch, plan.error) == (None, None, "cuda", None)

    def test_resume_without_diarization_needs_no_device(self, no_gpu):
        plan = _plan("cuda", run_asr=False, run_torch=False)
        assert (plan.asr, plan.torch, plan.error) == (None, None, None)

    def test_asr_implies_torch(self, gpu):
        assert _plan(None, run_torch=False).torch == "cuda"

    @pytest.mark.parametrize("requested", ["gpu", "cuda:0"])
    def test_unknown_device_is_an_error(self, gpu, requested):
        assert "unknown device" in _plan(requested).error

    def test_probes_skipped_for_stages_that_dont_run(self, monkeypatch):
        def _boom():
            raise AssertionError("probed a library that runs no stage")

        monkeypatch.setattr(asr, "check_cuda_available", _boom)
        monkeypatch.setattr(asr, "torch_cuda_available", _boom)
        _plan("cuda", run_asr=False, run_torch=False)


class TestVadDevice:
    """whisperX builds its pyannote VAD on the ASR device unless we pass one."""

    @pytest.fixture
    def fake_whisperx(self, monkeypatch):
        calls: dict = {}
        fake = types.ModuleType("whisperx")

        def load_model(name, device, compute_type="float16", asr_options=None, vad_model=None):
            calls.update(device=device, vad_model=vad_model)
            return "model"

        fake.load_model = load_model
        monkeypatch.setitem(sys.modules, "whisperx", fake)
        monkeypatch.setattr(asr, "_build_vad", lambda device: f"vad@{device}")
        return calls

    def test_vad_built_on_cpu_when_it_differs(self, fake_whisperx):
        asr.load_whisperx_model("large-v3", "cuda", "float16", 5, vad_device="cpu")
        assert fake_whisperx == {"device": "cuda", "vad_model": "vad@cpu"}

    @pytest.mark.parametrize("vad_device", [None, "cuda"])
    def test_whisperx_default_vad_otherwise(self, fake_whisperx, vad_device):
        asr.load_whisperx_model("large-v3", "cuda", "float16", 5, vad_device=vad_device)
        assert fake_whisperx == {"device": "cuda", "vad_model": None}

    def test_cpu_vad_unsupported_without_whisperx(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "whisperx", None)
        assert asr.cpu_vad_supported() is False


class TestBatchSize:
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_explicit_value_wins(self, device):
        assert asr.resolve_batch_size(3, device) == 3

    def test_cuda_default(self):
        assert asr.resolve_batch_size(None, "cuda") == 16

    def test_cpu_default(self):
        assert asr.resolve_batch_size(None, "cpu") == 4

    def test_non_positive_means_auto(self):
        assert asr.resolve_batch_size(0, "cpu") == 4


class TestThreads:
    def test_explicit_value_wins(self, monkeypatch):
        monkeypatch.setattr(asr, "available_cpu_count", lambda: 8)
        assert asr.resolve_cpu_threads(3) == 3

    def test_default_is_cpus_minus_one(self, monkeypatch):
        monkeypatch.setattr(asr, "available_cpu_count", lambda: 8)
        assert asr.resolve_cpu_threads(None) == 7

    def test_single_cpu_keeps_one_thread(self, monkeypatch):
        monkeypatch.setattr(asr, "available_cpu_count", lambda: 1)
        assert asr.resolve_cpu_threads(None) == 1

    def test_available_cpu_count_is_positive(self):
        assert asr.available_cpu_count() >= 1

    def test_logical_count_without_affinity_apis(self, monkeypatch):
        # macOS / Windows on Python < 3.13: only os.cpu_count exists.
        monkeypatch.delattr(asr.os, "process_cpu_count", raising=False)
        monkeypatch.delattr(asr.os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(asr.os, "cpu_count", lambda: 6)
        assert asr._logical_cpu_count() == 6

    def test_logical_count_when_cpu_count_unknown(self, monkeypatch):
        monkeypatch.delattr(asr.os, "process_cpu_count", raising=False)
        monkeypatch.delattr(asr.os, "sched_getaffinity", raising=False)
        monkeypatch.setattr(asr.os, "cpu_count", lambda: None)
        assert asr._logical_cpu_count() == 1

    def test_cgroup_quota_caps_the_count(self, monkeypatch):
        # podman run --cpus=4 on a 24-CPU host.
        monkeypatch.setattr(asr, "_logical_cpu_count", lambda: 24)
        monkeypatch.setattr(asr, "_cgroup_cpu_limit", lambda: 4)
        assert asr.available_cpu_count() == 4
        assert asr.resolve_cpu_threads(None) == 3

    def test_no_cgroup_quota(self, monkeypatch):
        monkeypatch.setattr(asr, "_logical_cpu_count", lambda: 8)
        monkeypatch.setattr(asr, "_cgroup_cpu_limit", lambda: None)
        assert asr.available_cpu_count() == 8

    def test_quota_above_logical_count_is_ignored(self, monkeypatch):
        monkeypatch.setattr(asr, "_logical_cpu_count", lambda: 8)
        monkeypatch.setattr(asr, "_cgroup_cpu_limit", lambda: 32)
        assert asr.available_cpu_count() == 8


class TestCgroupCpuLimit:
    def _proc(self, tmp_path, content):
        p = tmp_path / "proc_cgroup"
        p.write_text(content)
        return p

    def test_v2_quota_in_namespaced_container(self, tmp_path):
        root = tmp_path / "cgroup"
        root.mkdir()
        (root / "cpu.max").write_text("400000 100000\n")
        assert asr._cgroup_cpu_limit(self._proc(tmp_path, "0::/\n"), root) == 4

    def test_v2_fractional_quota_rounds_up(self, tmp_path):
        root = tmp_path / "cgroup"
        root.mkdir()
        (root / "cpu.max").write_text("150000 100000\n")
        assert asr._cgroup_cpu_limit(self._proc(tmp_path, "0::/\n"), root) == 2

    def test_v2_unlimited(self, tmp_path):
        root = tmp_path / "cgroup"
        root.mkdir()
        (root / "cpu.max").write_text("max 100000\n")
        assert asr._cgroup_cpu_limit(self._proc(tmp_path, "0::/\n"), root) is None

    def test_v2_tightest_ancestor_wins(self, tmp_path):
        root = tmp_path / "cgroup"
        leaf = root / "machine.slice" / "libpod-abc.scope"
        leaf.mkdir(parents=True)
        (leaf / "cpu.max").write_text("max 100000\n")
        (root / "machine.slice" / "cpu.max").write_text("200000 100000\n")
        proc = self._proc(tmp_path, "0::/machine.slice/libpod-abc.scope\n")
        assert asr._cgroup_cpu_limit(proc, root) == 2

    def test_v1_cfs_quota(self, tmp_path):
        root = tmp_path / "cgroup"
        cpu = root / "cpu,cpuacct"
        cpu.mkdir(parents=True)
        (cpu / "cpu.cfs_quota_us").write_text("300000\n")
        (cpu / "cpu.cfs_period_us").write_text("100000\n")
        proc = self._proc(tmp_path, "4:cpu,cpuacct:/docker/abc\n3:memory:/docker/abc\n")
        assert asr._cgroup_cpu_limit(proc, root) == 3

    def test_v1_unlimited(self, tmp_path):
        root = tmp_path / "cgroup"
        cpu = root / "cpu,cpuacct"
        cpu.mkdir(parents=True)
        (cpu / "cpu.cfs_quota_us").write_text("-1\n")
        (cpu / "cpu.cfs_period_us").write_text("100000\n")
        proc = self._proc(tmp_path, "4:cpu,cpuacct:/\n")
        assert asr._cgroup_cpu_limit(proc, root) is None

    def test_no_proc_cgroup(self, tmp_path):
        assert asr._cgroup_cpu_limit(tmp_path / "missing", tmp_path) is None


class TestLoadWhisperxModelThreads:
    def _install_fake_whisperx(self, monkeypatch, load_model):
        # Drop a real whisperx.asr possibly imported by earlier tests, so the
        # signature lookup only sees the fake.
        monkeypatch.delitem(sys.modules, "whisperx.asr", raising=False)
        fake = types.ModuleType("whisperx")
        fake.load_model = load_model
        monkeypatch.setitem(sys.modules, "whisperx", fake)

    def test_threads_forwarded_when_supported(self, monkeypatch):
        calls = {}

        def load_model(name, device, compute_type="float16", asr_options=None, threads=4):
            calls.update(device=device, compute_type=compute_type, threads=threads)
            return "model"

        self._install_fake_whisperx(monkeypatch, load_model)
        assert asr.load_whisperx_model("large-v3", "cpu", "float32", 5, threads=7) == "model"
        assert calls == {"device": "cpu", "compute_type": "float32", "threads": 7}

    def test_threads_forwarded_through_lazy_wrapper(self, monkeypatch):
        # Recent whisperX: top-level load_model(*args, **kwargs) wraps whisperx.asr.load_model.
        calls = {}

        def real_load_model(name, device, compute_type="float16", asr_options=None, threads=4):
            calls.update(threads=threads)
            return "model"

        fake = types.ModuleType("whisperx")
        fake_asr = types.ModuleType("whisperx.asr")
        fake_asr.load_model = real_load_model
        fake.asr = fake_asr
        fake.load_model = lambda *a, **kw: fake_asr.load_model(*a, **kw)
        monkeypatch.setitem(sys.modules, "whisperx", fake)
        monkeypatch.setitem(sys.modules, "whisperx.asr", fake_asr)

        assert asr.load_whisperx_model("large-v3", "cpu", "float32", 5, threads=7) == "model"
        assert calls == {"threads": 7}

    def test_threads_omitted_when_signature_lacks_it(self, monkeypatch):
        calls = {}

        def load_model(name, device, compute_type="float16", asr_options=None):
            calls.update(device=device, compute_type=compute_type)
            return "model"

        self._install_fake_whisperx(monkeypatch, load_model)
        assert asr.load_whisperx_model("large-v3", "cpu", "float32", 5, threads=7) == "model"
        assert calls == {"device": "cpu", "compute_type": "float32"}
