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

    def test_pascal_gpu_hint_is_float32_not_int8(self, monkeypatch):
        # GTX 10xx (compute capability 6.1): no float16, but int8 kernels.
        # The default must stay lossless: float32, never upstream's int8.
        _install_fake_ctranslate2(
            monkeypatch, device_count=1, compute_types={"float32", "int8", "int8_float32"}
        )
        assert asr.check_cuda_available() == (True, "float32")

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

    @pytest.mark.parametrize("value", ["INT8", " Int8 "])
    def test_case_and_spaces_dont_lose_the_quantization(self, no_gpu, value):
        assert asr.resolve_compute_type("cpu", value) == "int8"

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

    @pytest.fixture
    def old_gpu(self, monkeypatch):
        """A GPU below the float16 minimum (e.g. GTX 1080), CUDA torch."""
        monkeypatch.setattr(asr, "check_cuda_available", lambda: (True, "float32"))
        monkeypatch.setattr(asr, "torch_cuda_available", lambda: True)

    def test_old_gpu_auto_runs_asr_on_cpu(self, old_gpu):
        plan = _plan(None)
        assert (plan.asr, plan.vad, plan.torch, plan.error) == ("cpu", "cpu", "cuda", None)
        assert len(plan.notes) == 1 and "below the float16 minimum" in plan.notes[0]
        assert asr.resolve_compute_type(plan.asr, None) == "float32"

    def test_old_gpu_explicit_cuda_uses_it_in_float32(self, old_gpu):
        plan = _plan("cuda")
        assert (plan.asr, plan.vad, plan.torch, plan.error) == ("cuda", "cuda", "cuda", None)
        assert "float32" in plan.notes[0]
        assert asr.resolve_compute_type(plan.asr, None) == "float32"

    def test_old_gpu_without_torch_cuda_is_all_cpu(self, monkeypatch):
        monkeypatch.setattr(asr, "check_cuda_available", lambda: (True, "float32"))
        monkeypatch.setattr(asr, "torch_cuda_available", lambda: False)
        plan = _plan(None)
        assert (plan.asr, plan.vad, plan.torch) == ("cpu", "cpu", "cpu")
        assert len(plan.notes) == 1 and "below the float16 minimum" in plan.notes[0]

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


class TestBuildAsrOptions:
    def test_beam_only_by_default(self):
        opts = asr.build_asr_options(beam_size=5, initial_prompt=None, hotwords=None, anti_loop=False)
        assert opts == {"beam_size": 5}

    def test_anti_loop_parity_values(self):
        # Reference-project parity: exactly these three values.
        opts = asr.build_asr_options(beam_size=5, initial_prompt=None, hotwords=None, anti_loop=True)
        assert opts == {
            "beam_size": 5,
            "condition_on_previous_text": False,
            "compression_ratio_threshold": 2.0,
            "no_speech_threshold": 0.5,
        }

    def test_prompt_and_hotwords(self):
        opts = asr.build_asr_options(
            beam_size=10, initial_prompt="Riunione.", hotwords="OpenShift", anti_loop=False
        )
        assert opts == {"beam_size": 10, "initial_prompt": "Riunione.", "hotwords": "OpenShift"}

    @pytest.mark.parametrize("empty", [None, ""])
    def test_empty_prompt_and_hotwords_are_omitted(self, empty):
        opts = asr.build_asr_options(beam_size=5, initial_prompt=empty, hotwords=empty, anti_loop=False)
        assert opts == {"beam_size": 5}

    def test_options_reach_whisperx_load_model(self, monkeypatch):
        seen: dict = {}
        fake = types.ModuleType("whisperx")

        def load_model(name, device, compute_type="float16", asr_options=None):
            seen.update(asr_options)
            return "model"

        fake.load_model = load_model
        monkeypatch.setitem(sys.modules, "whisperx", fake)
        opts = {"beam_size": 5, "initial_prompt": "Riunione."}
        asr.load_whisperx_model("large-v3", "cpu", "float32", opts)
        assert seen == opts


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
        asr.load_whisperx_model("large-v3", "cuda", "float16", {"beam_size": 5}, vad_device="cpu")
        assert fake_whisperx == {"device": "cuda", "vad_model": "vad@cpu"}

    @pytest.mark.parametrize("vad_device", [None, "cuda"])
    def test_whisperx_default_vad_otherwise(self, fake_whisperx, vad_device):
        asr.load_whisperx_model("large-v3", "cuda", "float16", {"beam_size": 5}, vad_device=vad_device)
        assert fake_whisperx == {"device": "cuda", "vad_model": None}

    def test_whisperx_notice_goes_to_stderr(self, monkeypatch, capsys):
        fake = types.ModuleType("whisperx")

        def load_model(name, device, compute_type="float16", asr_options=None, vad_model=None):
            print("Use manually assigned vad_model. vad_method is ignored.")
            return "model"

        fake.load_model = load_model
        monkeypatch.setitem(sys.modules, "whisperx", fake)
        monkeypatch.setattr(asr, "_build_vad", lambda device: f"vad@{device}")
        asr.load_whisperx_model("large-v3", "cuda", "float16", {"beam_size": 5}, vad_device="cpu")
        out, err = capsys.readouterr()
        assert out == "" and "manually assigned vad_model" in err

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


class TestEnvThreadLimit:
    def test_unset(self):
        assert asr.env_thread_limit() is None

    def test_omp_wins_over_mkl(self, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        monkeypatch.setenv("MKL_NUM_THREADS", "2")
        assert asr.env_thread_limit() == ("OMP_NUM_THREADS", 4)

    def test_mkl_alone(self, monkeypatch):
        monkeypatch.setenv("MKL_NUM_THREADS", "3")
        assert asr.env_thread_limit() == ("MKL_NUM_THREADS", 3)

    def test_nested_omp_list_takes_the_first_level(self, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "6,2")
        assert asr.env_thread_limit() == ("OMP_NUM_THREADS", 6)

    @pytest.mark.parametrize("raw", ["0", "-2", "four", " "])
    def test_invalid_values_are_ignored(self, monkeypatch, caplog, raw):
        monkeypatch.setenv("OMP_NUM_THREADS", raw)
        with caplog.at_level(logging.WARNING, logger="transcript_video.asr"):
            assert asr.env_thread_limit() is None
        if raw.strip():
            assert "Ignoring OMP_NUM_THREADS" in caplog.text


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
        assert asr.load_whisperx_model("large-v3", "cpu", "float32", {"beam_size": 5}, threads=7) == "model"
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

        assert asr.load_whisperx_model("large-v3", "cpu", "float32", {"beam_size": 5}, threads=7) == "model"
        assert calls == {"threads": 7}

    def test_threads_omitted_when_signature_lacks_it(self, monkeypatch):
        calls = {}

        def load_model(name, device, compute_type="float16", asr_options=None):
            calls.update(device=device, compute_type=compute_type)
            return "model"

        self._install_fake_whisperx(monkeypatch, load_model)
        assert asr.load_whisperx_model("large-v3", "cpu", "float32", {"beam_size": 5}, threads=7) == "model"
        assert calls == {"device": "cpu", "compute_type": "float32"}


class TestEngineSelection:
    """--model auto / parakeet / whisper names, and where each engine runs."""

    @pytest.mark.parametrize(
        "model, engine",
        [(None, "auto"), ("auto", "auto"), ("AUTO", "auto"), ("parakeet", "parakeet"),
         ("parakeet-tdt-0.6b-v3", "parakeet"), ("large-v3", "whisperx"), ("small", "whisperx")],
    )
    def test_requested_engine(self, model, engine):
        assert asr.requested_engine(model) == engine

    def test_resolve_model_name(self):
        assert asr.resolve_model_name("auto", asr.ENGINE_WHISPERX) == "large-v3"
        assert asr.resolve_model_name("small", asr.ENGINE_WHISPERX) == "small"
        assert asr.resolve_model_name("auto", asr.ENGINE_PARAKEET) == "parakeet-tdt-0.6b-v3"
        assert asr.resolve_model_name("Parakeet", asr.ENGINE_PARAKEET) == "parakeet-tdt-0.6b-v3"

    def test_resolve_beam_size(self):
        assert asr.resolve_beam_size(None) == 5
        assert asr.resolve_beam_size(10) == 10

    @staticmethod
    def _gpu(monkeypatch, *, ct2, torch, hint="float16"):
        monkeypatch.setattr(asr, "check_cuda_available", lambda: (ct2, hint if ct2 else None))
        monkeypatch.setattr(asr, "torch_cuda_available", lambda: torch)

    def test_auto_on_a_float16_gpu_is_whisperx_on_cuda(self, monkeypatch):
        self._gpu(monkeypatch, ct2=True, torch=True)
        plan = asr.plan_devices("auto", run_asr=True, run_torch=True, engine="auto")
        assert (plan.engine, plan.asr, plan.torch, plan.vad) == ("whisperx", "cuda", "cuda", "cuda")
        assert plan.auto_reason == "float16 GPU"

    def test_auto_mixed_devices_keep_whisperx_when_its_asr_stays_on_cuda(self, monkeypatch):
        self._gpu(monkeypatch, ct2=True, torch=False)
        monkeypatch.setattr(asr, "cpu_vad_supported", lambda: True)
        plan = asr.plan_devices("auto", run_asr=True, run_torch=True, engine="auto")
        assert (plan.engine, plan.asr, plan.vad, plan.torch) == ("whisperx", "cuda", "cpu", "cpu")

    def test_auto_is_parakeet_when_the_whisperx_asr_would_fall_back_to_cpu(self, monkeypatch):
        """A float16 GPU, but torch can't use CUDA and whisperX can't split its VAD off."""
        self._gpu(monkeypatch, ct2=True, torch=False)
        monkeypatch.setattr(asr, "cpu_vad_supported", lambda: False)
        plan = asr.plan_devices("auto", run_asr=True, run_torch=True, engine="auto")
        assert (plan.engine, plan.asr, plan.vad, plan.torch) == ("parakeet", "cpu", None, "cpu")
        assert "VAD apart" in plan.auto_reason
        assert "--model large-v3" in plan.notes[0]

    def test_explicit_parakeet_has_no_auto_reason(self, monkeypatch):
        self._gpu(monkeypatch, ct2=False, torch=False)
        assert asr.plan_devices("auto", run_asr=True, run_torch=False, engine="parakeet").auto_reason is None

    def test_auto_without_gpu_is_parakeet_with_no_torch_vad(self, monkeypatch):
        self._gpu(monkeypatch, ct2=False, torch=False)
        plan = asr.plan_devices("auto", run_asr=True, run_torch=True, engine="auto")
        assert (plan.engine, plan.asr, plan.torch, plan.vad) == ("parakeet", "cpu", "cpu", None)
        assert plan.notes == ()  # a CPU-only machine is not worth a warning
        assert plan.auto_reason == "no CUDA GPU for the whisperX ASR"

    def test_auto_with_device_cpu_is_parakeet_without_probing(self, monkeypatch):
        def no_probe():
            raise AssertionError("no CUDA probe for --device cpu")

        monkeypatch.setattr(asr, "check_cuda_available", no_probe)
        plan = asr.plan_devices("cpu", run_asr=True, run_torch=False, engine="auto")
        assert (plan.engine, plan.asr, plan.torch) == ("parakeet", "cpu", None)
        assert plan.auto_reason == "--device cpu"

    def test_auto_on_a_gpu_below_float16_is_parakeet_with_a_note(self, monkeypatch):
        self._gpu(monkeypatch, ct2=True, torch=True, hint="float32")
        plan = asr.plan_devices("auto", run_asr=True, run_torch=True, engine="auto")
        assert (plan.engine, plan.asr, plan.torch) == ("parakeet", "cpu", "cuda")
        assert "--model large-v3 --device cuda" in plan.notes[0]

    def test_auto_with_explicit_cuda_and_no_gpu_is_an_error_not_parakeet(self, monkeypatch):
        self._gpu(monkeypatch, ct2=False, torch=False)
        plan = asr.plan_devices("cuda", run_asr=True, run_torch=True, engine="auto")
        assert plan.error and "ASR (ctranslate2)" in plan.error
        assert plan.engine is None

    def test_parakeet_needs_no_torch_without_diarization(self, monkeypatch):
        self._gpu(monkeypatch, ct2=True, torch=True)
        plan = asr.plan_devices("auto", run_asr=True, run_torch=False, engine="parakeet")
        assert (plan.engine, plan.asr, plan.torch, plan.vad) == ("parakeet", "cpu", None, None)

    def test_parakeet_with_explicit_cuda_diarizes_on_the_gpu(self, monkeypatch):
        self._gpu(monkeypatch, ct2=True, torch=True)
        plan = asr.plan_devices("cuda", run_asr=True, run_torch=True, engine="parakeet")
        assert (plan.asr, plan.torch, plan.error) == ("cpu", "cuda", None)
        assert "applies to diarization" in plan.notes[0]

    def test_parakeet_with_explicit_cuda_and_no_torch_cuda_is_an_error(self, monkeypatch):
        self._gpu(monkeypatch, ct2=True, torch=False)
        plan = asr.plan_devices("cuda", run_asr=True, run_torch=True, engine="parakeet")
        assert plan.error and "diarization (torch)" in plan.error

    def test_resume_has_no_engine(self, monkeypatch):
        self._gpu(monkeypatch, ct2=True, torch=True)
        plan = asr.plan_devices("auto", run_asr=False, run_torch=True, engine="auto")
        assert (plan.engine, plan.asr, plan.torch) == (None, None, "cuda")


class TestParakeetConflicts:
    def _cfg(self, **kw):
        base = dict(language=None, initial_prompt=None, hotwords=None, beam_size=None, anti_loop=False)
        return types.SimpleNamespace(**{**base, **kw})

    def test_nothing_set(self):
        assert asr.parakeet_conflicts(self._cfg(), auto_reason="--device cpu") is None

    def test_explicitly_disabled_prompt_and_hotwords_are_fine(self):
        assert asr.parakeet_conflicts(self._cfg(initial_prompt="", hotwords="")) is None

    def test_every_option_is_named(self):
        msg = asr.parakeet_conflicts(
            self._cfg(language="it", initial_prompt="x", hotwords="y", beam_size=10, anti_loop=True),
        )
        for shown in ("--language it", "--prompt", "--hotwords", "--beam_size 10", "--anti-loop",
                      "--model large-v3"):
            assert shown in msg
        assert "auto chose" not in msg

    def test_auto_explains_the_choice(self):
        msg = asr.parakeet_conflicts(self._cfg(language="it"), auto_reason="--device cpu")
        assert "--model auto chose Parakeet (--device cpu)" in msg
