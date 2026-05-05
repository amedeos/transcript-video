"""Tests for the pre-flight checks. Network-dependent paths are exercised via mocks."""

from __future__ import annotations

import sys
import types

import pytest

from transcript_video.config import DiarizationConfig, RunConfig
from transcript_video.preflight import (
    CheckResult,
    check_cuda,
    check_diarize_model_access,
    check_ffmpeg,
    check_hf_token,
    check_whisperx_api,
    report_results,
    run_preflight,
)


@pytest.fixture
def cfg(tmp_path):
    """Minimal RunConfig pointing at a fake input file."""
    fake = tmp_path / "video.mp4"
    fake.write_bytes(b"")
    return RunConfig(input_file=fake)


class TestCheckFfmpeg:
    def test_present(self, monkeypatch):
        monkeypatch.setattr("transcript_video.preflight.shutil.which", lambda _: "/usr/bin/ffmpeg")
        r = check_ffmpeg()
        assert r.ok and "PATH" in r.message

    def test_missing(self, monkeypatch):
        monkeypatch.setattr("transcript_video.preflight.shutil.which", lambda _: None)
        r = check_ffmpeg()
        assert not r.ok and "ffmpeg" in r.message


class TestCheckCuda:
    def test_cpu_request_skipped(self, monkeypatch):
        # check_cuda_available shouldn't matter when device=cpu.
        monkeypatch.setattr(
            "transcript_video.preflight.asr_mod.check_cuda_available",
            lambda: (False, None),
        )
        r = check_cuda("cpu")
        assert r.ok and "skipped" in r.message

    def test_cuda_available(self, monkeypatch):
        monkeypatch.setattr(
            "transcript_video.preflight.asr_mod.check_cuda_available",
            lambda: (True, "float16"),
        )
        r = check_cuda("auto")
        assert r.ok and "float16" in r.message

    def test_cuda_requested_but_missing(self, monkeypatch):
        monkeypatch.setattr(
            "transcript_video.preflight.asr_mod.check_cuda_available",
            lambda: (False, None),
        )
        r = check_cuda("cuda")
        assert not r.ok and "CUDA" in r.message

    def test_auto_with_no_cuda_is_ok(self, monkeypatch):
        # auto + no CUDA = warning, not blocking.
        monkeypatch.setattr(
            "transcript_video.preflight.asr_mod.check_cuda_available",
            lambda: (False, None),
        )
        r = check_cuda("auto")
        assert r.ok and "fall back" in r.message


class TestCheckWhisperxApi:
    def test_missing_whisperx(self, monkeypatch):
        # Block the whisperx import.
        monkeypatch.setitem(sys.modules, "whisperx", None)
        r = check_whisperx_api()
        # `import whisperx` raises ImportError when sys.modules entry is None.
        assert not r.ok

    def test_diarization_pipeline_present_at_top_level(self, monkeypatch):
        fake_whisperx = types.ModuleType("whisperx")

        class FakePipeline:
            pass

        fake_whisperx.DiarizationPipeline = FakePipeline
        monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
        r = check_whisperx_api()
        assert r.ok

    def test_diarization_pipeline_in_submodule(self, monkeypatch):
        fake_whisperx = types.ModuleType("whisperx")
        fake_diarize = types.ModuleType("whisperx.diarize")

        class FakePipeline:
            pass

        fake_diarize.DiarizationPipeline = FakePipeline
        fake_whisperx.diarize = fake_diarize
        monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
        monkeypatch.setitem(sys.modules, "whisperx.diarize", fake_diarize)
        r = check_whisperx_api()
        assert r.ok

    def test_pipeline_class_missing(self, monkeypatch):
        fake_whisperx = types.ModuleType("whisperx")
        # No DiarizationPipeline attribute, no submodule.
        monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
        r = check_whisperx_api()
        assert not r.ok


class TestCheckHfToken:
    def test_no_token(self):
        r = check_hf_token(None, network=False)
        assert not r.ok

    def test_offline_token_present(self):
        r = check_hf_token("hf_xxx", network=False)
        assert r.ok and "validity not checked" in r.message

    def test_online_valid(self, monkeypatch):
        class FakeApi:
            def whoami(self, token):
                return {"name": "amedeo"}

        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.HfApi = FakeApi
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
        r = check_hf_token("hf_xxx", network=True)
        assert r.ok and "amedeo" in r.message

    def test_online_invalid(self, monkeypatch):
        class FakeApi:
            def whoami(self, token):
                raise RuntimeError("401 Unauthorized")

        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.HfApi = FakeApi
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
        r = check_hf_token("hf_xxx", network=True)
        assert not r.ok and "rejected" in r.message


class TestCheckDiarizeModelAccess:
    def test_no_token_skipped(self):
        r = check_diarize_model_access(None, "pyannote/foo")
        assert r.ok and "skipped" in r.message

    def test_accessible(self, monkeypatch):
        class FakeApi:
            def model_info(self, mid, token):
                return {"id": mid}

        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.HfApi = FakeApi
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
        r = check_diarize_model_access("hf_xxx", "pyannote/foo")
        assert r.ok and "pyannote/foo" in r.message

    def test_gated_403(self, monkeypatch):
        class GatedRepoError(Exception):
            pass

        class FakeApi:
            def model_info(self, mid, token):
                raise GatedRepoError("403 access denied")

        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.HfApi = FakeApi
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
        r = check_diarize_model_access("hf_xxx", "pyannote/foo")
        assert not r.ok and "Agree" in r.message


class TestRunPreflight:
    def test_diarization_disabled_skips_token_checks(self, cfg, monkeypatch):
        monkeypatch.setattr(
            "transcript_video.preflight.shutil.which", lambda _: "/usr/bin/ffmpeg"
        )
        monkeypatch.setattr(
            "transcript_video.preflight.asr_mod.check_cuda_available", lambda: (True, "float16")
        )
        monkeypatch.setitem(
            sys.modules,
            "whisperx",
            types.SimpleNamespace(DiarizationPipeline=type("X", (), {})),
        )
        cfg.diarization = DiarizationConfig(enabled=False)
        results = run_preflight(cfg, network=True)
        names = [r.name for r in results]
        assert "hf_token" not in names
        assert "diarize_model_access" not in names

    def test_offline_skips_network_checks(self, cfg, monkeypatch):
        monkeypatch.setattr(
            "transcript_video.preflight.shutil.which", lambda _: "/usr/bin/ffmpeg"
        )
        monkeypatch.setattr(
            "transcript_video.preflight.asr_mod.check_cuda_available", lambda: (True, "float16")
        )
        monkeypatch.setitem(
            sys.modules,
            "whisperx",
            types.SimpleNamespace(DiarizationPipeline=type("X", (), {})),
        )
        monkeypatch.setattr(
            "transcript_video.preflight.diarize_mod.resolve_hf_token", lambda _: "hf_xxx"
        )
        results = run_preflight(cfg, network=False)
        names = [r.name for r in results]
        # Token presence still checked offline.
        assert "hf_token" in names
        # Network-only check skipped.
        assert "diarize_model_access" not in names


class TestReportResults:
    def test_all_ok_returns_true(self):
        rs = [CheckResult("a", True, "x"), CheckResult("b", True, "y")]
        assert report_results(rs) is True

    def test_any_failure_returns_false(self):
        rs = [CheckResult("a", True, "x"), CheckResult("b", False, "y")]
        assert report_results(rs) is False
