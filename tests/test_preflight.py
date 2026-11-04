"""Tests for the pre-flight checks. Network-dependent paths are exercised via mocks."""

from __future__ import annotations

import sys
import types

import pytest

from transcript_video.config import DiarizationConfig, IdentifyConfig, RunConfig
from transcript_video.preflight import (
    CheckResult,
    check_cuda,
    check_diarize_model_access,
    check_embedding_model_access,
    check_ffmpeg,
    check_hf_token,
    check_identify_db,
    check_whisperx_api,
    report_results,
    run_preflight,
)
from transcript_video.speaker_db import save_db


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
    @staticmethod
    def _gpu(monkeypatch, *, ct2: bool, torch: bool):
        monkeypatch.setattr(
            "transcript_video.preflight.asr_mod.check_cuda_available",
            lambda: (ct2, "float16" if ct2 else None),
        )
        monkeypatch.setattr(
            "transcript_video.preflight.asr_mod.torch_cuda_available", lambda: torch
        )

    def test_cpu_request_skipped(self, monkeypatch):
        # CUDA availability shouldn't matter when device=cpu.
        self._gpu(monkeypatch, ct2=False, torch=False)
        r = check_cuda("cpu")
        assert r.ok and "skipped" in r.message

    def test_cuda_available(self, monkeypatch):
        self._gpu(monkeypatch, ct2=True, torch=True)
        r = check_cuda("auto")
        assert r.ok and "float16" in r.message

    def test_cuda_requested_but_missing_fails(self, monkeypatch):
        # Explicit --device cuda + no GPU = stop, never a silent CPU run.
        self._gpu(monkeypatch, ct2=False, torch=False)
        r = check_cuda("cuda")
        assert not r.ok
        assert "ctranslate2" in r.message and "torch" in r.message
        assert "--device auto" in r.message

    def test_cuda_requested_but_torch_lacks_cuda_fails(self, monkeypatch):
        # e.g. a CPU-only torch wheel on a GPU machine.
        self._gpu(monkeypatch, ct2=True, torch=False)
        r = check_cuda("cuda")
        assert not r.ok and "torch" in r.message and "ctranslate2" not in r.message

    def test_auto_with_no_cuda_is_ok(self, monkeypatch):
        # auto + no CUDA = falls back to CPU, not blocking.
        self._gpu(monkeypatch, ct2=False, torch=False)
        r = check_cuda("auto")
        assert r.ok and "fall back" in r.message

    def test_auto_with_partial_cuda_is_ok(self, monkeypatch):
        self._gpu(monkeypatch, ct2=True, torch=False)
        monkeypatch.setattr("transcript_video.preflight.asr_mod.cpu_vad_supported", lambda: True)
        r = check_cuda("auto")
        assert r.ok and "partial" in r.message
        assert "ASR on cuda" in r.message and "VAD on cpu" in r.message
        assert "alignment/diarization on cpu" in r.message

    def test_all_cpu_after_vad_fallback_is_not_partial(self, monkeypatch):
        # ct2 sees CUDA, torch doesn't, and whisperX can't take a CPU VAD:
        # the ASR moves to CPU too, so nothing uses the GPU.
        self._gpu(monkeypatch, ct2=True, torch=False)
        monkeypatch.setattr("transcript_video.preflight.asr_mod.cpu_vad_supported", lambda: False)
        r = check_cuda("auto")
        assert r.ok and "partial" not in r.message
        assert "fall back to CPU" in r.message and "ASR runs on cpu too" in r.message

    def test_old_gpu_is_reported_and_asr_goes_to_cpu(self, monkeypatch):
        monkeypatch.setattr(
            "transcript_video.preflight.asr_mod.check_cuda_available", lambda: (True, "float32")
        )
        monkeypatch.setattr("transcript_video.preflight.asr_mod.torch_cuda_available", lambda: True)
        r = check_cuda("auto")
        assert r.ok and "ASR on cpu" in r.message and "alignment/diarization on cuda" in r.message
        assert "below the float16 minimum" in r.message

    def test_resume_with_cuda_ignores_ctranslate2(self, monkeypatch):
        # A resume runs no ASR: only torch must be able to use CUDA.
        self._gpu(monkeypatch, ct2=False, torch=True)
        r = check_cuda("cuda", run_asr=False)
        assert r.ok and "available" in r.message

    def test_resume_without_diarization_needs_no_gpu(self, monkeypatch):
        self._gpu(monkeypatch, ct2=False, torch=False)
        r = check_cuda("cuda", run_asr=False, run_torch=False)
        assert r.ok and "no stage needs a device" in r.message

    def test_run_preflight_passes_the_stages_that_run(self, monkeypatch, cfg, tmp_path):
        self._gpu(monkeypatch, ct2=False, torch=False)
        # Only the device wiring is under test. Stub the other checks so the real
        # whisperX (when installed) isn't imported: its cached whisperx.diarize
        # would leak into the fake-module tests of check_whisperx_api.
        for name in ("check_ffmpeg", "check_whisperx_api"):
            monkeypatch.setattr(
                f"transcript_video.preflight.{name}", lambda n=name: CheckResult(n, True, "stub")
            )
        cfg.device = "cuda"
        cfg.resume_from_aligned = tmp_path / "x.aligned.json"
        cfg.diarization = DiarizationConfig(enabled=False)
        results = {r.name: r for r in run_preflight(cfg, network=False)}
        assert results["cuda"].ok


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
        fake_whisperx.assign_word_speakers = lambda diarized, aligned: aligned
        monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
        r = check_whisperx_api()
        assert r.ok

    def test_diarization_pipeline_in_submodule(self, monkeypatch):
        fake_whisperx = types.ModuleType("whisperx")
        fake_diarize = types.ModuleType("whisperx.diarize")

        class FakePipeline:
            pass

        fake_diarize.DiarizationPipeline = FakePipeline
        fake_diarize.assign_word_speakers = lambda diarized, aligned: aligned
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

    def test_submodule_only_via_explicit_import(self, monkeypatch):
        """Regression: real whisperX ≥3.4 doesn't auto-import .diarize on `import whisperx`.

        ``whisperx.diarize`` is missing from the parent module's attributes
        until something does ``from whisperx.diarize import ...`` — at which
        point Python registers the submodule. The preflight must trigger that
        import itself rather than relying on attribute lookup.
        """
        fake_whisperx = types.ModuleType("whisperx")
        fake_diarize = types.ModuleType("whisperx.diarize")

        class FakeDP:
            pass

        def fake_assign(diarized, aligned):
            return aligned

        fake_diarize.DiarizationPipeline = FakeDP
        fake_diarize.assign_word_speakers = fake_assign

        # Crucially: the parent module does NOT expose `.diarize` as an attribute,
        # but the submodule IS registered in sys.modules so `from whisperx.diarize
        # import ...` succeeds.
        monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
        monkeypatch.setitem(sys.modules, "whisperx.diarize", fake_diarize)
        assert not hasattr(fake_whisperx, "diarize")  # confirm the failure mode

        r = check_whisperx_api()
        assert r.ok, r.message

    def test_assign_word_speakers_missing(self, monkeypatch):
        """DiarizationPipeline resolves but assign_word_speakers is gone."""
        fake_whisperx = types.ModuleType("whisperx")

        class FakeDP:
            pass

        fake_whisperx.DiarizationPipeline = FakeDP
        # Note: no assign_word_speakers anywhere.
        monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
        # Make the explicit submodule import fail too.
        monkeypatch.setitem(sys.modules, "whisperx.diarize", None)
        r = check_whisperx_api()
        assert not r.ok and "assign_word_speakers" in r.message


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


class TestCheckEmbeddingModelAccess:
    def test_no_token_skipped(self):
        r = check_embedding_model_access(None, "pyannote/foo")
        assert r.ok and "skipped" in r.message

    def test_accessible(self, monkeypatch):
        class FakeApi:
            def model_info(self, mid, token):
                return {"id": mid}

        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.HfApi = FakeApi
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
        r = check_embedding_model_access("hf_xxx", "pyannote/wespeaker-foo")
        assert r.ok and "pyannote/wespeaker-foo" in r.message

    def test_uses_default_when_model_id_none(self, monkeypatch):
        seen = {}

        class FakeApi:
            def model_info(self, mid, token):
                seen["mid"] = mid
                return {"id": mid}

        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.HfApi = FakeApi
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
        r = check_embedding_model_access("hf_xxx", None)
        assert r.ok
        # The default embedding model is the wespeaker checkpoint.
        assert "wespeaker" in seen["mid"]

    def test_gated_403(self, monkeypatch):
        class GatedRepoError(Exception):
            pass

        class FakeApi:
            def model_info(self, mid, token):
                raise GatedRepoError("403 access denied")

        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.HfApi = FakeApi
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
        r = check_embedding_model_access("hf_xxx", "pyannote/wespeaker-foo")
        assert not r.ok and "Agree" in r.message


class TestCheckIdentifyDb:
    def test_missing_db_treated_as_empty(self, tmp_path):
        # cold start: missing file is OK, just means the DB hasn't been used yet.
        r = check_identify_db(tmp_path / "absent.json")
        assert r.ok
        assert "empty" in r.message

    def test_populated_db(self, tmp_path):
        from transcript_video.speaker_embed import DEFAULT_EMBEDDING_MODEL

        path = tmp_path / "voices.json"
        save_db({
            "schema_version": 1,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "speakers": {"Mario": [{"embedding": [0.1], "source": "x"}]},
        }, path)
        r = check_identify_db(path)
        assert r.ok
        assert "1 speaker" in r.message

    def test_model_mismatch_fails(self, tmp_path):
        path = tmp_path / "voices.json"
        save_db({
            "schema_version": 1,
            "embedding_model": "old-model",
            "speakers": {"X": [{"embedding": [0.1], "source": "s"}]},
        }, path)
        r = check_identify_db(path)
        assert not r.ok
        assert "old-model" in r.message
        assert "fresh --voice-db" in r.message or "start over" in r.message

    def test_malformed_db_fails(self, tmp_path):
        path = tmp_path / "voices.json"
        path.write_text("not valid json", encoding="utf-8")
        r = check_identify_db(path)
        assert not r.ok
        assert "could not load" in r.message


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
        assert "embedding_model_access" not in names

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
        # Network-only checks skipped.
        assert "diarize_model_access" not in names
        assert "embedding_model_access" not in names

    def test_diarization_enabled_includes_embedding_check(self, cfg, monkeypatch):
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

        class FakeApi:
            def whoami(self, token):
                return {"name": "u"}

            def model_info(self, mid, token):
                return {"id": mid}

        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.HfApi = FakeApi
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

        cfg.diarization = DiarizationConfig(enabled=True)
        results = run_preflight(cfg, network=True)
        names = [r.name for r in results]
        assert "diarize_model_access" in names
        assert "embedding_model_access" in names

    def test_identify_enabled_includes_db_check(self, cfg, monkeypatch, tmp_path):
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

        cfg.identify = IdentifyConfig(enabled=True, voice_db=tmp_path / "voices.json")
        # Offline so we don't depend on huggingface_hub mocking here.
        results = run_preflight(cfg, network=False)
        names = [r.name for r in results]
        assert "identify_db" in names

    def test_identify_disabled_skips_db_check(self, cfg, monkeypatch):
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
        # identify defaults to disabled.
        results = run_preflight(cfg, network=False)
        names = [r.name for r in results]
        assert "identify_db" not in names


class TestReportResults:
    def test_all_ok_returns_true(self):
        rs = [CheckResult("a", True, "x"), CheckResult("b", True, "y")]
        assert report_results(rs) is True

    def test_any_failure_returns_false(self):
        rs = [CheckResult("a", True, "x"), CheckResult("b", False, "y")]
        assert report_results(rs) is False
