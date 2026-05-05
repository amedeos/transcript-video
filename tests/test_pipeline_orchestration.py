"""Mocked end-to-end orchestration tests for the transcription pipeline.

These tests run **without** torch/whisperX installed: every heavy boundary in
:mod:`pipeline` is monkeypatched. They guard the orchestration logic itself
(resume path, aligned snapshot write, output selection, payload construction)
against regressions like the ones we hit when whisperX renamed
``DiarizationPipeline``'s constructor argument.

Tests requiring a real whisperX install belong in a separate module and use
the ``integration`` pytest marker (skipped in CI by default).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transcript_video import pipeline as pipeline_mod
from transcript_video.config import (
    DiarizationConfig,
    FrontmatterConfig,
    OutputConfig,
    RunConfig,
)


@pytest.fixture
def fake_audio():
    """Sentinel returned by the mocked load_audio."""
    return object()


@pytest.fixture
def fake_input(tmp_path: Path) -> Path:
    p = tmp_path / "video.mp4"
    p.write_bytes(b"\x00" * 16)
    return p


@pytest.fixture
def stub_pipeline(monkeypatch, fake_audio):
    """Stub all torch/whisperX boundaries used by run_pipeline."""

    class FakeModel:
        def transcribe(self, audio, batch_size, **kwargs):
            return {
                "language": "it",
                "language_probability": 0.95,
                "segments": [
                    {"id": 0, "start": 0.0, "end": 2.0, "text": "Ciao a tutti"},
                    {"id": 1, "start": 2.5, "end": 5.0, "text": "Iniziamo subito"},
                ],
            }

    monkeypatch.setattr(pipeline_mod.asr_mod, "resolve_device_and_compute", lambda d, c: ("cuda", "float16"))
    monkeypatch.setattr(pipeline_mod.asr_mod, "load_whisperx_model", lambda *a, **kw: FakeModel())
    monkeypatch.setattr(pipeline_mod.asr_mod, "load_audio", lambda _path: fake_audio)
    monkeypatch.setattr(
        pipeline_mod.asr_mod,
        "align_segments",
        lambda segs, lang, audio, device: {"segments": segs, "word_segments": []},
    )

    def _fake_diarize(aligned, audio, **kwargs):
        # Assign alternating speakers so stats come out non-trivial.
        out = []
        labels = ["SPEAKER_00", "SPEAKER_01"]
        for i, s in enumerate(aligned.get("segments", [])):
            out.append({**s, "speaker": labels[i % 2]})
        return {"segments": out, "word_segments": []}

    monkeypatch.setattr(pipeline_mod.diarize_mod, "diarize_and_assign", _fake_diarize)
    monkeypatch.setattr(pipeline_mod.diarize_mod, "resolve_hf_token", lambda _: "hf_fake")


def _config(input_file: Path, *, write_md=True) -> RunConfig:
    return RunConfig(
        input_file=input_file,
        diarization=DiarizationConfig(enabled=True, hf_token="hf_fake"),
        output=OutputConfig(write_json=True, write_md=write_md),
        frontmatter=FrontmatterConfig(date="2026-05-05", tags=["demo"]),
    )


class TestRunPipelineHappyPath:
    def test_full_run_writes_expected_artifacts(self, stub_pipeline, fake_input, tmp_path):
        config = _config(fake_input)
        written = pipeline_mod.run_pipeline(config)

        assert "json" in written
        assert "md" in written
        # Aligned snapshot was written as a free safety net.
        aligned_path = tmp_path / "video_transcript.aligned.json"
        assert aligned_path.exists()
        # Final JSON has stage="complete".
        final = json.loads(written["json"].read_text(encoding="utf-8"))
        assert final["stage"] == "complete"
        # Speaker stats populated.
        assert set(final["stats"]["speakers"].keys()) == {"SPEAKER_00", "SPEAKER_01"}
        # MD has speaker headings.
        md = written["md"].read_text(encoding="utf-8")
        assert "## [00:00:00] SPEAKER_00" in md

    def test_no_diarize_skips_aligned_snapshot(self, stub_pipeline, fake_input, tmp_path):
        config = _config(fake_input)
        config.diarization = DiarizationConfig(enabled=False)
        pipeline_mod.run_pipeline(config)

        # No diarization → no need for the aligned safety net.
        assert not (tmp_path / "video_transcript.aligned.json").exists()
        # Final JSON still produced.
        assert (tmp_path / "video_transcript.json").exists()


class TestRunPipelineDiarizeFailure:
    def test_aligned_snapshot_survives_diarize_crash(self, stub_pipeline, fake_input, tmp_path, monkeypatch, caplog):
        """If diarization aborts, the aligned snapshot is still on disk."""
        import sys as _sys

        def _explode(*a, **kw):
            print("Error: gated repo", file=_sys.stderr)
            _sys.exit(1)

        monkeypatch.setattr(pipeline_mod.diarize_mod, "diarize_and_assign", _explode)

        config = _config(fake_input)
        with pytest.raises(SystemExit):
            pipeline_mod.run_pipeline(config)

        aligned_path = tmp_path / "video_transcript.aligned.json"
        assert aligned_path.exists()
        snapshot = json.loads(aligned_path.read_text(encoding="utf-8"))
        assert snapshot["stage"] == "aligned"
        # Segments preserved without speaker attributions.
        for seg in snapshot["segments"]:
            assert "speaker" not in seg


class TestResumeFromAligned:
    def test_resume_skips_asr_and_alignment(self, stub_pipeline, fake_input, tmp_path, monkeypatch):
        # First run: produce an aligned snapshot.
        config = _config(fake_input)
        pipeline_mod.run_pipeline(config)
        aligned_path = tmp_path / "video_transcript.aligned.json"
        assert aligned_path.exists()

        # Now invalidate the ASR/align stubs so a re-run that re-does them would crash.
        def _crash(*a, **kw):
            raise AssertionError("ASR/align should not be called when resuming")

        monkeypatch.setattr(pipeline_mod.asr_mod, "load_whisperx_model", _crash)
        monkeypatch.setattr(pipeline_mod.asr_mod, "align_segments", _crash)

        # Re-run with --resume-from-aligned: must not invoke ASR/align.
        resume_config = _config(fake_input)
        resume_config.resume_from_aligned = aligned_path
        written = pipeline_mod.run_pipeline(resume_config)

        # The final JSON should have stage="complete" and speaker labels.
        final = json.loads(written["json"].read_text(encoding="utf-8"))
        assert final["stage"] == "complete"
        assert any("speaker" in s for s in final["segments"])


class TestPayloadConstruction:
    def test_aligned_payload_no_speakers(self, fake_input):
        cfg = _config(fake_input)
        payload = pipeline_mod._build_payload(
            stage=pipeline_mod.STAGE_ALIGNED,
            source_file=fake_input,
            transcribed_at=__import__("datetime").datetime(2026, 5, 5, 10, 0, 0),
            config=cfg,
            device="cuda",
            compute_type="float16",
            detected_language="it",
            language_probability=0.95,
            segments=[{"start": 0.0, "end": 2.0, "text": "Ciao"}],
            processing_seconds=1.5,
        )
        assert payload["stage"] == pipeline_mod.STAGE_ALIGNED
        assert payload["stats"]["num_speakers"] == 0
        assert payload["stats"]["speakers"] == {}

    def test_complete_payload_with_speakers(self, fake_input):
        cfg = _config(fake_input)
        payload = pipeline_mod._build_payload(
            stage=pipeline_mod.STAGE_COMPLETE,
            source_file=fake_input,
            transcribed_at=__import__("datetime").datetime(2026, 5, 5, 10, 0, 0),
            config=cfg,
            device="cuda",
            compute_type="float16",
            detected_language="it",
            language_probability=0.95,
            segments=[
                {"start": 0.0, "end": 5.0, "text": "x", "speaker": "S0"},
                {"start": 5.0, "end": 8.0, "text": "y", "speaker": "S1"},
            ],
            processing_seconds=10.0,
        )
        assert payload["stage"] == pipeline_mod.STAGE_COMPLETE
        assert payload["stats"]["num_speakers"] == 2
        assert "S0" in payload["stats"]["speakers"]


class TestResolveDiarizationApi:
    """Direct unit tests for the whisperX adapter — the spot where API drift bites.

    These guard the regressions we hit on real machines: class moved between
    ``whisperx`` and ``whisperx.diarize``, and constructor kwarg renamed from
    ``use_auth_token`` to ``token``.
    """

    def test_pipeline_class_in_submodule(self, monkeypatch):
        import sys as _sys
        import types as _types

        from transcript_video import diarize as diarize_mod

        fake_whisperx = _types.ModuleType("whisperx")
        fake_diarize = _types.ModuleType("whisperx.diarize")

        class FakeDP:
            pass

        def fake_assign(diarized, aligned):
            return aligned

        fake_diarize.DiarizationPipeline = FakeDP
        fake_diarize.assign_word_speakers = fake_assign
        fake_whisperx.diarize = fake_diarize
        monkeypatch.setitem(_sys.modules, "whisperx", fake_whisperx)
        monkeypatch.setitem(_sys.modules, "whisperx.diarize", fake_diarize)

        cls, fn = diarize_mod._resolve_diarization_api()
        assert cls is FakeDP
        assert fn is fake_assign

    def test_pipeline_class_at_top_level(self, monkeypatch):
        import sys as _sys
        import types as _types

        from transcript_video import diarize as diarize_mod

        fake_whisperx = _types.ModuleType("whisperx")

        class FakeDP:
            pass

        def fake_assign(diarized, aligned):
            return aligned

        fake_whisperx.DiarizationPipeline = FakeDP
        fake_whisperx.assign_word_speakers = fake_assign
        monkeypatch.setitem(_sys.modules, "whisperx", fake_whisperx)

        cls, fn = diarize_mod._resolve_diarization_api()
        assert cls is FakeDP
        assert fn is fake_assign

    def test_init_kwargs_uses_token_when_signature_renames(self):
        from transcript_video.diarize import _build_pipeline_init_kwargs

        class NewApi:
            def __init__(self, model_name=None, token=None, device=None):  # whisperX modern signature
                pass

        kwargs = _build_pipeline_init_kwargs(NewApi, "hf_xxx", "cuda", "pyannote/foo")
        assert kwargs["token"] == "hf_xxx"
        assert "use_auth_token" not in kwargs
        assert kwargs["device"] == "cuda"
        assert kwargs["model_name"] == "pyannote/foo"

    def test_init_kwargs_uses_use_auth_token_when_signature_old(self):
        from transcript_video.diarize import _build_pipeline_init_kwargs

        class OldApi:
            def __init__(self, model_name=None, use_auth_token=None, device=None):  # legacy signature
                pass

        kwargs = _build_pipeline_init_kwargs(OldApi, "hf_xxx", "cuda", None)
        assert kwargs["use_auth_token"] == "hf_xxx"
        assert "token" not in kwargs
