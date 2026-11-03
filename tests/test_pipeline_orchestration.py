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
    IdentifyConfig,
    OutputConfig,
    RunConfig,
)
from transcript_video.speaker_db import add_sample, save_db


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
        # Same signature as whisperX 3.8's FasterWhisperPipeline.transcribe:
        # decoding options (prompt, hotwords, anti-loop) are NOT accepted here,
        # so passing one fails the test as it fails the real run.
        def transcribe(
            self, audio, batch_size=None, num_workers=0, language=None, task=None,
            chunk_size=30, print_progress=False, combined_progress=False,
            verbose=False, progress_callback=None,
        ):
            return {
                "language": "it",
                "language_probability": 0.95,
                "segments": [
                    {"id": 0, "start": 0.0, "end": 2.0, "text": "Ciao a tutti"},
                    {"id": 1, "start": 2.5, "end": 5.0, "text": "Iniziamo subito"},
                ],
            }

    monkeypatch.setattr(pipeline_mod.asr_mod, "check_cuda_available", lambda: (True, "float16"))
    monkeypatch.setattr(pipeline_mod.asr_mod, "torch_cuda_available", lambda: True)
    monkeypatch.setattr(pipeline_mod.asr_mod, "apply_torch_threads", lambda n: None)
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

    def _fake_extract(audio_path, segments, **kwargs):
        # Build a plausible-shaped clusters dict from whatever speaker labels
        # show up in the segments. Embeddings are tiny but well-formed.
        seen: dict[str, dict] = {}
        for s in segments:
            label = s.get("speaker")
            if not label:
                continue
            entry = seen.setdefault(
                label,
                {
                    "embedding": [0.1, 0.2, 0.3],
                    "duration_s": 0.0,
                    "n_segments": 0,
                    "embedding_model": "pyannote/fake-embedding",
                },
            )
            entry["duration_s"] += float(s.get("end", 0.0)) - float(s.get("start", 0.0))
            entry["n_segments"] += 1
        return seen

    monkeypatch.setattr(
        pipeline_mod.speaker_embed_mod, "extract_cluster_embeddings", _fake_extract
    )


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

    def test_schema_version_is_two(self, stub_pipeline, fake_input):
        config = _config(fake_input, write_md=False)
        written = pipeline_mod.run_pipeline(config)
        final = json.loads(written["json"].read_text(encoding="utf-8"))
        assert final["schema_version"] == 2

    def test_speaker_clusters_cached_in_complete_payload(self, stub_pipeline, fake_input):
        config = _config(fake_input, write_md=False)
        written = pipeline_mod.run_pipeline(config)
        final = json.loads(written["json"].read_text(encoding="utf-8"))
        clusters = final["speaker_clusters"]
        assert set(clusters.keys()) == {"SPEAKER_00", "SPEAKER_01"}
        for info in clusters.values():
            assert isinstance(info["embedding"], list)
            assert info["embedding"]  # non-empty
            assert info["n_segments"] >= 1
            assert info["duration_s"] > 0
            assert info["embedding_model"]

    def test_aligned_snapshot_has_empty_clusters(self, stub_pipeline, fake_input, tmp_path):
        config = _config(fake_input, write_md=False)
        pipeline_mod.run_pipeline(config)
        aligned = json.loads(
            (tmp_path / "video_transcript.aligned.json").read_text(encoding="utf-8")
        )
        # Aligned stage runs before diarization → no clusters.
        assert aligned["schema_version"] == 2
        assert aligned["speaker_clusters"] == {}

    def test_no_diarize_skips_aligned_snapshot(self, stub_pipeline, fake_input, tmp_path):
        config = _config(fake_input)
        config.diarization = DiarizationConfig(enabled=False)
        pipeline_mod.run_pipeline(config)

        # No diarization → no need for the aligned safety net.
        assert not (tmp_path / "video_transcript.aligned.json").exists()
        # Final JSON still produced.
        assert (tmp_path / "video_transcript.json").exists()

    def test_no_diarize_final_has_empty_clusters(self, stub_pipeline, fake_input, tmp_path):
        config = _config(fake_input, write_md=False)
        config.diarization = DiarizationConfig(enabled=False)
        pipeline_mod.run_pipeline(config)
        final = json.loads(
            (tmp_path / "video_transcript.json").read_text(encoding="utf-8")
        )
        assert final["speaker_clusters"] == {}

    def test_embedding_extraction_failure_does_not_crash(
        self, stub_pipeline, fake_input, tmp_path, monkeypatch
    ):
        def boom(*a, **kw):
            raise RuntimeError("simulated embedding model failure")

        monkeypatch.setattr(
            pipeline_mod.speaker_embed_mod, "extract_cluster_embeddings", boom
        )

        config = _config(fake_input, write_md=False)
        written = pipeline_mod.run_pipeline(config)
        final = json.loads(written["json"].read_text(encoding="utf-8"))
        # Pipeline succeeded, transcript is intact, only clusters are empty.
        assert final["stage"] == "complete"
        assert final["speaker_clusters"] == {}
        assert len(final["segments"]) >= 1


def _seed_voice_db(path, mapping: dict[str, list[float]], *, model: str = "pyannote/fake-embedding") -> None:
    """Write a voice DB at ``path`` populated with one sample per name in ``mapping``."""
    db = {"schema_version": 1, "embedding_model": model, "speakers": {}}
    for name, embedding in mapping.items():
        add_sample(
            db, name, embedding,
            source="seed.json", embedding_model=model,
            cluster="SPEAKER_seed", duration_s=10.0,
        )
    save_db(db, path)


class TestSpeakerIdentities:
    def test_disabled_yields_empty_identities(self, stub_pipeline, fake_input):
        # identify defaults to disabled.
        config = _config(fake_input, write_md=False)
        written = pipeline_mod.run_pipeline(config)
        final = json.loads(written["json"].read_text(encoding="utf-8"))
        assert final["speaker_identities"] == {}

    def test_enabled_with_matching_db_populates_auto_entries(
        self, stub_pipeline, fake_input, tmp_path
    ):
        # Fake clusters from stub use embedding [0.1, 0.2, 0.3].
        db_path = tmp_path / "voices.json"
        _seed_voice_db(db_path, {"Mario": [0.1, 0.2, 0.3]})

        config = _config(fake_input, write_md=False)
        config.identify = IdentifyConfig(enabled=True, voice_db=db_path, threshold=0.5)
        written = pipeline_mod.run_pipeline(config)
        final = json.loads(written["json"].read_text(encoding="utf-8"))

        identities = final["speaker_identities"]
        # Both clusters use the same embedding in the stub → both match Mario.
        assert "SPEAKER_00" in identities
        assert identities["SPEAKER_00"]["name"] == "Mario"
        assert identities["SPEAKER_00"]["source"] == "auto"
        assert identities["SPEAKER_00"]["score"] is not None

    def test_manual_speaker_map_overrides_auto(
        self, stub_pipeline, fake_input, tmp_path
    ):
        db_path = tmp_path / "voices.json"
        _seed_voice_db(db_path, {"AutoName": [0.1, 0.2, 0.3]})

        config = _config(fake_input, write_md=False)
        config.identify = IdentifyConfig(enabled=True, voice_db=db_path, threshold=0.5)
        config.speaker_map = {"SPEAKER_00": "ManualName"}
        written = pipeline_mod.run_pipeline(config)
        final = json.loads(written["json"].read_text(encoding="utf-8"))

        # SPEAKER_00 was overridden by the CLI map.
        assert final["speaker_identities"]["SPEAKER_00"] == {
            "name": "ManualName", "score": None, "source": "manual",
        }
        # SPEAKER_01 still gets the auto match (same embedding → AutoName).
        sp01 = final["speaker_identities"].get("SPEAKER_01")
        assert sp01 is not None
        assert sp01["source"] == "auto"

    def test_missing_db_is_non_fatal(self, stub_pipeline, fake_input, tmp_path):
        # Empty DB (missing file) yields cold-start: no matches, no error.
        config = _config(fake_input, write_md=False)
        config.identify = IdentifyConfig(
            enabled=True, voice_db=tmp_path / "absent.json", threshold=0.5,
        )
        written = pipeline_mod.run_pipeline(config)
        final = json.loads(written["json"].read_text(encoding="utf-8"))
        # No auto entries (empty DB), no manual entries (no speaker_map), no error.
        assert final["speaker_identities"] == {}

    def test_incompatible_model_skips_auto_keeps_manual(
        self, stub_pipeline, fake_input, tmp_path
    ):
        db_path = tmp_path / "voices.json"
        _seed_voice_db(db_path, {"Old": [0.1, 0.2, 0.3]}, model="some-other-model")

        config = _config(fake_input, write_md=False)
        config.identify = IdentifyConfig(enabled=True, voice_db=db_path, threshold=0.5)
        config.speaker_map = {"SPEAKER_00": "Manual"}
        written = pipeline_mod.run_pipeline(config)
        final = json.loads(written["json"].read_text(encoding="utf-8"))

        # No auto matches (model mismatch), but manual still recorded.
        identities = final["speaker_identities"]
        assert identities["SPEAKER_00"] == {
            "name": "Manual", "score": None, "source": "manual",
        }
        assert "SPEAKER_01" not in identities

    def test_below_threshold_no_match(self, stub_pipeline, fake_input, tmp_path):
        db_path = tmp_path / "voices.json"
        # Orthogonal to cluster embeddings → cosine 0.
        _seed_voice_db(db_path, {"Mario": [-0.3, -0.2, -0.1]})

        config = _config(fake_input, write_md=False)
        config.identify = IdentifyConfig(enabled=True, voice_db=db_path, threshold=0.5)
        written = pipeline_mod.run_pipeline(config)
        final = json.loads(written["json"].read_text(encoding="utf-8"))
        # No cluster matched.
        assert final["speaker_identities"] == {}


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

    def test_resume_refuses_a_finished_transcript_with_speakers(
        self, stub_pipeline, fake_input, tmp_path, monkeypatch, capsys
    ):
        """Invariant #5: re-diarizing a finished JSON would clobber its speakers."""
        written = pipeline_mod.run_pipeline(_config(fake_input))

        def _never(*a, **kw):
            raise AssertionError("diarization must not start")

        monkeypatch.setattr(pipeline_mod.diarize_mod, "diarize_and_assign", _never)
        config = _config(fake_input)
        config.resume_from_aligned = written["json"]
        with pytest.raises(SystemExit):
            pipeline_mod.run_pipeline(config)
        assert "speakers already assigned" in capsys.readouterr().err

    def test_resume_accepts_a_no_diarize_transcript(self, stub_pipeline, fake_input, tmp_path):
        """A --no-diarize run writes no aligned snapshot: its JSON is the way to diarize later."""
        first = _config(fake_input)
        first.diarization = DiarizationConfig(enabled=False)
        written = pipeline_mod.run_pipeline(first)
        assert not (tmp_path / "video_transcript.aligned.json").exists()

        config = _config(fake_input)
        config.resume_from_aligned = written["json"]
        final = json.loads(pipeline_mod.run_pipeline(config)["json"].read_text(encoding="utf-8"))
        assert any("speaker" in seg for seg in final["segments"])

    def test_resume_gpu_snapshot_on_cpu_only_machine(
        self, stub_pipeline, fake_input, tmp_path, monkeypatch
    ):
        """A snapshot made on a GPU box diarizes on CPU when no GPU is present now."""
        pipeline_mod.run_pipeline(_config(fake_input))
        aligned_path = tmp_path / "video_transcript.aligned.json"
        assert json.loads(aligned_path.read_text(encoding="utf-8"))["parameters"]["device"] == "cuda"

        # Now on a machine without CUDA.
        monkeypatch.setattr(pipeline_mod.asr_mod, "check_cuda_available", lambda: (False, None))
        monkeypatch.setattr(pipeline_mod.asr_mod, "torch_cuda_available", lambda: False)
        seen: dict[str, str] = {}

        def _fake_diarize(aligned, audio, **kwargs):
            seen["diarize"] = kwargs["device"]
            return {
                "segments": [{**s, "speaker": "SPEAKER_00"} for s in aligned["segments"]],
                "word_segments": [],
            }

        def _fake_extract(audio_path, segments, **kwargs):
            seen["embed"] = kwargs["device"]
            return {}

        monkeypatch.setattr(pipeline_mod.diarize_mod, "diarize_and_assign", _fake_diarize)
        monkeypatch.setattr(
            pipeline_mod.speaker_embed_mod, "extract_cluster_embeddings", _fake_extract
        )

        resume_config = _config(fake_input)
        resume_config.resume_from_aligned = aligned_path
        written = pipeline_mod.run_pipeline(resume_config)

        assert seen == {"diarize": "cpu", "embed": "cpu"}
        # The JSON still records the device/compute type the ASR actually ran with.
        params = json.loads(written["json"].read_text(encoding="utf-8"))["parameters"]
        assert params["device"] == "cuda"
        assert params["compute_type"] == "float16"

    def test_resume_with_explicit_cuda_on_cpu_only_machine_stops(
        self, stub_pipeline, fake_input, tmp_path, monkeypatch
    ):
        pipeline_mod.run_pipeline(_config(fake_input))
        aligned_path = tmp_path / "video_transcript.aligned.json"
        monkeypatch.setattr(pipeline_mod.asr_mod, "check_cuda_available", lambda: (False, None))
        monkeypatch.setattr(pipeline_mod.asr_mod, "torch_cuda_available", lambda: False)

        def _never(*a, **kw):
            raise AssertionError("diarization must not start")

        monkeypatch.setattr(pipeline_mod.diarize_mod, "diarize_and_assign", _never)
        resume_config = _config(fake_input)
        resume_config.resume_from_aligned = aligned_path
        resume_config.device = "cuda"
        with pytest.raises(SystemExit):
            pipeline_mod.run_pipeline(resume_config)

    def test_resume_honors_explicit_cpu(self, stub_pipeline, fake_input, tmp_path, monkeypatch):
        pipeline_mod.run_pipeline(_config(fake_input))
        aligned_path = tmp_path / "video_transcript.aligned.json"
        seen: dict[str, str] = {}

        def _fake_diarize(aligned, audio, **kwargs):
            seen["diarize"] = kwargs["device"]
            return {"segments": aligned["segments"], "word_segments": []}

        monkeypatch.setattr(pipeline_mod.diarize_mod, "diarize_and_assign", _fake_diarize)

        resume_config = _config(fake_input)
        resume_config.resume_from_aligned = aligned_path
        resume_config.device = "cpu"
        pipeline_mod.run_pipeline(resume_config)

        assert seen["diarize"] == "cpu"


class TestDevices:
    def _capture_stage_devices(self, monkeypatch):
        seen: dict[str, str] = {}

        def _fake_align(segs, lang, audio, device):
            seen["align"] = device
            return {"segments": segs, "word_segments": []}

        def _fake_diarize(aligned, audio, **kwargs):
            seen["diarize"] = kwargs["device"]
            return {"segments": aligned["segments"], "word_segments": []}

        def _fake_extract(audio_path, segments, **kwargs):
            seen["embed"] = kwargs["device"]
            return {}

        monkeypatch.setattr(pipeline_mod.asr_mod, "align_segments", _fake_align)
        monkeypatch.setattr(pipeline_mod.diarize_mod, "diarize_and_assign", _fake_diarize)
        monkeypatch.setattr(
            pipeline_mod.speaker_embed_mod, "extract_cluster_embeddings", _fake_extract
        )
        return seen

    def test_devices_resolved_once_per_run(self, stub_pipeline, fake_input, monkeypatch):
        """All torch stages get the same device, probed once up front.

        Regression: on a CPU-only host ctranslate2's CUDA probe flipped
        between calls, which sent ASR to cpu and pyannote to cuda in one run.
        """
        calls = {"n": 0}
        real_plan = pipeline_mod.asr_mod.plan_devices

        def _counting(*a, **kw):
            calls["n"] += 1
            return real_plan(*a, **kw)

        monkeypatch.setattr(pipeline_mod.asr_mod, "plan_devices", _counting)
        seen = self._capture_stage_devices(monkeypatch)

        pipeline_mod.run_pipeline(_config(fake_input, write_md=False))

        assert calls["n"] == 1
        assert seen == {"align": "cuda", "diarize": "cuda", "embed": "cuda"}

    def test_cpu_only_torch_runs_torch_stages_on_cpu(self, stub_pipeline, fake_input, monkeypatch):
        """GPU visible to ctranslate2, CPU-only torch wheel, --device auto.

        whisperX's VAD is a torch model: it must be built on CPU too, while
        the ASR keeps the GPU. Regression: whisperX put it on cuda:0 and the
        run crashed at model load.
        """
        monkeypatch.setattr(pipeline_mod.asr_mod, "torch_cuda_available", lambda: False)
        monkeypatch.setattr(pipeline_mod.asr_mod, "cpu_vad_supported", lambda: True)
        seen = self._capture_stage_devices(monkeypatch)
        real_load = pipeline_mod.asr_mod.load_whisperx_model

        def _load(name, device, *a, **kw):
            seen["asr"], seen["vad"] = device, kw.get("vad_device")
            return real_load(name, device, *a, **kw)

        monkeypatch.setattr(pipeline_mod.asr_mod, "load_whisperx_model", _load)

        written = pipeline_mod.run_pipeline(_config(fake_input, write_md=False))

        assert seen == {"asr": "cuda", "vad": "cpu", "align": "cpu", "diarize": "cpu", "embed": "cpu"}
        params = json.loads(written["json"].read_text(encoding="utf-8"))["parameters"]
        assert params["device"] == "cuda"  # the ASR's device

    def test_resume_without_diarization_ignores_explicit_cuda(
        self, stub_pipeline, fake_input, tmp_path, monkeypatch
    ):
        """Nothing runs on a device, so a shared `device = "cuda"` config mustn't block it."""
        pipeline_mod.run_pipeline(_config(fake_input))
        aligned_path = tmp_path / "video_transcript.aligned.json"
        monkeypatch.setattr(pipeline_mod.asr_mod, "check_cuda_available", lambda: (False, None))
        monkeypatch.setattr(pipeline_mod.asr_mod, "torch_cuda_available", lambda: False)

        config = _config(fake_input)
        config.resume_from_aligned = aligned_path
        config.diarization = DiarizationConfig(enabled=False)
        config.device = "cuda"
        written = pipeline_mod.run_pipeline(config)

        assert json.loads(written["json"].read_text(encoding="utf-8"))["stage"] == "complete"

    def test_explicit_cuda_without_torch_cuda_stops_before_asr(
        self, stub_pipeline, fake_input, monkeypatch
    ):
        monkeypatch.setattr(pipeline_mod.asr_mod, "torch_cuda_available", lambda: False)

        def _never(*a, **kw):
            raise AssertionError("the model must not be loaded")

        monkeypatch.setattr(pipeline_mod.asr_mod, "load_whisperx_model", _never)
        config = _config(fake_input)
        config.device = "cuda"
        with pytest.raises(SystemExit):
            pipeline_mod.run_pipeline(config)


class TestThreads:
    def test_explicit_threads_reach_asr_and_torch(self, stub_pipeline, fake_input, monkeypatch):
        seen: dict[str, int] = {}

        class FakeModel:
            def transcribe(self, audio, batch_size, **kwargs):
                return {"language": "it", "segments": [{"start": 0.0, "end": 1.0, "text": "x"}]}

        def _fake_load(*a, **kw):
            seen["asr"] = kw["threads"]
            return FakeModel()

        monkeypatch.setattr(pipeline_mod.asr_mod, "load_whisperx_model", _fake_load)
        monkeypatch.setattr(
            pipeline_mod.asr_mod, "apply_torch_threads", lambda n: seen.__setitem__("torch", n)
        )

        config = _config(fake_input, write_md=False)
        config.threads = 3
        pipeline_mod.run_pipeline(config)

        assert seen == {"asr": 3, "torch": 3}

    def test_cpu_run_default_threads_are_cpus_minus_one(
        self, stub_pipeline, fake_input, monkeypatch
    ):
        seen: dict[str, int | None] = {}
        monkeypatch.setattr(pipeline_mod.asr_mod, "available_cpu_count", lambda: 8)
        monkeypatch.setattr(pipeline_mod.asr_mod, "check_cuda_available", lambda: (False, None))
        monkeypatch.setattr(pipeline_mod.asr_mod, "torch_cuda_available", lambda: False)
        monkeypatch.setattr(
            pipeline_mod.asr_mod, "apply_torch_threads", lambda n: seen.__setitem__("torch", n)
        )
        real_fake_model = pipeline_mod.asr_mod.load_whisperx_model

        def _fake_load(*a, **kw):
            seen["asr"] = kw["threads"]
            return real_fake_model(*a, **kw)

        monkeypatch.setattr(pipeline_mod.asr_mod, "load_whisperx_model", _fake_load)

        pipeline_mod.run_pipeline(_config(fake_input, write_md=False))

        assert seen == {"torch": 7, "asr": 7}

    def test_gpu_run_keeps_library_defaults(self, stub_pipeline, fake_input, monkeypatch):
        """On GPU the CPUs-1 policy isn't applied; torch keeps its own default."""
        seen: dict[str, int | None] = {}
        monkeypatch.setattr(
            pipeline_mod.asr_mod, "apply_torch_threads", lambda n: seen.__setitem__("torch", n)
        )
        real_fake_model = pipeline_mod.asr_mod.load_whisperx_model

        def _fake_load(*a, **kw):
            seen["asr"] = kw["threads"]
            return real_fake_model(*a, **kw)

        monkeypatch.setattr(pipeline_mod.asr_mod, "load_whisperx_model", _fake_load)

        pipeline_mod.run_pipeline(_config(fake_input, write_md=False))

        assert seen == {"asr": None}


class TestDecodingOptions:
    """Prompt, hotwords and anti-loop go in at model load, never to transcribe()."""

    def test_options_go_to_load_model(self, stub_pipeline, fake_input, monkeypatch):
        seen: dict = {}
        real_load = pipeline_mod.asr_mod.load_whisperx_model

        def _load(name, device, compute_type, asr_options, **kw):
            seen.update(asr_options)
            return real_load(name, device, compute_type, asr_options, **kw)

        monkeypatch.setattr(pipeline_mod.asr_mod, "load_whisperx_model", _load)
        config = _config(fake_input, write_md=False)
        config.initial_prompt = "Riunione di lavoro."
        config.hotwords = "OpenShift"
        config.anti_loop = True
        config.beam_size = 10
        pipeline_mod.run_pipeline(config)  # the strict FakeModel.transcribe must not raise

        assert seen == {
            "beam_size": 10,
            "initial_prompt": "Riunione di lavoro.",
            "hotwords": "OpenShift",
            "condition_on_previous_text": False,
            "compression_ratio_threshold": 2.0,
            "no_speech_threshold": 0.5,
        }


class TestThreadPolicyEdges:
    def test_zero_threads_in_runconfig_is_auto(self, stub_pipeline, fake_input, monkeypatch):
        """RunConfig(threads=0) on a GPU run keeps library defaults, like None."""
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            pipeline_mod.asr_mod, "apply_torch_threads", lambda n: seen.__setitem__("torch", n)
        )
        config = _config(fake_input, write_md=False)
        config.threads = 0
        pipeline_mod.run_pipeline(config)
        assert "torch" not in seen

    def test_cpu_vad_gets_cpu_threads_even_with_torch_on_cuda(self, fake_input, monkeypatch):
        """ASR + VAD on CPU, alignment/diarization on CUDA: torch still does CPU work."""
        seen: dict[str, object] = {}
        monkeypatch.setattr(pipeline_mod.asr_mod, "available_cpu_count", lambda: 8)
        monkeypatch.setattr(
            pipeline_mod.asr_mod, "apply_torch_threads", lambda n: seen.__setitem__("torch", n)
        )
        plan = pipeline_mod.asr_mod.DevicePlan(asr="cpu", torch="cuda", vad="cpu")
        config = _config(fake_input, write_md=False)
        assert pipeline_mod._configure_threads(config, plan) == 7
        assert seen == {"torch": 7}

    def test_no_stage_logs_nothing_about_threads(self, fake_input, caplog):
        plan = pipeline_mod.asr_mod.DevicePlan(asr=None, torch=None)
        config = _config(fake_input, write_md=False)
        config.threads = 4
        with caplog.at_level("INFO", logger="transcript_video.pipeline"):
            assert pipeline_mod._configure_threads(config, plan) is None
        assert "CPU threads" not in caplog.text


class TestBatchSize:
    """ASR batch size: 16 on CUDA, 4 on CPU, explicit value wins."""

    @pytest.fixture
    def seen_batch(self, monkeypatch) -> dict[str, int]:
        seen: dict[str, int] = {}

        class FakeModel:
            def transcribe(self, audio, batch_size, **kwargs):
                seen["batch_size"] = batch_size
                return {"language": "it", "segments": [{"start": 0.0, "end": 1.0, "text": "x"}]}

        monkeypatch.setattr(
            pipeline_mod.asr_mod, "load_whisperx_model", lambda *a, **kw: FakeModel()
        )
        return seen

    @staticmethod
    def _on_cpu(monkeypatch):
        monkeypatch.setattr(pipeline_mod.asr_mod, "check_cuda_available", lambda: (False, None))
        monkeypatch.setattr(pipeline_mod.asr_mod, "torch_cuda_available", lambda: False)

    def test_cuda_default_is_16(self, stub_pipeline, seen_batch, fake_input):
        config = _config(fake_input, write_md=False)
        config.beam_size = 10
        pipeline_mod.run_pipeline(config)
        assert seen_batch == {"batch_size": 16}

    def test_cpu_default_is_4(self, stub_pipeline, seen_batch, fake_input, monkeypatch):
        self._on_cpu(monkeypatch)
        config = _config(fake_input, write_md=False)
        config.beam_size = 10
        pipeline_mod.run_pipeline(config)
        assert seen_batch == {"batch_size": 4}

    def test_explicit_batch_size_wins_on_cpu(self, stub_pipeline, seen_batch, fake_input, monkeypatch):
        self._on_cpu(monkeypatch)
        config = _config(fake_input, write_md=False)
        config.beam_size = 10
        config.batch_size = 3
        pipeline_mod.run_pipeline(config)
        assert seen_batch == {"batch_size": 3}


class TestMemoryLog:
    def test_reads_rss_and_peak_from_proc_status(self, tmp_path):
        status = tmp_path / "status"
        status.write_text("Name:\tpython\nVmHWM:\t 3145728 kB\nVmRSS:\t 1048576 kB\n")
        assert pipeline_mod._read_memory_gb(status) == (1.0, 3.0)

    def test_missing_status_file_is_none(self, tmp_path):
        assert pipeline_mod._read_memory_gb(tmp_path / "nope") is None


class TestAudioDecodedOnce:
    """The waveform decoded for ASR feeds diarization and embeddings too."""

    @pytest.fixture
    def seen(self, monkeypatch, fake_audio) -> dict:
        seen: dict = {"loads": 0}

        def _load(_path):
            seen["loads"] += 1
            return fake_audio

        def _diarize(aligned, audio, **kwargs):
            seen["diarize_audio"] = audio
            return {"segments": aligned["segments"], "word_segments": []}

        def _extract(audio_path, segments, **kwargs):
            seen["embed_audio"] = kwargs.get("audio")
            return {}

        monkeypatch.setattr(pipeline_mod.asr_mod, "load_audio", _load)
        monkeypatch.setattr(pipeline_mod.diarize_mod, "diarize_and_assign", _diarize)
        monkeypatch.setattr(
            pipeline_mod.speaker_embed_mod, "extract_cluster_embeddings", _extract
        )
        return seen

    def test_full_run_decodes_once(self, stub_pipeline, seen, fake_input, fake_audio):
        pipeline_mod.run_pipeline(_config(fake_input, write_md=False))
        assert seen["loads"] == 1
        assert seen["diarize_audio"] is fake_audio
        assert seen["embed_audio"] is fake_audio

    def test_resume_decodes_once_and_reuses_for_embeddings(
        self, stub_pipeline, seen, fake_input, fake_audio, tmp_path
    ):
        # First run writes the aligned snapshot; reset the counter, then resume.
        pipeline_mod.run_pipeline(_config(fake_input, write_md=False))
        snapshot = pipeline_mod._resolve_output_paths(_config(fake_input))["aligned"]
        seen["loads"] = 0

        config = _config(fake_input, write_md=False)
        config.resume_from_aligned = snapshot
        pipeline_mod.run_pipeline(config)

        assert seen["loads"] == 1
        assert seen["diarize_audio"] is fake_audio
        assert seen["embed_audio"] is fake_audio


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
