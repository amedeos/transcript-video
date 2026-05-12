"""Tests for the transcript-to-md CLI: --list-speakers and end-to-end re-render."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transcript_video.cli_to_md import (
    _format_speaker_overview,
    _refresh_auto_identities,
    _resolve_speaker_stats,
    main,
)
from transcript_video.speaker_db import save_db


def _two_speaker_transcript() -> dict:
    """A small canonical transcript with two speakers."""
    return {
        "schema_version": 1,
        "source_file": "/path/to/meeting.mp4",
        "transcribed_at": "2026-05-01T10:30:00",
        "parameters": {"backend": "whisperx", "model": "large-v3", "beam_size": 5},
        "audio_info": {"language_detected": "it", "duration_seconds": 60.0},
        "stats": {
            "num_segments": 3,
            "num_speakers": 2,
            "processing_seconds": 10.0,
            "speakers": {
                "SPEAKER_00": {"duration_seconds": 40.0, "percentage": 80.0, "num_turns": 2, "first_text": "Allora..."},
                "SPEAKER_01": {"duration_seconds": 10.0, "percentage": 20.0, "num_turns": 1, "first_text": "Sì..."},
            },
        },
        "segments": [
            {"start": 0.0, "end": 20.0, "text": "Allora oggi parliamo", "speaker": "SPEAKER_00"},
            {"start": 20.0, "end": 30.0, "text": "Sì certo", "speaker": "SPEAKER_01"},
            {"start": 30.0, "end": 50.0, "text": "Continuo", "speaker": "SPEAKER_00"},
        ],
    }


class TestResolveSpeakerStats:
    def test_uses_cached_stats(self):
        transcript = _two_speaker_transcript()
        stats = _resolve_speaker_stats(transcript)
        # Cached value wins — these numbers are not the recomputation.
        assert stats["SPEAKER_00"]["percentage"] == 80.0

    def test_falls_back_to_recompute_when_missing(self):
        transcript = _two_speaker_transcript()
        del transcript["stats"]["speakers"]
        stats = _resolve_speaker_stats(transcript)
        # Recomputed: SPEAKER_00 totals 40s, SPEAKER_01 totals 10s, total 50s.
        assert stats["SPEAKER_00"]["duration_seconds"] == pytest.approx(40.0)
        assert stats["SPEAKER_00"]["percentage"] == 80.0

    def test_no_segments_returns_empty(self):
        transcript = {"stats": {}, "segments": []}
        assert _resolve_speaker_stats(transcript) == {}


class TestFormatSpeakerOverview:
    def test_includes_all_speakers(self):
        out = _format_speaker_overview(_two_speaker_transcript(), {})
        assert "SPEAKER_00" in out
        assert "SPEAKER_01" in out
        assert "00:00:40" in out  # SPEAKER_00 duration formatted as HH:MM:SS
        assert "80.0%" in out

    def test_uses_speaker_map_for_name_column(self):
        out = _format_speaker_overview(_two_speaker_transcript(), {"SPEAKER_00": "Amedeo"})
        assert "Amedeo" in out
        # Unmapped labels still appear under both Label and Name columns.
        assert "SPEAKER_01" in out

    def test_includes_suspect_column(self):
        # Set num_suspect on the cached stats and verify it shows up.
        transcript = _two_speaker_transcript()
        transcript["stats"]["speakers"]["SPEAKER_00"]["num_suspect"] = 3
        transcript["stats"]["speakers"]["SPEAKER_01"]["num_suspect"] = 0
        out = _format_speaker_overview(transcript, {})
        assert "Suspect" in out
        # The "3" suspect count for SPEAKER_00 must appear on its row.
        speaker_00_line = next(line for line in out.splitlines() if line.startswith("SPEAKER_00"))
        assert " 3 " in speaker_00_line or speaker_00_line.endswith(" 3  Allora...")

    def test_no_speakers_message(self):
        out = _format_speaker_overview({"stats": {}, "segments": []}, {})
        assert "No speakers detected" in out


class TestListSpeakersCli:
    def test_list_speakers_does_not_write_md(self, tmp_path: Path, capsys):
        json_path = tmp_path / "t.json"
        json_path.write_text(json.dumps(_two_speaker_transcript()), encoding="utf-8")

        main([str(json_path), "--list-speakers"])

        captured = capsys.readouterr().out
        assert "SPEAKER_00" in captured
        assert "Duration" in captured

        # No MD file should have been written.
        assert not (tmp_path / "t.md").exists()

    def test_list_speakers_with_speaker_map(self, tmp_path: Path, capsys):
        json_path = tmp_path / "t.json"
        json_path.write_text(json.dumps(_two_speaker_transcript()), encoding="utf-8")

        main([str(json_path), "--list-speakers", "--speaker-map", "SPEAKER_00=Amedeo"])

        captured = capsys.readouterr().out
        assert "Amedeo" in captured

    def test_normal_render_still_works(self, tmp_path: Path):
        json_path = tmp_path / "t.json"
        json_path.write_text(json.dumps(_two_speaker_transcript()), encoding="utf-8")

        main([str(json_path)])
        out = tmp_path / "t.md"
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "## [00:00:00] SPEAKER_00" in text


def _transcript_with_clusters(*, identities=None) -> dict:
    """Schema-v2 transcript with cached cluster embeddings."""
    payload = _two_speaker_transcript()
    payload["schema_version"] = 2
    payload["speaker_clusters"] = {
        "SPEAKER_00": {
            "embedding": [1.0, 0.0, 0.0],
            "duration_s": 40.0, "n_segments": 2,
            "embedding_model": "pyannote/fake",
        },
        "SPEAKER_01": {
            "embedding": [0.0, 1.0, 0.0],
            "duration_s": 10.0, "n_segments": 1,
            "embedding_model": "pyannote/fake",
        },
    }
    if identities is not None:
        payload["speaker_identities"] = identities
    return payload


def _seed_db(path, mapping: dict[str, list[float]], *, model: str = "pyannote/fake") -> None:
    db = {"schema_version": 1, "embedding_model": model, "speakers": {}}
    for name, emb in mapping.items():
        db["speakers"][name] = [{"embedding": emb, "source": "seed.json", "duration_s": 10.0}]
    save_db(db, path)


class TestRefreshAutoIdentities:
    def test_populates_identities_from_db(self, tmp_path):
        db_path = tmp_path / "voices.json"
        _seed_db(db_path, {"Mario": [1.0, 0.0, 0.0], "Luca": [0.0, 1.0, 0.0]})

        transcript = _transcript_with_clusters()
        _refresh_auto_identities(transcript, voice_db=str(db_path), threshold=0.5)

        ids = transcript["speaker_identities"]
        assert ids["SPEAKER_00"]["name"] == "Mario"
        assert ids["SPEAKER_00"]["source"] == "auto"
        assert ids["SPEAKER_01"]["name"] == "Luca"
        assert ids["SPEAKER_01"]["source"] == "auto"

    def test_preserves_manual_entries(self, tmp_path):
        """Manual entries from transcribe-time must survive re-render auto-id."""
        db_path = tmp_path / "voices.json"
        _seed_db(db_path, {"AutoMario": [1.0, 0.0, 0.0], "AutoLuca": [0.0, 1.0, 0.0]})

        transcript = _transcript_with_clusters(identities={
            "SPEAKER_00": {"name": "ManualSet", "score": None, "source": "manual"},
            "SPEAKER_01": {"name": "OldAutoName", "score": 0.7, "source": "auto"},
        })
        _refresh_auto_identities(transcript, voice_db=str(db_path), threshold=0.5)

        ids = transcript["speaker_identities"]
        # Manual entry preserved.
        assert ids["SPEAKER_00"] == {"name": "ManualSet", "score": None, "source": "manual"}
        # Old auto entry replaced by fresh match.
        assert ids["SPEAKER_01"]["name"] == "AutoLuca"
        assert ids["SPEAKER_01"]["source"] == "auto"

    def test_missing_clusters_logs_warning_and_returns(self, tmp_path, caplog):
        transcript = _two_speaker_transcript()  # v1, no speaker_clusters
        caplog.set_level("WARNING")
        _refresh_auto_identities(transcript, voice_db=None, threshold=0.5)
        # No identities created; warning emitted.
        assert "speaker_clusters" in caplog.text
        assert "speaker_identities" not in transcript or not transcript.get("speaker_identities")

    def test_missing_db_logs_warning_and_returns(self, tmp_path, caplog):
        transcript = _transcript_with_clusters()
        caplog.set_level("WARNING")
        _refresh_auto_identities(
            transcript, voice_db=str(tmp_path / "absent.json"), threshold=0.5,
        )
        # Empty DB → warning, no identities populated.
        assert "empty" in caplog.text.lower()
        assert not transcript.get("speaker_identities")

    def test_model_mismatch_skips_auto(self, tmp_path, caplog):
        db_path = tmp_path / "voices.json"
        _seed_db(db_path, {"Mario": [1.0, 0.0, 0.0]}, model="other-model")

        transcript = _transcript_with_clusters()
        caplog.set_level("WARNING")
        _refresh_auto_identities(transcript, voice_db=str(db_path), threshold=0.5)
        assert "embedding_model" in caplog.text
        assert not transcript.get("speaker_identities")

    def test_below_threshold_no_match(self, tmp_path):
        db_path = tmp_path / "voices.json"
        # Orthogonal embeddings vs SPEAKER_00 ([1,0,0]) and SPEAKER_01 ([0,1,0]).
        _seed_db(db_path, {"Mario": [0.0, 0.0, 1.0]})

        transcript = _transcript_with_clusters()
        _refresh_auto_identities(transcript, voice_db=str(db_path), threshold=0.5)
        assert transcript["speaker_identities"] == {}


class TestIdentifySpeakersCli:
    def test_end_to_end_render_uses_identified_names(self, tmp_path: Path):
        db_path = tmp_path / "voices.json"
        _seed_db(db_path, {"Mario": [1.0, 0.0, 0.0], "Luca": [0.0, 1.0, 0.0]})

        json_path = tmp_path / "t.json"
        json_path.write_text(json.dumps(_transcript_with_clusters()), encoding="utf-8")

        main([
            str(json_path),
            "--identify-speakers",
            "--voice-db", str(db_path),
            "--id-threshold", "0.5",
        ])

        md = (tmp_path / "t.md").read_text(encoding="utf-8")
        assert "## [00:00:00] Mario" in md
        assert "## [00:00:20] Luca" in md

    def test_cli_speaker_map_still_overrides_auto_at_render(self, tmp_path: Path):
        db_path = tmp_path / "voices.json"
        _seed_db(db_path, {"AutoMario": [1.0, 0.0, 0.0], "AutoLuca": [0.0, 1.0, 0.0]})

        json_path = tmp_path / "t.json"
        json_path.write_text(json.dumps(_transcript_with_clusters()), encoding="utf-8")

        main([
            str(json_path),
            "--identify-speakers",
            "--voice-db", str(db_path),
            "--id-threshold", "0.5",
            "--speaker-map", "SPEAKER_00=Forced",
        ])

        md = (tmp_path / "t.md").read_text(encoding="utf-8")
        # CLI map wins for SPEAKER_00.
        assert "## [00:00:00] Forced" in md
        # SPEAKER_01 still gets auto-id.
        assert "## [00:00:20] AutoLuca" in md

    def test_list_speakers_reflects_auto_identification(self, tmp_path: Path, capsys):
        db_path = tmp_path / "voices.json"
        _seed_db(db_path, {"Mario": [1.0, 0.0, 0.0], "Luca": [0.0, 1.0, 0.0]})

        json_path = tmp_path / "t.json"
        json_path.write_text(json.dumps(_transcript_with_clusters()), encoding="utf-8")

        main([
            str(json_path), "--list-speakers",
            "--identify-speakers", "--voice-db", str(db_path), "--id-threshold", "0.5",
        ])
        out = capsys.readouterr().out
        # Auto-identified names appear in the Name column.
        assert "Mario" in out
        assert "Luca" in out
