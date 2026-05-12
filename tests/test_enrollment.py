"""Tests for the torch-free enrollment logic (``transcript-learn``'s core)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transcript_video.enrollment import EnrollmentError, learn_from_transcript
from transcript_video.speaker_db import load_db, save_db


def _make_transcript_json(
    path: Path,
    *,
    schema_version: int = 2,
    clusters: dict | None = None,
) -> None:
    payload = {"schema_version": schema_version}
    if clusters is not None:
        payload["speaker_clusters"] = clusters
    path.write_text(json.dumps(payload), encoding="utf-8")


def _default_clusters() -> dict:
    return {
        "SPEAKER_00": {
            "embedding": [0.1, 0.2, 0.3],
            "duration_s": 10.0,
            "n_segments": 4,
            "embedding_model": "pyannote/fake-embedding",
        },
        "SPEAKER_01": {
            "embedding": [0.4, 0.5, 0.6],
            "duration_s": 5.0,
            "n_segments": 2,
            "embedding_model": "pyannote/fake-embedding",
        },
    }


class TestHappyPath:
    def test_two_speakers_added_to_fresh_db(self, tmp_path):
        json_path = tmp_path / "ep1_transcript.json"
        _make_transcript_json(json_path, clusters=_default_clusters())
        db_path = tmp_path / "voices.json"

        result = learn_from_transcript(
            json_path,
            {"SPEAKER_00": "Mario", "SPEAKER_01": "Luca"},
            db_path,
        )

        assert result["added"] == {"SPEAKER_00": "Mario", "SPEAKER_01": "Luca"}
        assert result["skipped_no_cluster"] == []
        assert result["embedding_model"] == "pyannote/fake-embedding"
        assert result["dry_run"] is False

        db = load_db(db_path)
        assert db["embedding_model"] == "pyannote/fake-embedding"
        assert set(db["speakers"]) == {"Mario", "Luca"}
        assert db["speakers"]["Mario"][0]["embedding"] == [0.1, 0.2, 0.3]
        assert db["speakers"]["Mario"][0]["source"] == "ep1_transcript.json"
        assert db["speakers"]["Mario"][0]["cluster"] == "SPEAKER_00"
        assert db["speakers"]["Mario"][0]["duration_s"] == 10.0

    def test_partial_map_only_adds_mapped_labels(self, tmp_path):
        json_path = tmp_path / "t.json"
        _make_transcript_json(json_path, clusters=_default_clusters())
        db_path = tmp_path / "voices.json"

        result = learn_from_transcript(json_path, {"SPEAKER_00": "Mario"}, db_path)
        assert result["added"] == {"SPEAKER_00": "Mario"}
        assert result["skipped_no_cluster"] == []
        assert set(load_db(db_path)["speakers"]) == {"Mario"}

    def test_appending_to_existing_speaker_accumulates(self, tmp_path):
        db_path = tmp_path / "voices.json"

        json1 = tmp_path / "ep1.json"
        json2 = tmp_path / "ep2.json"
        _make_transcript_json(json1, clusters=_default_clusters())
        _make_transcript_json(json2, clusters={
            "SPEAKER_00": {
                "embedding": [0.9, 0.9, 0.9],
                "duration_s": 7.0,
                "n_segments": 3,
                "embedding_model": "pyannote/fake-embedding",
            },
        })

        learn_from_transcript(json1, {"SPEAKER_00": "Mario"}, db_path)
        learn_from_transcript(json2, {"SPEAKER_00": "Mario"}, db_path)

        db = load_db(db_path)
        assert len(db["speakers"]["Mario"]) == 2
        sources = {s["source"] for s in db["speakers"]["Mario"]}
        assert sources == {"ep1.json", "ep2.json"}


class TestSkip:
    def test_label_with_no_cluster_is_skipped(self, tmp_path):
        json_path = tmp_path / "t.json"
        _make_transcript_json(json_path, clusters=_default_clusters())
        db_path = tmp_path / "voices.json"

        result = learn_from_transcript(
            json_path,
            {"SPEAKER_00": "Mario", "SPEAKER_99": "Ghost"},
            db_path,
        )
        assert result["added"] == {"SPEAKER_00": "Mario"}
        assert result["skipped_no_cluster"] == ["SPEAKER_99"]

    def test_empty_embedding_skipped(self, tmp_path):
        json_path = tmp_path / "t.json"
        _make_transcript_json(json_path, clusters={
            "SPEAKER_00": {
                "embedding": [],  # corrupted / empty
                "duration_s": 5.0,
                "n_segments": 1,
                "embedding_model": "pyannote/fake-embedding",
            },
        })
        db_path = tmp_path / "voices.json"
        result = learn_from_transcript(json_path, {"SPEAKER_00": "Mario"}, db_path)
        assert result["added"] == {}
        assert result["skipped_no_cluster"] == ["SPEAKER_00"]


class TestDryRun:
    def test_dry_run_does_not_write_db(self, tmp_path):
        json_path = tmp_path / "t.json"
        _make_transcript_json(json_path, clusters=_default_clusters())
        db_path = tmp_path / "voices.json"

        result = learn_from_transcript(
            json_path, {"SPEAKER_00": "Mario"}, db_path, dry_run=True
        )
        assert result["dry_run"] is True
        assert result["added"] == {"SPEAKER_00": "Mario"}
        assert not db_path.exists(), "dry-run must not create the DB file"


class TestSchemaValidation:
    def test_schema_v1_rejected(self, tmp_path):
        json_path = tmp_path / "t.json"
        _make_transcript_json(json_path, schema_version=1, clusters=_default_clusters())
        with pytest.raises(EnrollmentError, match="schema_version"):
            learn_from_transcript(json_path, {"SPEAKER_00": "Mario"}, tmp_path / "voices.json")

    def test_missing_schema_version_rejected(self, tmp_path):
        json_path = tmp_path / "t.json"
        json_path.write_text(json.dumps({"speaker_clusters": _default_clusters()}))
        with pytest.raises(EnrollmentError, match="schema_version"):
            learn_from_transcript(json_path, {"SPEAKER_00": "Mario"}, tmp_path / "voices.json")

    def test_missing_speaker_clusters_rejected(self, tmp_path):
        json_path = tmp_path / "t.json"
        json_path.write_text(json.dumps({"schema_version": 2}))
        with pytest.raises(EnrollmentError, match="speaker_clusters"):
            learn_from_transcript(json_path, {"SPEAKER_00": "Mario"}, tmp_path / "voices.json")

    def test_empty_speaker_clusters_rejected(self, tmp_path):
        json_path = tmp_path / "t.json"
        _make_transcript_json(json_path, clusters={})
        with pytest.raises(EnrollmentError, match="speaker_clusters"):
            learn_from_transcript(json_path, {"SPEAKER_00": "Mario"}, tmp_path / "voices.json")

    def test_inconsistent_embedding_models_rejected(self, tmp_path):
        json_path = tmp_path / "t.json"
        _make_transcript_json(json_path, clusters={
            "SPEAKER_00": {
                "embedding": [0.1], "duration_s": 1.0, "n_segments": 1,
                "embedding_model": "model-a",
            },
            "SPEAKER_01": {
                "embedding": [0.2], "duration_s": 1.0, "n_segments": 1,
                "embedding_model": "model-b",
            },
        })
        with pytest.raises(EnrollmentError, match="inconsistent"):
            learn_from_transcript(json_path, {"SPEAKER_00": "Mario"}, tmp_path / "voices.json")

    def test_missing_embedding_model_rejected(self, tmp_path):
        json_path = tmp_path / "t.json"
        _make_transcript_json(json_path, clusters={
            "SPEAKER_00": {"embedding": [0.1], "duration_s": 1.0, "n_segments": 1},
        })
        with pytest.raises(EnrollmentError, match="embedding_model"):
            learn_from_transcript(json_path, {"SPEAKER_00": "Mario"}, tmp_path / "voices.json")


class TestDbCompatibility:
    def test_mismatched_db_model_rejected(self, tmp_path):
        db_path = tmp_path / "voices.json"
        # Pre-populate DB with a different model.
        save_db(
            {"schema_version": 1, "embedding_model": "old-model", "speakers": {}},
            db_path,
        )

        json_path = tmp_path / "t.json"
        _make_transcript_json(json_path, clusters=_default_clusters())

        with pytest.raises(EnrollmentError, match="cannot mix"):
            learn_from_transcript(json_path, {"SPEAKER_00": "Mario"}, db_path)


class TestIoErrors:
    def test_missing_json_file(self, tmp_path):
        with pytest.raises(EnrollmentError, match="JSON file not found"):
            learn_from_transcript(
                tmp_path / "missing.json", {"SPEAKER_00": "Mario"}, tmp_path / "voices.json"
            )

    def test_invalid_json(self, tmp_path):
        json_path = tmp_path / "bad.json"
        json_path.write_text("{ not valid json", encoding="utf-8")
        with pytest.raises(EnrollmentError, match="Invalid JSON"):
            learn_from_transcript(
                json_path, {"SPEAKER_00": "Mario"}, tmp_path / "voices.json"
            )
