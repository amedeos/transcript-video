"""Tests for the user-level voice-print DB.

The module under test is torch-free. These tests are pure-Python and rely
only on the standard library and pytest, matching the constraints enforced
by :mod:`tests.test_torch_free`.
"""

from __future__ import annotations

import json
import stat

import pytest

from transcript_video.speaker_db import (
    DB_SCHEMA_VERSION,
    add_sample,
    default_db_path,
    embedding_model_compatible,
    load_db,
    match,
    resolve_db_path,
    save_db,
)


class TestResolveDbPath:
    def test_cli_path_wins_over_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRANSCRIPT_VIDEO_VOICE_DB", str(tmp_path / "env.json"))
        result = resolve_db_path(str(tmp_path / "cli.json"))
        assert result == tmp_path / "cli.json"

    def test_env_var_used_when_no_cli(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRANSCRIPT_VIDEO_VOICE_DB", str(tmp_path / "env.json"))
        result = resolve_db_path(None)
        assert result == tmp_path / "env.json"

    def test_default_uses_xdg_data_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        monkeypatch.delenv("TRANSCRIPT_VIDEO_VOICE_DB", raising=False)
        result = resolve_db_path(None)
        assert result == tmp_path / "transcript-video" / "voices.json"

    def test_default_falls_back_to_local_share(self, tmp_path, monkeypatch):
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.delenv("TRANSCRIPT_VIDEO_VOICE_DB", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        result = resolve_db_path(None)
        assert result == tmp_path / ".local" / "share" / "transcript-video" / "voices.json"

    def test_user_expansion_applied(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        result = resolve_db_path("~/custom/voices.json")
        assert result == tmp_path / "custom" / "voices.json"

    def test_default_db_path_matches_resolve_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        monkeypatch.delenv("TRANSCRIPT_VIDEO_VOICE_DB", raising=False)
        assert default_db_path() == resolve_db_path(None)


class TestLoadSave:
    def test_load_missing_returns_empty_db(self, tmp_path):
        db = load_db(tmp_path / "does-not-exist.json")
        assert db["schema_version"] == DB_SCHEMA_VERSION
        assert db["speakers"] == {}
        assert db["embedding_model"] is None
        assert "created_at" in db

    def test_save_then_load_roundtrip(self, tmp_path):
        path = tmp_path / "voices.json"
        db = load_db(path)
        add_sample(
            db, "Mario", [0.1, 0.2, 0.3],
            source="ep1.json", embedding_model="model-x",
            cluster="SPEAKER_00", duration_s=42.0,
        )
        save_db(db, path)
        loaded = load_db(path)
        assert loaded["embedding_model"] == "model-x"
        assert loaded["speakers"]["Mario"][0]["embedding"] == [0.1, 0.2, 0.3]
        assert loaded["speakers"]["Mario"][0]["cluster"] == "SPEAKER_00"
        assert loaded["speakers"]["Mario"][0]["duration_s"] == 42.0
        assert loaded["speakers"]["Mario"][0]["source"] == "ep1.json"

    def test_save_creates_nested_directory(self, tmp_path):
        path = tmp_path / "a" / "b" / "voices.json"
        save_db(load_db(path), path)
        assert path.exists()

    def test_save_sets_strict_permissions(self, tmp_path):
        path = tmp_path / "voices.json"
        save_db(load_db(path), path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    def test_save_tightens_loose_directory_permissions(self, tmp_path):
        # Pre-create the directory with loose perms, then save_db should tighten.
        d = tmp_path / "voices_dir"
        d.mkdir(mode=0o755)
        path = d / "voices.json"
        save_db(load_db(path), path)
        assert stat.S_IMODE(d.stat().st_mode) == 0o700

    def test_save_preserves_symlink(self, tmp_path):
        target = tmp_path / "real.json"
        target.write_text(json.dumps({
            "schema_version": DB_SCHEMA_VERSION,
            "embedding_model": None,
            "speakers": {},
        }))
        link = tmp_path / "link.json"
        link.symlink_to(target)

        db = load_db(link)
        add_sample(db, "Mario", [0.1, 0.2], source="ep1.json", embedding_model="m")
        save_db(db, link)

        assert link.is_symlink(), "symlink at the user-visible path must survive atomic write"
        assert "Mario" in load_db(target)["speakers"]

    def test_save_is_atomic_under_failure(self, tmp_path, monkeypatch):
        path = tmp_path / "voices.json"
        save_db({"schema_version": DB_SCHEMA_VERSION, "embedding_model": "m", "speakers": {}}, path)
        original_content = path.read_text()

        import os as _os
        real_replace = _os.replace

        def boom(src, dst):
            raise OSError("simulated disk failure")

        monkeypatch.setattr(_os, "replace", boom)
        with pytest.raises(OSError):
            save_db({"schema_version": DB_SCHEMA_VERSION, "embedding_model": "m", "speakers": {"X": []}}, path)

        # Original file is intact; no half-written tmp left in the dir.
        assert path.read_text() == original_content
        leftover = [p for p in path.parent.iterdir() if p.name.startswith(".voices.")]
        assert leftover == [], f"tempfile leak: {leftover}"

        # Sanity: with replace restored, save_db works again.
        monkeypatch.setattr(_os, "replace", real_replace)


class TestAddSample:
    def _empty(self) -> dict:
        return {"schema_version": DB_SCHEMA_VERSION, "embedding_model": None, "speakers": {}}

    def test_first_add_sets_model(self):
        db = self._empty()
        add_sample(db, "Mario", [0.1], source="s", embedding_model="m")
        assert db["embedding_model"] == "m"

    def test_consistent_model_accepted(self):
        db = self._empty()
        add_sample(db, "Mario", [0.1], source="s", embedding_model="m")
        add_sample(db, "Luca", [0.2], source="s", embedding_model="m")
        assert set(db["speakers"]) == {"Mario", "Luca"}

    def test_inconsistent_model_rejected(self):
        db = self._empty()
        add_sample(db, "Mario", [0.1], source="s", embedding_model="m1")
        with pytest.raises(ValueError, match="cannot add sample"):
            add_sample(db, "Mario", [0.2], source="s", embedding_model="m2")

    def test_multiple_samples_per_speaker_accumulate(self):
        db = self._empty()
        add_sample(db, "Mario", [0.1, 0.2], source="ep1", embedding_model="m")
        add_sample(db, "Mario", [0.3, 0.4], source="ep2", embedding_model="m")
        assert len(db["speakers"]["Mario"]) == 2
        assert [s["source"] for s in db["speakers"]["Mario"]] == ["ep1", "ep2"]

    def test_empty_name_rejected(self):
        db = self._empty()
        with pytest.raises(ValueError):
            add_sample(db, "", [0.1], source="s", embedding_model="m")

    def test_empty_embedding_rejected(self):
        db = self._empty()
        with pytest.raises(ValueError):
            add_sample(db, "Mario", [], source="s", embedding_model="m")

    def test_empty_model_rejected(self):
        db = self._empty()
        with pytest.raises(ValueError):
            add_sample(db, "Mario", [0.1], source="s", embedding_model="")


class TestMatch:
    def _db(self, speakers: dict[str, list[dict]]) -> dict:
        return {
            "schema_version": DB_SCHEMA_VERSION,
            "embedding_model": "m",
            "speakers": speakers,
        }

    def test_empty_db_returns_none(self):
        assert match([1.0, 0.0], self._db({})) is None

    def test_exact_alignment_scores_one(self):
        db = self._db({"Mario": [{"embedding": [1.0, 0.0], "source": "s"}]})
        result = match([1.0, 0.0], db, threshold=0.5)
        assert result is not None
        assert result.name == "Mario"
        assert result.score == pytest.approx(1.0)
        assert result.n_samples == 1

    def test_orthogonal_returns_none(self):
        db = self._db({"Mario": [{"embedding": [1.0, 0.0], "source": "s"}]})
        assert match([0.0, 1.0], db, threshold=0.5) is None

    def test_picks_best_candidate(self):
        db = self._db({
            "Mario": [{"embedding": [1.0, 0.0], "source": "s"}],
            "Luca":  [{"embedding": [0.5, 0.5], "source": "s"}],
        })
        result = match([1.0, 0.0], db, threshold=0.1)
        assert result is not None
        assert result.name == "Mario"

    def test_top_k_one_uses_max_sample(self):
        # Mario has one poor and one perfect sample; top_k=1 means the perfect wins.
        db = self._db({"Mario": [
            {"embedding": [0.5, 0.5], "source": "noisy"},
            {"embedding": [1.0, 0.0], "source": "clean"},
        ]})
        result = match([1.0, 0.0], db, threshold=0.5, top_k=1)
        assert result is not None
        assert result.score == pytest.approx(1.0)

    def test_top_k_two_averages_top_two(self):
        # Both samples score 1.0; average is also 1.0.
        db = self._db({"Mario": [
            {"embedding": [1.0, 0.0], "source": "a"},
            {"embedding": [1.0, 0.0], "source": "b"},
            {"embedding": [0.0, 1.0], "source": "c"},  # orthogonal — would drag avg down if averaged in
        ]})
        result = match([1.0, 0.0], db, threshold=0.5, top_k=2)
        assert result is not None
        assert result.score == pytest.approx(1.0)

    def test_threshold_filters_out_weak_match(self):
        db = self._db({"Mario": [{"embedding": [0.7, 0.7], "source": "s"}]})  # cos vs [1,0] ≈ 0.707
        # 0.707 passes threshold 0.5...
        assert match([1.0, 0.0], db, threshold=0.5) is not None
        # ...but not threshold 0.8.
        assert match([1.0, 0.0], db, threshold=0.8) is None

    def test_zero_vector_in_db_is_safe(self):
        db = self._db({"Mario": [{"embedding": [0.0, 0.0], "source": "s"}]})
        assert match([1.0, 0.0], db, threshold=0.1) is None

    def test_mismatched_lengths_safe(self):
        db = self._db({"Mario": [{"embedding": [1.0, 0.0, 0.0], "source": "s"}]})
        # Different dim → cosine 0.0 → no match.
        assert match([1.0, 0.0], db, threshold=0.1) is None

    def test_empty_speaker_list_skipped(self):
        # A speaker with zero samples should be silently skipped, not crash.
        db = self._db({"Ghost": [], "Mario": [{"embedding": [1.0, 0.0], "source": "s"}]})
        result = match([1.0, 0.0], db, threshold=0.5)
        assert result is not None
        assert result.name == "Mario"


class TestEmbeddingModelCompatible:
    def test_empty_db_accepts_any_model(self):
        db = {"schema_version": DB_SCHEMA_VERSION, "embedding_model": None, "speakers": {}}
        assert embedding_model_compatible(db, "anything")

    def test_matching_model_compatible(self):
        db = {"schema_version": DB_SCHEMA_VERSION, "embedding_model": "m1", "speakers": {}}
        assert embedding_model_compatible(db, "m1")

    def test_different_model_incompatible(self):
        db = {"schema_version": DB_SCHEMA_VERSION, "embedding_model": "m1", "speakers": {}}
        assert not embedding_model_compatible(db, "m2")
