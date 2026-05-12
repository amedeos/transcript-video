"""Tests for the ``transcript-voices`` CLI entry point."""

from __future__ import annotations

import pytest

from transcript_video.cli_voices import main
from transcript_video.speaker_db import add_sample, load_db, save_db


def _seed_db(path, speakers: dict | None = None, model: str = "pyannote/fake") -> None:
    db = {"schema_version": 1, "embedding_model": model, "speakers": {}}
    if speakers:
        for name, samples in speakers.items():
            for s in samples:
                add_sample(
                    db, name, s["embedding"],
                    source=s["source"],
                    embedding_model=model,
                    cluster=s.get("cluster"),
                    duration_s=s.get("duration_s"),
                )
    save_db(db, path)


class TestList:
    def test_empty_db(self, tmp_path, capsys):
        db_path = tmp_path / "voices.json"
        main(["list", "--voice-db", str(db_path)])
        out = capsys.readouterr().out
        assert str(db_path) in out
        assert "no speakers enrolled" in out.lower()

    def test_lists_speakers_alphabetically(self, tmp_path, capsys):
        db_path = tmp_path / "voices.json"
        _seed_db(db_path, {
            "Mario": [{"embedding": [0.1], "source": "ep1", "duration_s": 5.0}],
            "Anna":  [{"embedding": [0.2], "source": "ep1", "duration_s": 3.0}],
            "Luca":  [
                {"embedding": [0.3], "source": "ep1", "duration_s": 7.0},
                {"embedding": [0.4], "source": "ep2", "duration_s": 9.0},
            ],
        })
        main(["list", "--voice-db", str(db_path)])
        out = capsys.readouterr().out
        # Alphabetic order: Anna, Luca, Mario.
        anna_pos = out.find("Anna")
        luca_pos = out.find("Luca")
        mario_pos = out.find("Mario")
        assert -1 < anna_pos < luca_pos < mario_pos
        assert "2 samples" in out
        assert "1 sample" in out  # singular form

    def test_default_action_is_list(self, tmp_path, capsys):
        # `transcript-voices --voice-db X` (no sub-command) should default to list.
        db_path = tmp_path / "voices.json"
        # Without --voice-db on top-level: passing just no args defaults to list
        # on the default DB. We seed env so the default points at a temp dir.
        _seed_db(db_path, {"Mario": [{"embedding": [0.1], "source": "ep1", "duration_s": 5}]})
        # Simulate env-driven default to avoid touching real ~/.local/share.
        # The cleanest test: pass [] and ensure no SystemExit and we hit list path.
        # We point the default to our tmp DB via env var.
        # Reproduce by invoking the explicit list path; the bare-no-arg path is
        # covered separately below.
        main(["list", "--voice-db", str(db_path)])
        out = capsys.readouterr().out
        assert "Mario" in out

    def test_bare_invocation_defaults_to_list(self, tmp_path, capsys, monkeypatch):
        db_path = tmp_path / "voices.json"
        _seed_db(db_path, {"Mario": [{"embedding": [0.1], "source": "ep1", "duration_s": 5}]})
        monkeypatch.setenv("TRANSCRIPT_VIDEO_VOICE_DB", str(db_path))
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        main([])  # bare: no sub-command, no --voice-db
        out = capsys.readouterr().out
        assert "Mario" in out
        assert "Voice DB" in out


class TestShow:
    def test_existing_speaker(self, tmp_path, capsys):
        db_path = tmp_path / "voices.json"
        _seed_db(db_path, {
            "Mario": [
                {"embedding": [0.1], "source": "ep1.json", "cluster": "SPEAKER_00", "duration_s": 234.5},
                {"embedding": [0.2], "source": "ep2.json", "cluster": "SPEAKER_01", "duration_s": 87.0},
            ],
        })
        main(["show", "Mario", "--voice-db", str(db_path)])
        out = capsys.readouterr().out
        assert "Mario" in out
        assert "2 sample" in out
        assert "ep1.json" in out
        assert "ep2.json" in out
        assert "SPEAKER_00" in out
        assert "SPEAKER_01" in out
        assert "234.5" in out

    def test_missing_speaker(self, tmp_path, capsys):
        db_path = tmp_path / "voices.json"
        _seed_db(db_path, {"Mario": [{"embedding": [0.1], "source": "s", "duration_s": 1.0}]})
        main(["show", "Ghost", "--voice-db", str(db_path)])
        out = capsys.readouterr().out
        assert "No speaker named 'Ghost'" in out


class TestForget:
    def test_forget_all_with_yes(self, tmp_path, capsys):
        db_path = tmp_path / "voices.json"
        _seed_db(db_path, {
            "Mario": [
                {"embedding": [0.1], "source": "ep1", "duration_s": 5.0},
                {"embedding": [0.2], "source": "ep2", "duration_s": 5.0},
            ],
            "Luca": [{"embedding": [0.3], "source": "ep1", "duration_s": 3.0}],
        })
        main(["forget", "Mario", "--yes", "--voice-db", str(db_path)])
        out = capsys.readouterr().out
        assert "Removed 2 sample" in out

        db = load_db(db_path)
        assert "Mario" not in db["speakers"]
        assert "Luca" in db["speakers"], "untouched speakers preserved"

    def test_forget_specific_source_with_yes(self, tmp_path, capsys):
        db_path = tmp_path / "voices.json"
        _seed_db(db_path, {
            "Mario": [
                {"embedding": [0.1], "source": "ep1", "duration_s": 5.0},
                {"embedding": [0.2], "source": "ep2", "duration_s": 5.0},
            ],
        })
        main([
            "forget", "Mario",
            "--source", "ep1",
            "--yes",
            "--voice-db", str(db_path),
        ])
        out = capsys.readouterr().out
        assert "Removed 1 sample" in out

        db = load_db(db_path)
        assert len(db["speakers"]["Mario"]) == 1
        assert db["speakers"]["Mario"][0]["source"] == "ep2"

    def test_forget_missing_speaker(self, tmp_path, capsys):
        db_path = tmp_path / "voices.json"
        _seed_db(db_path, {"Mario": [{"embedding": [0.1], "source": "ep1", "duration_s": 5}]})
        main(["forget", "Ghost", "--yes", "--voice-db", str(db_path)])
        out = capsys.readouterr().out
        assert "No speaker named 'Ghost'" in out
        # DB is untouched.
        assert "Mario" in load_db(db_path)["speakers"]

    def test_forget_missing_source(self, tmp_path, capsys):
        db_path = tmp_path / "voices.json"
        _seed_db(db_path, {"Mario": [{"embedding": [0.1], "source": "ep1", "duration_s": 5}]})
        main([
            "forget", "Mario",
            "--source", "nope.json",
            "--yes",
            "--voice-db", str(db_path),
        ])
        out = capsys.readouterr().out
        assert "No samples for 'Mario' with source 'nope.json'" in out
        # DB is untouched.
        assert load_db(db_path)["speakers"]["Mario"], "non-matching source must not delete anything"

    def test_confirmation_y_proceeds(self, tmp_path, capsys, monkeypatch):
        db_path = tmp_path / "voices.json"
        _seed_db(db_path, {"Mario": [{"embedding": [0.1], "source": "ep1", "duration_s": 5}]})
        monkeypatch.setattr("builtins.input", lambda _: "y")

        main(["forget", "Mario", "--voice-db", str(db_path)])
        out = capsys.readouterr().out
        assert "About to remove" in out
        assert "Removed 1 sample" in out
        assert "Mario" not in load_db(db_path)["speakers"]

    def test_confirmation_n_aborts(self, tmp_path, capsys, monkeypatch):
        db_path = tmp_path / "voices.json"
        _seed_db(db_path, {"Mario": [{"embedding": [0.1], "source": "ep1", "duration_s": 5}]})
        monkeypatch.setattr("builtins.input", lambda _: "n")

        with pytest.raises(SystemExit) as exc:
            main(["forget", "Mario", "--voice-db", str(db_path)])
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "Aborted" in out
        # DB untouched.
        assert "Mario" in load_db(db_path)["speakers"]

    def test_confirmation_eof_aborts(self, tmp_path, capsys, monkeypatch):
        db_path = tmp_path / "voices.json"
        _seed_db(db_path, {"Mario": [{"embedding": [0.1], "source": "ep1", "duration_s": 5}]})

        def _eof(_):
            raise EOFError()

        monkeypatch.setattr("builtins.input", _eof)
        with pytest.raises(SystemExit) as exc:
            main(["forget", "Mario", "--voice-db", str(db_path)])
        assert exc.value.code == 1
        # DB untouched.
        assert "Mario" in load_db(db_path)["speakers"]
