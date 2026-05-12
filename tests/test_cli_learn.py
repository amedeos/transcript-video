"""Tests for the ``transcript-learn`` CLI entry point."""

from __future__ import annotations

import json

import pytest

from transcript_video.cli_learn import main
from transcript_video.speaker_db import load_db


def _make_transcript(path, *, clusters=None):
    payload = {"schema_version": 2}
    if clusters is None:
        clusters = {
            "SPEAKER_00": {
                "embedding": [0.1, 0.2],
                "duration_s": 10.0,
                "n_segments": 3,
                "embedding_model": "pyannote/fake",
            },
            "SPEAKER_01": {
                "embedding": [0.3, 0.4],
                "duration_s": 5.0,
                "n_segments": 2,
                "embedding_model": "pyannote/fake",
            },
        }
    payload["speaker_clusters"] = clusters
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestHappyPath:
    def test_adds_samples_and_prints_summary(self, tmp_path, capsys):
        json_path = tmp_path / "ep1_transcript.json"
        _make_transcript(json_path)
        db_path = tmp_path / "voices.json"

        main([
            str(json_path),
            "--speaker-map", "SPEAKER_00=Mario,SPEAKER_01=Luca",
            "--voice-db", str(db_path),
        ])

        out = capsys.readouterr().out
        assert "Added 2 sample(s)" in out
        assert "SPEAKER_00 -> Mario" in out
        assert "SPEAKER_01 -> Luca" in out

        db = load_db(db_path)
        assert set(db["speakers"]) == {"Mario", "Luca"}

    def test_dry_run_does_not_write_db(self, tmp_path, capsys):
        json_path = tmp_path / "t.json"
        _make_transcript(json_path)
        db_path = tmp_path / "voices.json"

        main([
            str(json_path),
            "--speaker-map", "SPEAKER_00=Mario",
            "--voice-db", str(db_path),
            "--dry-run",
        ])

        out = capsys.readouterr().out
        assert "Would add" in out
        assert not db_path.exists()

    def test_skipped_labels_reported_on_stderr(self, tmp_path, capsys):
        json_path = tmp_path / "t.json"
        _make_transcript(json_path)
        db_path = tmp_path / "voices.json"

        main([
            str(json_path),
            "--speaker-map", "SPEAKER_00=Mario,SPEAKER_99=Ghost",
            "--voice-db", str(db_path),
        ])

        captured = capsys.readouterr()
        assert "Added 1 sample(s)" in captured.out
        assert "SPEAKER_99" in captured.err

    def test_voice_db_env_var_used_when_no_flag(self, tmp_path, capsys, monkeypatch):
        json_path = tmp_path / "t.json"
        _make_transcript(json_path)
        env_db = tmp_path / "from_env.json"
        monkeypatch.setenv("TRANSCRIPT_VIDEO_VOICE_DB", str(env_db))
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)

        main([str(json_path), "--speaker-map", "SPEAKER_00=Mario"])

        assert env_db.exists()


class TestErrorPaths:
    def test_missing_json_file_exits(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            main([str(tmp_path / "nope.json"), "--speaker-map", "SPEAKER_00=Mario"])
        assert exc.value.code == 1
        assert "not found" in capsys.readouterr().err

    def test_no_speaker_map_exits(self, tmp_path, capsys):
        json_path = tmp_path / "t.json"
        _make_transcript(json_path)
        with pytest.raises(SystemExit) as exc:
            main([str(json_path), "--voice-db", str(tmp_path / "v.json")])
        assert exc.value.code == 1
        assert "no speaker mapping" in capsys.readouterr().err.lower()

    def test_schema_v1_exits_with_clear_message(self, tmp_path, capsys):
        json_path = tmp_path / "old.json"
        json_path.write_text(json.dumps({"schema_version": 1, "segments": []}))

        with pytest.raises(SystemExit) as exc:
            main([
                str(json_path),
                "--speaker-map", "SPEAKER_00=Mario",
                "--voice-db", str(tmp_path / "v.json"),
            ])
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "schema_version" in err
        assert "Re-process" in err

    def test_no_clusters_exits_with_clear_message(self, tmp_path, capsys):
        json_path = tmp_path / "nodiar.json"
        json_path.write_text(json.dumps({"schema_version": 2}))

        with pytest.raises(SystemExit) as exc:
            main([
                str(json_path),
                "--speaker-map", "SPEAKER_00=Mario",
                "--voice-db", str(tmp_path / "v.json"),
            ])
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "speaker_clusters" in err

    def test_mismatched_db_model_exits(self, tmp_path, capsys):
        from transcript_video.speaker_db import save_db

        json_path = tmp_path / "t.json"
        _make_transcript(json_path)
        db_path = tmp_path / "voices.json"
        save_db(
            {"schema_version": 1, "embedding_model": "other-model", "speakers": {}},
            db_path,
        )

        with pytest.raises(SystemExit) as exc:
            main([
                str(json_path),
                "--speaker-map", "SPEAKER_00=Mario",
                "--voice-db", str(db_path),
            ])
        assert exc.value.code == 1
        assert "cannot mix" in capsys.readouterr().err
