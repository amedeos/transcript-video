"""Tests for the transcript-to-md CLI: --list-speakers and end-to-end re-render."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transcript_video.cli_to_md import (
    _format_speaker_overview,
    _resolve_speaker_stats,
    main,
)


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
