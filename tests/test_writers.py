"""Tests for the JSON/SRT/VTT/TXT writers."""

from __future__ import annotations

import json

from transcript_video.writers import write_json, write_srt, write_txt, write_vtt


def _make_segments() -> list[dict]:
    return [
        {"id": 0, "start": 0.0, "end": 2.5, "text": " Hello world. "},
        {"id": 1, "start": 3.0, "end": 5.5, "text": " How are you? "},
    ]


def _make_diarized_segments() -> list[dict]:
    return [
        {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello world.", "speaker": "SPEAKER_00"},
        {"id": 1, "start": 3.0, "end": 5.5, "text": "How are you?", "speaker": "SPEAKER_01"},
    ]


class TestWriteJson:
    def test_writes_utf8_indented(self, tmp_path):
        out = tmp_path / "deep" / "out.json"
        write_json({"hello": "città"}, out)
        text = out.read_text(encoding="utf-8")
        # ensure_ascii=False keeps non-ASCII bytes as-is.
        assert "città" in text
        # Indented output (indent=2) for human readability.
        assert "\n" in text
        # Re-parsing must yield the original.
        assert json.loads(text) == {"hello": "città"}

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "a" / "b" / "c.json"
        write_json({"x": 1}, out)
        assert out.exists()


class TestWriteSrt:
    def test_format(self, tmp_path):
        out = tmp_path / "subs.srt"
        write_srt(_make_segments(), out)
        content = out.read_text(encoding="utf-8")
        # Index lines start at 1.
        assert content.startswith("1\n")
        assert "2\n" in content
        # SRT timestamps with comma separator.
        assert "00:00:00,000 --> 00:00:02,500" in content
        assert "00:00:03,000 --> 00:00:05,500" in content
        # Text is stripped (no leading/trailing spaces).
        assert "Hello world.\n" in content
        # Blank line between cues.
        assert "\n\n" in content

    def test_empty_segments(self, tmp_path):
        out = tmp_path / "empty.srt"
        write_srt([], out)
        assert out.read_text(encoding="utf-8") == ""

    def test_speaker_names_default_off(self, tmp_path):
        # Without a speaker_names map, cues carry only the raw text.
        out = tmp_path / "subs.srt"
        write_srt(_make_diarized_segments(), out)
        content = out.read_text(encoding="utf-8")
        assert "Hello world.\n" in content
        assert "SPEAKER_00" not in content

    def test_speaker_names_prefix(self, tmp_path):
        out = tmp_path / "subs.srt"
        names = {"SPEAKER_00": "Amedeo", "SPEAKER_01": "Tizio"}
        write_srt(_make_diarized_segments(), out, speaker_names=names)
        content = out.read_text(encoding="utf-8")
        assert "Amedeo: Hello world.\n" in content
        assert "Tizio: How are you?\n" in content

    def test_speaker_names_unmapped_falls_back_to_label(self, tmp_path):
        # A label with no name in the map keeps the raw SPEAKER_XX label.
        out = tmp_path / "subs.srt"
        write_srt(_make_diarized_segments(), out, speaker_names={"SPEAKER_00": "Amedeo"})
        content = out.read_text(encoding="utf-8")
        assert "Amedeo: Hello world.\n" in content
        assert "SPEAKER_01: How are you?\n" in content


class TestWriteVtt:
    def test_format(self, tmp_path):
        out = tmp_path / "subs.vtt"
        write_vtt(_make_segments(), out)
        content = out.read_text(encoding="utf-8")
        # WebVTT files must begin with the WEBVTT header.
        assert content.startswith("WEBVTT\n\n")
        # Index lines start at 1.
        assert "1\n" in content
        # VTT timestamps use a dot separator.
        assert "00:00:00.000 --> 00:00:02.500" in content
        assert "00:00:03.000 --> 00:00:05.500" in content
        # Text is stripped.
        assert "Hello world.\n" in content

    def test_empty_segments_writes_header_only(self, tmp_path):
        out = tmp_path / "empty.vtt"
        write_vtt([], out)
        assert out.read_text(encoding="utf-8") == "WEBVTT\n\n"

    def test_speaker_names_prefix(self, tmp_path):
        out = tmp_path / "subs.vtt"
        names = {"SPEAKER_00": "Amedeo", "SPEAKER_01": "Tizio"}
        write_vtt(_make_diarized_segments(), out, speaker_names=names)
        content = out.read_text(encoding="utf-8")
        assert "Amedeo: Hello world.\n" in content
        assert "Tizio: How are you?\n" in content


class TestWriteTxt:
    def test_one_segment_per_line(self, tmp_path):
        out = tmp_path / "out.txt"
        write_txt(_make_segments(), out)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines == ["Hello world.", "How are you?"]

    def test_handles_missing_text(self, tmp_path):
        out = tmp_path / "out.txt"
        write_txt([{"start": 0.0, "end": 1.0}], out)
        # Missing text produces an empty line (still one line per segment).
        assert out.read_text(encoding="utf-8") == "\n"
