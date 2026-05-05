"""Tests for speaker map parsing (inline string + YAML/JSON sidecars)."""

from __future__ import annotations

import json

import pytest

from transcript_video.speakers import (
    display_name,
    parse_speaker_map_file,
    parse_speaker_map_inline,
    resolve_speaker_map,
)


class TestParseSpeakerMapInline:
    def test_typical(self):
        assert parse_speaker_map_inline("SPEAKER_00=Amedeo,SPEAKER_01=Tizio") == {
            "SPEAKER_00": "Amedeo",
            "SPEAKER_01": "Tizio",
        }

    def test_empty_string(self):
        assert parse_speaker_map_inline("") == {}

    def test_whitespace_is_stripped(self):
        assert parse_speaker_map_inline(" SPEAKER_00 = Amedeo , SPEAKER_01 = Tizio ") == {
            "SPEAKER_00": "Amedeo",
            "SPEAKER_01": "Tizio",
        }

    def test_value_can_contain_equals(self):
        # split with maxsplit=1 — first '=' is the separator.
        assert parse_speaker_map_inline("SPEAKER_00=foo=bar") == {"SPEAKER_00": "foo=bar"}

    def test_skips_empty_chunks(self):
        assert parse_speaker_map_inline("SPEAKER_00=A,,SPEAKER_01=B") == {
            "SPEAKER_00": "A",
            "SPEAKER_01": "B",
        }

    def test_missing_equals_exits(self):
        with pytest.raises(SystemExit):
            parse_speaker_map_inline("SPEAKER_00")

    def test_empty_label_exits(self):
        with pytest.raises(SystemExit):
            parse_speaker_map_inline("=Amedeo")

    def test_empty_name_exits(self):
        with pytest.raises(SystemExit):
            parse_speaker_map_inline("SPEAKER_00=")


class TestParseSpeakerMapFile:
    def test_yaml(self, tmp_path):
        p = tmp_path / "map.yaml"
        p.write_text("SPEAKER_00: Amedeo\nSPEAKER_01: Tizio\n", encoding="utf-8")
        assert parse_speaker_map_file(p) == {"SPEAKER_00": "Amedeo", "SPEAKER_01": "Tizio"}

    def test_yml_extension(self, tmp_path):
        p = tmp_path / "map.yml"
        p.write_text("SPEAKER_00: Amedeo\n", encoding="utf-8")
        assert parse_speaker_map_file(p) == {"SPEAKER_00": "Amedeo"}

    def test_json(self, tmp_path):
        p = tmp_path / "map.json"
        p.write_text(json.dumps({"SPEAKER_00": "Amedeo", "SPEAKER_01": "Tizio"}), encoding="utf-8")
        assert parse_speaker_map_file(p) == {"SPEAKER_00": "Amedeo", "SPEAKER_01": "Tizio"}

    def test_missing_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            parse_speaker_map_file(tmp_path / "missing.yaml")

    def test_invalid_yaml_exits(self, tmp_path):
        p = tmp_path / "broken.yaml"
        p.write_text("SPEAKER_00: [unterminated\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            parse_speaker_map_file(p)

    def test_invalid_json_exits(self, tmp_path):
        p = tmp_path / "broken.json"
        p.write_text("{not json", encoding="utf-8")
        with pytest.raises(SystemExit):
            parse_speaker_map_file(p)

    def test_unsupported_extension_exits(self, tmp_path):
        p = tmp_path / "map.toml"
        p.write_text("SPEAKER_00 = 'Amedeo'\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            parse_speaker_map_file(p)

    def test_non_mapping_root_exits(self, tmp_path):
        p = tmp_path / "list.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(SystemExit):
            parse_speaker_map_file(p)


class TestResolveSpeakerMap:
    def test_inline_wins_over_file(self, tmp_path):
        # If inline is provided, the file is ignored.
        p = tmp_path / "map.yaml"
        p.write_text("SPEAKER_00: FromFile\n", encoding="utf-8")
        result = resolve_speaker_map("SPEAKER_00=Inline", str(p))
        assert result == {"SPEAKER_00": "Inline"}

    def test_file_used_when_no_inline(self, tmp_path):
        p = tmp_path / "map.json"
        p.write_text('{"SPEAKER_00": "FromFile"}', encoding="utf-8")
        assert resolve_speaker_map(None, str(p)) == {"SPEAKER_00": "FromFile"}

    def test_neither_returns_empty(self):
        assert resolve_speaker_map(None, None) == {}


class TestDisplayName:
    def test_lookup_hits(self):
        assert display_name("SPEAKER_00", {"SPEAKER_00": "Amedeo"}) == "Amedeo"

    def test_fallback_to_label(self):
        assert display_name("SPEAKER_99", {"SPEAKER_00": "Amedeo"}) == "SPEAKER_99"
