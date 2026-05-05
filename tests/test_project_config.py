"""Tests for the project-level TOML config resolution."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from transcript_video import project_config


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


class TestFindConfigFile:
    def test_explicit_path(self, tmp_path):
        p = _write(tmp_path, "custom.toml", "beam_size = 12")
        assert project_config.find_config_file(p, cwd=tmp_path) == p

    def test_explicit_missing_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            project_config.find_config_file(tmp_path / "nope.toml", cwd=tmp_path)

    def test_cwd_lookup(self, tmp_path):
        p = _write(tmp_path, "transcript-video.toml", "beam_size = 8")
        assert project_config.find_config_file(None, cwd=tmp_path) == p

    def test_input_dir_lookup(self, tmp_path):
        sub = tmp_path / "videos"
        sub.mkdir()
        p = _write(sub, "transcript-video.toml", "beam_size = 8")
        # cwd has no config; the video lives in sub/.
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        fake_input = sub / "video.mp4"
        fake_input.write_bytes(b"")
        assert project_config.find_config_file(None, cwd=elsewhere, input_file=fake_input) == p

    def test_no_config_returns_none(self, tmp_path):
        elsewhere = tmp_path / "no-config-here"
        elsewhere.mkdir()
        assert project_config.find_config_file(None, cwd=elsewhere) is None


class TestApplyProfile:
    def test_no_profile(self, tmp_path):
        p = _write(
            tmp_path,
            "c.toml",
            """
            beam_size = 10
            anti_loop = true
            """,
        )
        data = project_config.load_config(p)
        flat, smap = project_config.apply_profile(data, profile=None)
        assert flat == {"beam_size": 10, "anti_loop": True}
        assert smap == {}

    def test_profile_overlays_top_level(self, tmp_path):
        p = _write(
            tmp_path,
            "c.toml",
            """
            beam_size = 5

            [profiles.meeting]
            beam_size = 10
            anti_loop = true
            """,
        )
        data = project_config.load_config(p)
        flat, _ = project_config.apply_profile(data, profile="meeting")
        assert flat["beam_size"] == 10  # overridden
        assert flat["anti_loop"] is True

    def test_profile_inherits_unset_keys(self, tmp_path):
        p = _write(
            tmp_path,
            "c.toml",
            """
            beam_size = 5
            language = "it"

            [profiles.meeting]
            anti_loop = true
            """,
        )
        data = project_config.load_config(p)
        flat, _ = project_config.apply_profile(data, profile="meeting")
        assert flat == {"beam_size": 5, "language": "it", "anti_loop": True}

    def test_unknown_profile_exits(self, tmp_path):
        p = _write(
            tmp_path,
            "c.toml",
            """
            [profiles.meeting]
            anti_loop = true
            """,
        )
        data = project_config.load_config(p)
        with pytest.raises(SystemExit):
            project_config.apply_profile(data, profile="podcast")

    def test_speaker_map_extracted(self, tmp_path):
        p = _write(
            tmp_path,
            "c.toml",
            """
            beam_size = 5

            [speaker_map]
            SPEAKER_00 = "Amedeo"
            SPEAKER_01 = "Marco"
            """,
        )
        data = project_config.load_config(p)
        flat, smap = project_config.apply_profile(data, profile=None)
        assert "speaker_map" not in flat
        assert smap == {"SPEAKER_00": "Amedeo", "SPEAKER_01": "Marco"}

    def test_profile_overrides_speaker_map(self, tmp_path):
        p = _write(
            tmp_path,
            "c.toml",
            """
            [speaker_map]
            SPEAKER_00 = "Default"

            [profiles.alt]

            [profiles.alt.speaker_map]
            SPEAKER_00 = "OverrideName"
            """,
        )
        data = project_config.load_config(p)
        _, smap = project_config.apply_profile(data, profile="alt")
        assert smap == {"SPEAKER_00": "OverrideName"}


class TestLoadConfigErrors:
    def test_invalid_toml_exits(self, tmp_path):
        p = _write(tmp_path, "broken.toml", "not = valid = toml")
        with pytest.raises(SystemExit):
            project_config.load_config(p)


class TestResolve:
    def test_no_config_returns_empty(self, tmp_path):
        elsewhere = tmp_path / "x"
        elsewhere.mkdir()
        flat, smap, path = project_config.resolve(explicit=None, profile=None, cwd=elsewhere)
        assert flat == {} and smap == {} and path is None

    def test_full_round_trip(self, tmp_path):
        _write(
            tmp_path,
            "transcript-video.toml",
            """
            beam_size = 7
            tags = ["alpha", "beta"]

            [speaker_map]
            SPEAKER_00 = "Anna"

            [profiles.podcast]
            beam_size = 12
            """,
        )
        flat, smap, path = project_config.resolve(
            explicit=None, profile="podcast", cwd=tmp_path
        )
        assert flat["beam_size"] == 12
        assert flat["tags"] == ["alpha", "beta"]
        assert smap == {"SPEAKER_00": "Anna"}
        assert path is not None
