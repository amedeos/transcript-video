"""CLI → RunConfig wiring for ``transcript-from-video`` (no pipeline run)."""

from __future__ import annotations

from pathlib import Path

import pytest

from transcript_video import cli_from_video


@pytest.fixture
def captured(monkeypatch, tmp_path: Path) -> dict:
    """Run ``main`` up to ``run_pipeline`` and capture the RunConfig it builds."""
    seen: dict = {}
    monkeypatch.setattr(cli_from_video, "run_pipeline", lambda cfg: seen.setdefault("config", cfg))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "talk.wav").write_bytes(b"\x00")
    return seen


def _run(argv: list[str]) -> None:
    cli_from_video.main(["talk.wav", "--no-check", *argv])


class TestBatchSizeFlag:
    def test_unset_means_auto(self, captured):
        _run([])
        assert captured["config"].batch_size is None

    def test_cli_flag(self, captured):
        _run(["--batch-size", "2"])
        assert captured["config"].batch_size == 2

    def test_config_key(self, captured, tmp_path):
        (tmp_path / "transcript-video.toml").write_text("batch_size = 3\n")
        _run([])
        assert captured["config"].batch_size == 3

    def test_cli_wins_over_config(self, captured, tmp_path):
        (tmp_path / "transcript-video.toml").write_text("batch_size = 3\n")
        _run(["--batch-size", "1"])
        assert captured["config"].batch_size == 1

    @pytest.mark.parametrize("value", ["-1", "4.0", "true"])
    def test_invalid_config_value_is_rejected(self, captured, tmp_path, value):
        (tmp_path / "transcript-video.toml").write_text(f"batch_size = {value}\n")
        with pytest.raises(SystemExit):
            _run([])
        assert "config" not in captured

    def test_negative_on_cli_is_rejected(self, captured):
        with pytest.raises(SystemExit):
            _run(["--batch-size", "-1"])


class TestZeroMeansAuto:
    """0 on the CLI resets a config value back to the automatic default."""

    @pytest.mark.parametrize(("flag", "key"), [("--batch-size", "batch_size"), ("--threads", "threads")])
    def test_cli_zero_overrides_config(self, captured, tmp_path, flag, key):
        (tmp_path / "transcript-video.toml").write_text(f"{key} = 2\n")
        _run([flag, "0"])
        assert getattr(captured["config"], key) is None

    def test_config_zero_is_auto(self, captured, tmp_path):
        (tmp_path / "transcript-video.toml").write_text("threads = 0\n")
        _run([])
        assert captured["config"].threads is None


class TestDeviceValue:
    """A config `device` bypasses argparse choices, so main() validates it."""

    def test_config_device_is_lowercased(self, captured, tmp_path):
        (tmp_path / "transcript-video.toml").write_text('device = "CUDA"\n')
        _run([])
        assert captured["config"].device == "cuda"

    @pytest.mark.parametrize("value", ["gpu", "cuda:0"])
    def test_unknown_config_device_is_rejected(self, captured, tmp_path, value):
        (tmp_path / "transcript-video.toml").write_text(f'device = "{value}"\n')
        with pytest.raises(SystemExit):
            _run([])
        assert "config" not in captured

    def test_auto_maps_to_none(self, captured):
        _run(["--device", "auto"])
        assert captured["config"].device is None
