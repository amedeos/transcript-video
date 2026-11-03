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


class TestCliWinsInMutexGroups:
    """Invariant #7 inside mutually exclusive groups: a CLI flag beats the config."""

    @pytest.fixture
    def logging_seen(self, monkeypatch) -> dict:
        seen: dict = {}
        monkeypatch.setattr(
            cli_from_video, "setup_logging",
            lambda quiet, verbose: seen.update(quiet=quiet, verbose=verbose),
        )
        return seen

    def test_cli_prompt_file_beats_config_prompt(self, captured, tmp_path):
        (tmp_path / "transcript-video.toml").write_text('prompt = "from config"\n')
        (tmp_path / "glossary.txt").write_text("from file")
        _run(["--prompt-file", "glossary.txt"])
        assert captured["config"].initial_prompt == "from file"

    def test_cli_prompt_beats_config_no_prompt(self, captured, tmp_path):
        (tmp_path / "transcript-video.toml").write_text("no_prompt = true\n")
        _run(["--prompt", "from cli"])
        assert captured["config"].initial_prompt == "from cli"

    def test_cli_no_hotwords_beats_config_hotwords(self, captured, tmp_path):
        (tmp_path / "transcript-video.toml").write_text('hotwords = "OpenShift"\n')
        _run(["--no-hotwords"])
        assert captured["config"].hotwords == ""

    def test_cli_verbose_beats_config_quiet(self, captured, logging_seen, tmp_path):
        (tmp_path / "transcript-video.toml").write_text("quiet = true\n")
        _run(["-v"])
        assert logging_seen == {"quiet": False, "verbose": True}

    def test_config_alone_still_applies(self, captured, logging_seen, tmp_path):
        (tmp_path / "transcript-video.toml").write_text('prompt = "from config"\nquiet = true\n')
        _run([])
        assert captured["config"].initial_prompt == "from config"
        assert logging_seen["quiet"] is True

    def test_config_setting_two_members_is_rejected(self, captured, tmp_path, capsys):
        (tmp_path / "transcript-video.toml").write_text(
            'prompt = "a"\nprompt_file = "b.txt"\n'
        )
        with pytest.raises(SystemExit):
            _run([])
        assert "mutually exclusive" in capsys.readouterr().err

    def test_false_flag_in_config_is_not_a_conflict(self, captured, tmp_path):
        (tmp_path / "transcript-video.toml").write_text('prompt = "a"\nno_prompt = false\n')
        _run([])
        assert captured["config"].initial_prompt == "a"
