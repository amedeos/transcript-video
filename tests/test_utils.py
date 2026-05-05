"""Tests for timestamp formatters and small file helpers."""

from __future__ import annotations

import warnings

import pytest

from transcript_video.utils import (
    format_timestamp_hms,
    format_timestamp_short,
    format_timestamp_srt,
    read_text_file,
    silence_known_noisy_warnings,
)


class TestFormatTimestampSrt:
    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0.0, "00:00:00,000"),
            (0.5, "00:00:00,500"),
            (1.0, "00:00:01,000"),
            (60.0, "00:01:00,000"),
            (3600.0, "01:00:00,000"),
            (3661.123, "01:01:01,123"),
            (3661.999, "01:01:01,999"),
        ],
    )
    def test_typical(self, seconds, expected):
        assert format_timestamp_srt(seconds) == expected

    def test_negative_clamps_to_zero(self):
        assert format_timestamp_srt(-1.0) == "00:00:00,000"

    def test_millis_rollover(self):
        # 0.9999s rounds to 1000 ms, which must roll into the next second.
        assert format_timestamp_srt(0.9999) == "00:00:01,000"


class TestFormatTimestampHms:
    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0.0, "00:00:00"),
            (12.0, "00:00:12"),
            (105.0, "00:01:45"),
            (3661.4, "01:01:01"),
            (6135.0, "01:42:15"),  # the example from the user spec
        ],
    )
    def test_typical(self, seconds, expected):
        assert format_timestamp_hms(seconds) == expected

    def test_negative_clamps(self):
        assert format_timestamp_hms(-5.0) == "00:00:00"


class TestFormatTimestampShort:
    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0.0, "[00:00]"),
            (75.0, "[01:15]"),
            (3661.0, "[61:01]"),  # hours collapse into minutes for the short form
        ],
    )
    def test_typical(self, seconds, expected):
        assert format_timestamp_short(seconds) == expected


class TestSilenceKnownNoisyWarnings:
    def test_torchcodec_warning_silenced(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.resetwarnings()
            silence_known_noisy_warnings()
            warnings.warn(
                "torchcodec is not installed correctly so built-in audio decoding will fail.",
                UserWarning,
                stacklevel=2,
            )
            assert not captured

    def test_tf32_warning_silenced(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.resetwarnings()
            silence_known_noisy_warnings()
            warnings.warn("TensorFloat-32 (TF32) has been disabled", UserWarning, stacklevel=2)
            assert not captured

    def test_unrelated_warnings_pass_through(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.resetwarnings()
            warnings.simplefilter("always")
            silence_known_noisy_warnings()
            warnings.warn("something else entirely", UserWarning, stacklevel=2)
            assert any("something else" in str(w.message) for w in captured)


class TestReadTextFile:
    def test_reads_and_strips(self, tmp_path):
        p = tmp_path / "hello.txt"
        p.write_text("  hello world  \n", encoding="utf-8")
        assert read_text_file(p, "demo") == "hello world"

    def test_missing_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            read_text_file(tmp_path / "missing.txt", "demo")
