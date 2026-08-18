"""Tests for timestamp formatters and small file helpers."""

from __future__ import annotations

import logging
import warnings

import pytest

from transcript_video.utils import (
    format_timestamp_hms,
    format_timestamp_short,
    format_timestamp_srt,
    format_timestamp_vtt,
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


class TestFormatTimestampVtt:
    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0.0, "00:00:00.000"),
            (0.5, "00:00:00.500"),
            (1.0, "00:00:01.000"),
            (60.0, "00:01:00.000"),
            (3600.0, "01:00:00.000"),
            (3661.123, "01:01:01.123"),
        ],
    )
    def test_typical(self, seconds, expected):
        # WebVTT uses a dot (not SRT's comma) for the millisecond separator.
        assert format_timestamp_vtt(seconds) == expected

    def test_negative_clamps_to_zero(self):
        assert format_timestamp_vtt(-1.0) == "00:00:00.000"

    def test_millis_rollover(self):
        # 0.9999s rounds to 1000 ms, which must roll into the next second.
        assert format_timestamp_vtt(0.9999) == "00:00:01.000"


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
    # Real warning text from pyannote.audio: triple-quoted f-string that begins
    # with a newline. The filter must handle this multiline shape, otherwise
    # `.` in the default-flag regex doesn't span the leading "\n" and the
    # filter silently does nothing.
    REAL_TORCHCODEC_MESSAGE = (
        "\ntorchcodec is not installed correctly so built-in audio decoding "
        "will fail. Solutions are:\n* use audio preloaded in-memory ..."
    )
    REAL_TF32_MESSAGE = (
        "TensorFloat-32 (TF32) has been disabled as it might lead to "
        "reproducibility issues and lower accuracy.\nIt can be re-enabled "
        "by calling\n   >>> import torch"
    )
    # Lightning emits this exact prefix from
    # `lightning.pytorch.utilities.migration.utils._pl_migrate_checkpoint`
    # via `logging.info`, not `warnings.warn`. The runtime filter must drop it.
    REAL_LIGHTNING_MESSAGE = (
        "Lightning automatically upgraded your loaded checkpoint from v1.5.4 "
        "to v2.6.1. To apply the upgrade to your files permanently, run "
        "`python -m lightning.pytorch.utilities.upgrade_checkpoint ...`"
    )

    def test_torchcodec_warning_silenced(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.resetwarnings()
            silence_known_noisy_warnings()
            warnings.warn(self.REAL_TORCHCODEC_MESSAGE, UserWarning, stacklevel=2)
            assert not captured

    def test_tf32_warning_silenced(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.resetwarnings()
            silence_known_noisy_warnings()
            warnings.warn(self.REAL_TF32_MESSAGE, UserWarning, stacklevel=2)
            assert not captured

    def test_unrelated_warnings_pass_through(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.resetwarnings()
            warnings.simplefilter("always")
            silence_known_noisy_warnings()
            warnings.warn("something else entirely", UserWarning, stacklevel=2)
            assert any("something else" in str(w.message) for w in captured)

    @pytest.fixture
    def lightning_logger_clean(self):
        """Restore the Lightning migration logger to its pre-test filter state.

        The runtime filter is global and durable by design: once attached it
        stays attached for the process lifetime, so subsequent runs (real CLI
        invocations) don't re-emit the notice. Tests that assert filter
        behavior must clean up so they don't leak state into sibling tests or
        the unrelated-warnings test.

        The logger level is forced to DEBUG here because the real CLI calls
        ``setup_logging()`` which lowers the root to INFO, but pytest does not,
        so INFO records would be dropped by the logger-level gate before the
        filter is even consulted — masking the behavior under test.
        """
        logger_name = "lightning.pytorch.utilities.migration.utils"
        logger = logging.getLogger(logger_name)
        original_filters = list(logger.filters)
        original_level = logger.level
        original_propagate = logger.propagate
        logger.setLevel(logging.DEBUG)
        yield logger
        logger.filters = original_filters
        logger.level = original_level
        logger.propagate = original_propagate

    def test_lightning_upgrade_message_silenced(self, lightning_logger_clean):
        # The filter drops the record before it reaches handlers, so a
        # capturing handler on the emitting logger sees nothing.
        logger = lightning_logger_clean
        silence_known_noisy_warnings()
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        # Reset filters on the logger to baseline so only the silence filter
        # (added by silence_known_noisy_warnings) is present; the capture
        # handler must still see through if the filter lets records pass.
        capture = _Capture(level=logging.DEBUG)
        logger.addHandler(capture)
        try:
            logger.info(self.REAL_LIGHTNING_MESSAGE)
        finally:
            logger.removeHandler(capture)
        assert not records, (
            "Lightning upgrade notice was not silenced — filter did not drop it"
        )

    def test_lightning_other_messages_pass_through(self, lightning_logger_clean):
        # Sanity: the filter is specific to the upgrade marker, not a blanket
        # mute on the whole logger. Other info records from the same module
        # must still propagate to handlers.
        logger = lightning_logger_clean
        silence_known_noisy_warnings()
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        capture = _Capture(level=logging.DEBUG)
        logger.addHandler(capture)
        try:
            logger.info("some other lightning message")
        finally:
            logger.removeHandler(capture)
        assert any(
            "some other lightning message" in r.getMessage() for r in records
        ), "Filter is too broad — it muted an unrelated Lightning record"

    def test_lightning_filter_idempotent(self, lightning_logger_clean):
        # Calling silence_known_noisy_warnings() repeatedly must not stack
        # duplicate filter instances on the logger. Otherwise each call adds
        # another copy and the idempotency contract documented in utils.py
        # is violated (no functional difference, but a leak across re-imports
        # / repeated CLI invocations in the same process).
        logger = lightning_logger_clean
        silence_known_noisy_warnings()
        silence_known_noisy_warnings()
        silence_known_noisy_warnings()
        # The cached filter instance is the one attached; count occurrences.
        filt = silence_known_noisy_warnings._lightning_filter  # type: ignore[attr-defined]
        count = sum(1 for f in logger.filters if f is filt)
        assert count == 1, (
            f"Expected exactly 1 attached filter instance, found {count} — "
            "silence_known_noisy_warnings is not idempotent"
        )


class TestReadTextFile:
    def test_reads_and_strips(self, tmp_path):
        p = tmp_path / "hello.txt"
        p.write_text("  hello world  \n", encoding="utf-8")
        assert read_text_file(p, "demo") == "hello world"

    def test_missing_file_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            read_text_file(tmp_path / "missing.txt", "demo")
