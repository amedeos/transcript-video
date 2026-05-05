"""Tests for per-speaker statistics."""

from __future__ import annotations

import pytest

from transcript_video.stats import compute_speaker_stats, count_unique_speakers


class TestCountUniqueSpeakers:
    def test_basic(self):
        segs = [{"speaker": "A"}, {"speaker": "B"}, {"speaker": "A"}]
        assert count_unique_speakers(segs) == 2

    def test_skips_missing(self):
        segs = [{"speaker": "A"}, {}, {"speaker": None}]
        assert count_unique_speakers(segs) == 1

    def test_empty(self):
        assert count_unique_speakers([]) == 0


class TestComputeSpeakerStats:
    def test_durations_and_percentages(self):
        segs = [
            {"speaker": "A", "start": 0.0, "end": 30.0, "text": "Hello"},
            {"speaker": "B", "start": 30.0, "end": 40.0, "text": "World"},
            {"speaker": "A", "start": 40.0, "end": 70.0, "text": "Again"},
        ]
        stats = compute_speaker_stats(segs)
        assert stats["A"]["duration_seconds"] == 60.0
        assert stats["B"]["duration_seconds"] == 10.0
        # Total diarized time = 70s; A = 60/70 = 85.7%, B = 10/70 = 14.3%.
        assert stats["A"]["percentage"] == 85.7
        assert stats["B"]["percentage"] == 14.3

    def test_turns_count_consecutive_runs(self):
        # A speaks twice in a row → 1 turn; then B; then A again → 2 turns total for A.
        segs = [
            {"speaker": "A", "start": 0.0, "end": 5.0, "text": "x"},
            {"speaker": "A", "start": 5.0, "end": 10.0, "text": "x"},
            {"speaker": "B", "start": 10.0, "end": 15.0, "text": "y"},
            {"speaker": "A", "start": 15.0, "end": 20.0, "text": "x"},
        ]
        stats = compute_speaker_stats(segs)
        assert stats["A"]["num_turns"] == 2
        assert stats["B"]["num_turns"] == 1

    def test_first_text_truncation(self):
        long = "Allora, oggi parliamo della migrazione da SDN a OVN-Kubernetes con tutti i dettagli del caso."
        segs = [{"speaker": "A", "start": 0.0, "end": 5.0, "text": long}]
        stats = compute_speaker_stats(segs)
        assert stats["A"]["first_text"].endswith("...")
        assert len(stats["A"]["first_text"]) <= 84  # 80 + "..."

    def test_first_text_short_no_ellipsis(self):
        segs = [{"speaker": "A", "start": 0.0, "end": 5.0, "text": "Short."}]
        stats = compute_speaker_stats(segs)
        assert stats["A"]["first_text"] == "Short."

    def test_first_appearance_order_preserved(self):
        segs = [
            {"speaker": "C", "start": 0.0, "end": 1.0, "text": "x"},
            {"speaker": "A", "start": 1.0, "end": 2.0, "text": "x"},
            {"speaker": "B", "start": 2.0, "end": 3.0, "text": "x"},
        ]
        stats = compute_speaker_stats(segs)
        assert list(stats.keys()) == ["C", "A", "B"]

    def test_segments_without_speaker_skipped_and_break_turn_continuity(self):
        segs = [
            {"speaker": "A", "start": 0.0, "end": 5.0, "text": "x"},
            {"start": 5.0, "end": 6.0, "text": "no-speaker"},  # gap in attribution
            {"speaker": "A", "start": 6.0, "end": 10.0, "text": "x"},
        ]
        stats = compute_speaker_stats(segs)
        # The unattributed segment resets last_speaker, so A's two runs count as 2 turns.
        assert stats["A"]["num_turns"] == 2
        assert stats["A"]["duration_seconds"] == pytest.approx(9.0)

    def test_empty_segments(self):
        assert compute_speaker_stats([]) == {}

    def test_zero_duration_does_not_crash(self):
        segs = [
            {"speaker": "A", "start": 5.0, "end": 5.0, "text": "x"},
            {"speaker": "B", "start": 5.0, "end": 5.0, "text": "y"},
        ]
        stats = compute_speaker_stats(segs)
        assert stats["A"]["duration_seconds"] == 0.0
        assert stats["A"]["percentage"] == 0.0
