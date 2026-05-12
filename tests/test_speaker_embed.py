"""Integration tests for the per-cluster embedding extraction.

Skipped in CI (which does not install pyannote / torch); run locally with:

    pytest -m integration tests/test_speaker_embed.py

Module-level :func:`pytest.importorskip` makes the file safe to *collect*
even without pyannote — pytest reports the file as skipped and moves on.
A working ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``) is required because
the embedding model is fetched on first run.
"""

from __future__ import annotations

import os
import random
import struct
import wave
from pathlib import Path

import pytest

pyannote_audio = pytest.importorskip("pyannote.audio")

pytestmark = pytest.mark.integration


SAMPLE_RATE = 16000
EMBEDDING_DIM = 256  # pyannote/wespeaker-voxceleb-resnet34-LM output dimension


def _hf_token() -> str | None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(var)
        if v:
            return v
    return None


def _write_synthetic_wav(path: Path, seconds: float, *, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a deterministic pseudo-random WAV — enough to exercise the pipeline.

    The audio has no real speaker content; the resulting embeddings are
    meaningless but the function must still return well-shaped output.
    """
    rng = random.Random(42)
    n_samples = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        frames = b"".join(struct.pack("<h", rng.randint(-16000, 16000)) for _ in range(n_samples))
        w.writeframes(frames)


def test_returns_empty_for_no_speaker_labels(tmp_path):
    from transcript_video.speaker_embed import extract_cluster_embeddings

    audio = tmp_path / "silence.wav"
    _write_synthetic_wav(audio, seconds=2)
    segments = [
        {"start": 0.0, "end": 1.0, "text": "hi"},  # no `speaker`
    ]
    # No speakers → no inference happens → no token needed.
    result = extract_cluster_embeddings(audio, segments, hf_token="not-used")
    assert result == {}


def test_smoke_two_clusters(tmp_path):
    from transcript_video.speaker_embed import (
        DEFAULT_EMBEDDING_MODEL,
        extract_cluster_embeddings,
    )

    token = _hf_token()
    if not token:
        pytest.skip("no HF_TOKEN in env; cannot download embedding model")

    audio = tmp_path / "two_speakers.wav"
    _write_synthetic_wav(audio, seconds=10)

    segments = [
        {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 9.0, "speaker": "SPEAKER_01"},
    ]

    result = extract_cluster_embeddings(audio, segments, hf_token=token)

    assert set(result.keys()) == {"SPEAKER_00", "SPEAKER_01"}
    for _label, info in result.items():
        assert isinstance(info["embedding"], list)
        assert all(isinstance(x, float) for x in info["embedding"])
        assert len(info["embedding"]) == EMBEDDING_DIM
        assert info["n_segments"] == 1
        assert info["duration_s"] == pytest.approx(4.0)
        assert info["embedding_model"] == DEFAULT_EMBEDDING_MODEL


def test_short_segments_filtered_with_fallback(tmp_path):
    """If all segments are below ``min_segment_seconds``, we still embed them
    (fallback) rather than dropping the cluster."""
    from transcript_video.speaker_embed import extract_cluster_embeddings

    token = _hf_token()
    if not token:
        pytest.skip("no HF_TOKEN in env; cannot download embedding model")

    audio = tmp_path / "short.wav"
    _write_synthetic_wav(audio, seconds=3)

    # Both segments are < default 1.5s. Fallback path should kick in.
    segments = [
        {"start": 0.0, "end": 0.8, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 1.7, "speaker": "SPEAKER_00"},
    ]

    result = extract_cluster_embeddings(audio, segments, hf_token=token)
    assert "SPEAKER_00" in result
    assert result["SPEAKER_00"]["n_segments"] == 2
