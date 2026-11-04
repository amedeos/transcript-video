"""Tests for diarize._share_embedding_backbone (one network pass per window).

The fallback tests need nothing. Everything that runs the shared path needs
torch (the self-check runs it), and the equivalence test needs pyannote
(random weights, no download): those are marked ``integration`` and skipped
when the library is missing.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from transcript_video import diarize as diarize_mod

LOGGER = "transcript_video.diarize"


class _FakeNetwork:
    """forward() = forward_embedding(forward_frames()), counting frame passes."""

    def __init__(self):
        self.frame_calls = 0
        self.frame_rows = 0
        self.fail = False

    def forward_frames(self, waveforms):
        if self.fail:
            raise RuntimeError("boom")
        self.frame_calls += 1
        self.frame_rows += len(waveforms)
        return waveforms * 2.0  # (batch, 1, samples)

    def forward_embedding(self, frames, weights=None):
        # (batch, 1, samples) x (batch, frames) -> (batch, 2): mean audio, masked mean
        import torch

        pooled = frames.mean(dim=(1, 2))
        if weights is None:
            weights = torch.ones(len(frames), 1)
        weighted = (weights * pooled[:, None]).sum(dim=1) / (weights.sum(dim=1) + 1e-8)
        return torch.stack([pooled, weighted], dim=1)

    def __call__(self, waveforms, weights=None):
        return self.forward_embedding(self.forward_frames(waveforms), weights=weights)


class _FakeEmbedding:
    """Mimics PyannoteAudioPretrainedSpeakerEmbedding.__call__."""

    def __init__(self, network):
        self.model_ = network
        self.device = "cpu"

    def __call__(self, waveforms, masks=None):
        if masks is None:
            return self.model_(waveforms).cpu().numpy()
        return self.model_(waveforms, weights=masks).cpu().numpy()


class _SlottedEmbedding:
    """A wrapper whose class can't be swapped (``__slots__`` layout)."""

    __slots__ = ("model_", "device")

    def __init__(self, network):
        self.model_ = network
        self.device = "cpu"

    def __call__(self, waveforms, masks=None):
        return None


def _fake_whisperx_pipeline(embedding, *, batch=32):
    return SimpleNamespace(model=SimpleNamespace(_embedding=embedding, embedding_batch_size=batch))


def _window_rows(torch, windows=4, speakers=3, samples=1600, frames=50):
    """Rows as pyannote builds them: each window's audio repeated once per local speaker."""
    torch.manual_seed(0)
    audio = [torch.randn(1, 1, samples) for _ in range(windows)]
    waveforms = torch.vstack([w for w in audio for _ in range(speakers)])
    masks = (torch.rand(windows * speakers, frames) > 0.5).float()
    masks[2] = 0.0  # a local speaker that never talks
    return waveforms, masks


class TestStaysStock:
    """Cases that must leave pyannote untouched; no torch needed."""

    def test_pipeline_without_embedding_model(self, caplog):
        caplog.set_level(logging.INFO, logger=LOGGER)
        assert diarize_mod._share_embedding_backbone(SimpleNamespace()) is False
        assert "no pyannote embedding model" in caplog.text

    def test_network_without_split_forward(self, caplog):
        caplog.set_level(logging.INFO, logger=LOGGER)
        embedding = SimpleNamespace(model_=SimpleNamespace(), device="cpu")
        pipeline = _fake_whisperx_pipeline(embedding)
        assert diarize_mod._share_embedding_backbone(pipeline) is False
        assert type(embedding) is SimpleNamespace
        assert "no forward_frames/forward_embedding split" in caplog.text

    def test_class_swap_error_keeps_stock(self, caplog):
        embedding = _SlottedEmbedding(_FakeNetwork())
        assert diarize_mod._share_embedding_backbone(_fake_whisperx_pipeline(embedding)) is False
        assert type(embedding) is _SlottedEmbedding
        assert "shared embedding network is not usable" in caplog.text

    def test_diarize_and_assign_survives_an_unexpected_error(self, monkeypatch, caplog):
        def explode(_pipeline):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(diarize_mod, "_share_embedding_backbone", explode)
        calls = _install_fake_whisperx(monkeypatch, _FakeEmbedding(_FakeNetwork()))
        out = diarize_mod.diarize_and_assign({"segments": []}, "audio", hf_token="t", device="cpu")
        assert out == {"segments": [], "assigned": True}
        assert calls == ["diarize"]
        assert "stock pyannote embeddings (RuntimeError: unexpected)" in caplog.text


def _install_fake_whisperx(monkeypatch, embedding):
    """Fake whisperX diarization API; records whether sharing was on when diarizing."""
    calls: list[str] = []

    class FakeDiarizationPipeline:
        def __init__(self, token=None, device=None, model_name=None):
            self.model = SimpleNamespace(_embedding=embedding, embedding_batch_size=32)

        def __call__(self, audio, **kwargs):
            shared = getattr(type(self.model._embedding), "_shares_backbone", False)
            calls.append("diarize-shared" if shared else "diarize")
            return "diarization"

    def assign(diarization, aligned):
        return {**aligned, "assigned": True}

    monkeypatch.setattr(diarize_mod, "_resolve_diarization_api", lambda: (FakeDiarizationPipeline, assign))
    return calls


@pytest.mark.integration
class TestShared:
    @pytest.fixture(autouse=True)
    def _torch(self):
        self.torch = pytest.importorskip("torch")

    def test_sharing_turns_on_and_keeps_the_batch(self, caplog):
        caplog.set_level(logging.INFO, logger=LOGGER)
        embedding = _FakeEmbedding(_FakeNetwork())
        pipeline = _fake_whisperx_pipeline(embedding, batch=32)
        assert diarize_mod._share_embedding_backbone(pipeline) is True
        assert isinstance(embedding, _FakeEmbedding)  # subclass: properties keep working
        assert pipeline.model.embedding_batch_size == 32  # same rows per call as stock
        assert "run once per window" in caplog.text

    def test_applying_twice_is_a_no_op(self):
        embedding = _FakeEmbedding(_FakeNetwork())
        pipeline = _fake_whisperx_pipeline(embedding)
        diarize_mod._share_embedding_backbone(pipeline)
        cls = type(embedding)
        assert diarize_mod._share_embedding_backbone(pipeline) is True
        assert type(embedding) is cls

    def test_self_check_mismatch_keeps_stock(self, caplog):
        class SkewedNetwork(_FakeNetwork):
            def __call__(self, waveforms, weights=None):  # forward() adds a step the halves don't
                return super().__call__(waveforms, weights=weights) + 1.0

        embedding = _FakeEmbedding(SkewedNetwork())
        assert diarize_mod._share_embedding_backbone(_fake_whisperx_pipeline(embedding)) is False
        assert type(embedding) is _FakeEmbedding
        assert "embeddings differ from stock" in caplog.text

    def test_same_embeddings_with_one_network_pass_per_window(self):
        torch = self.torch
        waveforms, masks = _window_rows(torch)
        stock = _FakeEmbedding(_FakeNetwork())
        expected = [stock(waveforms[:7], masks=masks[:7]), stock(waveforms[7:], masks=masks[7:])]

        network = _FakeNetwork()
        shared = _FakeEmbedding(network)
        diarize_mod._share_embedding_backbone(_fake_whisperx_pipeline(shared))
        network.frame_calls = network.frame_rows = 0  # forget the self-check
        # a batch boundary inside window 2 (rows 6-8): windows 0, 1, 2 | 2, 3
        got = [shared(waveforms[:7], masks=masks[:7]), shared(waveforms[7:], masks=masks[7:])]

        for e, g in zip(expected, got, strict=True):
            assert (e == g).all()
        assert network.frame_calls == 2
        assert network.frame_rows == 5  # stock: 12

    def test_no_masks_uses_the_stock_call(self):
        network = _FakeNetwork()
        shared = _FakeEmbedding(network)
        diarize_mod._share_embedding_backbone(_fake_whisperx_pipeline(shared))
        network.frame_rows = 0
        waveforms, _ = _window_rows(self.torch, windows=2, speakers=2)
        assert shared(waveforms).shape == (4, 2)
        assert network.frame_rows == 4  # no grouping without masks

    def test_error_during_the_run_falls_back_to_stock(self, caplog):
        network = _FakeNetwork()
        shared = _FakeEmbedding(network)
        diarize_mod._share_embedding_backbone(_fake_whisperx_pipeline(shared))
        waveforms, masks = _window_rows(self.torch)
        network.fail = True
        with pytest.raises(RuntimeError):  # stock fails too with this network...
            shared(waveforms, masks=masks)
        assert type(shared) is _FakeEmbedding  # ...but the stock class is back
        assert "falling back to stock pyannote embeddings" in caplog.text

    def test_diarize_and_assign_shares_before_diarizing(self, monkeypatch):
        calls = _install_fake_whisperx(monkeypatch, _FakeEmbedding(_FakeNetwork()))
        out = diarize_mod.diarize_and_assign({"segments": []}, "audio", hf_token="t", device="cpu")
        assert out["assigned"] is True
        assert calls == ["diarize-shared"]


@pytest.mark.integration
def test_real_wespeaker_resnet_gives_the_same_embeddings():
    """Random-weight WeSpeaker ResNet34 through pyannote's own wrapper (no download).

    Same row batches as stock pyannote, as in the pipeline. Exact equality
    held on the machine this was written on; the tolerance covers backends
    whose convolution kernels round differently for another batch shape.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("pyannote.audio")
    import numpy as np
    from pyannote.audio.models.embedding.wespeaker import WeSpeakerResNet34
    from pyannote.audio.pipelines.speaker_verification import (
        PyannoteAudioPretrainedSpeakerEmbedding,
    )

    torch.manual_seed(0)
    network = WeSpeakerResNet34()
    network.eval()
    embedding = PyannoteAudioPretrainedSpeakerEmbedding(network)
    waveforms, masks = _window_rows(torch, windows=4, speakers=3, samples=32000, frames=200)
    waveforms = waveforms * 0.1

    def run():
        return np.vstack([embedding(waveforms[:7], masks=masks[:7]), embedding(waveforms[7:], masks=masks[7:])])

    stock = run()
    assert diarize_mod._share_embedding_backbone(_fake_whisperx_pipeline(embedding)) is True
    passes = []
    forward_frames = network.forward_frames
    network.forward_frames = lambda w: passes.append(len(w)) or forward_frames(w)
    shared = run()

    assert np.array_equal(np.isnan(stock), np.isnan(shared))
    assert np.allclose(stock, shared, rtol=0.0, atol=1e-6, equal_nan=True)
    assert sum(passes) == 5  # windows through the network; stock pyannote runs 12
