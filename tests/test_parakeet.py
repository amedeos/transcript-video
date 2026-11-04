"""Tests for the Parakeet engine module, with onnx-asr and huggingface_hub faked."""

from __future__ import annotations

import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from transcript_video import parakeet


class TestNames:
    @pytest.mark.parametrize("value", ["parakeet", "Parakeet", "parakeet-tdt-0.6b-v3", " PARAKEET "])
    def test_parakeet_names(self, value):
        assert parakeet.is_parakeet(value)

    @pytest.mark.parametrize("value", [None, "", "auto", "large-v3", "parakeet-ctc"])
    def test_other_names(self, value):
        assert not parakeet.is_parakeet(value)

    @pytest.mark.parametrize(
        "compute_type, weights",
        [("float32", "float32"), ("int8", "int8"), ("int8_float32", "int8"), ("INT8", "int8"), (None, "float32")],
    )
    def test_weights_for(self, compute_type, weights):
        assert parakeet.weights_for(compute_type) == weights

    def test_float32_files_include_the_external_data(self):
        files = parakeet.model_files("float32")
        assert "encoder-model.onnx.data" in files
        assert {"config.json", "vocab.txt"} <= set(files)
        assert not any(".int8." in f for f in files)

    def test_int8_files(self):
        assert parakeet.model_files("int8") == (
            "config.json", "vocab.txt", "encoder-model.int8.onnx", "decoder_joint-model.int8.onnx",
        )

    def test_folders_sit_next_to_the_hub_cache_and_carry_the_revision(self, fake_hub):
        root = fake_hub.cache.parent / "transcript-video"
        assert parakeet.model_dir() == root / f"{parakeet.MODEL_NAME}-{parakeet.HF_REVISION[:12]}"
        assert parakeet.vad_dir() == root / f"silero-vad-{parakeet.VAD_REVISION[:12]}"


class TestWords:
    def test_tokens_grouped_into_words_with_file_times(self):
        words = parakeet.words_from_tokens(
            [" Ciao", ",", " come", " st", "ai", "?"],
            [0.0, 0.16, 0.48, 0.96, 1.04, 1.12],
            [-0.1, -0.3, -0.2, -0.4, -0.6, -0.1],
            offset=10.0,
            chunk_end=12.0,
        )
        assert [w["word"] for w in words] == ["Ciao,", "come", "stai?"]
        assert [w["start"] for w in words] == pytest.approx([10.0, 10.48, 10.96])
        assert words[0]["end"] == pytest.approx(10.48)  # next word's start
        assert words[2]["end"] == pytest.approx(11.12 + parakeet.WORD_TAIL_S)  # tail cap
        assert words[0]["_lp"] == pytest.approx(-0.4)
        assert words[0]["_n"] == 2

    def test_end_is_capped_before_a_pause(self):
        words = parakeet.words_from_tokens([" sì", " no"], [0.0, 3.0], [-0.1, -0.1], offset=0.0, chunk_end=4.0)
        assert words[0]["end"] == pytest.approx(parakeet.WORD_TAIL_S)

    def test_last_word_ends_no_later_than_the_chunk(self):
        words = parakeet.words_from_tokens([" ok"], [0.9], [-0.1], offset=0.0, chunk_end=1.0)
        assert words[0]["end"] == pytest.approx(1.0)

    def test_minimum_length_never_passes_the_chunk_end(self):
        words = parakeet.words_from_tokens([" sì"], [0.95], [-0.1], offset=0.0, chunk_end=1.0)
        assert words[0]["end"] == pytest.approx(1.0)  # not 0.95 + MIN_WORD_S

    def test_two_words_on_one_frame_keep_a_minimum_length(self):
        words = parakeet.words_from_tokens([" a", " b"], [0.4, 0.4], [-0.1, -0.1], offset=0.0, chunk_end=2.0)
        assert words[0]["end"] == pytest.approx(0.4 + parakeet.MIN_WORD_S)

    def test_first_token_without_space_opens_a_word(self):
        words = parakeet.words_from_tokens(["Ok", "."], [0.0, 0.08], None, offset=0.0, chunk_end=1.0)
        assert [w["word"] for w in words] == ["Ok."]
        assert words[0]["_n"] == 0  # no logprobs

    def test_no_tokens(self):
        assert parakeet.words_from_tokens([], [], [], offset=0.0, chunk_end=1.0) == []
        assert parakeet.words_from_tokens(None, None, None, offset=0.0, chunk_end=1.0) == []


class TestSegments:
    def _words(self, spec):
        """spec: (text, start, token logprobs)"""
        out = []
        for i, (text, start, lps) in enumerate(spec):
            end = spec[i + 1][1] if i + 1 < len(spec) else start + 0.3
            out.append({"word": text, "start": start, "end": end, "_lp": sum(lps), "_n": len(lps)})
        return out

    def test_split_after_sentence_end(self):
        segs = parakeet.split_segments(self._words([
            ("Ciao.", 0.0, [-0.2]), ("Come", 0.5, [-0.1]), ("stai?", 0.8, [-0.3, -0.1]), ("Bene", 1.2, [-0.5]),
        ]))
        assert [s["text"] for s in segs] == ["Ciao.", "Come stai?", "Bene"]
        assert segs[1]["start"] == 0.5
        assert segs[1]["end"] == pytest.approx(1.2)
        assert segs[1]["token_logprob"] == pytest.approx(-0.5 / 3, abs=1e-4)
        assert segs[1]["words"][1] == {"word": "stai?", "start": 0.8, "end": 1.2, "logprob": -0.2}

    def test_sentence_end_before_a_closing_quote_or_bracket(self):
        segs = parakeet.split_segments(self._words([
            ("«Ciao.»", 0.0, [-0.1]), ("(Fine.)", 0.5, [-0.1]), ("Poi", 1.0, [-0.1]),
        ]))
        assert [s["text"] for s in segs] == ["«Ciao.»", "(Fine.)", "Poi"]

    def test_split_at_a_long_pause_mid_sentence(self):
        words = self._words([("e", 0.0, [-0.1]), ("poi", 0.3, [-0.1])])
        words[0]["end"] = 0.3
        words[1]["start"], words[1]["end"] = 0.3 + parakeet.SENTENCE_GAP_S, 2.0
        assert [s["text"] for s in parakeet.split_segments(words)] == ["e", "poi"]

    def test_segments_never_carry_whisperx_confidence_fields(self):
        seg = parakeet.split_segments(self._words([("Sì.", 0.0, [-0.1])]))[0]
        assert "avg_logprob" not in seg and "no_speech_prob" not in seg
        assert "score" not in seg["words"][0]

    def test_word_without_logprobs(self):
        words = self._words([("x", 0.0, [])])
        seg = parakeet.split_segments(words)[0]
        assert seg["token_logprob"] is None
        assert seg["words"][0]["logprob"] is None


# --- model files -------------------------------------------------------------------------


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """A fake huggingface_hub whose cache is tmp_path/hub; records downloads."""
    cache = tmp_path / "hf" / "hub"
    cache.mkdir(parents=True)
    downloads: list[str] = []
    revisions = {parakeet.HF_REPO: parakeet.HF_REVISION, parakeet.VAD_REPO: parakeet.VAD_REVISION}
    module = types.ModuleType("huggingface_hub")

    def blob(repo, name):
        return cache / repo.replace("/", "--") / name

    def hf_hub_download(repo, name, *, revision):
        assert revision == revisions[repo]
        path = blob(repo, name)
        if not path.exists():
            downloads.append(f"{repo}/{name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"weights of {name}")
        return str(path)

    def try_to_load_from_cache(repo, name, *, revision):
        assert revision == revisions[repo]
        path = blob(repo, name)
        return str(path) if path.exists() else None

    module.hf_hub_download = hf_hub_download
    module.try_to_load_from_cache = try_to_load_from_cache
    module.constants = types.SimpleNamespace(HF_HUB_CACHE=str(cache))
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return types.SimpleNamespace(cache=cache, downloads=downloads, blob=blob)


def _all_files(compute_type):
    return [f"{parakeet.HF_REPO}/{n}" for n in parakeet.model_files(compute_type)] + [
        f"{parakeet.VAD_REPO}/{n}" for n in parakeet.VAD_FILES
    ]


class TestModelFiles:
    def test_files_are_hard_linked_from_the_cache(self, fake_hub):
        folder, vad_folder = parakeet.prepare_files("float32")
        for name in parakeet.model_files("float32"):
            assert (folder / name).stat().st_ino == fake_hub.blob(parakeet.HF_REPO, name).stat().st_ino
        for name in parakeet.VAD_FILES:
            assert (vad_folder / name).stat().st_ino == fake_hub.blob(parakeet.VAD_REPO, name).stat().st_ino
        assert sorted(fake_hub.downloads) == sorted(_all_files("float32"))
        assert not list(folder.glob("*.partial")) and not list(folder.glob(".*.partial"))

    def test_copy_when_linking_fails(self, fake_hub, monkeypatch):
        def no_link(src, dst):
            raise OSError("cross-device link")

        monkeypatch.setattr(parakeet.os, "link", no_link)
        folder, _ = parakeet.prepare_files("int8")
        name = "encoder-model.int8.onnx"
        assert (folder / name).read_text() == f"weights of {name}"
        assert (folder / name).stat().st_ino != fake_hub.blob(parakeet.HF_REPO, name).stat().st_ino
        assert not list(folder.glob(".*.partial"))

    def test_present_files_are_not_fetched_again(self, fake_hub):
        parakeet.prepare_files("int8")
        fake_hub.downloads.clear()
        fake_hub.blob(parakeet.HF_REPO, "vocab.txt").unlink()  # even if the cache lost it
        parakeet.prepare_files("int8")
        assert fake_hub.downloads == []

    def test_missing_files_cover_the_model_and_the_vad(self, fake_hub):
        assert parakeet.missing_files("int8") == _all_files("int8")
        blob = fake_hub.blob(parakeet.HF_REPO, "config.json")  # in the HF cache
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_text("{}")
        parakeet.vad_dir().mkdir(parents=True)
        (parakeet.vad_dir() / "silero_vad.onnx").write_text("")  # in our folder
        assert parakeet.missing_files("int8") == [
            f"{parakeet.HF_REPO}/vocab.txt",
            f"{parakeet.HF_REPO}/encoder-model.int8.onnx",
            f"{parakeet.HF_REPO}/decoder_joint-model.int8.onnx",
            f"{parakeet.VAD_REPO}/config.json",
        ]


# --- transcribe ----------------------------------------------------------------------------


@dataclass
class _Chunk:
    start: float
    end: float
    text: str
    timestamps: list[float]
    tokens: list[str]
    logprobs: list[float]


class _FakeRecognizer:
    def __init__(self, chunks, seen):
        self._chunks, self._seen = chunks, seen

    def with_timestamps(self):
        return self

    def recognize(self, waveform, *, sample_rate):
        self._seen["recognize"] = (waveform.dtype.name, sample_rate)
        return iter(self._chunks)


class _FakeAsr:
    def __init__(self, chunks, seen):
        self._chunks, self._seen = chunks, seen

    def with_vad(self, vad, **options):
        self._seen["vad"] = vad
        self._seen["vad_options"] = options
        return _FakeRecognizer(self._chunks, self._seen)


class TestTranscribe:
    def _model(self, chunks):
        seen: dict = {}
        return parakeet.ParakeetModel(_FakeAsr(chunks, seen), "VAD"), seen

    def test_result_shape_and_vad_options(self):
        np = pytest.importorskip("numpy")
        model, seen = self._model([
            _Chunk(1.0, 3.0, "Ciao. Sì", [0.0, 0.08, 0.5], [" Ciao", ".", " Sì"], [-0.1, -0.1, -0.2]),
            _Chunk(5.0, 6.0, "ok", [0.1], [" ok"], [-0.3]),
        ])
        result = model.transcribe(np.zeros(16000 * 7, dtype=np.float64), batch_size=4)
        assert result["language"] is None
        assert [s["text"] for s in result["segments"]] == ["Ciao.", "Sì", "ok"]
        assert result["segments"][2]["start"] == pytest.approx(5.1)
        assert seen["vad"] == "VAD"
        assert seen["vad_options"] == {
            "batch_size": 4,
            "max_speech_duration_s": parakeet.MAX_CHUNK_S,
            "min_silence_duration_ms": parakeet.MIN_SILENCE_MS,
            "speech_pad_ms": parakeet.SPEECH_PAD_MS,
        }
        assert seen["recognize"] == ("float32", parakeet.SAMPLE_RATE)

    def test_a_cut_is_logged_and_the_overlap_word_kept_once(self, caplog):
        """onnx-asr's cut of continuous speech: (0, 29.976) then (29.916, ...), 60 ms of overlap."""
        np = pytest.importorskip("numpy")
        model, _ = self._model([
            _Chunk(0.0, 29.976, "a b", [0.0, 29.93], [" a", " b"], [-0.1, -0.2]),
            _Chunk(29.916, 34.0, "b c", [0.02, 1.0], [" b", " c"], [-0.1, -0.1]),
        ])
        with caplog.at_level(logging.WARNING, logger="transcript_video.parakeet"):
            result = model.transcribe(np.zeros(16000, dtype=np.float32), batch_size=4)
        words = [w["word"] for s in result["segments"] for w in s["words"]]
        assert words == ["a", "b", "c"]
        assert "1 cut(s) at 30 s" in caplog.text
        assert "29.9s" in caplog.text

    def test_chunks_split_at_a_pause_are_not_reported_as_cuts(self, caplog):
        np = pytest.importorskip("numpy")
        model, _ = self._model([
            _Chunk(0.0, 29.9994, "a", [0.0], [" a"], [-0.1]),  # closed normally, just under 30 s
            _Chunk(31.0, 33.0, "b", [0.0], [" b"], [-0.1]),
        ])
        with caplog.at_level(logging.WARNING, logger="transcript_video.parakeet"):
            model.transcribe(np.zeros(16000, dtype=np.float32), batch_size=4)
        assert "cut(s)" not in caplog.text

    def test_load_model_wires_threads_weights_and_folder(self, monkeypatch, tmp_path):
        calls: dict = {}
        onnx_asr = types.ModuleType("onnx_asr")
        onnx_asr.load_model = lambda name, path, **kw: calls.__setitem__("model", (name, Path(path), kw))
        onnx_asr.load_vad = lambda name, path, **kw: calls.__setitem__("vad", (name, Path(path), kw))
        ort = types.ModuleType("onnxruntime")

        class SessionOptions:
            intra_op_num_threads = 0
            inter_op_num_threads = 0

        ort.SessionOptions = SessionOptions
        monkeypatch.setitem(sys.modules, "onnx_asr", onnx_asr)
        monkeypatch.setitem(sys.modules, "onnxruntime", ort)
        monkeypatch.setattr(parakeet, "prepare_files", lambda compute_type: (tmp_path, tmp_path / "vad"))

        model = parakeet.load_model("int8_float32", threads=6)
        assert isinstance(model, parakeet.ParakeetModel)
        name, path, kw = calls["model"]
        assert (name, path, kw["quantization"]) == (parakeet.ONNX_ASR_MODEL, tmp_path, "int8")
        assert kw["sess_options"].intra_op_num_threads == 6
        assert kw["sess_options"].inter_op_num_threads == 1
        assert kw["providers"] == ["CPUExecutionProvider"]
        vad_name, vad_path, vad_kw = calls["vad"]
        assert (vad_name, vad_path) == ("silero", tmp_path / "vad")
        assert vad_kw["sess_options"] is kw["sess_options"]
        assert vad_kw["providers"] == ["CPUExecutionProvider"]

        parakeet.load_model("float32")
        assert calls["model"][2]["quantization"] is None  # fp32 weights
        assert calls["model"][2]["sess_options"].intra_op_num_threads == 0  # ORT default


class TestMergeOverlaps:
    def _w(self, word, start, end):
        return {"word": word, "start": start, "end": end, "_lp": 0.0, "_n": 0}

    def test_no_overlap_changes_nothing(self):
        chunks = [(0.0, 5.0, [self._w("a", 1.0, 4.9)]), (6.0, 8.0, [self._w("b", 6.1, 7.0)])]
        assert parakeet.merge_overlaps(chunks) == []
        assert [w["end"] for _, _, ws in chunks for w in ws] == [4.9, 7.0]

    def test_small_overlap_at_a_short_pause_is_deduplicated_but_not_a_cut(self):
        chunks = [
            (0.0, 10.03, [self._w("a", 9.0, 10.03)]),
            (10.01, 12.0, [self._w("b", 10.05, 11.0)]),  # 20 ms overlap: pause of 40 ms
        ]
        assert parakeet.merge_overlaps(chunks) == []
        assert chunks[0][2][0]["end"] == pytest.approx(10.02)  # capped at the midpoint
        assert [w["word"] for w in chunks[1][2]] == ["b"]
