"""NVIDIA Parakeet TDT 0.6B v3 speech recognition on CPU, through onnx-asr.

The second ASR engine next to whisperX. It runs on ONNX Runtime's CPU
provider only, transcribes 25 European languages, detects the language by
itself (it cannot be told which one), and gives token timestamps (80 ms
steps), so no alignment stage is needed.

:func:`load_model` returns an object whose ``transcribe(audio,
batch_size=...)`` returns a result shaped like whisperX's, ``{"segments":
[...], "language": None}``, with sentence-level segments that already carry
``words``. The call differs: there is no ``language`` argument (Parakeet
can't be forced) and the language is never reported. Measured on a laptop CPU (i7-1280P, 6 threads, fp32): 0.40x the
audio length against ~1.6x for whisperX large-v3.

Segmentation uses onnx-asr's Silero VAD with the configuration that was
measured. Speech separated by pauses shorter than :data:`MIN_SILENCE_MS` is
merged into one chunk, so short utterances get enough context: with short
chunks Parakeet slipped into English or Cyrillic. A chunk is closed at the
last pause before :data:`MAX_CHUNK_S`. Only continuous speech longer than
that, without any pause the VAD can see, is cut where it stands. Those cuts
are logged, and the words in the overlap onnx-asr leaves around them are
kept once (:func:`merge_overlaps`).

Everything runs on ONNX Runtime's CPU provider, even when another one is
installed, and the model files come from pinned HuggingFace revisions.

The heavy imports (onnx_asr, onnxruntime, huggingface_hub, numpy) happen
inside the functions, so importing this module costs nothing.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("transcript_video.parakeet")

MODEL_NAME = "parakeet-tdt-0.6b-v3"
"""Canonical name, recorded as ``parameters.model``."""
MODEL_ALIASES = ("parakeet", MODEL_NAME)
"""``--model`` values that select this engine (case-insensitive)."""
ONNX_ASR_MODEL = "nemo-parakeet-tdt-0.6b-v3"
HF_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
HF_REVISION = "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce"
"""Pinned commit of :data:`HF_REPO` (2026-02-17), the one measured on the laptop."""
VAD_NAME = "silero"
VAD_REPO = "istupakov/silero-vad-onnx"
VAD_REVISION = "b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6"
"""Pinned commit of :data:`VAD_REPO` (2026-03-22)."""
VAD_FILES = ("config.json", "silero_vad.onnx")
CPU_PROVIDERS = ["CPUExecutionProvider"]

SAMPLE_RATE = 16000
MIN_SILENCE_MS = 1000
"""VAD: pauses shorter than this don't split speech."""
MAX_CHUNK_S = 30.0
"""VAD: longest chunk sent to the model."""
SPEECH_PAD_MS = 30
"""VAD: padding around each chunk (onnx-asr's default, passed explicitly)."""
WORD_TAIL_S = 0.5
"""A word ends at the next word's start, but at most this long after its last token starts."""
MIN_WORD_S = 0.08
"""Shortest word duration (one encoder frame)."""
SENTENCE_GAP_S = 1.0
"""A pause this long between two words starts a new segment, even mid-sentence."""
SENTENCE_END = (".", "?", "!", "…")
_CLOSING = "\"'»”’)]}"
"""Closing quotes and brackets that may follow a sentence's final punctuation."""

DOWNLOAD_SIZE = {"float32": "2.5 GB", "int8": "670 MB"}
"""Approximate download of the weight files (encoder + decoder); the rest is a few MB."""
ONNX_ASR_SERIES = "0.12."
"""The onnx-asr releases this module was written against (pyproject pins >=0.12,<0.13)."""
_COMMON_FILES = ("config.json", "vocab.txt")
_MODEL_FILES = {
    "float32": ("encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx"),
    "int8": ("encoder-model.int8.onnx", "decoder_joint-model.int8.onnx"),
}


def is_parakeet(model: str | None) -> bool:
    """Whether a ``--model`` value selects this engine."""
    return (model or "").strip().lower() in MODEL_ALIASES


def weights_for(compute_type: str | None) -> str:
    """ONNX weights for a compute type: ``int8`` for any int8 type, else ``float32``."""
    return "int8" if (compute_type or "").lower().startswith("int8") else "float32"


def resolve_weights(compute_type: str | None) -> tuple[str, str | None]:
    """Parakeet's weights for ``--compute-type``, and a warning when it isn't exact.

    Parakeet ships float32 and int8 weights only. Unset, ``auto``,
    ``default`` and ``float32`` give float32 (lossless, the CPU default).
    Every int8 type gives int8. ``int16``, a quantization asked for to save
    RAM, gets the closest quantized weights (int8) rather than full size;
    ``float16`` / ``bfloat16`` get float32, as ctranslate2 does on CPU.
    """
    requested = (compute_type or "auto").lower()
    if requested in ("auto", "default", "float32"):
        return "float32", None
    if requested.startswith("int8"):
        return "int8", None
    weights = "int8" if requested == "int16" else "float32"
    return weights, (
        f"Parakeet has float32 and int8 weights only: --compute-type {compute_type} runs {weights}."
    )


def model_files(compute_type: str | None) -> tuple[str, ...]:
    """The repo files one compute type needs."""
    return _COMMON_FILES + _MODEL_FILES[weights_for(compute_type)]


def _files_root() -> Path:
    """Next to the HuggingFace hub cache, so hard links from it stay on one filesystem."""
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE).parent / "transcript-video"


def model_dir() -> Path:
    """Where the Parakeet files are kept as real files (one folder per pinned revision)."""
    return _files_root() / f"{MODEL_NAME}-{HF_REVISION[:12]}"


def vad_dir() -> Path:
    """Where the Silero VAD files are kept (one folder per pinned revision)."""
    return _files_root() / f"silero-vad-{VAD_REVISION[:12]}"


def _wanted(compute_type: str | None) -> list[tuple[str, str, tuple[str, ...], Path]]:
    return [
        (HF_REPO, HF_REVISION, model_files(compute_type), model_dir()),
        (VAD_REPO, VAD_REVISION, VAD_FILES, vad_dir()),
    ]


def missing_files(compute_type: str | None) -> list[tuple[str, str, str]]:
    """``(repo, revision, name)`` of the files neither in our folders nor in the HF cache.

    A first run downloads them.
    """
    from huggingface_hub import try_to_load_from_cache

    missing = []
    for repo, revision, names, folder in _wanted(compute_type):
        for name in names:
            if (folder / name).is_file():
                continue
            cached = try_to_load_from_cache(repo, name, revision=revision)
            if not (isinstance(cached, str) and Path(cached).is_file()):
                missing.append((repo, revision, name))
    return missing


def download_estimate(compute_type: str | None, missing: list[tuple[str, str, str]]) -> str:
    """Rough size of downloading ``missing``: the weight size when a weight file is in it."""
    heavy = set(_MODEL_FILES[weights_for(compute_type)])
    if any(repo == HF_REPO and name in heavy for repo, _, name in missing):
        return DOWNLOAD_SIZE[weights_for(compute_type)]
    return "a few MB"


def prepare_files(compute_type: str | None) -> tuple[Path, Path]:
    """Make sure the Parakeet and VAD files are real files in their folders; return both folders."""
    for repo, revision, names, folder in _wanted(compute_type):
        _link_files(repo, revision, names, folder)
    return model_dir(), vad_dir()


def _link_files(repo: str, revision: str, names: tuple[str, ...], dest: Path) -> None:
    """Download into the HuggingFace cache, then hard-link (or copy) each file into ``dest``.

    The cache alone doesn't work for Parakeet's fp32 weights: it keeps each
    file as a blob in a shared folder, so ``encoder-model.onnx`` and its
    external ``encoder-model.onnx.data`` resolve to different directories,
    and ONNX Runtime (1.31) refuses external data that escapes the model's
    folder. The folder name carries the revision, so files of different
    revisions never mix; each file goes in under a unique temporary name and
    is renamed into place, so concurrent first runs can't hand each other a
    half-copied file.
    """
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = dest / name
        if target.is_file():
            continue
        source = Path(hf_hub_download(repo, name, revision=revision)).resolve()
        partial = dest / f".{name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.partial"
        try:
            try:
                os.link(source, partial)
            except OSError:
                shutil.copy2(source, partial)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)


def _session_options(threads: int | None):
    import onnxruntime as ort

    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = int(threads)
    options.inter_op_num_threads = 1
    return options


def load_model(compute_type: str | None, threads: int | None = None) -> ParakeetModel:
    """Load Parakeet and the Silero VAD on ONNX Runtime's CPU provider.

    ``threads`` sets the intra-op threads (``None``: ONNX Runtime's default).
    The provider is forced: with onnxruntime-gpu (or a platform provider)
    installed, onnx-asr would otherwise pick it, ignoring the device plan.
    """
    import onnx_asr

    folder, vad_folder = prepare_files(compute_type)
    weights = weights_for(compute_type)
    options = _session_options(threads)
    asr = onnx_asr.load_model(
        ONNX_ASR_MODEL,
        folder,
        quantization="int8" if weights == "int8" else None,
        sess_options=options,
        providers=CPU_PROVIDERS,
    )
    vad = onnx_asr.load_vad(VAD_NAME, vad_folder, sess_options=options, providers=CPU_PROVIDERS)
    return ParakeetModel(asr, vad)


class ParakeetModel:
    """A loaded Parakeet + VAD pair; :meth:`transcribe` mirrors whisperX's result shape."""

    def __init__(self, asr: Any, vad: Any):
        self._asr = asr
        self._vad = vad

    def transcribe(self, audio: Any, *, batch_size: int) -> dict[str, Any]:
        """Transcribe 16 kHz mono float32 ``audio``; return ``{"segments", "language": None}``."""
        import numpy as np

        recognizer = self._asr.with_vad(
            self._vad,
            batch_size=batch_size,
            max_speech_duration_s=MAX_CHUNK_S,
            min_silence_duration_ms=MIN_SILENCE_MS,
            speech_pad_ms=SPEECH_PAD_MS,
        ).with_timestamps()
        waveform = np.asarray(audio, dtype=np.float32)
        chunks = [
            (
                chunk.start,
                chunk.end,
                words_from_tokens(
                    chunk.tokens, chunk.timestamps, chunk.logprobs, offset=chunk.start, chunk_end=chunk.end
                ),
            )
            for chunk in recognizer.recognize(waveform, sample_rate=SAMPLE_RATE)
        ]
        hard_cuts = merge_overlaps(chunks)
        if hard_cuts:
            shown = ", ".join(f"{t:.1f}s" for t in hard_cuts[:5]) + (" …" if len(hard_cuts) > 5 else "")
            logger.warning(
                "Parakeet: %d cut(s) at %g s inside continuous speech (no pause found), at %s; "
                "a word at a cut may be split.",
                len(hard_cuts), MAX_CHUNK_S, shown,
            )
        segments = [seg for _, _, words in chunks for seg in split_segments(words)]
        return {"segments": segments, "language": None}


def merge_overlaps(chunks: list[tuple[float, float, list[dict[str, Any]]]]) -> list[float]:
    """Keep each word once where consecutive VAD chunks overlap; return the cut times.

    onnx-asr pads every chunk by :data:`SPEECH_PAD_MS` on both sides. Where
    it cuts continuous speech at :data:`MAX_CHUNK_S` (and where two chunks
    are split at a pause shorter than twice the padding) the next chunk
    starts before the previous one ends, so a word in the overlap can be
    decoded twice. Both chunks keep only the words on their side of the
    overlap's midpoint, and no word of the first one ends past it. Mutates
    the word lists in place. A full ``2 x pad`` overlap is the mark of a
    cut: those times are returned, for the log.
    """
    pad = SPEECH_PAD_MS / 1000
    cuts: list[float] = []
    for (_, prev_end, prev_words), (start, _, words) in zip(chunks, chunks[1:], strict=False):
        overlap = prev_end - start
        if overlap <= 0:
            continue
        if overlap >= 2 * pad - 1e-3:
            cuts.append(start + pad)
        middle = start + overlap / 2
        prev_words[:] = [w for w in prev_words if w["start"] < middle]
        for w in prev_words:
            w["end"] = min(w["end"], max(middle, w["start"]))
        words[:] = [w for w in words if w["start"] >= middle]
    return cuts


def words_from_tokens(
    tokens: list[str] | None,
    timestamps: list[float] | None,
    logprobs: list[float] | None,
    *,
    offset: float,
    chunk_end: float,
) -> list[dict[str, Any]]:
    """Group sub-word tokens into words with times in the file's timeline.

    A token that starts with a space opens a word (onnx-asr turns the
    SentencePiece ``▁`` into a space); others, punctuation included, extend
    the current word. ``start`` is the first token's time; ``end`` is the
    next word's start, capped at :data:`WORD_TAIL_S` after the word's last
    token so a pause isn't swallowed, and at least :data:`MIN_WORD_S` long
    (except at the very end of the chunk, which no word passes).
    Each word keeps the sum and count of its token logprobs (``_lp``,
    ``_n``) for :func:`split_segments`.
    """
    if not tokens or not timestamps:
        return []
    probs = logprobs if logprobs and len(logprobs) == len(tokens) else [None] * len(tokens)
    words: list[dict[str, Any]] = []
    for token, ts, lp in zip(tokens, timestamps, probs, strict=False):
        t = offset + float(ts)
        if token.startswith(" ") or not words:
            words.append({"word": token.strip(), "start": t, "_last": t, "_lp": 0.0, "_n": 0})
        else:
            words[-1]["word"] += token
            words[-1]["_last"] = t
        if lp is not None:
            words[-1]["_lp"] += float(lp)
            words[-1]["_n"] += 1
    words = [w for w in words if w["word"]]
    for i, w in enumerate(words):
        nxt = words[i + 1]["start"] if i + 1 < len(words) else chunk_end
        end = max(min(nxt, w["_last"] + WORD_TAIL_S), w["start"] + MIN_WORD_S)
        w["end"] = max(min(end, chunk_end), w["start"])  # never past the chunk
        del w["_last"]
    return words


def split_segments(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cut a chunk's words into sentence-level segments.

    A segment ends after a word that ends a sentence (``.`` ``?`` ``!``
    ``…``, possibly followed by a closing quote or bracket; Parakeet
    punctuates), or before a word that follows a pause of
    :data:`SENTENCE_GAP_S` or more. Each segment gets ``start``, ``end``,
    ``text``, ``words`` (``word``, ``start``, ``end``, ``logprob``: mean over
    the word's tokens) and ``token_logprob`` (mean over all its tokens).
    The names differ from whisperX's ``avg_logprob`` / ``score`` on purpose:
    the values mean something else, and the suspect thresholds tuned for
    whisperX must not read them.
    """
    groups: list[list[dict[str, Any]]] = []
    for w in words:
        new = (
            not groups
            or groups[-1][-1]["word"].rstrip(_CLOSING).endswith(SENTENCE_END)
            or w["start"] - groups[-1][-1]["end"] >= SENTENCE_GAP_S
        )
        if new:
            groups.append([w])
        else:
            groups[-1].append(w)
    segments = []
    for group in groups:
        lp_sum = sum(w["_lp"] for w in group)
        n = sum(w["_n"] for w in group)
        segments.append({
            "start": round(group[0]["start"], 3),
            "end": round(group[-1]["end"], 3),
            "text": " ".join(w["word"] for w in group),
            "words": [
                {
                    "word": w["word"],
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                    "logprob": round(w["_lp"] / w["_n"], 4) if w["_n"] else None,
                }
                for w in group
            ],
            "token_logprob": round(lp_sum / n, 4) if n else None,
        })
    return segments
