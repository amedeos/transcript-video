"""User-level voice-print database for cross-video speaker identification.

Stores speaker embeddings keyed by name in a single JSON file under
``$XDG_DATA_HOME/transcript-video/voices.json`` (default
``~/.local/share/transcript-video/voices.json``). Lookup is cosine
similarity over per-speaker enrolled samples; an identity is only
assigned when the score crosses a caller-provided threshold.

This module is intentionally torch-free: it stores and compares vectors
that the heavy path (:mod:`transcript_video.speaker_embed`) computed
during the pipeline. It can therefore run from ``transcript-to-md`` on a
machine without GPU or PyTorch installed. The architectural guard lives
in :mod:`tests.test_torch_free`.

Two distinct schema versions are at play in the project:

- The *transcript JSON* ``schema_version`` (see :mod:`pipeline`) — bumped
  when the on-disk transcript payload changes shape.
- The *voice DB* :data:`DB_SCHEMA_VERSION` defined here — bumped if the
  DB file format itself changes shape.

They are independent.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DB_SCHEMA_VERSION = 1
DEFAULT_THRESHOLD = 0.65
DEFAULT_TOP_K = 3


def default_db_path() -> Path:
    """Return the default user-level DB path, respecting ``XDG_DATA_HOME``."""
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "transcript-video" / "voices.json"


def resolve_db_path(cli_path: str | Path | None) -> Path:
    """Resolve the DB path with the documented precedence.

    Order: ``cli_path`` > ``$TRANSCRIPT_VIDEO_VOICE_DB`` > :func:`default_db_path`.
    User-expansion (``~``) is applied to all sources.
    """
    if cli_path:
        return Path(cli_path).expanduser()
    env = os.environ.get("TRANSCRIPT_VIDEO_VOICE_DB")
    if env:
        return Path(env).expanduser()
    return default_db_path()


def _empty_db() -> dict[str, Any]:
    return {
        "schema_version": DB_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "embedding_model": None,
        "speakers": {},
    }


def load_db(path: str | Path) -> dict[str, Any]:
    """Load the DB from ``path``, or return a fresh empty DB if it doesn't exist.

    A missing file is *not* an error: identification and enrollment both start
    from an empty DB on cold-start. A malformed file is an error and surfaces
    the underlying :class:`json.JSONDecodeError`.
    """
    p = Path(path)
    if not p.exists():
        return _empty_db()
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_db(db: dict[str, Any], path: str | Path) -> None:
    """Atomically write the DB to ``path`` with 0600 permissions.

    The parent directory is created with 0700. The write is atomic via
    ``tempfile`` + :func:`os.replace`, which also preserves a symlink at
    ``path`` if one is present (the symlink target is overwritten in place).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Tighten dir permissions if it pre-existed with looser bits.
    with contextlib.suppress(OSError):
        os.chmod(p.parent, 0o700)

    # If `p` is a symlink, write into its parent and replace the resolved target
    # so the symlink itself survives.
    target = p.resolve() if p.is_symlink() else p
    fd, tmp_path = tempfile.mkstemp(
        prefix=".voices.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, target)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def add_sample(
    db: dict[str, Any],
    name: str,
    embedding: list[float],
    *,
    source: str,
    embedding_model: str,
    cluster: str | None = None,
    duration_s: float | None = None,
) -> None:
    """Append a sample to ``name``'s enrolled list. Mutates ``db`` in place.

    The first call on an empty DB sets :data:`embedding_model` for the whole
    file; subsequent calls with a different ``embedding_model`` raise
    :class:`ValueError`. This is intentional: cosine similarity between
    embeddings produced by different models is meaningless.
    """
    if not name:
        raise ValueError("speaker name must be non-empty")
    if not embedding:
        raise ValueError("embedding must be non-empty")
    if not embedding_model:
        raise ValueError("embedding_model must be non-empty")

    existing = db.get("embedding_model")
    if existing is None:
        db["embedding_model"] = embedding_model
    elif existing != embedding_model:
        raise ValueError(
            f"DB embedding_model is {existing!r}; cannot add sample produced by {embedding_model!r}"
        )

    sample: dict[str, Any] = {
        "embedding": list(embedding),
        "source": source,
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }
    if cluster:
        sample["cluster"] = cluster
    if duration_s is not None:
        sample["duration_s"] = float(duration_s)

    db.setdefault("speakers", {}).setdefault(name, []).append(sample)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Returns ``0.0`` for mismatched lengths, empty inputs, or zero-norm vectors
    (which can occur if an embedding was clipped before storage).
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


@dataclass(frozen=True)
class MatchResult:
    name: str
    score: float
    n_samples: int


def match(
    embedding: list[float],
    db: dict[str, Any],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
) -> MatchResult | None:
    """Return the best-matching speaker for ``embedding``, or ``None`` below threshold.

    Per-speaker score is the average of the ``top_k`` highest per-sample cosine
    similarities (capped at the number of samples). ``top_k=1`` therefore
    reduces to "best matching sample wins" — the most permissive setting and
    the right choice when a speaker has only a few samples enrolled.
    """
    speakers = db.get("speakers") or {}
    if not speakers:
        return None

    best: MatchResult | None = None
    for name, samples in speakers.items():
        if not samples:
            continue
        scores = sorted(
            (_cosine(embedding, s.get("embedding", [])) for s in samples),
            reverse=True,
        )
        k = max(1, min(top_k, len(scores)))
        score = sum(scores[:k]) / k
        if best is None or score > best.score:
            best = MatchResult(name=name, score=score, n_samples=len(samples))

    if best is not None and best.score >= threshold:
        return best
    return None


def embedding_model_compatible(db: dict[str, Any], model_id: str) -> bool:
    """True if the DB has no committed model, or its model matches ``model_id``."""
    existing = db.get("embedding_model")
    return existing is None or existing == model_id
