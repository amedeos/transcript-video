"""Append speakers from a transcript JSON to the user-level voice-print DB.

The "learning" half of the speaker-identification workflow:

1. ``transcript-from-video`` produces a JSON whose ``speaker_clusters`` field
   carries one cached embedding per diarized cluster (computed at pipeline
   time, when pyannote was already loaded).
2. The user inspects the transcript, figures out who each ``SPEAKER_XX``
   actually is, and runs ``transcript-learn`` with that mapping.
3. This module reads the cached embeddings from the JSON and appends them
   to the DB under the user-provided names — **no audio or GPU required**.

The whole "learn later" workflow rests on the embeddings being cached in
the JSON. If they aren't (schema v1, or v2 without diarization, or
extraction failed at pipeline time), enrollment requires re-processing.

This module is torch-free; the guard lives in :mod:`tests.test_torch_free`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import speaker_db

REQUIRED_SCHEMA_VERSION = 2


class EnrollmentError(ValueError):
    """Fail-fast condition the caller should surface to the user verbatim."""


def _load_transcript(json_path: Path) -> dict[str, Any]:
    try:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise EnrollmentError(f"JSON file not found: {json_path}") from e
    except json.JSONDecodeError as e:
        raise EnrollmentError(f"Invalid JSON in {json_path}: {e}") from e


def _extract_clusters(transcript: dict[str, Any], json_path: Path) -> dict[str, dict[str, Any]]:
    version = transcript.get("schema_version")
    if not isinstance(version, int) or version < REQUIRED_SCHEMA_VERSION:
        raise EnrollmentError(
            f"{json_path} has schema_version {version!r}; transcript-learn needs "
            f">= {REQUIRED_SCHEMA_VERSION} (with cached speaker_clusters). "
            "Re-process the video with the current transcript-from-video to upgrade."
        )
    clusters = transcript.get("speaker_clusters")
    if not isinstance(clusters, dict) or not clusters:
        raise EnrollmentError(
            f"{json_path} has no cached speaker_clusters; either diarization was "
            "disabled or embedding extraction failed at pipeline time. "
            "Re-process the video to get the embeddings cached."
        )
    return clusters


def _consistent_embedding_model(clusters: dict[str, dict[str, Any]]) -> str:
    """Return the single embedding_model used by all clusters, or raise.

    A well-formed JSON has the same model across all clusters (one pipeline
    run uses one model). Mixed models suggest a corrupt or hand-edited file.
    """
    models = {info.get("embedding_model") for info in clusters.values()}
    models.discard(None)
    if not models:
        raise EnrollmentError(
            "Cluster embeddings carry no embedding_model field; cannot enroll "
            "without knowing which model produced them. Re-process the video."
        )
    if len(models) != 1:
        raise EnrollmentError(
            f"Cluster embeddings reference inconsistent models {models!r}; "
            "this JSON appears corrupt. Re-process the video."
        )
    return next(iter(models))


def learn_from_transcript(
    json_path: str | Path,
    speaker_map: dict[str, str],
    db_path: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Append samples from ``json_path`` to ``db_path`` per ``speaker_map``.

    For each ``label -> name`` in ``speaker_map``, look up the cached
    cluster embedding and append it to the DB under ``name``. Labels with
    no matching cluster (typos, or labels referring to a different video)
    are reported via ``skipped_no_cluster`` instead of failing the whole
    operation.

    Returns a summary dict::

        {
            "added": {label: name, ...},            # actually appended
            "skipped_no_cluster": [label, ...],     # mapped but no cluster
            "embedding_model": "...",               # the model used here
            "db_path": str,
            "dry_run": bool,
        }

    Raises :class:`EnrollmentError` for fail-fast conditions: bad JSON,
    schema too old, missing/empty ``speaker_clusters``, embedding-model
    mismatch with an existing DB.
    """
    json_path = Path(json_path)
    db_path = Path(db_path)

    transcript = _load_transcript(json_path)
    clusters = _extract_clusters(transcript, json_path)
    embedding_model = _consistent_embedding_model(clusters)

    db = speaker_db.load_db(db_path)
    if not speaker_db.embedding_model_compatible(db, embedding_model):
        raise EnrollmentError(
            f"DB at {db_path} was created with embedding_model "
            f"{db['embedding_model']!r}; cannot mix samples from {embedding_model!r}. "
            "Use --voice-db to point at a fresh DB, or back up and remove the existing one."
        )

    source_id = json_path.name
    added: dict[str, str] = {}
    skipped: list[str] = []

    for label, name in speaker_map.items():
        cluster = clusters.get(label)
        if cluster is None:
            skipped.append(label)
            continue
        embedding = cluster.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            skipped.append(label)
            continue
        speaker_db.add_sample(
            db, name, embedding,
            source=source_id,
            embedding_model=embedding_model,
            cluster=label,
            duration_s=cluster.get("duration_s"),
        )
        added[label] = name

    if added and not dry_run:
        speaker_db.save_db(db, db_path)

    return {
        "added": added,
        "skipped_no_cluster": skipped,
        "embedding_model": embedding_model,
        "db_path": str(db_path),
        "dry_run": dry_run,
    }
