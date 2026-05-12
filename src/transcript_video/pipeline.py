"""Orchestrate the full transcription pipeline: ASR → align → diarize → write.

Two persistence layers:

- ``*_transcript.aligned.json`` — written immediately after alignment, before
  diarization runs. Acts as a free safety net: if diarization fails (gated repo,
  HF outage, missing model, ...) the costly ASR + alignment work is preserved
  and can be resumed via ``--resume-from-aligned``.
- ``*_transcript.json`` — the final canonical artifact, written after the full
  pipeline completes (with or without diarization).

The ``stage`` field at the top level discriminates the two payloads.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from . import asr as asr_mod
from . import diarize as diarize_mod
from . import speaker_db as speaker_db_mod
from . import speaker_embed as speaker_embed_mod
from . import stats as stats_mod
from .config import RunConfig
from .markdown import render_markdown, write_markdown
from .utils import format_timestamp_short
from .writers import write_json, write_srt, write_txt

logger = logging.getLogger("transcript_video.pipeline")

STAGE_ALIGNED = "aligned"
STAGE_COMPLETE = "complete"

KNOWN_MEDIA_EXTENSIONS = frozenset({
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".wmv",
    ".mpeg", ".mpg", ".m4v", ".ts", ".mts", ".m2ts", ".ogv", ".3gp",
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus",
    ".wma", ".aiff", ".aif",
})


def _resolve_output_paths(config: RunConfig) -> dict[str, Path]:
    input_path = config.input_file.resolve()
    output_dir = config.output.output_dir.resolve() if config.output.output_dir else input_path.parent
    basename = config.output.basename or input_path.stem
    suffix = "transcript"
    return {
        "json": output_dir / f"{basename}_{suffix}.json",
        "aligned": output_dir / f"{basename}_{suffix}.aligned.json",
        "srt": output_dir / f"{basename}_{suffix}.srt",
        "txt": output_dir / f"{basename}_{suffix}.txt",
        "md": output_dir / f"{basename}_{suffix}.md",
    }


def _normalize_segment(seg: dict[str, Any]) -> dict[str, Any]:
    """Normalize a whisperX segment into our JSON schema."""
    out: dict[str, Any] = {
        "id": seg.get("id"),
        "start": float(seg.get("start") or 0.0),
        "end": float(seg.get("end") or 0.0),
        "text": (seg.get("text") or "").strip(),
    }
    if "speaker" in seg and seg["speaker"]:
        out["speaker"] = seg["speaker"]
    if "avg_logprob" in seg:
        out["avg_logprob"] = seg["avg_logprob"]
    if "no_speech_prob" in seg:
        out["no_speech_prob"] = seg["no_speech_prob"]
    if "words" in seg and seg["words"]:
        words: list[dict[str, Any]] = []
        for w in seg["words"]:
            entry: dict[str, Any] = {"word": w.get("word", "")}
            if "start" in w:
                entry["start"] = w["start"]
            if "end" in w:
                entry["end"] = w["end"]
            if "score" in w:
                entry["score"] = w["score"]
            if "speaker" in w and w["speaker"]:
                entry["speaker"] = w["speaker"]
            words.append(entry)
        out["words"] = words
    return out


def _build_payload(
    *,
    stage: str,
    source_file: Path,
    transcribed_at: datetime,
    config: RunConfig,
    device: str,
    compute_type: str,
    detected_language: str,
    language_probability: float | None,
    segments: list[dict[str, Any]],
    processing_seconds: float,
    speaker_clusters: dict[str, Any] | None = None,
    speaker_identities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical JSON payload. Used for both aligned and complete stages.

    ``speaker_clusters`` is the per-speaker embedding cache produced by
    :mod:`speaker_embed` after diarization. Aligned-stage and no-diarize-stage
    payloads carry an empty dict here.

    ``speaker_identities`` maps cluster labels to ``{name, score, source}``
    entries when ``--identify-speakers`` is on; otherwise empty. ``source``
    is ``"auto"`` for DB matches and ``"manual"`` for entries coming from
    the user's ``--speaker-map``.

    Schema version 2 added both ``speaker_clusters`` and ``speaker_identities``.
    Consumers that only look at ``segments`` / ``parameters`` / ``stats``
    (notably :mod:`markdown`) are forward- and backward-compatible without
    changes.
    """
    duration_seconds = float(segments[-1].get("end") or 0.0) if segments else 0.0
    # Annotate segments with `suspect` / `suspect_reasons` before computing the
    # per-speaker breakdown so num_suspect picks them up. Mutates segments in
    # place; idempotent on resume because the same thresholds always apply.
    stats_mod.mark_suspect_segments(segments)
    return {
        "schema_version": 2,
        "stage": stage,
        "source_file": str(source_file),
        "transcribed_at": transcribed_at.isoformat(timespec="seconds"),
        "parameters": {
            "backend": "whisperx",
            "asr": f"whisperx-{config.model}",
            "model": config.model,
            "device": device,
            "compute_type": compute_type,
            "beam_size": config.beam_size,
            "vad_filter": True,
            "language_forced": config.language,
            "initial_prompt": config.initial_prompt or None,
            "hotwords": config.hotwords or None,
            "anti_loop": config.anti_loop,
            "condition_on_previous_text": False if config.anti_loop else None,
            "compression_ratio_threshold": 2.0 if config.anti_loop else None,
            "no_speech_threshold": 0.5 if config.anti_loop else None,
            "diarization": {
                "enabled": config.diarization.enabled,
                "model_name": config.diarization.model_name,
                "num_speakers": config.diarization.num_speakers,
                "min_speakers": config.diarization.min_speakers,
                "max_speakers": config.diarization.max_speakers,
            },
            "suspect_thresholds": {
                "avg_logprob": stats_mod.DEFAULT_AVG_LOGPROB_THRESHOLD,
                "no_speech_prob": stats_mod.DEFAULT_NO_SPEECH_PROB_THRESHOLD,
            },
        },
        "audio_info": {
            "language_detected": detected_language,
            "language_probability": language_probability,
            "duration_seconds": duration_seconds,
        },
        "stats": {
            "num_segments": len(segments),
            "num_speakers": stats_mod.count_unique_speakers(segments),
            "num_suspect": stats_mod.count_suspect_segments(segments),
            "processing_seconds": processing_seconds,
            "speakers": stats_mod.compute_speaker_stats(segments),
        },
        "speaker_clusters": speaker_clusters or {},
        "speaker_identities": speaker_identities or {},
        "segments": segments,
        "full_text": "\n".join(seg["text"] for seg in segments if seg.get("text")),
    }


def _run_asr_and_align(config: RunConfig, *, paths: dict[str, Path]) -> dict[str, Any]:
    """Run model load → ASR → alignment, returning the aligned-stage payload.

    Side effect: when diarization is enabled, persists the aligned payload to
    ``paths['aligned']`` so a downstream failure doesn't lose the work.
    """
    input_path = config.input_file.resolve()
    device, compute_type = asr_mod.resolve_device_and_compute(config.device, config.compute_type)
    if device == "cuda":
        logger.info("Using CUDA acceleration (compute_type=%s).", compute_type)
    else:
        logger.warning("Running on CPU (slower).")

    logger.info("Loading whisperX model '%s'...", config.model)
    model = asr_mod.load_whisperx_model(config.model, device, compute_type, config.beam_size)

    transcribe_kwargs = asr_mod.build_transcribe_kwargs(
        language=config.language,
        initial_prompt=config.initial_prompt,
        hotwords=config.hotwords,
        anti_loop=config.anti_loop,
    )

    prompt_display = "(none)" if not config.initial_prompt else (
        config.initial_prompt if len(config.initial_prompt) <= 80 else config.initial_prompt[:80] + "..."
    )
    hotwords_display = "(none)" if not config.hotwords else config.hotwords
    language_display = config.language if config.language else "(autodetect)"

    logger.info("Starting transcription of: %s", input_path.name)
    logger.info("  beam_size:       %s", config.beam_size)
    logger.info("  language:        %s", language_display)
    logger.info("  initial prompt:  %s", prompt_display)
    logger.info("  hotwords:        %s", hotwords_display)
    logger.info("  anti-loop:       %s", "on" if config.anti_loop else "off")
    logger.info("  diarization:     %s", "on" if config.diarization.enabled else "off")

    audio = asr_mod.load_audio(input_path)

    start_time = datetime.now()
    asr_result = model.transcribe(audio, batch_size=16, **transcribe_kwargs)

    detected_language = asr_result.get("language") or config.language or "und"
    raw_segments: list[dict[str, Any]] = list(asr_result.get("segments", []))
    logger.info("ASR done. Detected language: %s. %d segments.", detected_language, len(raw_segments))
    for seg in raw_segments:
        text = (seg.get("text") or "").strip()
        logger.debug("%s %s", format_timestamp_short(float(seg.get("start") or 0.0)), text)

    logger.info("Aligning to word level...")
    aligned = asr_mod.align_segments(raw_segments, detected_language, audio, device)
    aligned_segments = [_normalize_segment(s) for s in aligned.get("segments", [])]
    align_time = datetime.now()
    align_seconds = (align_time - start_time).total_seconds()

    payload = _build_payload(
        stage=STAGE_ALIGNED,
        source_file=input_path,
        transcribed_at=align_time,
        config=config,
        device=device,
        compute_type=compute_type,
        detected_language=detected_language,
        language_probability=asr_result.get("language_probability"),
        segments=aligned_segments,
        processing_seconds=align_seconds,
    )

    # Free safety net: write the aligned payload before diarization touches anything.
    if config.diarization.enabled:
        write_json(payload, paths["aligned"])
        logger.info("Aligned snapshot saved: %s", paths["aligned"])

    return payload


def _load_aligned_payload(path: Path) -> dict[str, Any]:
    """Load a previously-written aligned snapshot."""
    import json

    if not path.exists():
        print(f"Error: aligned snapshot not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    stage = payload.get("stage")
    if stage not in (STAGE_ALIGNED, STAGE_COMPLETE):
        print(
            f"Error: {path} has unexpected stage '{stage}'; expected '{STAGE_ALIGNED}'.",
            file=sys.stderr,
        )
        sys.exit(1)
    return payload


def _diarize_payload(payload: dict[str, Any], config: RunConfig) -> dict[str, Any]:
    """Run diarization on an aligned payload and return the complete payload."""
    input_path = config.input_file.resolve()
    device = payload["parameters"].get("device") or "cpu"

    logger.info("Running speaker diarization (pyannote)...")
    audio = asr_mod.load_audio(input_path)
    aligned_for_whisperx = {
        "segments": payload["segments"],
        "word_segments": [],
    }
    token = diarize_mod.resolve_hf_token(config.diarization.hf_token)
    diarized = diarize_mod.diarize_and_assign(
        aligned_for_whisperx,
        audio,
        hf_token=token,
        device=device,
        num_speakers=config.diarization.num_speakers,
        min_speakers=config.diarization.min_speakers,
        max_speakers=config.diarization.max_speakers,
        model_name=config.diarization.model_name,
    )
    diarized_segments = [_normalize_segment(s) for s in diarized.get("segments", [])]

    # Best-effort cluster-embedding extraction. Failure is logged and the
    # pipeline continues with an empty `speaker_clusters`; the transcript is
    # still useful, only the future auto-identification capability is lost
    # for this run (and can be recovered by re-processing).
    speaker_clusters: dict[str, Any] = {}
    if token and diarized_segments:
        try:
            logger.info("Extracting per-cluster speaker embeddings...")
            speaker_clusters = speaker_embed_mod.extract_cluster_embeddings(
                input_path,
                diarized_segments,
                hf_token=token,
                device=device,
            )
            logger.info(
                "Cached embeddings for %d cluster(s).", len(speaker_clusters)
            )
        except Exception as e:
            logger.warning(
                "Speaker embedding extraction failed: %s. "
                "Cluster embeddings will not be cached for this run.", e,
            )

    speaker_identities = _resolve_speaker_identities(speaker_clusters, config)

    return _build_payload(
        stage=STAGE_COMPLETE,
        source_file=input_path,
        transcribed_at=datetime.now(),
        config=config,
        device=device,
        compute_type=payload["parameters"].get("compute_type") or "",
        detected_language=payload["audio_info"].get("language_detected") or "und",
        language_probability=payload["audio_info"].get("language_probability"),
        segments=diarized_segments,
        processing_seconds=float(payload.get("stats", {}).get("processing_seconds") or 0.0),
        speaker_clusters=speaker_clusters,
        speaker_identities=speaker_identities,
    )


def _resolve_speaker_identities(
    speaker_clusters: dict[str, Any], config: RunConfig
) -> dict[str, Any]:
    """Build ``speaker_identities`` for the JSON payload.

    Only runs when ``config.identify.enabled`` is True. Combines two sources:

    - **Auto** (``source="auto"``): each cluster's best DB match above
      ``config.identify.threshold``, via :func:`speaker_db.auto_resolve_speaker_map`.
    - **Manual** (``source="manual"``): every entry in ``config.speaker_map``
      whose label is present in ``speaker_clusters``. Manual entries override
      auto ones for the same label.

    Any DB-side failure (missing file, unreadable, model mismatch) is logged
    and treated as "no auto matches"; manual entries are still recorded.
    """
    if not config.identify.enabled:
        return {}

    identities: dict[str, Any] = {}

    if speaker_clusters:
        try:
            db_path = speaker_db_mod.resolve_db_path(config.identify.voice_db)
            db = speaker_db_mod.load_db(db_path)

            first = next(iter(speaker_clusters.values()))
            model = first.get("embedding_model") if isinstance(first, dict) else None
            if model and not speaker_db_mod.embedding_model_compatible(db, model):
                logger.warning(
                    "Voice DB at %s uses embedding_model %r; clusters use %r. "
                    "Auto-identification skipped for this run.",
                    db_path, db.get("embedding_model"), model,
                )
            else:
                auto = speaker_db_mod.auto_resolve_speaker_map(
                    speaker_clusters, db, threshold=config.identify.threshold,
                )
                for label, info in auto.items():
                    identities[label] = {
                        "name": info["name"],
                        "score": info["score"],
                        "source": "auto",
                    }
                if auto:
                    logger.info(
                        "Auto-identified %d/%d cluster(s) from %s.",
                        len(auto), len(speaker_clusters), db_path,
                    )
                else:
                    logger.info(
                        "Auto-identification ran against %s but no cluster cleared the threshold (%.2f).",
                        db_path, config.identify.threshold,
                    )
        except Exception as e:
            logger.warning("Auto-identification failed: %s. Manual entries (if any) still recorded.", e)

    # Manual entries override auto for the same label.
    for label, name in (config.speaker_map or {}).items():
        if label in speaker_clusters:
            identities[label] = {
                "name": name,
                "score": None,
                "source": "manual",
            }

    return identities


def _write_outputs(payload: dict[str, Any], config: RunConfig, paths: dict[str, Path]) -> dict[str, Path]:
    """Write the requested output files. JSON is always produced."""
    written: dict[str, Path] = {}
    segments = payload.get("segments", [])

    if config.output.write_json:
        write_json(payload, paths["json"])
        written["json"] = paths["json"]
    if config.output.write_srt:
        write_srt(segments, paths["srt"])
        written["srt"] = paths["srt"]
    if config.output.write_txt:
        write_txt(segments, paths["txt"])
        written["txt"] = paths["txt"]
    if config.output.write_md:
        md = render_markdown(
            payload,
            speaker_map=config.speaker_map,
            fm_date=config.frontmatter.date,
            tags=config.frontmatter.tags,
            fm_source=config.frontmatter.source,
        )
        write_markdown(md, paths["md"])
        written["md"] = paths["md"]

    return written


def run_pipeline(config: RunConfig) -> dict[str, Path]:
    """Run the full pipeline and return the dict of paths actually written.

    When ``config.resume_from_aligned`` is set, ASR + alignment are skipped and
    we go straight from the cached snapshot to diarization (if enabled) and the
    final outputs.
    """
    input_path = config.input_file.resolve()
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if input_path.suffix.lower() not in KNOWN_MEDIA_EXTENSIONS:
        logger.warning(
            "Input has an unrecognized media extension; proceeding anyway (ffmpeg will decide): %s",
            input_path.name,
        )

    paths = _resolve_output_paths(config)

    if config.resume_from_aligned:
        logger.info("Resuming from aligned snapshot: %s", config.resume_from_aligned)
        aligned_payload = _load_aligned_payload(config.resume_from_aligned)
    else:
        aligned_payload = _run_asr_and_align(config, paths=paths)

    if config.diarization.enabled:
        try:
            final_payload = _diarize_payload(aligned_payload, config)
        except SystemExit:
            # Pre-flight or diarize printed an error; the aligned snapshot is still
            # on disk so the user can retry with --resume-from-aligned.
            if not config.resume_from_aligned:
                logger.error(
                    "Tip: aligned snapshot preserved at %s — retry with --resume-from-aligned.",
                    paths["aligned"],
                )
            raise
    else:
        # No diarization: promote the aligned payload to "complete" stage as-is.
        final_payload = dict(aligned_payload)
        final_payload["stage"] = STAGE_COMPLETE

    written = _write_outputs(final_payload, config, paths)

    logger.info("Pipeline done.")
    print("Files written:")
    for kind, p in written.items():
        print(f"  - {kind:5s}: {p}")
    return written
