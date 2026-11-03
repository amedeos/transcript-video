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

import gc
import logging
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from . import asr as asr_mod
from . import diarize as diarize_mod
from . import speaker_db as speaker_db_mod
from . import speaker_embed as speaker_embed_mod
from . import stats as stats_mod
from .config import RunConfig
from .markdown import build_effective_speaker_map, render_markdown, write_markdown
from .utils import format_timestamp_short
from .writers import write_json, write_srt, write_txt, write_vtt

logger = logging.getLogger("transcript_video.pipeline")

STAGE_ALIGNED = "aligned"
STAGE_COMPLETE = "complete"

# Transcript JSON schema. v2: speaker_clusters + speaker_identities.
# v3: parameters.batch_size (the ASR batch can shift the transcript by
# floating-point noise, so record what produced it).
SCHEMA_VERSION = 3

KNOWN_MEDIA_EXTENSIONS = frozenset({
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".wmv",
    ".mpeg", ".mpg", ".m4v", ".ts", ".mts", ".m2ts", ".ogv", ".3gp",
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus",
    ".wma", ".aiff", ".aif",
})


def _read_memory_gb(status_path: Path = Path("/proc/self/status")) -> tuple[float, float] | None:
    """``(current_rss, peak_rss)`` of this process in GB, or ``None`` off Linux."""
    try:
        text = status_path.read_text()
    except OSError:
        return None
    values: dict[str, float] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            fields = rest.split()
            if fields and fields[0].isdigit():
                values[key] = int(fields[0]) / (1024 * 1024)  # kB -> GB
    if "VmRSS" not in values or "VmHWM" not in values:
        return None
    return values["VmRSS"], values["VmHWM"]


def _log_memory(label: str) -> None:
    """Log RAM in use and the peak so far, to show which stage needs the most memory."""
    mem = _read_memory_gb()
    if mem is not None:
        logger.info("RAM %s: %.1f GB (peak %.1f GB)", label, mem[0], mem[1])


def _resolve_output_paths(config: RunConfig) -> dict[str, Path]:
    input_path = config.input_file.resolve()
    output_dir = config.output.output_dir.resolve() if config.output.output_dir else input_path.parent
    basename = config.output.basename or input_path.stem
    suffix = "transcript"
    return {
        "json": output_dir / f"{basename}_{suffix}.json",
        "aligned": output_dir / f"{basename}_{suffix}.aligned.json",
        "srt": output_dir / f"{basename}_{suffix}.srt",
        "vtt": output_dir / f"{basename}_{suffix}.vtt",
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
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Build the canonical JSON payload. Used for both aligned and complete stages.

    ``speaker_clusters`` is the per-speaker embedding cache produced by
    :mod:`speaker_embed` after diarization. Aligned-stage and no-diarize-stage
    payloads carry an empty dict here.

    ``speaker_identities`` maps cluster labels to ``{name, score, source}``
    entries when ``--identify-speakers`` is on; otherwise empty. ``source``
    is ``"auto"`` for DB matches and ``"manual"`` for entries coming from
    the user's ``--speaker-map``.

    ``batch_size`` is the ASR batch that produced the segments; ``None`` when
    unknown (a resume from a v2 snapshot).

    Schema version 2 added both ``speaker_clusters`` and ``speaker_identities``;
    version 3 added ``parameters.batch_size``.
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
        "schema_version": SCHEMA_VERSION,
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
            "batch_size": batch_size,
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


def _resolve_devices(config: RunConfig) -> asr_mod.DevicePlan:
    """Resolve the devices for the stages that will run, before any heavy work.

    The rules live in :func:`asr.plan_devices`, shared with the pre-flight
    check. An explicit ``--device cuda`` that this machine can't honor stops
    the run here, in seconds instead of after hours on CPU, even with
    ``--no-check``. The returned plan also carries the ASR compute type.
    """
    plan = asr_mod.plan_devices(
        config.device, run_asr=config.runs_asr, run_torch=config.runs_torch
    )
    if plan.error:
        print(f"Error: {plan.error}", file=sys.stderr)
        sys.exit(1)
    for note in plan.notes:
        logger.warning(note)
    if plan.asr:
        plan = replace(plan, compute_type=asr_mod.resolve_compute_type(plan.asr, config.compute_type))
    return plan


def _configure_threads(config: RunConfig, devices: asr_mod.DevicePlan) -> int | None:
    """Apply the CPU thread policy; return the thread count for ctranslate2 (or ``None``).

    An explicit count applies to every library that runs a stage. It comes
    from ``--threads`` / config ``threads``, else from ``OMP_NUM_THREADS`` /
    ``MKL_NUM_THREADS`` (a limit set in the environment, e.g. on a shared
    server, is honored). Otherwise the default (CPUs - 1) is applied only to
    the libraries running on CPU: ctranslate2 when the ASR does, torch when
    any torch work does (VAD, alignment, diarization). A library on the GPU
    keeps its own default; torch, for instance, defaults to the physical core
    count. A library with no stage to run is left alone.
    """
    requested = config.threads if config.threads is not None and config.threads > 0 else None
    source = ""
    if requested is None and (env := asr_mod.env_thread_limit()):
        var, requested = env
        source = f" (from {var})"
    explicit = requested is not None
    threads = asr_mod.resolve_cpu_threads(requested)
    torch_runs = devices.torch is not None or devices.vad is not None
    asr_threads = (
        threads if devices.asr is not None and (explicit or devices.asr == "cpu") else None
    )
    torch_threads = threads if torch_runs and (explicit or devices.torch_on_cpu) else None
    if torch_threads is not None:
        asr_mod.apply_torch_threads(torch_threads)
    if asr_threads is not None or torch_threads is not None:
        logger.info("CPU threads: %d%s", threads, source)
    elif devices.uses_device:
        logger.info("CPU threads: library defaults (GPU run; --threads N to override).")
    return asr_threads


def _run_asr_and_align(
    config: RunConfig,
    *,
    paths: dict[str, Path],
    devices: asr_mod.DevicePlan,
    threads: int | None = None,
) -> tuple[dict[str, Any], Any]:
    """Run model load → ASR → alignment; return ``(aligned_payload, audio)``.

    ASR runs on ``devices.asr`` (ctranslate2) and alignment on
    ``devices.torch``. Both were resolved once by :func:`_resolve_devices`.
    The decoded audio is returned so diarization can reuse it instead of
    decoding the file again.

    Side effect: when diarization is enabled, persists the aligned payload to
    ``paths['aligned']`` so a downstream failure doesn't lose the work.
    """
    input_path = config.input_file.resolve()
    device, compute_type = devices.asr, devices.compute_type
    if device == "cuda":
        logger.info("Using CUDA acceleration (compute_type=%s).", compute_type)
    else:
        logger.warning("Running on CPU (compute_type=%s, slower).", compute_type)

    logger.info("Loading whisperX model '%s'...", config.model)
    asr_options = asr_mod.build_asr_options(
        beam_size=config.beam_size,
        initial_prompt=config.initial_prompt,
        hotwords=config.hotwords,
        anti_loop=config.anti_loop,
    )
    model = asr_mod.load_whisperx_model(
        config.model, device, compute_type, asr_options, threads=threads,
        vad_device=devices.vad,
    )
    _log_memory("after model load")
    batch_size = asr_mod.resolve_batch_size(config.batch_size, device)
    batch_display = str(batch_size) if config.batch_size else f"{batch_size} (auto: {device})"

    prompt_display = "(none)" if not config.initial_prompt else (
        config.initial_prompt if len(config.initial_prompt) <= 80 else config.initial_prompt[:80] + "..."
    )
    hotwords_display = "(none)" if not config.hotwords else config.hotwords
    language_display = config.language if config.language else "(autodetect)"

    logger.info("Starting transcription of: %s", input_path.name)
    logger.info("  beam_size:       %s", config.beam_size)
    logger.info("  batch_size:      %s", batch_display)
    logger.info("  language:        %s", language_display)
    logger.info("  initial prompt:  %s", prompt_display)
    logger.info("  hotwords:        %s", hotwords_display)
    logger.info("  anti-loop:       %s", "on" if config.anti_loop else "off")
    logger.info("  diarization:     %s", "on" if config.diarization.enabled else "off")

    audio = asr_mod.load_audio(input_path)

    start_time = datetime.now()
    # Decoding options went in at load time (asr_options): whisperX's batched
    # transcribe() only takes the audio, batch size and language.
    asr_result = model.transcribe(audio, batch_size=batch_size, language=config.language)
    _log_memory("after ASR")
    # Free the ASR model before alignment/diarization load their own models:
    # large-v3 in float32 on CPU holds several GB of RAM.
    del model
    gc.collect()
    _log_memory("after freeing the ASR model")

    detected_language = asr_result.get("language") or config.language or "und"
    raw_segments: list[dict[str, Any]] = list(asr_result.get("segments", []))
    logger.info("ASR done. Detected language: %s. %d segments.", detected_language, len(raw_segments))
    for seg in raw_segments:
        text = (seg.get("text") or "").strip()
        logger.debug("%s %s", format_timestamp_short(float(seg.get("start") or 0.0)), text)

    logger.info("Aligning to word level...")
    aligned = asr_mod.align_segments(raw_segments, detected_language, audio, devices.torch)
    # The alignment model was local to align_segments: collect it before
    # diarization loads pyannote.
    gc.collect()
    _log_memory("after alignment")
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
        batch_size=batch_size,
    )

    # Free safety net: write the aligned payload before diarization touches anything.
    if config.diarization.enabled:
        write_json(payload, paths["aligned"])
        logger.info("Aligned snapshot saved: %s", paths["aligned"])

    return payload, audio


def _has_speakers(payload: dict[str, Any]) -> bool:
    """Whether a payload already carries diarization results."""
    if payload.get("speaker_clusters") or payload.get("speaker_identities"):
        return True
    return any(seg.get("speaker") for seg in payload.get("segments") or [])


def _load_aligned_payload(path: Path) -> dict[str, Any]:
    """Load a snapshot to resume from (invariant #5).

    Accepted: the ``*_transcript.aligned.json`` snapshot (stage ``aligned``),
    and a finished transcript (stage ``complete``) that carries no speakers.
    That is the output of a ``--no-diarize`` run, which writes no aligned
    snapshot, so it is the only way to add diarization later. A finished
    transcript *with* speakers is refused: re-diarizing it would overwrite
    its speaker labels and identities.
    """
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
    if stage == STAGE_COMPLETE and _has_speakers(payload):
        print(
            f"Error: {path} is a finished transcript with speakers already assigned. "
            "Resuming would re-diarize it and overwrite its speaker labels and "
            "identities. Resume from the *_transcript.aligned.json snapshot instead, "
            "or use transcript-to-md to re-render it.",
            file=sys.stderr,
        )
        sys.exit(1)
    return payload


def _diarize_payload(
    payload: dict[str, Any], config: RunConfig, *, device: str, audio: Any = None
) -> dict[str, Any]:
    """Run diarization on an aligned payload and return the complete payload.

    Diarization and embeddings run on ``device``: the torch device resolved
    for *this* run, never the snapshot's. So an aligned snapshot made on a GPU
    machine can be resumed on a CPU-only one, and vice versa.
    ``parameters.device`` in the JSON keeps the device the ASR ran on (read
    from the payload), so it stays consistent with ``compute_type``.

    ``audio`` is the waveform the ASR already decoded; ``None`` (on resume)
    decodes the input file here.
    """
    input_path = config.input_file.resolve()
    asr_device = payload["parameters"].get("device") or "cpu"

    logger.info("Running speaker diarization (pyannote)...")
    if audio is None:
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
    gc.collect()
    _log_memory("after diarization")

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
                audio=audio,
            )
            logger.info(
                "Cached embeddings for %d cluster(s).", len(speaker_clusters)
            )
            _log_memory("after speaker embeddings")
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
        device=asr_device,
        compute_type=payload["parameters"].get("compute_type") or "",
        detected_language=payload["audio_info"].get("language_detected") or "und",
        language_probability=payload["audio_info"].get("language_probability"),
        segments=diarized_segments,
        processing_seconds=float(payload.get("stats", {}).get("processing_seconds") or 0.0),
        speaker_clusters=speaker_clusters,
        speaker_identities=speaker_identities,
        batch_size=payload["parameters"].get("batch_size"),
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

    # Effective {label: name} map for subtitle cues, only when requested.
    speaker_names = (
        build_effective_speaker_map(payload, config.speaker_map)
        if config.output.subtitle_speakers
        else None
    )

    if config.output.write_json:
        write_json(payload, paths["json"])
        written["json"] = paths["json"]
    if config.output.write_srt:
        write_srt(segments, paths["srt"], speaker_names=speaker_names)
        written["srt"] = paths["srt"]
    if config.output.write_vtt:
        write_vtt(segments, paths["vtt"], speaker_names=speaker_names)
        written["vtt"] = paths["vtt"]
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

    devices = _resolve_devices(config)
    asr_threads = _configure_threads(config, devices)

    audio = None
    if config.resume_from_aligned:
        logger.info("Resuming from aligned snapshot: %s", config.resume_from_aligned)
        aligned_payload = _load_aligned_payload(config.resume_from_aligned)
    else:
        aligned_payload, audio = _run_asr_and_align(
            config, paths=paths, devices=devices, threads=asr_threads
        )

    if config.diarization.enabled:
        try:
            final_payload = _diarize_payload(
                aligned_payload, config, device=devices.torch, audio=audio
            )
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
    audio = None  # no stage needs the waveform past this point

    written = _write_outputs(final_payload, config, paths)

    logger.info("Pipeline done.")
    print("Files written:")
    for kind, p in written.items():
        print(f"  - {kind:5s}: {p}")
    return written
