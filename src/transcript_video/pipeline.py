"""Orchestrate the full transcription pipeline: ASR → align → diarize → write."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from . import asr as asr_mod
from . import diarize as diarize_mod
from .config import RunConfig
from .markdown import render_markdown, write_markdown
from .utils import format_timestamp_short
from .writers import write_json, write_srt, write_txt


def _resolve_output_paths(config: RunConfig) -> dict[str, Path]:
    input_path = config.input_file.resolve()
    output_dir = config.output.output_dir.resolve() if config.output.output_dir else input_path.parent
    basename = config.output.basename or input_path.stem
    suffix = "transcript"
    return {
        "json": output_dir / f"{basename}_{suffix}.json",
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


def run_pipeline(config: RunConfig) -> dict[str, Path]:
    """Run the full pipeline and return the dict of paths actually written."""
    input_path = config.input_file.resolve()
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if input_path.suffix.lower() != ".mp4":
        print(f"Warning: input does not have an .mp4 extension; proceeding anyway: {input_path.name}")

    device, compute_type = asr_mod.resolve_device_and_compute(config.device, config.compute_type)
    if device == "cuda":
        print(f"Using CUDA acceleration (compute_type={compute_type}).")
    else:
        print("Warning: running on CPU (slower).")

    print(f"Loading whisperX model '{config.model}'...")
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

    print(f"Starting transcription of: {input_path.name}")
    print(f"  beam_size:       {config.beam_size}")
    print(f"  language:        {language_display}")
    print(f"  initial prompt:  {prompt_display}")
    print(f"  hotwords:        {hotwords_display}")
    print(f"  anti-loop:       {'on' if config.anti_loop else 'off'}")
    print(f"  diarization:     {'on' if config.diarization.enabled else 'off'}")
    print("-" * 60)

    audio = asr_mod.load_audio(input_path)

    start_time = datetime.now()
    asr_result = model.transcribe(audio, batch_size=16, **transcribe_kwargs)

    detected_language = asr_result.get("language") or config.language or "und"
    raw_segments: list[dict[str, Any]] = list(asr_result.get("segments", []))
    print(f"ASR done. Detected language: {detected_language}. {len(raw_segments)} segments.")
    for seg in raw_segments:
        text = (seg.get("text") or "").strip()
        print(f"{format_timestamp_short(float(seg.get('start') or 0.0))} {text}")

    print("-" * 60)
    print("Aligning to word level...")
    aligned = asr_mod.align_segments(raw_segments, detected_language, audio, device)

    if config.diarization.enabled:
        print("Running speaker diarization (pyannote)...")
        token = diarize_mod.resolve_hf_token(config.diarization.hf_token)
        aligned = diarize_mod.diarize_and_assign(
            aligned,
            audio,
            hf_token=token,
            device=device,
            num_speakers=config.diarization.num_speakers,
            min_speakers=config.diarization.min_speakers,
            max_speakers=config.diarization.max_speakers,
            model_name=config.diarization.model_name,
        )

    aligned_segments = list(aligned.get("segments", []))
    normalized = [_normalize_segment(seg) for seg in aligned_segments]
    end_time = datetime.now()
    processing_seconds = (end_time - start_time).total_seconds()
    print(f"Pipeline done in {processing_seconds:.1f}s.")

    duration_seconds = 0.0
    if normalized:
        duration_seconds = float(normalized[-1].get("end") or 0.0)

    json_payload: dict[str, Any] = {
        "schema_version": 1,
        "source_file": str(input_path),
        "transcribed_at": end_time.isoformat(timespec="seconds"),
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
        },
        "audio_info": {
            "language_detected": detected_language,
            "language_probability": asr_result.get("language_probability"),
            "duration_seconds": duration_seconds,
        },
        "stats": {
            "num_segments": len(normalized),
            "num_speakers": diarize_mod.count_unique_speakers(normalized),
            "processing_seconds": processing_seconds,
        },
        "segments": normalized,
        "full_text": "\n".join(seg["text"] for seg in normalized if seg.get("text")),
    }

    paths = _resolve_output_paths(config)
    written: dict[str, Path] = {}

    if config.output.write_json:
        write_json(json_payload, paths["json"])
        written["json"] = paths["json"]
    if config.output.write_srt:
        write_srt(normalized, paths["srt"])
        written["srt"] = paths["srt"]
    if config.output.write_txt:
        write_txt(normalized, paths["txt"])
        written["txt"] = paths["txt"]
    if config.output.write_md:
        md = render_markdown(
            json_payload,
            speaker_map=config.speaker_map,
            fm_date=config.frontmatter.date,
            tags=config.frontmatter.tags,
            fm_source=config.frontmatter.source,
        )
        write_markdown(md, paths["md"])
        written["md"] = paths["md"]

    print("\nFiles written:")
    for kind, p in written.items():
        print(f"  - {kind:5s}: {p}")
    return written
