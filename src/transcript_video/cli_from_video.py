"""``transcript-from-video`` CLI: full ASR + alignment + diarization pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import project_config
from .config import (
    DiarizationConfig,
    FrontmatterConfig,
    IdentifyConfig,
    OutputConfig,
    RunConfig,
)
from .pipeline import run_pipeline
from .preflight import report_results, run_preflight
from .speakers import resolve_speaker_map
from .utils import read_text_file, setup_logging, silence_known_noisy_warnings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcript-from-video",
        description=(
            "Transcribe and diarize a video/audio file using whisperX "
            "(faster-whisper backend + pyannote speaker diarization). "
            "JSON is always written; SRT, TXT, and Markdown are opt-in."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4
  %(prog)s video.mp4 --beam_size 10 --srt
  %(prog)s video.mp4 --language en --txt --md
  %(prog)s video.mp4 --prompt "Glossary: API, GPU, microservices."
  %(prog)s video.mp4 --hotwords "Anthropic Claude faster-whisper"
  %(prog)s video.mp4 --anti-loop
  %(prog)s meeting.mp4 --md --speaker-map "SPEAKER_00=Amedeo,SPEAKER_01=Tizio" --tag openshift
        """,
    )

    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help=(
            "Path to the input video/audio file (typically MP4). Optional with --check, or "
            "with --resume-from-aligned (in which case the source file is taken from the "
            "aligned snapshot)."
        ),
    )

    model_group = parser.add_argument_group("Model / device")
    model_group.add_argument("--model", default="large-v3", help="whisperX model name (default: large-v3).")
    model_group.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Beam search width (default: 5).",
    )
    model_group.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to run on (default: auto-detect).",
    )
    model_group.add_argument(
        "--compute-type",
        default=None,
        help="Override compute type (e.g. float16, int8, int8_float16). Default: auto.",
    )

    parser.add_argument(
        "--language",
        default=None,
        help="Force a language code (e.g. it, en, de, fr, es). Default: autodetect.",
    )

    prompt_group = parser.add_argument_group("Initial prompt (mutually exclusive, opt-in)").add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default=None, help="Inline initial-prompt text (~224 token cap).")
    prompt_group.add_argument("--prompt-file", default=None, help="Read initial prompt from a UTF-8 file.")
    prompt_group.add_argument("--no-prompt", action="store_true", help="Explicitly disable the initial prompt.")

    hotwords_group = parser.add_argument_group("Hotwords (mutually exclusive, opt-in)").add_mutually_exclusive_group()
    hotwords_group.add_argument("--hotwords", default=None, help="Inline space-separated hotwords.")
    hotwords_group.add_argument("--hotwords-file", default=None, help="Read hotwords from a UTF-8 file.")
    hotwords_group.add_argument("--no-hotwords", action="store_true", help="Explicitly disable hotwords.")

    parser.add_argument(
        "--anti-loop",
        action="store_true",
        help=(
            "Mitigate Whisper's cyclic hallucinations: condition_on_previous_text=False, "
            "compression_ratio_threshold=2.0, no_speech_threshold=0.5."
        ),
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress info-level output; only warnings and errors.",
    )
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level output (per-segment ASR progress).",
    )

    resume_group = parser.add_argument_group("Resume / cache")
    resume_group.add_argument(
        "--resume-from-aligned",
        default=None,
        help=(
            "Path to a previously-written *_transcript.aligned.json. Skips ASR + alignment "
            "and jumps straight to diarization. The source video is taken from the snapshot "
            "(or override by passing input_file)."
        ),
    )

    preflight_group = parser.add_argument_group("Pre-flight checks")
    preflight_group.add_argument(
        "--check",
        action="store_true",
        help=(
            "Run pre-flight checks (ffmpeg, CUDA, HF token, gated-model access, whisperX API) "
            "and exit. Useful before paying for a long ASR run."
        ),
    )
    preflight_group.add_argument(
        "--no-check",
        action="store_true",
        help="Skip the default pre-flight that runs at the start of every transcription.",
    )
    preflight_group.add_argument(
        "--offline-check",
        action="store_true",
        help=(
            "Skip the network-dependent parts of the pre-flight (HF token validity and "
            "gated-model access). Implied when no internet is available."
        ),
    )

    diar_group = parser.add_argument_group("Diarization")
    diar_group.add_argument("--no-diarize", action="store_true", help="Skip speaker diarization.")
    diar_group.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token; falls back to HF_TOKEN env var, then ~/.cache/huggingface/token.",
    )
    diar_group.add_argument(
        "--diarize-model",
        default=None,
        help=(
            "pyannote model id (default: whisperX's built-in, currently "
            "pyannote/speaker-diarization-community-1). Common alternative: "
            "pyannote/speaker-diarization-3.1. Both are gated repos and require "
            "accepting their terms on huggingface.co."
        ),
    )
    diar_group.add_argument("--num-speakers", type=int, default=None, help="Exact number of speakers.")
    diar_group.add_argument("--min-speakers", type=int, default=None, help="Lower bound for speaker count.")
    diar_group.add_argument("--max-speakers", type=int, default=None, help="Upper bound for speaker count.")
    diar_group.add_argument(
        "--speaker-map",
        default=None,
        help='Inline label-to-name map, e.g. "SPEAKER_00=Amedeo,SPEAKER_01=Tizio".',
    )
    diar_group.add_argument(
        "--speaker-map-file",
        default=None,
        help="YAML or JSON sidecar with the speaker map.",
    )

    identify_group = parser.add_argument_group("Speaker auto-identification")
    identify_group.add_argument(
        "--identify-speakers",
        action="store_true",
        help=(
            "After diarization, auto-match each cluster against the voice-print DB "
            "(enroll speakers first with transcript-learn). Results are written to "
            "the JSON's speaker_identities field; --speaker-map still wins per label."
        ),
    )
    identify_group.add_argument(
        "--voice-db",
        default=None,
        help=(
            "Path to the voice DB used by --identify-speakers. Overrides "
            "$TRANSCRIPT_VIDEO_VOICE_DB and the XDG default."
        ),
    )
    identify_group.add_argument(
        "--id-threshold",
        type=float,
        default=0.65,
        help=(
            "Cosine-similarity threshold for auto-identification (default: 0.65). "
            "Higher = fewer false positives; lower = more matches."
        ),
    )

    out_group = parser.add_argument_group("Outputs")
    out_group.add_argument("--srt", action="store_true", help="Write a SubRip subtitle file.")
    out_group.add_argument("--vtt", action="store_true", help="Write a WebVTT subtitle file.")
    out_group.add_argument("--txt", action="store_true", help="Write a plain-text transcript file.")
    out_group.add_argument("--md", action="store_true", help="Write a human-readable Markdown transcript.")
    out_group.add_argument(
        "--subtitle-speakers",
        action="store_true",
        help="Prefix each subtitle cue with the speaker name (applies to --srt and --vtt).",
    )
    out_group.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files (default: same directory as the input).",
    )
    out_group.add_argument(
        "--basename",
        default=None,
        help="Base filename (without extension) for outputs (default: stem of the input).",
    )

    fm_group = parser.add_argument_group("Markdown frontmatter (only used with --md)")
    fm_group.add_argument(
        "--date",
        dest="fm_date",
        default=None,
        help="Frontmatter date in YYYY-MM-DD format (default: today's date).",
    )
    fm_group.add_argument(
        "--tag",
        action="append",
        dest="tags",
        default=[],
        help="Frontmatter tag (repeatable). Collected into the YAML 'tags' list.",
    )
    fm_group.add_argument(
        "--source",
        dest="fm_source",
        default=None,
        help="Override the frontmatter 'source' field (default: input file basename).",
    )

    config_group = parser.add_argument_group("Project config")
    config_group.add_argument(
        "--config",
        default=None,
        help=(
            "Path to a transcript-video.toml config file. If omitted, the tool "
            "looks for one in the current directory and then alongside the input video."
        ),
    )
    config_group.add_argument(
        "--profile",
        default=None,
        help="Apply [profiles.NAME] from the config on top of the top-level defaults.",
    )

    return parser


def _build_pre_parser() -> argparse.ArgumentParser:
    """Mini parser for the --config / --profile / input_file early lookup.

    Used so the project config file can be resolved BEFORE the main parser
    decides on defaults; without this, --profile would be applied too late.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--profile", default=None)
    pre.add_argument("input_file", nargs="?", default=None)
    return pre


def _apply_config_defaults(
    parser: argparse.ArgumentParser, cli_defaults: dict
) -> tuple[set[str], set[str]]:
    """Filter the config's flat dict to keys recognized by the main parser.

    Returns ``(applied, ignored)`` for diagnostics. Also calls
    ``parser.set_defaults`` for the applied keys.
    """
    known = {a.dest for a in parser._actions if a.dest != "help"}
    applied: dict = {}
    ignored: set[str] = set()
    for key, value in cli_defaults.items():
        if key in known:
            applied[key] = value
        else:
            ignored.add(key)
    if applied:
        parser.set_defaults(**applied)
    return set(applied.keys()), ignored


def _resolve_prompt(args: argparse.Namespace) -> str | None:
    if args.no_prompt:
        return ""
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        return read_text_file(args.prompt_file, "prompt")
    return None


def _resolve_hotwords(args: argparse.Namespace) -> str | None:
    if args.no_hotwords:
        return ""
    if args.hotwords is not None:
        return args.hotwords
    if args.hotwords_file is not None:
        return read_text_file(args.hotwords_file, "hotwords")
    return None


def main(argv: list[str] | None = None) -> None:
    # Pre-parse to find --config / --profile / input_file so we can locate the
    # config file BEFORE the main parser computes defaults.
    pre_args, _ = _build_pre_parser().parse_known_args(argv)
    pre_input_path: Path | None = Path(pre_args.input_file) if pre_args.input_file else None
    cli_defaults, speaker_map_from_config, config_path = project_config.resolve(
        explicit=pre_args.config,
        profile=pre_args.profile,
        input_file=pre_input_path,
    )

    parser = _build_parser()
    applied: set[str] = set()
    ignored: set[str] = set()
    if cli_defaults:
        applied, ignored = _apply_config_defaults(parser, cli_defaults)

    args = parser.parse_args(argv)

    setup_logging(quiet=args.quiet, verbose=args.verbose)
    silence_known_noisy_warnings()

    if config_path:
        import logging as _logging

        _logging.getLogger("transcript_video.cli").info(
            "Loaded config %s%s%s",
            config_path,
            f" (profile={pre_args.profile})" if pre_args.profile else "",
            f"; applied: {sorted(applied)}" if applied else "",
        )
        if ignored:
            _logging.getLogger("transcript_video.cli").warning(
                "Config keys ignored (no matching CLI flag): %s", sorted(ignored)
            )

    # Resolve the input file: from CLI, or from the aligned snapshot when resuming.
    input_file: Path | None = Path(args.input_file) if args.input_file else None
    resume_path: Path | None = Path(args.resume_from_aligned) if args.resume_from_aligned else None

    if input_file is None and resume_path is not None:
        # Pull source_file from the snapshot.
        import json as _json

        try:
            with open(resume_path, encoding="utf-8") as _f:
                _snap = _json.load(_f)
            source = _snap.get("source_file")
            if source:
                input_file = Path(source)
        except OSError as e:
            parser.error(f"could not read --resume-from-aligned snapshot: {e}")

    if not args.check and input_file is None:
        parser.error("input_file is required (omit only with --check, or with --resume-from-aligned and a snapshot that records source_file)")

    config = RunConfig(
        # When --check is the only thing requested, input_file may be missing;
        # the placeholder is never read because we exit before run_pipeline.
        input_file=input_file if input_file else Path("/dev/null"),
        model=args.model,
        beam_size=args.beam_size,
        device=None if args.device == "auto" else args.device,
        compute_type=args.compute_type,
        language=args.language,
        initial_prompt=_resolve_prompt(args),
        hotwords=_resolve_hotwords(args),
        anti_loop=args.anti_loop,
        diarization=DiarizationConfig(
            enabled=not args.no_diarize,
            hf_token=args.hf_token,
            model_name=args.diarize_model,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        ),
        output=OutputConfig(
            write_json=True,
            write_srt=args.srt,
            write_vtt=args.vtt,
            write_txt=args.txt,
            write_md=args.md,
            subtitle_speakers=args.subtitle_speakers,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            basename=args.basename,
        ),
        frontmatter=FrontmatterConfig(
            date=args.fm_date,
            tags=list(args.tags or []),
            source=args.fm_source,
        ),
        identify=IdentifyConfig(
            enabled=args.identify_speakers,
            voice_db=Path(args.voice_db) if args.voice_db else None,
            threshold=args.id_threshold,
        ),
        speaker_map=resolve_speaker_map(
            args.speaker_map, args.speaker_map_file, fallback=speaker_map_from_config
        ),
        resume_from_aligned=resume_path,
    )

    # --identify-speakers needs diarized clusters to match against.
    if args.identify_speakers and args.no_diarize:
        print(
            "Warning: --identify-speakers has no effect with --no-diarize "
            "(no clusters to identify). Continuing without auto-identification.",
            file=sys.stderr,
        )
        config.identify.enabled = False

    # --check: run only the pre-flight (with network) and exit 0/1.
    if args.check:
        results = run_preflight(config, network=not args.offline_check)
        ok = report_results(results)
        sys.exit(0 if ok else 1)

    # Default pre-flight on every run, unless --no-check.
    if not args.no_check:
        results = run_preflight(config, network=not args.offline_check)
        if not report_results(results):
            print(
                "\nPre-flight failed. Fix the issues above or pass --no-check to skip "
                "(at your own risk).",
                file=sys.stderr,
            )
            sys.exit(1)

    run_pipeline(config)


if __name__ == "__main__":
    main()
