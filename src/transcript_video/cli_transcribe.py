"""``transcribe-video`` CLI: full ASR + alignment + diarization pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DiarizationConfig, FrontmatterConfig, OutputConfig, RunConfig
from .pipeline import run_pipeline
from .speakers import resolve_speaker_map
from .utils import read_text_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe-video",
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

    parser.add_argument("input_file", help="Path to the input video/audio file (typically MP4).")

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

    diar_group = parser.add_argument_group("Diarization")
    diar_group.add_argument("--no-diarize", action="store_true", help="Skip speaker diarization.")
    diar_group.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token; falls back to HF_TOKEN env var, then ~/.cache/huggingface/token.",
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

    out_group = parser.add_argument_group("Outputs")
    out_group.add_argument("--srt", action="store_true", help="Write a SubRip subtitle file.")
    out_group.add_argument("--txt", action="store_true", help="Write a plain-text transcript file.")
    out_group.add_argument("--md", action="store_true", help="Write a human-readable Markdown transcript.")
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

    return parser


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
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = RunConfig(
        input_file=Path(args.input_file),
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
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        ),
        output=OutputConfig(
            write_json=True,
            write_srt=args.srt,
            write_txt=args.txt,
            write_md=args.md,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            basename=args.basename,
        ),
        frontmatter=FrontmatterConfig(
            date=args.fm_date,
            tags=list(args.tags or []),
            source=args.fm_source,
        ),
        speaker_map=resolve_speaker_map(args.speaker_map, args.speaker_map_file),
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()
