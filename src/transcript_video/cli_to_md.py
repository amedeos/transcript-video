"""``transcript-to-md`` CLI: re-render Markdown from a JSON artifact.

This entry point intentionally avoids any heavy imports (no torch, whisperX, or
pyannote): it works on the JSON file produced by ``transcribe-video``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .markdown import load_transcript_json, render_markdown, write_markdown
from .speakers import resolve_speaker_map


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcript-to-md",
        description=(
            "Re-render a human-readable Markdown transcript from the JSON artifact "
            "produced by transcribe-video. Runs without GPU or ML models."
        ),
    )

    parser.add_argument("json_path", help="Path to the *_transcript.json file.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output Markdown path (default: <stem>.md next to the JSON).",
    )
    parser.add_argument(
        "--speaker-map",
        default=None,
        help='Inline label-to-name map, e.g. "SPEAKER_00=Amedeo,SPEAKER_01=Tizio".',
    )
    parser.add_argument(
        "--speaker-map-file",
        default=None,
        help="YAML or JSON sidecar with the speaker map.",
    )
    parser.add_argument(
        "--date",
        dest="fm_date",
        default=None,
        help="Frontmatter date in YYYY-MM-DD format (default: today's date).",
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        default=[],
        help="Frontmatter tag (repeatable).",
    )
    parser.add_argument(
        "--source",
        dest="fm_source",
        default=None,
        help="Override the frontmatter 'source' field.",
    )
    parser.add_argument(
        "--merge-gap-seconds",
        type=float,
        default=1.5,
        help=(
            "Merge consecutive same-speaker segments whose silent gap is at most this many "
            "seconds (default: 1.5). Set to 0 to disable merging."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    transcript = load_transcript_json(json_path)
    speaker_map = resolve_speaker_map(args.speaker_map, args.speaker_map_file)

    md = render_markdown(
        transcript,
        speaker_map=speaker_map,
        fm_date=args.fm_date,
        tags=list(args.tags or []),
        fm_source=args.fm_source,
        merge_gap_seconds=args.merge_gap_seconds,
    )

    output_path = Path(args.output) if args.output else json_path.with_suffix(".md")

    write_markdown(md, output_path)
    print(f"Markdown written: {output_path}")


if __name__ == "__main__":
    main()
