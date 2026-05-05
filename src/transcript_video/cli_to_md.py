"""``transcript-to-md`` CLI: re-render Markdown from a JSON artifact.

This entry point intentionally avoids any heavy imports (no torch, whisperX, or
pyannote): it works on the JSON file produced by ``transcribe-video``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .markdown import load_transcript_json, render_markdown, write_markdown
from .speakers import display_name, resolve_speaker_map
from .stats import compute_speaker_stats
from .utils import format_timestamp_hms, setup_logging


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
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help=(
            "Print a one-line summary per speaker (duration, percentage, turn count, first words) "
            "and exit without writing Markdown. Useful when figuring out who is who before "
            "filling in --speaker-map."
        ),
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-q", "--quiet", action="store_true", help="Suppress info-level output.")
    verbosity.add_argument("-v", "--verbose", action="store_true", help="Enable debug-level output.")
    return parser


def _resolve_speaker_stats(transcript: dict) -> dict:
    """Return ``stats.speakers`` from the JSON, recomputing on-the-fly if absent.

    Older JSONs (produced before this feature) do not carry the breakdown, so
    we recompute it from ``segments`` to keep ``--list-speakers`` working.
    """
    cached = (transcript.get("stats") or {}).get("speakers")
    if isinstance(cached, dict) and cached:
        return cached
    return compute_speaker_stats(transcript.get("segments") or [])


def _format_speaker_overview(transcript: dict, speaker_map: dict[str, str]) -> str:
    """Render a fixed-width table summarizing each speaker."""
    stats = _resolve_speaker_stats(transcript)
    if not stats:
        return "No speakers detected (diarization disabled or no diarization data in JSON)."

    rows = []
    for label, bucket in stats.items():
        rows.append(
            (
                label,
                display_name(label, speaker_map),
                format_timestamp_hms(float(bucket.get("duration_seconds") or 0.0)),
                f"{float(bucket.get('percentage') or 0.0):5.1f}%",
                str(bucket.get("num_turns") or 0),
                str(bucket.get("first_text") or ""),
            )
        )

    label_w = max(len("Label"), max(len(r[0]) for r in rows))
    name_w = max(len("Name"), max(len(r[1]) for r in rows))

    header = (
        f"{'Label':<{label_w}}  {'Name':<{name_w}}  {'Duration':>8}  {'%':>6}  {'Turns':>5}  First words"
    )
    sep = "-" * len(header)
    body = [
        f"{label:<{label_w}}  {name:<{name_w}}  {dur:>8}  {pct:>6}  {turns:>5}  {first}"
        for (label, name, dur, pct, turns, first) in rows
    ]
    return "\n".join([header, sep, *body])


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    setup_logging(quiet=args.quiet, verbose=args.verbose)

    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    transcript = load_transcript_json(json_path)
    speaker_map = resolve_speaker_map(args.speaker_map, args.speaker_map_file)

    if args.list_speakers:
        print(_format_speaker_overview(transcript, speaker_map))
        return

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
