"""``transcript-learn`` CLI: append speakers from a transcript JSON to the DB.

Torch-free entry point. Consumes the cached ``speaker_clusters`` embeddings
from the JSON artifact; no audio, no GPU, no model downloads at runtime.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .enrollment import EnrollmentError, learn_from_transcript
from .speaker_db import resolve_db_path
from .speakers import resolve_speaker_map
from .utils import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcript-learn",
        description=(
            "Append speakers from a transcript JSON to the voice-print database. "
            "Uses embeddings that were cached at pipeline time; no audio, GPU, "
            "or model download required."
        ),
    )
    parser.add_argument(
        "json_path",
        help="Path to the *_transcript.json file (schema version >= 2).",
    )
    speaker_group = parser.add_argument_group("speaker mapping")
    speaker_group.add_argument(
        "--speaker-map",
        default=None,
        help='Inline label-to-name map, e.g. "SPEAKER_00=Mario,SPEAKER_01=Luca".',
    )
    speaker_group.add_argument(
        "--speaker-map-file",
        default=None,
        help="YAML or JSON sidecar with the speaker map.",
    )
    parser.add_argument(
        "--voice-db",
        default=None,
        help=(
            "Path to the voice DB. Overrides $TRANSCRIPT_VIDEO_VOICE_DB and the "
            "default ~/.local/share/transcript-video/voices.json."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be added without writing to the DB.",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress info-level output."
    )
    verbosity.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug-level output."
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    setup_logging(quiet=args.quiet, verbose=args.verbose)

    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    speaker_map = resolve_speaker_map(args.speaker_map, args.speaker_map_file)
    if not speaker_map:
        print(
            "Error: no speaker mapping provided. Pass --speaker-map or --speaker-map-file.",
            file=sys.stderr,
        )
        sys.exit(1)

    db_path = resolve_db_path(args.voice_db)

    try:
        result = learn_from_transcript(
            json_path, speaker_map, db_path, dry_run=args.dry_run
        )
    except EnrollmentError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    action = "Would add" if result["dry_run"] else "Added"
    if result["added"]:
        print(f"{action} {len(result['added'])} sample(s) to {result['db_path']}:")
        for label, name in result["added"].items():
            print(f"  {label} -> {name}")
    else:
        print("No samples added (no overlap between speaker-map and cluster labels).")

    if result["skipped_no_cluster"]:
        print(
            f"Skipped (no matching cluster in JSON): "
            f"{', '.join(result['skipped_no_cluster'])}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
