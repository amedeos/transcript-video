"""``transcript-voices`` CLI: inspect and manage the voice-print database.

Sub-commands:

- ``list``    — one line per enrolled speaker (default when no sub-command).
- ``show``    — sample-by-sample detail for a named speaker.
- ``forget``  — remove a speaker or specific samples; prompts unless ``--yes``.

Torch-free; reads and writes the local DB only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .speaker_db import (
    all_speaker_names,
    load_db,
    remove_sample,
    remove_speaker,
    resolve_db_path,
    save_db,
    speaker_sample_count,
)
from .utils import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcript-voices",
        description=(
            "Inspect and manage the voice-print database. Reads/writes the local "
            "DB file only — no audio, no GPU, no model download."
        ),
    )

    voice_db_parent = argparse.ArgumentParser(add_help=False)
    voice_db_parent.add_argument(
        "--voice-db",
        default=None,
        help=(
            "Path to the voice DB. Overrides $TRANSCRIPT_VIDEO_VOICE_DB and the "
            "default ~/.local/share/transcript-video/voices.json."
        ),
    )

    subparsers = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    subparsers.add_parser(
        "list",
        parents=[voice_db_parent],
        help="List enrolled speakers with sample counts (default).",
    )

    show = subparsers.add_parser(
        "show",
        parents=[voice_db_parent],
        help="Show sample-by-sample detail for a speaker.",
    )
    show.add_argument("name", help="Speaker name to inspect.")

    forget = subparsers.add_parser(
        "forget",
        parents=[voice_db_parent],
        help="Remove a speaker, or specific samples for a speaker.",
    )
    forget.add_argument("name", help="Speaker name to forget.")
    forget.add_argument(
        "--source",
        default=None,
        help="Limit removal to samples whose source matches this value (e.g. 'ep1_transcript.json').",
    )
    forget.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )

    # Defaults so the no-subcommand path (transcript-voices alone) still has a
    # well-formed Namespace.
    parser.set_defaults(cmd="list", voice_db=None, name=None, source=None, yes=False)
    return parser


def _do_list(db_path: Path, db: dict[str, Any]) -> None:
    print(f"Voice DB: {db_path}")
    names = all_speaker_names(db)
    if not names:
        print("(no speakers enrolled)")
        return
    model = db.get("embedding_model") or "(not set)"
    print(f"Embedding model: {model}")
    print(f"Speakers ({len(names)}):")
    max_name = max(len(n) for n in names)
    for n in names:
        count = speaker_sample_count(db, n)
        plural = "" if count == 1 else "s"
        print(f"  {n:<{max_name}}  {count} sample{plural}")


def _do_show(db_path: Path, db: dict[str, Any], name: str) -> None:
    samples = (db.get("speakers") or {}).get(name, [])
    if not samples:
        print(f"No speaker named '{name}' in {db_path}.")
        return
    print(f"{name}: {len(samples)} sample(s) in {db_path}")
    # Compute a clean, fixed-width layout.
    sources = [str(s.get("source", "?")) for s in samples]
    clusters = [str(s.get("cluster", "?")) for s in samples]
    src_w = max(len("source"), max(len(s) for s in sources))
    clu_w = max(len("cluster"), max(len(c) for c in clusters))
    header = f"  {'source':<{src_w}}  {'cluster':<{clu_w}}  {'duration':>9}  added"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in samples:
        src = str(s.get("source", "?"))
        clu = str(s.get("cluster", "?"))
        dur = float(s.get("duration_s") or 0.0)
        added = str(s.get("added_at", "?"))
        print(f"  {src:<{src_w}}  {clu:<{clu_w}}  {dur:8.1f}s  {added}")


def _confirm(prompt: str) -> bool:
    """Prompt for a y/N answer. Returns True only on explicit 'y'/'yes'."""
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def _do_forget(
    db_path: Path,
    db: dict[str, Any],
    name: str,
    *,
    source: str | None,
    skip_prompt: bool,
) -> None:
    samples = (db.get("speakers") or {}).get(name, [])
    if not samples:
        print(f"No speaker named '{name}' in {db_path}.")
        return

    if source is not None:
        matching = [s for s in samples if s.get("source") == source]
        if not matching:
            print(f"No samples for '{name}' with source '{source}' in {db_path}.")
            return
        action_desc = f"{len(matching)} sample(s) of '{name}' (source: {source})"
    else:
        action_desc = f"all {len(samples)} sample(s) of '{name}'"

    if not skip_prompt:
        print(f"About to remove {action_desc} from {db_path}.")
        if not _confirm("Continue? [y/N]: "):
            print("Aborted.")
            sys.exit(1)

    removed = (
        remove_sample(db, name, source) if source is not None else remove_speaker(db, name)
    )
    save_db(db, db_path)
    print(f"Removed {removed} sample(s) for '{name}'.")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    setup_logging(quiet=False, verbose=False)

    db_path = resolve_db_path(args.voice_db)
    db = load_db(db_path)

    if args.cmd == "list":
        _do_list(db_path, db)
    elif args.cmd == "show":
        _do_show(db_path, db, args.name)
    elif args.cmd == "forget":
        _do_forget(db_path, db, args.name, source=args.source, skip_prompt=args.yes)
    else:  # pragma: no cover — defended by argparse's choices
        parser.error(f"unknown command: {args.cmd!r}")


if __name__ == "__main__":
    main()
