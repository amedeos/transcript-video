"""``transcript-to-md`` CLI: re-render Markdown from a JSON artifact.

This entry point intentionally avoids any heavy imports (no torch, whisperX, or
pyannote): it works on the JSON file produced by ``transcript-from-video``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import project_config, speaker_db
from .markdown import (
    build_effective_speaker_map,
    load_transcript_json,
    render_markdown,
    write_markdown,
)
from .speakers import display_name, resolve_speaker_map
from .stats import compute_speaker_stats
from .utils import format_timestamp_hms, setup_logging, silence_known_noisy_warnings
from .writers import write_srt, write_vtt

logger = logging.getLogger("transcript_video.cli_to_md")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcript-to-md",
        description=(
            "Re-render a human-readable Markdown transcript from the JSON artifact "
            "produced by transcript-from-video. Runs without GPU or ML models."
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
        "--paragraph-chars",
        type=int,
        default=400,
        help=(
            "Break long speaker blocks into paragraphs at sentence boundaries when "
            "the running paragraph exceeds this many characters (default: 400). "
            "Set to 0 to disable splitting (single paragraph per merged block)."
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

    identify_group = parser.add_argument_group("Speaker auto-identification (torch-free)")
    identify_group.add_argument(
        "--identify-speakers",
        action="store_true",
        help=(
            "Auto-match the JSON's cached cluster embeddings against the voice DB "
            "and inject the names into speaker_identities at re-render time. "
            "Requires a schema v2 JSON (cluster embeddings present); no audio, no GPU."
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
    subtitle_group = parser.add_argument_group("Subtitle outputs (torch-free)")
    subtitle_group.add_argument(
        "--srt",
        action="store_true",
        help=(
            "Also write a SubRip (.srt) subtitle file next to the JSON. "
            "Readable by VLC and most desktop players; no GPU needed."
        ),
    )
    subtitle_group.add_argument(
        "--vtt",
        action="store_true",
        help="Also write a WebVTT (.vtt) subtitle file next to the JSON.",
    )
    subtitle_group.add_argument(
        "--subtitle-speakers",
        action="store_true",
        help="Prefix each subtitle cue with the speaker name (applies to --srt and --vtt).",
    )
    parser.add_argument(
        "--mark-suspect",
        action="store_true",
        help=(
            "Prefix each segment flagged as suspect (low ASR confidence or "
            "high silence probability) with `[?]` inline in the Markdown body. "
            "Default off — keeps the rendered transcript clean."
        ),
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-q", "--quiet", action="store_true", help="Suppress info-level output.")
    verbosity.add_argument("-v", "--verbose", action="store_true", help="Enable debug-level output.")

    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to a transcript-video.toml file. If omitted, the tool looks for one "
            "in the current directory and then alongside the JSON."
        ),
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Apply [profiles.NAME] from the config on top of the top-level defaults.",
    )
    return parser


def _build_pre_parser() -> argparse.ArgumentParser:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--profile", default=None)
    pre.add_argument("json_path", nargs="?", default=None)
    return pre


def _apply_config_defaults(parser: argparse.ArgumentParser, cli_defaults: dict) -> tuple[set[str], set[str]]:
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


def _resolve_speaker_stats(transcript: dict) -> dict:
    """Return ``stats.speakers`` from the JSON, recomputing on-the-fly if absent.

    Older JSONs (produced before this feature) do not carry the breakdown, so
    we recompute it from ``segments`` to keep ``--list-speakers`` working.
    """
    cached = (transcript.get("stats") or {}).get("speakers")
    if isinstance(cached, dict) and cached:
        return cached
    return compute_speaker_stats(transcript.get("segments") or [])


def _refresh_auto_identities(
    transcript: dict, *, voice_db: str | None, threshold: float
) -> None:
    """Refresh ``transcript["speaker_identities"]`` from the current DB state.

    Strategy:

    1. Read ``speaker_clusters`` from the JSON; bail out (with a warning) if
       absent — the user needs a v2 JSON with cached embeddings.
    2. Load the DB; bail out gracefully if missing / empty / incompatible.
    3. Run :func:`speaker_db.auto_resolve_speaker_map` against the clusters.
    4. **Replace** the existing auto entries with the fresh matches; **preserve**
       manual entries (``source == "manual"``) that were recorded at transcribe
       time, since the user's explicit assertion shouldn't be undone by today's
       DB state.

    Torch-free: only ``speaker_db`` is touched; no pyannote/torch import.
    """
    clusters = transcript.get("speaker_clusters") or {}
    if not clusters:
        logger.warning(
            "--identify-speakers: this JSON has no cached speaker_clusters. "
            "Re-process the video with transcript-from-video to populate them."
        )
        return

    db_path = speaker_db.resolve_db_path(voice_db)
    try:
        db = speaker_db.load_db(db_path)
    except Exception as e:
        logger.warning("--identify-speakers: could not load voice DB at %s: %s", db_path, e)
        return

    if not db.get("speakers"):
        logger.warning(
            "--identify-speakers: voice DB at %s is empty. Enroll speakers with "
            "transcript-learn first.", db_path,
        )
        return

    first = next(iter(clusters.values()))
    cluster_model = first.get("embedding_model") if isinstance(first, dict) else None
    if cluster_model and not speaker_db.embedding_model_compatible(db, cluster_model):
        logger.warning(
            "--identify-speakers: voice DB at %s uses embedding_model %r; "
            "JSON clusters use %r. Skipping auto-identification.",
            db_path, db.get("embedding_model"), cluster_model,
        )
        return

    auto = speaker_db.auto_resolve_speaker_map(clusters, db, threshold=threshold)

    existing = transcript.get("speaker_identities") or {}
    refreshed: dict = {}
    for label, info in auto.items():
        refreshed[label] = {
            "name": info["name"],
            "score": info["score"],
            "source": "auto",
        }
    # Manual entries from transcribe-time survive; they overwrite any fresh auto.
    for label, entry in existing.items():
        if isinstance(entry, dict) and entry.get("source") == "manual":
            refreshed[label] = entry

    transcript["speaker_identities"] = refreshed
    n_auto = sum(1 for e in refreshed.values() if e.get("source") == "auto")
    logger.info(
        "--identify-speakers: %d/%d cluster(s) auto-identified from %s.",
        n_auto, len(clusters), db_path,
    )


def _format_speaker_overview(transcript: dict, speaker_map: dict[str, str]) -> str:
    """Render a fixed-width table summarizing each speaker."""
    stats = _resolve_speaker_stats(transcript)
    if not stats:
        return "No speakers detected (diarization disabled or no diarization data in JSON)."

    effective_map = build_effective_speaker_map(transcript, speaker_map)
    rows = []
    for label, bucket in stats.items():
        rows.append(
            (
                label,
                display_name(label, effective_map),
                format_timestamp_hms(float(bucket.get("duration_seconds") or 0.0)),
                f"{float(bucket.get('percentage') or 0.0):5.1f}%",
                str(bucket.get("num_turns") or 0),
                str(bucket.get("num_suspect") or 0),
                str(bucket.get("first_text") or ""),
            )
        )

    label_w = max(len("Label"), max(len(r[0]) for r in rows))
    name_w = max(len("Name"), max(len(r[1]) for r in rows))

    header = (
        f"{'Label':<{label_w}}  {'Name':<{name_w}}  "
        f"{'Duration':>8}  {'%':>6}  {'Turns':>5}  {'Suspect':>7}  First words"
    )
    sep = "-" * len(header)
    body = [
        f"{label:<{label_w}}  {name:<{name_w}}  "
        f"{dur:>8}  {pct:>6}  {turns:>5}  {susp:>7}  {first}"
        for (label, name, dur, pct, turns, susp, first) in rows
    ]
    return "\n".join([header, sep, *body])


def main(argv: list[str] | None = None) -> None:
    pre_args, _ = _build_pre_parser().parse_known_args(argv)
    json_input = Path(pre_args.json_path) if pre_args.json_path else None
    cli_defaults, speaker_map_from_config, _ = project_config.resolve(
        explicit=pre_args.config,
        profile=pre_args.profile,
        input_file=json_input,
    )

    parser = _build_parser()
    if cli_defaults:
        _apply_config_defaults(parser, cli_defaults)

    args = parser.parse_args(argv)

    setup_logging(quiet=args.quiet, verbose=args.verbose)
    silence_known_noisy_warnings()

    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    transcript = load_transcript_json(json_path)
    speaker_map = resolve_speaker_map(
        args.speaker_map, args.speaker_map_file, fallback=speaker_map_from_config
    )

    if args.identify_speakers:
        _refresh_auto_identities(
            transcript, voice_db=args.voice_db, threshold=args.id_threshold,
        )

    if args.list_speakers:
        print(_format_speaker_overview(transcript, speaker_map))
        return

    if args.srt or args.vtt:
        speaker_names = (
            build_effective_speaker_map(transcript, speaker_map)
            if args.subtitle_speakers
            else None
        )
        segments = transcript.get("segments") or []
        if args.srt:
            srt_path = json_path.with_suffix(".srt")
            write_srt(segments, srt_path, speaker_names=speaker_names)
            print(f"SubRip written: {srt_path}")
        if args.vtt:
            vtt_path = json_path.with_suffix(".vtt")
            write_vtt(segments, vtt_path, speaker_names=speaker_names)
            print(f"WebVTT written: {vtt_path}")

    md = render_markdown(
        transcript,
        speaker_map=speaker_map,
        fm_date=args.fm_date,
        tags=list(args.tags or []),
        fm_source=args.fm_source,
        merge_gap_seconds=args.merge_gap_seconds,
        mark_suspect=args.mark_suspect,
        paragraph_chars=args.paragraph_chars,
    )

    output_path = Path(args.output) if args.output else json_path.with_suffix(".md")

    write_markdown(md, output_path)
    print(f"Markdown written: {output_path}")


if __name__ == "__main__":
    main()
