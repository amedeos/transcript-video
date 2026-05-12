"""Render a human-readable Markdown transcript from the canonical JSON artifact.

This module is intentionally lightweight: it depends only on the standard library
and (for the speaker map sidecar parsing in :mod:`speakers`) PyYAML. It is the
sole module imported by ``transcript-to-md`` so re-rendering can run without
torch / whisperX / pyannote installed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

from .speakers import display_name
from .utils import format_timestamp_hms

UNKNOWN_SPEAKER = "Unknown"


def load_transcript_json(json_path: str | Path) -> dict[str, Any]:
    """Load the canonical JSON artifact from disk."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def build_effective_speaker_map(
    transcript: dict[str, Any], cli_speaker_map: dict[str, str] | None
) -> dict[str, str]:
    """Merge JSON's ``speaker_identities`` (fallback) with the CLI map (wins).

    Schema-v2 transcripts may carry a ``speaker_identities`` field populated
    by ``transcript-from-video --identify-speakers`` (or by an enrollment
    step). Each entry is ``{name, score, source: "auto" | "manual"}``. The
    rendering layer treats those names as a fallback: if the user passes
    ``--speaker-map SPEAKER_00=Luca`` at render time, "Luca" still wins.

    Returns a flat ``{label: name}`` dict ready for the rendering pipeline.
    Malformed identity entries (missing ``name``, not a dict) are skipped
    silently — the rest of the map continues to work.
    """
    identities = transcript.get("speaker_identities") or {}
    merged: dict[str, str] = {}
    for label, info in identities.items():
        if not isinstance(info, dict):
            continue
        name = info.get("name")
        if name:
            merged[str(label)] = str(name)
    if cli_speaker_map:
        merged.update(cli_speaker_map)
    return merged


def _yaml_quote(value: str) -> str:
    """Quote a value as a YAML double-quoted scalar (escapes ``\\`` and ``"``)."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _yaml_inline_value(value: str) -> str:
    """Render a YAML scalar inline.

    Strings that are safe (no leading/trailing whitespace, no special chars,
    not a YAML reserved word) are emitted bare. Everything else is double-quoted.
    """
    if not value:
        return '""'
    reserved = {"true", "false", "null", "yes", "no", "on", "off", "~"}
    needs_quoting = (
        value.lower() in reserved
        or value != value.strip()
        or any(ch in value for ch in ":#&*!|>'\"%@`,[]{}\n\r\t")
    )
    # Numbers should also be quoted to keep them as strings.
    try:
        float(value)
        needs_quoting = True
    except ValueError:
        pass
    if needs_quoting:
        return _yaml_quote(value)
    return value


def _render_tags(tags: Iterable[str]) -> str:
    items = [t.strip() for t in tags if t and t.strip()]
    if not items:
        return "[]"
    return "[" + ", ".join(_yaml_inline_value(t) for t in items) + "]"


def _build_asr_label(parameters: dict[str, Any]) -> str:
    """Build the ``asr:`` frontmatter field (e.g. ``whisperx-large-v3``)."""
    backend = parameters.get("backend", "whisperx")
    model = parameters.get("model", "")
    return f"{backend}-{model}" if model else str(backend)


def _resolved_language(parameters: dict[str, Any], audio_info: dict[str, Any]) -> str:
    forced = parameters.get("language_forced")
    if forced:
        return str(forced)
    detected = audio_info.get("language_detected")
    return str(detected) if detected else "unknown"


def _collect_speakers(segments: list[dict[str, Any]]) -> list[str]:
    """Return distinct speaker labels in the order they first appear."""
    seen: list[str] = []
    for seg in segments:
        label = seg.get("speaker")
        if not label:
            continue
        if label not in seen:
            seen.append(label)
    return seen


def render_frontmatter(
    transcript: dict[str, Any],
    *,
    speaker_map: dict[str, str],
    fm_date: str | None,
    tags: list[str],
    fm_source: str | None,
) -> str:
    """Render the YAML frontmatter block (including the surrounding ``---`` lines)."""
    parameters = transcript.get("parameters", {}) or {}
    audio_info = transcript.get("audio_info", {}) or {}
    segments = transcript.get("segments", []) or []

    duration_s = float(audio_info.get("duration_seconds") or 0.0)
    language = _resolved_language(parameters, audio_info)
    asr_label = _build_asr_label(parameters)
    beam_size = parameters.get("beam_size")

    if fm_source:
        source = fm_source
    else:
        source_path = transcript.get("source_file") or ""
        source = Path(str(source_path)).name or str(source_path)

    fm_date_value = fm_date or date.today().isoformat()

    speakers_seen = _collect_speakers(segments)
    if not speakers_seen:
        speakers_seen = [UNKNOWN_SPEAKER]

    lines = ["---"]
    lines.append(f"date: {fm_date_value}")
    lines.append(f'duration: "{format_timestamp_hms(duration_s)}"')
    lines.append(f"language: {_yaml_inline_value(language)}")
    lines.append(f"source: {_yaml_inline_value(source)}")
    lines.append(f"asr: {_yaml_inline_value(asr_label)}")
    if beam_size is not None:
        lines.append(f"beam_size: {beam_size}")
    lines.append("speakers:")
    for label in speakers_seen:
        name = display_name(label, speaker_map)
        lines.append(f"  {label}: {_yaml_inline_value(name)}")
    lines.append(f"tags: {_render_tags(tags)}")
    lines.append("---")
    return "\n".join(lines) + "\n"


SUSPECT_MARKER = "[?]"

# Sentence boundaries: split on whitespace following ``.``, ``!``, or ``?``.
# Italian-friendly enough; minor false positives on abbreviations like ``es.``
# are acceptable given the typical 400-char paragraph budget.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def split_into_paragraphs(text: str, max_chars: int) -> list[str]:
    """Split a long block of speaker text into paragraphs at sentence boundaries.

    The algorithm packs whole sentences into paragraphs, starting a new
    paragraph as soon as adding the next sentence would push the running
    paragraph past ``max_chars``. The first sentence of each paragraph is
    always kept whole — a single 600-char sentence stays on its own line
    rather than being mid-cut.

    ``max_chars <= 0`` disables splitting (returns the input as a single-element
    list), which is the escape hatch for users who prefer the legacy single-
    paragraph rendering.
    """
    text = text.strip()
    if not text or max_chars <= 0:
        return [text] if text else []

    sentences = [s.strip() for s in _SENTENCE_BOUNDARY_RE.split(text) if s.strip()]
    if not sentences:
        return [text]

    paragraphs: list[str] = []
    current: list[str] = []
    current_len = 0
    for s in sentences:
        if current and current_len + len(s) > max_chars:
            paragraphs.append(" ".join(current))
            current = [s]
            current_len = len(s)
        else:
            current.append(s)
            current_len += len(s) + 1  # +1 accounts for the joining space.
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def _group_segments(
    segments: list[dict[str, Any]],
    merge_gap_seconds: float,
    *,
    mark_suspect: bool = False,
) -> list[dict[str, Any]]:
    """Merge consecutive same-speaker segments separated by <= ``merge_gap_seconds``.

    Each output block is a dict with ``start``, ``end``, ``speaker`` (raw label),
    and ``text``. Segments with no speaker label are bucketed under
    :data:`UNKNOWN_SPEAKER`.

    When ``mark_suspect=True``, each segment flagged ``suspect: true`` is
    prefixed with ``[?]`` *inline* (preserving its position within the merged
    block) instead of breaking the block. This keeps the speaker's turn
    coherent while making the unreliable spans visible at a glance.
    """
    blocks: list[dict[str, Any]] = []
    for seg in segments:
        speaker = seg.get("speaker") or UNKNOWN_SPEAKER
        raw_text = (seg.get("text") or "").strip()
        if not raw_text:
            continue
        text = f"{SUSPECT_MARKER} {raw_text}" if mark_suspect and seg.get("suspect") else raw_text
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)

        if blocks and merge_gap_seconds > 0:
            prev = blocks[-1]
            gap = start - prev["end"]
            if prev["speaker"] == speaker and gap <= merge_gap_seconds:
                prev["text"] = f"{prev['text']} {text}".strip()
                prev["end"] = end
                continue

        blocks.append({"start": start, "end": end, "speaker": speaker, "text": text})
    return blocks


def render_body(
    segments: list[dict[str, Any]],
    *,
    speaker_map: dict[str, str],
    merge_gap_seconds: float = 1.5,
    mark_suspect: bool = False,
    paragraph_chars: int = 400,
) -> str:
    """Render the speaker-labeled Markdown body.

    ``paragraph_chars`` controls how aggressively long speaker blocks are
    broken into paragraphs at sentence boundaries: a new paragraph starts
    as soon as the running paragraph would exceed this many characters.
    Pass ``0`` to disable splitting (legacy single-paragraph behavior).
    """
    blocks = _group_segments(segments, merge_gap_seconds, mark_suspect=mark_suspect)
    if not blocks:
        return ""

    parts: list[str] = []
    for block in blocks:
        timestamp = format_timestamp_hms(block["start"])
        name = display_name(block["speaker"], speaker_map)
        parts.append(f"## [{timestamp}] {name}")
        paragraphs = split_into_paragraphs(block["text"], paragraph_chars)
        parts.append("\n\n".join(paragraphs) if paragraphs else block["text"])
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def render_markdown(
    transcript: dict[str, Any],
    *,
    speaker_map: dict[str, str] | None = None,
    fm_date: str | None = None,
    tags: list[str] | None = None,
    fm_source: str | None = None,
    merge_gap_seconds: float = 1.5,
    mark_suspect: bool = False,
    paragraph_chars: int = 400,
) -> str:
    """Render the full Markdown document (frontmatter + body).

    The effective speaker map is the union of the transcript's
    ``speaker_identities`` (fallback) and ``speaker_map`` (wins). The CLI
    map therefore retains its final-authority semantics it has always had.
    """
    effective_map = build_effective_speaker_map(transcript, speaker_map)
    tags = tags or []
    segments = transcript.get("segments", []) or []
    fm = render_frontmatter(
        transcript,
        speaker_map=effective_map,
        fm_date=fm_date,
        tags=tags,
        fm_source=fm_source,
    )
    body = render_body(
        segments,
        speaker_map=effective_map,
        merge_gap_seconds=merge_gap_seconds,
        mark_suspect=mark_suspect,
        paragraph_chars=paragraph_chars,
    )
    return f"{fm}\n{body}" if body else fm


def write_markdown(content: str, output_path: Path) -> None:
    """Write the rendered Markdown to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
