"""Speaker label → display-name mapping.

Two input forms are supported:

- Inline string: ``SPEAKER_00=Amedeo,SPEAKER_01=Tizio``
- Sidecar file (YAML or JSON, auto-detected by extension)::

    # YAML
    SPEAKER_00: Amedeo
    SPEAKER_01: Tizio

    // JSON
    { "SPEAKER_00": "Amedeo", "SPEAKER_01": "Tizio" }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def parse_speaker_map_inline(value: str) -> dict[str, str]:
    """Parse ``SPEAKER_00=Name0,SPEAKER_01=Name1`` into a dict.

    Whitespace around labels and names is stripped. Empty entries are skipped.
    Exits with a clear error on malformed input.
    """
    mapping: dict[str, str] = {}
    if not value:
        return mapping

    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            print(
                f"Error: invalid --speaker-map entry '{chunk}'. Expected 'LABEL=Name'.",
                file=sys.stderr,
            )
            sys.exit(1)
        label, name = chunk.split("=", 1)
        label = label.strip()
        name = name.strip()
        if not label or not name:
            print(
                f"Error: invalid --speaker-map entry '{chunk}'. Both label and name required.",
                file=sys.stderr,
            )
            sys.exit(1)
        mapping[label] = name
    return mapping


def parse_speaker_map_file(path: str | Path) -> dict[str, str]:
    """Load a speaker map from a YAML or JSON sidecar file."""
    p = Path(path)
    if not p.exists():
        print(f"Error: speaker map file not found: {p}", file=sys.stderr)
        sys.exit(1)

    suffix = p.suffix.lower()
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        print(f"Error reading speaker map file '{p}': {e}", file=sys.stderr)
        sys.exit(1)

    data: object
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError:
            print(
                "Error: PyYAML is required to read YAML speaker map files. "
                "Install with `pip install pyyaml`.",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            print(f"Error: invalid YAML in {p}: {e}", file=sys.stderr)
            sys.exit(1)
    elif suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"Error: invalid JSON in {p}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(
            f"Error: unsupported speaker map file extension '{suffix}'. "
            "Use .yaml, .yml, or .json.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not isinstance(data, dict):
        print(f"Error: speaker map file '{p}' must contain a mapping.", file=sys.stderr)
        sys.exit(1)

    mapping: dict[str, str] = {}
    for k, v in data.items():
        mapping[str(k)] = str(v)
    return mapping


def resolve_speaker_map(
    inline: str | None,
    file_path: str | None,
    fallback: dict[str, str] | None = None,
) -> dict[str, str]:
    """Resolve a speaker map from CLI inputs.

    Precedence: ``--speaker-map`` > ``--speaker-map-file`` > ``fallback`` > ``{}``.
    The ``fallback`` is the inline map declared in a project config file
    (``transcript-video.toml``'s ``[speaker_map]``).
    """
    if inline:
        return parse_speaker_map_inline(inline)
    if file_path:
        return parse_speaker_map_file(file_path)
    if fallback:
        return dict(fallback)
    return {}


def display_name(label: str, speaker_map: dict[str, str]) -> str:
    """Look up a display name for ``label``; fall back to the label itself."""
    return speaker_map.get(label, label)
