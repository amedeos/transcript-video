"""Project-level configuration via ``transcript-video.toml``.

A simple TOML file lets you persist the per-project flags you would otherwise
retype on every invocation (beam size, hotwords, anti-loop, speaker map, tags,
...). Profiles allow swapping whole bundles of settings, e.g. ``[profiles.meeting]``
vs ``[profiles.podcast]``.

Resolution order, when ``--config`` is not passed explicitly:

1. ``./transcript-video.toml`` (current working directory)
2. ``<input_file_dir>/transcript-video.toml`` (alongside the video, if a positional
   input was provided)
3. None — use CLI defaults / explicit flags only.

Schema (all keys optional)::

    # Top-level keys map directly to argparse ``dest`` names. Anything you can
    # pass on the command line as `--foo-bar VALUE` is settable here as `foo_bar`.
    beam_size      = 10
    anti_loop      = true
    diarize_model  = "pyannote/speaker-diarization-3.1"
    min_speakers   = 2
    max_speakers   = 6
    hotwords       = "OpenShift Cgroups Kubernetes"
    tags           = ["openshift", "cgroups-v2"]
    paragraph_chars = 400

    # Optional inline speaker-name map. CLI --speaker-map still wins.
    [speaker_map]
    SPEAKER_00 = "Amedeo"
    SPEAKER_01 = "Marco"

    # Profiles overlay on top of the top-level defaults when --profile is set.
    [profiles.meeting]
    anti_loop      = true
    min_speakers   = 2
    max_speakers   = 8

    [profiles.podcast]
    beam_size      = 12
    anti_loop      = true

CLI flags always win over the config file. The config provides defaults; it
never restricts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:  # Python 3.11+ has tomllib in the stdlib.
    import tomllib as _tomllib
except ImportError:  # pragma: no cover — only triggered on Python 3.10.
    try:
        import tomli as _tomllib  # type: ignore[no-redef]
    except ImportError:
        _tomllib = None  # type: ignore[assignment]

CONFIG_FILENAME = "transcript-video.toml"
SPEAKER_MAP_KEY = "speaker_map"
PROFILES_KEY = "profiles"


def _load_toml(path: Path) -> dict[str, Any]:
    if _tomllib is None:
        print(
            "Error: tomllib/tomli is required to read a TOML config but neither is "
            "available. Install `tomli` (`pip install tomli`) or upgrade to Python "
            ">=3.11.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        with open(path, "rb") as f:
            return _tomllib.load(f)
    except OSError as e:
        print(f"Error reading config file '{path}': {e}", file=sys.stderr)
        sys.exit(1)
    except _tomllib.TOMLDecodeError as e:
        print(f"Error: invalid TOML in '{path}': {e}", file=sys.stderr)
        sys.exit(1)


def find_config_file(
    explicit: str | Path | None,
    *,
    input_file: Path | None = None,
    cwd: Path | None = None,
) -> Path | None:
    """Locate a config file using the documented resolution order.

    ``explicit`` takes precedence; missing-file is an error in that case.
    Otherwise we look in ``cwd`` (default: actual cwd) then alongside ``input_file``.
    Returns ``None`` if no config is found.
    """
    if explicit:
        p = Path(explicit)
        if not p.exists():
            print(f"Error: --config file not found: {p}", file=sys.stderr)
            sys.exit(1)
        return p

    cwd = cwd or Path.cwd()
    candidate = cwd / CONFIG_FILENAME
    if candidate.exists():
        return candidate

    if input_file is not None:
        candidate = input_file.resolve().parent / CONFIG_FILENAME
        if candidate.exists():
            return candidate

    return None


def load_config(path: Path) -> dict[str, Any]:
    """Read and validate a config file. Returns the parsed TOML as a dict."""
    return _load_toml(path)


def _split_special_keys(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Separate ``[speaker_map]`` and ``[profiles]`` from the flat defaults."""
    flat: dict[str, Any] = {}
    speaker_map: dict[str, str] = {}
    for key, value in data.items():
        if key == SPEAKER_MAP_KEY and isinstance(value, dict):
            speaker_map = {str(k): str(v) for k, v in value.items()}
        elif key == PROFILES_KEY:
            continue  # handled separately by ``apply_profile``
        else:
            flat[key] = value
    return flat, speaker_map


def apply_profile(
    data: dict[str, Any], profile: str | None
) -> tuple[dict[str, Any], dict[str, str]]:
    """Merge the named profile (if any) on top of the top-level defaults.

    Returns ``(cli_defaults, speaker_map)`` ready to feed into argparse via
    ``parser.set_defaults(**cli_defaults)`` plus the inline speaker map.
    """
    flat, speaker_map = _split_special_keys(data)

    if profile:
        profiles = data.get(PROFILES_KEY) or {}
        if profile not in profiles:
            available = ", ".join(sorted(profiles.keys())) or "(none defined)"
            print(
                f"Error: profile '{profile}' not found in config. Available: {available}",
                file=sys.stderr,
            )
            sys.exit(1)
        profile_data = profiles[profile]
        if not isinstance(profile_data, dict):
            print(f"Error: [profiles.{profile}] must be a table.", file=sys.stderr)
            sys.exit(1)
        # Profile overrides flat defaults; speaker_map inside a profile also wins.
        for key, value in profile_data.items():
            if key == SPEAKER_MAP_KEY and isinstance(value, dict):
                speaker_map = {str(k): str(v) for k, v in value.items()}
            else:
                flat[key] = value

    return flat, speaker_map


def resolve(
    *,
    explicit: str | Path | None,
    profile: str | None,
    input_file: Path | None = None,
    cwd: Path | None = None,
) -> tuple[dict[str, Any], dict[str, str], Path | None]:
    """High-level helper used by both CLIs.

    Returns ``(cli_defaults, speaker_map_from_config, path_used)``. When no
    config is found and none was requested, returns ``({}, {}, None)``.
    """
    path = find_config_file(explicit, input_file=input_file, cwd=cwd)
    if path is None:
        return {}, {}, None
    data = load_config(path)
    cli_defaults, speaker_map = apply_profile(data, profile)
    return cli_defaults, speaker_map, path
