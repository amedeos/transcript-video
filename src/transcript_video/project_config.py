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
    # beam_size, anti_loop, hotwords, prompt and language are whisperX options:
    # with Parakeet (the default --model auto picks it on CPU) they stop the
    # run, so a config that sets them also pins the whisperX model.
    model          = "large-v3"
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

import argparse
import datetime as _dt
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


class ConfigValueError(ValueError):
    """A config value that doesn't fit the CLI flag it sets."""


def _coerce_scalar(action: argparse.Action, value: Any) -> Any:
    """Convert one config value the way argparse converts a CLI string.

    TOML already gives typed values, so the rules are: an ``int`` flag takes an
    integer (or a numeric string), never a float or a boolean; a ``float``
    flag takes any number but not a boolean; a plain string flag takes a
    string, or a TOML date/time (written unquoted), rendered as ISO text.
    """
    conv = action.type
    if isinstance(value, bool):
        raise ConfigValueError("must not be a boolean")
    if conv is int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
        raise ConfigValueError("must be an integer")
    if conv is float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        raise ConfigValueError("must be a number")
    if conv is None:
        if isinstance(value, str):
            return value
        if isinstance(value, (_dt.date, _dt.time)):  # datetime is a date subclass
            return value.isoformat()
        raise ConfigValueError("must be a string")
    return conv(value) if isinstance(value, str) else value


def _coerce(action: argparse.Action, value: Any) -> Any:
    """Validate and convert a config value for ``action``; raise ConfigValueError."""
    if action.nargs == 0:  # store_true / store_false flags
        if not isinstance(value, bool):
            raise ConfigValueError("must be true or false")
        return value
    if isinstance(action, argparse._AppendAction):  # e.g. tags = ["a", "b"]
        items = value if isinstance(value, list) else [value]
        return [_coerce_scalar(action, item) for item in items]
    if value is None:
        return None
    value = _coerce_scalar(action, value)
    if action.choices is not None and value not in action.choices:
        # Config files are forgiving about case ("CUDA" == "cuda"); the CLI isn't.
        folded = {str(c).lower(): c for c in action.choices}
        if isinstance(value, str) and value.lower() in folded:
            return folded[value.lower()]
        raise ConfigValueError(f"must be one of {', '.join(map(str, action.choices))}")
    return value


def apply_to_parser(
    parser: argparse.ArgumentParser, cli_defaults: dict[str, Any]
) -> tuple[set[str], set[str]]:
    """Apply config values as parser defaults, validated like CLI input.

    ``parser.set_defaults`` skips ``type=`` and ``choices`` for non-string
    values, so without this a TOML ``beam_size = 5.5`` or ``device = "gpu"``
    would reach the pipeline unchecked. Each key known to the parser is
    converted and checked against its flag; an invalid value stops with a
    parser error naming the key. Returns ``(applied, ignored)`` key sets;
    ignored keys match no flag. CLI flags still win: they are parsed after
    these defaults (invariant #7).
    """
    actions = {a.dest: a for a in parser._actions if a.dest != "help"}
    applied: dict[str, Any] = {}
    ignored: set[str] = set()
    for key, value in cli_defaults.items():
        action = actions.get(key)
        if action is None:
            ignored.add(key)
            continue
        try:
            applied[key] = _coerce(action, value)
        except (ConfigValueError, ValueError, TypeError) as e:
            parser.error(f"config '{key}' {e} (got {value!r})")
    for group in parser._mutually_exclusive_groups:
        set_in_config = [
            a.dest for a in group._group_actions
            if a.dest in applied and _is_set(a, applied[a.dest])
        ]
        if len(set_in_config) > 1:
            parser.error(
                f"config sets {', '.join(repr(d) for d in set_in_config)}: "
                "they are mutually exclusive, keep one"
            )
    if applied:
        parser.set_defaults(**applied)
    return set(applied), ignored


def _is_set(action: argparse.Action, value: Any) -> bool:
    """Whether a config value actually selects ``action`` (``false`` / unset don't)."""
    if action.nargs == 0:
        return value is True
    return value is not None


_UNSET = object()


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: list[str] | None,
    cli_defaults: dict[str, Any],
) -> tuple[argparse.Namespace, set[str], set[str]]:
    """Apply the config, parse ``argv``, and make CLI flags win inside mutex groups.

    argparse only detects conflicts between flags typed on the command line;
    config values arrive as defaults it never "sees". Without this, a config
    ``prompt = "X"`` would beat a CLI ``--prompt-file``, and a config
    ``quiet = true`` would silence a CLI ``-v`` (invariant #7: the CLI always
    wins). So when a member of a mutually exclusive group is given on the
    command line, the other members set by the config go back to their
    built-in defaults. Returns ``(args, applied, ignored)`` like
    :func:`apply_to_parser`.
    """
    groups = [list(g._group_actions) for g in parser._mutually_exclusive_groups]
    builtin = {a.dest: a.default for group in groups for a in group}
    applied, ignored = apply_to_parser(parser, cli_defaults) if cli_defaults else (set(), set())
    args = parser.parse_args(argv)

    if applied & set(builtin):
        # Re-parse with the group members pre-set to a sentinel: argparse fills
        # in defaults only for missing attributes, so whatever is no longer the
        # sentinel was typed on the command line.
        probe = argparse.Namespace(**{dest: _UNSET for dest in builtin})
        parser.parse_args(argv, namespace=probe)
        given = {dest for dest in builtin if getattr(probe, dest) is not _UNSET}
        for group in groups:
            dests = {a.dest for a in group}
            if dests & given:
                for dest in (dests - given) & applied:
                    setattr(args, dest, builtin[dest])
    return args, applied, ignored
