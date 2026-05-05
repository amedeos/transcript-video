"""Run configuration dataclasses for the transcription pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DiarizationConfig:
    enabled: bool = True
    hf_token: str | None = None
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    model_name: str | None = None  # ``None`` = whisperX default (currently community-1).


@dataclass
class OutputConfig:
    write_json: bool = True
    write_srt: bool = False
    write_txt: bool = False
    write_md: bool = False
    output_dir: Path | None = None
    basename: str | None = None


@dataclass
class FrontmatterConfig:
    """Used only when ``--md`` is set."""

    date: str | None = None  # ISO ``YYYY-MM-DD``; defaults to today.
    tags: list[str] = field(default_factory=list)
    source: str | None = None  # Defaults to the input filename.


@dataclass
class RunConfig:
    input_file: Path
    model: str = "large-v3"
    beam_size: int = 5
    device: str | None = None  # ``cuda`` / ``cpu`` / ``None`` = auto.
    compute_type: str | None = None  # Override (e.g. ``int8_float16``); ``None`` = auto.
    language: str | None = None  # ``None`` = autodetect.

    initial_prompt: str | None = None  # ``None`` = unset, ``""`` = explicitly disabled.
    hotwords: str | None = None  # Same convention as ``initial_prompt``.
    anti_loop: bool = False

    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    frontmatter: FrontmatterConfig = field(default_factory=FrontmatterConfig)
    speaker_map: dict[str, str] = field(default_factory=dict)

    # Path to a previously-saved ``*_transcript.aligned.json`` to resume from.
    # When set, ASR + alignment are skipped and the pipeline jumps straight to
    # diarization (if enabled) and the final outputs.
    resume_from_aligned: Path | None = None
