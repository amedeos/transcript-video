# transcript-video

[![CI](https://github.com/amedeos/transcript-video/actions/workflows/ci.yml/badge.svg)](https://github.com/amedeos/transcript-video/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#)

Transcribe and diarize video/audio using [whisperX](https://github.com/m-bain/whisperX) — which combines [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with wav2vec2 forced alignment and [pyannote.audio](https://github.com/pyannote/pyannote-audio) speaker diarization — and emit JSON, SRT, plain text, and a human-readable Markdown transcript with YAML frontmatter.

This project is the English-language successor of [transcript-italian-video](https://github.com/amedeos/transcript-italian-video). It keeps the reference's initial prompt and hotwords (and accepts its `--anti-loop` flag, which whisperX's batched decoding currently ignores; see the options table) and adds:

- **Speaker diarization** via pyannote (whisperX integration).
- **Language autodetect** by default (`--language` still available to force).
- **Output policy reversal**: JSON is always written; SRT, TXT, and Markdown are opt-in.
- **Markdown export with YAML frontmatter and speaker headings** (`--md`), including an inline or sidecar speaker-name map.
- **Re-render path**: a separate `transcript-to-md` binary regenerates the Markdown from an existing JSON without loading any model.
- **Pre-flight checks** (`--check`): catch missing ffmpeg, bad HF token, or gated-model access in seconds — before paying the GPU bill.
- **Resumable pipeline** (`--resume-from-aligned`): an aligned snapshot is saved before diarization so a downstream failure never wastes the slow ASR + alignment work.
- **Per-speaker stats** in the JSON (talk time, percentage, turns, suspect counts) and a `transcript-to-md --list-speakers` overview to help with mapping labels to names.
- **Suspect-segment flagging**: low-confidence ASR segments are marked in the JSON with `suspect: true` and `suspect_reasons: [...]`. Optional `--mark-suspect` adds an inline `[?]` marker in the rendered Markdown, exactly where the dubious span starts.
- **Paragraph splitting**: long speaker blocks are broken at sentence boundaries when they exceed `--paragraph-chars` (default 400) — keeps the rendered transcript scannable without fragmenting turns.
- **Cross-video speaker auto-identification** (`--identify-speakers` + `transcript-learn` + `transcript-voices`): after enrolling speakers from a few transcripts, future videos containing the same voices are auto-labeled. Identification works both at pipeline time (`transcript-from-video --identify-speakers`) and at re-render time (`transcript-to-md --identify-speakers`, torch-free).
- **Project config file** (`transcript-video.toml`): persist per-project flags (model, beam_size, hotwords, tags, speaker map, ...) and switch bundles via `--profile NAME`.

## Requirements

- Python 3.10+
- ffmpeg available on `$PATH`
- For GPU runs: NVIDIA driver + CUDA (≥10 GB VRAM recommended; 16 GB+ for `large-v3`)
- CPU-only machines work too, with no accuracy loss but much slower — see [Running on CPU only](#running-on-cpu-only)
- For diarization: a HuggingFace account with the diarization model's terms accepted, plus an HF token. whisperX's current default is [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1); the older [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) is also supported via `--diarize-model`. **Both are gated repos** — visit the model page once and click "Agree and access repository" before running diarization, otherwise the pipeline aborts with a 403 GatedRepo error.
- For the speaker-embedding cache (used by `transcript-learn`): the pipeline also fetches [pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) when diarization is enabled. The pre-flight check warns if access is missing; if it is, the embedding cache is skipped non-fatally and the rest of the pipeline proceeds.

## Installation

### uv (primary)

```bash
uv venv
uv pip install -e .            # CPU
uv pip install -e ".[cuda]"    # CPU + CUDA extras
```

### pip (fallback)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Running the binaries

`uv pip install` finds `.venv` automatically without activating it, but the installed console scripts (`transcript-from-video`, `transcript-to-md`, plus `pytest` / `ruff` from the dev extra) still need their environment active to be on `$PATH`. Two options:

```bash
# Activate the venv (classic):
source .venv/bin/activate
transcript-from-video video.mp4

# Or use `uv run` for one-off invocations (no activation):
uv run transcript-from-video video.mp4
```

## HuggingFace token (for diarization)

Diarization needs a token. Resolved in this order:

1. `--hf-token TOKEN`
2. `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) environment variable
3. `~/.cache/huggingface/token` (written by `huggingface-cli login`)

## Usage

### `transcript-from-video`

```bash
transcript-from-video INPUT [options]
```

JSON is always produced. SRT, TXT, and Markdown are opt-in.

#### Examples

```bash
# Minimal: autodetect language, JSON only, diarization on
transcript-from-video video.mp4

# All artifacts
transcript-from-video video.mp4 --srt --txt --md

# Force English, higher-accuracy beam, anti-loop mitigations
transcript-from-video interview.mp4 --language en --beam_size 10 --anti-loop

# Initial prompt for style/glossary, hotwords for proper nouns
transcript-from-video podcast.mp4 \
  --prompt "Glossary: API, GPU, microservices, Kubernetes." \
  --hotwords "Anthropic Claude faster-whisper"

# Diarization with bounded speaker count + named speakers + frontmatter
transcript-from-video video.mp4 --md \
  --min-speakers 2 --max-speakers 4 \
  --speaker-map "SPEAKER_00=Amedeo,SPEAKER_01=Tizio" \
  --tag openshift --tag ovn-kubernetes --tag troubleshooting

# Skip diarization (faster, no HF token needed)
transcript-from-video lecture.mp4 --no-diarize --txt --srt
```

#### Argument groups

**Model / device**

| Flag | Description | Default |
|------|-------------|---------|
| `--model NAME` | whisperX model | `large-v3` |
| `--beam_size N` | Beam search width | `5` |
| `--device {auto,cuda,cpu}` | Device. `auto` uses the GPU when one is usable, otherwise CPU. `cuda` stops with an error if no GPU is usable | `auto` |
| `--compute-type TYPE` | Override (e.g. `int8`, `float16`; any case). Only ctranslate2's types are accepted, so a typo stops with an error. A type the device can't run is replaced, with a warning: `int8*` by the closest supported `int8*` type, anything else by the device default. `auto` and `default` mean the lossless default below | `float16` on CUDA, `float32` on CPU |
| `--threads N` | CPU threads for ASR, alignment and diarization. `0` means auto (undoes a config value) | logical CPUs − 1 (capped by a container CPU quota) for stages on CPU; library defaults for stages on GPU |
| `--batch-size N` | ASR chunks decoded at once. Changes speed and RAM, never the transcript. Lower it if the run is killed for lack of memory. `0` means auto | `16` on CUDA, `4` on CPU |

**Language**

| Flag | Description | Default |
|------|-------------|---------|
| `--language CODE` | Force a language code | autodetect |

**Initial prompt** (mutex, opt-in — same semantics as the reference project)

| Flag | Description |
|------|-------------|
| `--prompt "..."` | Inline initial-prompt text (~224 token cap) |
| `--prompt-file PATH` | Read prompt from a UTF-8 file |
| `--no-prompt` | Explicitly disable |

**Hotwords** (mutex, opt-in)

| Flag | Description |
|------|-------------|
| `--hotwords "..."` | Inline hotwords (space-separated) |
| `--hotwords-file PATH` | Read hotwords from a UTF-8 file |
| `--no-hotwords` | Explicitly disable |

**Decoding**

| Flag | Description |
|------|-------------|
| `--anti-loop` | `condition_on_previous_text=False`, `compression_ratio_threshold=2.0`, `no_speech_threshold=0.5`, from the reference project. With whisperX's batched decoding these currently have no effect: each 30 s chunk is already decoded on its own, without conditioning on the previous text |

**Verbosity**

| Flag | Description |
|------|-------------|
| `-q`, `--quiet` | Suppress info-level output; only warnings and errors. |
| `-v`, `--verbose` | Enable debug-level output (per-segment ASR progress). |

**Pre-flight checks**

By default a pre-flight runs at the start of every transcription. It validates ffmpeg / CUDA / HF token / gated-model access / whisperX API in 1–2 seconds, before the model is loaded.

| Flag | Description |
|------|-------------|
| `--check` | Run only the pre-flight (with network) and exit 0/1. Useful before kicking off a long run. |
| `--no-check` | Skip the default pre-flight (escape hatch). |
| `--offline-check` | Skip the network parts (HF validity + gated-model access). |

**Resume / cache**

After alignment, an `*_transcript.aligned.json` snapshot is written automatically when diarization is enabled. If the diarize step fails for any reason, the costly ASR + alignment work is preserved and you can resume:

| Flag | Description |
|------|-------------|
| `--resume-from-aligned PATH` | Skip ASR + alignment; load the snapshot, run diarization, write outputs. The source video path is read from the snapshot if not given as a positional. |

**Project config**

| Flag | Description |
|------|-------------|
| `--config PATH` | Explicit `transcript-video.toml`. Otherwise found in the current directory or alongside the input video. |
| `--profile NAME` | Apply `[profiles.NAME]` from the config on top of the top-level defaults. CLI flags still win. |

See [Project config file](#project-config-file) for the full schema.

**Diarization**

| Flag | Description |
|------|-------------|
| `--no-diarize` | Skip pyannote |
| `--hf-token TOKEN` | HF token (else env / cached file) |
| `--diarize-model ID` | pyannote model id (default: whisperX's built-in, currently `pyannote/speaker-diarization-community-1`; alternative: `pyannote/speaker-diarization-3.1`) |
| `--num-speakers N` | Exact speaker count |
| `--min-speakers N` / `--max-speakers N` | Bounds (improves quality) |
| `--speaker-map "L0=Name0,L1=Name1"` | Inline label→name map |
| `--speaker-map-file PATH` | YAML or JSON sidecar |

**Speaker auto-identification** (opt-in; requires diarization)

| Flag | Description |
|------|-------------|
| `--identify-speakers` | After diarization, match each cluster against the voice DB and populate `speaker_identities` in the JSON. Off by default. |
| `--voice-db PATH` | Override the default DB location (`~/.local/share/transcript-video/voices.json`). |
| `--id-threshold FLOAT` | Cosine-similarity threshold (default `0.65`). Higher = fewer false positives. |

`--speaker-map` keeps final-authority semantics: any label the user maps wins (recorded as `source="manual"` in the JSON). Below-threshold clusters are left unidentified — never silently mislabeled.

**Outputs** — JSON is always produced:

| Flag | Description |
|------|-------------|
| `--srt` | Write SubRip (`.srt`) subtitles |
| `--vtt` | Write WebVTT (`.vtt`) subtitles |
| `--txt` | Write plain text |
| `--md` | Write Markdown with frontmatter |
| `--subtitle-speakers` | Prefix each subtitle cue with the speaker name (applies to `--srt` and `--vtt`). Uses the effective speaker map: CLI `--speaker-map` > manual identities > auto identities > raw `SPEAKER_XX`. |
| `--output-dir DIR` | Output directory (default: alongside the input) |
| `--basename NAME` | Filename stem (default: stem of the input) |

Both `.srt` and `.vtt` are read natively by VLC, mpv and most desktop players: give the subtitle file the same base name as the video in the same folder, or load it from *Subtitles → Add subtitle file*. `.vtt` is also the format HTML5 `<track>` expects for web video.

**Markdown frontmatter** (only used with `--md`)

| Flag | Description |
|------|-------------|
| `--date YYYY-MM-DD` | Frontmatter date (default: today) |
| `--tag TAG` | Repeatable; appended to `tags:` |
| `--source NAME` | Override the `source:` field |

### `transcript-to-md`

Regenerate the Markdown from an existing JSON artifact — no GPU, no models, runs in milliseconds. Useful when you want to refine the speaker map, tags, or merge gap without rerunning ASR.

```bash
transcript-to-md PATH/TO/foo_transcript.json \
  --speaker-map "SPEAKER_00=Amedeo,SPEAKER_01=Tizio" \
  --tag standup --tag retro \
  --merge-gap-seconds 2.0 \
  -o foo.md
```

| Flag | Description |
|------|-------------|
| `-o`, `--output PATH` | Output path (default: `<stem>.md` next to the JSON) |
| `--srt` / `--vtt` | Also write a `.srt` / `.vtt` subtitle file next to the JSON — torch-free, no GPU. Regenerate player-ready subtitles from an existing transcript. |
| `--subtitle-speakers` | Prefix each subtitle cue with the speaker name (applies to `--srt` and `--vtt`); same precedence as on `transcript-from-video`. |
| `--speaker-map "..."` / `--speaker-map-file PATH` | Same as `transcript-from-video` |
| `--date`, `--tag`, `--source` | Frontmatter overrides |
| `--merge-gap-seconds FLOAT` | Merge consecutive same-speaker segments whose silent gap is at most this many seconds (default `1.5`; `0` disables merging) |
| `--paragraph-chars N` | Break long speaker blocks into paragraphs at sentence boundaries when the running paragraph exceeds N characters (default 400; `0` disables splitting) |
| `--mark-suspect` | Inline `[?]` marker before each segment flagged as suspect (low ASR confidence or high silence probability) |
| `--list-speakers` | Print a per-speaker overview (label, name, duration, %, turns, suspect count, first words) and exit without writing Markdown. Use this to figure out who is who before filling in `--speaker-map`. |
| `--identify-speakers` | Refresh `speaker_identities` against the current voice DB before rendering. Torch-free — no GPU or model download. Manual entries survive; auto entries get re-matched. |
| `--voice-db PATH` | Override the default DB location for `--identify-speakers`. |
| `--id-threshold FLOAT` | Cosine-similarity threshold for `--identify-speakers` (default `0.65`). |
| `--config PATH` / `--profile NAME` | See [Project config file](#project-config-file). |
| `-q`, `--quiet` / `-v`, `--verbose` | Output verbosity. |

#### Speaker-mapping workflow

```bash
# 1. Inspect to see who said what (no MD written):
transcript-to-md video_transcript.json --list-speakers

# Output (example):
# Label       Name        Duration         %  Turns  First words
# ----------  ----------  --------  --------  -----  ------------------------------------------
# SPEAKER_00  SPEAKER_00  00:28:30     59.0%     47  Allora oggi parliamo della migrazione...
# SPEAKER_01  SPEAKER_01  00:08:12     17.0%     12  Sì una domanda io ce l'ho a me...

# 2. Now you know who is who → re-render with the map:
transcript-to-md video_transcript.json \
  --speaker-map "SPEAKER_00=Amedeo,SPEAKER_01=Marco" \
  --tag openshift --tag retro
```

### Speaker enrollment and auto-identification

The voice-print database is a single JSON file under your home (default `~/.local/share/transcript-video/voices.json` — XDG-compliant). Three pieces wire it together:

1. The pipeline caches per-cluster embeddings in every transcribed JSON (`speaker_clusters`, schema v2) — no opt-in needed.
2. `transcript-learn` consumes those embeddings to enroll speakers into the DB.
3. `--identify-speakers` (on both `transcript-from-video` and `transcript-to-md`) consults the DB to auto-label recurring speakers.

The full loop:

```bash
# 1. First video: transcribe, then label manually once.
transcript-from-video ep1.mp4
transcript-to-md ep1_transcript.json --list-speakers   # see who is SPEAKER_00, SPEAKER_01
transcript-learn ep1_transcript.json \
  --speaker-map "SPEAKER_00=Mario,SPEAKER_01=Luca"

# 2. Future videos: auto-identify those speakers.
transcript-from-video ep2.mp4 --identify-speakers --md
#  → the rendered Markdown labels SPEAKER_XX clusters that match the DB

# 3. Retroactively re-label old transcripts (no GPU needed):
transcript-to-md ep0_transcript.json --identify-speakers --md
```

**Identity precedence at render time** (highest to lowest):

1. CLI `--speaker-map` / `--speaker-map-file` — always wins.
2. JSON `speaker_identities` entries with `source="manual"` — recorded when `--speaker-map` was passed at transcribe time.
3. JSON `speaker_identities` entries with `source="auto"` — DB matches above threshold.
4. The raw `SPEAKER_XX` label — when nothing else identifies the cluster.

Below-threshold clusters never receive a silent auto-label — they keep their `SPEAKER_XX` label and are visible as unidentified.

The database is a single JSON file, sync-friendly: back it up, share between machines, version-control it alongside project notes. Overridable via `--voice-db PATH` or the `TRANSCRIPT_VIDEO_VOICE_DB` environment variable on every binary that touches it.

#### `transcript-learn`

Append speakers from a transcript JSON to the database.

```bash
transcript-learn video_transcript.json \
  --speaker-map "SPEAKER_00=Mario,SPEAKER_01=Luca"
```

| Flag | Description |
|------|-------------|
| `JSON_PATH` | The `*_transcript.json` produced by `transcript-from-video` (schema version ≥ 2). |
| `--speaker-map "L0=Name,..."` | Inline label-to-name map. |
| `--speaker-map-file PATH` | YAML or JSON sidecar (same shape as `transcript-from-video`'s sidecar). |
| `--voice-db PATH` | Override the default DB location. |
| `--dry-run` | Print what would be added; don't write the DB. |
| `-q`, `--quiet` / `-v`, `--verbose` | Output verbosity. |

Labels in `--speaker-map` that don't match any cluster in the JSON (typos, or speakers from a different video) are reported and skipped — they don't fail the operation. Re-running `transcript-learn` on the same speaker across different videos accumulates samples: the more videos the speaker appears in, the more robust their voice print becomes.

#### `transcript-voices`

Inspect and maintain the database.

```bash
transcript-voices                  # list enrolled speakers (default)
transcript-voices show Mario       # sample-by-sample detail
transcript-voices forget Mario     # remove Mario (prompts y/N)
transcript-voices forget Mario --source ep1_transcript.json --yes
```

| Sub-command | Description |
|-------------|-------------|
| `list` (default when no sub-command) | One line per enrolled speaker, with sample count and embedding model. |
| `show NAME` | Per-sample detail: source transcript, cluster label, duration, when added. |
| `forget NAME [--source X] [-y]` | Remove a speaker entirely, or just samples from a given source. Prompts for confirmation unless `-y`/`--yes`. |

All sub-commands accept `--voice-db PATH` to override the default location.

## Outputs

All outputs default to the input directory with the suffix `_transcript`. With `--basename my_meeting`, files become `my_meeting_transcript.{json,srt,vtt,txt,md}`.

### JSON (canonical artifact)

```jsonc
{
  "schema_version": 2,
  "stage": "complete",
  "source_file": "/abs/path/video.mp4",
  "transcribed_at": "2026-05-05T10:30:00",
  "parameters": {
    "backend": "whisperx",
    "asr": "whisperx-large-v3",
    "model": "large-v3",
    "device": "cuda",
    "compute_type": "float16",
    "beam_size": 5,
    "vad_filter": true,
    "language_forced": null,
    "initial_prompt": null,
    "hotwords": null,
    "anti_loop": false,
    "diarization": {
      "enabled": true,
      "num_speakers": null,
      "min_speakers": null,
      "max_speakers": null
    }
  },
  "audio_info": { "language_detected": "it", "language_probability": 0.98, "duration_seconds": 6135.2 },
  "stats": { "num_segments": 450, "num_speakers": 2, "processing_seconds": 180.3 },
  "speaker_clusters": {
    "SPEAKER_00": {
      "embedding": [0.234, -0.112, "...(256 floats)"],
      "duration_s": 3611.6,
      "n_segments": 261,
      "embedding_model": "pyannote/wespeaker-voxceleb-resnet34-LM"
    },
    "SPEAKER_01": { "...": "..." }
  },
  "speaker_identities": {
    "SPEAKER_00": { "name": "Mario", "score": 0.81, "source": "auto" },
    "SPEAKER_01": { "name": "Luca",  "score": null, "source": "manual" }
  },
  "segments": [
    {
      "id": 0,
      "start": 12.3,
      "end": 18.7,
      "text": "Allora, oggi parliamo della migrazione...",
      "speaker": "SPEAKER_00",
      "words": [{ "word": "Allora,", "start": 12.3, "end": 12.6, "speaker": "SPEAKER_00" }]
    }
  ],
  "full_text": "Allora, oggi parliamo...\n..."
}
```

`speaker_clusters` is populated only when diarization is enabled and the embedding extraction succeeds; otherwise empty `{}`. It's the cache read by `transcript-learn` and by `--identify-speakers`. `speaker_identities` is populated only when `--identify-speakers` was passed (at pipeline time or re-render time); each entry tracks whether it came from the DB (`source: "auto"`) or from an explicit `--speaker-map` (`source: "manual"`). See [Speaker enrollment and auto-identification](#speaker-enrollment-and-auto-identification).

### Markdown

```markdown
---
date: 2026-05-01
duration: "01:42:15"
language: it
source: meeting-foo.mp4
asr: whisperx-large-v3
beam_size: 10
speakers:
  SPEAKER_00: Amedeo
  SPEAKER_01: Tizio
tags: [openshift, ovn-kubernetes, troubleshooting]
---

## [00:00:12] Amedeo
Allora, oggi parliamo della migrazione da SDN a OVN-Kubernetes...

## [00:01:45] Tizio
Sì, il problema che vediamo è...
```

Consecutive segments by the same speaker separated by at most `--merge-gap-seconds` (default `1.5`) are merged into a single block.

## Project config file

Persist per-project flags so you don't retype them every run. Place a `transcript-video.toml` in the directory you run from (or alongside the input video):

```toml
# Top-level keys map to argparse `dest` names — same shape as the CLI flags
# but with underscores instead of dashes.
beam_size       = 10
anti_loop       = true
diarize_model   = "pyannote/speaker-diarization-3.1"
hotwords        = "OpenShift Cgroups Kubernetes RHACS"
tags            = ["openshift", "cgroups-v2"]
paragraph_chars = 400

# Optional inline speaker-name map. CLI --speaker-map still wins.
[speaker_map]
SPEAKER_00 = "Amedeo"
SPEAKER_01 = "Marco"

# Profiles overlay on top of the top-level defaults when --profile is set.
[profiles.meeting]
anti_loop    = true
min_speakers = 2
max_speakers = 8

[profiles.podcast]
beam_size    = 12
anti_loop    = true
```

Resolution order:

1. `--config PATH` (explicit; missing file is an error)
2. `./transcript-video.toml` (current working directory)
3. `<input_dir>/transcript-video.toml` (alongside the video)
4. None — CLI defaults

Activate a profile with `--profile NAME`:

```bash
transcript-from-video video.mp4 --profile meeting --md
```

CLI flags always override the config. The config provides defaults; it never restricts. Each value is checked like the matching CLI flag: `beam_size = 5.5` or `device = "gpu"` stops with an error naming the key, instead of failing later in the pipeline.

## Running on CPU only

No NVIDIA GPU is required. With the default `--device auto` the pipeline uses the GPU when one is usable and switches to CPU on its own otherwise, so the same command works on a GPU desktop and on a laptop. An explicit `--device cuda` (or `device = "cuda"` in the config file) means "GPU or nothing": without a usable GPU the run stops at start-up with an error instead of quietly running for hours on CPU. Keep `device` out of a config file you share between machines.

- **No accuracy loss by default.** On CPU the ASR runs in `float32`, which is at least as precise as the `float16` used on GPU (ctranslate2 has no `float16` on CPU). Alignment (wav2vec2) and diarization (pyannote) run in `float32` on both devices. Expect small numeric differences between machines, not worse transcripts.
- **Threads.** By default stages running on CPU use all logical CPUs but one, which leaves the machine responsive. Inside a container with a CPU quota (`podman run --cpus=4`), the quota counts as the number of CPUs. Stages running on the GPU keep their libraries' own defaults. Override with `--threads N` or `threads = N` in the config file, which applies to every stage.
- **RAM.** `large-v3` in `float32` needs about 6 GB for the model weights alone. On top of that, each chunk in the ASR batch adds about 0.65 GB (estimate). That is why the CPU default is `--batch-size 4` (roughly 10 GB peak) instead of the GPU's 16 (roughly 18 GB, more with a wide beam). The batch never changes the transcript: raise it for speed if you have RAM to spare, lower it to `1` (roughly 8 GB) on a tight machine. `--beam_size` costs mostly time; its RAM cost shows up only at high batch sizes. The ASR model is freed before alignment and diarization load their own models, and the log prints `RAM after …` at each stage.
- **Faster, lighter, slightly less accurate.** Pass `--compute-type int8`: about 4× less memory for the weights and faster, at a small accuracy cost on hard audio (noise, overlapping speakers, proper names). Picking a smaller `--model` costs much more accuracy than `int8`. `large-v3-turbo` is the usual compromise; avoid `distil-*` models for non-English audio.
- **Resuming across machines.** `--resume-from-aligned` picks the device from the current machine and the current `--device`, not from the snapshot: you can run ASR on a GPU box and diarize on a laptop, or the other way round. `parameters.device` / `compute_type` in the JSON keep recording what the ASR ran with.
- **Two libraries, two GPU checks.** ASR uses ctranslate2; alignment, diarization and embeddings use torch. Each is checked on its own. With `auto`, a library that can't use CUDA (e.g. a CPU-only torch wheel) runs its stages on CPU and the log says so; whisperX's voice-activity detection, a torch model, then runs on CPU while the ASR keeps the GPU. With `cuda`, every library that has work to do must be able to use the GPU (a `--resume-from-aligned` runs no ASR, so it only needs torch).

## Container (Podman / Docker)

A [`Containerfile`](Containerfile) ships with the repo. It builds an image with whisperX + pyannote ready to run, on top of `nvcr.io/nvidia/cuda:12.8.1-cudnn-runtime-ubi9` — Red Hat-aligned (UBI9 + NVIDIA's own registry), CUDA tag pinned to match the `+cu128` PyTorch wheel pip installs from PyPI. ffmpeg is fetched as a static GPL build (UBI/RHEL omits ffmpeg for licensing reasons).

Build:

```bash
podman build -t transcript-video -f Containerfile .
```

Run on a GPU host (NVIDIA Container Device Interface):

```bash
podman run --rm \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$(pwd):/data:Z" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:Z" \
  transcript-video \
  video.mp4 --md --tag openshift
```

The container's working directory is `/data`, so paths in the command line are relative to the host directory bind-mounted there. The HuggingFace cache is shared with the host so the diarization model is downloaded once across runs.

For CPU-only execution, drop `--device nvidia.com/gpu=all` and don't pass `--device cuda`: with the default `--device auto` the pipeline falls back to CPU on its own (see [Running on CPU only](#running-on-cpu-only)). It is much slower than a GPU. A `--cpus` quota on the container is respected by the default thread count.

### Rootless podman: keep host groups for GPU access

Out of the box, rootless podman maps your user to `nobody` inside the container, so the `/dev/nvidia*` devices appear unreadable and CUDA fails with `no CUDA-capable device is detected` (the diagnostic in the container shows `nvidia-smi: Failed to initialize NVML: Insufficient Permissions` and `ctranslate2.get_cuda_device_count() == 0`). Add `--group-add keep-groups` so the container inherits your host's `video` / `render` groups and keeps device-file access:

```bash
podman run --rm \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  --group-add keep-groups \
  -e HF_TOKEN="$(cat ~/.cache/huggingface/token)" \
  -v "$(pwd):/data" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  transcript-video \
  video.mp4 --md
```

Make sure your host user is in the right groups (`groups | grep -E 'video|render'`); add yourself with `sudo usermod -aG video,render $USER` and re-login if you aren't. Running rootful (`sudo podman run ...`) is an alternative that sidesteps the namespace mapping entirely.

### HuggingFace token in the container

If you authenticated on the host with `huggingface-cli login`, the token lives in `~/.cache/huggingface/token`, not in `$HF_TOKEN`. Pass it explicitly with `-e HF_TOKEN="$(cat ~/.cache/huggingface/token)"`, or bind-mount the whole cache directory (as the example above does) — the latter also avoids re-downloading the diarization model on every run.

## Troubleshooting

### CUDA not detected

With the default `--device auto` a missing GPU doesn't stop the run: the pipeline falls back to CPU, which is much slower. To check that both GPU users (ctranslate2 for ASR, torch for alignment/diarization) can see CUDA, run the pre-flight with `--device cuda`:

```bash
transcript-from-video --check --device cuda
# [OK  ] cuda   available (compute_type hint: float16)                    ← GPU in use
# [FAIL] cuda   device 'cuda' was requested (--device or config) but alignment/diarization (torch) cannot use a CUDA device. ...   ← names the library
```

The probe only checks that each library sees a CUDA device. If it reports `available` but the model load then fails on missing CUDA libraries (e.g. `libcudnn`), the CUDA wheels are missing: install the `[cuda]` extra. If it fails:

```bash
nvidia-smi
uv pip install -e ".[cuda]" --force-reinstall
```

### Diarization fails with 401/403 / GatedRepoError

The diarization model is gated — you must accept its terms on huggingface.co before the first download. The pipeline now detects this and prints actionable instructions, but the steps are:

1. Visit the model page and click **Agree and access repository**:
   - https://huggingface.co/pyannote/speaker-diarization-community-1 (whisperX default)
   - or https://huggingface.co/pyannote/speaker-diarization-3.1 if you pass `--diarize-model pyannote/speaker-diarization-3.1`
   - some pipelines also depend on https://huggingface.co/pyannote/segmentation-3.0
2. Confirm your token has `read` scope at https://hf.co/settings/tokens
3. Re-run; the model is downloaded once and cached locally

### "Alignment model unavailable for language"

whisperX ships alignment models for a fixed list of languages. If yours is missing, the pipeline keeps going with segment-level (not word-level) timestamps; the Markdown speaker boundaries may be slightly less precise.

### Out of memory (GPU or CPU)

On CPU the symptom is a bare `Killed` (the kernel's OOM killer), usually during ASR. The `RAM after …` log lines show how much each stage used. Try these in order. The first keeps the transcript as it is; the other two trade accuracy for memory:

1. **Lower the batch**: `--batch-size 1` (or `2`). It costs speed, not accuracy, and it is the main lever: ASR memory grows mostly with the batch. A wide `--beam_size` adds a few GB only at high batch sizes.
2. **Quantize**: `--compute-type int8` makes the weights about 4× smaller, at a small accuracy cost on hard audio.
3. **Smaller model**: `--model large-v3-turbo` or `--model medium`. This costs the most accuracy.

```bash
transcript-from-video video.mp4 --batch-size 1
```

## License

GPL v3 — see [LICENSE](LICENSE).
