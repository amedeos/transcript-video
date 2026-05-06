# transcript-video

[![CI](https://github.com/amedeos/transcript-video/actions/workflows/ci.yml/badge.svg)](https://github.com/amedeos/transcript-video/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#)

Transcribe and diarize video/audio using [whisperX](https://github.com/m-bain/whisperX) — which combines [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with wav2vec2 forced alignment and [pyannote.audio](https://github.com/pyannote/pyannote-audio) speaker diarization — and emit JSON, SRT, plain text, and a human-readable Markdown transcript with YAML frontmatter.

This project is the English-language successor of [transcript-italian-video](https://github.com/amedeos/transcript-italian-video). It keeps every optimization from the reference (initial prompt, hotwords, anti-loop) and adds:

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
- **Project config file** (`transcript-video.toml`): persist per-project flags (model, beam_size, hotwords, tags, speaker map, ...) and switch bundles via `--profile NAME`.

## Requirements

- Python 3.10+
- ffmpeg available on `$PATH`
- For GPU runs: NVIDIA driver + CUDA (≥10 GB VRAM recommended; 16 GB+ for `large-v3`)
- For diarization: a HuggingFace account with the diarization model's terms accepted, plus an HF token. whisperX's current default is [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1); the older [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) is also supported via `--diarize-model`. **Both are gated repos** — visit the model page once and click "Agree and access repository" before running diarization, otherwise the pipeline aborts with a 403 GatedRepo error.

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
| `--device {auto,cuda,cpu}` | Device | `auto` |
| `--compute-type TYPE` | Override (e.g. `int8`, `float16`) | auto |

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
| `--anti-loop` | `condition_on_previous_text=False`, `compression_ratio_threshold=2.0`, `no_speech_threshold=0.5` — apply when you observe Whisper's cyclic hallucinations |

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

**Outputs** — JSON is always produced:

| Flag | Description |
|------|-------------|
| `--srt` | Write SubRip subtitles |
| `--txt` | Write plain text |
| `--md` | Write Markdown with frontmatter |
| `--output-dir DIR` | Output directory (default: alongside the input) |
| `--basename NAME` | Filename stem (default: stem of the input) |

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
| `--speaker-map "..."` / `--speaker-map-file PATH` | Same as `transcript-from-video` |
| `--date`, `--tag`, `--source` | Frontmatter overrides |
| `--merge-gap-seconds FLOAT` | Merge consecutive same-speaker segments whose silent gap is at most this many seconds (default `1.5`; `0` disables merging) |
| `--paragraph-chars N` | Break long speaker blocks into paragraphs at sentence boundaries when the running paragraph exceeds N characters (default 400; `0` disables splitting) |
| `--mark-suspect` | Inline `[?]` marker before each segment flagged as suspect (low ASR confidence or high silence probability) |
| `--list-speakers` | Print a per-speaker overview (label, name, duration, %, turns, suspect count, first words) and exit without writing Markdown. Use this to figure out who is who before filling in `--speaker-map`. |
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

## Outputs

All outputs default to the input directory with the suffix `_transcript`. With `--basename my_meeting`, files become `my_meeting_transcript.{json,srt,txt,md}`.

### JSON (canonical artifact)

```jsonc
{
  "schema_version": 1,
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

CLI flags always override the config. The config provides defaults; it never restricts.

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

For CPU-only execution, drop `--device nvidia.com/gpu=all` and pass `--device cpu` to `transcript-from-video` inside the container (slow — only sensible for short clips or smoke tests).

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

### GPU memory pressure

Switch to a smaller model:

```bash
transcript-from-video video.mp4 --model medium
```

## License

GPL v3 — see [LICENSE](LICENSE).
