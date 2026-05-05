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

## Requirements

- Python 3.10+
- ffmpeg available on `$PATH`
- For GPU runs: NVIDIA driver + CUDA (≥10 GB VRAM recommended; 16 GB+ for `large-v3`)
- For diarization: a HuggingFace account with the [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) model terms accepted, plus an HF token

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

`uv pip install` finds `.venv` automatically without activating it, but the installed console scripts (`transcribe-video`, `transcript-to-md`, plus `pytest` / `ruff` from the dev extra) still need their environment active to be on `$PATH`. Two options:

```bash
# Activate the venv (classic):
source .venv/bin/activate
transcribe-video video.mp4

# Or use `uv run` for one-off invocations (no activation):
uv run transcribe-video video.mp4
```

## HuggingFace token (for diarization)

Diarization needs a token. Resolved in this order:

1. `--hf-token TOKEN`
2. `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) environment variable
3. `~/.cache/huggingface/token` (written by `huggingface-cli login`)

## Usage

### `transcribe-video`

```bash
transcribe-video INPUT [options]
```

JSON is always produced. SRT, TXT, and Markdown are opt-in.

#### Examples

```bash
# Minimal: autodetect language, JSON only, diarization on
transcribe-video meeting.mp4

# All artifacts
transcribe-video meeting.mp4 --srt --txt --md

# Force English, higher-accuracy beam, anti-loop mitigations
transcribe-video interview.mp4 --language en --beam_size 10 --anti-loop

# Initial prompt for style/glossary, hotwords for proper nouns
transcribe-video podcast.mp4 \
  --prompt "Glossary: API, GPU, microservices, Kubernetes." \
  --hotwords "Anthropic Claude faster-whisper"

# Diarization with bounded speaker count + named speakers + frontmatter
transcribe-video meeting.mp4 --md \
  --min-speakers 2 --max-speakers 4 \
  --speaker-map "SPEAKER_00=Amedeo,SPEAKER_01=Tizio" \
  --tag openshift --tag ovn-kubernetes --tag troubleshooting

# Skip diarization (faster, no HF token needed)
transcribe-video lecture.mp4 --no-diarize --txt --srt
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

**Diarization**

| Flag | Description |
|------|-------------|
| `--no-diarize` | Skip pyannote |
| `--hf-token TOKEN` | HF token (else env / cached file) |
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
| `--speaker-map "..."` / `--speaker-map-file PATH` | Same as `transcribe-video` |
| `--date`, `--tag`, `--source` | Frontmatter overrides |
| `--merge-gap-seconds FLOAT` | Merge consecutive same-speaker segments whose silent gap is at most this many seconds (default `1.5`; `0` disables merging) |

## Outputs

All outputs default to the input directory with the suffix `_transcript`. With `--basename my_meeting`, files become `my_meeting_transcript.{json,srt,txt,md}`.

### JSON (canonical artifact)

```jsonc
{
  "schema_version": 1,
  "source_file": "/abs/path/meeting.mp4",
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

## Troubleshooting

### CUDA not detected

```bash
nvidia-smi
uv pip install -e ".[cuda]" --force-reinstall
```

### Diarization fails with 401/403

Accept the model terms on HuggingFace and ensure your token has read access:

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

### "Alignment model unavailable for language"

whisperX ships alignment models for a fixed list of languages. If yours is missing, the pipeline keeps going with segment-level (not word-level) timestamps; the Markdown speaker boundaries may be slightly less precise.

### GPU memory pressure

Switch to a smaller model:

```bash
transcribe-video video.mp4 --model medium
```

## License

GPL v3 — see [LICENSE](LICENSE).
