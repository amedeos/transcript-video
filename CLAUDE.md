# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`transcript-video` is the **English-language successor** of [transcript-italian-video](https://github.com/amedeos/transcript-italian-video). It transcribes and diarizes video/audio using whisperX (faster-whisper + wav2vec2 alignment + pyannote 3.x diarization) and produces JSON / SRT / TXT / Markdown.

Everything in this repository — code, comments, docstrings, README, CLI help — is in **English**. Italian is allowed only inside example transcripts in docs.

## Common commands

Install (uv is primary; pip is supported for back-compat):

```bash
uv venv && uv pip install -e .             # CPU
uv pip install -e ".[cuda]"                # + NVIDIA CUDA wheels
uv pip install -e ".[dev]"                 # + ruff, pytest

# pip fallback
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

Lint / sanity:

```bash
ruff check src/
python -m py_compile src/transcript_video/*.py
```

Run the entry points (after install):

```bash
transcribe-video INPUT [options]
transcript-to-md PATH/TO/foo_transcript.json [options]
```

## Architecture: the four invariants

These are non-obvious from the code and **must hold across any refactor**.

### 1. `markdown.py` must stay torch-free

`transcript-to-md` exists so users can re-render Markdown from a saved JSON without a GPU or model download. It transitively imports only:

- `markdown.py` → `speakers.py` → `utils.py`
- stdlib + PyYAML

Do **not** add imports from `asr`, `diarize`, `pipeline`, `whisperx`, `torch`, `pyannote`, `transformers`, `ctranslate2`, or `faster_whisper` to those four modules. If you need a helper from there, lift it into `utils.py` instead. There is a smoke test for this: see "Verification" in the original plan.

### 2. JSON is the contract between the two binaries

The shape produced by `pipeline.py` is the input to `markdown.py`. Adding/renaming a field in one place requires the other. Bump `schema_version` in `pipeline.py` if you change the shape; have `markdown.py` tolerate older versions or fail with a clear message.

Top-level keys: `schema_version`, `source_file`, `transcribed_at`, `parameters`, `audio_info`, `stats`, `segments`, `full_text`. All field names are English (this is the explicit rename vs. the reference project, which used Italian field names like `file_sorgente`, `parametri`, `segmenti`).

### 3. Output policy is the reverse of the reference project

- The reference project always emitted TXT + SRT + JSON.
- Here, **JSON is always written**; **`--srt`, `--txt`, `--md` are opt-in**.

Don't restore the old behavior "for convenience".

### 4. Default language is autodetect

The reference project defaulted to `--language it`. **This project does not** — the user explicitly wanted no default. `--language` only sets a forced override; `parameters.language_forced` in the JSON is `null` whenever autodetect was used.

## Reference-project parity

These optimizations are intentionally preserved verbatim from `trascrivi.py` (only translated to English) and their behavior must not drift:

- **Anti-loop** (`--anti-loop`): sets exactly `condition_on_previous_text=False`, `compression_ratio_threshold=2.0`, `no_speech_threshold=0.5`. See `asr.build_transcribe_kwargs`.
- **Initial prompt** (mutex group): `--prompt` / `--prompt-file` / `--no-prompt`. `None` means "unset", `""` means "explicitly disabled".
- **Hotwords** (mutex group): same tri-state convention as the prompt group.
- **CUDA detection** (`asr.check_cuda_available`): probes `ctranslate2.get_supported_compute_types("cuda")` and prefers `float16`. Falls back to `int8` on CPU. Same logic as `trascrivi.py:15-22`.

If you find yourself "improving" any of these, double-check the upstream behavior first — they were chosen deliberately.

## CLI design rules

- Two binaries, not one with subcommands: `transcribe-video` (full pipeline) and `transcript-to-md` (pure re-render). They are wired in `pyproject.toml` under `[project.scripts]`.
- Mutex groups for `--prompt*` and `--hotwords*` because the three modes (inline / file / explicit-disable) are exclusive by design.
- Frontmatter overrides (`--date`, `--tag`, `--source`) live on **both** binaries so re-rendering can correct or extend metadata after the fact.
- `--merge-gap-seconds 0` means "never merge consecutive same-speaker segments". The check in `markdown._group_segments` is `merge_gap_seconds > 0` *then* `gap <= merge_gap_seconds` — keep that ordering.

## HuggingFace token resolution

`diarize.resolve_hf_token` checks (in order): `--hf-token`, `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACE_TOKEN`, then `~/.cache/huggingface/token`. Diarization aborts with a clear message if none is found. Don't silently disable diarization on missing token — that's `--no-diarize`'s job.

## License

GPL v3 (inherited from the reference project). Don't relicense without explicit user approval.
