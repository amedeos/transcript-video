# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working agreement: never auto-commit

The user reviews and validates every commit individually before it lands. **Do not run `git commit` on your own**, even when:

- a previous instruction said "go ahead and commit" — that approval is one-shot, not a standing license
- multiple logical units are obviously ready to be split across commits
- auto mode is active (auto mode covers code changes, not publishing actions)

After completing a change, stage the files explicitly and stop. Show the user a summary of what would be committed and wait for an explicit per-commit "ok" before running `git commit`. If you have several batches of changes, get approval for each batch separately rather than chaining them.

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
transcript-from-video INPUT [options]
transcript-to-md PATH/TO/foo_transcript.json [options]
```

## Architecture: the invariants

These are non-obvious from the code and **must hold across any refactor**.

### 1. The torch-free re-render path

`transcript-to-md` exists so users can re-render Markdown from a saved JSON without a GPU or model download. The protected set (enforced by `tests/test_torch_free.py`):

- `utils.py`, `speakers.py`, `stats.py`, `markdown.py`, `writers.py`, `project_config.py`, `cli_to_md.py`
- Allowed deps: stdlib + PyYAML + `tomli`/`tomllib`

Do **not** add imports from `asr`, `diarize`, `pipeline`, `preflight`, `whisperx`, `torch`, `pyannote`, `transformers`, `ctranslate2`, or `faster_whisper` to those modules. If you need a helper from there, lift it into `utils.py` / `stats.py` instead.

### 2. JSON is the contract between the two binaries

The shape produced by `pipeline.py` is the input to `markdown.py`. Adding/renaming a field in one place requires the other. Bump `schema_version` in `pipeline.py` if you change the shape; have `markdown.py` tolerate older versions or fail with a clear message.

Top-level keys: `schema_version`, `source_file`, `transcribed_at`, `parameters`, `audio_info`, `stats`, `segments`, `full_text`. All field names are English (this is the explicit rename vs. the reference project, which used Italian field names like `file_sorgente`, `parametri`, `segmenti`).

### 3. Output policy is the reverse of the reference project

- The reference project always emitted TXT + SRT + JSON.
- Here, **JSON is always written**; **`--srt`, `--txt`, `--md` are opt-in**.

Don't restore the old behavior "for convenience".

### 4. Default language is autodetect

The reference project defaulted to `--language it`. **This project does not** — the user explicitly wanted no default. `--language` only sets a forced override; `parameters.language_forced` in the JSON is `null` whenever autodetect was used.

### 5. Two-stage pipeline + `stage` field

The pipeline writes two persistence layers:

- `*_transcript.aligned.json` — written immediately after alignment, only when diarization is enabled. The free safety net: if diarization fails, ASR + alignment work is preserved.
- `*_transcript.json` — the canonical output, written after the full pipeline.

Both share the same schema. The top-level `stage` field discriminates them (`"aligned"` vs `"complete"`). When refactoring `_build_payload()`, keep both stages going through the same builder so they stay structurally identical.

`--resume-from-aligned PATH` skips ASR + alignment and feeds the snapshot straight to diarization. The aligned snapshot is the only legitimate input form for resuming — don't accept `*_transcript.json` for resume (its segments already carry speakers, so re-diarizing would clobber them).

### 6. Pre-flight runs by default

`preflight.run_preflight(config)` is invoked at the start of every transcription unless `--no-check` is passed. The pre-flight tests we hit on real machines (HF token validity, gated-model access, whisperX API resolution) catch failures in 1–2 seconds instead of 5 minutes. When adding a new external dependency to the pipeline, add a pre-flight check for it.

### 7. CLI flags always win over the config file

`project_config.toml` provides defaults via `parser.set_defaults(**)`; explicit CLI flags overwrite them at parse time. Don't introduce config keys that "force" a value (e.g. by reading the config AFTER `parser.parse_args`); the contract is one-directional. The order of resolution for the config itself is: `--config` explicit → `./transcript-video.toml` → `<input_dir>/transcript-video.toml` → none.

`[speaker_map]` is special-cased: it's not a CLI flag (the CLI uses `--speaker-map` / `--speaker-map-file`), so we extract it separately and pass it as `fallback` to `resolve_speaker_map`. When refactoring this, keep the precedence: `--speaker-map` > `--speaker-map-file` > config `[speaker_map]` > `{}`.

### 8. Suspect thresholds are policy, not data

`stats.DEFAULT_AVG_LOGPROB_THRESHOLD = -1.0` and `DEFAULT_NO_SPEECH_PROB_THRESHOLD = 0.6` are tuning constants for whisperX large-v3. They are persisted in `parameters.suspect_thresholds` so a downstream consumer can tell *which* thresholds produced the flags. Don't change the defaults silently — bump `schema_version` if the new defaults would meaningfully reshuffle the suspect set on existing JSONs.

`mark_suspect_segments` mutates segments in place. Calling it on segments that were already flagged is idempotent (the existing `suspect: true` is overwritten with the same value); but if you change thresholds between calls, the previous flags are NOT cleared first. If that ever matters, normalize at the start of the function.

## Reference-project parity

These optimizations are intentionally preserved verbatim from `trascrivi.py` (only translated to English) and their behavior must not drift:

- **Anti-loop** (`--anti-loop`): sets exactly `condition_on_previous_text=False`, `compression_ratio_threshold=2.0`, `no_speech_threshold=0.5`. See `asr.build_transcribe_kwargs`.
- **Initial prompt** (mutex group): `--prompt` / `--prompt-file` / `--no-prompt`. `None` means "unset", `""` means "explicitly disabled".
- **Hotwords** (mutex group): same tri-state convention as the prompt group.
- **CUDA detection** (`asr.check_cuda_available`): probes `ctranslate2.get_supported_compute_types("cuda")` and prefers `float16`. Falls back to `int8` on CPU. Same logic as `trascrivi.py:15-22`.

If you find yourself "improving" any of these, double-check the upstream behavior first — they were chosen deliberately.

## CLI design rules

- Two binaries, not one with subcommands: `transcript-from-video` (full pipeline) and `transcript-to-md` (pure re-render). They are wired in `pyproject.toml` under `[project.scripts]`.
- Mutex groups for `--prompt*` and `--hotwords*` because the three modes (inline / file / explicit-disable) are exclusive by design.
- Frontmatter overrides (`--date`, `--tag`, `--source`) live on **both** binaries so re-rendering can correct or extend metadata after the fact.
- `--merge-gap-seconds 0` means "never merge consecutive same-speaker segments". The check in `markdown._group_segments` is `merge_gap_seconds > 0` *then* `gap <= merge_gap_seconds` — keep that ordering.

## HuggingFace token resolution

`diarize.resolve_hf_token` checks (in order): `--hf-token`, `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACE_TOKEN`, then `~/.cache/huggingface/token`. Diarization aborts with a clear message if none is found. Don't silently disable diarization on missing token — that's `--no-diarize`'s job.

## whisperX API drift

whisperX's API has churned across 3.x. We adapt at runtime instead of pinning a tight version:

- **Class location**: `_resolve_diarization_api()` looks up `DiarizationPipeline` in both `whisperx.diarize` (modern) and top-level `whisperx` (older). Don't add `whisperx.diarize.Pipeline` to that lookup — it re-exports pyannote's raw `Pipeline` and has an incompatible `__init__`.
- **Constructor signature**: `_build_pipeline_init_kwargs()` introspects `__init__.parameters` and picks the first matching token kwarg from `("token", "use_auth_token", "auth_token")`. The `device` and `model_name`/`model` kwargs are forwarded only if the signature accepts them.
- **Default diarization model**: whisperX currently defaults to `pyannote/speaker-diarization-community-1` (gated). The older `pyannote/speaker-diarization-3.1` works too via `--diarize-model`. Both require accepting their terms on huggingface.co.
- **Gated-repo errors**: `diarize_and_assign()` catches `GatedRepoError` / 403 / Forbidden during pipeline construction and prints an actionable hint with the URL to visit. Keep that branch when refactoring — silently re-raising the HF stack trace is what the previous version did, and it's user-hostile.

## License

GPL v3 (inherited from the reference project). Don't relicense without explicit user approval.
