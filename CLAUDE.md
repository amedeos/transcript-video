# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working agreement: never auto-commit

The user reviews and validates every commit individually before it lands. **Do not run `git commit` on your own**, even when:

- a previous instruction said "go ahead and commit" — that approval is one-shot, not a standing license
- multiple logical units are obviously ready to be split across commits
- auto mode is active (auto mode covers code changes, not publishing actions)

After completing a change, stage the files explicitly and stop. Show the user a summary of what would be committed and wait for an explicit per-commit "ok" before running `git commit`. If you have several batches of changes, get approval for each batch separately rather than chaining them.

## Working agreement: branch before modifying

Never edit files directly on `main`. Before starting any modification, create a dedicated branch from an up-to-date `main`:

```bash
git checkout main && git pull --ff-only
git checkout -b <prefix>/<short-slug>
```

Branch name prefixes (pick the one that matches the intent):

- `feature/` — new functionality
- `fix/` — bug fix
- `refactor/` — restructuring without behavior change
- `docs/` — documentation-only changes (README, CLAUDE.md, comments-only edits)

The slug after the prefix is short, kebab-case, and describes the change (e.g. `feature/srt-output`, `fix/diarize-token-kwarg`, `refactor/json-builder`, `docs/branch-policy`).

This rule applies to every modification, including small docs touch-ups. Stacked branches (one branch off another, when the first is not yet merged) are allowed when the user explicitly opts in.

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

`transcript-to-md`, `transcript-learn`, and `transcript-voices` all exist so users can do post-processing work — re-render Markdown, enroll speakers into the DB, manage the DB — without a GPU or model download. The protected set (enforced by `tests/test_torch_free.py`):

- `utils.py`, `speakers.py`, `stats.py`, `markdown.py`, `writers.py`, `project_config.py`, `cli_to_md.py`
- `speaker_db.py`, `enrollment.py`, `cli_learn.py`, `cli_voices.py`
- Allowed deps: stdlib + PyYAML + `tomli`/`tomllib`

Do **not** add imports from `asr`, `diarize`, `pipeline`, `preflight`, `speaker_embed`, `whisperx`, `torch`, `pyannote`, `transformers`, `ctranslate2`, `faster_whisper`, or `numpy` to those modules. If you need a helper from there, lift it into `utils.py` / `stats.py` / `speaker_db.py` instead.

The auto-identification refresh in `cli_to_md.py --identify-speakers` is part of this set: cosine matching runs in pure Python (no numpy) against the JSON's cached `speaker_clusters`. See invariant #9.

### 2. JSON is the contract across all four binaries

The shape produced by `pipeline.py` is the input to `markdown.py`, `enrollment.py`, and the auto-id path in `cli_to_md.py`. Adding/renaming a field in one place requires the others. Bump `schema_version` in `pipeline.py` when you change the shape; have consumers tolerate older versions or fail with a clear message.

Top-level keys at the current schema version (`2`):

- `schema_version`, `stage`, `source_file`, `transcribed_at`, `parameters`, `audio_info`, `stats`, `segments`, `full_text`
- `speaker_clusters` (since v2): per-cluster cached embeddings — empty when diarization is off or extraction failed.
- `speaker_identities` (since v2): per-cluster `{name, score, source}` — populated only when `--identify-speakers` is on, at pipeline time or re-render time.

All field names are English (the explicit rename vs. the reference project, which used Italian field names like `file_sorgente`, `parametri`, `segmenti`).

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

### 8. Speaker identification: CLI map > manual identities > auto identities

The `speaker_identities` field in the JSON (schema v2) is a *hint* about who each `SPEAKER_XX` cluster is, populated by `--identify-speakers` at transcribe time and/or re-render time. Each entry carries a `source` discriminator:

- `source: "auto"` — DB match above threshold. **Refresh-able**: `transcript-to-md --identify-speakers` overwrites these with the current DB state. Stale auto entries should never block fresh matches.
- `source: "manual"` — user assertion via `--speaker-map` at transcribe time. **Preserved on refresh**: a re-render auto-id pass leaves these alone. The user's intent at transcribe time is not undone by today's DB state.

Rendering — `markdown.py` and `cli_to_md._format_speaker_overview` — builds an effective map via `markdown.build_effective_speaker_map(transcript, cli_speaker_map)`. The precedence (highest to lowest):

1. CLI `--speaker-map` / `--speaker-map-file` at render time
2. `speaker_identities[label]` with `source="manual"`
3. `speaker_identities[label]` with `source="auto"`
4. The raw `SPEAKER_XX` label

Below-threshold clusters get **no entry** in `speaker_identities` — they keep their raw label and are visible as unidentified. Never silently mislabel.

The whole identification path (`speaker_db.auto_resolve_speaker_map`, `_refresh_auto_identities` in `cli_to_md`, `_resolve_speaker_identities` in `pipeline`, `build_effective_speaker_map` in `markdown`) is torch-free at the consumer side. Only `pipeline._resolve_speaker_identities` runs in the heavy path because the pipeline already has torch loaded; the same logic is replayed at re-render time with no torch import.

### 9. Suspect thresholds are policy, not data

`stats.DEFAULT_AVG_LOGPROB_THRESHOLD = -1.0` and `DEFAULT_NO_SPEECH_PROB_THRESHOLD = 0.6` are tuning constants for whisperX large-v3. They are persisted in `parameters.suspect_thresholds` so a downstream consumer can tell *which* thresholds produced the flags. Don't change the defaults silently — bump `schema_version` if the new defaults would meaningfully reshuffle the suspect set on existing JSONs.

`mark_suspect_segments` mutates segments in place. Calling it on segments that were already flagged is idempotent (the existing `suspect: true` is overwritten with the same value); but if you change thresholds between calls, the previous flags are NOT cleared first. If that ever matters, normalize at the start of the function.

The auto-id default threshold `speaker_db.DEFAULT_THRESHOLD = 0.65` is the same kind of policy constant. Surfaced through `--id-threshold` on both binaries so users can tune for their cohort. The plan was: prefer false-negative (no match) over false-positive (wrong speaker). Adjust the default with the same caution as the suspect thresholds.

## Reference-project parity

These optimizations are intentionally preserved verbatim from `trascrivi.py` (only translated to English) and their behavior must not drift:

- **Anti-loop** (`--anti-loop`): sets exactly `condition_on_previous_text=False`, `compression_ratio_threshold=2.0`, `no_speech_threshold=0.5`. See `asr.build_transcribe_kwargs`.
- **Initial prompt** (mutex group): `--prompt` / `--prompt-file` / `--no-prompt`. `None` means "unset", `""` means "explicitly disabled".
- **Hotwords** (mutex group): same tri-state convention as the prompt group.
- **CUDA detection** (`asr.check_cuda_available`): probes `ctranslate2.get_supported_compute_types("cuda")` and prefers `float16`. Falls back to `int8` on CPU. Same logic as `trascrivi.py:15-22`.

If you find yourself "improving" any of these, double-check the upstream behavior first — they were chosen deliberately.

## CLI design rules

- **Four binaries, not one with subcommands**: `transcript-from-video` (full pipeline), `transcript-to-md` (torch-free re-render), `transcript-learn` (torch-free DB enrollment), `transcript-voices` (torch-free DB introspection/cleanup). All wired in `pyproject.toml` under `[project.scripts]`. The three torch-free ones extend invariant #1.
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
