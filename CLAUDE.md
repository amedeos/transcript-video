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

`transcript-video` is the **English-language successor** of [transcript-italian-video](https://github.com/amedeos/transcript-italian-video). It transcribes and diarizes video/audio using whisperX (faster-whisper + wav2vec2 alignment + pyannote 3.x diarization) and produces JSON / SRT / TXT / Markdown. A second ASR engine, NVIDIA **Parakeet TDT 0.6B v3** on CPU through onnx-asr / ONNX Runtime (`parakeet.py`), is what the default `--model auto` runs when there is no float16 GPU: see "Second ASR engine: Parakeet on CPU" below.

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

Do **not** add imports from `asr`, `diarize`, `parakeet`, `pipeline`, `preflight`, `speaker_embed`, `whisperx`, `torch`, `pyannote`, `transformers`, `ctranslate2`, `faster_whisper`, `onnx_asr`, `onnxruntime`, or `numpy` to those modules. If you need a helper from there, lift it into `utils.py` / `stats.py` / `speaker_db.py` instead.

The auto-identification refresh in `cli_to_md.py --identify-speakers` is part of this set: cosine matching runs in pure Python (no numpy) against the JSON's cached `speaker_clusters`. See invariant #9.

### 2. JSON is the contract across all four binaries

The shape produced by `pipeline.py` is the input to `markdown.py`, `enrollment.py`, and the auto-id path in `cli_to_md.py`. Adding/renaming a field in one place requires the others. Bump `schema_version` in `pipeline.py` when you change the shape; have consumers tolerate older versions or fail with a clear message.

Top-level keys at the current schema version (`4`, `pipeline.SCHEMA_VERSION`):

- `schema_version`, `stage`, `source_file`, `transcribed_at`, `parameters`, `audio_info`, `stats`, `segments`, `full_text`
- `speaker_clusters` (since v2): per-cluster cached embeddings — empty when diarization is off or extraction failed.
- `speaker_identities` (since v2): per-cluster `{name, score, source}` — populated only when `--identify-speakers` is on, at pipeline time or re-render time.
- `parameters.batch_size` (since v3): the ASR batch that produced the segments. On a resume it is carried from the snapshot like every other ASR parameter (see invariant #5), and it is `null` when the snapshot is v2.
- Since v4 (the Parakeet engine): `parameters.model_revision` (the pinned HF commit for Parakeet, `null` for whisperX) and `parameters.vad` (Parakeet's Silero settings, `null` for whisperX); `parameters.backend` is `whisperx` or `onnx-asr`. Both engines write the same keys: what doesn't apply is `null`. For Parakeet that means `beam_size`, `language_forced`, prompt, hotwords, the anti-loop values, `suspect_thresholds`, and `audio_info.language_detected` / `language_probability` (Parakeet doesn't report the language). Parakeet segments carry `token_logprob` and words carry `logprob` (mean token log-probabilities), never `avg_logprob` / `score`: they mean something else. `_snapshot_asr_parameters` backfills `model_revision` / `vad` as `null` when resuming an older snapshot.

All field names are English (the explicit rename vs. the reference project, which used Italian field names like `file_sorgente`, `parametri`, `segmenti`).

### 3. Output policy is the reverse of the reference project

- The reference project always emitted TXT + SRT + JSON.
- Here, **JSON is always written**; **`--srt`, `--txt`, `--md` are opt-in**.

Don't restore the old behavior "for convenience".

### 4. Default language is autodetect

The reference project defaulted to `--language it`. **This project does not** — the user explicitly wanted no default. `--language` only sets a forced override; `parameters.language_forced` in the JSON is `null` whenever autodetect was used. Parakeet can't be forced at all: with it, `--language` is an error (see the Parakeet section).

### 5. Two-stage pipeline + `stage` field

The pipeline writes two persistence layers:

- `*_transcript.aligned.json` — written immediately after alignment, only when diarization is enabled. The free safety net: if diarization fails, ASR + alignment work is preserved.
- `*_transcript.json` — the canonical output, written after the full pipeline.

Both share the same schema. The top-level `stage` field discriminates them (`"aligned"` vs `"complete"`). When refactoring `_build_payload()`, keep both stages going through the same builder so they stay structurally identical.

`--resume-from-aligned PATH` skips ASR + alignment and feeds the snapshot straight to diarization. The aligned snapshot is the legitimate input form for resuming. `pipeline._load_aligned_payload` refuses a `*_transcript.json` that already carries speakers (segments, `speaker_clusters` or `speaker_identities`), because re-diarizing would clobber them. The one exception is a finished transcript with no speakers: a `--no-diarize` run writes no aligned snapshot, so its JSON is the only way to add diarization later.

On a resume, **everything under `parameters` except `diarization` is carried from the snapshot** (`pipeline._snapshot_asr_parameters`). Those fields describe the ASR that produced the segments, which did not run again: the current run's `--model`, `--language`, prompt or beam flags must not relabel them. Only `diarization` describes the current run. `_build_payload` takes the ASR block ready-made (`_asr_parameters` on a fresh run), so both stages still go through one builder. A resume with `--no-diarize` is rebuilt the same way (`_complete_without_diarization`), not copied: the result says diarization was off and has the current schema.

### 6. Pre-flight runs by default

`preflight.run_preflight(config)` is invoked at the start of every transcription unless `--no-check` is passed. The pre-flight tests we hit on real machines (HF token validity, gated-model access, whisperX API resolution) catch failures in 1–2 seconds instead of 5 minutes. When adding a new external dependency to the pipeline, add a pre-flight check for it. For ASR runs it also reports the engine (`asr_engine`, FAIL on options Parakeet can't apply) and, for Parakeet, checks onnx-asr and the model files (`parakeet`).

### 7. CLI flags always win over the config file

`project_config.toml` provides defaults via `parser.set_defaults(**)`; explicit CLI flags overwrite them at parse time. `set_defaults` skips `type=` and `choices` for non-string values, so both binaries go through `project_config.apply_to_parser`, which converts and checks each config value against its flag (int / float / bool / string / append / choices, case-insensitive choices, unquoted TOML dates as ISO text) and stops with a parser error naming the key. Add new semantic checks there or after parsing, never by reading the config again. Both binaries parse through `project_config.parse_args_with_config`: argparse only sees conflicts between flags typed on the command line, so when one member of a mutually exclusive group (`--prompt*`, `--hotwords*`, `-q`/`-v`) is given on the CLI, the members the config set go back to their built-in defaults. A config that sets two members of one group is an error. Don't introduce config keys that "force" a value (e.g. by reading the config AFTER `parser.parse_args`); the contract is one-directional. The order of resolution for the config itself is: `--config` explicit → `./transcript-video.toml` → `<input_dir>/transcript-video.toml` → none.

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

`stats.DEFAULT_AVG_LOGPROB_THRESHOLD = -1.0` and `DEFAULT_NO_SPEECH_PROB_THRESHOLD = 0.6` are tuning constants for whisperX large-v3. They are persisted in `parameters.suspect_thresholds` so a downstream consumer can tell *which* thresholds produced the flags. Don't change the defaults silently — bump `schema_version` if the new defaults would meaningfully reshuffle the suspect set on existing JSONs. The record is always the thresholds this run applied (`pipeline._applied_suspect_thresholds`), a resume included: the flags are recomputed, so a snapshot's copy is not carried. `null` means an ASR without confidences these thresholds can read: then nothing is flagged.

`mark_suspect_segments` mutates segments in place. Calling it on segments that were already flagged is idempotent (the existing `suspect: true` is overwritten with the same value); but if you change thresholds between calls, the previous flags are NOT cleared first. If that ever matters, normalize at the start of the function.

The auto-id default threshold `speaker_db.DEFAULT_THRESHOLD = 0.65` is the same kind of policy constant. Surfaced through `--id-threshold` on both binaries so users can tune for their cohort. The plan was: prefer false-negative (no match) over false-positive (wrong speaker). Adjust the default with the same caution as the suspect thresholds.

## Reference-project parity

These optimizations are intentionally preserved verbatim from `trascrivi.py` (only translated to English) and their behavior must not drift:

- **Anti-loop** (`--anti-loop`): sets exactly `condition_on_previous_text=False`, `compression_ratio_threshold=2.0`, `no_speech_threshold=0.5`. See `asr.build_asr_options`. Caveat: whisperX's batched decoding (3.8) reads none of the three. Each 30 s VAD chunk is decoded on its own, with no previous-text conditioning or temperature fallback, so the flag is kept for parity but has no effect today.
- **Decoding options go in at model load, not to `transcribe()`.** whisperX's batched `FasterWhisperPipeline.transcribe` accepts only audio / batch size / language / task / chunk size / progress arguments. Prompt, hotwords, beam size and the anti-loop values are `asr_options` of `whisperx.load_model` (`asr.build_asr_options` → `asr.load_whisperx_model`). Passing them to `transcribe()` raised `TypeError` on every `--prompt` / `--hotwords` / `--anti-loop` run. The test stub in `test_pipeline_orchestration.py` mirrors the real `transcribe` signature, so the bug can't come back silently.
- **Initial prompt** (mutex group): `--prompt` / `--prompt-file` / `--no-prompt`. `None` means "unset", `""` means "explicitly disabled".
- **Hotwords** (mutex group): same tri-state convention as the prompt group.
- **CUDA compute-type preference** (`asr.check_cuda_available`): like `trascrivi.py:15-22`, it reads `ctranslate2.get_supported_compute_types("cuda")` and prefers `float16`. The probe itself deliberately differs from upstream: it requires a CUDA device, is cached, and has a fallback order. Without float16 it falls back to `float32`, never to upstream's `int8`. See "CUDA probe is cached and requires a device" below, and don't restore the upstream probe.

If you find yourself "improving" any of these, double-check the upstream behavior first — they were chosen deliberately. All of them are whisperX options: with Parakeet they stop the run (see "Second ASR engine").

### Deliberate divergence: CPU runs (do not "restore" upstream)

- **CPU compute type is `float32`, not upstream's `int8`** (`asr.CPU_DEFAULT_COMPUTE_TYPE`). This is about whisperX (ctranslate2) on CPU, i.e. `--model large-v3` or another whisper name; the default `--model auto` runs Parakeet on CPU (see below), whose weights default to float32 too. The user wants CPU-only machines (e.g. a laptop with no NVIDIA/AMD GPU) to lose no accuracy compared with GPU `float16`. ctranslate2 has no `float16` on CPU, and `float32` is its lossless equivalent. `int8` stays available as an explicit `--compute-type int8`. Our `auto` and `default` both mean this lossless default (`asr.resolve_compute_type`). Never pass them through: ctranslate2's `auto` picks the fastest type (int8 on CPU), and whisperX rewrites `default` to float16/float32 without checking what the GPU supports. Resolving them here also makes the JSON record the type that actually ran. Only the names in `asr.COMPUTE_TYPES` are accepted (`--compute-type` has `choices`, case-insensitive on the CLI and in the config), so a typo or `INT8` can no longer fall through to the device default and silently lose a quantization asked for to save RAM. Any other explicit compute type the resolved device can't run is replaced, with a warning: `int8*` by the closest supported `int8*` type (e.g. `int8_float16` → `int8_float32` on CPU, so a quantized choice made for low RAM stays quantized), anything else by the device default.
- **Minimum GPU for the ASR: float16** (`asr.MIN_GPU_COMPUTE_TYPE`: compute capability ≥ 7.0). The user chose this over upstream's int8 fallback. `plan_devices` treats a ctranslate2 GPU without float16 (hint `float32`) as below the minimum. With `auto` the ASR then runs on CPU in float32 (lossless) and torch keeps the GPU. With explicit `cuda` the ASR uses that GPU in float32, because an explicit request is honored, not moved to CPU. Below-minimum GPUs never default to int8. With `--model auto` (the default) a GPU below the minimum means Parakeet on CPU instead, with a note; an explicit whisper model keeps the rule above.
- **GPU → CPU fallback only in `auto`.** An explicit `--device cuda` (CLI or config) with no usable GPU **stops the run** (pre-flight FAIL; `pipeline._resolve_devices` prints the error and exits before any model loads, even with `--no-check`). Only the stages that will run count: a resume needs no ctranslate2 CUDA, and a resume with `--no-diarize` needs no GPU at all. `device` from the config is validated (case-insensitively) by `project_config.apply_to_parser`, because argparse `choices` don't apply to config defaults. Never turn it into a silent CPU fallback: the user decided a GPU request must not quietly become hours of CPU work.
- **Two GPU users, probed separately, resolved once per run.** ASR uses ctranslate2 (`check_cuda_available`); alignment, diarization and embeddings use torch (`asr.torch_cuda_available`). `asr.plan_devices` is the single source of the device rules: a pure function (no exit) used by both the pre-flight `check_cuda` and `pipeline._resolve_devices`, so the two can't drift. It runs once before any heavy work, and every stage gets those values: no stage re-probes. With `auto` the libraries may differ (e.g. a CPU-only torch wheel on a GPU box puts the torch stages on CPU) and a warning is logged; with `cuda` both must be usable. **whisperX's pyannote VAD is a torch model that whisperX builds on the ASR device**: when ASR is on CUDA and torch can't use it, `asr.load_whisperx_model(vad_device="cpu")` builds the VAD on CPU and passes it as `vad_model` (whisperX still applies its own `vad_params`). If the installed whisperX can't take a `vad_model` (`asr.cpu_vad_supported`), the ASR moves to CPU too. On `--resume-from-aligned` the torch device comes from the current machine and config, not from the snapshot. The JSON's `parameters.device` / `compute_type` keep the ASR's values.
- **CUDA probe is cached and requires a device** (`asr._probe_cuda`). With no NVIDIA driver, ctranslate2's `get_supported_compute_types("cuda")` raises on the first call and returns `{"float32"}` on later calls. So the probe also requires `get_cuda_device_count() > 0` and runs once per process (`functools.lru_cache`). Never call ctranslate2's CUDA APIs directly elsewhere; go through `check_cuda_available` / `_supported_compute_types`. The CUDA hint is always one of the supported types.
- **ASR batch size is 4 on CPU, 16 on CUDA** (`asr.resolve_batch_size`, `CPU_DEFAULT_BATCH_SIZE` / `GPU_DEFAULT_BATCH_SIZE`). The old hard-coded 16 got a 32 GB laptop OOM-killed during ASR (large-v3, float32, beam 10). Measured on tiny/small: ASR peak RAM grows with the **batch** (each chunk carries its own encoder activations and cross-attention cache, extrapolated to about 0.65 GB per chunk for large-v3 float32), while `beam_size` makes no measurable difference. So the default doesn't depend on the beam. `--batch-size N` / config `batch_size` always wins. The batch only groups independent 30 s VAD chunks, so it doesn't change what gets decoded. But different matrix shapes can shift the floating-point results enough to flip a near-tie beam choice or nudge an `avg_logprob` (and so a suspect flag). That is why it is recorded in `parameters.batch_size` (schema v3). Don't raise the CPU default without re-measuring peak RAM (the `RAM after …` log lines) with large-v3 on a 32 GB machine. For Parakeet the batch is the number of VAD chunks per ONNX Runtime call (always the CPU default, 4).
- **Threads**: `--threads N` / config `threads`; an explicit value applies to every stage. Without it, `OMP_NUM_THREADS` then `MKL_NUM_THREADS` (`asr.env_thread_limit`, first level of an OpenMP list) count as an explicit value: the user chose to honor an environment limit, e.g. on shared servers. `tests/conftest.py` clears both variables for every test. The default is available CPUs − 1 (`asr.resolve_cpu_threads`). "Available" means affinity-aware logical CPUs, capped by a cgroup v1/v2 CPU quota (container `--cpus`, `asr._cgroup_cpu_limit`). The default is applied only to the libraries running on CPU (`pipeline._configure_threads`): ctranslate2 via `whisperx.load_model(threads=)`, torch via `torch.set_num_threads`. On the GPU, libraries keep their own defaults. whisperX's top-level `load_model` is a `*args, **kwargs` lazy wrapper, so `asr._load_model_accepts` introspects `whisperx.asr.load_model` instead. With Parakeet the ASR count goes to ONNX Runtime (`intra_op_num_threads`; Parakeet is always on CPU, so it always gets the default). On a hybrid Intel laptop (i7-1280P) 6 threads beat CPUs − 1 = 19 for whisperX; Parakeet was measured at 6 only, so laptops should set `threads = 6` until a P-core default exists.

### Second ASR engine: Parakeet on CPU (user decisions, 2026-11-04)

- **`--model auto` is the default** (`asr.requested_engine` / `asr._plan_auto`). It means whisperX `large-v3` when the whisperX plan puts the ASR on CUDA (a float16 GPU), or when `--device cuda` was requested (so an impossible GPU request stays an error, never a Parakeet run). Otherwise Parakeet. The choice follows the **whole** whisperX plan, not just the ctranslate2 probe: with a float16 GPU but a CPU-only torch and a whisperX that can't split its VAD off, the whisperX ASR would land on CPU, and on CPU the answer is Parakeet. `DevicePlan.engine` / `auto_reason` carry the result; the pre-flight and the pipeline use the same plan rules. `--model parakeet` (alias of `parakeet-tdt-0.6b-v3`) picks it anywhere; any other value is a whisper model.
- **Why**: on the user's laptop large-v3 runs at ~1.6x the audio length on CPU, Parakeet fp32 at 0.40x (agreement with large-v3: 89.4% of words on a 2-minute clip; it misses domain jargon such as "collaudo", and has no prompt to fix it). This overrides "CPU runs lose no accuracy" for the *choice of model* only. The user still compares the two on real audio (large-v3 on the GPU desktop vs `--model parakeet`).
- **Strict options** (`asr.parakeet_conflicts`, used by pre-flight `asr_engine` and by `pipeline._resolve_devices`): with Parakeet, `--language`, a non-empty prompt or hotwords, an explicit `--beam_size` (even 5) and `--anti-loop` **stop the run** before any model loads. Parakeet detects the language itself (it can't be told) and takes none of them. `""` (`--no-prompt` / `--no-hotwords`) is fine. The user chose errors over warn-and-continue. That is why `--beam_size` has no default (`None`, whisperX resolves it to 5). Not checked on a resume (no ASR runs).
- **Always CPU**: ONNX Runtime with `CPUExecutionProvider` forced for the model and the VAD (onnx-asr would otherwise pick onnxruntime-gpu or a platform provider and ignore the device plan). Diarization follows `--device` as usual. Parakeet needs no torch without diarization (`plan_devices(run_torch=...)` means torch work *besides* the ASR's own). `--device cuda` + `--model parakeet` + `--no-diarize` runs with a note: nothing uses the GPU (same "only stages that run count" rule as a no-diarize resume).
- **Weights** (`parakeet.resolve_weights`): float32 by default (lossless), int8 for every int8 type; `int16` gets int8 and `float16` / `bfloat16` get float32, with a warning. Recorded as `compute_type` `float32` / `int8`.
- **Files** (`parakeet.prepare_files`): Parakeet (`istupakov/parakeet-tdt-0.6b-v3-onnx`) and the Silero VAD at **pinned revisions** (`HF_REVISION`, `VAD_REVISION`), downloaded into the HF cache and then hard-linked (copied as a fallback) into per-revision folders next to it (`<HF_HUB_CACHE>/../transcript-video/<name>-<rev12>`). The reason: the HF cache keeps each file as a blob in a shared folder, so `encoder-model.onnx` and its external `encoder-model.onnx.data` resolve to different directories, and ONNX Runtime 1.31 refuses external data outside the model's folder ("External data path escapes model directory"). Next to the cache keeps hard links on one filesystem (containers mount `~/.cache/huggingface`). Temporary names are unique, so concurrent first runs don't collide. Bump a revision only with a re-measurement, and record it in `parameters.model_revision`.
- **Segmentation**: onnx-asr 0.12's Silero VAD with the measured settings: pauses shorter than 1000 ms merged, chunks closed at the last pause before 30 s, 30 ms padding. Merging short utterances matters: with onnx-asr's default 100 ms, short chunks slipped into English or Cyrillic. Continuous speech longer than 30 s, with no pause the VAD sees, is cut by onnx-asr with a 60 ms overlap. `parakeet.merge_overlaps` keeps each word once and logs the cuts. Chunks are split into sentence-level segments at final punctuation (closing quotes and brackets allowed) or at pauses of 1 s or more.
- **No alignment, no suspect flags**: Parakeet's token timestamps (80 ms steps) become the words, so there is no wav2vec2 stage. It has no `avg_logprob` / `no_speech_prob`, so `suspect_thresholds` is `null` and nothing is flagged (`pipeline._applied_suspect_thresholds`). The token logprobs are saved so a threshold can be calibrated later, from the GPU-vs-CPU comparison.
- **Dependency and pre-flight**: `onnx-asr[cpu,hub]>=0.12,<0.13` is a core dependency, because `auto` needs it on CPU. Pre-flight `parakeet` checks onnx-asr 0.12.x (`parakeet.ONNX_ASR_SERIES`), the CPU provider, and the model files: missing files are a download notice, and with network each one is looked up at its pinned revision. Any import error becomes a FAIL line. Audio is still decoded through `whisperx.load_audio`, which imports torch.

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
- **Shared embedding network** (`diarize._share_embedding_backbone`): pyannote's `get_embeddings` runs the whole WeSpeaker ResNet once per (window, local speaker). That is the same audio 3 times per window, with only the final pooling reading the speaker mask, and it is nearly all the diarization time on CPU. We swap the class of the pipeline's embedding wrapper (`DiarizationPipeline.model._embedding`) for a subclass that runs `forward_frames` once per distinct window in the batch and `forward_embedding` once per row. `embedding_batch_size` is **not** changed, so pooling and the linear layers see the same row batches as stock. A larger row batch changes float rounding (~4e-8, measured); same batches gave 0.0 in tests. The speed was measured on a laptop with an earlier variant (32 windows per call): community-1 went from 237 s to 65 s, DER 0.0% against stock. The final variant runs the same number of network passes in smaller batches. These are pyannote internals, so (1) the pieces are feature-detected; (2) a self-check compares shared and stock on a small random batch before use; (3) any error in the swap, the check or a shared call restores the stock class (WARNING log); and (4) the call site in `diarize_and_assign` is guarded too. The speed-up must never stop diarization. Don't remove these guards. `speaker-diarization-3.1` uses the same WeSpeaker ResNet under pyannote 4, so it is shared too. The proper long-term home is upstream pyannote: the split methods exist there, and the pipeline doesn't call them.

## License

GPL v3 (inherited from the reference project). Don't relicense without explicit user approval.
