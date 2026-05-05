"""Guard the architectural invariant: ``transcript-to-md`` must run without
torch / whisperX / pyannote installed.

This is the most important test in the suite. It blocks the heavy ML imports
via :data:`sys.meta_path` and exercises the full re-render path. If a future
refactor accidentally pulls a torch-dependent module into ``markdown.py`` (or
any of its transitive imports), this test fails immediately.

See CLAUDE.md > "Architecture: the four invariants" > #1.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

BLOCKED_PREFIXES = (
    "torch",
    "torchaudio",
    "whisperx",
    "pyannote",
    "transformers",
    "ctranslate2",
    "faster_whisper",
)

# Modules that must remain torch-free.
PROTECTED_MODULES = (
    "transcript_video.utils",
    "transcript_video.speakers",
    "transcript_video.markdown",
    "transcript_video.writers",
    "transcript_video.cli_to_md",
)


class _BlockingFinder:
    """A meta-path finder that raises ImportError for the blocked prefixes."""

    def find_spec(self, name, path=None, target=None):
        for prefix in BLOCKED_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                raise ImportError(
                    f"Heavy ML import '{name}' is blocked: this code path must remain torch-free."
                )
        return None


@pytest.fixture
def block_heavy_imports(monkeypatch):
    """Install the blocking finder and clear any already-imported protected modules.

    Yields nothing; just provides the contextual side-effects.
    """
    finder = _BlockingFinder()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])

    for name in list(sys.modules):
        if any(name == p or name.startswith(p + ".") for p in BLOCKED_PREFIXES):
            monkeypatch.delitem(sys.modules, name, raising=False)
        for protected in PROTECTED_MODULES:
            if name == protected:
                monkeypatch.delitem(sys.modules, name, raising=False)
    yield


def test_protected_modules_import_without_torch(block_heavy_imports):
    """All four torch-free modules must import cleanly under the block."""
    for name in PROTECTED_MODULES:
        importlib.import_module(name)


def test_render_markdown_runs_without_torch(block_heavy_imports, tmp_path: Path):
    """End-to-end: load a JSON transcript and emit Markdown with torch blocked."""
    from transcript_video.markdown import (
        load_transcript_json,
        render_markdown,
        write_markdown,
    )

    transcript = {
        "schema_version": 1,
        "source_file": "/path/to/sample.mp4",
        "transcribed_at": "2026-05-05T10:30:00",
        "parameters": {"backend": "whisperx", "model": "large-v3", "beam_size": 5},
        "audio_info": {"language_detected": "en", "duration_seconds": 90.0},
        "stats": {"num_segments": 1, "num_speakers": 1, "processing_seconds": 1.0},
        "segments": [
            {"id": 0, "start": 0.0, "end": 4.0, "text": "Hello there.", "speaker": "SPEAKER_00"},
        ],
    }
    json_path = tmp_path / "foo.json"
    json_path.write_text(json.dumps(transcript), encoding="utf-8")

    loaded = load_transcript_json(json_path)
    md = render_markdown(loaded, speaker_map={"SPEAKER_00": "Alice"})
    assert "## [00:00:00] Alice" in md
    assert "Hello there." in md

    out_path = tmp_path / "foo.md"
    write_markdown(md, out_path)
    assert out_path.read_text(encoding="utf-8") == md


def test_blocking_finder_actually_blocks(block_heavy_imports):
    """Sanity check: the finder really does raise on a blocked prefix."""
    with pytest.raises(ImportError, match="must remain torch-free"):
        importlib.import_module("torch")
