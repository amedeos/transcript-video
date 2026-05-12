"""Tests for Markdown rendering: frontmatter, body, and segment grouping."""

from __future__ import annotations

import json
from datetime import date

import pytest

from transcript_video import markdown as md_mod
from transcript_video.markdown import (
    UNKNOWN_SPEAKER,
    _build_asr_label,
    _collect_speakers,
    _group_segments,
    _resolved_language,
    _yaml_inline_value,
    build_effective_speaker_map,
    load_transcript_json,
    render_body,
    render_frontmatter,
    render_markdown,
    split_into_paragraphs,
    write_markdown,
)


def _base_transcript(**overrides) -> dict:
    """Build a minimal but complete transcript dict for tests."""
    base = {
        "schema_version": 1,
        "source_file": "/path/to/meeting-foo.mp4",
        "transcribed_at": "2026-05-01T10:30:00",
        "parameters": {
            "backend": "whisperx",
            "model": "large-v3",
            "beam_size": 10,
            "language_forced": None,
        },
        "audio_info": {
            "language_detected": "it",
            "language_probability": 0.98,
            "duration_seconds": 6135.0,  # 01:42:15 — the example from the spec
        },
        "stats": {"num_segments": 4, "num_speakers": 2, "processing_seconds": 200.0},
        "segments": [
            {"id": 0, "start": 12.0, "end": 14.5, "text": "Allora, oggi parliamo della migrazione", "speaker": "SPEAKER_00"},
            {"id": 1, "start": 15.0, "end": 17.5, "text": "da SDN a OVN-Kubernetes...", "speaker": "SPEAKER_00"},
            {"id": 2, "start": 105.0, "end": 107.0, "text": "Sì, il problema che vediamo è...", "speaker": "SPEAKER_01"},
            {"id": 3, "start": 108.0, "end": 110.0, "text": "la latenza tra i pod.", "speaker": "SPEAKER_01"},
        ],
    }
    base.update(overrides)
    return base


class TestYamlInlineValue:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("it", "it"),
            ("hello", "hello"),
            ("meeting-foo.mp4", "meeting-foo.mp4"),
            ("whisperx-large-v3", "whisperx-large-v3"),
        ],
    )
    def test_safe_values_emitted_bare(self, value, expected):
        assert _yaml_inline_value(value) == expected

    @pytest.mark.parametrize(
        "value",
        ["true", "false", "null", "yes", "no", "on", "off", "~", "TRUE", "Yes"],
    )
    def test_yaml_reserved_words_quoted(self, value):
        assert _yaml_inline_value(value).startswith('"')

    @pytest.mark.parametrize("value", ["1.5", "42", "-3.14"])
    def test_numeric_strings_quoted(self, value):
        assert _yaml_inline_value(value).startswith('"')

    @pytest.mark.parametrize("value", ["has: colon", "has,comma", "has[bracket"])
    def test_special_chars_quoted(self, value):
        assert _yaml_inline_value(value).startswith('"')

    def test_empty_string_quoted(self):
        assert _yaml_inline_value("") == '""'

    def test_quotes_escaped(self):
        assert _yaml_inline_value('he said "hi"') == '"he said \\"hi\\""'


class TestBuildAsrLabel:
    def test_typical(self):
        assert _build_asr_label({"backend": "whisperx", "model": "large-v3"}) == "whisperx-large-v3"

    def test_default_backend(self):
        assert _build_asr_label({"model": "medium"}) == "whisperx-medium"

    def test_no_model(self):
        assert _build_asr_label({"backend": "whisperx"}) == "whisperx"


class TestResolvedLanguage:
    def test_forced_wins(self):
        assert _resolved_language({"language_forced": "en"}, {"language_detected": "it"}) == "en"

    def test_falls_back_to_detected(self):
        assert _resolved_language({"language_forced": None}, {"language_detected": "it"}) == "it"

    def test_unknown_when_neither(self):
        assert _resolved_language({}, {}) == "unknown"


class TestCollectSpeakers:
    def test_preserves_first_seen_order(self):
        segs = [
            {"speaker": "SPEAKER_01"},
            {"speaker": "SPEAKER_00"},
            {"speaker": "SPEAKER_01"},  # duplicate ignored
            {"speaker": "SPEAKER_02"},
        ]
        assert _collect_speakers(segs) == ["SPEAKER_01", "SPEAKER_00", "SPEAKER_02"]

    def test_skips_missing_labels(self):
        segs = [{"speaker": "SPEAKER_00"}, {"text": "no speaker"}, {"speaker": None}]
        assert _collect_speakers(segs) == ["SPEAKER_00"]


class TestGroupSegments:
    def test_merges_within_gap(self):
        segs = [
            {"start": 0.0, "end": 2.0, "text": "Hello", "speaker": "S0"},
            {"start": 2.5, "end": 4.0, "text": "world.", "speaker": "S0"},
        ]
        blocks = _group_segments(segs, merge_gap_seconds=1.5)
        assert len(blocks) == 1
        assert blocks[0]["text"] == "Hello world."
        assert blocks[0]["end"] == 4.0

    def test_does_not_merge_beyond_gap(self):
        segs = [
            {"start": 0.0, "end": 2.0, "text": "Hello.", "speaker": "S0"},
            {"start": 5.0, "end": 6.0, "text": "Later.", "speaker": "S0"},
        ]
        blocks = _group_segments(segs, merge_gap_seconds=1.5)
        assert len(blocks) == 2

    def test_does_not_merge_across_speakers(self):
        segs = [
            {"start": 0.0, "end": 2.0, "text": "A.", "speaker": "S0"},
            {"start": 2.1, "end": 3.0, "text": "B.", "speaker": "S1"},
        ]
        blocks = _group_segments(segs, merge_gap_seconds=1.5)
        assert len(blocks) == 2
        assert blocks[0]["speaker"] == "S0"
        assert blocks[1]["speaker"] == "S1"

    def test_merge_gap_zero_disables_merging(self):
        # Even when the gap is 0, merge_gap_seconds=0 means "never merge".
        segs = [
            {"start": 0.0, "end": 2.0, "text": "A.", "speaker": "S0"},
            {"start": 2.0, "end": 3.0, "text": "B.", "speaker": "S0"},
        ]
        blocks = _group_segments(segs, merge_gap_seconds=0.0)
        assert len(blocks) == 2

    def test_skips_empty_text(self):
        segs = [
            {"start": 0.0, "end": 2.0, "text": "", "speaker": "S0"},
            {"start": 3.0, "end": 4.0, "text": "  ", "speaker": "S0"},
            {"start": 5.0, "end": 6.0, "text": "Hi", "speaker": "S0"},
        ]
        blocks = _group_segments(segs, merge_gap_seconds=1.5)
        assert len(blocks) == 1
        assert blocks[0]["text"] == "Hi"

    def test_missing_speaker_bucketed_as_unknown(self):
        segs = [{"start": 0.0, "end": 2.0, "text": "Hi"}]
        blocks = _group_segments(segs, merge_gap_seconds=1.5)
        assert blocks[0]["speaker"] == UNKNOWN_SPEAKER


class TestRenderFrontmatter:
    def test_matches_user_spec(self):
        transcript = _base_transcript()
        fm = render_frontmatter(
            transcript,
            speaker_map={"SPEAKER_00": "Amedeo", "SPEAKER_01": "Tizio"},
            fm_date="2026-05-01",
            tags=["openshift", "ovn-kubernetes", "troubleshooting"],
            fm_source="meeting-foo.mp4",
        )
        # Field order and values match the user's spec verbatim.
        expected = (
            "---\n"
            "date: 2026-05-01\n"
            'duration: "01:42:15"\n'
            "language: it\n"
            "source: meeting-foo.mp4\n"
            "asr: whisperx-large-v3\n"
            "beam_size: 10\n"
            "speakers:\n"
            "  SPEAKER_00: Amedeo\n"
            "  SPEAKER_01: Tizio\n"
            "tags: [openshift, ovn-kubernetes, troubleshooting]\n"
            "---\n"
        )
        assert fm == expected

    def test_default_date_is_today(self):
        fm = render_frontmatter(
            _base_transcript(), speaker_map={}, fm_date=None, tags=[], fm_source=None
        )
        assert f"date: {date.today().isoformat()}\n" in fm

    def test_default_source_is_input_basename(self):
        fm = render_frontmatter(
            _base_transcript(), speaker_map={}, fm_date="2026-05-01", tags=[], fm_source=None
        )
        assert "source: meeting-foo.mp4" in fm

    def test_empty_tags(self):
        fm = render_frontmatter(
            _base_transcript(), speaker_map={}, fm_date="2026-05-01", tags=[], fm_source=None
        )
        assert "tags: []\n" in fm

    def test_forced_language_wins(self):
        t = _base_transcript()
        t["parameters"]["language_forced"] = "en"
        fm = render_frontmatter(t, speaker_map={}, fm_date="2026-05-01", tags=[], fm_source=None)
        assert "language: en" in fm

    def test_unmapped_speaker_keeps_label(self):
        # SPEAKER_02 has no entry in the map → falls back to its raw label.
        t = _base_transcript()
        t["segments"].append({"start": 200.0, "end": 201.0, "text": "...", "speaker": "SPEAKER_02"})
        fm = render_frontmatter(
            t, speaker_map={"SPEAKER_00": "Amedeo"}, fm_date="2026-05-01", tags=[], fm_source=None
        )
        assert "  SPEAKER_02: SPEAKER_02" in fm

    def test_diarization_disabled_falls_back_to_unknown(self):
        t = _base_transcript()
        for seg in t["segments"]:
            seg.pop("speaker", None)
        fm = render_frontmatter(t, speaker_map={}, fm_date="2026-05-01", tags=[], fm_source=None)
        assert f"  {UNKNOWN_SPEAKER}: {UNKNOWN_SPEAKER}" in fm

    def test_omits_beam_size_when_missing(self):
        t = _base_transcript()
        t["parameters"].pop("beam_size", None)
        fm = render_frontmatter(t, speaker_map={}, fm_date="2026-05-01", tags=[], fm_source=None)
        assert "beam_size" not in fm


class TestRenderBody:
    def test_matches_user_spec(self):
        transcript = _base_transcript()
        body = render_body(
            transcript["segments"],
            speaker_map={"SPEAKER_00": "Amedeo", "SPEAKER_01": "Tizio"},
            merge_gap_seconds=1.5,
        )
        # Two blocks (one per speaker), each with the merged text.
        assert "## [00:00:12] Amedeo\nAllora, oggi parliamo della migrazione da SDN a OVN-Kubernetes...\n" in body
        assert "## [00:01:45] Tizio\nSì, il problema che vediamo è... la latenza tra i pod.\n" in body

    def test_empty_segments_returns_empty_string(self):
        assert render_body([], speaker_map={}) == ""

    def test_mark_suspect_off_keeps_body_clean(self):
        segs = [
            {"speaker": "A", "start": 0.0, "end": 5.0, "text": "Maybe wrong", "suspect": True},
            {"speaker": "A", "start": 5.0, "end": 10.0, "text": "Definitely right"},
        ]
        body = render_body(segs, speaker_map={}, merge_gap_seconds=1.5)
        assert "[?]" not in body

    def test_mark_suspect_inline_marker(self):
        segs = [
            {"speaker": "A", "start": 0.0, "end": 5.0, "text": "Definitely right"},
            {"speaker": "A", "start": 5.0, "end": 10.0, "text": "Maybe wrong", "suspect": True},
            {"speaker": "A", "start": 10.0, "end": 15.0, "text": "Right again"},
        ]
        body = render_body(segs, speaker_map={}, merge_gap_seconds=1.5, mark_suspect=True)
        # Single merged block, with the marker preserving position of the suspect span.
        assert body.count("## [") == 1
        assert "Definitely right [?] Maybe wrong Right again" in body

    def test_mark_suspect_no_op_when_no_suspect_segments(self):
        segs = [{"speaker": "A", "start": 0.0, "end": 5.0, "text": "Hello"}]
        body = render_body(segs, speaker_map={}, merge_gap_seconds=1.5, mark_suspect=True)
        assert "[?]" not in body


class TestRenderMarkdown:
    def test_full_document(self):
        transcript = _base_transcript()
        md = render_markdown(
            transcript,
            speaker_map={"SPEAKER_00": "Amedeo", "SPEAKER_01": "Tizio"},
            fm_date="2026-05-01",
            tags=["openshift"],
            fm_source="meeting-foo.mp4",
        )
        assert md.startswith("---\n")
        # Frontmatter ends with --- then a blank line then the body.
        assert "\n---\n\n## [" in md

    def test_no_segments_still_emits_frontmatter(self):
        t = _base_transcript()
        t["segments"] = []
        md = render_markdown(t, speaker_map={}, fm_date="2026-05-01", tags=[], fm_source=None)
        assert md.startswith("---\n")
        assert md.rstrip().endswith("---")

    def test_speaker_identities_used_as_fallback(self):
        """When the JSON carries speaker_identities and the CLI passes no map,
        the identity names are used for headings and frontmatter."""
        t = _base_transcript()
        t["speaker_identities"] = {
            "SPEAKER_00": {"name": "Amedeo", "score": 0.81, "source": "auto"},
            "SPEAKER_01": {"name": "Tizio", "score": 0.77, "source": "auto"},
        }
        md = render_markdown(t, speaker_map={}, fm_date="2026-05-01", tags=[], fm_source=None)
        assert "## [00:00:12] Amedeo" in md
        assert "## [00:01:45] Tizio" in md
        # Frontmatter also reflects auto-id.
        assert "SPEAKER_00: Amedeo" in md
        assert "SPEAKER_01: Tizio" in md

    def test_cli_speaker_map_overrides_speaker_identities(self):
        """The CLI/config speaker_map keeps its final-authority semantics."""
        t = _base_transcript()
        t["speaker_identities"] = {
            "SPEAKER_00": {"name": "Amedeo", "score": 0.81, "source": "auto"},
            "SPEAKER_01": {"name": "Tizio", "score": 0.77, "source": "auto"},
        }
        md = render_markdown(
            t,
            speaker_map={"SPEAKER_00": "Mario"},  # overrides "Amedeo"
            fm_date="2026-05-01", tags=[], fm_source=None,
        )
        # SPEAKER_00 → CLI wins.
        assert "## [00:00:12] Mario" in md
        assert "Amedeo" not in md
        # SPEAKER_01 → identity fallback still kicks in.
        assert "## [00:01:45] Tizio" in md

    def test_no_identities_field_behaves_as_before(self):
        """Backwards compatibility: transcripts without speaker_identities render exactly as before."""
        t = _base_transcript()
        assert "speaker_identities" not in t
        md = render_markdown(
            t,
            speaker_map={"SPEAKER_00": "Mario"},
            fm_date="2026-05-01", tags=[], fm_source=None,
        )
        assert "## [00:00:12] Mario" in md
        # SPEAKER_01 has no mapping, so the raw label survives.
        assert "## [00:01:45] SPEAKER_01" in md


class TestBuildEffectiveSpeakerMap:
    def test_empty_inputs_return_empty(self):
        assert build_effective_speaker_map({}, None) == {}
        assert build_effective_speaker_map({}, {}) == {}

    def test_speaker_identities_only(self):
        t = {"speaker_identities": {
            "SPEAKER_00": {"name": "Mario", "score": 0.8, "source": "auto"},
            "SPEAKER_01": {"name": "Luca", "score": 0.9, "source": "manual"},
        }}
        result = build_effective_speaker_map(t, None)
        assert result == {"SPEAKER_00": "Mario", "SPEAKER_01": "Luca"}

    def test_cli_map_wins_on_overlap(self):
        t = {"speaker_identities": {
            "SPEAKER_00": {"name": "Mario", "score": 0.8, "source": "auto"},
        }}
        result = build_effective_speaker_map(t, {"SPEAKER_00": "Luca"})
        assert result == {"SPEAKER_00": "Luca"}

    def test_cli_extends_identities(self):
        t = {"speaker_identities": {
            "SPEAKER_00": {"name": "Mario", "source": "auto"},
        }}
        result = build_effective_speaker_map(t, {"SPEAKER_99": "Ghost"})
        assert result == {"SPEAKER_00": "Mario", "SPEAKER_99": "Ghost"}

    def test_malformed_identity_skipped(self):
        t = {"speaker_identities": {
            "SPEAKER_00": {"name": "Mario", "source": "auto"},
            "SPEAKER_01": "not-a-dict",
            "SPEAKER_02": {"source": "auto"},  # no name
            "SPEAKER_03": {"name": "", "source": "auto"},  # empty name
        }}
        result = build_effective_speaker_map(t, None)
        assert result == {"SPEAKER_00": "Mario"}

    def test_missing_field_safe(self):
        # Transcript with no speaker_identities at all.
        assert build_effective_speaker_map({"foo": "bar"}, {"S": "Mario"}) == {"S": "Mario"}


class TestLoadTranscriptJson:
    def test_round_trip(self, tmp_path):
        transcript = _base_transcript()
        p = tmp_path / "t.json"
        p.write_text(json.dumps(transcript), encoding="utf-8")
        loaded = load_transcript_json(p)
        assert loaded["audio_info"]["duration_seconds"] == 6135.0


class TestWriteMarkdown:
    def test_writes_and_creates_dirs(self, tmp_path):
        out = tmp_path / "deep" / "out.md"
        write_markdown("# hi\n", out)
        assert out.read_text(encoding="utf-8") == "# hi\n"


class TestModuleLevelFlags:
    def test_unknown_speaker_constant(self):
        # Sanity check that the constant is exposed for tests.
        assert md_mod.UNKNOWN_SPEAKER == "Unknown"


class TestSplitIntoParagraphs:
    def test_no_split_below_threshold(self):
        text = "Short sentence. Another short one."
        assert split_into_paragraphs(text, max_chars=400) == [text]

    def test_split_when_long(self):
        # Each sentence is ~50 chars; threshold 100 → ~2 sentences per paragraph.
        sentences = ". ".join([f"Sentence number {i} with enough words to be real" for i in range(6)]) + "."
        paragraphs = split_into_paragraphs(sentences, max_chars=100)
        assert len(paragraphs) >= 3
        # Every paragraph must end with sentence-final punctuation.
        for p in paragraphs:
            assert p.rstrip().endswith((".", "!", "?"))

    def test_single_long_sentence_kept_whole(self):
        # No way to split; return as a single paragraph even if it overflows.
        text = "This single sentence is longer than the threshold but cannot be split."
        out = split_into_paragraphs(text, max_chars=20)
        assert out == [text]

    def test_zero_max_disables_splitting(self):
        text = ("Sentence one. " * 100).strip()
        assert split_into_paragraphs(text, max_chars=0) == [text]

    def test_negative_max_disables_splitting(self):
        text = "Sentence one. Sentence two."
        assert split_into_paragraphs(text, max_chars=-1) == [text]

    def test_empty_text(self):
        assert split_into_paragraphs("", max_chars=400) == []
        assert split_into_paragraphs("   ", max_chars=400) == []


class TestRenderBodyParagraphSplitting:
    def test_long_block_split_by_default(self):
        # A single 1000-char block from one speaker.
        long_text = ". ".join(
            [f"Frase numero {i}, contenente abbastanza parole per essere realistica e leggibile" for i in range(15)]
        ) + "."
        segs = [{"speaker": "A", "start": 0.0, "end": 100.0, "text": long_text}]
        body = render_body(segs, speaker_map={}, merge_gap_seconds=1.5, paragraph_chars=400)
        # Single heading, multiple paragraphs (separated by blank lines).
        assert body.count("## [") == 1
        assert "\n\n" in body  # paragraph separator

    def test_paragraph_chars_zero_no_split(self):
        long_text = ". ".join([f"Frase {i}" for i in range(50)]) + "."
        segs = [{"speaker": "A", "start": 0.0, "end": 10.0, "text": long_text}]
        body = render_body(segs, speaker_map={}, merge_gap_seconds=1.5, paragraph_chars=0)
        # No paragraph splits inside the block: only the trailing blank line
        # separating block from EOF, no double newline within the block body.
        # Strip the trailing "\n" then the final "\n" from rstrip and check.
        block_lines = body.rstrip().split("\n")
        # Lines: heading + single paragraph = 2 lines.
        assert len(block_lines) == 2
