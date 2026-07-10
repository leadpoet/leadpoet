"""W1 source-access v2: ranged reads + range-aware dedupe (sourceexperiments.md §2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.research_lab.code_build import (
    FULL_FILE_RANGE_END,
    ParentImageSourceContext,
    _read_source_file_for_model,
    resolve_source_inspection_requests,
)
from research_lab.code_editing import (
    CodeEditSourceInspectionRequest,
    build_code_edit_source_inspection_messages,
)


def _write_numbered_file(path: Path, line_count: int, *, prefix: str = "line") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{prefix} {index} of {path.name}\n" for index in range(1, line_count + 1)),
        encoding="utf-8",
    )


def _source_context(tmp_path: Path, editable_files: tuple[str, ...]) -> ParentImageSourceContext:
    return ParentImageSourceContext(
        source_root=tmp_path,
        source_mode="parent_image_extract",
        parent_image_digest_hash="sha256:test-digest",
        source_tree_hash="sha256:test-tree",
        top_level_paths=("sourcing_model/",),
        editable_files=editable_files,
        file_previews=(),
    )


def _read(path: str, *, start_line: int = 0, max_lines: int = 0) -> CodeEditSourceInspectionRequest:
    return CodeEditSourceInspectionRequest(
        operation="read_file", path=path, rationale="test", start_line=start_line, max_lines=max_lines
    )


@pytest.fixture()
def big_file_context(tmp_path: Path) -> ParentImageSourceContext:
    _write_numbered_file(tmp_path / "sourcing_model" / "big.py", 900)
    return _source_context(tmp_path, ("sourcing_model/big.py",))


class TestRangedReadHappyPath:
    def test_ranged_read_returns_requested_slice_with_full_file_metadata(self, big_file_context):
        batch = resolve_source_inspection_requests(
            big_file_context,
            [_read("sourcing_model/big.py", start_line=400, max_lines=50)],
            already_read_paths=(),
            max_files=12,
            max_file_bytes=24_000,
            max_total_bytes=120_000,
            max_search_matches=30,
            source_access_v2=True,
        )
        (result,) = batch.model_context["results"]
        assert result["start_line"] == 400
        assert result["end_line"] == 449
        assert result["line_count"] == 50
        assert result["total_line_count"] == 900
        assert result["truncated"] is True  # file continues past line 449
        assert result["range_truncated"] is False  # the requested range fit
        assert result["content"].startswith("line 400 of big.py\n")
        assert result["content"].rstrip("\n").endswith("line 449 of big.py")
        assert batch.read_ranges == {"sourcing_model/big.py": ((400, 449),)}

    def test_byte_cap_applies_to_the_slice_at_line_granularity(self, big_file_context):
        batch = resolve_source_inspection_requests(
            big_file_context,
            [_read("sourcing_model/big.py", start_line=1, max_lines=900)],
            already_read_paths=(),
            max_files=12,
            max_file_bytes=600,  # ~30 lines worth
            max_total_bytes=120_000,
            max_search_matches=30,
            source_access_v2=True,
        )
        (result,) = batch.model_context["results"]
        assert result["range_truncated"] is True
        assert result["truncated"] is True
        # No split lines: content ends with a complete line.
        assert result["content"].endswith("\n")
        assert result["end_line"] == result["line_count"]
        assert result["bytes_returned"] <= 600

    def test_whole_file_read_under_cap_reports_untruncated(self, tmp_path):
        _write_numbered_file(tmp_path / "sourcing_model" / "small.py", 10)
        context = _source_context(tmp_path, ("sourcing_model/small.py",))
        batch = resolve_source_inspection_requests(
            context,
            [_read("sourcing_model/small.py")],
            already_read_paths=(),
            max_files=12,
            max_file_bytes=24_000,
            max_total_bytes=120_000,
            max_search_matches=30,
            source_access_v2=True,
        )
        (result,) = batch.model_context["results"]
        assert result["truncated"] is False
        assert result["total_line_count"] == 10
        assert result["end_line"] == 10
        # Full-file coverage recorded with the sentinel so unranged repeats skip.
        assert batch.read_ranges["sourcing_model/small.py"] == ((1, FULL_FILE_RANGE_END),)

    def test_start_line_past_eof_returns_empty_content_with_real_line_count(self, tmp_path):
        _write_numbered_file(tmp_path / "sourcing_model" / "small.py", 10)
        context = _source_context(tmp_path, ("sourcing_model/small.py",))
        batch = resolve_source_inspection_requests(
            context,
            [_read("sourcing_model/small.py", start_line=5000, max_lines=50)],
            already_read_paths=(),
            max_files=12,
            max_file_bytes=24_000,
            max_total_bytes=120_000,
            max_search_matches=30,
            source_access_v2=True,
        )
        (result,) = batch.model_context["results"]
        assert result["content"] == ""
        assert result["line_count"] == 0
        assert result["total_line_count"] == 10


class TestRangeAwareDedupe:
    def test_disjoint_range_reread_of_same_path_is_allowed(self, big_file_context):
        first = resolve_source_inspection_requests(
            big_file_context,
            [_read("sourcing_model/big.py", start_line=1, max_lines=100)],
            already_read_paths=(),
            max_files=12,
            max_file_bytes=24_000,
            max_total_bytes=120_000,
            max_search_matches=30,
            source_access_v2=True,
        )
        second = resolve_source_inspection_requests(
            big_file_context,
            [_read("sourcing_model/big.py", start_line=500, max_lines=100)],
            already_read_paths=first.read_paths,
            already_read_ranges=first.read_ranges,
            max_files=12,
            max_file_bytes=24_000,
            max_total_bytes=120_000,
            max_search_matches=30,
            source_access_v2=True,
        )
        (result,) = second.model_context["results"]
        assert result["start_line"] == 500
        assert result["end_line"] == 599
        assert second.read_ranges["sourcing_model/big.py"] == ((1, 100), (500, 599))

    def test_contained_range_reread_skips_without_consuming_budget(self, big_file_context):
        first = resolve_source_inspection_requests(
            big_file_context,
            [_read("sourcing_model/big.py", start_line=100, max_lines=200)],
            already_read_paths=(),
            max_files=12,
            max_file_bytes=24_000,
            max_total_bytes=120_000,
            max_search_matches=30,
            source_access_v2=True,
        )
        second = resolve_source_inspection_requests(
            big_file_context,
            [_read("sourcing_model/big.py", start_line=150, max_lines=50)],
            already_read_paths=first.read_paths,
            already_read_ranges=first.read_ranges,
            max_files=12,
            max_file_bytes=24_000,
            max_total_bytes=120_000,
            max_search_matches=30,
            source_access_v2=True,
        )
        assert second.model_context["results"] == []
        assert second.bytes_returned == 0
        skipped = [item for item in second.event_doc["results"] if item.get("skipped")]
        assert skipped and skipped[0]["skipped"] == "range_already_read"

    def test_ranges_per_path_cap_blocks_a_fourth_range(self, big_file_context):
        covered = {"sourcing_model/big.py": ((1, 50), (100, 150), (200, 250))}
        batch = resolve_source_inspection_requests(
            big_file_context,
            [_read("sourcing_model/big.py", start_line=700, max_lines=50)],
            already_read_paths=("sourcing_model/big.py",),
            already_read_ranges=covered,
            max_files=12,
            max_file_bytes=24_000,
            max_total_bytes=120_000,
            max_search_matches=30,
            source_access_v2=True,
            max_ranges_per_path=3,
        )
        assert batch.model_context["results"] == []
        skipped = [item for item in batch.event_doc["results"] if item.get("skipped")]
        assert skipped and skipped[0]["skipped"] == "max_ranges_reached"

    def test_unranged_repeat_of_fully_read_file_skips(self, tmp_path):
        _write_numbered_file(tmp_path / "sourcing_model" / "small.py", 10)
        context = _source_context(tmp_path, ("sourcing_model/small.py",))
        first = resolve_source_inspection_requests(
            context,
            [_read("sourcing_model/small.py")],
            already_read_paths=(),
            max_files=12,
            max_file_bytes=24_000,
            max_total_bytes=120_000,
            max_search_matches=30,
            source_access_v2=True,
        )
        second = resolve_source_inspection_requests(
            context,
            [_read("sourcing_model/small.py")],
            already_read_paths=first.read_paths,
            already_read_ranges=first.read_ranges,
            max_files=12,
            max_file_bytes=24_000,
            max_total_bytes=120_000,
            max_search_matches=30,
            source_access_v2=True,
        )
        skipped = [item for item in second.event_doc["results"] if item.get("skipped")]
        assert skipped and skipped[0]["skipped"] == "range_already_read"

    def test_deeper_range_after_byte_truncated_full_read_is_allowed(self, big_file_context):
        # Aggravator A regression: the prompt says "request the deeper range you
        # still need" — that follow-up must return content, not skip.
        first = resolve_source_inspection_requests(
            big_file_context,
            [_read("sourcing_model/big.py")],
            already_read_paths=(),
            max_files=12,
            max_file_bytes=2_000,  # forces byte truncation ~line 100
            max_total_bytes=120_000,
            max_search_matches=30,
            source_access_v2=True,
        )
        (head,) = first.model_context["results"]
        assert head["truncated"] is True
        follow_up_start = head["end_line"] + 1
        second = resolve_source_inspection_requests(
            big_file_context,
            [_read("sourcing_model/big.py", start_line=follow_up_start, max_lines=100)],
            already_read_paths=first.read_paths,
            already_read_ranges=first.read_ranges,
            max_files=12,
            max_file_bytes=24_000,
            max_total_bytes=120_000,
            max_search_matches=30,
            source_access_v2=True,
        )
        (deeper,) = second.model_context["results"]
        assert deeper["start_line"] == follow_up_start
        assert deeper["line_count"] == 100

    def test_max_files_counts_distinct_paths_not_ranges(self, tmp_path):
        _write_numbered_file(tmp_path / "sourcing_model" / "a.py", 50)
        _write_numbered_file(tmp_path / "sourcing_model" / "b.py", 50)
        context = _source_context(tmp_path, ("sourcing_model/a.py", "sourcing_model/b.py"))
        batch = resolve_source_inspection_requests(
            context,
            [
                _read("sourcing_model/a.py", start_line=1, max_lines=10),
                _read("sourcing_model/a.py", start_line=20, max_lines=10),
                _read("sourcing_model/b.py", start_line=1, max_lines=10),
            ],
            already_read_paths=(),
            max_files=1,
            max_file_bytes=24_000,
            max_total_bytes=120_000,
            max_search_matches=30,
            source_access_v2=True,
        )
        # Two ranges on a.py both resolve (one distinct path); b.py hits max_files.
        assert len(batch.model_context["results"]) == 2
        skipped = [item for item in batch.event_doc["results"] if item.get("skipped")]
        assert skipped and skipped[0]["path"] == "sourcing_model/b.py"
        assert skipped[0]["skipped"] == "max_files_reached"


class TestLegacyByteIdentical:
    def test_v2_off_matches_legacy_read_shape_exactly(self, big_file_context):
        batch = resolve_source_inspection_requests(
            big_file_context,
            [_read("sourcing_model/big.py", start_line=400, max_lines=50)],
            already_read_paths=(),
            max_files=12,
            max_file_bytes=2_000,
            max_total_bytes=120_000,
            max_search_matches=30,
            source_access_v2=False,
        )
        (result,) = batch.model_context["results"]
        # Legacy: range fields ignored, byte-clipped from the top of the file.
        assert "start_line" not in result
        assert "end_line" not in result
        assert "total_line_count" not in result
        assert result["bytes_returned"] == 2_000
        assert result["content"].startswith("line 1 of big.py")
        assert result["truncated"] is True
        assert batch.read_ranges == {}
        assert "read_ranges" not in batch.model_context

    def test_v2_off_repeat_read_still_skips_as_already_read(self, big_file_context):
        first = resolve_source_inspection_requests(
            big_file_context,
            [_read("sourcing_model/big.py")],
            already_read_paths=(),
            max_files=12,
            max_file_bytes=24_000,
            max_total_bytes=120_000,
            max_search_matches=30,
        )
        second = resolve_source_inspection_requests(
            big_file_context,
            [_read("sourcing_model/big.py", start_line=500, max_lines=100)],
            already_read_paths=first.read_paths,
            max_files=12,
            max_file_bytes=24_000,
            max_total_bytes=120_000,
            max_search_matches=30,
        )
        skipped = [item for item in second.event_doc["results"] if item.get("skipped")]
        assert skipped and skipped[0]["skipped"] == "already_read"

    def test_default_call_without_v2_kwargs_is_unchanged(self, tmp_path):
        _write_numbered_file(tmp_path / "sourcing_model" / "small.py", 5)
        context = _source_context(tmp_path, ("sourcing_model/small.py",))
        batch = resolve_source_inspection_requests(
            context,
            [_read("sourcing_model/small.py")],
            already_read_paths=(),
            max_files=8,
            max_file_bytes=24_000,
            max_total_bytes=120_000,
            max_search_matches=30,
        )
        (result,) = batch.model_context["results"]
        # Legacy line_count quirk (finding 6) intentionally preserved when v2 off:
        # newline count + 1 for non-empty content.
        assert result["line_count"] == 6
        assert batch.read_paths == ("sourcing_model/small.py",)


class TestRedactionPostSlice:
    def test_secret_values_inside_a_requested_range_are_masked(self, tmp_path):
        target = tmp_path / "sourcing_model" / "config_like.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# filler line {index}\n" for index in range(1, 101)]
        lines[49] = 'API_KEY = "sk-or-abcdefghijklmnop1234"\n'
        target.write_text("".join(lines), encoding="utf-8")
        context = _source_context(tmp_path, ("sourcing_model/config_like.py",))
        batch = resolve_source_inspection_requests(
            context,
            [_read("sourcing_model/config_like.py", start_line=40, max_lines=20)],
            already_read_paths=(),
            max_files=12,
            max_file_bytes=24_000,
            max_total_bytes=120_000,
            max_search_matches=30,
            source_access_v2=True,
        )
        (result,) = batch.model_context["results"]
        assert "sk-or-abcdefghijklmnop1234" not in result["content"]
        assert result["start_line"] == 40
        # Masking is 1:1 so the slice keeps its line structure.
        assert result["line_count"] == 20


class TestReadFileForModelUnit:
    def test_multibyte_content_is_not_split_mid_character(self, tmp_path):
        target = tmp_path / "unicode.py"
        target.write_text("# émoji 🎯 heavy line\n" * 40, encoding="utf-8")
        result = _read_source_file_for_model(
            tmp_path, "unicode.py", max_bytes=100, line_based=True
        )
        # Whole lines only: no U+FFFD replacement characters from split chars.
        assert "�" not in result["content"]
        assert result["content"].endswith("\n")

    def test_single_line_longer_than_budget_is_returned_whole(self, tmp_path):
        target = tmp_path / "long.py"
        target.write_text("x = " + "a" * 500 + "\nsecond\n", encoding="utf-8")
        result = _read_source_file_for_model(tmp_path, "long.py", max_bytes=100, line_based=True)
        assert result["line_count"] == 1
        assert result["content"].rstrip("\n").endswith("a")
        assert result["range_truncated"] is True


class TestPromptFlipTogether:
    def _messages(self, *, source_access_v2):
        return build_code_edit_source_inspection_messages(
            ticket={"ticket_id": "t1"},
            artifact_manifest={},
            component_registry={},
            benchmark_public_summary={},
            runtime_source_index={"editable_files": []},
            source_inspection_context={},
            budget_context={},
            max_requests=4,
            source_access_v2=source_access_v2,
        )

    def test_explicit_v2_true_advertises_ranged_reads(self, monkeypatch):
        monkeypatch.delenv("RESEARCH_LAB_SOURCE_ACCESS_V2", raising=False)
        user = self._messages(source_access_v2=True)[1]["content"]
        assert "start_line" in user
        assert "search match" in user

    def test_explicit_v2_false_wins_over_env_flag(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_LAB_SOURCE_ACCESS_V2", "true")
        user = self._messages(source_access_v2=False)[1]["content"]
        assert "start_line" not in user

    def test_none_falls_back_to_env_flag(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_LAB_SOURCE_ACCESS_V2", "true")
        user = self._messages(source_access_v2=None)[1]["content"]
        assert "start_line" in user

    def test_none_defaults_to_v2_when_env_is_unset(self, monkeypatch):
        monkeypatch.delenv("RESEARCH_LAB_SOURCE_ACCESS_V2", raising=False)
        user = self._messages(source_access_v2=None)[1]["content"]
        assert "start_line" in user

    def test_none_honors_explicit_false_env(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_LAB_SOURCE_ACCESS_V2", "false")
        user = self._messages(source_access_v2=None)[1]["content"]
        assert "start_line" not in user
