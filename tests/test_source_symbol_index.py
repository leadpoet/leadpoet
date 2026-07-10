"""Sanitized planner symbol-index and one-shot reference lookup tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway.research_lab.code_build import ParentImageSourceContext
from gateway.research_lab.source_symbol_index import (
    MAX_INDEX_FILES,
    MAX_INDEX_SYMBOLS,
    MAX_PLANNER_INDEX_BYTES,
    build_source_symbol_index,
    resolve_source_references,
    unresolved_references_from_context,
    valid_unresolved_reference,
)
from research_lab.code_editing import loop_direction_plan_from_mapping


def _build(root: Path, editable_files: tuple[str, ...]) -> dict:
    return build_source_symbol_index(
        source_root=root,
        editable_files=editable_files,
        source_tree_hash="sha256:tree",
        parent_image_digest_hash="sha256:image",
    )


def test_index_is_deterministic_and_records_ast_shape_without_values(tmp_path):
    target = tmp_path / "model" / "provider.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        """import json as codec
from .client import fetch as fetch_client

SECRET_VALUE = "do-not-project-this-value"

class Provider:
    \"\"\"Calls https://user:pass@example.test with token=abcdefghijklmnop.\"\"\"

    async def search(self, query, *, limit=SECRET_VALUE):
        \"\"\"Return bounded results. A second sentence is omitted.\"\"\"
        def normalize(item):
            return item
        return normalize(query)
""",
        encoding="utf-8",
    )

    first = _build(tmp_path, ("model/provider.py", "model/provider.py"))
    second = _build(tmp_path, ("model/provider.py",))

    assert first["index_hash"] == second["index_hash"]
    assert first["source_tree_hash"] == "sha256:tree"
    assert first["parent_image_digest_hash"] == "sha256:image"
    assert first["editable_file_count"] == 1
    symbols = first["files"][0]["symbols"]
    by_name = {item["qualified_name"]: item for item in symbols}
    assert by_name["Provider"]["kind"] == "class"
    assert by_name["Provider.search"]["kind"] == "async_method"
    assert by_name["Provider.search.normalize"]["kind"] == "function"
    assert by_name["Provider.search"]["parameters"] == ["self", "query", "limit"]
    assert by_name["Provider.search"]["summary"] == "Return bounded results."
    serialized = json.dumps(first, sort_keys=True)
    assert "do-not-project-this-value" not in serialized
    assert "user:pass" not in serialized
    assert "abcdefghijklmnop" not in serialized


def test_index_finds_symbol_beyond_first_24kb(tmp_path):
    target = tmp_path / "model" / "large.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        ("# filler that keeps the symbol outside the old preview window\n" * 700)
        + "def deep_candidate_router(company, icp):\n"
        + "    \"\"\"Route a candidate after deep source inspection.\"\"\"\n"
        + "    return company\n",
        encoding="utf-8",
    )
    assert target.stat().st_size > 24_000

    index = _build(tmp_path, ("model/large.py",))
    symbol = next(
        item
        for item in index["files"][0]["symbols"]
        if item["name"] == "deep_candidate_router"
    )
    assert symbol["start_line"] > 400

    resolved = resolve_source_references(
        index_doc=index,
        source_root=tmp_path,
        references=("deep_candidate_router",),
    )
    assert resolved["resolved_reference_count"] == 1
    assert resolved["results"][0]["matches"][0]["path"] == "model/large.py"
    assert resolved["results"][0]["matches"][0]["start_line"] == symbol["start_line"]


def test_malformed_python_is_hash_only_and_does_not_break_other_files(tmp_path):
    bad = tmp_path / "model" / "bad.py"
    good = tmp_path / "model" / "good.py"
    bad.parent.mkdir(parents=True)
    bad.write_text("def broken(:\n    secret = 'sk-or-never-project-this'\n", encoding="utf-8")
    good.write_text("def healthy():\n    return True\n", encoding="utf-8")

    index = _build(tmp_path, ("model/bad.py", "model/good.py"))
    bad_doc = next(item for item in index["files"] if item["path"] == "model/bad.py")
    assert bad_doc["parse_status"] == "parse_failed"
    assert bad_doc["parse_error_class"] == "SyntaxError"
    assert bad_doc["parse_error_hash"].startswith("sha256:")
    assert "sk-or-" not in json.dumps(index)
    assert index["parse_error_count"] == 1


def test_index_never_reads_an_editable_path_outside_extracted_root(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = 'must-not-be-indexed'\n", encoding="utf-8")

    index = build_source_symbol_index(
        source_root=source_root,
        editable_files=("../outside.py",),
        source_tree_hash="sha256:tree",
        parent_image_digest_hash="sha256:image",
    )

    assert index["files"][0]["parse_status"] == "unreadable"
    assert "must-not-be-indexed" not in json.dumps(index)


def test_index_caps_files_symbols_and_projection_bytes(tmp_path):
    editable = []
    for file_index in range(MAX_INDEX_FILES + 2):
        rel = f"model/file_{file_index:03d}.py"
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                f"def symbol_{file_index}_{symbol_index}(argument):\n    return argument"
                for symbol_index in range(6)
            )
            + "\n",
            encoding="utf-8",
        )
        editable.append(rel)

    index = _build(tmp_path, tuple(reversed(editable)))
    assert index["editable_file_count"] == MAX_INDEX_FILES + 2
    assert index["file_count"] == MAX_INDEX_FILES
    assert index["symbol_count"] <= MAX_INDEX_SYMBOLS
    assert index["truncated"] is True
    assert len(json.dumps(index, sort_keys=True, separators=(",", ":")).encode("utf-8")) <= MAX_PLANNER_INDEX_BYTES
    assert [item["path"] for item in index["files"]] == sorted(editable)[:MAX_INDEX_FILES]


def test_reference_resolution_order_and_safe_reference_filtering(tmp_path):
    target = tmp_path / "model" / "routing.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "class Router:\n"
        "    def discover_companies(self, query):\n"
        "        marker = 'provider_result_marker'\n"
        "        return query\n",
        encoding="utf-8",
    )
    index = _build(tmp_path, ("model/routing.py",))

    resolved = resolve_source_references(
        index_doc=index,
        source_root=tmp_path,
        references=(
            "model/routing.py",
            "Router.discover_companies",
            "discover_companies",
            "provider_result_marker",
        ),
    )
    assert [item["matches"][0]["kind"] for item in resolved["results"]] == [
        "exact_path",
        "qualified_symbol",
        "bare_symbol",
        "source_text",
    ]
    assert valid_unresolved_reference("../secret.py") == ""
    assert valid_unresolved_reference("https://example.test/path") == ""
    assert valid_unresolved_reference("free form paragraph") == ""
    assert valid_unresolved_reference("x" * 121) == ""
    assert unresolved_references_from_context(
        explicit=("Router.discover_companies", "../bad"),
        reason="missing provider_result_marker",
    ) == ("Router.discover_companies", "provider_result_marker")

    with pytest.raises(ValueError, match="exceeds 120 characters"):
        loop_direction_plan_from_mapping(
            {
                "no_new_safe_path": True,
                "reason": "unknown reference",
                "unresolved_references": ["x" * 121],
            }
        )


def test_repeated_inspection_inventory_omits_symbols_and_summaries(tmp_path):
    target = tmp_path / "model" / "routing.py"
    target.parent.mkdir(parents=True)
    target.write_text('def route(query):\n    """Private planner summary."""\n    return query\n', encoding="utf-8")
    index = _build(tmp_path, ("model/routing.py",))
    context = ParentImageSourceContext(
        source_root=tmp_path,
        source_mode="parent_image_extract",
        parent_image_digest_hash="sha256:image",
        source_tree_hash="sha256:tree",
        top_level_paths=("model/",),
        editable_files=("model/routing.py",),
        file_previews=(),
        planner_source_index=index,
    )

    assert context.planner_index()["files"][0]["symbols"][0]["name"] == "route"
    inventory_file = context.inspection_index()["file_inventory"][0]
    assert inventory_file["path"] == "model/routing.py"
    assert "symbols" not in inventory_file
    assert "summary" not in json.dumps(context.inspection_index())
