"""Symbol-slice starting context: extraction, budgets, and the phase-0 case.

The decisive scenario mirrors a production failure where the plan
referenced helpers early in a large file, the interactive excerpt missed
them, and the loop failed after its repairs. Seeding must deliver exactly those spans.
"""

from pathlib import Path

from gateway.research_lab.source_slice_context import (
    DEFAULT_SLICE_BUDGET_SHARE,
    build_slice_context,
    slice_results_for_inspection_context,
)
from gateway.research_lab.source_symbol_index import build_source_symbol_index


BIG_FILE = '''\
import re
import json

WINDOW_DAYS = 30
_PRESS_RE = re.compile(r"press|news")


def _event_terms(record):
    """Terms identifying one event."""
    return set(str(record.get("title", "")).lower().split())


@staticmethod
def _press_release_url_classifier(url, company_domain):
    return bool(_PRESS_RE.search(url or ""))


class Classifier:
    """Groups press classification helpers."""

    def __init__(self, domain):
        self.domain = domain

    def is_third_party(self, url):
        return self.domain not in (url or "")

'''


def _pad(text: str, total_lines: int) -> str:
    lines = text.splitlines()
    while len(lines) < total_lines:
        lines.append(f"# filler line {len(lines) + 1}")
    return "\n".join(lines) + "\n"


def _make_tree(tmp_path: Path) -> Path:
    root = tmp_path / "src"
    (root / "example_pkg").mkdir(parents=True)
    (root / "example_pkg" / "pipeline.py").write_text(
        _pad(BIG_FILE, 400), encoding="utf-8"
    )
    (root / "example_pkg" / "tiny.py").write_text(
        "VALUE = 1\n\n\ndef helper():\n    return VALUE\n", encoding="utf-8"
    )
    return root


def _index(root: Path) -> dict:
    return build_source_symbol_index(
        source_root=root,
        editable_files=["example_pkg/pipeline.py", "example_pkg/tiny.py"],
        source_tree_hash="sha256:" + "0" * 64,
        parent_image_digest_hash="sha256:" + "1" * 64,
    )


def _symbol_reference(index: dict, path: str, qualified_name: str) -> dict:
    for file_doc in index["files"]:
        if file_doc["path"] != path:
            continue
        for symbol in file_doc.get("symbols", []):
            if symbol["qualified_name"] == qualified_name:
                return {
                    "path": path,
                    "qualified_name": qualified_name,
                    "start_line": symbol["start_line"],
                    "end_line": symbol["end_line"],
                }
    raise AssertionError(f"{qualified_name} not indexed")


def test_phase0_case_referenced_helpers_are_seeded(tmp_path):
    root = _make_tree(tmp_path)
    index = _index(root)
    refs = [
        _symbol_reference(index, "example_pkg/pipeline.py", "_event_terms"),
        _symbol_reference(
            index, "example_pkg/pipeline.py", "_press_release_url_classifier"
        ),
    ]
    context = build_slice_context(
        source_root=root,
        index_doc=index,
        bound_references=refs,
        source_byte_budget=200_000,
    )
    joined = "\n".join(s.content for s in context.slices)
    assert "def _event_terms" in joined
    assert "def _press_release_url_classifier" in joined
    # Module constants the helpers rely on arrive via the top region.
    assert "WINDOW_DAYS" in joined
    assert "_PRESS_RE" in joined
    assert context.unresolved_references == []


def test_decorated_function_keeps_its_decorator(tmp_path):
    root = _make_tree(tmp_path)
    index = _index(root)
    ref = _symbol_reference(
        index, "example_pkg/pipeline.py", "_press_release_url_classifier"
    )
    context = build_slice_context(
        source_root=root,
        index_doc=index,
        bound_references=[ref],
        source_byte_budget=200_000,
    )
    symbol_slices = [s for s in context.slices if s.reason == "referenced_symbol"]
    assert symbol_slices and "@staticmethod" in symbol_slices[0].content


def test_method_reference_includes_class_header(tmp_path):
    root = _make_tree(tmp_path)
    index = _index(root)
    ref = _symbol_reference(
        index, "example_pkg/pipeline.py", "Classifier.is_third_party"
    )
    context = build_slice_context(
        source_root=root,
        index_doc=index,
        bound_references=[ref],
        source_byte_budget=200_000,
    )
    reasons = {s.reason for s in context.slices}
    assert "enclosing_class" in reasons
    class_slice = next(s for s in context.slices if s.reason == "enclosing_class")
    assert "class Classifier" in class_slice.content
    assert "__init__" in class_slice.content


def test_small_files_ship_whole(tmp_path):
    root = _make_tree(tmp_path)
    index = _index(root)
    ref = _symbol_reference(index, "example_pkg/tiny.py", "helper")
    context = build_slice_context(
        source_root=root,
        index_doc=index,
        bound_references=[ref],
        source_byte_budget=200_000,
    )
    assert [s.reason for s in context.slices] == ["whole_file"]
    assert "VALUE = 1" in context.slices[0].content


def test_budget_share_is_enforced(tmp_path):
    root = _make_tree(tmp_path)
    index = _index(root)
    refs = [
        _symbol_reference(index, "example_pkg/pipeline.py", "_event_terms"),
        _symbol_reference(
            index, "example_pkg/pipeline.py", "_press_release_url_classifier"
        ),
        _symbol_reference(
            index, "example_pkg/pipeline.py", "Classifier.is_third_party"
        ),
    ]
    tiny_budget = 300  # bytes; share of it smaller still
    context = build_slice_context(
        source_root=root,
        index_doc=index,
        bound_references=refs,
        source_byte_budget=tiny_budget,
    )
    assert context.budget_bytes == int(tiny_budget * DEFAULT_SLICE_BUDGET_SHARE)
    assert context.total_bytes <= context.budget_bytes
    assert context.skipped_over_budget > 0


def test_unresolvable_path_falls_through_not_fails(tmp_path):
    root = _make_tree(tmp_path)
    index = _index(root)
    context = build_slice_context(
        source_root=root,
        index_doc=index,
        bound_references=[{"path": "example_pkg/ghost.py", "qualified_name": "x"}],
        source_byte_budget=200_000,
    )
    assert context.slices == []
    assert context.unresolved_references == ["x"]


def test_traversal_outside_root_is_refused(tmp_path):
    root = _make_tree(tmp_path)
    index = _index(root)
    context = build_slice_context(
        source_root=root,
        index_doc=index,
        bound_references=[{"path": "../../etc/passwd", "qualified_name": "evil"}],
        source_byte_budget=200_000,
    )
    assert context.slices == []
    assert context.unresolved_references == ["evil"]


def test_read_ranges_and_result_shape(tmp_path):
    root = _make_tree(tmp_path)
    index = _index(root)
    ref = _symbol_reference(index, "example_pkg/pipeline.py", "_event_terms")
    context = build_slice_context(
        source_root=root,
        index_doc=index,
        bound_references=[ref],
        source_byte_budget=200_000,
    )
    ranges = context.read_ranges()
    assert "example_pkg/pipeline.py" in ranges
    results = slice_results_for_inspection_context(context)
    assert all(r["op"] == "seeded_symbol_slice" for r in results)
    assert all(r["content"] for r in results)


def test_determinism(tmp_path):
    root = _make_tree(tmp_path)
    index = _index(root)
    refs = [
        _symbol_reference(index, "example_pkg/pipeline.py", "_event_terms"),
        _symbol_reference(
            index, "example_pkg/pipeline.py", "Classifier.is_third_party"
        ),
    ]
    first = build_slice_context(
        source_root=root, index_doc=index, bound_references=refs, source_byte_budget=50_000
    )
    second = build_slice_context(
        source_root=root, index_doc=index, bound_references=refs, source_byte_budget=50_000
    )
    assert [
        (s.path, s.start_line, s.end_line, s.reason) for s in first.slices
    ] == [(s.path, s.start_line, s.end_line, s.reason) for s in second.slices]
    assert first.total_bytes == second.total_bytes
