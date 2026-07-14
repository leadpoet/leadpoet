"""Deterministic symbol-slice starting context for code-edit runs.

The binding plan names the symbols it intends to change and the source
symbol index records exactly where each one lives. This module cuts those
spans (plus the code a slice needs to be read correctly: decorators, the
enclosing class header, the module top region) into synthetic "already
read" entries that seed the code-edit source-inspection context. The
interactive search / ranged-read system is untouched and remains the
fallback: seeding only changes what the model has already seen when round
one starts.

Every slice derives purely from ``(source tree, index, bound references)``
so the seeded context is deterministic and replay-safe. Slices are capped
to a fixed share of the source byte budget so they can never starve the
interactive reads that succeed today.

Motivating failure class (verified against production loop history): a
code-edit run reads the right file, but the returned excerpt misses the
plan's referenced helpers, and the run fails after exhausting its repair
attempts even though the symbols were indexed at exact spans all along.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# Slices may consume at most this share of the code-edit source byte
# budget; the remainder stays reserved for the model's own interactive
# reads and searches.
DEFAULT_SLICE_BUDGET_SHARE = 0.4

# Files at or below this line count ship whole: slicing tiny files adds
# provenance noise without saving meaningful budget.
WHOLE_FILE_LINE_THRESHOLD = 120

_MAX_SLICES = 24


@dataclass
class SourceSlice:
    path: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    reason: str  # "referenced_symbol" | "enclosing_class" | "module_top" | "whole_file"
    qualified_name: str
    content: str

    @property
    def byte_size(self) -> int:
        return len(self.content.encode("utf-8"))


@dataclass
class SliceContext:
    slices: list[SourceSlice] = field(default_factory=list)
    total_bytes: int = 0
    budget_bytes: int = 0
    skipped_over_budget: int = 0
    unresolved_references: list[str] = field(default_factory=list)

    def read_ranges(self) -> dict[str, tuple[tuple[int, int], ...]]:
        """Spans in the shape the inspection loop tracks as already-read."""
        by_path: dict[str, list[tuple[int, int]]] = {}
        for item in self.slices:
            by_path.setdefault(item.path, []).append((item.start_line, item.end_line))
        return {path: tuple(sorted(spans)) for path, spans in by_path.items()}


def _decorator_adjusted_start(tree: ast.AST, def_line: int) -> int:
    """The index records the ``def`` line; decorators sit above it."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if int(getattr(node, "lineno", -1)) != def_line:
                continue
            decorator_lines = [
                int(getattr(dec, "lineno", def_line)) for dec in node.decorator_list
            ]
            if decorator_lines:
                return min(min(decorator_lines), def_line)
            return def_line
    return def_line


def _module_top_end(tree: ast.AST, line_count: int) -> int:
    """Last line before the first def/class: imports plus module constants."""
    first_def_line = line_count + 1
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first_def_line = int(getattr(node, "lineno", first_def_line))
            break
    return max(0, min(first_def_line - 1, line_count))


def _enclosing_class_span(
    file_symbols: Sequence[Mapping[str, Any]], qualified_name: str
) -> tuple[int, int] | None:
    if "." not in qualified_name:
        return None
    class_qualified = qualified_name.rsplit(".", 1)[0]
    for symbol in file_symbols:
        if (
            str(symbol.get("qualified_name") or "") == class_qualified
            and str(symbol.get("kind") or "") == "class"
        ):
            return int(symbol.get("start_line") or 0), int(symbol.get("end_line") or 0)
    return None


def _extract_lines(lines: Sequence[str], start: int, end: int) -> str:
    start = max(1, start)
    end = min(len(lines), max(start, end))
    return "\n".join(lines[start - 1 : end])


def _spans_overlap(existing: Sequence[tuple[int, int]], start: int, end: int) -> bool:
    return any(not (end < s or start > e) for s, e in existing)


def build_slice_context(
    *,
    source_root: Path,
    index_doc: Mapping[str, Any],
    bound_references: Sequence[Mapping[str, Any]],
    source_byte_budget: int,
    slice_budget_share: float = DEFAULT_SLICE_BUDGET_SHARE,
) -> SliceContext:
    """Cut deterministic starting slices for the plan's bound references.

    ``bound_references`` rows need ``path`` and, for symbol references,
    ``qualified_name``/``start_line``/``end_line`` (the shape
    ``bind_source_references_exact`` resolves to). Rows that cannot be
    sliced fall through silently — the interactive read system remains the
    fallback for them, preserving today's behavior as the floor.
    """

    budget = max(0, int(source_byte_budget * max(0.0, min(1.0, slice_budget_share))))
    context = SliceContext(budget_bytes=budget)
    files_by_path: dict[str, Mapping[str, Any]] = {
        str(item.get("path") or ""): item
        for item in index_doc.get("files", [])
        if isinstance(item, Mapping)
    }
    seen_spans: dict[str, list[tuple[int, int]]] = {}
    parsed_cache: dict[str, tuple[list[str], ast.AST] | None] = {}

    def _load(path: str) -> tuple[list[str], ast.AST] | None:
        if path in parsed_cache:
            return parsed_cache[path]
        try:
            resolved = (source_root / path).resolve()
            resolved.relative_to(source_root.resolve())
            text = resolved.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(text, filename=path)
            parsed_cache[path] = (text.splitlines(), tree)
        except (OSError, ValueError, SyntaxError):
            parsed_cache[path] = None
        return parsed_cache[path]

    def _admit(candidate: SourceSlice) -> bool:
        if len(context.slices) >= _MAX_SLICES:
            context.skipped_over_budget += 1
            return False
        spans = seen_spans.setdefault(candidate.path, [])
        if _spans_overlap(spans, candidate.start_line, candidate.end_line):
            return True  # already covered; not a failure
        if context.total_bytes + candidate.byte_size > budget:
            context.skipped_over_budget += 1
            return False
        spans.append((candidate.start_line, candidate.end_line))
        context.slices.append(candidate)
        context.total_bytes += candidate.byte_size
        return True

    for reference in bound_references:
        if not isinstance(reference, Mapping):
            continue
        path = str(reference.get("path") or "")
        file_doc = files_by_path.get(path)
        loaded = _load(path) if path else None
        if not path or loaded is None:
            context.unresolved_references.append(
                str(reference.get("qualified_name") or reference.get("reference") or path)
            )
            continue
        lines, tree = loaded

        # Small files ship whole — cheaper and less noisy than slicing.
        if len(lines) <= WHOLE_FILE_LINE_THRESHOLD:
            _admit(
                SourceSlice(
                    path=path,
                    start_line=1,
                    end_line=len(lines),
                    reason="whole_file",
                    qualified_name=path,
                    content=_extract_lines(lines, 1, len(lines)),
                )
            )
            continue

        # Module top region: imports and module-level constants the sliced
        # code will reference.
        top_end = _module_top_end(tree, len(lines))
        if top_end > 0:
            _admit(
                SourceSlice(
                    path=path,
                    start_line=1,
                    end_line=top_end,
                    reason="module_top",
                    qualified_name=f"{path}:module_top",
                    content=_extract_lines(lines, 1, top_end),
                )
            )

        start_line = int(reference.get("start_line") or 0)
        end_line = int(reference.get("end_line") or 0)
        qualified = str(reference.get("qualified_name") or "")
        if start_line <= 0 or end_line < start_line:
            # A file-only reference: top region already added; the
            # interactive reads cover the rest.
            continue

        adjusted_start = _decorator_adjusted_start(tree, start_line)
        _admit(
            SourceSlice(
                path=path,
                start_line=adjusted_start,
                end_line=end_line,
                reason="referenced_symbol",
                qualified_name=qualified,
                content=_extract_lines(lines, adjusted_start, end_line),
            )
        )

        # Methods additionally get their class header/__init__ so the model
        # sees construction state, not a floating function.
        symbols = (
            file_doc.get("symbols", []) if isinstance(file_doc, Mapping) else []
        )
        class_span = _enclosing_class_span(symbols, qualified)
        if class_span:
            class_start, class_end = class_span
            init_end = class_start
            for symbol in symbols:
                if str(symbol.get("qualified_name") or "").endswith(".__init__") and str(
                    symbol.get("qualified_name") or ""
                ).startswith(qualified.rsplit(".", 1)[0]):
                    init_end = int(symbol.get("end_line") or class_start)
                    break
            header_end = min(init_end, class_end, end_line if init_end == class_start else init_end)
            if header_end > class_start:
                _admit(
                    SourceSlice(
                        path=path,
                        start_line=_decorator_adjusted_start(tree, class_start),
                        end_line=header_end,
                        reason="enclosing_class",
                        qualified_name=qualified.rsplit(".", 1)[0],
                        content=_extract_lines(lines, class_start, header_end),
                    )
                )

    return context


def slice_results_for_inspection_context(context: SliceContext) -> list[dict[str, Any]]:
    """Slices in the read-result shape the inspection context records."""
    results: list[dict[str, Any]] = []
    for item in context.slices:
        results.append(
            {
                "op": "seeded_symbol_slice",
                "path": item.path,
                "start_line": item.start_line,
                "end_line": item.end_line,
                "reason": item.reason,
                "qualified_name": item.qualified_name,
                "content": item.content,
            }
        )
    return results
