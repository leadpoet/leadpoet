"""Sanitized AST index and bounded reference lookup for code-edit planning."""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from research_lab.canonical import sha256_json


MAX_INDEX_FILES = 300
MAX_INDEX_SYMBOLS = 1_000
MAX_SYMBOLS_PER_FILE = 80
MAX_IMPORTS_PER_FILE = 40
MAX_SUMMARY_CHARS = 240
MAX_PLANNER_INDEX_BYTES = 128 * 1024
MAX_REFERENCE_COUNT = 8
MAX_REFERENCE_CHARS = 120
MAX_MATCHES_PER_REFERENCE = 5
MAX_REFERENCE_SEARCH_FILE_BYTES = 1 * 1024 * 1024
MAX_REFERENCE_SEARCH_TOTAL_BYTES = 8 * 1024 * 1024
_INDEX_HASH_RESERVE_BYTES = 128

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_REFERENCE_RE = re.compile(r"^[A-Za-z0-9_./:\-]+$")
_SECRET_VALUE_PATTERNS = (
    re.compile(r"sk-or-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sb_secret_[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b[a-z][a-z0-9+.\-]{1,20}://[^\s'\"@/]+:[^\s'\"@]+@[^\s'\"]+"),
    re.compile(
        r"(?i)(\b(?:api[_-]?key|secret|token|password|passwd|private_key)\b\s*[:=]\s*[\"']?)"
        r"([A-Za-z0-9_\-]{16,})"
    ),
)
_URL_RE = re.compile(r"(?i)\b(?:https?|s3)://\S+")


@dataclass(frozen=True)
class ExactSourceReferenceBinding:
    normalized_references: tuple[str, ...]
    missing_references: tuple[str, ...]
    invalid_references: tuple[str, ...]
    ambiguous_references: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not (
            self.missing_references
            or self.invalid_references
            or self.ambiguous_references
        )


def valid_unresolved_reference(value: Any) -> str:
    """Return a safe path/identifier-like reference, or an empty string."""

    text = str(value or "").strip()
    if len(text) > MAX_REFERENCE_CHARS:
        return ""
    if not text or _CONTROL_RE.search(text) or ".." in text or "://" in text:
        return ""
    if not _REFERENCE_RE.fullmatch(text):
        return ""
    lowered = text.lower()
    if any(marker in lowered for marker in ("sk-or-", "sb_secret", "password", "private_key")):
        return ""
    return text


def build_source_symbol_index(
    *,
    source_root: Path,
    editable_files: Sequence[str],
    source_tree_hash: str,
    parent_image_digest_hash: str,
) -> dict[str, Any]:
    """Build a deterministic, value-free index from one extracted parent image."""

    unique_editable_files = tuple(sorted(dict.fromkeys(str(item) for item in editable_files)))
    files: list[dict[str, Any]] = []
    total_symbols_seen = 0
    total_imports_seen = 0
    parse_error_count = 0
    remaining_symbols = MAX_INDEX_SYMBOLS
    for rel in unique_editable_files[:MAX_INDEX_FILES]:
        file_doc: dict[str, Any] = {"path": rel, "size_bytes": 0, "line_count": 0}
        try:
            path = _safe_source_path(source_root, rel)
            raw_bytes = path.read_bytes()
        except (OSError, ValueError):
            file_doc["parse_status"] = "unreadable"
            files.append(file_doc)
            continue
        text = raw_bytes.decode("utf-8", errors="replace")
        file_doc["size_bytes"] = len(raw_bytes)
        file_doc["line_count"] = len(text.splitlines())
        if not rel.endswith(".py"):
            file_doc["parse_status"] = "not_python"
            files.append(file_doc)
            continue
        try:
            tree = ast.parse(text, filename=rel)
        except (SyntaxError, ValueError, TypeError) as exc:
            parse_error_count += 1
            file_doc.update(
                {
                    "parse_status": "parse_failed",
                    "parse_error_class": type(exc).__name__,
                    "parse_error_hash": sha256_json({"path": rel, "error_class": type(exc).__name__}),
                }
            )
            files.append(file_doc)
            continue

        imports = _imports_from_tree(tree)
        symbols = _symbols_from_tree(tree)
        total_imports_seen += len(imports)
        total_symbols_seen += len(symbols)
        included_symbols = symbols[: min(MAX_SYMBOLS_PER_FILE, remaining_symbols)]
        remaining_symbols -= len(included_symbols)
        file_doc.update(
            {
                "parse_status": "parsed",
                "imports": imports[:MAX_IMPORTS_PER_FILE],
                "symbols": included_symbols,
                "import_count": len(imports),
                "symbol_count": len(symbols),
                "imports_truncated": len(imports) > MAX_IMPORTS_PER_FILE,
                "symbols_truncated": len(included_symbols) < len(symbols),
            }
        )
        files.append(file_doc)

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "source_tree_hash": str(source_tree_hash),
        "parent_image_digest_hash": str(parent_image_digest_hash),
        "editable_file_count": len(unique_editable_files),
        "files": files,
        "file_count": len(files),
        "symbol_count": sum(len(item.get("symbols") or []) for item in files),
        "symbol_count_seen": total_symbols_seen,
        "import_count": sum(len(item.get("imports") or []) for item in files),
        "import_count_seen": total_imports_seen,
        "parse_error_count": parse_error_count,
        "truncated": (
            len(unique_editable_files) > MAX_INDEX_FILES
            or any(bool(item.get("symbols_truncated") or item.get("imports_truncated")) for item in files)
        ),
    }
    payload = _fit_planner_projection(payload)
    payload["index_hash"] = sha256_json(payload)
    encoded_size = len(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    )
    if encoded_size > MAX_PLANNER_INDEX_BYTES:
        raise ValueError("planner symbol index exceeds the hard projection limit")
    return payload


def compact_file_inventory(index_doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return file metadata only for repeated source-inspection prompts."""

    return [
        {
            key: item.get(key)
            for key in ("path", "size_bytes", "line_count", "parse_status", "symbol_count", "import_count")
            if item.get(key) not in (None, "")
        }
        for item in index_doc.get("files", [])
        if isinstance(item, Mapping)
    ]


def bind_source_references_exact(
    *,
    index_doc: Mapping[str, Any],
    source_root: Path,
    editable_files: Sequence[str],
    references: Sequence[Any],
) -> ExactSourceReferenceBinding:
    """Bind source references exactly; fuzzy matches never satisfy this gate."""

    editable_paths = tuple(sorted(dict.fromkeys(str(item) for item in editable_files if item)))
    editable_set = set(editable_paths)
    file_docs = [item for item in index_doc.get("files", []) if isinstance(item, Mapping)]
    indexed_by_path = {str(item.get("path") or ""): item for item in file_docs}
    index_complete = not bool(index_doc.get("truncated")) and all(
        path in indexed_by_path for path in editable_paths
    ) and all(
        not path.endswith(".py") or str(indexed_by_path[path].get("parse_status") or "") == "parsed"
        for path in editable_paths
    )

    normalized: list[str] = []
    missing: list[str] = []
    invalid: list[str] = []
    ambiguous: list[str] = []
    seen_inputs: set[str] = set()

    for raw_reference in references:
        reference = str(raw_reference or "").strip()
        if reference in seen_inputs:
            continue
        seen_inputs.add(reference)
        safe_reference = valid_unresolved_reference(reference)
        if not safe_reference:
            _append_unique(invalid, reference or "<empty>")
            _append_unique(normalized, reference)
            continue
        if safe_reference in editable_set:
            _append_unique(normalized, safe_reference)
            continue

        composite_path, composite_symbol = _split_composite_reference(safe_reference)
        if composite_path:
            if composite_path not in editable_set:
                _append_unique(missing, safe_reference)
                _append_unique(normalized, safe_reference)
                continue
            symbols, complete = _exact_symbols_for_file(source_root, composite_path)
            if not complete:
                _append_unique(invalid, safe_reference)
                _append_unique(normalized, safe_reference)
                continue
            matches = _matching_symbols(symbols, composite_symbol)
            if len(matches) == 1:
                canonical = f"{composite_path}::{matches[0]['qualified_name']}"
                _append_unique(normalized, canonical)
            elif len(matches) > 1:
                _append_unique(ambiguous, safe_reference)
                _append_unique(normalized, safe_reference)
            else:
                _append_unique(missing, safe_reference)
                _append_unique(normalized, safe_reference)
            continue

        if _looks_like_file_reference(safe_reference):
            _append_unique(missing, safe_reference)
            _append_unique(normalized, safe_reference)
            continue

        if index_complete:
            symbol_rows = _indexed_symbol_rows(file_docs)
            scan_complete = True
        else:
            symbol_rows, scan_complete = _scan_editable_symbols(
                source_root=source_root,
                editable_files=editable_paths,
            )
        matches = [
            row
            for row in symbol_rows
            if str(row.get("name") or "") == safe_reference
            or str(row.get("qualified_name") or "") == safe_reference
        ]
        if len(matches) == 1 and scan_complete:
            canonical = f"{matches[0]['path']}::{matches[0]['qualified_name']}"
            _append_unique(normalized, canonical)
        elif len(matches) > 1 or not scan_complete:
            _append_unique(ambiguous, safe_reference)
            _append_unique(normalized, safe_reference)
        else:
            _append_unique(missing, safe_reference)
            _append_unique(normalized, safe_reference)

    return ExactSourceReferenceBinding(
        normalized_references=tuple(normalized),
        missing_references=tuple(missing),
        invalid_references=tuple(invalid),
        ambiguous_references=tuple(ambiguous),
    )


def unresolved_references_from_context(
    *,
    explicit: Sequence[Any] = (),
    must_inspect: Sequence[Any] = (),
    reason: str = "",
) -> tuple[str, ...]:
    """Collect bounded references without treating ordinary prose as code."""

    candidates: list[Any] = [*explicit, *must_inspect]
    candidates.extend(
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_./:\-]{2,119}", str(reason or ""))
        if "." in token or "/" in token or "_" in token
    )
    resolved: list[str] = []
    for candidate in candidates:
        value = valid_unresolved_reference(candidate)
        if value and value not in resolved:
            resolved.append(value)
        if len(resolved) >= MAX_REFERENCE_COUNT:
            break
    return tuple(resolved)


def resolve_source_references(
    *,
    index_doc: Mapping[str, Any],
    source_root: Path,
    references: Sequence[Any],
) -> dict[str, Any]:
    """Resolve references against paths/symbols, then a bounded text search."""

    safe_references = unresolved_references_from_context(explicit=references)
    file_docs = [item for item in index_doc.get("files", []) if isinstance(item, Mapping)]
    results: list[dict[str, Any]] = []
    for reference in safe_references:
        candidates: list[tuple[int, float, dict[str, Any]]] = []
        composite_path, composite_symbol = _split_composite_reference(reference)
        path_reference = composite_path or reference
        symbol_reference = composite_symbol or reference
        ref_lower = symbol_reference.lower()
        ref_tail = ref_lower.rsplit(".", 1)[-1].rsplit("/", 1)[-1]
        for file_doc in file_docs:
            path = str(file_doc.get("path") or "")
            path_lower = path.lower()
            path_ref_lower = path_reference.lower()
            if path_lower == path_ref_lower:
                candidates.append((0, 1.0, {"kind": "exact_path", "path": path}))
            elif path_lower.endswith(path_ref_lower) or path_lower.endswith("/" + path_ref_lower):
                candidates.append((1, 1.0, {"kind": "path_suffix", "path": path}))
            if composite_path and path != composite_path:
                continue
            for symbol in file_doc.get("symbols", []):
                if not isinstance(symbol, Mapping):
                    continue
                qualified = str(symbol.get("qualified_name") or "")
                name = str(symbol.get("name") or "")
                qualified_lower = qualified.lower()
                name_lower = name.lower()
                match: tuple[int, float, str] | None = None
                if qualified_lower == ref_lower:
                    match = (2, 1.0, "qualified_symbol")
                elif name_lower == ref_lower or name_lower == ref_tail:
                    match = (3, 1.0, "bare_symbol")
                else:
                    similarity = max(
                        SequenceMatcher(None, ref_lower, qualified_lower).ratio(),
                        SequenceMatcher(None, ref_tail, name_lower).ratio(),
                    )
                    if similarity >= 0.78:
                        match = (4, similarity, "similar_symbol")
                if match is not None:
                    candidates.append(
                        (
                            match[0],
                            match[1],
                            {
                                "kind": match[2],
                                "path": path,
                                "symbol": qualified,
                                "symbol_kind": str(symbol.get("kind") or ""),
                                "start_line": int(symbol.get("start_line") or 0),
                                "end_line": int(symbol.get("end_line") or 0),
                                "summary": str(symbol.get("summary") or "")[:MAX_SUMMARY_CHARS],
                            },
                        )
                    )
        if not candidates:
            candidates.extend(_bounded_text_matches(source_root, file_docs, reference))
        candidates.sort(key=lambda item: (item[0], -item[1], str(item[2].get("path") or ""), int(item[2].get("start_line") or 0)))
        matches: list[dict[str, Any]] = []
        seen: set[str] = set()
        for _rank, _similarity, match_doc in candidates:
            key = json.dumps(match_doc, sort_keys=True, separators=(",", ":"))
            if key in seen:
                continue
            seen.add(key)
            matches.append(match_doc)
            if len(matches) >= MAX_MATCHES_PER_REFERENCE:
                break
        results.append(
            {
                "reference": reference,
                "reference_hash": sha256_json({"reference": reference}),
                "match_count": len(matches),
                "matches": matches,
            }
        )
    payload = {
        "schema_version": "1.0",
        "symbol_index_hash": str(index_doc.get("index_hash") or ""),
        "reference_count": len(safe_references),
        "resolved_reference_count": sum(1 for item in results if item["match_count"] > 0),
        "results": results,
    }
    payload["resolution_hash"] = sha256_json(payload)
    return payload


def _split_composite_reference(reference: str) -> tuple[str, str]:
    if "::" in reference:
        path, symbol = reference.split("::", 1)
        return path.strip(), symbol.strip()
    match = re.fullmatch(r"(.+\.py):([^:]+)", reference)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", ""


def _looks_like_file_reference(reference: str) -> bool:
    return "/" in reference or bool(re.search(r"\.[A-Za-z0-9]{1,8}$", reference))


def _exact_symbols_for_file(source_root: Path, rel: str) -> tuple[list[dict[str, Any]], bool]:
    try:
        path = _safe_source_path(source_root, rel)
        raw = path.read_bytes()
        if len(raw) > MAX_REFERENCE_SEARCH_FILE_BYTES:
            return [], False
        tree = ast.parse(raw.decode("utf-8", errors="replace"), filename=rel)
    except (OSError, ValueError, SyntaxError, TypeError):
        return [], False
    return _symbols_from_tree(tree), True


def _matching_symbols(symbols: Sequence[Mapping[str, Any]], reference: str) -> list[dict[str, Any]]:
    return [
        dict(symbol)
        for symbol in symbols
        if str(symbol.get("name") or "") == reference
        or str(symbol.get("qualified_name") or "") == reference
    ]


def _indexed_symbol_rows(file_docs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_doc in file_docs:
        path = str(file_doc.get("path") or "")
        for symbol in file_doc.get("symbols", []):
            if isinstance(symbol, Mapping):
                rows.append({"path": path, **dict(symbol)})
    return rows


def _scan_editable_symbols(
    *,
    source_root: Path,
    editable_files: Sequence[str],
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    complete = True
    for rel in editable_files:
        if not rel.endswith(".py"):
            continue
        try:
            path = _safe_source_path(source_root, rel)
            raw = path.read_bytes()
        except (OSError, ValueError):
            complete = False
            continue
        if len(raw) > MAX_REFERENCE_SEARCH_FILE_BYTES:
            complete = False
            continue
        total_bytes += len(raw)
        if total_bytes > MAX_REFERENCE_SEARCH_TOTAL_BYTES:
            complete = False
            break
        try:
            tree = ast.parse(raw.decode("utf-8", errors="replace"), filename=rel)
        except (SyntaxError, ValueError, TypeError):
            complete = False
            continue
        rows.extend({"path": rel, **symbol} for symbol in _symbols_from_tree(tree))
    return rows, complete


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.scope_kinds: list[str] = []
        self.symbols: list[dict[str, Any]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self._append(node, kind="class", parameters=())
        self.stack.append(node.name)
        self.scope_kinds.append("class")
        self.generic_visit(node)
        self.scope_kinds.pop()
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._visit_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._visit_function(node, is_async=True)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, *, is_async: bool) -> None:
        is_method = bool(self.scope_kinds and self.scope_kinds[-1] == "class")
        kind = (
            "async_method"
            if is_async and is_method
            else "method"
            if is_method
            else "async_function"
            if is_async
            else "function"
        )
        self._append(node, kind=kind, parameters=_parameter_names(node.args))
        self.stack.append(node.name)
        self.scope_kinds.append("function")
        self.generic_visit(node)
        self.scope_kinds.pop()
        self.stack.pop()

    def _append(self, node: ast.AST, *, kind: str, parameters: Sequence[str]) -> None:
        name = str(getattr(node, "name", ""))
        qualified = ".".join([*self.stack, name]) if self.stack else name
        summary = _docstring_summary(node)
        record = {
            "name": name,
            "qualified_name": qualified,
            "kind": kind,
            "parameters": list(parameters),
            "start_line": int(getattr(node, "lineno", 0) or 0),
            "end_line": int(getattr(node, "end_lineno", 0) or getattr(node, "lineno", 0) or 0),
        }
        if summary:
            record["summary"] = summary
        self.symbols.append(record)


def _symbols_from_tree(tree: ast.AST) -> list[dict[str, Any]]:
    visitor = _SymbolVisitor()
    visitor.visit(tree)
    return visitor.symbols


def _imports_from_tree(tree: ast.AST) -> list[dict[str, Any]]:
    imports: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    {
                        "module": str(alias.name)[:200],
                        "bound_name": str(alias.asname or alias.name.split(".", 1)[0])[:120],
                        "line": int(getattr(node, "lineno", 0) or 0),
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            module = ("." * int(node.level or 0)) + str(node.module or "")
            for alias in node.names:
                imports.append(
                    {
                        "module": module[:200],
                        "name": str(alias.name)[:120],
                        "bound_name": str(alias.asname or alias.name)[:120],
                        "line": int(getattr(node, "lineno", 0) or 0),
                    }
                )
    imports.sort(key=lambda item: (int(item.get("line") or 0), str(item.get("module") or ""), str(item.get("name") or "")))
    return imports


def _parameter_names(arguments: ast.arguments) -> tuple[str, ...]:
    names = [arg.arg for arg in (*arguments.posonlyargs, *arguments.args)]
    if arguments.vararg:
        names.append("*" + arguments.vararg.arg)
    names.extend(arg.arg for arg in arguments.kwonlyargs)
    if arguments.kwarg:
        names.append("**" + arguments.kwarg.arg)
    return tuple(str(item)[:120] for item in names)


def _docstring_summary(node: ast.AST) -> str:
    try:
        value = ast.get_docstring(node, clean=True) or ""
    except TypeError:
        value = ""
    first = next((line.strip() for line in value.splitlines() if line.strip()), "")
    first = re.split(r"(?<=[.!?])\s+", first, maxsplit=1)[0]
    return _sanitize_summary(first)[:MAX_SUMMARY_CHARS]


def _sanitize_summary(value: str) -> str:
    text = _CONTROL_RE.sub(" ", str(value or ""))
    text = _URL_RE.sub("[redacted-url]", text)
    for pattern in _SECRET_VALUE_PATTERNS:
        text = pattern.sub("[redacted]", text)
    return re.sub(r"\s+", " ", text).strip()


def _fit_planner_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    fitted = deepcopy(dict(payload))

    def size() -> int:
        return len(json.dumps(fitted, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))

    byte_limit = MAX_PLANNER_INDEX_BYTES - _INDEX_HASH_RESERVE_BYTES
    if size() <= byte_limit:
        return fitted
    fitted["truncated"] = True
    for optional_key in ("summary", "parameters"):
        for file_doc in reversed(fitted.get("files", [])):
            for symbol in reversed(file_doc.get("symbols", [])):
                symbol.pop(optional_key, None)
                if size() <= byte_limit:
                    return fitted
    for file_doc in reversed(fitted.get("files", [])):
        imports = file_doc.get("imports")
        while isinstance(imports, list) and imports:
            imports.pop()
            file_doc["imports_truncated"] = True
            if size() <= byte_limit:
                return fitted
    for file_doc in reversed(fitted.get("files", [])):
        symbols = file_doc.get("symbols")
        while isinstance(symbols, list) and len(symbols) > 1:
            symbols.pop()
            file_doc["symbols_truncated"] = True
            if size() <= byte_limit:
                return fitted
    return fitted


def _bounded_text_matches(
    source_root: Path,
    file_docs: Sequence[Mapping[str, Any]],
    reference: str,
) -> list[tuple[int, float, dict[str, Any]]]:
    matches: list[tuple[int, float, dict[str, Any]]] = []
    needle = reference.lower()
    remaining_bytes = MAX_REFERENCE_SEARCH_TOTAL_BYTES
    for file_doc in file_docs:
        if remaining_bytes <= 0:
            break
        path = str(file_doc.get("path") or "")
        try:
            source_path = _safe_source_path(source_root, path)
            with source_path.open("rb") as handle:
                raw = handle.read(min(MAX_REFERENCE_SEARCH_FILE_BYTES, remaining_bytes))
        except (OSError, ValueError):
            continue
        remaining_bytes -= len(raw)
        lines = raw.decode("utf-8", errors="replace").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if needle in line.lower():
                matches.append(
                    (
                        5,
                        1.0,
                        {"kind": "source_text", "path": path, "start_line": line_number, "end_line": line_number},
                    )
                )
                if len(matches) >= MAX_MATCHES_PER_REFERENCE:
                    return matches
    return matches


def _safe_source_path(source_root: Path, relative_path: str) -> Path:
    root = source_root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("source path escapes extracted root") from exc
    return candidate
