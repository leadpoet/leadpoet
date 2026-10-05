"""Bounded, local comparison of Arena source archives.

Only a caller that has checked source disclosure may send a reference archive to
the review provider. Private comparison context never contains reference text.
"""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import re
import copy
from dataclasses import dataclass
from typing import Any

from lab_arena import code_review, source_bundle


MAX_STRUCTURAL_FILE_BYTES = 256 * 1024
MAX_PUBLIC_REFERENCE_BYTES = 512 * 1024
MAX_PUBLIC_REFERENCE_FILE_BYTES = 256 * 1024
_REFLECTION_NAMES = frozenset({
    "__file__", "__code__", "__dict__", "__doc__", "inspect", "linecache", "tokenize",
    "getsource", "open", "read_text", "read_bytes", "eval", "exec", "compile",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "importlib",
    "sys", "traceback", "dis", "settrace", "setprofile", "__traceback__",
})


@dataclass(frozen=True)
class SourceIdentity:
    archive_sha256: str
    normalized_sha256: str
    files: tuple[code_review.ReviewFile, ...]


@dataclass(frozen=True)
class SimilarityComparison:
    status: str  # exact, ambiguous, or distinct
    score: float
    change_kind: str
    candidate_paths: tuple[str, ...]
    change_kinds: tuple[str, ...] = ()


def _normalized_text(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n")


def _python_tree(content: str) -> ast.AST | None:
    if len(content.encode("utf-8")) > MAX_STRUCTURAL_FILE_BYTES:
        return None
    try:
        return ast.parse(_normalized_text(content))
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None


def _reflective(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.Name) and node.id in _REFLECTION_NAMES
        or isinstance(node, ast.Attribute) and node.attr in _REFLECTION_NAMES
        or isinstance(node, ast.Constant) and isinstance(node.value, str)
        and any(name in node.value for name in ("__file__", "getsource", "read_text"))
        for node in ast.walk(tree)
    )


def _archive_reflective(files: tuple[code_review.ReviewFile, ...]) -> bool:
    for item in files:
        if not item.path.endswith(".py"):
            continue
        tree = _python_tree(item.content)
        # Unknown Python is treated as reflective for cross-file comparison.
        if tree is None or _reflective(tree):
            return True
    return False


def _canonical_content(path: str, content: str) -> bytes:
    normalized = _normalized_text(content)
    if path.endswith(".py"):
        tree = _python_tree(normalized)
        if tree is not None and not _reflective(tree):
            try:
                return ("ast:" + ast.dump(tree, include_attributes=False)).encode("utf-8")
            except (MemoryError, RecursionError):
                pass
    return ("text:" + normalized).encode("utf-8")


def inspect_archive(payload: bytes) -> SourceIdentity:
    """Validate and fingerprint the complete source tree, ignoring tar root."""
    if not isinstance(payload, bytes):
        raise code_review.CodeReviewError("review_source_invalid")
    full_files = code_review._read_all_text_files(payload)
    facts = source_bundle.validate_source_archive(payload)
    root = str(facts["source_root"])
    prefix = root + "/" if root else ""
    files = tuple(sorted((
        code_review.ReviewFile(
            item.path.removeprefix(prefix), item.size_bytes, item.content
        ) for item in full_files
    ), key=lambda item: item.path))
    # Source inspection in one module can observe formatting in another.
    archive_reflective = _archive_reflective(files)
    digest = hashlib.sha256()
    for item in files:
        path = item.path.encode("utf-8")
        content = (
            ("text:" + _normalized_text(item.content)).encode("utf-8")
            if archive_reflective else _canonical_content(item.path, item.content)
        )
        digest.update(len(path).to_bytes(4, "big"))
        digest.update(path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return SourceIdentity(
        "sha256:" + hashlib.sha256(payload).hexdigest(),
        "sha256:" + digest.hexdigest(), files,
    )


def _rename_only(old: str, new: str, *, strip_inert: bool = False) -> bool:
    """Prove only function-local assignment renames with identical AST shape."""
    if max(len(old.encode("utf-8")), len(new.encode("utf-8"))) > MAX_STRUCTURAL_FILE_BYTES:
        return False
    old_tree = _python_tree(old)
    new_tree = _python_tree(new)
    if old_tree is None or new_tree is None or _reflective(old_tree) or _reflective(new_tree):
        return False
    try:
        if ast.dump(old_tree, include_attributes=False) == ast.dump(new_tree, include_attributes=False):
            return False
    except (MemoryError, RecursionError):
        return False

    def normalized(tree: ast.AST) -> str | None:
        clone = copy.deepcopy(tree)
        if strip_inert:
            clone = _StripInertStatements().visit(clone)
        for function in ast.walk(clone):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Do not infer lexical binding through nested scopes, imports,
            # closures, unpacking, or explicit dynamic scope operations.
            descendants = list(ast.walk(function))
            if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef, ast.Lambda, ast.Global,
                                     ast.Nonlocal, ast.Import, ast.ImportFrom,
                                     ast.NamedExpr, ast.For, ast.AsyncFor,
                                     ast.With, ast.AsyncWith, ast.ExceptHandler,
                                     ast.ListComp, ast.SetComp, ast.DictComp,
                                     ast.GeneratorExp))
                   for node in descendants[1:]):
                return None
            # Defaults, decorators and annotations run in the outer scope.
            # Rename only references inside the function body.
            body_nodes = [node for statement in function.body for node in ast.walk(statement)]
            bindings: dict[str, str] = {}
            arguments = function.args
            if arguments.vararg is not None or arguments.kwarg is not None:
                return None
            for index, argument in enumerate(
                arguments.posonlyargs + arguments.args + arguments.kwonlyargs
            ):
                bindings[argument.arg] = f"@arena:arg:{index}"
                argument.arg = bindings[argument.arg]
            for node in body_nodes:
                if isinstance(node, ast.Assign):
                    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                        return None
                    name = node.targets[0].id
                    if name not in bindings:
                        local_count = sum(value.startswith("@arena:local:") for value in bindings.values())
                        bindings[name] = f"@arena:local:{local_count}"
                elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                    return None
            if not bindings:
                return None
            for node in body_nodes:
                if isinstance(node, ast.Name) and node.id in bindings:
                    node.id = bindings[node.id]
        try:
            return ast.dump(clone, include_attributes=False)
        except (MemoryError, RecursionError):
            return None

    old_normalized = normalized(old_tree)
    return old_normalized is not None and old_normalized == normalized(new_tree)


def _prose_cosmetic_only(path: str, old: str, new: str) -> bool:
    """Mark plain prompt punctuation/spacing edits as uncertain, never exact."""
    if not path.startswith("prompts/") or not path.endswith((".txt", ".md")):
        return False
    if max(len(old.encode("utf-8")), len(new.encode("utf-8"))) > MAX_STRUCTURAL_FILE_BYTES:
        return False
    if any(character in old + new for character in "{}[]<>/\\:=`$"):
        return False
    words_old = re.findall(r"\w+", old, flags=re.UNICODE)
    words_new = re.findall(r"\w+", new, flags=re.UNICODE)
    return len(words_old) >= 4 and words_old == words_new and old != new


class _StripInertStatements(ast.NodeTransformer):
    def generic_visit(self, node: ast.AST) -> ast.AST:
        for field, value in ast.iter_fields(node):
            if isinstance(value, list) and value and all(isinstance(item, ast.stmt) for item in value):
                kept = [item for item in value if not isinstance(item, ast.Pass)
                        and not (isinstance(item, ast.Expr)
                                 and isinstance(item.value, ast.Constant))]
                setattr(node, field, [self.visit(item) for item in kept])
            elif isinstance(value, ast.AST):
                setattr(node, field, self.visit(value))
        return node


def _cosmetic_python_only(old: str, new: str) -> bool:
    """Spot pass/literal-expression edits, including docstrings, as uncertain."""
    old_tree, new_tree = _python_tree(old), _python_tree(new)
    if old_tree is None or new_tree is None or _reflective(old_tree) or _reflective(new_tree):
        return False


    try:
        old_dump = ast.dump(_StripInertStatements().visit(copy.deepcopy(old_tree)),
                            include_attributes=False)
        new_dump = ast.dump(_StripInertStatements().visit(copy.deepcopy(new_tree)),
                            include_attributes=False)
    except (MemoryError, RecursionError):
        return False
    return old_dump == new_dump and not _same_tree(old_tree, new_tree)


def _documentation_path(path: str) -> bool:
    return (
        path == "README" or path.startswith("README.")
        or (path.startswith("docs/") and path.endswith((".md", ".txt", ".rst")))
    )


def _added_inert_python(content: str) -> bool:
    tree = _python_tree(content)
    return tree is not None and not _reflective(tree) and all(
        isinstance(node, ast.Pass)
        or isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        for node in tree.body
    )


def _change_class(path: str, old: str, new: str, *, reflective: bool) -> str | None:
    if _normalized_text(old) == _normalized_text(new):
        return "line_endings"
    if path.endswith(".py"):
        old_tree, new_tree = _python_tree(old), _python_tree(new)
        if old_tree is not None and new_tree is not None and _same_tree(old_tree, new_tree):
            return "formatting_with_source_introspection" if reflective else "formatting_or_comments"
        if not reflective and _rename_only(old, new):
            return "identifier_only"
        if not reflective and _cosmetic_python_only(old, new):
            return "inert_python_statements"
        if not reflective and _rename_only(old, new, strip_inert=True):
            return "identifier_and_inert_statements"
    if _prose_cosmetic_only(path, old, new):
        return "prose_punctuation_or_spacing"
    if _documentation_path(path) and max(len(old.encode()), len(new.encode())) <= MAX_STRUCTURAL_FILE_BYTES:
        return "documentation_text"
    return None


def compare(candidate: SourceIdentity, reference: SourceIdentity) -> SimilarityComparison:
    """Return exact only for source equality with conservative runtime guards."""
    candidate_files = {item.path: item for item in candidate.files}
    reference_files = {item.path: item for item in reference.files}
    paths = tuple(sorted(set(candidate_files) | set(reference_files)))
    candidate_paths = tuple(path for path in paths if path in candidate_files)
    if any(path not in candidate_files for path in reference_files):
        return SimilarityComparison("distinct", 0.0, "file_set_changed", candidate_paths)
    changed = tuple(path for path in paths if path not in reference_files
                    or candidate_files[path].content != reference_files[path].content)
    if not changed:
        return SimilarityComparison("exact", 1.0, "same_files", ())
    if candidate.normalized_sha256 == reference.normalized_sha256:
        return SimilarityComparison("exact", 1.0, "formatting_or_comments", changed)
    reflective = _archive_reflective(candidate.files) or _archive_reflective(reference.files)
    kinds: set[str] = set()
    for path in changed:
        if path not in reference_files:
            content = candidate_files[path].content
            if path.endswith(".py") and _added_inert_python(content):
                kind = "added_inert_python_file"
            elif _documentation_path(path) and len(content.encode()) <= MAX_STRUCTURAL_FILE_BYTES:
                kind = "added_documentation_file"
            else:
                kind = None
        else:
            kind = _change_class(path, reference_files[path].content,
                                 candidate_files[path].content, reflective=reflective)
        if kind is None:
            break
        if kind == "identifier_and_inert_statements":
            kinds.update(("identifier_only", "inert_python_statements"))
        else:
            kinds.add(kind)
    else:
        ordered = tuple(sorted(kinds))
        kind = ordered[0] if len(ordered) == 1 else "mixed_cosmetic"
        return SimilarityComparison("ambiguous", 0.99, kind, changed, ordered)
    # A numeric score is useful for preselection, not an adjudication. Use only
    # bounded path equality here so no large diff or quadratic matcher runs.
    unchanged = len(paths) - len(changed)
    return SimilarityComparison(
        "distinct", round(unchanged / max(1, len(paths)), 6),
        "semantic_or_unclassified_edit", changed,
    )


def _same_tree(left: ast.AST, right: ast.AST) -> bool:
    try:
        return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)
    except (MemoryError, RecursionError):
        return False


def _public_near(candidate: SourceIdentity, reference: SourceIdentity) -> float:
    """Linear, bounded bag-of-tokens screen; never a duplicate verdict."""
    if sum(item.size_bytes for item in candidate.files) > MAX_PUBLIC_REFERENCE_BYTES:
        return 0.0
    if sum(item.size_bytes for item in reference.files) > MAX_PUBLIC_REFERENCE_BYTES:
        return 0.0
    def tokens(identity: SourceIdentity) -> Counter[str]:
        result: Counter[str] = Counter()
        for item in identity.files:
            result.update(re.findall(r"\w+|[^\w\s]", item.content, flags=re.UNICODE))
        return result
    left, right = tokens(candidate), tokens(reference)
    denominator = max(sum(left.values()), sum(right.values()), 1)
    return sum((left & right).values()) / denominator


def comparison_context(
    candidate: SourceIdentity,
    reference: SourceIdentity,
    *,
    reference_public: bool,
    comparison_id: str,
) -> dict[str, Any] | None:
    """Make one bounded context for the existing full-candidate review call."""
    if not isinstance(comparison_id, str) or not 1 <= len(comparison_id) <= 80 or not comparison_id.isascii():
        raise ValueError("comparison_id_invalid")
    result = compare(candidate, reference)
    if result.status == "distinct":
        if not reference_public:
            return None
        near_score = _public_near(candidate, reference)
        if near_score < 0.80:
            return None
        result = SimilarityComparison(
            "ambiguous", round(near_score, 6), "public_near_match",
            result.candidate_paths,
        )
    context: dict[str, Any] = {
        "comparison_id": comparison_id,
        "local_status": result.status,
        "score": result.score,
        "reference_public": reference_public,
        "change_kind": result.change_kind,
        "change_kinds": list(result.change_kinds or (result.change_kind,)),
        "candidate_paths": list(result.candidate_paths[:100]),
        "candidate_path_count": len(result.candidate_paths),
    }
    if reference_public:
        reference_files = [
            {"path": item.path, "content": item.content}
            for item in reference.files
        ]
        if any(item.size_bytes > MAX_PUBLIC_REFERENCE_FILE_BYTES for item in reference.files):
            return context if result.change_kind != "public_near_match" else None
        encoded = json.dumps(reference_files, ensure_ascii=False).encode("utf-8")
        if len(encoded) <= MAX_PUBLIC_REFERENCE_BYTES:
            context["public_reference_files"] = reference_files
        elif result.change_kind == "public_near_match":
            return None
    return context


__all__ = [
    "SourceIdentity", "SimilarityComparison", "inspect_archive", "compare",
    "comparison_context", "MAX_PUBLIC_REFERENCE_BYTES",
]
