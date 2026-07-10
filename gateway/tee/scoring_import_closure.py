"""Build and verify the gateway enclave's Research Lab scoring import closure.

The production gateway checkout is split: ``gateway/`` lives under
``$HOME/gateway`` while shared packages live under ``$HOME`` and are staged
under ``gateway/_attested_runtime`` before the EIF build. This module records
the exact local Python files reachable from the scoring entrypoints and fails
the build if any recorded file is absent or has different contents.

It is deliberately static analysis. Importing scoring modules here would run
configuration and provider initialization during the image build.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from gateway.tee.normalize_attested_runtime import normalized_file_mode
except Exception:
    from normalize_attested_runtime import normalized_file_mode


SCHEMA_VERSION = "leadpoet.gateway_scoring_import_closure.v1"
MANIFEST_RELATIVE_PATH = "_attested_runtime/scoring_import_closure.json"

PACKAGE_NAMES = (
    "gateway",
    "leadpoet_canonical",
    "leadpoet_verifier",
    "qualification",
    "research_lab",
    "validator_models",
)

ENTRYPOINT_MODULES = (
    "gateway.tee.tee_service",
    "gateway.tee.scoring_executor",
    "gateway.tee.scoring_job_manager",
    "gateway.research_lab.scoring_worker",
    "gateway.research_lab.allocations",
    "gateway.research_lab.promotion",
    "research_lab.eval.evaluator",
    "qualification.scoring.lead_scorer",
    "leadpoet_canonical.attested_receipts",
)

# The evaluator loads these with importlib so AST imports cannot discover them.
DYNAMIC_IMPORT_MODULES = (
    "gateway.qualification.models",
    "gateway.qualification.config",
    "gateway.qualification.utils.helpers",
    "qualification.scoring.lead_scorer",
    "qualification.scoring.verification_helpers",
)


class ScoringClosureError(RuntimeError):
    """Raised when a scoring import closure cannot be built or verified."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _module_name(package: str, package_root: Path, path: Path) -> str:
    relative = path.relative_to(package_root)
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join([package] + parts) if parts else package


def _iter_python_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return ()
    return (
        path
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    )


def build_module_index(*, gateway_root: Path, source_root: Path) -> Dict[str, Path]:
    gateway_root = gateway_root.resolve()
    source_root = source_root.resolve()
    index: Dict[str, Path] = {}
    for package in PACKAGE_NAMES:
        package_root = gateway_root if package == "gateway" else source_root / package
        if not package_root.is_dir():
            raise ScoringClosureError("required package root is missing: %s" % package_root)
        for path in _iter_python_files(package_root):
            module = _module_name(package, package_root, path)
            existing = index.get(module)
            if existing is not None and existing != path:
                raise ScoringClosureError("duplicate module %s: %s and %s" % (module, existing, path))
            index[module] = path
    return index


def _resolve_relative_module(current_module: str, level: int, module: Optional[str]) -> str:
    package_parts = current_module.split(".")[:-1]
    if level > len(package_parts) + 1:
        return ""
    keep = len(package_parts) - level + 1
    base = package_parts[: max(0, keep)]
    if module:
        base.extend(module.split("."))
    return ".".join(base)


def _candidate_imports(path: Path, module_name: str) -> Set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        raise ScoringClosureError("cannot parse %s: %s" % (path, exc)) from exc
    candidates: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            candidates.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = (
                _resolve_relative_module(module_name, node.level, node.module)
                if node.level
                else str(node.module or "")
            )
            if base:
                candidates.add(base)
                candidates.update(
                    base + "." + alias.name
                    for alias in node.names
                    if alias.name != "*"
                )
    return candidates


def _literal_environment_names(path: Path) -> Set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        raise ScoringClosureError("cannot parse %s: %s" % (path, exc)) from exc
    names: Set[str] = set()
    string_constants: Dict[str, str] = {}
    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Constant)
            and isinstance(value.value, str)
        ):
            string_constants[target.id] = value.value

    def resolved_name(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return string_constants.get(node.id)
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args and isinstance(node.func, ast.Attribute):
            first = resolved_name(node.args[0])
            if not first:
                continue
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr == "getenv"
            ):
                names.add(first)
            elif (
                node.func.attr == "get"
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "os"
                and node.func.value.attr == "environ"
            ):
                names.add(first)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.func, ast.Name):
            first = resolved_name(node.args[0])
            helper_name = node.func.id.lower()
            if (
                first
                and re.fullmatch(r"[A-Z][A-Z0-9_]+", first)
                and ("env" in helper_name or helper_name in {"_flag", "flag"})
            ):
                names.add(first)
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "os"
            and node.value.attr == "environ"
        ):
            slice_value = node.slice
            if hasattr(ast, "Index") and isinstance(slice_value, ast.Index):
                slice_value = slice_value.value
            resolved = resolved_name(slice_value)
            if resolved:
                names.add(resolved)
    return names


def _parents(module_name: str) -> Iterable[str]:
    parts = module_name.split(".")
    for index in range(1, len(parts)):
        yield ".".join(parts[:index])


def discover_scoring_modules(index: Dict[str, Path]) -> Tuple[str, ...]:
    roots = tuple(dict.fromkeys(ENTRYPOINT_MODULES + DYNAMIC_IMPORT_MODULES))
    missing_roots = [module for module in roots if module not in index]
    if missing_roots:
        raise ScoringClosureError("scoring entrypoint modules are missing: %s" % ", ".join(missing_roots))

    pending: List[str] = list(roots)
    discovered: Set[str] = set()
    while pending:
        module = pending.pop()
        if module in discovered:
            continue
        path = index.get(module)
        if path is None:
            continue
        discovered.add(module)
        for parent in _parents(module):
            if parent in index and parent not in discovered:
                pending.append(parent)
        for candidate in _candidate_imports(path, module):
            if candidate in index and candidate not in discovered:
                pending.append(candidate)
                continue
            parts = candidate.split(".")
            while len(parts) > 1:
                parts.pop()
                parent = ".".join(parts)
                if parent in index and parent not in discovered:
                    pending.append(parent)
                    break
    return tuple(sorted(discovered))


def _staged_relative_path(path: Path, *, gateway_root: Path, source_root: Path) -> str:
    path = path.resolve()
    gateway_root = gateway_root.resolve()
    source_root = source_root.resolve()
    try:
        return path.relative_to(gateway_root).as_posix()
    except ValueError:
        pass
    try:
        relative = path.relative_to(source_root)
    except ValueError as exc:
        raise ScoringClosureError("source path is outside known roots: %s" % path) from exc
    return (Path("_attested_runtime") / relative).as_posix()


def build_manifest(*, gateway_root: Path, source_root: Path) -> dict:
    index = build_module_index(gateway_root=gateway_root, source_root=source_root)
    modules = discover_scoring_modules(index)
    files = []
    environment_variables: Set[str] = set()
    for module in modules:
        path = index[module]
        environment_variables.update(_literal_environment_names(path))
        staged_path = _staged_relative_path(
            path,
            gateway_root=gateway_root,
            source_root=source_root,
        )
        item = {
                "module": module,
                "staged_path": staged_path,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
        }
        if staged_path.startswith("_attested_runtime/"):
            item["normalized_mode"] = "%04o" % normalized_file_mode(path)
            item["normalized_mtime_ns"] = 0
        files.append(item)
    body = {
        "schema_version": SCHEMA_VERSION,
        "entrypoint_modules": list(ENTRYPOINT_MODULES),
        "dynamic_import_modules": list(DYNAMIC_IMPORT_MODULES),
        "environment_variables": sorted(environment_variables),
        "files": files,
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        **body,
        "manifest_hash": "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def write_manifest(manifest: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def verify_staged_manifest(*, gateway_root: Path, manifest_path: Optional[Path] = None) -> dict:
    gateway_root = gateway_root.resolve()
    manifest_path = manifest_path or gateway_root / MANIFEST_RELATIVE_PATH
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoringClosureError("cannot read scoring import manifest: %s" % exc) from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ScoringClosureError("unsupported scoring import manifest schema")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ScoringClosureError("scoring import manifest has no files")
    environment_variables = manifest.get("environment_variables")
    if (
        not isinstance(environment_variables, list)
        or environment_variables != sorted(set(environment_variables))
        or any(not isinstance(item, str) or not item for item in environment_variables)
    ):
        raise ScoringClosureError("scoring import manifest environment variables are invalid")
    body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    expected_manifest_hash = "sha256:" + hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if manifest.get("manifest_hash") != expected_manifest_hash:
        raise ScoringClosureError("scoring import manifest hash mismatch")
    seen_paths: Set[str] = set()
    for item in files:
        relative = str(item.get("staged_path") or "")
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise ScoringClosureError("invalid staged path: %s" % relative)
        if relative in seen_paths:
            raise ScoringClosureError("duplicate staged path: %s" % relative)
        seen_paths.add(relative)
        staged = gateway_root / relative
        if not staged.is_file():
            raise ScoringClosureError("staged scoring dependency is missing: %s" % relative)
        if staged.stat().st_size != int(item.get("size_bytes") or -1):
            raise ScoringClosureError("staged scoring dependency size mismatch: %s" % relative)
        if _sha256_file(staged) != item.get("sha256"):
            raise ScoringClosureError("staged scoring dependency hash mismatch: %s" % relative)
        expected_mode = item.get("normalized_mode")
        if expected_mode is not None:
            actual_mode = "%04o" % stat.S_IMODE(staged.stat().st_mode)
            if actual_mode != expected_mode:
                raise ScoringClosureError("staged scoring dependency mode mismatch: %s" % relative)
            if staged.stat().st_mtime_ns != int(item.get("normalized_mtime_ns", -1)):
                raise ScoringClosureError("staged scoring dependency mtime mismatch: %s" % relative)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--gateway-root", required=True, type=Path)
    build.add_argument("--source-root", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    verify = subparsers.add_parser("verify-staged")
    verify.add_argument("--gateway-root", required=True, type=Path)
    verify.add_argument("--manifest", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        manifest = build_manifest(gateway_root=args.gateway_root, source_root=args.source_root)
        write_manifest(manifest, args.output)
        print("scoring_import_closure_files=%s" % len(manifest["files"]))
        print("scoring_import_closure_hash=%s" % manifest["manifest_hash"])
        return 0
    manifest = verify_staged_manifest(gateway_root=args.gateway_root, manifest_path=args.manifest)
    print("scoring_import_closure_verified=%s" % len(manifest["files"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
