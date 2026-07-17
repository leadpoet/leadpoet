"""Build and verify the gateway enclave's Research Lab execution import closure.

The production gateway checkout is split: ``gateway/`` lives under
``$HOME/gateway`` while shared packages live under ``$HOME`` and are staged
under ``gateway/_attested_runtime`` before the EIF build. This module records
the exact local Python files reachable from the scoring and auto-research
entrypoints and fails the build if any recorded file is absent or has different
contents.

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
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from gateway.tee.normalize_attested_runtime import normalized_file_mode
except Exception:
    from normalize_attested_runtime import normalized_file_mode


SCHEMA_VERSION = "leadpoet.gateway_execution_import_closure.v2"
MANIFEST_RELATIVE_PATH = "_attested_runtime/scoring_import_closure.json"

MEASURED_DATA_PATHS = (
    "gateway/api/role_patterns.json",
    "gateway/tee/protected_workflows.json",
    "gateway/tee/runsc-runtime.lock.json",
    "gateway/tee/topology.json",
    "gateway/utils/area_city_mappings.json",
    "gateway/utils/english_word_cities.txt",
    "gateway/utils/geo_lookup_fast.json",
    "gateway/utils/industry_equivalence.json",
    "schemas/evidence_bundle.schema.json",
    "schemas/execution_trace.schema.json",
    "schemas/research_evaluation_score_bundle.schema.json",
    "schemas/research_loop_start_contract.schema.json",
    "schemas/research_reimbursement.schema.json",
    "schemas/research_trajectory.schema.json",
    "schemas/results_ledger_row.schema.json",
)

PACKAGE_NAMES = (
    "gateway",
    "Leadpoet",
    "leadpoet_canonical",
    "leadpoet_verifier",
    "qualification",
    "research_lab",
    "validator_models",
)

ENTRYPOINT_MODULES = (
    "gateway.tee.tee_service",
    "gateway.tee.scoring_executor",
    "gateway.research_lab.scoring_worker",
    "gateway.research_lab.allocations",
    "gateway.research_lab.promotion",
    "research_lab.eval.evaluator",
    "qualification.scoring.lead_scorer",
    "leadpoet_canonical.attested_receipts",
    "leadpoet_canonical.attested_v2",
    "gateway.tee.protected_workflows",
)

# V2 autoresearch calculations execute in the measured autoresearch role. The
# parent remains an I/O adapter for signed host operations only. Listing every
# authority root here prevents the broad gateway tree COPY from hiding a
# missing executable dependency in a role-specific release manifest.
AUTORESEARCH_ENTRYPOINT_MODULES = (
    "gateway.tee.autoresearch_executor_v2",
    "gateway.tee.execution_job_manager_v2",
    "gateway.tee.host_operation_channel_v2",
    "gateway.tee.provider_client_v2",
    "gateway.tee.source_bundle_v2",
    "gateway.research_lab.worker_process",
    "gateway.research_lab.worker",
    "gateway.research_lab.autoresearch_runtime",
    "gateway.research_lab.code_loop_engine",
    "gateway.research_lab.code_build",
    "gateway.research_lab.dev_eval_runner",
    "gateway.research_lab.git_tree_evaluator",
    "gateway.research_lab.git_tree_models",
    "gateway.research_lab.git_tree_repository",
    "gateway.research_lab.git_tree_scheduler",
    "gateway.research_lab.git_tree_store",
    "research_lab.code_editing",
)

# The evaluator loads these with importlib so AST imports cannot discover them.
DYNAMIC_IMPORT_MODULES = (
    "gateway.qualification.models",
    "gateway.qualification.config",
    "gateway.qualification.utils.helpers",
    "qualification.scoring.lead_scorer",
    "qualification.scoring.verification_helpers",
)

ROLE_ENTRYPOINT_MODULES = {
    "gateway_coordinator": (
        "gateway.tee.tee_service",
        "gateway.tee.artifact_persistence_v2",
        "gateway.tee.artifact_vault_v2",
        "gateway.tee.coordinator_active_model_source_v2",
        "gateway.tee.coordinator_allocation_source_v2",
        "gateway.tee.coordinator_chain_source_v2",
        "gateway.tee.coordinator_epoch_cutover_v2",
        "gateway.tee.coordinator_executor_v2",
        "gateway.tee.coordinator_reward_source_v2",
        "gateway.tee.coordinator_source_add_v2",
        "gateway.tee.coordinator_weight_source_v2",
        "gateway.tee.egress_policy",
        "gateway.tee.egress_proxy",
        "gateway.tee.execution_job_manager_v2",
        "gateway.tee.inter_enclave_tls",
        "gateway.tee.inter_enclave_artifact_v2",
        "gateway.tee.kms_recipient_v2",
        "gateway.tee.mtls_identity",
        "gateway.tee.provider_broker_v2",
        "gateway.tee.provider_evidence_v2",
        "gateway.tee.provider_evidence_cache_store_v2",
        "gateway.tee.provider_semantics_v2",
        "gateway.tee.rpc_authority",
        "gateway.tee.runtime_identity_v2",
        "gateway.tee.topology",
        "gateway.tee.protected_workflows",
        "gateway.research_lab.active_model_authority_v2",
        "leadpoet_canonical.allocation_handoff_v2",
        "leadpoet_canonical.attested_v2",
        "leadpoet_canonical.weight_authority_v2",
        "gateway.research_lab.attested_v2_store",
    ),
    "gateway_scoring": ENTRYPOINT_MODULES
    + DYNAMIC_IMPORT_MODULES
    + (
        "gateway.tee.execution_job_manager_v2",
        "gateway.tee.inter_enclave_artifact_v2",
        "gateway.tee.model_sandbox_v2",
        "gateway.tee.mtls_identity",
        "gateway.tee.provider_client_v2",
        "gateway.tee.rpc_authority",
        "gateway.tee.runtime_identity_v2",
        "gateway.tee.sandbox_http_shim_v2",
        "gateway.tee.sandbox_provider_socket_v2",
        "gateway.tee.scoring_executor_v2",
        "gateway.tee.source_bundle_v2",
        "validator_models.automated_checks",
    ),
    "gateway_autoresearch": AUTORESEARCH_ENTRYPOINT_MODULES
    + (
        "gateway.tee.tee_service",
        "gateway.tee.topology",
        "gateway.tee.protected_workflows",
        "gateway.tee.mtls_identity",
        "gateway.tee.rpc_authority",
        "gateway.tee.runtime_identity_v2",
        "leadpoet_canonical.attested_v2",
    ),
}


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


def discover_modules(index: Dict[str, Path], roots: Sequence[str]) -> Tuple[str, ...]:
    roots = tuple(dict.fromkeys(roots))
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


def discover_scoring_modules(index: Dict[str, Path]) -> Tuple[str, ...]:
    return discover_modules(
        index,
        ENTRYPOINT_MODULES
        + AUTORESEARCH_ENTRYPOINT_MODULES
        + DYNAMIC_IMPORT_MODULES,
    )


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


def _measured_data_files(
    *, gateway_root: Path, source_root: Path
) -> List[dict]:
    output = []
    for relative_text in MEASURED_DATA_PATHS:
        relative = Path(relative_text)
        if relative.parts[0] == "gateway":
            source = gateway_root / Path(*relative.parts[1:])
        else:
            source = source_root / relative
        if not source.is_file():
            raise ScoringClosureError(
                "required runtime data file is missing: %s" % relative_text
            )
        staged_path = _staged_relative_path(
            source,
            gateway_root=gateway_root,
            source_root=source_root,
        )
        item = {
            "source_path": relative.as_posix(),
            "staged_path": staged_path,
            "sha256": _sha256_file(source),
            "size_bytes": source.stat().st_size,
        }
        if staged_path.startswith("_attested_runtime/"):
            item["normalized_mode"] = "%04o" % normalized_file_mode(source)
            item["normalized_mtime_ns"] = 0
        output.append(item)
    return output


def build_manifest(*, gateway_root: Path, source_root: Path) -> dict:
    index = build_module_index(gateway_root=gateway_root, source_root=source_root)
    role_modules = {
        role: discover_modules(index, roots)
        for role, roots in sorted(ROLE_ENTRYPOINT_MODULES.items())
    }
    modules = tuple(sorted({module for values in role_modules.values() for module in values}))
    files = []
    data_files = _measured_data_files(
        gateway_root=gateway_root,
        source_root=source_root,
    )
    environment_variables: Set[str] = set()
    files_by_module: Dict[str, dict] = {}
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
        files_by_module[module] = item
    role_manifests = {}
    for role, discovered in sorted(role_modules.items()):
        role_files = [files_by_module[module] for module in discovered]
        role_environment = sorted(
            {
                name
                for module in discovered
                for name in _literal_environment_names(index[module])
            }
        )
        role_body = {
            "role": role,
            "entrypoint_modules": list(ROLE_ENTRYPOINT_MODULES[role]),
            "environment_variables": role_environment,
            "files": role_files,
            "data_files": data_files,
        }
        role_encoded = json.dumps(
            role_body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        role_manifests[role] = {
            **role_body,
            "manifest_hash": "sha256:"
            + hashlib.sha256(role_encoded.encode("utf-8")).hexdigest(),
        }
    body = {
        "schema_version": SCHEMA_VERSION,
        "entrypoint_modules": list(ENTRYPOINT_MODULES),
        "autoresearch_entrypoint_modules": list(AUTORESEARCH_ENTRYPOINT_MODULES),
        "dynamic_import_modules": list(DYNAMIC_IMPORT_MODULES),
        "role_manifests": role_manifests,
        "environment_variables": sorted(environment_variables),
        "files": files,
        "data_files": data_files,
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


def verify_staged_manifest(
    *,
    gateway_root: Path,
    manifest_path: Optional[Path] = None,
    expected_role: str = "",
) -> dict:
    gateway_root = gateway_root.resolve()
    manifest_path = manifest_path or gateway_root / MANIFEST_RELATIVE_PATH
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoringClosureError("cannot read scoring import manifest: %s" % exc) from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ScoringClosureError("unsupported scoring import manifest schema")
    if manifest.get("entrypoint_modules") != list(ENTRYPOINT_MODULES):
        raise ScoringClosureError("scoring import manifest entrypoints are invalid")
    if manifest.get("autoresearch_entrypoint_modules") != list(
        AUTORESEARCH_ENTRYPOINT_MODULES
    ):
        raise ScoringClosureError("auto-research import manifest entrypoints are invalid")
    if manifest.get("dynamic_import_modules") != list(DYNAMIC_IMPORT_MODULES):
        raise ScoringClosureError("scoring dynamic import manifest roots are invalid")
    role_manifests = manifest.get("role_manifests")
    if not isinstance(role_manifests, Mapping) or set(role_manifests) != set(
        ROLE_ENTRYPOINT_MODULES
    ):
        raise ScoringClosureError("gateway role import manifests are invalid")
    if expected_role and expected_role not in role_manifests:
        raise ScoringClosureError("expected enclave role import manifest is missing")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ScoringClosureError("scoring import manifest has no files")
    data_files = manifest.get("data_files")
    if not isinstance(data_files, list) or not data_files:
        raise ScoringClosureError("scoring import manifest has no runtime data")
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
    for role, role_manifest in sorted(role_manifests.items()):
        if not isinstance(role_manifest, Mapping):
            raise ScoringClosureError("gateway role import manifest is invalid")
        role_body = {
            key: value for key, value in role_manifest.items() if key != "manifest_hash"
        }
        if role_body.get("role") != role:
            raise ScoringClosureError("gateway role import manifest role mismatch")
        if role_body.get("entrypoint_modules") != list(ROLE_ENTRYPOINT_MODULES[role]):
            raise ScoringClosureError("gateway role import entrypoints are invalid")
        expected_role_hash = "sha256:" + hashlib.sha256(
            json.dumps(
                role_body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        if role_manifest.get("manifest_hash") != expected_role_hash:
            raise ScoringClosureError("gateway role import manifest hash mismatch")
        role_files = role_body.get("files")
        if not isinstance(role_files, list) or not role_files:
            raise ScoringClosureError("gateway role import manifest has no files")
        if any(item not in files for item in role_files):
            raise ScoringClosureError("gateway role import file is absent from union closure")
        if role_body.get("data_files") != data_files:
            raise ScoringClosureError("gateway role runtime data closure is invalid")
    seen_paths: Set[str] = set()
    for item in files + data_files:
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
    verify.add_argument("--role", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        manifest = build_manifest(gateway_root=args.gateway_root, source_root=args.source_root)
        write_manifest(manifest, args.output)
        print("scoring_import_closure_files=%s" % len(manifest["files"]))
        print("scoring_import_closure_hash=%s" % manifest["manifest_hash"])
        return 0
    manifest = verify_staged_manifest(
        gateway_root=args.gateway_root,
        manifest_path=args.manifest,
        expected_role=args.role,
    )
    print("scoring_import_closure_verified=%s" % len(manifest["files"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
