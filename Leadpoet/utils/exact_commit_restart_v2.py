"""Fail-closed compatibility checks for an exact-commit V2 restart."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "leadpoet.exact_commit_restart_compatibility.v2"
PROTECTED_WORKFLOW_MANIFESTS = (
    (
        "gateway",
        "gateway/tee/protected_workflows.json",
        "leadpoet.protected_workflows.v2",
    ),
    (
        "validator",
        "validator_tee/enclave/protected_workflows_v2.json",
        "leadpoet.validator_protected_workflows.v2",
    ),
)
REQUIRED_RELEASE_MARKERS = {
    "gateway/api/weights.py": (
        '@router.get("/v2/release-evidence/{commit_sha}")',
        "leadpoet.auditor_release_evidence.v2",
    ),
    "gateway/main.py": (
        '@app.get("/health/v2-authority")',
        "leadpoet.gateway_v2_authority_health.v2",
    ),
    "gw_restart.sh": (
        "GATEWAY_RESTART_PHASE",
        "gateway.tee.release_channel_v2",
    ),
    "validator_restart.sh": (
        "validator_tee.host.restart_preflight_v2",
        "validator_tee.host.verify_release_gate_v2",
    ),
    "validator_models/containerizing/deploy_dynamic.sh": (
        "VALIDATOR_V2_DEPLOY_COMMIT",
        "authoritative_v2",
    ),
}
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class ExactCommitRestartCompatibilityError(RuntimeError):
    """The selected release cannot safely pair with the current V2 branch."""


class _StripDocstrings(ast.NodeTransformer):
    def _strip(self, node: Any) -> Any:
        self.generic_visit(node)
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(
                getattr(body[0], "value", None),
                (ast.Str, ast.Constant),
            )
            and isinstance(
                getattr(
                    body[0].value,
                    "s",
                    getattr(body[0].value, "value", None),
                ),
                str,
            )
        ):
            node.body = body[1:]
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        return self._strip(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return self._strip(node)

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> ast.AsyncFunctionDef:
        return self._strip(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        return self._strip(node)


def _git(
    repo_root: Path,
    args: Sequence[str],
    *,
    allow_ancestor_result: bool = False,
) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if allow_ancestor_result and result.returncode in (0, 1):
        return str(result.returncode)
    if result.returncode != 0:
        raise ExactCommitRestartCompatibilityError(
            "Git could not resolve the exact-commit compatibility contract"
        )
    return result.stdout


def _resolve_commit(repo_root: Path, value: str, label: str) -> str:
    commit = _git(repo_root, ["rev-parse", "--verify", value + "^{commit}"]).strip()
    if not _FULL_SHA_RE.fullmatch(commit):
        raise ExactCommitRestartCompatibilityError(
            "%s did not resolve to a full Git commit" % label
        )
    return commit


def _is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    return (
        _git(
            repo_root,
            ["merge-base", "--is-ancestor", ancestor, descendant],
            allow_ancestor_result=True,
        )
        == "0"
    )


def _git_file(repo_root: Path, commit: str, path: str) -> str:
    return _git(repo_root, ["show", "%s:%s" % (commit, path)])


def _git_file_hash(repo_root: Path, commit: str, path: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "show", "%s:%s" % (commit, path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ExactCommitRestartCompatibilityError(
            "Git could not read protected V2 source at %s:%s"
            % (commit, path)
        )
    return "sha256:" + hashlib.sha256(result.stdout).hexdigest()


def _manifest_hash(body: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(body),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _symbol_index(tree: ast.Module) -> Dict[str, ast.AST]:
    index = {}  # type: Dict[str, ast.AST]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            index[node.name] = node
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    index[node.name + "." + child.name] = child
    return index


def _symbol_hash(node: ast.AST) -> str:
    normalized = _StripDocstrings().visit(ast.fix_missing_locations(node))
    encoded = ast.dump(
        normalized,
        annotate_fields=True,
        include_attributes=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _protected_contract(
    repo_root: Path,
    commit: str,
    *,
    manifest_path: str,
    schema_version: str,
) -> Tuple[
    Tuple[Tuple[Tuple[str, str], str], ...],
    Tuple[Tuple[str, str], ...],
]:
    try:
        manifest = json.loads(
            _git_file(repo_root, commit, manifest_path)
        )
    except (TypeError, ValueError) as exc:
        raise ExactCommitRestartCompatibilityError(
            "protected V2 workflow manifest is invalid at %s" % commit
        ) from exc
    expected_fields = {
        "schema_version",
        "baseline_commit",
        "protected_source_commit",
        "entries",
        "manifest_hash",
    }
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != expected_fields
        or manifest.get("schema_version") != schema_version
        or not isinstance(manifest.get("entries"), list)
    ):
        raise ExactCommitRestartCompatibilityError(
            "protected V2 workflow manifest schema is invalid at %s" % commit
        )
    body = {
        key: manifest[key]
        for key in (
            "schema_version",
            "baseline_commit",
            "protected_source_commit",
            "entries",
        )
    }
    if manifest.get("manifest_hash") != _manifest_hash(body):
        raise ExactCommitRestartCompatibilityError(
            "protected V2 workflow manifest hash is invalid at %s" % commit
        )

    contract: Dict[Tuple[str, str], str] = {}
    for raw in manifest["entries"]:
        if not isinstance(raw, Mapping):
            raise ExactCommitRestartCompatibilityError(
                "protected V2 workflow entry is invalid at %s" % commit
            )
        key = (str(raw.get("path") or ""), str(raw.get("symbol") or ""))
        digest = str(raw.get("ast_sha256") or "")
        if (
            not key[0]
            or not key[1]
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
            or key in contract
        ):
            raise ExactCommitRestartCompatibilityError(
                "protected V2 workflow entry is malformed at %s" % commit
            )
        contract[key] = digest
    if not contract:
        raise ExactCommitRestartCompatibilityError(
            "protected V2 workflow contract is empty at %s" % commit
        )

    indexes = {}  # type: Dict[str, Dict[str, ast.AST]]
    for (path, symbol), expected_digest in sorted(contract.items()):
        if path not in indexes:
            try:
                tree = ast.parse(
                    _git_file(repo_root, commit, path),
                    filename="%s:%s" % (commit, path),
                )
            except (SyntaxError, ExactCommitRestartCompatibilityError) as exc:
                raise ExactCommitRestartCompatibilityError(
                    "protected V2 workflow source is invalid at %s:%s"
                    % (commit, path)
                ) from exc
            indexes[path] = _symbol_index(tree)
        node = indexes[path].get(symbol)
        if node is None or _symbol_hash(node) != expected_digest:
            raise ExactCommitRestartCompatibilityError(
                "protected V2 workflow manifest differs from source at %s:%s:%s"
                % (commit, path, symbol)
            )
    source_contract = tuple(
        (path, _git_file_hash(repo_root, commit, path))
        for path in sorted(indexes)
    )
    return tuple(sorted(contract.items())), source_contract


def _contract_hash(
    contract: Tuple[Tuple[Tuple[str, str], str], ...],
) -> str:
    document = [
        {"path": key[0], "symbol": key[1], "ast_sha256": digest}
        for key, digest in contract
    ]
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _source_contract_hash(
    contract: Tuple[Tuple[str, str], ...],
) -> str:
    document = [
        {"path": path, "source_sha256": digest}
        for path, digest in contract
    ]
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _verify_required_release_contract(
    repo_root: Path,
    selected_commit: str,
) -> None:
    for path, markers in REQUIRED_RELEASE_MARKERS.items():
        contents = _git_file(repo_root, selected_commit, path)
        missing = [marker for marker in markers if marker not in contents]
        if missing:
            raise ExactCommitRestartCompatibilityError(
                "selected release lacks required V2 contract marker in %s"
                % path
            )


def verify_exact_commit_restart_compatibility(
    *,
    repo_root: Path,
    selected_commit: str,
    branch_ref: str,
    compatibility_floor: str,
) -> Dict[str, Any]:
    root = repo_root.resolve()
    if not (root / ".git").exists():
        raise ExactCommitRestartCompatibilityError(
            "exact-commit compatibility repository is unavailable"
        )
    if not _FULL_SHA_RE.fullmatch(str(selected_commit or "")):
        raise ExactCommitRestartCompatibilityError(
            "selected release must be a lowercase full Git SHA"
        )
    if not _FULL_SHA_RE.fullmatch(str(compatibility_floor or "")):
        raise ExactCommitRestartCompatibilityError(
            "compatibility floor must be a lowercase full Git SHA"
        )

    selected = _resolve_commit(root, selected_commit, "selected release")
    branch = _resolve_commit(root, branch_ref, "configured branch")
    floor = _resolve_commit(root, compatibility_floor, "compatibility floor")
    if not _is_ancestor(root, selected, branch):
        raise ExactCommitRestartCompatibilityError(
            "selected release is not reachable from the configured branch"
        )
    if not _is_ancestor(root, floor, selected):
        raise ExactCommitRestartCompatibilityError(
            "selected release predates the supported V2 rollback floor"
        )

    _verify_required_release_contract(root, selected)
    contract_reports = {}
    combined_entries = []
    combined_sources = []
    for label, manifest_path, manifest_schema in PROTECTED_WORKFLOW_MANIFESTS:
        selected_contract, selected_sources = _protected_contract(
            root,
            selected,
            manifest_path=manifest_path,
            schema_version=manifest_schema,
        )
        branch_contract, branch_sources = _protected_contract(
            root,
            branch,
            manifest_path=manifest_path,
            schema_version=manifest_schema,
        )
        if selected_contract != branch_contract:
            selected_entries = dict(selected_contract)
            branch_entries = dict(branch_contract)
            changed = sorted(
                set(selected_entries).symmetric_difference(branch_entries)
                | {
                    key
                    for key in set(selected_entries).intersection(branch_entries)
                    if selected_entries[key] != branch_entries[key]
                }
            )
            first = changed[0] if changed else ("<unknown>", "<unknown>")
            raise ExactCommitRestartCompatibilityError(
                "selected release changes the %s protected V2 workflow "
                "contract required by current auditors: %s:%s"
                % (label, first[0], first[1])
            )
        if selected_sources != branch_sources:
            selected_files = dict(selected_sources)
            branch_files = dict(branch_sources)
            changed_paths = sorted(
                path
                for path in set(selected_files) | set(branch_files)
                if selected_files.get(path) != branch_files.get(path)
            )
            first_path = changed_paths[0] if changed_paths else "<unknown>"
            raise ExactCommitRestartCompatibilityError(
                "selected release changes a %s protected V2 source file "
                "required by current auditors: %s" % (label, first_path)
            )
        contract_reports[label] = {
            "entry_count": len(selected_contract),
            "contract_hash": _contract_hash(selected_contract),
            "source_file_count": len(selected_sources),
            "source_contract_hash": _source_contract_hash(selected_sources),
        }
        combined_entries.extend(
            ((("%s:%s" % (label, key[0]), key[1]), digest))
            for key, digest in selected_contract
        )
        combined_sources.extend(
            (("%s:%s" % (label, path), digest))
            for path, digest in selected_sources
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "compatible",
        "selected_commit": selected,
        "branch_commit": branch,
        "compatibility_floor": floor,
        "protected_workflow_contract_hash": _contract_hash(
            tuple(sorted(combined_entries))
        ),
        "protected_workflow_entry_count": len(combined_entries),
        "protected_source_contract_hash": _source_contract_hash(
            tuple(sorted(combined_sources))
        ),
        "protected_source_file_count": len(combined_sources),
        "protected_workflow_contracts": contract_reports,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--selected-commit", required=True)
    parser.add_argument("--branch-ref", default="origin/main")
    parser.add_argument("--compatibility-floor", required=True)
    args = parser.parse_args(argv)
    try:
        report = verify_exact_commit_restart_compatibility(
            repo_root=args.repo_root,
            selected_commit=args.selected_commit,
            branch_ref=args.branch_ref,
            compatibility_floor=args.compatibility_floor,
        )
    except ExactCommitRestartCompatibilityError as exc:
        raise SystemExit("ERROR: %s" % exc)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
