"""Narrow auditor-protocol checks for an exact-commit V2 restart.

Immutable release evidence, manifests, artifacts, PCR0 measurements, and
dependency locks are verified by the normal gateway and validator release
launchers. This helper answers only the cross-version question needed before
those launchers run: can the current public auditor consume the selected
release's authoritative V2 weight protocol?
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, Optional, Sequence


SCHEMA_VERSION = "leadpoet.exact_commit_restart_compatibility.v3"

# Each entry is:
#   (contract name, public value, selected-release markers, auditor markers)
#
# These are wire and cryptographic-envelope contracts only. Implementation
# hashes, protected-workflow manifests, ancestry, and later reliability fixes
# are deliberately excluded: their absence does not make an older attested
# release incompatible with current auditors.
AUDITOR_PROTOCOL_CONTRACT = (
    (
        "weight_protocol",
        "authoritative_v2",
        (
            (
                "validator_tee/host/weight_protocol_v2.py",
                'AUTHORITATIVE_V2_PROTOCOL = "authoritative_v2"',
            ),
        ),
        (
            (
                "neurons/auditor_validator.py",
                "AUDITOR_WEIGHT_PROTOCOL must be authoritative_v2",
            ),
        ),
    ),
    (
        "published_authority_endpoint",
        "/weights/v2/published/{netuid}/{epoch_id}",
        (
            (
                "gateway/api/weights.py",
                '@router.get("/v2/published/{netuid}/{epoch_id}")',
            ),
        ),
        (
            (
                "neurons/auditor_validator.py",
                "/weights/v2/published/{netuid}/{int(epoch_id)}",
            ),
        ),
    ),
    (
        "finalized_authority_endpoint",
        "/weights/v2/latest/{netuid}/{epoch_id}",
        (
            (
                "gateway/api/weights.py",
                '@router.get("/v2/latest/{netuid}/{epoch_id}")',
            ),
        ),
        (
            (
                "neurons/auditor_validator.py",
                "/weights/v2/latest/{netuid}/{int(epoch_id)}",
            ),
        ),
    ),
    (
        "release_evidence_endpoint",
        "/weights/v2/release-evidence/{commit_sha}",
        (
            (
                "gateway/api/weights.py",
                '@router.get("/v2/release-evidence/{commit_sha}")',
            ),
        ),
        (
            (
                "neurons/auditor_validator.py",
                "/weights/v2/release-evidence/{commit}",
            ),
        ),
    ),
    (
        "release_evidence_schema",
        "leadpoet.auditor_release_evidence.v2",
        (
            (
                "gateway/api/weights.py",
                '"leadpoet.auditor_release_evidence.v2"',
            ),
        ),
        (
            (
                "leadpoet_canonical/auditor_v2.py",
                '"leadpoet.auditor_release_evidence.v2"',
            ),
        ),
    ),
    (
        "published_authority_schema",
        "leadpoet.published_weight_authority_stage.v2",
        (
            (
                "gateway/research_lab/attested_v2_store.py",
                '"leadpoet.published_weight_authority_stage.v2"',
            ),
        ),
        (
            (
                "leadpoet_canonical/auditor_v2.py",
                '"leadpoet.published_weight_authority_stage.v2"',
            ),
        ),
    ),
    (
        "finalized_authority_schema",
        "leadpoet.published_weight_authority.v2",
        (
            (
                "gateway/research_lab/attested_v2_store.py",
                '"leadpoet.published_weight_authority.v2"',
            ),
        ),
        (
            (
                "leadpoet_canonical/auditor_v2.py",
                '"leadpoet.published_weight_authority.v2"',
            ),
        ),
    ),
    (
        "weight_bundle_schema",
        "leadpoet.published_weight_bundle.v2",
        (
            (
                "leadpoet_canonical/weight_authority_v2.py",
                '"leadpoet.published_weight_bundle.v2"',
            ),
        ),
        (
            (
                "leadpoet_canonical/weight_authority_v2.py",
                '"leadpoet.published_weight_bundle.v2"',
            ),
        ),
    ),
    (
        "weight_publication_schema",
        "leadpoet.weight_publication.v2",
        (
            (
                "gateway/research_lab/attested_v2_store.py",
                '"leadpoet.weight_publication.v2"',
            ),
        ),
        (
            (
                "leadpoet_canonical/auditor_v2.py",
                '"leadpoet.weight_publication.v2"',
            ),
        ),
    ),
    (
        "weight_snapshot_schema",
        "leadpoet.weight_snapshot.v2",
        (
            (
                "leadpoet_canonical/weight_authority_v2.py",
                '"leadpoet.weight_snapshot.v2"',
            ),
        ),
        (
            (
                "leadpoet_canonical/weight_authority_v2.py",
                '"leadpoet.weight_snapshot.v2"',
            ),
        ),
    ),
    (
        "weight_finalization_submission_schema",
        "leadpoet.weight_finalization_submission.v2",
        (
            (
                "gateway/research_lab/attested_v2_store.py",
                '"leadpoet.weight_finalization_submission.v2"',
            ),
        ),
        (
            (
                "leadpoet_canonical/weight_authority_v2.py",
                '"leadpoet.weight_finalization_submission.v2"',
            ),
        ),
    ),
    (
        "weight_computation_request_schema",
        "leadpoet.weight_computation_request.v1",
        (
            (
                "leadpoet_canonical/weight_computation.py",
                '"leadpoet.weight_computation_request.v1"',
            ),
        ),
        (
            (
                "leadpoet_canonical/weight_computation.py",
                '"leadpoet.weight_computation_request.v1"',
            ),
        ),
    ),
    (
        "weight_computation_result_schema",
        "leadpoet.weight_computation_result.v1",
        (
            (
                "leadpoet_canonical/weight_computation.py",
                '"leadpoet.weight_computation_result.v1"',
            ),
        ),
        (
            (
                "leadpoet_canonical/weight_computation.py",
                '"leadpoet.weight_computation_result.v1"',
            ),
        ),
    ),
    (
        "chain_signing_profile_schema",
        "leadpoet.chain_signing_profile.v2",
        (
            (
                "leadpoet_canonical/hotkey_authority_v2.py",
                '"leadpoet.chain_signing_profile.v2"',
            ),
        ),
        (
            (
                "leadpoet_canonical/hotkey_authority_v2.py",
                '"leadpoet.chain_signing_profile.v2"',
            ),
        ),
    ),
    (
        "weight_extrinsic_authorization_schema",
        "leadpoet.weight_extrinsic_authorization.v3",
        (
            (
                "leadpoet_canonical/hotkey_authority_v2.py",
                '"leadpoet.weight_extrinsic_authorization.v3"',
            ),
        ),
        (
            (
                "leadpoet_canonical/hotkey_authority_v2.py",
                '"leadpoet.weight_extrinsic_authorization.v3"',
            ),
        ),
    ),
    (
        "weight_receipt_role",
        "validator_weights",
        (
            (
                "leadpoet_canonical/attested_v2.py",
                'WEIGHT_ROLE = "validator_weights"',
            ),
        ),
        (
            (
                "leadpoet_canonical/attested_v2.py",
                'WEIGHT_ROLE = "validator_weights"',
            ),
        ),
    ),
    (
        "weight_receipt_purpose",
        "validator.weights.computed.v2",
        (
            (
                "leadpoet_canonical/attested_v2.py",
                '"validator.weights.computed.v2"',
            ),
        ),
        (
            (
                "leadpoet_canonical/attested_v2.py",
                '"validator.weights.computed.v2"',
            ),
        ),
    ),
    (
        "boot_identity_schema",
        "leadpoet.attested_boot_identity.v2",
        (
            (
                "leadpoet_canonical/attested_v2.py",
                '"leadpoet.attested_boot_identity.v2"',
            ),
        ),
        (
            (
                "leadpoet_canonical/attested_v2.py",
                '"leadpoet.attested_boot_identity.v2"',
            ),
        ),
    ),
    (
        "execution_receipt_schema",
        "leadpoet.attested_execution_receipt.v2",
        (
            (
                "leadpoet_canonical/attested_v2.py",
                '"leadpoet.attested_execution_receipt.v2"',
            ),
        ),
        (
            (
                "leadpoet_canonical/attested_v2.py",
                '"leadpoet.attested_execution_receipt.v2"',
            ),
        ),
    ),
    (
        "receipt_graph_schema",
        "leadpoet.attested_receipt_graph.v2",
        (
            (
                "leadpoet_canonical/attested_v2.py",
                '"leadpoet.attested_receipt_graph.v2"',
            ),
        ),
        (
            (
                "leadpoet_canonical/attested_v2.py",
                '"leadpoet.attested_receipt_graph.v2"',
            ),
        ),
    ),
)
CANONICAL_ALGORITHM_ROOTS = (
    (
        "leadpoet_canonical/weight_computation.py",
        ("compute_final_weights",),
    ),
    (
        "leadpoet_canonical/weights.py",
        ("bundle_weights_hash",),
    ),
)
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class ExactCommitRestartCompatibilityError(RuntimeError):
    """The selected release cannot speak the current auditor protocol."""


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
) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
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


def _git_file(repo_root: Path, commit: str, path: str) -> str:
    return _git(repo_root, ["show", "%s:%s" % (commit, path)])


def _top_level_definitions(tree: ast.Module) -> Dict[str, ast.AST]:
    definitions = {}  # type: Dict[str, ast.AST]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            definitions[node.target.id] = node
    return definitions


def _algorithm_dependency_contract(
    *,
    source: str,
    path: str,
    roots: Sequence[str],
) -> Dict[str, str]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        raise ExactCommitRestartCompatibilityError(
            "canonical weight algorithm source is invalid in %s" % path
        ) from exc
    definitions = _top_level_definitions(tree)
    pending = list(roots)
    included = set()
    while pending:
        name = pending.pop()
        if name in included:
            continue
        node = definitions.get(name)
        if node is None:
            raise ExactCommitRestartCompatibilityError(
                "canonical weight algorithm root is missing in %s: %s"
                % (path, name)
            )
        included.add(name)
        referenced = {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }
        pending.extend(sorted(referenced.intersection(definitions) - included))

    contract = {}
    for name in sorted(included):
        normalized = _StripDocstrings().visit(definitions[name])
        encoded = ast.dump(
            normalized,
            annotate_fields=True,
            include_attributes=False,
        ).encode("utf-8")
        contract[name] = "sha256:" + hashlib.sha256(encoded).hexdigest()
    return contract


def _canonical_algorithm_contract(
    repo_root: Path,
    selected_commit: str,
    auditor_commit: str,
) -> Dict[str, Any]:
    combined = []
    for path, roots in CANONICAL_ALGORITHM_ROOTS:
        selected = _algorithm_dependency_contract(
            source=_git_file(repo_root, selected_commit, path),
            path=path,
            roots=roots,
        )
        auditor = _algorithm_dependency_contract(
            source=_git_file(repo_root, auditor_commit, path),
            path=path,
            roots=roots,
        )
        if selected != auditor:
            changed = sorted(
                name
                for name in set(selected) | set(auditor)
                if selected.get(name) != auditor.get(name)
            )
            raise ExactCommitRestartCompatibilityError(
                "selected release canonical weight algorithm differs from "
                "current auditors in %s:%s"
                % (path, changed[0] if changed else "<unknown>")
            )
        combined.extend(
            {"path": path, "symbol": name, "ast_sha256": digest}
            for name, digest in sorted(selected.items())
        )
    encoded = json.dumps(
        combined,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return {
        "entry_count": len(combined),
        "contract_hash": "sha256:" + hashlib.sha256(encoded).hexdigest(),
    }


def _auditor_protocol_contract(
    repo_root: Path,
    selected_commit: str,
    auditor_commit: str,
) -> Dict[str, Any]:
    selected_sources = {}  # type: Dict[str, str]
    auditor_sources = {}  # type: Dict[str, str]
    document = []
    for name, value, selected_markers, auditor_markers in (
        AUDITOR_PROTOCOL_CONTRACT
    ):
        for path, marker in selected_markers:
            if path not in selected_sources:
                selected_sources[path] = _git_file(
                    repo_root,
                    selected_commit,
                    path,
                )
            if marker not in selected_sources[path]:
                raise ExactCommitRestartCompatibilityError(
                    "selected release lacks auditor protocol capability "
                    "%s in %s" % (name, path)
                )
        for path, marker in auditor_markers:
            if path not in auditor_sources:
                auditor_sources[path] = _git_file(
                    repo_root,
                    auditor_commit,
                    path,
                )
            if marker not in auditor_sources[path]:
                raise ExactCommitRestartCompatibilityError(
                    "current auditor protocol declaration is inconsistent "
                    "for %s in %s" % (name, path)
                )
        document.append({"name": name, "value": value})

    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return {
        "entry_count": len(document),
        "contract_hash": "sha256:" + hashlib.sha256(encoded).hexdigest(),
        "entries": document,
    }


def verify_exact_commit_restart_compatibility(
    *,
    repo_root: Path,
    selected_commit: str,
    branch_ref: str,
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

    selected = _resolve_commit(root, selected_commit, "selected release")
    branch = _resolve_commit(root, branch_ref, "configured branch")
    protocol = _auditor_protocol_contract(root, selected, branch)
    algorithm = _canonical_algorithm_contract(root, selected, branch)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "compatible",
        "selected_commit": selected,
        "branch_commit": branch,
        "compatibility_scope": "auditor_weight_protocol",
        "auditor_protocol_contract_hash": protocol["contract_hash"],
        "auditor_protocol_entry_count": protocol["entry_count"],
        "canonical_weight_algorithm_hash": algorithm["contract_hash"],
        "canonical_weight_algorithm_entry_count": algorithm["entry_count"],
        "implementation_history_compared": False,
        "release_evidence_validation": "exact_release_launchers",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--selected-commit", required=True)
    parser.add_argument("--branch-ref", default="origin/main")
    args = parser.parse_args(argv)
    try:
        report = verify_exact_commit_restart_compatibility(
            repo_root=args.repo_root,
            selected_commit=args.selected_commit,
            branch_ref=args.branch_ref,
        )
    except ExactCommitRestartCompatibilityError as exc:
        raise SystemExit("ERROR: %s" % exc)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
