"""Hash and verify protected validator V2 authority ASTs.

The validator EIF already measures every copied byte.  This manifest adds a
second, review-oriented gate around the calculation, receipt, signing, and
publication symbols that must not change as part of an adapter or packaging
edit.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Mapping, Sequence


SCHEMA_VERSION = "leadpoet.validator_protected_workflows.v2"
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "enclave"
    / "protected_workflows_v2.json"
)

PROTECTED_SYMBOLS = {
    "leadpoet_canonical/kms_recipient.py": (
        "decrypt_kms_recipient_ciphertext",
    ),
    "validator_tee/host/weight_protocol_v2.py": (
        "normalize_weight_protocol",
    ),
    "leadpoet_canonical/allocation_handoff_v2.py": (
        "build_allocation_handoff_v2",
        "validate_allocation_handoff_v2",
    ),
    "leadpoet_canonical/weight_computation.py": (
        "weight_config_document",
        "normalize_to_u16_with_uids_pure",
        "research_lab_uid_weights_from_allocation",
        "compute_final_weights",
    ),
    "leadpoet_canonical/weight_authority_v2.py": (
        "gateway_weight_input_value_documents_v2",
        "weight_input_value_documents_v2",
        "validate_weight_input_source_evidence_v2",
        "build_weight_snapshot_v2",
        "validate_published_weight_bundle_v2",
        "validate_weight_finalization_submission_v2",
    ),
    "leadpoet_canonical/hotkey_authority_v2.py": (
        "validate_chain_signing_profile",
        "build_weight_extrinsic_authorization_v2",
        "validate_weight_extrinsic_authorization_v2",
        "build_serve_axon_extrinsic_authorization_v2",
        "validate_serve_axon_extrinsic_authorization_v2",
        "classify_application_message_v2",
        "build_application_signature_request_v2",
        "validate_application_signature_request_v2",
    ),
    "leadpoet_canonical/auditor_v2.py": (
        "verify_attested_weight_bundle_v2",
        "verify_attested_weight_authority_v2",
        "verify_published_weight_authority_stage_v2",
        "_verify_publication_and_finalization",
    ),
    "validator_tee/enclave/weight_authority_v2.py": (
        "ValidatorWeightAuthorityV2.compute",
    ),
    "validator_tee/enclave/hotkey_authority_v2.py": (
        "ValidatorHotkeyAuthorityV2.provision_seed",
        "ValidatorHotkeyAuthorityV2.register_weight_result",
        "ValidatorHotkeyAuthorityV2.recover_weight_publication",
        "ValidatorHotkeyAuthorityV2.sign_application_message",
        "ValidatorHotkeyAuthorityV2.prepare_weight_commit",
        "ValidatorHotkeyAuthorityV2.sign_weight_extrinsic",
        "ValidatorHotkeyAuthorityV2.confirm_weight_publication",
        "ValidatorHotkeyAuthorityV2.sign_serve_axon_extrinsic",
    ),
    "validator_tee/enclave/chain_source_v2.py": (
        "EnclaveChainRpcTransportV2.call",
        "ValidatorChainSourceV2.read_finalized_snapshot",
        "ValidatorChainSourceV2.find_finalized_extrinsic_inclusion",
    ),
    "validator_tee/host/authoritative_weight_flow_v2.py": (
        "_verify_host_vector",
        "prepare_authoritative_weight_publication_v2",
        "resume_prepared_weight_publication_v2",
        "finalize_authoritative_weight_publication_v2",
    ),
    "neurons/validator.py": (
        "_validator_weight_protocol",
        "Validator.__init__",
        "Validator.set_weights",
        "Validator._research_lab_pre_weight_submission_guard",
        "Validator._publish_and_set_weights",
        "Validator._authorize_and_set_weights_v2",
        "Validator._recover_weight_publication_journal_v2",
        "Validator._set_weights_until_epoch_end",
        "Validator.submit_weights_at_epoch_end",
    ),
    "neurons/auditor_validator.py": (
        "AuditorValidator.fetch_attested_weights_v2",
        "AuditorValidator.fetch_verified_weight_authority",
        "AuditorValidator.verify_attested_weights_v2",
        "AuditorValidator.submit_weights_to_chain",
        "AuditorValidator.run",
    ),
}


class ValidatorProtectedWorkflowError(RuntimeError):
    """A protected validator symbol is absent or changed."""


class _StripDocstrings(ast.NodeTransformer):
    def _strip(self, node: Any) -> Any:
        self.generic_visit(node)
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), (ast.Str, ast.Constant))
            and isinstance(
                getattr(body[0].value, "s", getattr(body[0].value, "value", None)),
                str,
            )
        ):
            node.body = body[1:]
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        return self._strip(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        return self._strip(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        return self._strip(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        return self._strip(node)


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
        normalized, annotate_fields=True, include_attributes=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _manifest_hash(body: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(body), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip().lower()
    except Exception as exc:
        raise ValidatorProtectedWorkflowError(
            "cannot resolve validator protected source commit"
        ) from exc


def build_manifest(
    root: Path,
    *,
    baseline_commit: str = "",
    protected_source_commit: str = "",
) -> Dict[str, Any]:
    root = root.resolve()
    entries = []
    for relative_path, symbols in sorted(PROTECTED_SYMBOLS.items()):
        path = root / relative_path
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            raise ValidatorProtectedWorkflowError(
                "cannot parse protected validator file %s" % relative_path
            ) from exc
        index = _symbol_index(tree)
        for symbol in symbols:
            if symbol not in index:
                raise ValidatorProtectedWorkflowError(
                    "protected validator symbol %s:%s is missing"
                    % (relative_path, symbol)
                )
            entries.append(
                {
                    "path": relative_path,
                    "symbol": symbol,
                    "ast_sha256": _symbol_hash(index[symbol]),
                }
            )
    entries.sort(key=lambda item: (item["path"], item["symbol"]))
    body = {
        "schema_version": SCHEMA_VERSION,
        "baseline_commit": baseline_commit or _git_commit(root),
        "protected_source_commit": protected_source_commit or _git_commit(root),
        "entries": entries,
    }
    return {**body, "manifest_hash": _manifest_hash(body)}


def write_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(dict(manifest), sort_keys=True, indent=2, ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )


def load_manifest(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidatorProtectedWorkflowError(
            "cannot read validator protected workflow manifest"
        ) from exc
    expected_fields = {
        "schema_version",
        "baseline_commit",
        "protected_source_commit",
        "entries",
        "manifest_hash",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_fields
        or value.get("schema_version") != SCHEMA_VERSION
        or not isinstance(value.get("entries"), list)
    ):
        raise ValidatorProtectedWorkflowError(
            "validator protected workflow manifest schema is invalid"
        )
    body = {
        key: value[key]
        for key in (
            "schema_version",
            "baseline_commit",
            "protected_source_commit",
            "entries",
        )
    }
    if value.get("manifest_hash") != _manifest_hash(body):
        raise ValidatorProtectedWorkflowError(
            "validator protected workflow manifest hash is invalid"
        )
    return dict(value)


def verify_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    expected = build_manifest(
        root,
        baseline_commit=str(manifest.get("baseline_commit") or ""),
        protected_source_commit=str(
            manifest.get("protected_source_commit") or ""
        ),
    )
    if dict(manifest) == expected:
        return
    expected_by_key = {
        (item["path"], item["symbol"]): item["ast_sha256"]
        for item in expected["entries"]
    }
    observed_by_key = {
        (item.get("path"), item.get("symbol")): item.get("ast_sha256")
        for item in manifest.get("entries", [])
        if isinstance(item, dict)
    }
    changed = sorted(
        "%s:%s" % key
        for key in set(expected_by_key) | set(observed_by_key)
        if expected_by_key.get(key) != observed_by_key.get(key)
    )
    raise ValidatorProtectedWorkflowError(
        "validator protected workflow manifest mismatch: %s"
        % ", ".join(changed)
    )


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--baseline-commit", default="")
    parser.add_argument("--protected-source-commit", default="")
    args = parser.parse_args(list(argv) if argv else None)
    if args.write:
        write_manifest(
            build_manifest(
                args.root,
                baseline_commit=args.baseline_commit,
                protected_source_commit=args.protected_source_commit,
            ),
            args.manifest,
        )
    else:
        verify_manifest(args.root, load_manifest(args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
