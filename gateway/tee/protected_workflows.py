"""Hash and verify protected gateway qualification and artifact logic ASTs.

The manifest deliberately hashes selected function, class, and policy-constant
definitions rather than whole files. I/O adapters and imports can move around
those definitions while CI continues to fail if protected gateway behavior
changes unintentionally. Arena weights use their separate small signer.
The pre-hydration bootstrap boundary additionally protects each complete module
AST so its import bindings cannot be redirected independently of its functions.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


SCHEMA_VERSION = "leadpoet.protected_workflows.v2"
DEFAULT_MANIFEST = Path(__file__).with_name("protected_workflows.json")

PROTECTED_SYMBOLS = {'gateway/main.py': ('lifespan',),
 'gateway/tee/code_hash.py': ('ATTESTED_RUNTIME_DIR',
                              'ATTESTED_RUNTIME_PACKAGES',
                              'ATTESTED_RUNTIME_FILES',
                              'ATTESTED_RUNTIME_GENERATED_FILES',
                              '_ATTESTED_RUNTIME_ROLES',
                              '_FULL_COMMIT_RE',
                              '_FALLBACK_COMMAND_TIMEOUT_SECONDS',
                              'ROOT_FILES',
                              'INCLUDE_DIRS',
                              'HASH_SUFFIXES',
                              'EXCLUDED_DIRS',
                              'EXCLUDED_SUFFIXES',
                              'EXCLUDED_NAMES',
                              'GatewayCodeHashError',
                              '_is_hashable',
                              '_iter_files',
                              '_fallback_environment',
                              '_run_fallback_command',
                              '_fallback_commit',
                              'materialize_gateway_code_hash_runtime',
                              'iter_gateway_code_hash_files',
                              'iter_gateway_code_hash_payloads',
                              'compute_gateway_code_hash'),
 'gateway/tee/mtls_identity.py': ('ATTESTED_TLS_CERTIFICATE_LIFETIME',
                                  'ATTESTED_TLS_CERTIFICATE_CLOCK_SKEW',
                                  '_atomic_private_write',
                                  'generate_ephemeral_tls_identity',
                                  'write_identity_to_tmpfs',
                                  'create_mutual_tls_context'),
 'gateway/tee/protected_workflows.py': ('stage_external_protected_sources', 'main'),
 'gateway/tee/release_archive_v2.py': ('_path_exists_without_following',
                                       '_path_without_symlink_ancestry',
                                       '_real_directory',
                                       '_load_regular_json',
                                       '_normalize_role_pcr0s',
                                       'load_last_good_release',
                                       '_release_role_pcr0s',
                                       '_archived_role_pcr0s',
                                       '_verify_index_entry',
                                       '_verify_archive_index_locked',
                                       'verify_archive_index',
                                       '_sha256_file',
                                       '_measurement_pcr0',
                                       '_atomic_json',
                                       '_expected_sources',
                                       '_copy_regular',
                                       'verify_archive_directory',
                                       '_restored_runtime_files',
                                       '_install_replace',
                                       '_copy_regular_for_rollback',
                                       '_fsync_directory',
                                       '_install_restored_runtime',
                                       '_promote_verified_release_locked',
                                       'restore_verified_release',
                                       'archive_verified_release',
                                       'select_release_manifest'),
 'gateway/tee/release_channel_v2.py': ('__module__',
                                       'build_release_channel_v2',
                                       'validate_release_channel_v2',
                                       'fetch_release_channel_v2',
                                       'build_release_lineage_v2',
                                       'fetch_release_lineage_v2',
                                       'git_ancestor_commits_v2'),
 'gateway/tee/release_lineage_v2.py': ('__module__',
                                       'validate_compact_release_lineage_v2',
                                       'build_compact_release_lineage_boot_verifier_v2'),
 'gateway/tee/release_manifest_v2.py': ('validate_release_manifest',
                                        'validate_prior_release_manifest',
                                        'historical_two_role_specs'),
 'gateway/tee/restart_preflight_v2.py': ('__module__',
                                         'FULL_TOPOLOGY_INSTANCE_TYPE',
                                         '_COMMIT_RE',
                                         'GatewayRestartPreflightV2Error',
                                         '_json',
                                         '_imds_instance_type',
                                         '_configured_processor_count',
                                         '_observed_capacity',
                                         'verify_gateway_restart_preflight_v2',
                                         'main'),
 'gateway/tee/rpc_authority.py': ('COORDINATOR_ROLE',
                                  'active_enclave_role',
                                  'allowed_exact_methods',
                                  'rpc_method_allowed'),
 'gateway/tee/runtime_identity_v2.py': ('RuntimeIdentityV2',
                                        '_validate_public_configuration',
                                        '_validate_release_configuration'),
 'gateway/tee/supabase_schema_preflight_v2.py': ('verify_required_supabase_v2_schema',),
 'gateway/tee/tee_service.py': ('VSOCK_RPC_TRANSPORT_HEALTH_SCHEMA_VERSION',
                                'MAX_VSOCK_RPC_CLEANUP_EVENT_COUNT',
                                'VSOCK_RPC_SUPERVISOR_POLL_SECONDS',
                                'VSOCK_RPC_CLEANUP_ATTEMPTS_PER_RECOVERY_CYCLE',
                                'MAX_VSOCK_RPC_CLEANUP_ATTEMPT_COUNT',
                                '_VSOCK_RPC_TRANSIENT_ACCEPT_ERRNOS',
                                'vsock_rpc_transport_health_lock',
                                'vsock_rpc_pending_cleanup_failures',
                                'vsock_rpc_terminal_failure_event',
                                'vsock_rpc_cleanup_recovery_lock',
                                'handle_v2_runtime_rpc',
                                'VSOCKRPCCleanupError',
                                '_ExplicitVSOCKCloseFailure',
                                '_close_vsock_rpc_required',
                                '_record_vsock_rpc_cleanup',
                                '_retain_vsock_rpc_cleanup_failure',
                                '_recover_vsock_rpc_cleanup_failures',
                                'vsock_rpc_transport_health',
                                '_handle_vsock_connection',
                                '_serve_vsock_connections'),
 'gateway/tee/verify_topology.py': ('_V2_RUNTIME_CONFIG_SCHEMA',
                                    'TopologyHealthError',
                                    'verify_roles'),
 'leadpoet_canonical/attested_v2.py': ('build_transport_attempt',
                                      'validate_transport_attempt',
                                      'build_boot_identity_body',
                                      'verify_boot_identity_nitro'),
 'leadpoet_canonical/chain_source_v2.py': ('CHAIN_MAX_RUNTIME_METADATA_BYTES',
                                           '_REVEAL_PERIOD_METADATA_DEFAULTS_V2',
                                           'configure_chain_source_boundary_v2',
                                           'chain_source_policy_document',
                                           'chain_source_policy_hash',
                                           'chain_source_boundary_for_profile_v2',
                                           'reveal_period_epochs_storage_key',
                                           'decode_reveal_period_epochs_storage',
                                           'decode_runtime_metadata_commitment',
                                           'resolve_reveal_period_metadata_default_v2',
                                           'system_events_storage_key',
                                           'system_event_count_storage_key',
                                           'timelocked_weight_commits_storage_key',
                                           'decode_timelocked_weight_commits',
                                           'weights_storage_key',
                                           'decode_weights_storage',
                                           'last_update_storage_key',
                                           'decode_last_update_storage',
                                           'parse_finalized_header',
                                           'ss58_encode_account_id'),
 'leadpoet_canonical/lab_arena_rewards.py': ('validate_reward_constants',
                                             'validate_reward_basis',
                                             'signing_key_from_document',
                                             'verify_reward_basis_signature',
                                             'signing_key_hash_from_environment',
                                             'rewards_enabled_from_environment',
                                             'reward_week_index',
                                             'champion_share_for_week',
                                             'governing_reward_basis',
                                             'epoch_eligible',
                                             'champion_uid_for_hotkey',
                                             'champion_values'),
 'leadpoet_canonical/production_parity_boundary_v2.py': ('PRODUCTION_SUPABASE_ORIGIN',
                                                         'PRODUCTION_CHAIN_HOST',
                                                         'PRODUCTION_CHAIN_ARCHIVE_HOST',
                                                         'PRODUCTION_PARITY_ENV_NAMES',
                                                         '_parity_configuration',
                                                         'validate_production_parity_boundary_document_v2',
                                                         'validate_production_parity_boundary_v2',
                                                         'configured_boundary_document_v2',
                                                         'configured_supabase_origin_v2',
                                                         'production_parity_enabled_v2',
                                                         'configured_chain_source_boundary_v2',
                                                         'configured_chain_signing_profile_path_v2',
                                                         'configured_rebenchmark_now_v2'),
 'leadpoet_canonical/proxy_transport.py': ('__module__',),
 'leadpoet_canonical/subtensor_events_v2.py': ('__module__',),
 'qualification/scoring/lead_scorer.py': ('_decision_from_observed_employee_size',
                                          '_has_explicitly_unproven_fit_dimensions',
                                          '_reverify_decision',
                                          '_run_targeted_company_evidence_investigation',
                                          '_llm_reverify_company'),
 'scripts/verify_installed_gateway_controller_v1.py': ('__module__',
                                                       'SUPPORTED_CONTROLLER_COMMITS',
                                                       'RECOVERY_HOST_CONTROLLER_COMMITS',
                                                       'LEGACY_FOUR_FILE_CONTROLLER_BOUNDARY',
                                                       'LEGACY_FOUR_FILE_CONTROLLER_COMMITS',
                                                       'CONTROLLER_FILES',
                                                       'OPTIONAL_CONTROLLER_FILES',
                                                       '_COMMIT_RE',
                                                       '_UNSAFE_GIT_ENV_NAMES',
                                                       'InstalledGatewayControllerError',
                                                       '_safe_git_environment',
                                                       '_git',
                                                       '_git_commit_exists',
                                                       '_git_is_ancestor',
                                                       '_require_unmodified_git_authority',
                                                       '_open_parent_fd',
                                                       '_read_exact_file',
                                                       '_verify_directory',
                                                       '_reviewed_controller_parent_paths',
                                                       'verify_candidate_bound_controller_lineage',
                                                       'verify_installed_controller_bundle',
                                                       '_exec_verified_helper',
                                                       '_recover_exact_controller_checkout_drift',
                                                       'main'),
 'validator_tee/host/docker_operation_guard_v2.py': ('_EXACT_HOST_GATEWAY_ARGS',
                                                     '_HOST_GATEWAY_PYTHON_COMMAND',
                                                     '_MAX_HOST_GATEWAY_CMDLINE_BYTES',
                                                     'inspect_exact_host_gateway_runtime')}


class ProtectedWorkflowError(RuntimeError):
    """A protected symbol is absent or has changed from the baseline."""


class _StripDocstrings(ast.NodeTransformer):
    def _strip(self, node: Any) -> Any:
        self.generic_visit(node)
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), (ast.Str, ast.Constant))
            and isinstance(getattr(body[0].value, "s", getattr(body[0].value, "value", None)), str)
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
    index = {"__module__": tree}  # type: Dict[str, ast.AST]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            index[node.name] = node
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    index[target.id] = node
        if isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(
            node.target, ast.Name
        ):
            index[node.target.id] = node
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    index[node.name + "." + child.name] = child
    return index


def _symbol_hash(node: ast.AST) -> str:
    normalized = _StripDocstrings().visit(ast.fix_missing_locations(node))
    encoded = ast.dump(normalized, annotate_fields=True, include_attributes=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _manifest_hash(body: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(body),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _source_path(root: Path, relative_path: str) -> Path:
    direct = root / relative_path
    if direct.is_file():
        return direct
    if relative_path.startswith("gateway/"):
        gateway_relative = root / relative_path.split("/", 1)[1]
        if gateway_relative.is_file():
            return gateway_relative
    staged = root / "_attested_runtime" / relative_path
    if staged.is_file():
        return staged
    return direct


def stage_external_protected_sources(source_root: Path, destination_root: Path) -> int:
    """Stage non-gateway protected sources into the measured runtime tree."""

    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    staged_count = 0
    for relative_path in sorted(PROTECTED_SYMBOLS):
        if relative_path.startswith("gateway/"):
            continue
        source = source_root / relative_path
        if not source.is_file() or source.is_symlink():
            raise ProtectedWorkflowError(
                "external protected source is unavailable: %s" % relative_path
            )
        destination = destination_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not destination.is_file() or destination.is_symlink():
                raise ProtectedWorkflowError(
                    "staged protected source is invalid: %s" % relative_path
                )
            if destination.read_bytes() != source.read_bytes():
                raise ProtectedWorkflowError(
                    "staged protected source differs: %s" % relative_path
                )
        else:
            destination.write_bytes(source.read_bytes())
            destination.chmod(source.stat().st_mode & 0o777)
        staged_count += 1
    return staged_count


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
        raise ProtectedWorkflowError("cannot resolve baseline Git commit") from exc


def build_manifest(
    root: Path,
    *,
    baseline_commit: str = "",
    protected_source_commit: str = "",
) -> Dict[str, Any]:
    root = root.resolve()
    entries = []
    for relative_path, symbols in sorted(PROTECTED_SYMBOLS.items()):
        path = _source_path(root, relative_path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            raise ProtectedWorkflowError("cannot parse protected file %s" % relative_path) from exc
        index = _symbol_index(tree)
        for symbol in symbols:
            if symbol not in index:
                raise ProtectedWorkflowError(
                    "protected symbol %s:%s is missing" % (relative_path, symbol)
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
    encoded = json.dumps(
        dict(manifest),
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
    ) + "\n"
    path.write_text(encoded, encoding="utf-8")


def load_manifest(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProtectedWorkflowError("cannot read protected workflow manifest") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != SCHEMA_VERSION
        or not isinstance(value.get("entries"), list)
        or set(value)
        != {
            "schema_version",
            "baseline_commit",
            "protected_source_commit",
            "entries",
            "manifest_hash",
        }
    ):
        raise ProtectedWorkflowError("protected workflow manifest schema is invalid")
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
        raise ProtectedWorkflowError("protected workflow manifest hash is invalid")
    return dict(value)


def verify_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    expected = build_manifest(
        root,
        baseline_commit=str(manifest.get("baseline_commit") or ""),
        protected_source_commit=str(
            manifest.get("protected_source_commit") or ""
        ),
    )
    if dict(manifest) != expected:
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
        raise ProtectedWorkflowError(
            "protected workflow manifest mismatch: %s" % ", ".join(changed)
        )


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--stage-external-root", type=Path)
    parser.add_argument("--baseline-commit", default="")
    parser.add_argument("--protected-source-commit", default="")
    args = parser.parse_args(list(argv) if argv else None)
    if args.write and args.stage_external_root is not None:
        parser.error("--write and --stage-external-root are mutually exclusive")
    if args.stage_external_root is not None:
        count = stage_external_protected_sources(args.root, args.stage_external_root)
        print("protected_external_sources_staged=%s" % count)
    elif args.write:
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
