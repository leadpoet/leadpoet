#!/usr/bin/env python3.11
"""Validate that the exact restart launcher consumed every required contract."""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import json
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys

candidate_source_root = Path(
    os.environ.get("REHEARSAL_CANDIDATE_SOURCE_ROOT", "/source")
)
if candidate_source_root.is_dir():
    sys.path.insert(0, str(candidate_source_root))

with redirect_stdout(sys.stderr):
    from gateway.research_lab.git_tree_models import (
        GitTreeContractError,
        TreePolicy,
        TreeReplacement,
        derive_child_slot,
        derive_tree_id,
    )
    from gateway.tee.supabase_schema_preflight_v2 import (
        REQUIRED_SUPABASE_V2_RPCS,
        REQUIRED_SUPABASE_V2_SCHEMA,
    )
    from leadpoet_canonical.attested_v2 import sha256_json
    if __package__:
        from .postgres_v2_contract_probe import EXPECTED_APPLIED_MIGRATIONS
    else:
        from postgres_v2_contract_probe import EXPECTED_APPLIED_MIGRATIONS


TARGETED_REGRESSION_SCOPE = "weight_readiness_regression"
VALIDATOR_GATEWAY_ACTIVATION_INVARIANT = (
    "validator_activation_requires_exact_gateway_release"
)
EXPECTED_GATEWAY_PRIVATE_MODEL_ENV = {
    "RESEARCH_LAB_PRIVATE_REPO_BRANCH": "leadpoet-lab",
    "RESEARCH_LAB_PRIVATE_MODEL_MANIFEST_URI": (
        "s3://leadpoet-private-model-artifacts-493765492819/"
        "research-lab/sourcing-model/branches/leadpoet-lab/current.json"
    ),
    "RESEARCH_LAB_PRIVATE_MODEL_KMS_KEY_ID": (
        "alias/leadpoet-research-lab-artifact-signing"
    ),
}
GIT_TREE_SCHEMA_MIGRATIONS = {
    "scripts/95-research-lab-git-tree-autoresearch.sql",
    "scripts/115-research-lab-git-tree-root-replacement.sql",
}
GIT_TREE_VIEW_RELATIONS = {
    "research_lab_autoresearch_tree_node_current",
    "research_lab_autoresearch_operation_current",
    "research_lab_autoresearch_tree_current",
    "research_lab_autoresearch_run_tree_current",
}
REQUIRED_GIT_TREE_RELATION_CONTRACT = {
    relation: {
        "kind": "v" if relation in GIT_TREE_VIEW_RELATIONS else "r",
        "columns": frozenset(columns),
    }
    for migration, relation, columns in REQUIRED_SUPABASE_V2_SCHEMA
    if migration in GIT_TREE_SCHEMA_MIGRATIONS
}
KNOWN_INTERNAL_SUBSTITUTION_MODULES = {
    "Leadpoet.utils.restart_epoch_gate",
    "gateway.tee.prepare_gateway_envelopes_v2",
    "gateway.tee.release_archive_v2",
    "gateway.tee.release_channel_v2",
    "gateway.tee.restart_preflight_v2",
    "gateway.tee.verify_v2_runtime_ready",
    "gateway.research_lab.provider_profiles_v2",
    "gateway.utils.tee_kms_provision_v2",
    "gateway.utils.tee_v2_bootstrap",
    "validator_tee.host.docker_operation_guard_v2",
    "validator_tee.host.hotkey_bootstrap_v2",
    "validator_tee.host.refresh_hotkey_config_v2",
    "validator_tee.host.release_archive_v2",
    "validator_tee.host.restart_preflight_v2",
    "validator_tee.host.runtime_v2_bootstrap",
    "validator_tee.host.verify_chain_signing_profile_v2",
    "validator_tee.host.verify_release_gate_v2",
    "validator_tee.scripts.stage_runtime_artifacts_v2",
}
KNOWN_INTERNAL_SUBSTITUTION_SCRIPTS = {
    "docker_image_normalizer_v2.py",
    "host_memory_guard_v2.py",
    "release_manifest_v2.py",
    "sandbox_runtime_artifact.py",
    "scoring_wheelhouse.py",
    "stage_runtime_artifacts_v2.py",
    "verify_release_artifacts_v2.py",
    "verify_topology.py",
}
KNOWN_INTERNAL_SUBSTITUTION_PROCESSES = {
    "gateway.main",
    "gateway.tee_egress",
    "gateway.tee_relay",
    "validator.chain_relay",
}
KNOWN_INTERNAL_SUBSTITUTION_BOUNDARIES = {
    "bash.build_drand_cabi_v2",
    "http.local_gateway",
    "host.containerd_state",
    "host.cpu_capacity",
    "host.filesystem_capacity",
    "host.memory_capacity",
    "host.mount_namespace",
    "host.process_lookup",
    "host.process_termination",
    "host.socket_state",
    "host.systemd",
    "host.timing",
    "python.scrub_parent_environment",
    "python.validator_coordinator",
    "python_dependencies.download",
    "python_dependencies.bootstrap",
    "python_dependencies.install",
    "python_dependencies.uninstall",
}


def events(
    state_root: Path = Path("/rehearsal-state"),
) -> list[dict]:
    rows: list[dict] = []
    for name in ("events.jsonl", "local-postgrest-events.jsonl"):
        path = state_root / name
        if not path.is_file():
            continue
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return sorted(rows, key=lambda row: int(row.get("at_ns") or 0))


def serialized_adapter_events(
    state_root: Path = Path("/rehearsal-state"),
) -> list[dict]:
    """Return adapter events in their authoritative append order."""

    path = state_root / "events.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def require_order(values: list[str], required: list[str]) -> None:
    cursor = -1
    for expected in required:
        try:
            cursor = values.index(expected, cursor + 1)
        except ValueError as exc:
            raise SystemExit(
                f"required rehearsal event is missing or out of order: {expected}"
            ) from exc


def _first_event(
    rows: list[dict],
    predicate,
    *,
    after: int = -1,
    before: int | None = None,
    description: str,
) -> int:
    upper_bound = len(rows) if before is None else before
    for ordinal in range(after + 1, upper_bound):
        if predicate(rows[ordinal]):
            return ordinal
    raise SystemExit(
        "required validator activation-barrier evidence is missing or out "
        f"of order: {description}"
    )


def _is_gateway_request(
    row: dict,
    *,
    endpoint_suffix: str,
    served_commit: str,
) -> bool:
    return (
        row.get("kind") == "curl"
        and row.get("boundary") == "http_service"
        and row.get("operation") == "gateway_request"
        and str(row.get("url") or "").endswith(endpoint_suffix)
        and row.get("served_commit") == served_commit
    )


def selected_validator_late_activation_capability(
    candidate_roots: tuple[Path, ...],
) -> bool:
    relative = Path("validator_models/containerizing/deploy_dynamic.sh")
    for root in candidate_roots:
        source = root / relative
        if source.is_file():
            return (
                "VALIDATOR_GATEWAY_ACTIVATION_BARRIER_V2=1"
                in source.read_text(encoding="utf-8")
            )
    raise SystemExit(
        "candidate validator deployment source is unavailable for "
        "activation-barrier capability proof"
    )


def verify_validator_gateway_activation_barrier(
    rows: list[dict],
    *,
    from_sha: str,
    candidate_sha: str,
    late_activation_supported: bool,
) -> dict[str, bool]:
    """Prove preparation overlap without permitting pre-alignment activation."""

    validator_processes = [
        ordinal
        for ordinal, row in enumerate(rows)
        if row.get("kind") == "validator-process"
        and row.get("process") == "validator.coordinator"
        and row.get("status") == "started"
    ]
    if len(validator_processes) != 1:
        raise SystemExit(
            "validator activation barrier did not produce exactly one "
            "coordinator process"
        )
    process_ordinal = validator_processes[0]

    health_ordinals = [
        ordinal
        for ordinal, row in enumerate(rows)
        if row.get("kind") == "curl"
        and row.get("boundary") == "http_service"
        and row.get("operation") == "gateway_request"
        and str(row.get("url") or "").endswith("/health/v2-authority")
    ]
    if len(health_ordinals) < 3:
        raise SystemExit(
            "validator activation barrier did not exercise stale, aligned, "
            "and poststart gateway authority probes"
        )
    first_health = rows[health_ordinals[0]]
    if (
        from_sha != candidate_sha
        and (
            first_health.get("served_commit") != from_sha
            or first_health.get("gateway_probe_attempt") != 1
        )
    ):
        raise SystemExit(
            "validator activation barrier did not reject the installed "
            "gateway release before candidate alignment"
        )

    application_build = _first_event(
        rows,
        lambda row: (
            row.get("kind") == "docker"
            and row.get("operation") == "build"
            and "leadpoet-validator:latest" in (row.get("argv") or [])
        ),
        before=process_ordinal,
        description="candidate validator application image build",
    )
    revision_inspect = _first_event(
        rows,
        lambda row: (
            row.get("kind") == "docker"
            and row.get("operation") == "inspect"
            and "leadpoet-validator:latest" in (row.get("argv") or [])
            and any(
                "org.opencontainers.image.revision" in str(value)
                for value in (row.get("argv") or [])
            )
        ),
        after=application_build,
        before=process_ordinal,
        description="candidate validator application commit verification",
    )
    if late_activation_supported:
        preflight_health = _first_event(
            rows,
            lambda row: _is_gateway_request(
                row,
                endpoint_suffix="/health/v2-authority",
                served_commit=candidate_sha,
            ),
            before=application_build,
            description="candidate authority health during preflight",
        )
        preflight_build = _first_event(
            rows,
            lambda row: _is_gateway_request(
                row,
                endpoint_suffix="/build-info",
                served_commit=candidate_sha,
            ),
            after=preflight_health,
            before=application_build,
            description="candidate build identity during preflight",
        )
        preflight_release = _first_event(
            rows,
            lambda row: (
                _is_gateway_request(
                    row,
                    endpoint_suffix=candidate_sha,
                    served_commit=candidate_sha,
                )
                and "/weights/v2/release-evidence/"
                in str(row.get("url") or "")
            ),
            after=preflight_build,
            before=application_build,
            description="candidate release evidence during preflight",
        )
        prepared_id = _first_event(
            rows,
            lambda row: (
                row.get("kind") == "docker"
                and row.get("operation") == "inspect"
                and "leadpoet-validator:latest" in (row.get("argv") or [])
                and "{{.Id}}" in (row.get("argv") or [])
            ),
            after=revision_inspect,
            before=process_ordinal,
            description="prepared validator application image identity",
        )
        preparation_predicates = (
            (
                "validator hotkey refresh",
                lambda row: row.get("module")
                == "validator_tee.host.refresh_hotkey_config_v2",
            ),
            (
                "validator restart preflight",
                lambda row: row.get("module")
                == "validator_tee.host.restart_preflight_v2",
            ),
            (
                "validator enclave build",
                lambda row: row.get("kind") == "nitro"
                and row.get("operation") == "build_enclave",
            ),
            (
                "validator enclave launch",
                lambda row: row.get("kind") == "nitro"
                and row.get("operation") == "run_enclave",
            ),
            (
                "validator chain relay",
                lambda row: row.get("kind") == "process"
                and row.get("process") == "validator.chain_relay",
            ),
            (
                "validator runtime bootstrap",
                lambda row: row.get("module")
                == "validator_tee.host.runtime_v2_bootstrap",
            ),
            (
                "validator hotkey bootstrap",
                lambda row: row.get("module")
                == "validator_tee.host.hotkey_bootstrap_v2",
            ),
        )
        for description, predicate in preparation_predicates:
            _first_event(
                rows,
                predicate,
                before=prepared_id,
                description=f"{description} before prepared image identity",
            )
        aligned_health = _first_event(
            rows,
            lambda row: _is_gateway_request(
                row,
                endpoint_suffix="/health/v2-authority",
                served_commit=candidate_sha,
            ),
            after=prepared_id,
            before=process_ordinal,
            description="candidate authority health at activation barrier",
        )
        aligned_build = _first_event(
            rows,
            lambda row: _is_gateway_request(
                row,
                endpoint_suffix="/build-info",
                served_commit=candidate_sha,
            ),
            after=aligned_health,
            before=process_ordinal,
            description="candidate build identity at activation barrier",
        )
        aligned_release = _first_event(
            rows,
            lambda row: (
                _is_gateway_request(
                    row,
                    endpoint_suffix=candidate_sha,
                    served_commit=candidate_sha,
                )
                and "/weights/v2/release-evidence/"
                in str(row.get("url") or "")
            ),
            after=aligned_build,
            before=process_ordinal,
            description="candidate release evidence at activation barrier",
        )
        if not (
            preflight_health
            < preflight_build
            < preflight_release
            < application_build
            < revision_inspect
            < prepared_id
            < aligned_health
            < aligned_build
            < aligned_release
        ):
            raise SystemExit(
                "candidate validator preparation did not complete before "
                "the late gateway activation barrier"
            )
        _first_event(
            rows,
            lambda row: (
                row.get("kind") == "docker"
                and row.get("operation") == "inspect"
                and "leadpoet-validator:latest" in (row.get("argv") or [])
                and "{{.Id}}" in (row.get("argv") or [])
            ),
            after=aligned_release,
            before=process_ordinal,
            description="unchanged validator image identity after alignment",
        )
    else:
        aligned_health = _first_event(
            rows,
            lambda row: _is_gateway_request(
                row,
                endpoint_suffix="/health/v2-authority",
                served_commit=candidate_sha,
            ),
            before=process_ordinal,
            description="candidate authority health before activation",
        )
        aligned_build = _first_event(
            rows,
            lambda row: _is_gateway_request(
                row,
                endpoint_suffix="/build-info",
                served_commit=candidate_sha,
            ),
            after=aligned_health,
            before=process_ordinal,
            description="candidate build identity before activation",
        )
        aligned_release = _first_event(
            rows,
            lambda row: (
                _is_gateway_request(
                    row,
                    endpoint_suffix=candidate_sha,
                    served_commit=candidate_sha,
                )
                and "/weights/v2/release-evidence/"
                in str(row.get("url") or "")
            ),
            after=aligned_build,
            before=process_ordinal,
            description="candidate release evidence before activation",
        )
        if not (
            aligned_health
            < aligned_build
            < aligned_release
            < application_build
            < revision_inspect
            < process_ordinal
        ):
            raise SystemExit(
                "legacy validator deployer did not remain behind the exact "
                "gateway release fallback barrier"
            )

    poststart_health = _first_event(
        rows,
        lambda row: _is_gateway_request(
            row,
            endpoint_suffix="/health/v2-authority",
            served_commit=candidate_sha,
        ),
        after=process_ordinal,
        description="poststart candidate authority health",
    )
    poststart_build = _first_event(
        rows,
        lambda row: _is_gateway_request(
            row,
            endpoint_suffix="/build-info",
            served_commit=candidate_sha,
        ),
        after=poststart_health,
        description="poststart candidate build identity",
    )
    _first_event(
        rows,
        lambda row: (
            _is_gateway_request(
                row,
                endpoint_suffix=candidate_sha,
                served_commit=candidate_sha,
            )
            and "/weights/v2/release-evidence/" in str(row.get("url") or "")
        ),
        after=poststart_build,
        description="poststart candidate release evidence",
    )

    return {VALIDATOR_GATEWAY_ACTIVATION_INVARIANT: True}


def _substitution_identity(row: dict) -> str:
    return str(
        row.get("substitution")
        or row.get("module")
        or row.get("script")
        or row.get("process")
        or ""
    )


def _git_blob(
    roots: tuple[Path, ...],
    commit: str,
    git_path: str,
) -> bytes | None:
    for root in roots:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "show",
                f"{commit}:{git_path}",
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            return result.stdout
    return None


def _verify_installed_checkout_handoff(
    *,
    source_path: Path,
    source_git_path: str,
    candidate_sha: str,
    candidate_roots: tuple[Path, ...],
) -> bool:
    resolved_source = source_path.resolve()
    for root in candidate_roots:
        resolved_root = root.resolve()
        if resolved_source != resolved_root and resolved_root not in resolved_source.parents:
            continue
        head = subprocess.run(
            ["git", "-C", str(resolved_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if head.returncode != 0 or head.stdout.strip() != candidate_sha:
            return False
        candidate_blob = _git_blob(
            (resolved_root,),
            candidate_sha,
            source_git_path,
        )
        if candidate_blob is None:
            return not source_path.exists()
        return (
            source_path.is_file()
            and source_path.read_bytes() == candidate_blob
        )
    return False


def _verify_production_identity(
    row: dict,
    from_sha: str | None,
    candidate_sha: str,
    candidate_roots: tuple[Path, ...],
) -> None:
    source_path = Path(str(row.get("source_path") or ""))
    source_git_path = str(row.get("source_git_path") or "")
    source_kind = str(row.get("source_kind") or "")
    source_hash = str(row.get("source_sha256") or "")
    source_commit = str(row.get("source_commit") or candidate_sha)
    allowed_source_commits = {candidate_sha}
    if from_sha:
        allowed_source_commits.add(from_sha)
    if (
        row.get("candidate_sha") != candidate_sha
        or source_commit not in allowed_source_commits
        or not source_git_path
        or source_kind
        not in {"installed_checkout", "candidate_checkout", "candidate_archive"}
        or (
            source_kind == "installed_checkout"
            and (not from_sha or source_commit != from_sha)
        )
        or (
            source_kind in {"candidate_checkout", "candidate_archive"}
            and source_commit != candidate_sha
        )
    ):
        raise SystemExit(
            "candidate production source identity is invalid: %r" % row
        )

    git_blob = _git_blob(candidate_roots, source_commit, source_git_path)
    if git_blob is None or source_hash != hashlib.sha256(git_blob).hexdigest():
        raise SystemExit(
            "candidate production source Git identity is invalid: %r" % row
        )

    if source_path.is_file():
        if source_hash != hashlib.sha256(source_path.read_bytes()).hexdigest():
            if source_kind != "installed_checkout" or not (
                _verify_installed_checkout_handoff(
                    source_path=source_path,
                    source_git_path=source_git_path,
                    candidate_sha=candidate_sha,
                    candidate_roots=candidate_roots,
                )
            ):
                raise SystemExit(
                    "candidate production source bytes changed after execution: %r"
                    % row
                )
    elif source_kind == "installed_checkout":
        if not _verify_installed_checkout_handoff(
            source_path=source_path,
            source_git_path=source_git_path,
            candidate_sha=candidate_sha,
            candidate_roots=candidate_roots,
        ):
            raise SystemExit(
                "installed production source disappeared without an exact "
                "candidate checkout handoff: %r" % row
            )
    elif source_kind != "candidate_archive":
        raise SystemExit(
            "candidate checkout source disappeared after execution: %r" % row
        )


def verify_rehearsal_integrity(
    rows: list[dict],
    *,
    from_sha: str | None = None,
    candidate_sha: str,
    scope: str,
    candidate_roots: tuple[Path, ...] = (
        Path("/home/ec2-user/leadpoet_repo"),
        Path("/home/ec2-user/leadpoet/leadpoet"),
    ),
) -> None:
    substitutions = [
        row
        for row in rows
        if row.get("implementation") == "internal_substitution"
    ]
    synthetic_external_fixtures = [
        row
        for row in rows
        if row.get("fixture_authenticity") == "synthetic"
    ]
    if scope == "exact" and substitutions:
        identities = sorted(
            {
                _substitution_identity(row) or "<unknown>"
                for row in substitutions
            }
        )
        raise SystemExit(
            "exact restart rehearsal used repository-code substitutions: "
            + ", ".join(identities)
        )
    if scope == "exact" and synthetic_external_fixtures:
        identities = sorted(
            {
                str(
                    row.get("operation")
                    or row.get("boundary")
                    or row.get("kind")
                    or "<unknown>"
                )
                for row in synthetic_external_fixtures
            }
        )
        raise SystemExit(
            "exact restart rehearsal used synthetic external fixtures: "
            + ", ".join(identities)
        )
    if scope == TARGETED_REGRESSION_SCOPE:
        for row in substitutions:
            identity = _substitution_identity(row)
            allowed = (
                identity in KNOWN_INTERNAL_SUBSTITUTION_MODULES
                or identity in KNOWN_INTERNAL_SUBSTITUTION_SCRIPTS
                or identity in KNOWN_INTERNAL_SUBSTITUTION_PROCESSES
                or identity in KNOWN_INTERNAL_SUBSTITUTION_BOUNDARIES
            )
            if not allowed:
                raise SystemExit(
                    "targeted regression used an unclassified internal "
                    f"substitution: {identity or '<unknown>'}"
                )
    elif scope != "exact":
        raise SystemExit("unknown rehearsal scope: %s" % scope)

    for row in rows:
        implementation = row.get("implementation")
        if implementation in {"production_module", "production_script"}:
            _verify_production_identity(
                row,
                from_sha,
                candidate_sha,
                candidate_roots,
            )


def verify_migration_backed_database_contract(
    durable_schema_sha: str,
) -> str:
    path = Path("/rehearsal-state/postgres-v2-schema-contract.json")
    if not path.is_file():
        raise SystemExit(
            "migration-backed PostgreSQL contract evidence is missing"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    required_checks = {
        "maintenance_lease_contract_valid",
        "pre_128_transport_rejected",
        "post_128_transport_persisted",
        "transport_contract_valid",
        "pre_129_attested_local_transport_rejected",
        "post_129_attested_local_transport_persisted",
        "transport_terminal_contract_valid",
        "pre_133_provider_outcome_contract_rejected",
        "post_133_provider_outcome_contract_valid",
        "pre_134_provider_outcome_head_contract_rejected",
        "post_134_provider_outcome_head_contract_valid",
        "post_135_active_model_replay_contract_valid",
        "post_136_ancestry_checkpoint_contract_valid",
        "post_137_allocation_settlement_frontier_contract_valid",
        "post_138_ancestry_checkpoint_bootstrap_purpose_valid",
        "post_139_allocation_frontier_bootstrap_contract_valid",
        "post_141_allocation_frontier_source_contract_valid",
        "post_142_source_catalog_replay_contract_valid",
        "post_143_compact_checkpoint_contract_valid",
        "provider_outcome_append_atomic",
        "provider_outcome_contention_zero_rollback",
        "provider_outcome_conflict_head_exact",
        "pre_132_lifetime_credit_rejected",
        "post_132_lifetime_credit_persisted",
        "lifetime_credit_rpc_idempotent",
        "grandfathered_credit_unchanged",
        "lifetime_credit_contract_valid",
        "finalized_view_projection_exact",
        "finalized_view_seed_available",
        "historical_compute_schema_migrations_applied",
        "git_tree_autoresearch_schema_migrations_applied",
        "git_tree_autoresearch_persistence_contract_valid",
        "git_tree_stale_root_rejected_atomically",
        "git_tree_replacement_and_restart_replay_valid",
        "historical_compute_finalized_authority_seed_available",
        "historical_compute_allocation_conserved",
        "historical_compute_release_identity_bound",
        "settlement_authority_parsed",
        "measured_settlement_receipt_projection_exact",
        "tampered_weight_receipt_rejected",
        "required_schema_migrations_declared",
    }
    checks = document.get("checks")
    applied_migrations = document.get("applied_migrations")
    if (
        document.get("schema_version")
        != "leadpoet.restart_rehearsal.postgres_contract.v1"
        or document.get("candidate_sha") != durable_schema_sha
        or not isinstance(checks, dict)
        or set(checks) != required_checks
        or any(checks[name] is not True for name in required_checks)
        or not isinstance(applied_migrations, list)
        or tuple(applied_migrations) != EXPECTED_APPLIED_MIGRATIONS
    ):
        raise SystemExit(
            "migration-backed PostgreSQL contract evidence is incomplete"
        )
    if document.get("provider_outcome_contention_contract") != {
        "schema_version": "leadpoet.provider_outcome_contention_contract.v3",
        "lock_contention_status": "busy",
        "stale_lineage_status": "conflict",
        "candidate_checkpoint_hash": True,
        "conflict_head_checkpoint_row": "encrypted_or_null",
    }:
        raise SystemExit(
            "migration-backed provider outcome contract evidence is missing"
        )
    if document.get("maintenance_lease") != {
        "schema_version": "leadpoet.maintenance_lease_contract.v1",
        "atomic_acquire": True,
        "live_contention_rejected": True,
        "same_holder_renewed": True,
        "expired_holder_replaced": True,
        "invalid_ttl_rejected": True,
    }:
        raise SystemExit(
            "migration-backed maintenance lease evidence is missing"
        )
    git_tree = document.get("git_tree_autoresearch")
    expected_git_tree_fields = {
        "run_id",
        "policy",
        "initial_tree_id",
        "replacement_tree_id",
        "replacement_hash",
        "replacement",
        "predecessor_event",
        "cancellation_event",
        "candidate_id",
        "candidate_artifact_hash",
        "evaluation_usage",
        "row_counts",
        "restart_replay_exact",
        "stale_root_rejected_atomically",
    }
    git_tree_commitments_valid = False
    if isinstance(git_tree, dict) and set(git_tree) == expected_git_tree_fields:
        try:
            policy = TreePolicy.from_mapping(git_tree.get("policy") or {})
            replacement = TreeReplacement.from_mapping(
                git_tree.get("replacement") or {}
            )
            predecessor = git_tree.get("predecessor_event")
            cancellation = git_tree.get("cancellation_event")
            candidate_artifact_hash = str(
                git_tree.get("candidate_artifact_hash") or ""
            )
            expected_run_id = "00000000-0000-4000-8000-000000000095"
            expected_initial_root = sha256_json({"root": "a"})
            expected_replacement_root = sha256_json({"root": "b"})
            expected_initial_manifest = sha256_json({"manifest": "a"})
            expected_replacement_manifest = sha256_json({"manifest": "b"})
            expected_candidate_artifact = sha256_json(
                {"candidate": "selected"}
            )
            expected_policy = TreePolicy(mode="active")
            expected_initial_slot = derive_child_slot(
                tree_id=derive_tree_id(
                    run_id=expected_run_id,
                    root_artifact_hash=expected_initial_root,
                    policy=expected_policy,
                ),
                parent_node_id="root",
                root_branch_id="",
                depth=1,
                slot_index=0,
            )
            expected_initial_lineage_hash = sha256_json(
                {
                    "schema_version": "research_lab.git_tree_lineage.v1",
                    "tree_id": expected_initial_slot.tree_id,
                    "node_id": expected_initial_slot.node_id,
                    "git_commit": "c" * 64,
                    "composition": {"root_git_commit": "a" * 64},
                }
            )
            expected_initial_selection = {
                "tree_id": expected_initial_slot.tree_id,
                "selected_node_id": expected_initial_slot.node_id,
                "paid_finalist_count": 1,
                "selected_candidate_artifact_hash": expected_candidate_artifact,
                "selected_node_git_commit": "c" * 64,
                "selected_lineage_hash": expected_initial_lineage_hash,
            }
            expected_predecessor_doc = {
                "selection_hash": sha256_json(expected_initial_selection),
                "selection": expected_initial_selection,
            }
            predecessor_doc = (
                dict(predecessor.get("event_doc") or {})
                if isinstance(predecessor, dict)
                else {}
            )
            predecessor_hash = (
                sha256_json(
                    {
                        "schema_version": "research_lab.git_tree_event.v1",
                        "tree_id": str(predecessor.get("tree_id") or ""),
                        "event_type": str(predecessor.get("event_type") or ""),
                        "node_id": str(predecessor.get("node_id") or ""),
                        "previous_event_hash": str(
                            predecessor.get("previous_event_hash") or ""
                        ),
                        "event_doc": predecessor_doc,
                    }
                )
                if isinstance(predecessor, dict)
                else ""
            )
            cancellation_doc = (
                dict(cancellation.get("event_doc") or {})
                if isinstance(cancellation, dict)
                else {}
            )
            cancellation_hash = (
                sha256_json(
                    {
                        "schema_version": "research_lab.git_tree_event.v1",
                        "tree_id": str(cancellation.get("tree_id") or ""),
                        "event_type": str(
                            cancellation.get("event_type") or ""
                        ),
                        "node_id": str(cancellation.get("node_id") or ""),
                        "previous_event_hash": str(
                            cancellation.get("previous_event_hash") or ""
                        ),
                        "event_doc": cancellation_doc,
                    }
                )
                if isinstance(cancellation, dict)
                else ""
            )
            expected_initial_tree = derive_tree_id(
                run_id=expected_run_id,
                root_artifact_hash=expected_initial_root,
                policy=expected_policy,
            )
            expected_replacement_tree = derive_tree_id(
                run_id=expected_run_id,
                root_artifact_hash=expected_replacement_root,
                policy=expected_policy,
                replacement=replacement,
            )
            git_tree_commitments_valid = bool(
                git_tree.get("run_id") == expected_run_id
                and policy == expected_policy
                and replacement.generation == 1
                and replacement.replaces_tree_id == expected_initial_tree
                and replacement.cancellation_event_hash == cancellation_hash
                and replacement.prior_root_artifact_hash
                == expected_initial_root
                and replacement.prior_root_manifest_hash
                == expected_initial_manifest
                and replacement.prior_policy_hash == expected_policy.policy_hash
                and replacement.root_artifact_hash
                == expected_replacement_root
                and replacement.root_manifest_hash
                == expected_replacement_manifest
                and replacement.policy_hash == expected_policy.policy_hash
                and git_tree.get("initial_tree_id") == expected_initial_tree
                and git_tree.get("replacement_tree_id")
                == expected_replacement_tree
                and git_tree.get("replacement_hash")
                == replacement.replacement_hash
                and isinstance(predecessor, dict)
                and set(predecessor)
                == {
                    "tree_id",
                    "seq",
                    "event_type",
                    "node_id",
                    "previous_event_hash",
                    "event_doc",
                    "event_hash",
                }
                and predecessor.get("tree_id") == expected_initial_tree
                and isinstance(predecessor.get("seq"), int)
                and not isinstance(predecessor.get("seq"), bool)
                and int(predecessor["seq"]) >= 0
                and predecessor.get("event_type") == "final_selected"
                and predecessor.get("node_id")
                == expected_initial_slot.node_id
                and re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(predecessor.get("previous_event_hash") or ""),
                )
                and predecessor_doc == expected_predecessor_doc
                and predecessor.get("event_hash") == predecessor_hash
                and isinstance(cancellation, dict)
                and set(cancellation)
                == {
                    "tree_id",
                    "event_type",
                    "node_id",
                    "previous_event_hash",
                    "event_doc",
                    "event_hash",
                }
                and cancellation.get("tree_id") == expected_initial_tree
                and cancellation.get("event_type")
                == "tree_cancelled_root_changed"
                and cancellation.get("node_id") == ""
                and cancellation.get("previous_event_hash")
                == predecessor_hash
                and re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(cancellation.get("previous_event_hash") or ""),
                )
                and cancellation.get("event_hash") == cancellation_hash
                and cancellation_doc.get("old_root_artifact_hash")
                == expected_initial_root
                and cancellation_doc.get("new_root_artifact_hash")
                == expected_replacement_root
                and cancellation_doc.get("old_root_manifest_hash")
                == expected_initial_manifest
                and cancellation_doc.get("new_root_manifest_hash")
                == expected_replacement_manifest
                and cancellation_doc.get("old_policy_hash")
                == expected_policy.policy_hash
                and cancellation_doc.get("new_policy_hash")
                == expected_policy.policy_hash
                and candidate_artifact_hash == expected_candidate_artifact
                and git_tree.get("candidate_id")
                == "candidate:" + expected_candidate_artifact.split(":", 1)[1]
            )
        except (GitTreeContractError, TypeError, ValueError):
            git_tree_commitments_valid = False
    if (
        not isinstance(git_tree, dict)
        or not git_tree_commitments_valid
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(git_tree.get("initial_tree_id") or ""),
        )
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(git_tree.get("replacement_tree_id") or ""),
        )
        or git_tree.get("initial_tree_id")
        == git_tree.get("replacement_tree_id")
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(git_tree.get("replacement_hash") or ""),
        )
        or not re.fullmatch(
            r"candidate:[0-9a-f]{64}",
            str(git_tree.get("candidate_id") or ""),
        )
        or git_tree.get("restart_replay_exact") is not True
        or git_tree.get("stale_root_rejected_atomically") is not True
        or git_tree.get("evaluation_usage")
        != {
            "settled_cost_microusd": 250,
            "provider_call_count": 4,
            "terminal_operation_count": 2,
            "unsettled_operation_ids": [],
            "indeterminate_operation_ids": [],
        }
        or git_tree.get("row_counts")
        != {
            "trees": 2,
            "handoffs": 1,
            "candidates": 1,
            "evaluation_events": 1,
        }
    ):
        raise SystemExit(
            "migration-backed Git-tree autoresearch evidence is missing"
        )
    provider_append = document.get("provider_outcome_append")
    if (
        not isinstance(provider_append, dict)
        or provider_append.get("accepted_count") != 1
        or provider_append.get("rejected_count") != 1
        or provider_append.get("row_count") != 3
        or provider_append.get("contention_rollback_delta") != 0
        or provider_append.get("durable_head_conflict_verified") is not True
        or provider_append.get("empty_head_conflict_verified") is not True
    ):
        raise SystemExit(
            "migration-backed provider outcome append evidence is missing"
        )
    relations = document.get("relations")
    if (
        not isinstance(relations, dict)
        or not {
            "research_lab_finalized_allocation_epochs_v2",
            "research_lab_attested_ancestry_checkpoints_v2",
            "research_lab_attested_ancestry_activations_v2",
            "research_lab_allocation_settlement_frontiers_v2",
            "research_lab_allocation_settlement_frontier_activation_v2",
            *REQUIRED_GIT_TREE_RELATION_CONTRACT,
        } <= set(relations)
    ):
        raise SystemExit(
            "migration-backed durable relation evidence is missing"
        )
    for relation, expected in REQUIRED_GIT_TREE_RELATION_CONTRACT.items():
        observed = relations.get(relation)
        if (
            not isinstance(observed, dict)
            or observed.get("kind") != expected["kind"]
            or not isinstance(observed.get("columns"), list)
            or not expected["columns"].issubset(observed["columns"])
        ):
            raise SystemExit(
                "migration-backed Git-tree relation evidence differs: "
                + relation
            )
    rpcs = document.get("rpcs")
    required_git_tree_rpcs = {
        function_name
        for migration, function_name in REQUIRED_SUPABASE_V2_RPCS
        if migration in GIT_TREE_SCHEMA_MIGRATIONS
    }
    if (
        not isinstance(rpcs, list)
        or "persist_research_lab_ancestry_checkpoint_v2" not in rpcs
        or "persist_research_lab_allocation_settlement_frontier_v2" not in rpcs
        or "persist_research_lab_allocation_frontier_bootstrap_v2"
        not in rpcs
        or "research_lab_ancestry_checkpoint_bootstrap_contract_v2" not in rpcs
        or "research_lab_allocation_frontier_bootstrap_contract_v2" not in rpcs
        or "research_lab_compact_checkpoint_graph_contract_v1" not in rpcs
        or not required_git_tree_rpcs.issubset(rpcs)
    ):
        raise SystemExit(
            "migration-backed ancestry checkpoint RPC evidence is missing"
        )
    frontier = document.get("allocation_settlement_frontier")
    if (
        not isinstance(frontier, dict)
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(frontier.get("frontier_hash") or ""),
        )
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(frontier.get("source_receipt_hash") or ""),
        )
        or frontier.get("idempotent_replay") is not True
        or frontier.get("frontier_count") != 1
        or frontier.get("activation_count") != 1
    ):
        raise SystemExit(
            "migration-backed allocation settlement frontier evidence is missing"
        )
    frontier_bootstrap = document.get(
        "allocation_settlement_frontier_bootstrap"
    )
    if (
        not isinstance(frontier_bootstrap, dict)
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(frontier_bootstrap.get("frontier_hash") or ""),
        )
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(
                frontier_bootstrap.get("allocation_source_receipt_hash")
                or ""
            ),
        )
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(frontier_bootstrap.get("bootstrap_receipt_hash") or ""),
        )
        or frontier_bootstrap.get("idempotent_replay") is not True
        or frontier_bootstrap.get("unmeasured_source_rejected") is not True
        or frontier_bootstrap.get("frontier_count") != 1
        or frontier_bootstrap.get("activation_count") != 1
    ):
        raise SystemExit(
            "migration-backed allocation frontier bootstrap evidence is missing"
        )
    seed_rows = document.get("seed_rows")
    required_seed_relations = {
        "research_lab_finalized_allocation_epochs_v2",
        "research_lab_emission_allocation_current",
        "research_lab_legacy_finalized_allocation_migrations_v2",
        "research_lab_attested_boot_identities_v2",
        "research_lab_attested_execution_receipts_v2",
        "research_lab_attested_receipt_edges_v2",
        "research_lab_attested_receipt_transport_v2",
        "research_lab_attested_transport_attempts_v2",
    }
    if (
        not isinstance(seed_rows, dict)
        or set(seed_rows) != required_seed_relations
    ):
        raise SystemExit(
            "migration-backed allocation authority seeds are missing"
        )
    for relation in sorted(required_seed_relations):
        rows = seed_rows.get(relation)
        fixed_count = {
            "research_lab_finalized_allocation_epochs_v2": 2,
            "research_lab_emission_allocation_current": 1,
            "research_lab_legacy_finalized_allocation_migrations_v2": 1,
        }.get(relation)
        if (
            relation not in relations
            or not isinstance(rows, list)
            or (fixed_count is not None and len(rows) != fixed_count)
            or (
                relation
                in {
                    "research_lab_attested_boot_identities_v2",
                    "research_lab_attested_execution_receipts_v2",
                }
                and not rows
            )
            or any(
                not isinstance(row, dict)
                or set(row) != set(relations[relation]["columns"])
                for row in rows
            )
        ):
            raise SystemExit(
                "migration-backed allocation authority seed differs: "
                + relation
            )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_durable_boundary_state(
    durable_schema_sha: str,
) -> dict[str, object]:
    ready_path = Path("/rehearsal-state/local-postgrest.ready")
    state_path = Path(
        "/rehearsal-durable-state/postgrest-state.json"
    )
    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "durable PostgREST transition evidence is unreadable"
        ) from exc
    start_revision = ready.get("durable_revision")
    end_revision = state.get("revision")
    start_hash = ready.get("durable_state_hash")
    end_hash = state.get("state_hash")
    if (
        ready.get("durable_schema_sha") != durable_schema_sha
        or state.get("durable_schema_sha") != durable_schema_sha
        or state.get("schema_version")
        != "leadpoet.local_postgrest_durable_state.v1"
        or not isinstance(start_revision, int)
        or start_revision < 0
        or not isinstance(end_revision, int)
        or end_revision < start_revision
        or not isinstance(start_hash, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", start_hash)
        or not isinstance(end_hash, str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", end_hash)
    ):
        raise SystemExit(
            "durable PostgREST transition evidence differs"
        )
    canonical_state = {
        "schema_version": state.get("schema_version"),
        "durable_schema_sha": state.get("durable_schema_sha"),
        "revision": end_revision,
        "rows": state.get("rows"),
    }
    calculated_hash = "sha256:" + hashlib.sha256(
        json.dumps(
            canonical_state,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if end_hash != calculated_hash:
        raise SystemExit(
            "durable PostgREST transition hash differs"
        )
    return {
        "schema_version": (
            "leadpoet.restart_rehearsal.durable_boundary_state.v1"
        ),
        "durable_schema_sha": durable_schema_sha,
        "start_revision": start_revision,
        "start_state_hash": start_hash,
        "end_revision": end_revision,
        "end_state_hash": end_hash,
    }


def verify_gateway_private_model_environment(rows: list[dict]) -> None:
    gateway_processes = [
        row
        for row in rows
        if row.get("kind") == "process"
        and row.get("process") == "gateway.main"
        and row.get("status") == "started"
    ]
    if len(gateway_processes) != 1:
        raise SystemExit(
            "gateway rehearsal did not launch exactly one gateway.main process"
        )
    if (
        gateway_processes[0].get("environment_contract")
        != EXPECTED_GATEWAY_PRIVATE_MODEL_ENV
    ):
        raise SystemExit(
            "gateway.main private-model source environment differs from "
            "the canonical restart contract"
        )


def verify_gateway_provider_preflight(
    rows: list[dict],
    *,
    transition: str,
) -> None:
    if transition != "forward":
        return
    expected = {
        ("api.exa.ai", "/search"),
        ("api.scrapingdog.com", "/account"),
    }
    observed = {
        (str(row.get("host") or ""), str(row.get("path") or ""))
        for row in rows
        if row.get("operation") == "provider_transport"
        and row.get("status") == 200
    }
    if not expected <= observed:
        raise SystemExit(
            "gateway provider preflight did not complete through both "
            "authenticated provider boundaries"
        )
    append_rows = [
        (ordinal, row)
        for ordinal, row in enumerate(rows)
        if row.get("kind") == "local-postgrest"
        and row.get("operation")
        == "provider_outcome_checkpoint_appended"
        and row.get("status") == "ok"
        and row.get("result_status") in {"inserted", "existing"}
    ]
    checkpoint_hashes = {
        str(row.get("checkpoint_hash") or "")
        for _ordinal, row in append_rows
    }
    if len(append_rows) < 2 or len(checkpoint_hashes) < 2:
        raise SystemExit(
            "gateway provider preflight did not durably append both "
            "provider outcomes"
        )
    for _append_ordinal, append_row in append_rows:
        if (
            append_row.get("method") != "POST"
            or append_row.get("target")
            != "append_research_lab_provider_outcome_checkpoint_v2"
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(append_row.get("checkpoint_hash") or ""),
            )
            or not isinstance(append_row.get("sequence"), int)
            or isinstance(append_row.get("sequence"), bool)
            or int(append_row["sequence"]) <= 0
        ):
            raise SystemExit(
                "gateway provider preflight atomic checkpoint "
                "acknowledgement is invalid"
            )
    first_append_ordinal = append_rows[0][0]
    readback_rows = [
        (ordinal, row)
        for ordinal, row in enumerate(rows)
        if row.get("kind") == "local-postgrest"
        and row.get("operation")
        == "provider_outcome_checkpoint_readback"
        and row.get("status") == "ok"
    ]
    if not any(ordinal < first_append_ordinal for ordinal, _row in readback_rows):
        raise SystemExit(
            "gateway provider preflight did not exercise checkpoint "
            "restart recovery before appending"
        )
    if any(ordinal > first_append_ordinal for ordinal, _row in readback_rows):
        raise SystemExit(
            "gateway provider preflight issued a redundant checkpoint "
            "readback after atomic append"
        )
    if any(
        row.get("kind") == "local-postgrest"
        and row.get("status") == "rejected"
        and (
            "provider_outcome" in str(row.get("target") or "")
            or "provider_outcome" in str(row.get("path") or "")
        )
        for row in rows
    ):
        raise SystemExit(
            "gateway provider preflight encountered a rejected checkpoint "
            "operation"
        )


def verify_chain_settlement_durable_readback(rows: list[dict]) -> None:
    persistence_ordinals = [
        ordinal
        for ordinal, row in enumerate(rows)
        if row.get("kind") == "local-postgrest"
        and row.get("operation") == "chain_settlement_persisted"
        and row.get("status") == "ok"
        and row.get("target")
        in {
            "persist_research_lab_chain_realized_settlement_v1",
            "persist_research_lab_chain_realized_unattributed_v2",
            "persist_research_lab_chain_realized_lifetime_settlement_v2",
        }
    ]
    settlement_reads = [
        (ordinal, int(row.get("row_count") or 0))
        for ordinal, row in enumerate(rows)
        if row.get("kind") == "local-postgrest"
        and row.get("operation") == "select"
        and row.get("status") == "ok"
        and row.get("target")
        == "research_lab_chain_realized_epoch_settlements_v1"
    ]
    credit_reads = [
        (ordinal, row.get("row_count"))
        for ordinal, row in enumerate(rows)
        if row.get("kind") == "local-postgrest"
        and row.get("operation") == "select"
        and row.get("status") == "ok"
        and row.get("target")
        == "research_lab_chain_realized_obligation_credits_v1"
    ]
    if not persistence_ordinals:
        if any(row_count == 1 for _ordinal, row_count in settlement_reads):
            return
        raise SystemExit(
            "gateway rehearsal neither persisted nor read an existing "
            "chain-realized settlement"
        )
    first_persistence = min(persistence_ordinals)
    if not any(
        ordinal > first_persistence and row_count == 1
        for ordinal, row_count in settlement_reads
    ):
        raise SystemExit(
            "gateway rehearsal did not read back its durable settlement"
        )
    if not any(
        ordinal > first_persistence and row_count is not None
        for ordinal, row_count in credit_reads
    ):
        raise SystemExit(
            "gateway rehearsal did not read back durable settlement credits"
        )


def verify_restart_epoch_transient_recovery(rows: list[dict]) -> None:
    head_reads = [
        row
        for row in rows
        if row.get("boundary") == "stateful_subnet_chain"
        and row.get("operation") == "epoch_snapshot"
        and row.get("method") == "get_chain_head"
    ]
    failed_ordinals = [
        ordinal
        for ordinal, row in enumerate(head_reads)
        if row.get("injected_failure") is True
    ]
    successful_ordinals = [
        ordinal
        for ordinal, row in enumerate(head_reads)
        if row.get("injected_failure") is False
    ]
    if not failed_ordinals or not successful_ordinals:
        raise SystemExit(
            "restart launcher did not exercise transient epoch-read recovery"
        )
    if not any(
        successful > failed
        for failed in failed_ordinals
        for successful in successful_ordinals
    ):
        raise SystemExit(
            "restart launcher did not recover after the injected epoch-read failure"
        )


def selected_weight_storage_preflight_capability(
    candidate_roots: tuple[Path, ...],
) -> bool:
    relative = Path("gateway/tee/verify_weight_submission_ready_v2.py")
    for root in candidate_roots:
        source_path = root / relative
        if not source_path.is_file():
            continue
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )
        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and bool(node.args)
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--storage-read-preflight"
            for node in ast.walk(tree)
        )
    raise SystemExit(
        "candidate weight-readiness source is unavailable for capability proof"
    )


def verify_gateway_weight_readiness_invocations(
    rows: list[dict],
    *,
    candidate_sha: str,
    transition: str = "forward",
    storage_preflight_supported: bool = True,
) -> None:
    module = "gateway.tee.verify_weight_submission_ready_v2"
    if not storage_preflight_supported and transition != "rollback":
        raise SystemExit(
            "current release does not declare the required weight storage preflight"
        )
    expected_prefix = []
    if storage_preflight_supported:
        expected_prefix.append(
            {
                "argv": ["-m", module, "--storage-read-preflight"],
                "source_kind": "candidate_archive",
            }
        )
    chain_repair_contract = {
        "argv": ["-m", module, "--repair-chain-settlements"],
        "source_kind": "candidate_checkout",
    }
    http_contract = {
        "argv": [
            "-m",
            module,
            "--gateway-url",
            "http://localhost:8000",
            "--http-timeout-seconds",
            "360",
        ],
        "source_kind": "candidate_checkout",
    }
    observed = [
        row
        for row in rows
        if row.get("kind") == "python-module"
        and row.get("module") == module
    ]
    prefix_count = len(expected_prefix)
    if (
        len(observed) > prefix_count
        and observed[prefix_count].get("argv")
        == ["-m", module, "--storage-read-preflight"]
        and observed[prefix_count].get("source_kind")
        == "candidate_checkout"
    ):
        expected_prefix.append(
            {
                "argv": ["-m", module, "--storage-read-preflight"],
                "source_kind": "candidate_checkout",
            }
        )
        prefix_count += 1
    if len(observed) < prefix_count + 3:
        raise SystemExit(
            "gateway launcher did not execute the exact production weight "
            f"readiness invocation contract: {observed!r}"
        )
    body_rows = observed[prefix_count:-1]
    expected = list(expected_prefix)
    repair_rows = []
    active_epoch = None
    chain_repair_seen = False
    cycle_has_repair = False
    for row in body_rows:
        argv = row.get("argv")
        if argv == chain_repair_contract["argv"]:
            if chain_repair_seen and not cycle_has_repair:
                raise SystemExit(
                    "gateway chain settlement repair was not followed by "
                    "pinned weight preparation"
                )
            expected.append(chain_repair_contract)
            active_epoch = None
            chain_repair_seen = True
            cycle_has_repair = False
            continue
        if (
            chain_repair_seen
            and isinstance(argv, list)
            and len(argv) == 5
            and argv[:4] == ["-m", module, "--repair", "--epoch"]
            and str(argv[4]).isdigit()
        ):
            if active_epoch is None:
                active_epoch = str(argv[4])
            elif str(argv[4]) != active_epoch:
                raise SystemExit(
                    "gateway weight repair retries changed their pinned epoch"
                )
            repair_rows.append(row)
            cycle_has_repair = True
            expected.append(
                {
                    "argv": list(argv),
                    "source_kind": "candidate_checkout",
                }
            )
            continue
        raise SystemExit(
            "gateway launcher weight readiness sequence is invalid: "
            f"{body_rows!r}"
        )
    if not repair_rows or active_epoch is None or not cycle_has_repair:
        raise SystemExit(
            "gateway launcher did not execute pinned weight readiness repair"
        )
    expected.append(http_contract)
    injected = any(
        row.get("kind") == "fault-injection"
        and row.get("module") == module
        and row.get("status") == "injected-transient-failure"
        for row in rows
    )
    if injected and len(repair_rows) < 2:
        raise SystemExit(
            "gateway launcher did not retry weight readiness after the "
            "injected transient failure"
        )
    for ordinal, (row, contract) in enumerate(
        zip(observed, expected),
        start=1,
    ):
        if (
            row.get("status") != "started"
            or row.get("implementation") != "production_module"
            or row.get("candidate_sha") != candidate_sha
            or row.get("source_commit") != candidate_sha
            or row.get("source_git_path")
            != "gateway/tee/verify_weight_submission_ready_v2.py"
            or row.get("source_kind") != contract["source_kind"]
            or row.get("argv") != contract["argv"]
        ):
            raise SystemExit(
                "gateway production weight readiness invocation differs from "
                f"the launcher contract at ordinal {ordinal}: {row!r}"
            )
def main() -> int:
    component, from_sha, candidate_sha = sys.argv[1:4]
    scenario = (
        sys.argv[4]
        if len(sys.argv) > 4
        else "production_success"
    )
    scope = sys.argv[5] if len(sys.argv) > 5 else "exact"
    transition = sys.argv[6] if len(sys.argv) > 6 else "forward"
    durable_schema_sha = os.environ.get(
        "REHEARSAL_DURABLE_SCHEMA_SHA",
        candidate_sha,
    )
    if transition not in {"forward", "rollback"}:
        raise SystemExit("unknown rehearsal transition: %s" % transition)
    rows = events()
    rejected = [row for row in rows if row.get("status") == "rejected"]
    if rejected:
        raise SystemExit(f"contract adapter rejected operations: {rejected!r}")
    verify_rehearsal_integrity(
        rows,
        from_sha=from_sha,
        candidate_sha=candidate_sha,
        scope=scope,
    )
    postgres_contract_sha256 = verify_migration_backed_database_contract(
        durable_schema_sha
    )
    durable_boundary_state = verify_durable_boundary_state(
        durable_schema_sha
    )
    if transition == "forward":
        verify_restart_epoch_transient_recovery(rows)

    labels: list[str] = []
    for row in rows:
        kind = row.get("kind")
        module = row.get("module")
        operation = row.get("operation")
        stage = row.get("stage")
        process = row.get("process")
        if module:
            labels.append(f"module:{module}")
        elif kind == "nitro":
            labels.append(f"nitro:{operation}")
        elif kind == "docker":
            labels.append(f"docker:{operation}")
        elif kind == "process":
            labels.append(f"process:{process}")

    if component == "gateway":
        restart_invariants: dict[str, bool] = {}
        if scenario != "production_success":
            raise SystemExit(
                "targeted fault scenario cannot satisfy exact restart evidence"
            )
        storage_preflight_supported = (
            selected_weight_storage_preflight_capability(
                (
                    Path("/home/ec2-user/leadpoet_repo"),
                    Path("/home/ec2-user/leadpoet/leadpoet"),
                )
            )
        )
        verify_gateway_weight_readiness_invocations(
            rows,
            candidate_sha=candidate_sha,
            transition=transition,
            storage_preflight_supported=storage_preflight_supported,
        )
        verify_chain_settlement_durable_readback(rows)
        required_gateway_order = [
            "module:gateway.tee.release_channel_v2",
            "module:gateway.tee.prepare_gateway_envelopes_v2",
        ]
        if storage_preflight_supported:
            required_gateway_order.append(
                "module:gateway.tee.verify_weight_submission_ready_v2"
            )
        required_gateway_order.extend(
            [
                "module:gateway.tee.restart_preflight_v2",
                "nitro:build_enclave",
                "nitro:run_enclave",
                "module:gateway.utils.tee_v2_bootstrap",
                "module:gateway.utils.tee_kms_provision_v2",
                "module:gateway.tee.verify_v2_runtime_ready",
                "module:gateway.tee.verify_weight_submission_ready_v2",
                "process:gateway.main",
                "module:gateway.tee.verify_weight_submission_ready_v2",
            ]
        )
        require_order(labels, required_gateway_order)
        verify_gateway_private_model_environment(rows)
        verify_gateway_provider_preflight(rows, transition=transition)
        state = json.loads(
            Path("/rehearsal-state/state.json").read_text(encoding="utf-8")
        )
        if len(state.get("enclaves", [])) != 3:
            raise SystemExit("gateway did not start the exact three-enclave topology")
    else:
        restart_invariants = verify_validator_gateway_activation_barrier(
            serialized_adapter_events(),
            from_sha=from_sha,
            candidate_sha=candidate_sha,
            late_activation_supported=(
                selected_validator_late_activation_capability(
                    (
                        Path("/home/ec2-user/leadpoet_repo"),
                        Path("/home/ec2-user/leadpoet/leadpoet"),
                    )
                )
            ),
        )
        require_order(
            labels,
            [
                "module:gateway.tee.release_channel_v2",
                "module:validator_tee.host.refresh_hotkey_config_v2",
                "module:validator_tee.host.restart_preflight_v2",
                "nitro:build_enclave",
                "nitro:run_enclave",
                "process:validator.chain_relay",
                "module:validator_tee.host.runtime_v2_bootstrap",
                "module:validator_tee.host.hotkey_bootstrap_v2",
            ],
        )
        state = json.loads(
            Path("/rehearsal-state/state.json").read_text(encoding="utf-8")
        )
        validator = state.get("containers", {}).get("leadpoet-validator-main", {})
        validator_log = Path(str(validator.get("log_path") or ""))
        if (
            validator.get("running") is not True
            or validator.get("restart_count") != 0
            or not isinstance(validator.get("pid"), int)
            or int(validator["pid"]) <= 0
            or not validator_log.is_file()
        ):
            raise SystemExit("validator final container state is invalid")

    pcr0_values = {
        str((enclave.get("Measurements") or {}).get("PCR0") or "")
        for enclave in state.get("enclaves", [])
    }
    if len(pcr0_values) != 1 or "" in pcr0_values:
        raise SystemExit("launcher enclave PCR0 evidence is missing or ambiguous")

    print(
        json.dumps(
            {
                "schema_version": "leadpoet.local_restart_rehearsal.v2",
                "status": (
                    "passed"
                    if scope == "exact"
                    else "targeted_regression_passed"
                ),
                "component": component,
                "scope": scope,
                "from_sha": from_sha,
                "candidate_sha": candidate_sha,
                "scenario": scenario,
                "transition": transition,
                "event_count": len(rows),
                "pcr0": next(iter(pcr0_values)),
                "postgres_contract_sha256": postgres_contract_sha256,
                "durable_schema_sha": durable_schema_sha,
                "durable_boundary_state": durable_boundary_state,
                "restart_invariants": restart_invariants,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
