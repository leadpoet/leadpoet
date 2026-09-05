"""Candidate-owned behavioral contract for the local V2 release rehearsal.

The rehearsal executes repository-owned entry points from one frozen candidate
commit.  This module gives the launcher runner and the final evidence join one
shared, machine-readable description of the behavior that must be observed.
It intentionally derives source and policy commitments from candidate code so
internal production changes do not require duplicated expectations in the
test harness.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


SCHEMA_VERSION = "leadpoet.v2_rehearsal_behavior_contract.v2"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

_PROFILE_EPOCH_COUNTS = {
    "prepush": 1,
    "release": 100,
}

# These are stable behaviors, not implementation phases. The runner can change
# how it reaches them without weakening the release contract.
BEHAVIOR_SCENARIOS = (
    "restart-summary-deadline-classification",
    "compact-weight-joined-path",
    "company-fit-numeric-observation-projection",
    "measured-assigned-proxy-raw-transport",
    "measured-coordinator-raw-transport",
    "artifact-egress-sustained-readback",
    "chain-settlement-state-space",
    "historical-metagraph-layouts",
    "receipt-graph-aggregate-pagination",
    "receipt-graph-transport-deduplication",
    "fresh-weight-input-lineage",
    "stateful-compact-graph-readback",
    "research-lab-allocation-conservation",
    "settlement-frontier-terminal-retirement",
    "current-frontier-release-recovery",
    "validator-publication-release-recovery",
)

AUTHORITY_DIAGNOSTICS = (
    "candidate-bundle-generation",
    "host-bundle-composition",
    "primary-bundle-verification",
    "auditor-bundle-verification",
    "primary-auditor-vector-equality",
    "sdk-signing-bridge",
)

BEHAVIORAL_INVARIANTS = (
    "candidate_identity_exact",
    "protected_source_identity_exact",
    "restart_summary_deadline_classification_exact",
    "compact_weight_joined_path_verified",
    "compact_ancestry_unknown_commit_recovery_verified",
    "compact_primary_auditor_byte_identity_verified",
    "compact_publication_journal_recovery_verified",
    "company_fit_numeric_observation_projection_verified",
    "measured_assigned_proxy_raw_transport_verified",
    "measured_coordinator_raw_transport_verified",
    "artifact_egress_sustained_readback_verified",
    "chain_settlement_state_space_complete",
    "historical_metagraph_layouts_policy_bound",
    "receipt_graph_aggregate_evidence_paged",
    "receipt_graph_transport_deduplicated_and_verified",
    "fresh_weight_input_lineage_verified",
    "stateful_compact_graph_readback_verified",
    "research_lab_allocation_policy_config_bound",
    "research_lab_allocation_conserved",
    "settlement_frontier_terminal_retirement_verified",
    "current_frontier_release_recovery_verified",
    "validator_publication_release_recovery_verified",
    "canonical_vector_primary_auditor_equal",
    "receipt_ancestry_verified",
    "sdk_signing_bridge_verified",
    "submission_finalized",
    "last_update_readback_equal",
    "boundary_cleanup_complete",
    "unknown_boundaries_rejected",
)

RESTART_INVARIANTS = (
    "validator_activation_requires_exact_gateway_release",
    "validator_role_release_identity_exact",
)

EXACT_PRODUCTION_ENTRYPOINTS = (
    "gw_restart.sh",
    "validator_restart.sh",
    "scripts/restart_attested_release_local.sh",
    "gateway/main.py",
    "gateway/tee/active_release_requirements_v2.py",
    "gateway/tee/code_hash.py",
    "gateway/tee/prepare_active_release_lineage_v2.py",
    "gateway/tee/protected_workflows.py",
    "gateway/tee/release_channel_v2.py",
    "gateway/tee/release_lineage_v2.py",
    "gateway/tee/rehearsal_behavior_contract_v2.py",
    "gateway/tee/stage_attested_runtime.sh",
    "gateway/tee/verify_weight_submission_ready_v2.py",
    "gateway/tee/artifact_persistence_v2.py",
    "gateway/tee/egress_framing.py",
    "gateway/tee/egress_policy.py",
    "gateway/tee/egress_proxy.py",
    "gateway/tee/execution_job_manager_v2.py",
    "gateway/tee/provider_broker_v2.py",
    "gateway/tee/provider_client_v2.py",
    "gateway/tee/provider_outcome_store_v2.py",
    "gateway/tee/rpc_authority.py",
    "gateway/tee/release_manifest_v2.py",
    "gateway/tee/scoring_executor.py",
    "gateway/tee/scoring_executor_v2.py",
    "gateway/tee/tee_service.py",
    "gateway/tee/topology.py",
    "gateway/utils/tee_egress_forwarder.py",
    "lab_arena/scorer_entrypoint.py",
    "lab_arena/scoring.py",
    "lab_arena/verify.py",
    "qualification/scoring/competition.py",
    "qualification/scoring/lead_scorer.py",
    "gateway/research_lab/champion_settlement_v2.py",
    "gateway/research_lab/allocations.py",
    "gateway/research_lab/stateful_epoch_authority_v1.py",
    "gateway/research_lab/attested_coordinator_v2.py",
    "gateway/research_lab/attested_v2_store.py",
    "gateway/research_lab/api.py",
    "gateway/research_lab/provider_preflight.py",
    "gateway/research_lab/source_add_trial_runner.py",
    "gateway/research_lab/store.py",
    "gateway/research_lab/v2_authority.py",
    "research_lab/docker_operation_lock_v2.py",
    "gateway/api/weights.py",
    "gateway/tee/coordinator_chain_realized_settlement_v1.py",
    "gateway/tee/coordinator_chain_source_v2.py",
    "gateway/tee/coordinator_executor_v2.py",
    "gateway/tee/research_lab_runtime_config_v2.py",
    "gateway/tee/prepare_gateway_envelopes_v2.py",
    "gateway/tee/topology.json",
    "gateway/tee/provider_evidence_cache_store_v2.py",
    "gateway/tee/provider_semantics_v2.py",
    "leadpoet_canonical/chain_source_v2.py",
    "leadpoet_canonical/subtensor_events_v2.py",
    "leadpoet_canonical/subtensor_events_profile_v2.json",
    "tests/restart_rehearsal/fixtures/subtensor_metadata_spec452_parent8984915.scale.gz",
    "leadpoet_verifier/economics.py",
    "leadpoet_canonical/auditor_latest_verified_bundle_v2.py",
    "leadpoet_canonical/compact_auditor_authority_v2.py",
    "neurons/validator.py",
    "neurons/auditor_validator.py",
    "validator_tee/enclave/hotkey_authority_v2.py",
    "validator_tee/enclave/weight_authority_v2.py",
    "validator_tee/host/authoritative_weight_flow_v2.py",
    "validator_tee/host/enclave_hotkey_v2.py",
    "validator_tee/host/publication_journal_v2.py",
    "validator_models/containerizing/deploy_dynamic.sh",
    "validator_tee/scripts/verify_pinned_gateway_release_v2.sh",
    "scripts/run_production_parity_full_host.py",
    "validator_tee/host/docker_operation_guard_v2.py",
    "validator_tee/scripts/docker_operation_lock_v2.sh",
    "validator_tee/scripts/reclaim_docker_storage_v2.sh",
    "tests/restart_rehearsal/boundary_contract.json",
    "tests/restart_rehearsal/compact_weight_joined_runner.py",
    "tests/restart_rehearsal/production_workflow_runner.py",
)


class RehearsalBehaviorContractV2Error(RuntimeError):
    """The candidate rehearsal contract is missing, stale, or inconsistent."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_protected_manifest(source_root: Path) -> dict[str, Any]:
    path = source_root / "gateway/tee/protected_workflows.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RehearsalBehaviorContractV2Error(
            "candidate protected-workflow manifest is unreadable"
        ) from exc
    entries = value.get("entries")
    if (
        value.get("schema_version") != "leadpoet.protected_workflows.v2"
        or not _SHA_RE.fullmatch(str(value.get("baseline_commit") or ""))
        or not _SHA_RE.fullmatch(
            str(value.get("protected_source_commit") or "")
        )
        or not isinstance(entries, list)
        or not entries
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(value.get("manifest_hash") or ""),
        )
    ):
        raise RehearsalBehaviorContractV2Error(
            "candidate protected-workflow manifest is invalid"
        )
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or not entry["path"]
            or not isinstance(entry.get("symbol"), str)
            or not entry["symbol"]
            or not re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(entry.get("ast_sha256") or ""),
            )
        ):
            raise RehearsalBehaviorContractV2Error(
                "candidate protected-workflow entry is invalid"
            )
    manifest_body = {
        "schema_version": value["schema_version"],
        "baseline_commit": value["baseline_commit"],
        "protected_source_commit": value["protected_source_commit"],
        "entries": entries,
    }
    if value["manifest_hash"] != _sha256_json(manifest_body):
        raise RehearsalBehaviorContractV2Error(
            "candidate protected-workflow manifest hash differs"
        )
    return value


def _policy_commitments() -> dict[str, Any]:
    # Import lazily so this module remains safe for launcher identity checks
    # that only need its schema constants.
    from gateway.research_lab.config import ResearchLabGatewayConfig
    from leadpoet_canonical.chain_source_v2 import (
        chain_source_policy_document,
        chain_source_policy_hash,
    )

    allocation_policy = ResearchLabGatewayConfig.from_env().reimbursement_policy_doc(
        enabled=True
    )
    chain_source_policy = chain_source_policy_document()
    return {
        "chain_source": {
            "policy": chain_source_policy,
            "policy_hash": chain_source_policy_hash(),
        },
        "research_lab_allocation": {
            "policy": allocation_policy,
            "policy_hash": _sha256_json(allocation_policy),
        },
    }


def _candidate_fault_ids(source_root: Path, profile: str) -> list[str]:
    if profile == "prepush":
        return []
    path = (
        source_root
        / "tests/restart_rehearsal/fixtures/production_shaped_v2.json"
    )
    try:
        fixture = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RehearsalBehaviorContractV2Error(
            "candidate rehearsal fault contract is unreadable"
        ) from exc
    values = fixture.get("fault_matrix")
    if not isinstance(values, list):
        raise RehearsalBehaviorContractV2Error(
            "candidate rehearsal fault contract is invalid"
        )
    normalized = [str(value or "").strip() for value in values]
    if (
        any(not value for value in normalized)
        or len(normalized) != len(set(normalized))
        or not normalized
    ):
        raise RehearsalBehaviorContractV2Error(
            "candidate rehearsal fault identities are invalid"
        )
    return normalized


def build_rehearsal_behavior_contract_v2(
    *,
    source_root: Path,
    candidate_sha: str,
    profile: str,
    epoch_count: int,
) -> dict[str, Any]:
    """Build the exact candidate contract consumed by execution and readback."""

    normalized_root = source_root.resolve()
    normalized_sha = str(candidate_sha or "").strip().lower()
    if not _SHA_RE.fullmatch(normalized_sha):
        raise RehearsalBehaviorContractV2Error(
            "candidate SHA must be a full lowercase Git commit"
        )
    if profile not in _PROFILE_EPOCH_COUNTS:
        raise RehearsalBehaviorContractV2Error(
            "rehearsal profile is unsupported"
        )
    expected_epochs = _PROFILE_EPOCH_COUNTS[profile]
    if int(epoch_count) != expected_epochs:
        raise RehearsalBehaviorContractV2Error(
            f"{profile} requires exactly {expected_epochs} workflow epochs"
        )

    protected = _load_protected_manifest(normalized_root)
    protected_paths = sorted(
        {str(entry["path"]) for entry in protected["entries"]}
    )
    source_paths = sorted(
        set(protected_paths) | set(EXACT_PRODUCTION_ENTRYPOINTS)
    )
    missing_sources = [
        path for path in source_paths if not (normalized_root / path).is_file()
    ]
    if missing_sources:
        raise RehearsalBehaviorContractV2Error(
            "candidate production entrypoints are missing: "
            + ",".join(missing_sources)
        )

    normalized_faults = _candidate_fault_ids(normalized_root, profile)

    required_stages = [
        "input-contract",
        "production-allocation-input",
        *[f"source-identity:{path}" for path in source_paths],
        *[
            f"behavior:{scenario}"
            for scenario in BEHAVIOR_SCENARIOS
        ],
        *[
            f"diagnostic:{diagnostic}"
            for diagnostic in AUTHORITY_DIAGNOSTICS
        ],
    ]
    if profile == "release":
        required_stages.extend(
            f"fault:{ordinal}:{fault}"
            for ordinal, fault in enumerate(normalized_faults)
        )
        required_stages.append("concurrency")
    required_stages.extend(
        [
            "boundary-start",
            *[
                f"epoch-{30_000 + ordinal}"
                for ordinal in range(expected_epochs)
            ],
            "boundary-cleanup",
            "workflow-evidence-validation",
        ]
    )
    if len(required_stages) != len(set(required_stages)):
        raise RehearsalBehaviorContractV2Error(
            "candidate rehearsal stages are not unique"
        )

    body = {
        "schema_version": SCHEMA_VERSION,
        "candidate_sha": normalized_sha,
        "profile": profile,
        "epoch_count": expected_epochs,
        "protected_manifest_hash": protected["manifest_hash"],
        "protected_source_paths": protected_paths,
        "production_source_paths": source_paths,
        "behavior_scenarios": list(BEHAVIOR_SCENARIOS),
        "authority_diagnostics": list(AUTHORITY_DIAGNOSTICS),
        "fault_ids": normalized_faults,
        "required_stage_ids": required_stages,
        "required_invariant_ids": list(BEHAVIORAL_INVARIANTS),
        "required_restart_invariant_ids": list(RESTART_INVARIANTS),
        "policy_commitments": _policy_commitments(),
    }
    return {**body, "contract_hash": _sha256_json(body)}


def validate_rehearsal_behavior_contract_v2(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize a serialized candidate behavior contract."""

    document = dict(value)
    supplied_hash = str(document.pop("contract_hash", "") or "")
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or not _SHA_RE.fullmatch(str(document.get("candidate_sha") or ""))
        or document.get("profile") not in _PROFILE_EPOCH_COUNTS
        or int(document.get("epoch_count") or 0)
        != _PROFILE_EPOCH_COUNTS[str(document.get("profile"))]
    ):
        raise RehearsalBehaviorContractV2Error(
            "serialized rehearsal behavior contract is invalid"
        )
    for field in (
        "protected_source_paths",
        "production_source_paths",
        "behavior_scenarios",
        "authority_diagnostics",
        "fault_ids",
        "required_stage_ids",
        "required_invariant_ids",
        "required_restart_invariant_ids",
    ):
        values = document.get(field)
        if (
            not isinstance(values, list)
            or any(not isinstance(item, str) or not item for item in values)
            or len(values) != len(set(values))
        ):
            raise RehearsalBehaviorContractV2Error(
                f"serialized rehearsal contract {field} is invalid"
            )
    if not isinstance(document.get("policy_commitments"), dict):
        raise RehearsalBehaviorContractV2Error(
            "serialized rehearsal policy commitments are invalid"
        )
    expected_hash = _sha256_json(document)
    if supplied_hash != expected_hash:
        raise RehearsalBehaviorContractV2Error(
            "serialized rehearsal behavior contract hash differs"
        )
    return {**document, "contract_hash": expected_hash}


__all__ = [
    "AUTHORITY_DIAGNOSTICS",
    "BEHAVIORAL_INVARIANTS",
    "BEHAVIOR_SCENARIOS",
    "EXACT_PRODUCTION_ENTRYPOINTS",
    "RESTART_INVARIANTS",
    "RehearsalBehaviorContractV2Error",
    "SCHEMA_VERSION",
    "build_rehearsal_behavior_contract_v2",
    "validate_rehearsal_behavior_contract_v2",
]
