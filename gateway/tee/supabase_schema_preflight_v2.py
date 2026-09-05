"""Read-only PostgREST schema gate for the selected gateway V2 release."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Mapping
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REQUIRED_SUPABASE_V2_SCHEMA = (
    (
        "scripts/92-validator-sourcing-attested-v2.sql",
        "validator_sourcing_epoch_inputs_v2",
        ("epoch_id", "epoch_hash", "decision_root", "receipt_hash"),
    ),
    (
        "scripts/99-research-lab-v2-champion-settlement.sql",
        "research_lab_legacy_finalized_allocation_migrations_v2",
        ("netuid", "epoch_id", "allocation_hash", "settlement_receipt_hash"),
    ),
    (
        "scripts/101-stateful-subnet-epoch-authority.sql",
        "research_lab_stateful_subnet_epoch_candidates_v1",
        ("snapshot_hash", "netuid", "subnet_epoch_index"),
    ),
    (
        "scripts/101-stateful-subnet-epoch-authority.sql",
        "research_lab_stateful_subnet_epoch_cutovers_v1",
        ("cutover_authority_hash", "netuid", "first_subnet_epoch_index"),
    ),
    (
        "scripts/101-stateful-subnet-epoch-authority.sql",
        "research_lab_stateful_subnet_epoch_boundaries_v1",
        ("boundary_hash", "netuid", "subnet_epoch_index"),
    ),
    (
        "scripts/101-stateful-subnet-epoch-authority.sql",
        "research_lab_stateful_subnet_epoch_snapshots_v1",
        ("snapshot_hash", "netuid", "subnet_epoch_index"),
    ),
    (
        "scripts/101-stateful-subnet-epoch-authority.sql",
        "research_lab_stateful_subnet_epoch_cutover_state_v1",
        ("lifecycle_state", "mapping_hash", "netuid", "updated_at"),
    ),
    (
        "scripts/103-research-lab-legacy-allocation-nonfinalization.sql",
        "research_lab_legacy_allocation_nonfinalizations_v2",
        ("netuid", "epoch_id", "allocation_hash", "finding_receipt_hash"),
    ),
    (
        "scripts/125-research-lab-artifact-key-lineage.sql",
        "research_lab_provider_evidence_cache_v2",
        ("artifact_master_key_ref_hash",),
    ),
    (
        "scripts/125-research-lab-artifact-key-lineage.sql",
        "research_lab_provider_outcome_checkpoints_v2",
        ("artifact_master_key_ref_hash",),
    ),
    (
        "scripts/126-research-lab-chain-realized-settlement.sql",
        "research_lab_finalized_weight_vector_candidates_v1",
        (
            "netuid",
            "epoch_id",
            "validator_hotkey",
            "bundle_hash",
            "finalized_block",
            "uids",
            "weights_u16",
        ),
    ),
    (
        "scripts/126-research-lab-chain-realized-settlement.sql",
        "research_lab_chain_realized_epoch_settlements_v1",
        (
            "netuid",
            "epoch_id",
            "settlement_hash",
            "settlement_receipt_hash",
            "settlement_doc",
        ),
    ),
    (
        "scripts/126-research-lab-chain-realized-settlement.sql",
        "research_lab_chain_realized_settlement_activation_v1",
        (
            "netuid",
            "schema_version",
            "first_epoch_id",
            "source_bundle_hash",
            "source_bundle_epoch_id",
            "source_finalized_block",
        ),
    ),
    (
        "scripts/132-research-lab-champion-lifetime-credit.sql",
        "research_lab_chain_realized_obligation_credits_v1",
        (
            "netuid",
            "epoch_id",
            "settlement_hash",
            "obligation_kind",
            "obligation_source_id",
            "champion_credit_policy",
            "credit_hash",
            "credit_receipt_hash",
            "credit_doc",
        ),
    ),
    (
        "scripts/136-research-lab-ancestry-checkpoint-sidecars.sql",
        "research_lab_attested_ancestry_checkpoints_v2",
        (
            "root_receipt_hash",
            "schema_version",
            "lineage_id",
            "certificate_hash",
            "certificate_sequence",
            "issuer_boot_identity_hash",
            "proof_hash",
            "checkpoint_graph_hash",
            "certificate_doc",
            "proof_doc",
            "checkpoint_graph_doc",
        ),
    ),
    (
        "scripts/136-research-lab-ancestry-checkpoint-sidecars.sql",
        "research_lab_attested_ancestry_activations_v2",
        (
            "lineage_id",
            "activation_root_receipt_hash",
            "activation_certificate_hash",
        ),
    ),
    (
        "scripts/136-research-lab-ancestry-checkpoint-sidecars.sql",
        "research_lab_compact_weight_submissions_v2",
        (
            "compact_submission_hash",
            "bundle_hash",
            "netuid",
            "epoch_id",
            "validator_hotkey",
            "lineage_id",
            "binding_receipt_hash",
            "submission_doc",
        ),
    ),
    (
        "scripts/136-research-lab-ancestry-checkpoint-sidecars.sql",
        "research_lab_compact_weight_publication_intents_v2",
        (
            "bundle_hash",
            "compact_submission_hash",
            "netuid",
            "epoch_id",
            "validator_hotkey",
            "root_receipt_hash",
            "durable_readback_hash",
            "transparency_event_hash",
            "epoch_authority_hash",
            "intent_hash",
            "intent_doc",
        ),
    ),
    (
        "scripts/136-research-lab-ancestry-checkpoint-sidecars.sql",
        "research_lab_compact_weight_authorities_v2",
        (
            "bundle_hash",
            "netuid",
            "epoch_id",
            "validator_hotkey",
            "authority_stage",
            "schema_version",
            "lineage_id",
            "authority_hash",
            "compact_submission_hash",
            "publication_receipt_hash",
            "compact_finalization_hash",
            "finalization_receipt_hash",
            "authority_doc",
        ),
    ),
    (
        "scripts/137-research-lab-allocation-settlement-frontier.sql",
        "research_lab_allocation_settlement_frontiers_v2",
        (
            "netuid",
            "allocation_epoch",
            "settled_through_epoch",
            "schema_version",
            "frontier_hash",
            "predecessor_frontier_hash",
            "source_receipt_hash",
            "source_state_hash",
            "frontier_doc",
        ),
    ),
    (
        "scripts/137-research-lab-allocation-settlement-frontier.sql",
        "research_lab_allocation_settlement_frontier_activation_v2",
        (
            "netuid",
            "schema_version",
            "first_allocation_epoch",
            "first_frontier_hash",
            "source_receipt_hash",
        ),
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_submission_current",
        (
            "submission_id",
            "adapter_id",
            "miner_hotkey",
            "stage",
            "submission_doc",
            "precheck_status",
            "precheck_doc",
            "source_identity_hash",
            "source_identity_version",
        ),
    ),
    (
        "scripts/172-research-lab-source-add-claim-control.sql",
        "research_lab_source_add_control",
        (
            "singleton",
            "paused",
            "reason",
            "actor_ref",
            "updated_at",
            "restart_guard_commitment",
            "restart_guard_owner_commitment",
            "restart_guard_generation",
            "restart_guard_expires_at",
            "restart_guard_acquired_at",
            "restart_guard_actor_ref",
        ),
    ),
    (
        "scripts/174-research-lab-source-add-restart-state-restore.sql",
        "research_lab_source_add_control",
        ("restart_guard_restore_paused",),
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_work_items",
        (
            "work_id",
            "submission_id",
            "adapter_id",
            "work_kind",
            "work_status",
            "attempt_count",
            "available_at",
            "lease_token",
            "lease_expires_at",
            "job_doc",
            "result_doc",
        ),
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_probe_config_current",
        (
            "config_ref",
            "submission_id",
            "adapter_id",
            "config_status",
            "probe_doc",
            "credential_envelope",
        ),
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_functional_probe_current",
        (
            "attempt_ref",
            "submission_id",
            "adapter_id",
            "work_id",
            "evaluation_mode",
            "config_ref",
            "result_status",
            "route_hash",
            "receipt_hash",
            "business_artifact_hash",
            "result_doc",
        ),
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_provisioning_smoke_current",
        (
            "attempt_ref",
            "submission_id",
            "adapter_id",
            "work_id",
            "evaluation_mode",
            "config_ref",
            "result_status",
            "receipt_hash",
            "business_artifact_hash",
        ),
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_reward_intents",
        (
            "intent_id",
            "submission_id",
            "adapter_id",
            "miner_hotkey",
            "intent_status",
            "functional_receipt_hash",
            "business_artifact_hash",
            "available_at",
            "reward_ref",
        ),
    ),
    (
        "scripts/175-research-lab-source-add-provenance-leg1.sql",
        "research_lab_source_add_reward_intents",
        (
            "approval_kind",
            "provenance_receipt_hash",
            "provenance_artifact_hash",
        ),
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_reward_slots",
        (
            "slot_day",
            "slot_number",
            "intent_id",
            "work_id",
            "slot_status",
            "lease_token",
            "lease_expires_at",
            "reward_ref",
        ),
    ),
    (
        "scripts/175-research-lab-source-add-provenance-leg1.sql",
        "research_lab_source_add_reward_slots",
        ("approval_kind",),
    ),
    (
        "scripts/177-research-lab-source-add-provenance-authority-acl.sql",
        "research_lab_source_add_provenance_leg1_authority_v1",
        (
            "submission_id",
            "adapter_id",
            "miner_hotkey",
            "precheck_status",
            "provenance_receipt_hash",
            "provenance_artifact_hash",
            "provenance_created_at",
        ),
    ),
    (
        "scripts/178-research-lab-source-add-miner-status.sql",
        "research_lab_source_add_miner_status_v1",
        (
            "schema_version",
            "submission_id",
            "miner_hotkey",
            "source_name",
            "submitted_at",
            "updated_at",
            "decision_status",
            "decision_reason_code",
            "decision_reason",
            "reward_status",
            "alpha_percent",
            "reward_epochs",
            "start_epoch",
            "end_epoch",
        ),
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_identity_current",
        (
            "identity_version",
            "source_identity_hash",
            "submission_id",
            "adapter_id",
            "reservation_status",
            "seq",
        ),
    ),
    (
        "scripts/170-research-lab-source-add-provider-origin-uniqueness.sql",
        "research_lab_source_add_provider_origin_current",
        (
            "origin_version",
            "provider_origin_hash",
            "submission_id",
            "adapter_id",
            "miner_hotkey",
            "reservation_status",
            "seq",
        ),
    ),
    (
        "scripts/78-research-lab-source-add-catalog-provisioning.sql",
        "research_lab_source_add_provisioning_current",
        (
            "provision_ref",
            "catalog_id",
            "submission_id",
            "adapter_id",
            "miner_hotkey",
            "registry_provider_id",
            "provision_status",
            "provision_doc",
            "credential_envelope",
        ),
    ),
    (
        "scripts/72-research-lab-source-experiments.sql",
        "research_lab_source_catalog",
        (
            "catalog_id",
            "adapter_id",
            "miner_ref",
            "source_name",
            "source_kind",
            "declared_base_domains",
            "registry_provider_id",
        ),
    ),
    (
        "scripts/78-research-lab-source-add-catalog-provisioning.sql",
        "research_lab_source_catalog",
        ("source_identity_hash",),
    ),
    (
        "scripts/72-research-lab-source-experiments.sql",
        "research_lab_source_add_reward_current",
        (
            "reward_ref",
            "adapter_id",
            "catalog_id",
            "miner_hotkey",
            "leg",
            "alpha_percent",
            "reward_epochs",
            "start_epoch",
            "current_reward_status",
        ),
    ),
    (
        "scripts/183-lab-arena-miner-reward-basis.sql",
        "lab_arena_reward_basis_v1",
        (
            "round_id",
            "effective_reward_epoch",
            "reward_basis_hash",
            "reward_basis_doc",
            "signing_key_doc",
            "king_outcome",
            "king_hotkey",
            "king_start_epoch",
            "published_at",
        ),
    ),
)

REQUIRED_SUPABASE_V2_RPCS = (
    (
        "scripts/126-research-lab-chain-realized-settlement.sql",
        "persist_research_lab_chain_realized_settlement_v1",
    ),
    (
        "scripts/127-research-lab-chain-unattributed-settlement.sql",
        "persist_research_lab_chain_realized_unattributed_v2",
    ),
    (
        "scripts/128-research-lab-chain-settlement-transport-purposes.sql",
        "research_lab_attested_transport_purpose_contract_v2",
    ),
    (
        "scripts/129-research-lab-attested-local-transport.sql",
        "research_lab_attested_transport_terminal_contract_v2",
    ),
    (
        "scripts/133-research-lab-provider-outcome-contention-status.sql",
        "append_research_lab_provider_outcome_checkpoint_v2",
    ),
    (
        "scripts/133-research-lab-provider-outcome-contention-status.sql",
        "research_lab_provider_outcome_contention_contract_v2",
    ),
    (
        "scripts/134-research-lab-provider-outcome-head-contention.sql",
        "research_lab_provider_outcome_contention_contract_v3",
    ),
    (
        "scripts/144-research-lab-provider-persistence-batches.sql",
        "put_research_lab_provider_evidence_cache_v2",
    ),
    (
        "scripts/144-research-lab-provider-persistence-batches.sql",
        "append_research_lab_provider_outcome_checkpoints_v2",
    ),
    (
        "scripts/144-research-lab-provider-persistence-batches.sql",
        "research_lab_provider_persistence_batch_contract_v1",
    ),
    (
        "scripts/132-research-lab-champion-lifetime-credit.sql",
        "persist_research_lab_chain_realized_lifetime_settlement_v2",
    ),
    (
        "scripts/132-research-lab-champion-lifetime-credit.sql",
        "research_lab_champion_lifetime_credit_contract_v1",
    ),
    (
        "scripts/136-research-lab-ancestry-checkpoint-sidecars.sql",
        "persist_research_lab_ancestry_checkpoint_v2",
    ),
    (
        "scripts/155-research-lab-ancestry-disclosure-root-fast-path.sql",
        "research_lab_ancestry_disclosure_lookup_contract_v1",
    ),
    (
        "scripts/156-production-parity-readonly-role.sql",
        "leadpoet_production_parity_reader_contract_v1",
    ),
    (
        "scripts/137-research-lab-allocation-settlement-frontier.sql",
        "persist_research_lab_allocation_settlement_frontier_v2",
    ),
    (
        "scripts/138-research-lab-ancestry-checkpoint-bootstrap-purpose.sql",
        "research_lab_ancestry_checkpoint_bootstrap_contract_v2",
    ),
    (
        "scripts/139-research-lab-allocation-frontier-bootstrap.sql",
        "persist_research_lab_allocation_frontier_bootstrap_v2",
    ),
    (
        "scripts/139-research-lab-allocation-frontier-bootstrap.sql",
        "research_lab_allocation_frontier_bootstrap_contract_v2",
    ),
    (
        "scripts/141-research-lab-allocation-frontier-source-contract.sql",
        "research_lab_allocation_frontier_historical_source_contract_v1",
    ),
    (
        "scripts/142-research-lab-source-catalog-result-replay.sql",
        "research_lab_source_catalog_replay_contract_v2",
    ),
    (
        "scripts/143-research-lab-compact-ancestry-checkpoints.sql",
        "research_lab_compact_checkpoint_graph_contract_v1",
    ),
    (
        "scripts/149-research-lab-compact-weight-settlement-authority.sql",
        "research_lab_compact_weight_settlement_contract_v1",
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_admit",
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_begin_provider_execution",
    ),
    (
        "scripts/172-research-lab-source-add-claim-control.sql",
        "research_lab_source_add_claim_work",
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_finish_work",
    ),
    (
        "scripts/169-research-lab-source-add-post-accept-leg1.sql",
        "research_lab_source_add_configure_probe_v2",
    ),
    (
        "scripts/175-research-lab-source-add-provenance-leg1.sql",
        "research_lab_source_add_configure_probe_v3",
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_requeue_provenance",
    ),
    (
        "scripts/172-research-lab-source-add-claim-control.sql",
        "research_lab_source_add_set_paused",
    ),
    (
        "scripts/169-research-lab-source-add-post-accept-leg1.sql",
        "research_lab_source_add_reserve_leg1_slot_v2",
    ),
    (
        "scripts/173-research-lab-source-add-leg1-release-policy.sql",
        "research_lab_source_add_reserve_leg1_slot_v3",
    ),
    (
        "scripts/175-research-lab-source-add-provenance-leg1.sql",
        "research_lab_source_add_reserve_leg1_slot_v4",
    ),
    (
        "scripts/169-research-lab-source-add-post-accept-leg1.sql",
        "research_lab_source_add_finalize_leg1_v2",
    ),
    (
        "scripts/173-research-lab-source-add-leg1-release-policy.sql",
        "research_lab_source_add_finalize_leg1_v3",
    ),
    (
        "scripts/175-research-lab-source-add-provenance-leg1.sql",
        "research_lab_source_add_finalize_leg1_v4",
    ),
    (
        "scripts/175-research-lab-source-add-provenance-leg1.sql",
        "research_lab_source_add_enqueue_leg1_after_provenance_v1",
    ),
    (
        "scripts/175-research-lab-source-add-provenance-leg1.sql",
        "research_lab_source_add_reconcile_provenance_leg1_v1",
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_enqueue_provision_smoke",
    ),
    (
        "scripts/175-research-lab-source-add-provenance-leg1.sql",
        "research_lab_source_add_enqueue_provision_smoke_v2",
    ),
    (
        "scripts/169-research-lab-source-add-post-accept-leg1.sql",
        "research_lab_source_add_finalize_provision_v2",
    ),
    (
        "scripts/175-research-lab-source-add-provenance-leg1.sql",
        "research_lab_source_add_finalize_provision_v3",
    ),
    (
        "scripts/169-research-lab-source-add-post-accept-leg1.sql",
        "research_lab_source_add_reject_current_builtin_v2",
    ),
    (
        "scripts/175-research-lab-source-add-provenance-leg1.sql",
        "research_lab_source_add_reject_current_builtin_v3",
    ),
    (
        "scripts/145-research-lab-source-add-admission-control.sql",
        "research_lab_source_add_admission_control_contract_v1",
    ),
    (
        "scripts/169-research-lab-source-add-post-accept-leg1.sql",
        "research_lab_source_add_finalize_provision_smoke_v2",
    ),
    (
        "scripts/175-research-lab-source-add-provenance-leg1.sql",
        "research_lab_source_add_finalize_provision_smoke_v3",
    ),
    (
        "scripts/169-research-lab-source-add-post-accept-leg1.sql",
        "research_lab_source_add_post_accept_leg1_contract_v1",
    ),
    (
        "scripts/173-research-lab-source-add-leg1-release-policy.sql",
        "research_lab_source_add_post_accept_leg1_contract_v2",
    ),
    (
        "scripts/175-research-lab-source-add-provenance-leg1.sql",
        "research_lab_source_add_post_accept_leg1_contract_v3",
    ),
    (
        "scripts/176-research-lab-source-add-provenance-origin-repair.sql",
        "research_lab_source_add_post_accept_leg1_contract_v4",
    ),
    (
        "scripts/178-research-lab-source-add-miner-status.sql",
        "research_lab_source_add_miner_status_page_v1",
    ),
    (
        "scripts/178-research-lab-source-add-miner-status.sql",
        "research_lab_source_add_miner_status_contract_v1",
    ),
    (
        "scripts/170-research-lab-source-add-provider-origin-uniqueness.sql",
        "research_lab_source_add_admit_v2",
    ),
    (
        "scripts/170-research-lab-source-add-provider-origin-uniqueness.sql",
        "research_lab_source_add_requeue_provenance_v2",
    ),
    (
        "scripts/170-research-lab-source-add-provider-origin-uniqueness.sql",
        "research_lab_source_add_provider_origin_contract_v1",
    ),
    (
        "scripts/171-research-lab-source-add-duplicate-privacy.sql",
        "research_lab_source_add_admit_v3",
    ),
    (
        "scripts/171-research-lab-source-add-duplicate-privacy.sql",
        "research_lab_source_add_duplicate_privacy_contract_v1",
    ),
    (
        "scripts/172-research-lab-source-add-claim-control.sql",
        "research_lab_source_add_acquire_restart_guard_v1",
    ),
    (
        "scripts/172-research-lab-source-add-claim-control.sql",
        "research_lab_source_add_restart_quiescence_v1",
    ),
    (
        "scripts/172-research-lab-source-add-claim-control.sql",
        "research_lab_source_add_restart_guard_state_v1",
    ),
    (
        "scripts/172-research-lab-source-add-claim-control.sql",
        "research_lab_source_add_release_restart_guard_v1",
    ),
    (
        "scripts/172-research-lab-source-add-claim-control.sql",
        "research_lab_source_add_claim_control_contract_v1",
    ),
    (
        "scripts/174-research-lab-source-add-restart-state-restore.sql",
        "research_lab_source_add_acquire_restart_guard_v2",
    ),
    (
        "scripts/174-research-lab-source-add-restart-state-restore.sql",
        "research_lab_source_add_restart_guard_state_v2",
    ),
    (
        "scripts/174-research-lab-source-add-restart-state-restore.sql",
        "research_lab_source_add_release_restart_guard_v2",
    ),
    (
        "scripts/174-research-lab-source-add-restart-state-restore.sql",
        "research_lab_source_add_claim_control_contract_v2",
    ),
)


class SupabaseSchemaPreflightV2Error(RuntimeError):
    """The selected V2 release cannot use the live PostgREST schema."""


POSTGRES_IDENTIFIER_MAX_BYTES = 63


def _source_add_leg1_release_environment_policy_v1(
    parent_environment: Mapping[str, str],
) -> Dict[str, Any]:
    expected = {
        "RESEARCH_LAB_SOURCE_ADD_LEG1_ALPHA_PERCENT": Decimal("0.2"),
        "RESEARCH_LAB_SOURCE_ADD_LEG2_ALPHA_PERCENT": Decimal("0.0"),
        "RESEARCH_LAB_REWARD_EPOCHS": Decimal("20"),
        "RESEARCH_LAB_SOURCE_ADD_LEG1_MAX_PER_UTC_DAY": Decimal("50"),
    }
    observed: dict[str, Decimal] = {}
    for name, default in expected.items():
        raw = str(parent_environment.get(name, str(default))).strip()
        try:
            value = Decimal(raw)
        except (InvalidOperation, ValueError) as exc:
            raise SupabaseSchemaPreflightV2Error(
                "SOURCE_ADD Leg 1 release environment is invalid"
            ) from exc
        if not value.is_finite() or value != default:
            raise SupabaseSchemaPreflightV2Error(
                "SOURCE_ADD Leg 1 release environment differs"
            )
        observed[name] = value
    return {
        "schema_version": "leadpoet.source_add_leg1_release_policy.v1",
        "leg1_alpha_percent": float(
            observed["RESEARCH_LAB_SOURCE_ADD_LEG1_ALPHA_PERCENT"]
        ),
        "leg2_alpha_percent": float(
            observed["RESEARCH_LAB_SOURCE_ADD_LEG2_ALPHA_PERCENT"]
        ),
        "reward_epochs": int(observed["RESEARCH_LAB_REWARD_EPOCHS"]),
        "daily_cap": int(
            observed["RESEARCH_LAB_SOURCE_ADD_LEG1_MAX_PER_UTC_DAY"]
        ),
    }


def _verify_compact_weight_settlement_contract_v1(
    *,
    headers: Mapping[str, str],
    supabase_url: str,
    opener: Any,
    timeout_seconds: float,
) -> Dict[str, Any]:
    request = Request(
        (
            f"{supabase_url}/rest/v1/rpc/"
            "research_lab_compact_weight_settlement_contract_v1"
        ),
        data=b"{}",
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            status = int(response.getcode())
            encoded = response.read()
    except HTTPError as exc:
        raise SupabaseSchemaPreflightV2Error(
            "compact weight settlement schema contract is unavailable; apply "
            "scripts/149-research-lab-compact-weight-settlement-authority.sql "
            f"before restart (HTTP {exc.code})"
        ) from exc
    except Exception as exc:
        raise SupabaseSchemaPreflightV2Error(
            "compact weight settlement schema contract probe failed"
        ) from exc
    if status < 200 or status >= 300:
        raise SupabaseSchemaPreflightV2Error(
            "compact weight settlement schema contract is unavailable; apply "
            "scripts/149-research-lab-compact-weight-settlement-authority.sql "
            f"before restart (HTTP {status})"
        )
    try:
        contract = json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise SupabaseSchemaPreflightV2Error(
            "compact weight settlement schema contract response is invalid"
        ) from exc
    expected = {
        "schema_version": (
            "leadpoet.research_lab_compact_weight_settlement_contract.v1"
        ),
        "max_authority_bytes": 8_388_608,
        "size_constraint_valid": True,
        "append_only_trigger_enabled": True,
        "identity_unique_constraint_enabled": True,
        "row_level_security_enabled": True,
        "finalized_stage_supported": True,
    }
    if contract != expected:
        raise SupabaseSchemaPreflightV2Error(
            "compact weight settlement schema contract differs"
        )
    return dict(contract)


SOURCE_ADD_DUPLICATE_PRIVACY_FUNCTION_AUTHORITY_SHA256 = (
    "sha256:26bf34c94725b855f81c2e48b6afbd72d68db36a4aeffb5642494a5da32233e0"
)

SOURCE_ADD_MINER_STATUS_VIEW_AUTHORITY_SHA256 = (
    "sha256:8096dcc13409b33b56ad70f9606c9fe8ac7c644583b02b9c70f97322dfe86e26"
)
SOURCE_ADD_MINER_STATUS_PAGE_AUTHORITY_SHA256 = (
    "sha256:fefb9294135f34d9e0f329288f9ee11c42b54e36eaa4941d92e20b69e1a9d2e1"
)
SOURCE_ADD_MINER_STATUS_CONTRACT_AUTHORITY_SHA256 = (
    "sha256:b2d1ba1bf1062a911dc4ab3d6619d93b5cf282d4daa3896c553e99e0520b2c11"
)


def _verify_source_add_miner_status_contract_v1(
    *,
    headers: Mapping[str, str],
    supabase_url: str,
    opener: Any,
    timeout_seconds: float,
) -> Dict[str, Any]:
    request = Request(
        (
            f"{supabase_url}/rest/v1/rpc/"
            "research_lab_source_add_miner_status_contract_v1"
        ),
        data=b"{}",
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            status = int(response.getcode())
            encoded = response.read()
    except HTTPError as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD miner-status privacy contract is unavailable; apply "
            "scripts/178-research-lab-source-add-miner-status.sql before "
            f"restart (HTTP {exc.code})"
        ) from exc
    except Exception as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD miner-status privacy contract probe failed"
        ) from exc
    if status < 200 or status >= 300:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD miner-status privacy contract is unavailable; apply "
            "scripts/178-research-lab-source-add-miner-status.sql before "
            f"restart (HTTP {status})"
        )
    try:
        contract = json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD miner-status privacy contract response is invalid"
        ) from exc
    expected = {
        "schema_version": "leadpoet.source_add_miner_status_contract.v1",
        "view_name": "research_lab_source_add_miner_status_v1",
        "page_rpc": "research_lab_source_add_miner_status_page_v1",
        "page_signature": "text,text,integer",
        "view_columns": [
            "schema_version",
            "submission_id",
            "miner_hotkey",
            "source_name",
            "submitted_at",
            "updated_at",
            "decision_status",
            "decision_reason_code",
            "decision_reason",
            "reward_status",
            "alpha_percent",
            "reward_epochs",
            "start_epoch",
            "end_epoch",
        ],
        "view_security_invoker": True,
        "view_security_barrier": True,
        "page_security_invoker": True,
        "page_stable": True,
        "view_authority_sha256": (
            SOURCE_ADD_MINER_STATUS_VIEW_AUTHORITY_SHA256
        ),
        "page_authority_sha256": (
            SOURCE_ADD_MINER_STATUS_PAGE_AUTHORITY_SHA256
        ),
        "contract_authority_sha256": (
            SOURCE_ADD_MINER_STATUS_CONTRACT_AUTHORITY_SHA256
        ),
        "permissions": {
            "view_service_role_select": True,
            "view_anon_select": False,
            "view_authenticated_select": False,
            "view_public_select": False,
            "page_service_role_callable": True,
            "page_anon_callable": False,
            "page_authenticated_callable": False,
            "page_public_callable": False,
            "contract_service_role_callable": True,
            "contract_anon_callable": False,
            "contract_authenticated_callable": False,
        },
    }
    if contract != expected:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD miner-status privacy contract differs"
        )
    return dict(contract)


def _verify_source_add_duplicate_privacy_contract_v1(
    *,
    headers: Mapping[str, str],
    supabase_url: str,
    opener: Any,
    timeout_seconds: float,
) -> Dict[str, Any]:
    request = Request(
        (
            f"{supabase_url}/rest/v1/rpc/"
            "research_lab_source_add_duplicate_privacy_contract_v1"
        ),
        data=b"{}",
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            status = int(response.getcode())
            encoded = response.read()
    except HTTPError as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD duplicate-privacy contract is unavailable; apply "
            "scripts/171-research-lab-source-add-duplicate-privacy.sql "
            f"before restart (HTTP {exc.code})"
        ) from exc
    except Exception as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD duplicate-privacy contract probe failed"
        ) from exc
    if status < 200 or status >= 300:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD duplicate-privacy contract is unavailable; apply "
            "scripts/171-research-lab-source-add-duplicate-privacy.sql "
            f"before restart (HTTP {status})"
        )
    try:
        contract = json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD duplicate-privacy contract response is invalid"
        ) from exc
    expected = {
        "schema_version": "leadpoet.source_add_duplicate_privacy_contract.v1",
        "admission_rpc": "research_lab_source_add_admit_v3",
        "admission_signature": (
            "jsonb,text,text,text,text,text,integer,integer,integer,integer"
        ),
        "compatibility_rpc": "research_lab_source_add_admit_v2",
        "compatibility_signature": (
            "jsonb,text,text,text,text,text,integer,integer,integer"
        ),
        "compatibility_cooldown_seconds": 20,
        "cooldown_parameter_min_seconds": 1,
        "cooldown_parameter_max_seconds": 3600,
        "cooldown_clock": "clock_timestamp_after_advisory_locks",
        "cooldown_source": "durable_miner_provenance_work",
        "duplicate_precedes_cooldown": True,
        "lock_order": [
            "provider_origin_or_identity",
            "hotkey",
            "submission_or_work",
        ],
        "function_authority_sha256": (
            SOURCE_ADD_DUPLICATE_PRIVACY_FUNCTION_AUTHORITY_SHA256
        ),
        "functions": {
            "admit_v1": True,
            "admit_v2_compatibility": True,
            "admit_v3": True,
            "provider_origin_hash_v1": True,
            "provider_origin_host_v1": True,
        },
        "permissions": {
            "service_role_exists": True,
            "v3_service_role_callable": True,
            "v2_service_role_callable": True,
            "contract_service_role_callable": True,
            "anon_callable": False,
            "authenticated_callable": False,
        },
    }
    if contract != expected:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD duplicate-privacy contract differs"
        )
    return dict(contract)


SOURCE_ADD_CLAIM_CONTROL_FUNCTION_AUTHORITY_SHA256 = (
    "sha256:890a1e42b6dd28eb1c8515c3b8c33d31a9974058fbd2c43393bb0880c0ca21e6"
)
SOURCE_ADD_CLAIM_CONTROL_ROLLBACK_V1_CONTRACT_SHA256 = (
    "sha256:b74dbb957ca2ed1741aed6503351c0934fad76614614a52669d5b7b03d0c011f"
)
SOURCE_ADD_CLAIM_CONTROL_V2_FUNCTION_AUTHORITY_SHA256 = (
    "sha256:1082a75d70849b072299929ff00999b5c78a69adc9c7b03e544640ed60b02ff8"
)


def _verify_source_add_claim_control_contract_v1(
    *,
    headers: Mapping[str, str],
    supabase_url: str,
    opener: Any,
    timeout_seconds: float,
) -> Dict[str, Any]:
    request = Request(
        (
            f"{supabase_url}/rest/v1/rpc/"
            "research_lab_source_add_claim_control_contract_v1"
        ),
        data=b"{}",
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            status = int(response.getcode())
            encoded = response.read()
    except HTTPError as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD claim-control contract is unavailable; apply "
            "scripts/172-research-lab-source-add-claim-control.sql "
            f"before restart (HTTP {exc.code})"
        ) from exc
    except Exception as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD claim-control contract probe failed"
        ) from exc
    if status < 200 or status >= 300:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD claim-control contract is unavailable; apply "
            "scripts/172-research-lab-source-add-claim-control.sql "
            f"before restart (HTTP {status})"
        )
    try:
        contract = json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD claim-control contract response is invalid"
        ) from exc
    expected = {
        "schema_version": "leadpoet.source_add_claim_control_contract.v1",
        "control_lock": "source-add-control",
        "pause_rpc": "research_lab_source_add_set_paused",
        "pause_signature": "boolean,text,text",
        "claim_rpc": "research_lab_source_add_claim_work",
        "claim_signature": "text,integer",
        "acquire_guard_rpc": (
            "research_lab_source_add_acquire_restart_guard_v1"
        ),
        "acquire_guard_signature": "text,text,bigint,integer,text",
        "guard_state_rpc": (
            "research_lab_source_add_restart_guard_state_v1"
        ),
        "guard_state_signature": "",
        "release_guard_rpc": (
            "research_lab_source_add_release_restart_guard_v1"
        ),
        "release_guard_signature": "text,text,bigint,text",
        "guard_state_result_fields": [
            "schema_version",
            "paused",
            "guard_active",
            "guard_commitment",
            "owner_commitment",
            "guard_generation",
            "owner_generation_commitment",
            "guard_expires_at",
        ],
        "acquire_guard_result_fields": [
            "schema_version",
            "paused",
            "guard_active",
            "guard_commitment",
            "owner_commitment",
            "guard_generation",
            "owner_generation_commitment",
            "guard_expires_at",
        ],
        "release_guard_result_fields": [
            "schema_version",
            "released",
            "paused",
            "guard_active",
            "guard_generation",
            "owner_generation_commitment",
        ],
        "guard_id_format": "^source_add_restart_guard:[0-9a-f]{64}$",
        "guard_commitment": "sha256_utf8_guard_id",
        "owner_id_format": "^source_add_restart_owner:[0-9a-f]{64}$",
        "owner_commitment": "sha256_utf8_owner_id",
        "owner_generation_commitment": (
            "sha256_utf8_owner_commitment_colon_decimal_generation"
        ),
        "guard_lease_min_seconds": 60,
        "guard_lease_max_seconds": 14400,
        "active_guard_replay_extends_lease": True,
        "acquire_compare_and_swap": "expected_generation",
        "different_owner_takeover_increments_generation": True,
        "expired_reacquire_increments_generation": True,
        "generation_retained_after_release": True,
        "resume_requires_guard_clear": True,
        "expired_guard_recovery": (
            "explicit_reacquire_then_exact_release"
        ),
        "release_keeps_paused": True,
        "restart_quiescence_rpc": (
            "research_lab_source_add_restart_quiescence_v1"
        ),
        "restart_quiescence_signature": "text,text,bigint",
        "restart_quiescence_schema_version": (
            "leadpoet.source_add_restart_quiescence.v1"
        ),
        "restart_quiescence_result_fields": [
            "schema_version",
            "paused",
            "guard_active",
            "guard_matches",
            "owner_matches",
            "generation_matches",
            "guard_commitment",
            "owner_commitment",
            "guard_generation",
            "owner_generation_commitment",
            "guard_expires_at",
            "leased_work_count",
            "quiescent",
        ],
        "lock_before_paused_read": True,
        "leased_scope": "all_leased_regardless_of_expiry",
        "migration_requires_paused": True,
        "migration_requires_zero_leased": True,
        "function_authority_sha256": (
            SOURCE_ADD_CLAIM_CONTROL_FUNCTION_AUTHORITY_SHA256
        ),
        "functions": {
            "admission_guard": True,
            "acquire_restart_guard_v1": True,
            "claim_work": True,
            "pause": True,
            "release_restart_guard_v1": True,
            "restart_guard_state_v1": True,
            "restart_quiescence_v1": True,
        },
        "permissions": {
            "service_role_exists": True,
            "acquire_guard_service_role_callable": True,
            "claim_service_role_callable": True,
            "pause_service_role_callable": True,
            "quiescence_service_role_callable": True,
            "release_guard_service_role_callable": True,
            "guard_state_service_role_callable": True,
            "contract_service_role_callable": True,
            "anon_callable": False,
            "authenticated_callable": False,
        },
    }
    if contract != expected:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD claim-control contract differs"
        )
    return dict(contract)


def _verify_source_add_claim_control_contract_v2(
    *,
    headers: Mapping[str, str],
    supabase_url: str,
    opener: Any,
    timeout_seconds: float,
) -> Dict[str, Any]:
    request = Request(
        (
            f"{supabase_url}/rest/v1/rpc/"
            "research_lab_source_add_claim_control_contract_v2"
        ),
        data=b"{}",
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            status = int(response.getcode())
            encoded = response.read()
    except HTTPError as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD restart-state contract is unavailable; apply "
            "scripts/174-research-lab-source-add-restart-state-restore.sql "
            f"before restart (HTTP {exc.code})"
        ) from exc
    except Exception as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD restart-state contract probe failed"
        ) from exc
    if status < 200 or status >= 300:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD restart-state contract is unavailable; apply "
            "scripts/174-research-lab-source-add-restart-state-restore.sql "
            f"before restart (HTTP {status})"
        )
    try:
        contract = json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD restart-state contract response is invalid"
        ) from exc
    expected = {
        "schema_version": "leadpoet.source_add_claim_control_contract.v2",
        "control_lock": "source-add-control",
        "pause_rpc": "research_lab_source_add_set_paused",
        "pause_signature": "boolean,text,text",
        "claim_rpc": "research_lab_source_add_claim_work",
        "claim_signature": "text,integer",
        "acquire_guard_rpc": (
            "research_lab_source_add_acquire_restart_guard_v2"
        ),
        "acquire_guard_signature": "text,text,bigint,integer,text",
        "guard_state_rpc": (
            "research_lab_source_add_restart_guard_state_v2"
        ),
        "guard_state_signature": "",
        "release_guard_rpc": (
            "research_lab_source_add_release_restart_guard_v2"
        ),
        "release_guard_signature": "text,text,bigint,text",
        "restart_quiescence_rpc": (
            "research_lab_source_add_restart_quiescence_v1"
        ),
        "restart_quiescence_signature": "text,text,bigint",
        "guard_state_result_fields": [
            "schema_version",
            "paused",
            "guard_active",
            "guard_commitment",
            "owner_commitment",
            "guard_generation",
            "owner_generation_commitment",
            "guard_expires_at",
            "restore_paused",
        ],
        "acquire_guard_result_fields": [
            "schema_version",
            "paused",
            "guard_active",
            "guard_commitment",
            "owner_commitment",
            "guard_generation",
            "owner_generation_commitment",
            "guard_expires_at",
            "restore_paused",
        ],
        "release_guard_result_fields": [
            "schema_version",
            "released",
            "paused",
            "guard_active",
            "guard_generation",
            "owner_generation_commitment",
            "restored_pre_restart_state",
        ],
        "restore_state_column": "restart_guard_restore_paused",
        "acquire_captures_pre_restart_paused": True,
        "renewal_preserves_restore_state": True,
        "expired_takeover_preserves_restore_state": True,
        "operator_pause_wins": True,
        "release_restores_pre_restart_state": True,
        "failed_restart_keeps_paused": True,
        "rollback_v1_contract_schema_version": (
            "leadpoet.source_add_claim_control_contract.v1"
        ),
        "rollback_v1_contract_sha256": (
            SOURCE_ADD_CLAIM_CONTROL_ROLLBACK_V1_CONTRACT_SHA256
        ),
        "migration_requires_paused": True,
        "migration_requires_zero_leased": True,
        "migration_requires_guard_clear": True,
        "function_authority_sha256": (
            SOURCE_ADD_CLAIM_CONTROL_V2_FUNCTION_AUTHORITY_SHA256
        ),
        "functions": {
            "admission_guard": True,
            "acquire_restart_guard_v1": True,
            "acquire_restart_guard_v2": True,
            "claim_work": True,
            "pause": True,
            "release_restart_guard_v1": True,
            "release_restart_guard_v2": True,
            "restart_guard_state_v1": True,
            "restart_guard_state_v2": True,
            "restart_quiescence_v1": True,
            "restore_trigger_v2": True,
        },
        "permissions": {
            "service_role_exists": True,
            "service_role_callable": True,
            "anon_callable": False,
            "authenticated_callable": False,
        },
    }
    if contract != expected:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD restart-state contract differs"
        )
    return dict(contract)


def _verify_source_add_provider_origin_contract_v1(
    *,
    headers: Mapping[str, str],
    supabase_url: str,
    opener: Any,
    timeout_seconds: float,
) -> Dict[str, Any]:
    request = Request(
        (
            f"{supabase_url}/rest/v1/rpc/"
            "research_lab_source_add_provider_origin_contract_v1"
        ),
        data=b"{}",
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            status = int(response.getcode())
            encoded = response.read()
    except HTTPError as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD provider-origin contract is unavailable; apply "
            "scripts/170-research-lab-source-add-provider-origin-uniqueness.sql "
            f"before restart (HTTP {exc.code})"
        ) from exc
    except Exception as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD provider-origin contract probe failed"
        ) from exc
    if status < 200 or status >= 300:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD provider-origin contract is unavailable; apply "
            "scripts/170-research-lab-source-add-provider-origin-uniqueness.sql "
            f"before restart (HTTP {status})"
        )
    try:
        contract = json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD provider-origin contract response is invalid"
        ) from exc
    expected_keys = {
        "schema_version",
        "identity_version",
        "identity_scope",
        "admission_rpc",
        "recheck_rpc",
        "owner_count",
        "reserved_count",
        "coverage_complete",
        "collision_free",
        "submission_trigger_enabled",
        "catalog_trigger_enabled",
        "provision_trigger_enabled",
        "terminal_release_trigger_enabled",
        "append_only_trigger_enabled",
        "row_level_security_enabled",
        "service_role_policy_enabled",
    }
    if not isinstance(contract, Mapping) or set(contract) != expected_keys:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD provider-origin contract response is invalid"
        )
    owner_count = contract.get("owner_count")
    reserved_count = contract.get("reserved_count")
    required_true = expected_keys - {
        "schema_version",
        "identity_version",
        "identity_scope",
        "admission_rpc",
        "recheck_rpc",
        "owner_count",
        "reserved_count",
    }
    if (
        contract.get("schema_version")
        != "leadpoet.source_add_provider_origin_contract.v1"
        or contract.get("identity_version") != "v1"
        or contract.get("identity_scope") != "normalized_exact_host"
        or contract.get("admission_rpc") != "research_lab_source_add_admit_v2"
        or contract.get("recheck_rpc")
        != "research_lab_source_add_requeue_provenance_v2"
        or type(owner_count) is not int
        or type(reserved_count) is not int
        or owner_count < 0
        or reserved_count < 0
        or owner_count != reserved_count
        or any(contract.get(field) is not True for field in required_true)
    ):
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD provider-origin contract differs"
        )
    # Keep the provider-origin verification as the one protected preflight
    # seam while binding the v3 route's exact implementation, compatibility
    # wrapper, policy, signatures, and ACLs in migration 170.
    _verify_source_add_duplicate_privacy_contract_v1(
        headers=headers,
        supabase_url=supabase_url,
        opener=opener,
        timeout_seconds=timeout_seconds,
    )
    return dict(contract)


SOURCE_ADD_POST_ACCEPT_LEG1_ROLLBACK_V1_FUNCTION_AUTHORITY_SHA256 = (
    "sha256:80592287bb9dfed4bdc86b056f53ba71da2fb62d7ee82074c94a878c550eb83b"
)

SOURCE_ADD_POST_ACCEPT_LEG1_FUNCTION_AUTHORITY_SHA256 = (
    "sha256:6c09aa3c6b82b3fe666c6739c4f71a51ea8d6445e3e5a52ab08a4e2f8fa8d9ec"
)

SOURCE_ADD_PROVENANCE_LEG1_FUNCTION_AUTHORITY_SHA256 = (
    "sha256:fe7df9f9336217f3e738f420fae0d9720959042080df431c1bcb2d4baa8ee954"
)

SOURCE_ADD_PROVENANCE_LEG1_TRIGGER_AUTHORITY_SHA256 = (
    "sha256:208de2069d2b44826fe466de01a2d1a91f4c762869227b39bdba969c8586be16"
)

SOURCE_ADD_PROVENANCE_LEG1_VIEW_AUTHORITY_SHA256 = (
    "sha256:19f67626677803ff84f92196adeb9731d415c643247c603e41512a88c3f6291b"
)

SOURCE_ADD_PROVENANCE_ORIGIN_VIEW_AUTHORITY_SHA256 = (
    "sha256:36380661634fee55bbdb69631d81ee0872f96de9d1373a253d1b02db242f037a"
)

SOURCE_ADD_PROVENANCE_ORIGIN_REPAIR_FUNCTION_AUTHORITY_SHA256 = (
    "sha256:700345ac44ebad77f4568e6c80458238129fd4af6c9ada66d7558d1bca5c9491"
)


def _verify_source_add_post_accept_leg1_contract_v2(
    *,
    headers: Mapping[str, str],
    supabase_url: str,
    opener: Any,
    timeout_seconds: float,
) -> Dict[str, Any]:
    request = Request(
        (
            f"{supabase_url}/rest/v1/rpc/"
            "research_lab_source_add_post_accept_leg1_contract_v2"
        ),
        data=b"{}",
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            status = int(response.getcode())
            encoded = response.read()
    except HTTPError as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD post-accept Leg 1 contract is unavailable; apply "
            "scripts/173-research-lab-source-add-leg1-release-policy.sql "
            f"before restart (HTTP {exc.code})"
        ) from exc
    except Exception as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD post-accept Leg 1 contract probe failed"
        ) from exc
    if status < 200 or status >= 300:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD post-accept Leg 1 contract is unavailable; apply "
            "scripts/173-research-lab-source-add-leg1-release-policy.sql "
            f"before restart (HTTP {status})"
        )
    try:
        contract = json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD post-accept Leg 1 contract response is invalid"
        ) from exc
    expected = {
        "schema_version": "leadpoet.source_add_post_accept_leg1_contract.v2",
        "daily_cap": 50,
        "leg1_alpha_percent": 0.2,
        "leg1_reward_epochs": 20,
        "function_authority_sha256": (
            SOURCE_ADD_POST_ACCEPT_LEG1_FUNCTION_AUTHORITY_SHA256
        ),
        "functions": {
            "configure_probe_v2": True,
            "finalize_provision_v2": True,
            "reject_current_builtin_v2": True,
            "post_accept_contract_v1": True,
            "reserve_leg1_slot_v2": True,
            "finalize_leg1_v2": True,
            "reserve_leg1_slot_v3": True,
            "finalize_leg1_v3": True,
            "finalize_provision_smoke_v2": True,
        },
        "triggers": {
            "acceptance": True,
            "eligible": True,
            "leg1_work": True,
            "leg1_slot": True,
            "leg1_obligation": True,
            "leg1_initial_event": True,
        },
        "permissions": {
            "service_role_exists": True,
            "candidate_callable": True,
            "rollback_v2_callable": True,
            "legacy_not_callable": True,
        },
    }
    if contract != expected:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD post-accept Leg 1 contract differs"
        )
    return dict(contract)


def _verify_source_add_post_accept_leg1_contract_v3(
    *,
    headers: Mapping[str, str],
    supabase_url: str,
    opener: Any,
    timeout_seconds: float,
) -> Dict[str, Any]:
    request = Request(
        (
            f"{supabase_url}/rest/v1/rpc/"
            "research_lab_source_add_post_accept_leg1_contract_v3"
        ),
        data=b"{}",
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            status = int(response.getcode())
            encoded = response.read()
    except HTTPError as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD automatic provenance Leg 1 contract is unavailable; "
            "apply scripts/175-research-lab-source-add-provenance-leg1.sql "
            f"before restart (HTTP {exc.code})"
        ) from exc
    except Exception as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD automatic provenance Leg 1 contract probe failed"
        ) from exc
    if status < 200 or status >= 300:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD automatic provenance Leg 1 contract is unavailable; "
            "apply scripts/175-research-lab-source-add-provenance-leg1.sql "
            f"before restart (HTTP {status})"
        )
    try:
        contract = json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD automatic provenance Leg 1 contract response is "
            "invalid"
        ) from exc
    expected = {
        "schema_version": "leadpoet.source_add_post_accept_leg1_contract.v3",
        "daily_cap": 50,
        "leg1_alpha_percent": 0.2,
        "leg1_reward_epochs": 20,
        "approval_boundary": "provenance_precheck_passed",
        "backfill_policy": "all_exact_attested_provenance",
        "public_trigger_fields": [
            "precheck_status",
            "provenance_artifact_hash",
            "provenance_precheck_passed",
            "provenance_receipt_hash",
            "provenance_result_hash",
            "submission_id",
        ],
        "authority_view": (
            "research_lab_source_add_provenance_leg1_authority_v1"
        ),
        "function_authority_sha256": (
            SOURCE_ADD_PROVENANCE_LEG1_FUNCTION_AUTHORITY_SHA256
        ),
        "trigger_authority_sha256": (
            SOURCE_ADD_PROVENANCE_LEG1_TRIGGER_AUTHORITY_SHA256
        ),
        "view_authority_sha256": (
            SOURCE_ADD_PROVENANCE_LEG1_VIEW_AUTHORITY_SHA256
        ),
        "functions": {
            "configure_probe_v3": True,
            "enqueue_leg1_after_provenance_v1": True,
            "enqueue_provision_smoke_v2": True,
            "finalize_leg1_v4": True,
            "finalize_provision_smoke_v3": True,
            "finalize_provision_v3": True,
            "reject_current_builtin_v3": True,
            "reconcile_provenance_leg1_v1": True,
            "reserve_leg1_slot_v4": True,
        },
        "triggers": {
            "automatic_enqueue": True,
            "eligible_v2": True,
            "eligible_v3": True,
            "leg1_initial_event_v3": True,
            "leg1_obligation_v3": True,
            "leg1_slot_v3": True,
            "leg1_work_v3": True,
        },
        "columns": {
            "intent_approval_kind": True,
            "intent_provenance_artifact_hash": True,
            "intent_provenance_receipt_hash": True,
            "slot_approval_kind": True,
        },
        "permissions": {
            "service_role_exists": True,
            "candidate_callable": True,
            "internal_not_callable": True,
            "rollback_v2_callable": True,
        },
    }
    if contract != expected:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD automatic provenance Leg 1 contract differs"
        )
    return dict(contract)


def _verify_source_add_post_accept_leg1_contract_v4(
    *,
    headers: Mapping[str, str],
    supabase_url: str,
    opener: Any,
    timeout_seconds: float,
) -> Dict[str, Any]:
    request = Request(
        (
            f"{supabase_url}/rest/v1/rpc/"
            "research_lab_source_add_post_accept_leg1_contract_v4"
        ),
        data=b"{}",
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            status = int(response.getcode())
            encoded = response.read()
    except HTTPError as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD provenance-origin Leg 1 contract is unavailable; "
            "apply scripts/176-research-lab-source-add-provenance-origin-"
            f"repair.sql before restart (HTTP {exc.code})"
        ) from exc
    except Exception as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD provenance-origin Leg 1 contract probe failed"
        ) from exc
    if status < 200 or status >= 300:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD provenance-origin Leg 1 contract is unavailable; "
            "apply scripts/176-research-lab-source-add-provenance-origin-"
            f"repair.sql before restart (HTTP {status})"
        )
    try:
        contract = json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD provenance-origin Leg 1 contract response is invalid"
        ) from exc
    expected = {
        "schema_version": "leadpoet.source_add_post_accept_leg1_contract.v4",
        "required_migration": (
            "scripts/176-research-lab-source-add-provenance-origin-repair.sql"
        ),
        "daily_cap": 50,
        "leg1_alpha_percent": 0.2,
        "leg1_reward_epochs": 20,
        "approval_boundary": "provenance_precheck_passed",
        "backfill_policy": (
            "earliest_exact_attested_provenance_per_provider_origin"
        ),
        "provider_origin_scope": "normalized_exact_host",
        "provider_origin_winner_order": [
            "provenance_created_at",
            "submission_id",
        ],
        "cancelled_intents_are_authority": False,
        "public_trigger_fields": [
            "precheck_status",
            "provenance_artifact_hash",
            "provenance_precheck_passed",
            "provenance_receipt_hash",
            "provenance_result_hash",
            "submission_id",
        ],
        "authority_view": (
            "research_lab_source_add_provenance_leg1_authority_v1"
        ),
        "function_authority_sha256": (
            SOURCE_ADD_PROVENANCE_LEG1_FUNCTION_AUTHORITY_SHA256
        ),
        "trigger_authority_sha256": (
            SOURCE_ADD_PROVENANCE_LEG1_TRIGGER_AUTHORITY_SHA256
        ),
        "view_authority_sha256": (
            SOURCE_ADD_PROVENANCE_ORIGIN_VIEW_AUTHORITY_SHA256
        ),
        "repair_function_authority_sha256": (
            SOURCE_ADD_PROVENANCE_ORIGIN_REPAIR_FUNCTION_AUTHORITY_SHA256
        ),
        "functions": {
            "configure_probe_v3": True,
            "enqueue_leg1_after_provenance_v1": True,
            "enqueue_provision_smoke_v2": True,
            "finalize_leg1_v4": True,
            "finalize_provision_smoke_v3": True,
            "finalize_provision_v3": True,
            "reject_current_builtin_v3": True,
            "reconcile_provenance_leg1_v1": True,
            "reserve_leg1_slot_v4": True,
        },
        "triggers": {
            "automatic_enqueue": True,
            "eligible_v2": True,
            "eligible_v3": True,
            "leg1_initial_event_v3": True,
            "leg1_obligation_v3": True,
            "leg1_slot_v3": True,
            "leg1_work_v3": True,
        },
        "columns": {
            "intent_approval_kind": True,
            "intent_provenance_artifact_hash": True,
            "intent_provenance_receipt_hash": True,
            "slot_approval_kind": True,
        },
        "permissions": {
            "service_role_exists": True,
            "candidate_callable": True,
            "internal_not_callable": True,
            "rollback_v2_callable": True,
        },
    }
    if contract != expected:
        raise SupabaseSchemaPreflightV2Error(
            "SOURCE_ADD provenance-origin Leg 1 contract differs"
        )
    return dict(contract)


def _verify_chain_realized_activation_v1(
    parent_environment: Mapping[str, str],
    *,
    headers: Mapping[str, str],
    supabase_url: str,
    opener: Any,
    timeout_seconds: float,
    activation_authority: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    try:
        netuid = int(parent_environment.get("BITTENSOR_NETUID") or 71)
    except (TypeError, ValueError) as exc:
        raise SupabaseSchemaPreflightV2Error(
            "BITTENSOR_NETUID is invalid for chain-realized settlement"
        ) from exc
    if netuid <= 0:
        raise SupabaseSchemaPreflightV2Error(
            "BITTENSOR_NETUID is invalid for chain-realized settlement"
        )
    columns = (
        "netuid,schema_version,first_epoch_id,source_bundle_hash,"
        "source_bundle_epoch_id,source_finalized_block"
    )
    if activation_authority is None:
        query = urlencode(
            {
                "select": columns,
                "netuid": f"eq.{netuid}",
                "limit": "2",
            }
        )
        request = Request(
            (
                f"{supabase_url}/rest/v1/"
                "research_lab_chain_realized_settlement_activation_v1"
                f"?{query}"
            ),
            headers=dict(headers),
        )
        try:
            with opener(request, timeout=timeout_seconds) as response:
                status = int(response.getcode())
                encoded = response.read()
        except HTTPError as exc:
            raise SupabaseSchemaPreflightV2Error(
                "chain-realized settlement activation is unavailable; apply "
                "scripts/126-research-lab-chain-realized-settlement.sql after "
                f"at least one finalized V2 bundle exists (HTTP {exc.code})"
            ) from exc
        except Exception as exc:
            raise SupabaseSchemaPreflightV2Error(
                "chain-realized settlement activation probe failed"
            ) from exc
        if status < 200 or status >= 300:
            raise SupabaseSchemaPreflightV2Error(
                "chain-realized settlement activation is unavailable; apply "
                "scripts/126-research-lab-chain-realized-settlement.sql after "
                f"at least one finalized V2 bundle exists (HTTP {status})"
            )
        try:
            rows = json.loads(encoded.decode("utf-8"))
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise SupabaseSchemaPreflightV2Error(
                "chain-realized settlement activation response is invalid"
            ) from exc
    else:
        rows = [activation_authority]
    if not isinstance(rows, list) or len(rows) != 1:
        raise SupabaseSchemaPreflightV2Error(
            "chain-realized settlement activation is missing or ambiguous; "
            "apply scripts/126-research-lab-chain-realized-settlement.sql "
            "after at least one finalized V2 bundle exists"
        )
    row = rows[0]
    if not isinstance(row, Mapping) or set(row) != set(columns.split(",")):
        raise SupabaseSchemaPreflightV2Error(
            "chain-realized settlement activation response is invalid"
        )
    try:
        row_netuid = int(row["netuid"])
        first_epoch = int(row["first_epoch_id"])
        source_epoch = int(row["source_bundle_epoch_id"])
        finalized_block = int(row["source_finalized_block"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SupabaseSchemaPreflightV2Error(
            "chain-realized settlement activation response is invalid"
        ) from exc
    if (
        row_netuid != netuid
        or row.get("schema_version")
        != "leadpoet.research_lab_chain_realized_settlement_activation.v1"
        or first_epoch < 0
        or source_epoch != first_epoch
        or finalized_block < 0
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(row.get("source_bundle_hash") or ""),
        )
        is None
    ):
        raise SupabaseSchemaPreflightV2Error(
            "chain-realized settlement activation response is invalid"
        )
    return {
        "netuid": netuid,
        "first_epoch_id": first_epoch,
        "source_bundle_hash": str(row["source_bundle_hash"]),
        "source_finalized_block": finalized_block,
    }


def verify_required_supabase_v2_schema(
    parent_environment: Mapping[str, str],
    *,
    opener: Any = urlopen,
    timeout_seconds: float = 10.0,
    chain_realized_activation_authority: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    required_rpcs = REQUIRED_SUPABASE_V2_RPCS
    source_add_leg1_release_policy = (
        _source_add_leg1_release_environment_policy_v1(parent_environment)
    )
    activation_source = (
        "postgrest"
        if chain_realized_activation_authority is None
        else "provided-authority"
    )
    for migration, function_name in required_rpcs:
        if len(function_name.encode("utf-8")) > POSTGRES_IDENTIFIER_MAX_BYTES:
            raise SupabaseSchemaPreflightV2Error(
                "required Supabase V2 RPC identifier exceeds PostgreSQL's "
                f"{POSTGRES_IDENTIFIER_MAX_BYTES}-byte limit for "
                f"{function_name}; correct {migration} before restart"
            )
    supabase_url = str(parent_environment.get("SUPABASE_URL") or "").rstrip("/")
    service_role_key = str(parent_environment.get("SUPABASE_SERVICE_ROLE_KEY") or "")
    if not supabase_url or not service_role_key:
        raise SupabaseSchemaPreflightV2Error(
            "prepared parent environment lacks Supabase V2 schema credentials"
        )
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {service_role_key}",
        "apikey": service_role_key,
    }
    migrations = set()
    for migration, table, columns in REQUIRED_SUPABASE_V2_SCHEMA:
        query = urlencode({"select": ",".join(columns), "limit": "0"})
        request = Request(
            f"{supabase_url}/rest/v1/{table}?{query}",
            headers=headers,
        )
        try:
            with opener(request, timeout=timeout_seconds) as response:
                status = int(response.getcode())
                response.read(1)
        except HTTPError as exc:
            raise SupabaseSchemaPreflightV2Error(
                "required Supabase V2 schema is unavailable for "
                f"{table}; apply {migration} before restart (HTTP {exc.code})"
            ) from exc
        except Exception as exc:
            raise SupabaseSchemaPreflightV2Error(
                f"Supabase V2 schema probe failed for {table}"
            ) from exc
        if status < 200 or status >= 300:
            raise SupabaseSchemaPreflightV2Error(
                "required Supabase V2 schema is unavailable for "
                f"{table}; apply {migration} before restart (HTTP {status})"
            )
        migrations.add(migration)
    activation = _verify_chain_realized_activation_v1(
        parent_environment,
        headers=headers,
        supabase_url=supabase_url,
        opener=opener,
        timeout_seconds=timeout_seconds,
        activation_authority=chain_realized_activation_authority,
    )
    migrations.add(
        "scripts/126-research-lab-chain-realized-settlement.sql"
    )
    # PostgREST returns 200 to OPTIONS even for a nonexistent /rpc path, so an
    # OPTIONS probe cannot prove function availability. The service-role OpenAPI
    # document lists only functions present in the active schema cache and
    # executable by that role; inspect it once without executing any RPC.
    schema_request = Request(
        f"{supabase_url}/rest/v1/",
        headers={**headers, "Accept": "application/openapi+json"},
    )
    try:
        with opener(schema_request, timeout=timeout_seconds) as response:
            status = int(response.getcode())
            encoded_schema = response.read()
    except HTTPError as exc:
        raise SupabaseSchemaPreflightV2Error(
            f"Supabase V2 RPC schema probe failed (HTTP {exc.code})"
        ) from exc
    except Exception as exc:
        raise SupabaseSchemaPreflightV2Error(
            "Supabase V2 RPC schema probe failed"
        ) from exc
    if status < 200 or status >= 300:
        raise SupabaseSchemaPreflightV2Error(
            f"Supabase V2 RPC schema probe failed (HTTP {status})"
        )
    try:
        schema_document = json.loads(encoded_schema.decode("utf-8"))
        schema_paths = schema_document["paths"]
        if not isinstance(schema_paths, Mapping):
            raise TypeError("OpenAPI paths must be an object")
    except (KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise SupabaseSchemaPreflightV2Error(
            "Supabase V2 RPC schema document is invalid"
        ) from exc
    for migration, function_name in required_rpcs:
        if f"/rpc/{function_name}" not in schema_paths:
            raise SupabaseSchemaPreflightV2Error(
                "required Supabase V2 RPC is unavailable for "
                f"{function_name}; apply {migration} before restart"
            )
        migrations.add(migration)
    compact_weight_settlement_contract = (
        _verify_compact_weight_settlement_contract_v1(
            headers=headers,
            supabase_url=supabase_url,
            opener=opener,
            timeout_seconds=timeout_seconds,
        )
    )
    source_add_provider_origin_contract = (
        _verify_source_add_provider_origin_contract_v1(
            headers=headers,
            supabase_url=supabase_url,
            opener=opener,
            timeout_seconds=timeout_seconds,
        )
    )
    source_add_post_accept_leg1_contract = (
        _verify_source_add_post_accept_leg1_contract_v4(
            headers=headers,
            supabase_url=supabase_url,
            opener=opener,
            timeout_seconds=timeout_seconds,
        )
    )
    source_add_claim_control_contract = (
        _verify_source_add_claim_control_contract_v2(
            headers=headers,
            supabase_url=supabase_url,
            opener=opener,
            timeout_seconds=timeout_seconds,
        )
    )
    source_add_miner_status_contract = (
        _verify_source_add_miner_status_contract_v1(
            headers=headers,
            supabase_url=supabase_url,
            opener=opener,
            timeout_seconds=timeout_seconds,
        )
    )
    return {
        "status": "ready",
        "probe_count": len(REQUIRED_SUPABASE_V2_SCHEMA)
        + len(required_rpcs)
        + 6,
        "table_probe_count": len(REQUIRED_SUPABASE_V2_SCHEMA),
        "rpc_probe_count": len(required_rpcs),
        "data_probe_count": 6,
        "schema_document_probe_count": 1,
        "chain_realized_settlement_activation_http_probe_count": (
            1 if activation_source == "postgrest" else 0
        ),
        "chain_realized_settlement_activation_source": activation_source,
        "chain_realized_settlement_activation": activation,
        "compact_weight_settlement_contract": (
            compact_weight_settlement_contract
        ),
        "source_add_provider_origin_contract": (
            source_add_provider_origin_contract
        ),
        "source_add_post_accept_leg1_contract": (
            source_add_post_accept_leg1_contract
        ),
        "source_add_claim_control_contract": (
            source_add_claim_control_contract
        ),
        "source_add_miner_status_contract": (
            source_add_miner_status_contract
        ),
        "source_add_leg1_release_policy": source_add_leg1_release_policy,
        "migration_files": sorted(migrations),
    }
