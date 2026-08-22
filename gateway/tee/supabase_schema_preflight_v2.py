"""Read-only PostgREST schema gate for the selected gateway V2 release."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Mapping
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from leadpoet_canonical.attested_v2 import ROLE_PURPOSES


REQUIRED_SUPABASE_V2_SCHEMA = (
    (
        "scripts/92-validator-sourcing-attested-v2.sql",
        "validator_sourcing_epoch_inputs_v2",
        ("epoch_id", "epoch_hash", "decision_root", "receipt_hash"),
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "research_lab_autoresearch_trees",
        ("tree_id", "run_id", "root_artifact_hash"),
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "research_lab_autoresearch_tree_nodes",
        ("tree_id", "node_id", "parent_node_id"),
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "research_lab_autoresearch_tree_events",
        (
            "tree_id",
            "seq",
            "event_type",
            "node_id",
            "previous_event_hash",
            "event_doc",
            "event_hash",
            "created_at",
        ),
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "research_lab_autoresearch_operation_settlements",
        ("logical_operation_id", "seq", "tree_id"),
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "research_lab_autoresearch_frontier_commitments",
        (
            "tree_id",
            "round_index",
            "schema_version",
            "expected_previous_hash",
            "frontier_hash",
            "frontier_doc",
            "commitment_hash",
            "created_at",
        ),
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "research_lab_autoresearch_tree_handoffs",
        (
            "tree_id",
            "run_id",
            "candidate_id",
            "node_id",
            "handoff_hash",
        ),
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "research_lab_autoresearch_tree_node_current",
        (
            "tree_id",
            "node_id",
            "parent_node_id",
            "root_branch_id",
            "depth",
            "child_ordinal",
            "generation_operation_id",
            "node_doc",
            "identity_hash",
            "current_event_type",
            "current_event_doc",
            "current_event_hash",
        ),
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "research_lab_autoresearch_operation_current",
        (
            "logical_operation_id",
            "tree_id",
            "node_id",
            "operation_kind",
            "operation_status",
            "request_hash",
            "result_hash",
            "settled_cost_microusd",
            "provider_call_count",
            "settlement_doc",
        ),
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "research_lab_autoresearch_tree_current",
        (
            "tree_id",
            "run_id",
            "root_artifact_hash",
            "root_manifest_hash",
            "root_source_tree_hash",
            "root_git_commit",
            "root_image_digest",
            "policy_hash",
            "evaluator_commitment_hash",
            "tree_doc",
            "current_event_type",
            "current_event_doc",
            "current_event_hash",
            "current_round_index",
            "current_frontier_hash",
            "current_frontier_doc",
        ),
    ),
    (
        "scripts/115-research-lab-git-tree-root-replacement.sql",
        "research_lab_autoresearch_run_tree_current",
        (
            "tree_id",
            "run_id",
            "tree_generation",
            "root_artifact_hash",
            "replaces_tree_id",
            "root_manifest_hash",
            "root_source_tree_hash",
            "root_git_commit",
            "root_image_digest",
            "policy_hash",
            "evaluator_commitment_hash",
            "tree_doc",
            "current_event_type",
            "current_event_doc",
            "current_event_hash",
            "current_round_index",
            "current_frontier_hash",
            "current_frontier_doc",
        ),
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "research_lab_candidate_artifacts",
        (
            "git_tree_id",
            "git_tree_node_id",
            "git_tree_root_commit",
            "git_tree_node_commit",
            "git_tree_lineage_hash",
        ),
    ),
    (
        "scripts/97-research-lab-conditional-validation.sql",
        "research_lab_conditional_validation_events",
        ("event_id", "assignment_hash", "policy_hash"),
    ),
    (
        "scripts/97-research-lab-conditional-validation.sql",
        "research_lab_scoring_category_results",
        ("category_result_id", "source_kind", "category", "assignment_hash"),
    ),
    (
        "scripts/97-research-lab-conditional-validation.sql",
        "research_lab_scoring_job_candidate",
        (
            "conditional_total",
            "baseline_preliminary_score",
            "threshold_points",
            "preliminary_gate_status",
            "category_assignment_hash",
            "conditional_policy_hash",
        ),
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
        "scripts/118-research-lab-maintenance-lease.sql",
        "research_lab_maintenance_lease",
        ("lease_name", "holder_ref", "expires_at"),
    ),
    (
        "scripts/121-research-lab-atomic-candidate-claim.sql",
        "research_lab_candidate_claim",
        ("candidate_id", "holder_ref", "claimed_at", "expires_at"),
    ),
    (
        "scripts/122-research-lab-atomic-run-claim.sql",
        "research_loop_run_claim",
        ("run_id", "holder_ref", "claimed_at", "expires_at"),
    ),
    (
        "scripts/123-research-lab-corpus-completeness.sql",
        "research_lab_corpus_complete",
        ("trajectory_id", "run_id", "source_watermark", "completed_at"),
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
            "precheck_status",
            "precheck_doc",
            "source_identity_hash",
            "source_identity_version",
        ),
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_control",
        ("singleton", "paused", "reason", "actor_ref", "updated_at"),
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
    # Routing experiments are an independent, append-only authority. Keep
    # every durable relation in the restart gate so a worker cannot start
    # against a partially applied 157 schema.
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_experiments_v2",
        ("experiment_hash", "experiment_id", "spec_doc", "receipt_execution_mode"),
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_experiment_events_v2",
        ("event_hash", "experiment_hash", "event_type", "event_doc"),
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_experiment_claims_v2",
        ("claim_key", "experiment_hash", "claim_generation", "claim_state"),
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_experiment_claim_heartbeats_v2",
        ("heartbeat_key", "claim_key", "claim_generation", "lease_expires_at"),
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_experiment_claim_closures_v2",
        ("close_key", "claim_key", "claim_generation", "close_reason"),
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_experiment_claims_v3",
        (
            "claim_key",
            "experiment_hash",
            "request_hash",
            "lease_hash",
            "lease_generation",
            "claim_generation",
            "claim_state",
        ),
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_experiment_claim_heartbeats_v3",
        ("heartbeat_key", "claim_key", "claim_generation", "lease_expires_at"),
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_experiment_claim_closures_v3",
        ("close_key", "claim_key", "claim_generation", "close_reason"),
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_provider_attempts_v2",
        ("attempt_key", "experiment_hash", "provider_receipt_ref", "billing_state"),
    ),
    (
        "scripts/160-research-lab-routing-adapter-failures.sql",
        "research_lab_routing_adapter_failures_v2",
        (
            "failure_key",
            "experiment_hash",
            "provider_receipt_ref",
            "claim_key",
            "claim_generation",
            "failure_doc",
        ),
    ),
    (
        "scripts/162-research-lab-candidate-routing-experiments.sql",
        "research_lab_candidate_waterfall_receipts",
        (
            "receipt_id",
            "experiment_hash",
            "decision_receipt_id",
            "provider_receipt_ref",
            "attempt_receipt_sha256",
            "prior_attempt_receipt_sha256",
            "attempt_chain_sha256",
            "target_verified_qualified_count",
            "receipt_doc",
        ),
    ),
    (
        "scripts/162-research-lab-candidate-routing-experiments.sql",
        "research_lab_candidate_waterfall_metrics",
        (
            "metric_id",
            "evaluation_receipt_id",
            "experiment_hash",
            "variant_id",
            "split",
            "metric_doc",
        ),
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_decision_receipts_v2",
        ("receipt_id", "experiment_hash", "variant_id", "decision_doc"),
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_evaluation_receipts_v2",
        ("receipt_id", "experiment_hash", "evaluation_hash", "evaluation_doc"),
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_lab_references_v2",
        ("reference_hash", "experiment_hash", "evaluation_receipt_id", "reconciliation_doc"),
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_budget_events_v2",
        ("event_key", "reservation_id", "experiment_hash", "event_type"),
    ),
    (
        "scripts/159-research-lab-routing-execution-queue.sql",
        "research_lab_routing_execution_request_leases_v2",
        (
            "request_hash",
            "lease_hash",
            "lease_generation",
            "worker_ref",
            "lease_state",
        ),
    ),
)

REQUIRED_SUPABASE_V2_RPCS = (
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "create_research_lab_autoresearch_tree",
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "plan_research_lab_autoresearch_tree_node",
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "append_research_lab_autoresearch_tree_event",
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "transition_research_lab_autoresearch_operation",
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "commit_research_lab_autoresearch_frontier",
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "select_research_lab_autoresearch_tree_final",
    ),
    (
        "scripts/95-research-lab-git-tree-autoresearch.sql",
        "record_research_lab_autoresearch_tree_handoff",
    ),
    (
        "scripts/115-research-lab-git-tree-root-replacement.sql",
        "create_research_lab_git_tree_candidate_handoff",
    ),
    (
        "scripts/115-research-lab-git-tree-root-replacement.sql",
        "research_lab_autoresearch_run_evaluation_usage",
    ),
    (
        "scripts/117-research-lab-trajectory-antijoin.sql",
        "research_lab_missing_trajectory_ids",
    ),
    (
        "scripts/118-research-lab-maintenance-lease.sql",
        "research_lab_acquire_maintenance_lease",
    ),
    (
        "scripts/119-research-lab-provider-usage-batch-insert.sql",
        "insert_research_lab_provider_usage_ledger_rows",
    ),
    (
        "scripts/120-research-lab-trajectory-delta.sql",
        "research_lab_next_unprojected_terminal_runs",
    ),
    (
        "scripts/120-research-lab-trajectory-delta.sql",
        "research_lab_terminal_runs_missing_traces",
    ),
    (
        "scripts/121-research-lab-atomic-candidate-claim.sql",
        "claim_next_research_lab_candidate",
    ),
    (
        "scripts/122-research-lab-atomic-run-claim.sql",
        "claim_next_research_loop_run",
    ),
    (
        "scripts/123-research-lab-corpus-completeness.sql",
        "research_lab_corpus_source_watermark",
    ),
    (
        "scripts/123-research-lab-corpus-completeness.sql",
        "research_lab_mark_corpus_complete",
    ),
    (
        "scripts/123-research-lab-corpus-completeness.sql",
        "research_lab_terminal_runs_needing_corpus",
    ),
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
        "scripts/146-research-lab-private-benchmark-schema-v11.sql",
        "research_lab_private_benchmark_schema_contract_v1",
    ),
    (
        "scripts/148-research-lab-atomic-credit-resume.sql",
        "resume_research_lab_credit_blocked_run_v1",
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
        "scripts/135-research-lab-active-model-result-replay.sql",
        "research_lab_active_model_replay_contract_v2",
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
        "scripts/154-research-lab-model-compatibility-purpose.sql",
        "research_lab_candidate_hybrid_purpose_contract_v1",
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
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_claim_work",
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_finish_work",
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_configure_probe",
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_requeue_provenance",
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_set_paused",
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_reserve_leg1_slot",
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_finalize_leg1",
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_enqueue_provision_smoke",
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_finalize_provision",
    ),
    (
        "scripts/96-research-lab-source-add-functional-workflow.sql",
        "research_lab_source_add_finalize_provision_smoke",
    ),
    (
        "scripts/145-research-lab-source-add-admission-control.sql",
        "research_lab_source_add_admission_control_contract_v1",
    ),
    (
        "scripts/153-research-lab-private-model-lineage-generation.sql",
        "research_lab_private_model_lineage_generation",
    ),
    # These RPCs are the complete durable store surface used by the isolated
    # routing worker. The API only uses the first entry (submit); all others
    # remain restart-gated for a later explicitly registered worker.
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_submit_experiment_v2",
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_request_execution_v2",
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_claim_experiment_v3",
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_renew_claim_v3",
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_close_claim_v3",
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_append_fenced_event_v3",
    ),
    (
        "scripts/161-research-lab-exact-model-transitions.sql",
        "research_lab_routing_exact_model_transition_contract_v1",
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_recover_claim_v3",
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_append_provider_attempt_v3",
    ),
    (
        "scripts/160-research-lab-routing-adapter-failures.sql",
        "research_lab_routing_append_adapter_failure_v3",
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_append_decision_receipt_v3",
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_append_evaluation_v3",
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_reserve_budget_v3",
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_settle_budget_v3",
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_mark_budget_uncertain_v3",
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_recover_budget_v3",
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_list_expired_budget_reservations_v3",
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_list_unresolved_budget_reservations_v3",
    ),
    (
        "scripts/157-research-lab-routing-experiment-authority.sql",
        "research_lab_routing_promote_v3",
    ),
    (
        "scripts/159-research-lab-routing-execution-queue.sql",
        "research_lab_routing_claim_execution_requests_v2",
    ),
    (
        "scripts/159-research-lab-routing-execution-queue.sql",
        "research_lab_routing_renew_execution_request_lease_v2",
    ),
    (
        "scripts/159-research-lab-routing-execution-queue.sql",
        "research_lab_routing_close_execution_request_lease_v2",
    ),
    (
        "scripts/159-research-lab-routing-execution-queue.sql",
        "research_lab_routing_claim_execution_v3",
    ),
)

# 158 changes the attested receipt purpose allowlist in place, so it has no
# relation or RPC of its own. It is still part of the immutable restart
# migration inventory and must be present in the candidate source tree.
REQUIRED_SUPABASE_V2_POLICY_MIGRATIONS = (
    "scripts/158-research-lab-routing-experiment-purposes.sql",
)

REQUIRED_SUPABASE_V2_QUEUE_MIGRATIONS = (
    "scripts/159-research-lab-routing-execution-queue.sql",
)


class SupabaseSchemaPreflightV2Error(RuntimeError):
    """The selected V2 release cannot use the live PostgREST schema."""


POSTGRES_IDENTIFIER_MAX_BYTES = 63


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


def _role_purpose_pairs_from_constraint_v1(
    definition: str,
) -> Dict[str, frozenset[str]]:
    if not isinstance(definition, str) or not definition.startswith("CHECK ("):
        raise SupabaseSchemaPreflightV2Error(
            "candidate hybrid purpose constraint definition is invalid"
        )
    clauses = re.findall(
        r"\(role = '([^']+)'::text\)\s+AND\s+"
        r"\(purpose = ANY \(ARRAY\[(.*?)\]\)\)",
        definition,
        flags=re.DOTALL,
    )
    parsed: Dict[str, frozenset[str]] = {}
    for role, encoded_purposes in clauses:
        if role in parsed:
            raise SupabaseSchemaPreflightV2Error(
                "candidate hybrid purpose constraint repeats a role"
            )
        purposes = re.findall(r"'([^']+)'::text", encoded_purposes)
        if not purposes or len(purposes) != len(set(purposes)):
            raise SupabaseSchemaPreflightV2Error(
                "candidate hybrid purpose constraint has invalid purposes"
            )
        parsed[role] = frozenset(purposes)
    expected = {
        str(role): frozenset(str(purpose) for purpose in purposes)
        for role, purposes in ROLE_PURPOSES.items()
    }
    if parsed != expected:
        raise SupabaseSchemaPreflightV2Error(
            "candidate hybrid purpose constraint differs from canonical roles"
        )
    return parsed


def _verify_candidate_hybrid_purpose_contract_v1(
    *,
    headers: Mapping[str, str],
    supabase_url: str,
    opener: Any,
    timeout_seconds: float,
) -> Dict[str, Any]:
    request = Request(
        (
            f"{supabase_url}/rest/v1/rpc/"
            "research_lab_candidate_hybrid_purpose_contract_v1"
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
            "candidate hybrid purpose schema contract is unavailable; apply "
            "scripts/154-research-lab-model-compatibility-purpose.sql before "
            f"restart (HTTP {exc.code})"
        ) from exc
    except Exception as exc:
        raise SupabaseSchemaPreflightV2Error(
            "candidate hybrid purpose schema contract probe failed"
        ) from exc
    if status < 200 or status >= 300:
        raise SupabaseSchemaPreflightV2Error(
            "candidate hybrid purpose schema contract is unavailable; apply "
            "scripts/154-research-lab-model-compatibility-purpose.sql before "
            f"restart (HTTP {status})"
        )
    try:
        contract = json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise SupabaseSchemaPreflightV2Error(
            "candidate hybrid purpose schema contract response is invalid"
        ) from exc
    if not isinstance(contract, Mapping) or set(contract) != {
        "schema_version",
        "constraint_name",
        "constraint_valid",
        "constraint_definition",
    }:
        raise SupabaseSchemaPreflightV2Error(
            "candidate hybrid purpose schema contract response is invalid"
        )
    if (
        contract.get("schema_version")
        != "leadpoet.research_lab_candidate_hybrid_purpose_contract.v1"
        or contract.get("constraint_name")
        != "research_lab_attested_execution_receipts_v2_role_purpose_check"
        or contract.get("constraint_valid") is not True
    ):
        raise SupabaseSchemaPreflightV2Error(
            "candidate hybrid purpose schema contract differs"
        )
    definition = contract.get("constraint_definition")
    pairs = _role_purpose_pairs_from_constraint_v1(definition)
    return {
        "schema_version": contract["schema_version"],
        "constraint_name": contract["constraint_name"],
        "constraint_valid": True,
        "role_count": len(pairs),
        "role_purpose_pair_count": sum(len(value) for value in pairs.values()),
        "constraint_definition_sha256": "sha256:"
        + hashlib.sha256(definition.encode("utf-8")).hexdigest(),
    }


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
    activation_source = (
        "postgrest"
        if chain_realized_activation_authority is None
        else "provided-authority"
    )
    for migration, function_name in REQUIRED_SUPABASE_V2_RPCS:
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
    migrations.update(REQUIRED_SUPABASE_V2_POLICY_MIGRATIONS)
    migrations.update(REQUIRED_SUPABASE_V2_QUEUE_MIGRATIONS)
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
    for migration, function_name in REQUIRED_SUPABASE_V2_RPCS:
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
    candidate_hybrid_purpose_contract = (
        _verify_candidate_hybrid_purpose_contract_v1(
            headers=headers,
            supabase_url=supabase_url,
            opener=opener,
            timeout_seconds=timeout_seconds,
        )
    )
    return {
        "status": "ready",
        "probe_count": len(REQUIRED_SUPABASE_V2_SCHEMA)
        + len(REQUIRED_SUPABASE_V2_RPCS)
        + 3,
        "table_probe_count": len(REQUIRED_SUPABASE_V2_SCHEMA),
        "rpc_probe_count": len(REQUIRED_SUPABASE_V2_RPCS),
        "data_probe_count": 3,
        "schema_document_probe_count": 1,
        "chain_realized_settlement_activation_http_probe_count": (
            1 if activation_source == "postgrest" else 0
        ),
        "chain_realized_settlement_activation_source": activation_source,
        "chain_realized_settlement_activation": activation,
        "compact_weight_settlement_contract": (
            compact_weight_settlement_contract
        ),
        "candidate_hybrid_purpose_contract": (
            candidate_hybrid_purpose_contract
        ),
        "migration_files": sorted(migrations),
    }
