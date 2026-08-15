"""Hash and verify protected Research Lab business-logic ASTs.

The manifest deliberately hashes selected function, class, and policy-constant
definitions rather than whole files. I/O adapters and imports can move around
those definitions while CI continues to fail if scoring, autoresearch,
promotion, accounting, allocation, or weight behavior changes unintentionally.
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

PROTECTED_SYMBOLS = {
    "leadpoet_canonical/kms_recipient.py": (
        "decrypt_kms_recipient_ciphertext",
    ),
    "gateway/tee/coordinator_epoch_cutover_v2.py": (
        "attest_subnet_epoch_cutover_v2",
    ),
    "gateway/tee/execution_job_manager_v2.py": (
        "ExecutionJobManagerV2",
    ),
    "leadpoet_canonical/ancestry_checkpoint_v2.py": (
        "ANCESTRY_CHECKPOINT_BOOTSTRAP_REQUEST_SCHEMA_VERSION",
        "ANCESTRY_CHECKPOINT_BOOTSTRAP_RESULT_SCHEMA_VERSION",
        "derive_ancestry_lineage_id_v2",
        "validate_ancestry_lineage_id_v2",
        "validate_ancestry_projection_v2",
        "project_receipt_graph_v2",
        "build_ancestry_policy_v2",
        "validate_ancestry_policy_v2",
        "build_full_graph_parent_v2",
        "validate_full_graph_parent_v2",
        "build_full_graph_parent_authority_v2",
        "build_certificate_parent_authority_v2",
        "build_certificate_disclosure_parent_authority_v2",
        "issue_ancestry_certificate_v2",
        "validate_ancestry_certificate_v2",
        "validate_ancestry_parent_authority_v2",
        "validate_ancestry_delta_v2",
        "validate_local_delta_against_certificate_v2",
        "build_compact_ancestry_proof_from_delta_v2",
        "build_compact_ancestry_proof_v2",
        "validate_compact_ancestry_proof_v2",
        "build_checkpointed_receipt_graph_from_full_graph_v2",
        "select_ancestry_checkpoint_resume_frontier_v2",
        "_checkpoint_bootstrap_set_hash",
        "issue_legacy_ancestry_checkpoint_bootstrap_v2",
        "validate_ancestry_checkpoint_bootstrap_result_v2",
    ),
    "leadpoet_canonical/allocation_settlement_frontier_v2.py": (
        "ALLOCATION_SETTLEMENT_FRONTIER_SCHEMA_VERSION",
        "REWARD_SETTLEMENT_CHECKPOINT_SCHEMA_VERSION",
        "FRONTIER_MODES",
        "REWARD_KINDS",
        "MAX_REWARD_CHECKPOINTS",
        "MAX_NETUID",
        "MAX_EPOCH",
        "MAX_ALPHA_PERCENT",
        "ALPHA_QUANT",
        "AllocationSettlementFrontierV2Error",
        "_require",
        "_integer",
        "_identifier",
        "_hash",
        "_alpha_text",
        "_validated_alpha_text",
        "_decimal",
        "build_reward_settlement_checkpoint_v2",
        "validate_reward_settlement_checkpoint_v2",
        "reward_checkpoint_key_v2",
        "reward_checkpoint_index_v2",
        "build_allocation_settlement_frontier_v2",
        "validate_allocation_settlement_frontier_v2",
        "validate_frontier_successor_v2",
        "frontier_paid_maps_v2",
        "frontier_artifact_hashes_v2",
    ),
    "leadpoet_canonical/allocation_settlement_frontier_bootstrap_v2.py": (
        "ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_SCHEMA_VERSION",
        "ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_REQUEST_SCHEMA_VERSION",
        "ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_OPERATION",
        "ALLOCATION_SETTLEMENT_FRONTIER_BOOTSTRAP_PURPOSE",
        "AllocationSettlementFrontierBootstrapV2Error",
        "build_allocation_settlement_frontier_bootstrap_v2",
        "validate_allocation_settlement_frontier_bootstrap_v2",
        "frontier_bootstrap_artifact_hashes_v2",
    ),
    "leadpoet_canonical/compact_weight_authority_v2.py": (
        "compact_gateway_frontier_receipt_hashes_v2",
        "build_weight_ancestry_commitment_v2",
        "validate_compact_weight_submission_shape_v2",
        "compact_weight_bundle_hash_v2",
        "build_checkpointed_weight_graph_from_compact_v2",
        "validate_compact_weight_ancestry_v2",
        "reconstruct_published_weight_bundle_from_compact_v2",
        "build_stateful_epoch_evidence_from_compact_v2",
        "build_compact_weight_finalization_submission_v2",
        "validate_compact_weight_finalization_shape_v2",
        "validate_compact_weight_finalization_ancestry_v2",
        "build_checkpointed_weight_finalization_graph_v2",
        "reconstruct_weight_finalization_from_compact_v2",
    ),
    "leadpoet_canonical/compact_auditor_authority_v2.py": (
        "build_compact_published_weight_authority_v2",
        "validate_compact_published_weight_authority_shape_v2",
        "verify_compact_weight_submission_v2",
        "verify_compact_published_weight_authority_v2",
    ),
    "gateway/research_lab/tee_protocol.py": (
        "normalize_tee_protocol",
    ),
    "gateway/api/weights.py": (
        "get_weight_inputs_v2",
        "_submit_weights_v2_impl",
        "submit_weights_v2",
        "submit_compact_weights_v2",
        "finalize_weights_v2",
        "finalize_compact_weights_v2",
        "get_attested_weights_v2",
        "get_compact_published_weights_v2",
        "submit_weights",
    ),
    "research_lab/code_editing.py": (
        "build_loop_direction_planner_messages",
        "build_loop_direction_reference_repair_messages",
        "build_code_edit_source_inspection_messages",
        "build_code_edit_auto_research_messages",
        "build_code_edit_fallback_messages",
        "build_plan_alignment_judge_messages",
        "build_code_edit_repair_messages",
        "parse_loop_direction_plan_response",
        "parse_plan_alignment_judge_response",
        "parse_code_edit_source_inspection_response",
        "parse_code_edit_response",
        "parse_code_edit_repair_response",
        "_GIT_STRUCTURAL_DIFF_PREFIXES",
        "git_diff_structural_metadata",
        "validate_code_edit_draft",
        "code_edit_candidate_manifest",
    ),
    "gateway/research_lab/autoresearch_runtime.py": (
        "AutoResearchRuntimeSettings",
        "AutoResearchLoopEvent",
        "would_exceed_budget",
    ),
    "gateway/research_lab/code_loop_engine.py": (
        "CodeEditLoopEngine",
        "_bind_loop_direction_plan",
    ),
    "research_lab/sourcing_model_contract_check.py": (
        "REVIEWED_CONSUMER_SNAPSHOT_SPECS",
        "load_wrapper_contract",
        "reviewed_consumer_snapshots",
        "_snapshot_sha256",
        "resolve_reviewed_consumer_snapshot",
        "verify_source_tree_contract",
    ),
    "research_lab/eval/private_runtime.py": (
        "EXPECTED_ROUTING_COMPILER_VERSION",
        "_REVIEWED_CONSUMER_MANIFEST_PAIRS",
        "_PROVIDER_OUTAGE_TEXT_MARKERS",
        "_provider_error_line_is_loop_ending",
        "_raise_on_empty_provider_error",
        "_private_manifest_hash_payload",
        "_verify_consumer_contract_manifest",
        "verify_private_artifact_manifest_signature",
        "_verify_private_artifact_manifest_signature_cached",
        "_verify_private_artifact_manifest_signature_uncached",
        "build_local_private_artifact_manifest",
        "validate_sourcing_adapter_metadata",
    ),
    "gateway/research_lab/config.py": (
        "DEFAULT_PRIVATE_TEST_CMD",
        "DEFAULT_PRIVATE_BUILD_CMD",
    ),
    "gateway/research_lab/code_build.py": (
        "validate_private_code_edit_diff_artifact",
        "_verify_built_candidate_artifact",
        "CodeEditCandidateBuilder._build_under_deadline",
    ),
    "gateway/research_lab/provider_capabilities.py": (
        "_SOURCE_ADD_MANIFEST_RE",
        "_SOURCE_ADD_RUNTIME_PATH",
        "_SOURCE_ADD_BINDING_MANIFEST_SCHEMA_VERSION",
        "_SOURCE_ADD_V8_REQUEST_SCHEMA_VERSION",
        "_SOURCE_ADD_V7_REQUEST_SCHEMA_VERSION",
        "_SOURCE_ADD_REGISTRATION_FIELDS",
        "_SOURCE_ADD_V7_REGISTRATION_FIELDS",
        "_SOURCE_ADD_V7_LEGACY_REGISTRATION_FIELDS",
        "_SOURCE_ADD_V7_ORIGINAL_REGISTRATION_FIELDS",
        "_SOURCE_ADD_V7_GUIDANCE_WITHOUT_CATEGORIES_FIELDS",
        "_SOURCE_ADD_V8_REQUIRED_REGISTRATION_FIELDS",
        "_SOURCE_ADD_LITERAL_NAMES",
        "_SOURCE_ADD_EXECUTION_MODES",
        "_SOURCE_ADD_IDEMPOTENCY_MODES",
        "_SOURCE_ADD_COST_CLASSES",
        "_SOURCE_ADD_MANIFEST_FORBIDDEN_KEYS",
        "_SOURCE_ADD_V8_PLANNER_FIELDS",
        "EffectiveProviderCapabilities",
        "_source_add_model_string_tuple",
        "_source_add_manifest_string_tuple",
        "_source_add_bounded_int",
        "_source_add_bounded_float",
        "_source_add_description",
        "_source_add_binding_manifest",
        "_source_add_binding_manifest_digest",
        "_normalize_source_add_v8_registration",
        "_normalize_source_add_v7_registration",
        "normalize_source_add_planner_contract",
        "_source_add_provisioning_provenance",
        "_normalized_words",
        "_source_mention_forms",
        "_mentioned_source_add_providers",
        "approved_source_router_suggestions",
        "validate_source_add_registration_diff",
        "_credential_ready",
        "_resolved_credential_ready",
        "_provider_doc_from_source_row",
        "load_effective_provider_capabilities_sync",
    ),
    "gateway/research_lab/git_tree_models.py": (
        "TreePolicy",
        "TreeReplacement",
        "TreeEvaluation",
        "TreeNode",
        "TreeCheckpoint",
        "TreeResult",
        "derive_tree_id",
        "derive_frontier_commitment_hash",
        "derive_child_slot",
        "generation_operation_id",
        "build_operation_id",
        "cohort_evaluation_operation_id",
        "tree_rank_key",
        "select_finalist",
        "next_child_slot",
    ),
    "gateway/research_lab/git_tree_scheduler.py": (
        "GitTreeScheduler",
        "sanitized_branch_context",
    ),
    "gateway/research_lab/git_tree_repository.py": (
        "GitTreeRepository",
    ),
    "gateway/research_lab/git_tree_store.py": (
        "GitTreeStore",
    ),
    "gateway/research_lab/git_tree_evaluator.py": (
        "TreeEvaluationPlan",
        "classify_tree_evaluation",
        "classify_candidate_tree_evaluation",
    ),
    "gateway/research_lab/dev_eval_runner.py": (
        "snapshot_readiness",
        "DockerReplayDevEvaluator",
        "AttestedReplayDevEvaluatorV2",
    ),
    "gateway/research_lab/champion_settlement_v2.py": (
        "validate_chain_weight_observation_v1",
        "select_chain_realized_bundle_candidate_v1",
        "build_chain_realized_settlement_package_v1",
        "validate_finalized_allocation_authorities_v2",
        "validate_legacy_settlement_migrations_v2",
        "validate_legacy_allocation_nonfinalizations_v2",
        "validate_chain_realized_epoch_settlements_v1",
        "validate_chain_realized_obligation_credits_v1",
        "merge_finalized_allocation_histories_v2",
        "merge_settled_allocation_histories_v2",
        "load_finalized_allocation_history_v2",
        "load_chain_realized_allocation_history_v1",
        "validate_chain_realized_settlement_bootstrap_v1",
        "load_settled_allocation_history_v2",
        "load_legacy_allocation_nonfinalizations_v2",
        "champion_v2_cutover_readiness",
    ),
    "gateway/research_lab/attested_v2_store.py": (
        "_select_by_values",
        "persist_receipt_graph_v2",
        "persist_ancestry_checkpoint_v2",
        "load_ancestry_checkpoint_proofs_v2",
        "load_ancestry_checkpoint_proof_v2",
        "load_checkpointed_receipt_graphs_v2",
        "load_checkpointed_receipt_graph_v2",
        "persist_compact_weight_submission_v2",
        "persist_compact_weight_publication_intent_v2",
        "load_compact_weight_publication_intent_v2",
        "persist_compact_weight_authority_v2",
        "load_compact_weight_authority_v2",
        "load_compact_weight_authority_for_identity_v2",
        "_load_receipt_graph_batch_v2",
        "load_receipt_graph_v2",
        "load_receipt_graphs_v2",
        "load_business_artifact_graph_v2",
        "load_business_artifact_graphs_v2",
        "load_business_artifact_graph_by_ref_v2",
        "load_business_artifact_graphs_by_ref_v2",
        "load_execution_result_by_receipt_v2",
        "_validate_allocation_settlement_frontier_storage_v2",
        "load_allocation_settlement_frontier_context_v2",
        "persist_allocation_settlement_frontier_v2",
        "persist_legacy_finalized_allocation_migration_v2",
        "persist_chain_realized_settlement_v1",
        "persist_legacy_allocation_nonfinalization_v2",
    ),
    "gateway/research_lab/attested_artifacts_v2.py": (
        "_select_committed_encrypted_artifacts",
        "persist_execution_transport_artifacts_v2",
    ),
    "gateway/research_lab/attested_scoring_v2.py": (
        "_local_failed_receipt_hashes",
        "_is_checkpoint_bootstrap_scope",
        "_resolve_parent_ancestry_transport_v2",
        "_validate_output_ancestry_checkpoint_v2",
        "_validate_checkpointed_graph_proof_v2",
        "_persist_ancestry_checkpoint_after_graph_v2",
        "_persist_graph_then_ancestry_checkpoint_v2",
        "_compact_parent_graphs_for_transport",
        "execute_scoring_v2",
    ),
    "gateway/research_lab/attested_autoresearch_v2.py": (
        "_resolve_parent_ancestry_transport_v2",
        "_trusted_parent_authorities_v2",
        "_persist_ancestry_checkpoint_after_graph_v2",
        "execute_autoresearch_v2",
    ),
    "gateway/research_lab/attested_weight_inputs_v2.py": (
        "_compact_weight_ancestry",
        "build_gateway_weight_inputs_v2",
    ),
    "gateway/research_lab/autoresearch_authority_v2.py": (
        "run_authoritative_autoresearch_v2",
    ),
    "gateway/research_lab/snapshot_refresh.py": (
        "maybe_refresh_dev_snapshot",
    ),
    "gateway/tee/autoresearch_executor_v2.py": (
        "_HostGitTreeRepository",
        "_HostCandidateBuilder",
        "AutoresearchExecutorV2",
    ),
    "research_lab/eval/evaluator.py": (
        "evaluate_private_model_pair",
        "score_private_model_pair_items",
        "_score_single_icp",
        "_run_candidate_with_retries",
        "build_holdout_gate_result",
        "_score_with_private_holdout_gate",
        "benchmark_icp_score_from_company_scores",
        "_benchmark_icp_score",
        "build_score_bundle_from_scored_icps",
        "build_scoring_health_doc",
        "prepare_autoresearch_scoring_payload",
    ),
    "research_lab/eval/dev_eval.py": (
        "build_current_day_dev_bank",
        "select_current_day_dev_icps",
        "select_snapshot_dev_icps",
        "build_dev_icp_set",
        "DevEvalResult",
        "evaluate_dev",
        "_score_dev_items",
    ),
    "research_lab/eval/snapshot_store.py": (
        "_OPENROUTER_MODEL_ENDPOINT_SUFFIXES",
        "_OPENROUTER_CONTROL_ENDPOINTS_BY_METHOD",
        "build_snapshot_pointer_document",
        "verify_snapshot_pointer_document",
        "ProviderSnapshotStore",
    ),
    "research_lab/eval/baseline_summary.py": (
        "build_baseline_health",
        "daily_noise_budget_doc",
        "build_baseline_score_summary",
    ),
    "research_lab/eval/promotion_metric.py": (
        "_paired_lcb_promotion_metric",
        "promotion_improvement_metric",
        "promotion_gate_decision",
    ),
    "research_lab/eval/provider_evidence_cache.py": (
        "canonical_request_fingerprint",
        "icp_evidence_cache_key",
        "build_evidence_cache_from_trace_entries",
        "merge_evidence_caches",
    ),
    "research_lab/eval/provider_costs.py": (
        "estimate_provider_cost",
        "ProviderCostLedger",
        "summarize_provider_cost_events",
        "summarize_provider_cost_trace_entries",
    ),
    "research_lab/source_add_rewards.py": (
        "SourceAddRewardRecord",
        "validate_source_add_reward_record",
        "create_leg1_reward",
        "create_leg2_reward",
        "stop_reward_forward",
    ),
    "gateway/research_lab/source_add_provenance.py": (
        "sanitize_source_add_precheck_doc",
        "evaluate_source_add_provenance",
    ),
    "gateway/research_lab/source_add_workflow.py": (
        "build_automatic_probe_config",
        "process_source_add_work_item",
        "_process_provenance",
        "_process_functional_probe",
        "_process_leg1_reward",
        "_retry_allowed",
        "_retry_at",
    ),
    "gateway/research_lab/source_add_llm_judge.py": (
        "SourceAddJudgeVerdict",
        "judge_source_add_implementation",
        "_parse_verdict",
    ),
    "gateway/research_lab/allocations.py": (
        "_build_allocation_v2_singleflight",
        "build_research_lab_allocation_bundle",
        "_champion_finalized_paid_alpha_to_date",
        "_champion_obligation_caps",
        "_champion_paid_alpha_to_date_from_snapshots",
        "_champion_lifetime_credit_ledger_from_snapshots",
        "_champion_replay_obligation",
        "champion_reward_requires_allocation_history_v2",
        "_historical_compute_fallback_from_snapshot",
        "_epoch_active",
    ),
    "gateway/research_lab/maintenance.py": (
        "reconcile_champion_reward_statuses",
        "backfill_champion_reward_v2_authority",
        "backfill_champion_settlement_v2_authority",
        "backfill_historical_compute_fallback_v2_authority",
        "backfill_source_add_reward_v2_authority",
        "champion_v2_cutover_readiness_report",
    ),
    "gateway/research_lab/arweave_audit.py": (
        "_verified_rebuffer_event",
        "record_research_lab_checkpointed_events",
        "rebuffer_research_lab_buffered_audit_events",
        "recover_research_lab_checkpointed_audit_epochs",
    ),
    "gateway/research_lab/v2_authority.py": (
        "attest_historical_champion_reward_v2",
        "attest_historical_champion_settlement_v2",
        "attest_historical_source_add_reward_v2",
        "classify_historical_champion_allocation_v2",
        "_current_allocation_frontier_outcome_v2",
        "build_allocation_v2",
        "settle_chain_realized_epoch_v1",
        "ensure_chain_realized_settlements_v1",
        "_load_allocation_parent_graphs_v2",
        "execute_provider_preflight_v2",
    ),
    "gateway/research_lab/promotion.py": (
        "_promotion_reason_recorded",
        "_ensure_source_add_leg2_reward_activation",
        "reconcile_source_add_leg2_reward_activations",
        "confirmation_min_delta",
        "confirmation_attempt_budget",
        "_baseline_aggregate_excluding_icps",
        "ResearchLabPromotionController.process_scored_candidate",
        "ResearchLabPromotionController._promote_built_image_candidate",
        "ResearchLabPromotionController._maybe_create_source_add_implementation_rewards",
        "_load_valid_artifact",
        "_push_candidate_source_diff_to_repo",
        "_candidate_icp_score",
    ),
    "gateway/research_lab/scoring_worker.py": (
        "_baseline_preflight_monotonic",
        "_baseline_worker_index_for_attempt",
        "_baseline_runner_for_attempt",
        "_compatible_baseline_retry_extension",
        "_emit_private_baseline_retry_extension",
        "_load_baseline_scoring_progress",
        "_load_candidate_source_diff",
        "ResearchLabGatewayScoringWorker._is_private_baseline_owner",
        "ResearchLabGatewayScoringWorker._run_lease_held_recovery_and_preflight",
        "ResearchLabGatewayScoringWorker._run_owned_provider_preflight",
        "ResearchLabGatewayScoringWorker._enforce_baseline_wave_preflight_freshness",
        "ResearchLabGatewayScoringWorker._run_baseline_batch",
        "ResearchLabGatewayScoringWorker._run_baseline_batch_inner",
    ),
    "gateway/research_lab/provider_preflight.py": (
        "ProviderPreflight.check",
        "_cached_attested_preflight",
        "preflight_gate",
    ),
    "gateway/tee/update_gateway_rebenchmark_retry_secret.py": (
        "_json_object_without_duplicates",
        "_parse_shell_environment",
        "_parse_environment",
        "_render_environment",
        "update_gateway_rebenchmark_retry_secret",
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
    "leadpoet_canonical/chain_source_v2.py": (
        "weights_storage_key",
        "decode_weights_storage",
        "last_update_storage_key",
        "decode_last_update_storage",
        "validate_arweave_checkpoint_event",
    ),
    "leadpoet_canonical/legacy_settlement_v2.py": (
        "validate_legacy_weight_bundle_v2",
        "legacy_chain_vector_matches_bundle_v2",
        "validate_legacy_audit_event_v2",
        "validate_legacy_allocation_nonfinalization_v2",
        "validate_legacy_nonfinalization_document_v2",
        "validate_legacy_finalized_settlement_v2",
        "validate_legacy_settlement_document_v2",
    ),
    "leadpoet_verifier/economics.py": (
        "compute_reimbursement_award",
        "build_reimbursement_schedule",
        "build_champion_reward_obligation",
        "allocate_research_lab_epoch",
        "_allocate_research_lab_epoch_existing",
        "cap_reimbursement_schedules_by_epoch",
        "compose_final_weight_vector",
        "_allocate_reimbursements_at_set_rate",
        "_allocate_pro_rata_exact",
        "_allocate_fallback_reimbursements",
        "_distribute_reimbursement_surplus",
        "_allocate_champions",
        "_allocate_champions_minimum_window",
        "_allocate_source_add",
        "_allocate_capped_pro_rata",
        "_normalize_reimbursement_obligation",
        "_normalize_fallback_reimbursement_obligation",
        "_normalize_champion_obligation",
        "_champion_desired_alpha_percent",
        "_champion_total_due_alpha_percent",
        "_champion_paid_alpha_percent_to_date",
        "_champion_remaining_alpha_percent",
    ),
    "gateway/tee/coordinator_reward_source_v2.py": (
        "CoordinatorRewardSourceV2",
    ),
    "gateway/tee/coordinator_allocation_source_v2.py": (
        "CoordinatorAllocationSourceV2",
    ),
    "gateway/tee/coordinator_allocation_frontier_bootstrap_v2.py": (
        "select_latest_allocation_source_row_v2",
        "CoordinatorAllocationFrontierBootstrapV2",
    ),
    "gateway/tee/bootstrap_allocation_settlement_frontier_v2.py": (
        "load_latest_checkpointed_allocation_source_v2",
        "_load_candidate_reward_graphs_v2",
        "ensure_allocation_settlement_frontier_v2",
    ),
    "gateway/tee/supabase_source_v2.py": (
        "QUERY_POLICIES",
        "_filters",
        "SupabaseSourceReaderV2",
    ),
    "gateway/tee/coordinator_legacy_settlement_v2.py": (
        "CoordinatorLegacySettlementSourceV2",
    ),
    "gateway/tee/coordinator_executor_v2.py": (
        "CoordinatorExecutorV2",
    ),
    "gateway/tee/bootstrap_active_ancestry_checkpoints_v2.py": (
        "_lineage_id",
        "_load_release_manifest",
        "_LazyApprovedReleaseBootVerifier",
        "_verify_coordinator_capability",
        "_graph_root",
        "_load_frontier_bounded_allocation_graphs",
        "_select_active_graphs",
        "_bootstrap_one_graph",
        "bootstrap_active_ancestry_checkpoints_v2",
        "main",
    ),
    "gateway/tee/release_archive_v2.py": (
        "_path_exists_without_following",
        "_path_without_symlink_ancestry",
        "_real_directory",
        "_load_regular_json",
        "_normalize_role_pcr0s",
        "load_last_good_release",
        "_release_role_pcr0s",
        "_archived_role_pcr0s",
        "_verify_index_entry",
        "_verify_archive_index_locked",
        "verify_archive_index",
        "_sha256_file",
        "_measurement_pcr0",
        "_atomic_json",
        "_expected_sources",
        "_copy_regular",
        "verify_archive_directory",
        "_restored_runtime_files",
        "_install_replace",
        "_copy_regular_for_rollback",
        "_fsync_directory",
        "_install_restored_runtime",
        "_promote_verified_release_locked",
        "restore_verified_release",
        "archive_verified_release",
        "select_release_manifest",
    ),
    "gateway/tee/coordinator_active_model_source_v2.py": (
        "CoordinatorActiveModelSourceV2",
    ),
    "gateway/tee/coordinator_chain_source_v2.py": (
        "CoordinatorChainSourceV2",
    ),
    "gateway/tee/coordinator_chain_realized_settlement_v1.py": (
        "CoordinatorChainRealizedSettlementV1",
    ),
    "gateway/tee/coordinator_source_add_v2.py": (
        "CoordinatorSourceAddProvenanceV2",
        "CoordinatorSourceAddFunctionalProbeV2",
    ),
    "gateway/tee/source_add_runtime_v2.py": (
        "validate_source_add_credential_envelope_v2",
        "validate_source_add_sealed_job_envelope_v2",
        "build_source_add_probe_route_v2",
        "validate_source_add_runtime_route_v2",
        "build_source_add_probe_job_envelope_v2",
    ),
    "gateway/tee/coordinator_weight_source_v2.py": (
        "CoordinatorWeightSourceV2",
    ),
    "gateway/tee/model_sandbox_v2.py": (
        "MODEL_SANDBOX_CGROUP_V1_CONTROL_FILES",
        "MODEL_SANDBOX_VISIBLE_ROOT",
        "_pid_is_direct_cgroup_member",
        "_normalized_cgroup_relative_path",
        "_current_cgroup_path",
        "_current_cgroup_v1_paths",
        "_prepare_model_sandbox_cgroup_v1",
        "_runsc_run_command",
        "prepare_model_sandbox_cgroup_v2",
        "model_sandbox_job_cgroup_path",
        "_sandbox_visible_parent",
        "_sandbox_visible_workspace",
        "_sandbox_visible_path",
        "_copy_readonly_visible_tree",
        "_oci_config",
        "RunscModelSandboxV2",
    ),
    "gateway/tee/provider_broker_v2.py": (
        "MEASURED_TRANSPORT_REQUEST_HEADERS",
        "_extract_tls_metadata",
        "HTTPXProviderTransport",
        "ProviderBrokerV2",
        "provider_registry_document",
    ),
    "gateway/tee/egress_framing.py": (
        "TUNNEL_FRAMING_HEADER",
        "TUNNEL_FRAMING_MODE",
        "TUNNEL_FRAME_BYTES",
        "EgressTunnelFramingError",
        "_receive_exact_until",
        "send_tunnel_frame",
        "receive_tunnel_frame",
        "relay_raw_and_framed",
    ),
    "gateway/tee/egress_policy.py": (
        "EGRESS_POLICY_VERSION",
        "ALLOWED_PORTS",
        "policy_document",
        "destination_policy_hash",
    ),
    "gateway/tee/egress_proxy.py": (
        "_parse_proxy_request",
        "_FramedParentBridge",
        "_ManagedProxyStream",
        "EnclaveEgressProxy",
    ),
    "gateway/tee/artifact_persistence_v2.py": (
        "_ArtifactHTTPSProxyTransport",
        "_ArtifactVerificationTransportPool",
        "_ArtifactVerificationTransportSession",
        "ArtifactPersistenceVerifierV2",
    ),
    "gateway/utils/tee_egress_forwarder.py": (
        "_handle_connection",
        "TEEEgressForwarder",
    ),
    "gateway/tee/provider_semantics_v2.py": (
        "ProviderSemanticsAuthorityV2",
    ),
    "gateway/tee/mtls_identity.py": (
        "_atomic_private_write",
        "write_identity_to_tmpfs",
        "create_mutual_tls_context",
    ),
    "gateway/tee/inter_enclave_tls.py": (
        "MAX_REPLAY_CACHE_BYTES",
        "MAX_REPLAY_CACHE_ENTRIES",
        "MAX_RPC_DELIVERY_ATTEMPTS",
        "RPC_DELIVERY_BACKOFF_SECONDS",
        "REPLAY_CACHE_TTL_SECONDS",
        "REPLAY_WAIT_SECONDS",
        "InterEnclaveTLSError",
        "_RetryableInterEnclaveTransportError",
        "_recv_exact",
        "_send_frame",
        "_read_frame",
        "AttestedPeerRegistry",
        "AttestedTLSRPCClient",
        "AttestedTLSRPCServer",
        "build_rpc_request",
        "validate_rpc_request",
    ),
    "gateway/tee/tee_service.py": (
        "acknowledge_checkpoint",
        "build_checkpoint",
        "handle_v2_runtime_rpc",
        "get_v2_provider_broker",
        "get_v2_inter_enclave_client",
        "execute_v2_provider_request",
        "handle_inter_enclave_rpc",
        "start_v2_tls_service",
    ),
    "gateway/tee/rpc_authority.py": (
        "allowed_exact_methods",
        "rpc_method_allowed",
    ),
    "gateway/utils/arweave_client.py": (
        "checkpoint_payload_bytes",
        "upload_checkpoint",
        "wait_for_confirmation",
    ),
    "gateway/tasks/hourly_batch.py": (
        "build_arweave_checkpoint_log_event",
        "hourly_batch_task",
    ),
    "gateway/tee/reward_executor_v2.py": (
        "reward_receipt_projection_v2",
        "champion_reward_row_projection_v2",
        "source_add_reward_row_projection_v2",
        "reimbursement_reward_row_projection_v2",
        "execute_reward_decision_v2",
        "_source_add_migration",
    ),
    "gateway/tee/verify_weight_submission_ready_v2.py": (
        "_ancestry_safe_epoch_from_storage_readiness",
        "verify_weight_submission_storage_readable_v2",
        "repair_chain_realized_settlements_v1",
        "verify_weight_submission_ready_v2",
    ),
    "gateway/tee/release_lineage_v2.py": (
        "validate_compact_release_lineage_v2",
        "build_compact_release_lineage_boot_verifier_v2",
        "load_approved_release_lineage_v2",
        "build_release_lineage_boot_verifier_v2",
    ),
    "gateway/tee/scoring_executor_v2.py": (
        "ScoringExecutorV2",
        "ScoringExecutorV2._execute_provider_preflight",
    ),
}


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
    index = {}  # type: Dict[str, ast.AST]
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
