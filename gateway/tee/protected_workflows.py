"""Hash and verify protected Research Lab business-logic ASTs.

The manifest deliberately hashes selected function, class, and policy-constant
definitions rather than whole files. I/O adapters and imports can move around
those definitions while CI continues to fail if scoring, accounting,
allocation, or weight behavior changes unintentionally.
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

PROTECTED_SYMBOLS = {
    "gateway/main.py": (
        "lifespan",
        "_start_source_add_dispatcher_task",
        "_SOURCE_ADD_INDEPENDENT_PATHS",
        "_gateway_source_add_dispatcher_ready",
    ),
    "leadpoet_canonical/kms_recipient.py": (
        "decrypt_kms_recipient_ciphertext",
    ),
    "gateway/tee/coordinator_epoch_cutover_v2.py": (
        "attest_subnet_epoch_cutover_v2",
    ),
    "gateway/tee/code_hash.py": (
        "ATTESTED_RUNTIME_DIR",
        "ATTESTED_RUNTIME_PACKAGES",
        "ATTESTED_RUNTIME_FILES",
        "ATTESTED_RUNTIME_GENERATED_FILES",
        "_ATTESTED_RUNTIME_ROLES",
        "_FULL_COMMIT_RE",
        "_FALLBACK_COMMAND_TIMEOUT_SECONDS",
        "ROOT_FILES",
        "INCLUDE_DIRS",
        "HASH_SUFFIXES",
        "EXCLUDED_DIRS",
        "EXCLUDED_SUFFIXES",
        "EXCLUDED_NAMES",
        "GatewayCodeHashError",
        "_is_hashable",
        "_iter_files",
        "_fallback_environment",
        "_run_fallback_command",
        "_fallback_commit",
        "materialize_gateway_code_hash_runtime",
        "iter_gateway_code_hash_files",
        "iter_gateway_code_hash_payloads",
        "compute_gateway_code_hash",
    ),
    "gateway/tee/protected_workflows.py": (
        "stage_external_protected_sources",
        "main",
    ),
    "gateway/tee/supabase_schema_preflight_v2.py": (
        "SOURCE_ADD_POST_ACCEPT_LEG1_ROLLBACK_V1_FUNCTION_AUTHORITY_SHA256",
        "SOURCE_ADD_POST_ACCEPT_LEG1_FUNCTION_AUTHORITY_SHA256",
        "SOURCE_ADD_PROVENANCE_LEG1_FUNCTION_AUTHORITY_SHA256",
        "SOURCE_ADD_PROVENANCE_LEG1_TRIGGER_AUTHORITY_SHA256",
        "SOURCE_ADD_PROVENANCE_LEG1_VIEW_AUTHORITY_SHA256",
        "SOURCE_ADD_PROVENANCE_ORIGIN_VIEW_AUTHORITY_SHA256",
        "SOURCE_ADD_PROVENANCE_ORIGIN_REPAIR_FUNCTION_AUTHORITY_SHA256",
        "_source_add_leg1_release_environment_policy_v1",
        "_verify_source_add_post_accept_leg1_contract_v2",
        "_verify_source_add_post_accept_leg1_contract_v3",
        "_verify_source_add_post_accept_leg1_contract_v4",
        "SOURCE_ADD_CLAIM_CONTROL_FUNCTION_AUTHORITY_SHA256",
        "_verify_source_add_claim_control_contract_v1",
        "SOURCE_ADD_CLAIM_CONTROL_ROLLBACK_V1_CONTRACT_SHA256",
        "SOURCE_ADD_CLAIM_CONTROL_V2_FUNCTION_AUTHORITY_SHA256",
        "_verify_source_add_claim_control_contract_v2",
        "verify_required_supabase_v2_schema",
    ),
    "gateway/tee/execution_job_manager_v2.py": (
        "_DIRECT_SUPABASE_SIDECAR_NAMESPACES",
        "_execution_failure_code",
        "_job_input_limit_bytes",
        "ExecutionContextV2.record_transport",
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
        "_build_weight_inputs_v2_singleflight",
        "_weight_inputs_v2_has_authorized_work",
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
    "research_lab/docker_operation_lock_v2.py": (
        "DOCKER_OPERATION_LOCK_FILE_ENV",
        "DOCKER_OPERATION_ADMISSION_LOCK_FILE_ENV",
        "DOCKER_OPERATION_LOCK_TIMEOUT_ENV",
        "DOCKER_DAEMON_READY_TIMEOUT_ENV",
        "DEFAULT_DOCKER_OPERATION_LOCK_FILE",
        "DEFAULT_DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS",
        "DEFAULT_DOCKER_DAEMON_READY_TIMEOUT_SECONDS",
        "DockerOperationLockError",
        "_positive_integer_environment",
        "_bounded_deadline",
        "_docker_ready_environment",
        "docker_operation_lock_path",
        "docker_operation_admission_lock_path",
        "_acquire_file_lock_until",
        "wait_for_docker_daemon_ready",
        "shared_docker_operation_lock",
        "shared_docker_operation_source_paths",
    ),
    "validator_tee/host/docker_operation_guard_v2.py": (
        "_EXACT_HOST_GATEWAY_ARGS",
        "_HOST_GATEWAY_PYTHON_COMMAND",
        "_MAX_HOST_GATEWAY_CMDLINE_BYTES",
        "inspect_exact_host_gateway_runtime",
    ),
    "gateway/utils/pcr0_builder.py": (
        "DOCKER_OPERATION_LOCK_FILE",
        "DOCKER_OPERATION_ADMISSION_LOCK_FILE_ENV",
        "DOCKER_OPERATION_LOCK_TIMEOUT_SECONDS",
        "_docker_operation_admission_lock_file",
        "_docker_operation_lock_scope",
        "_run_sync_build_step_to_completion",
        "_communicate_build_process_to_completion",
    ),
    "gateway/research_lab/store.py": (
        "_TRANSIENT_ERROR_SIGNATURES",
        "_TRANSIENT_ERROR_TYPE_SIGNATURES",
        "_is_transient_store_error",
        "insert_rows",
        "_is_seq_conflict",
    ),
    "gateway/research_lab/source_add_trial_runner.py": (
        "_remove_interrupted_source_add_container",
        "build_source_add_sandbox_runner",
    ),
    "gateway/research_lab/provider_capabilities.py": (
        "_SOURCE_ADD_BINDING_MANIFEST_SCHEMA_VERSION",
        "_SOURCE_ADD_REGISTRATION_FIELDS",
        "_SOURCE_ADD_EXECUTION_MODES",
        "_SOURCE_ADD_IDEMPOTENCY_MODES",
        "_SOURCE_ADD_COST_CLASSES",
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
        "normalize_source_add_planner_contract",
        "_credential_ready",
        "_resolved_credential_ready",
        "_provider_doc_from_source_row",
        "load_effective_provider_capabilities_sync",
    ),
    "gateway/research_lab/champion_settlement_v2.py": (
        "CHAIN_WEIGHT_OBSERVATION_SCHEMA_VERSION_V2",
        "CHAIN_TIMELOCKED_REVEAL_PROOF_SCHEMA_VERSION_V2",
        "validate_timelocked_reveal_proof_v2",
        "validate_chain_weight_observation_v1",
        "_preliminary_compact_finalized_bundle_authority_v2",
        "select_compact_chain_realized_bundle_candidate_v2",
        "select_chain_realized_bundle_candidate_v1",
        "build_chain_realized_settlement_package_v1",
        "build_compact_chain_realized_settlement_package_v2",
        "build_unattributed_chain_realized_settlement_package_v2",
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
        "_ANCESTRY_CHECKPOINT_UNKNOWN_COMMIT_BACKOFF_SECONDS",
        "_EXACT_INSERT_BATCH_ROWS",
        "_ancestry_checkpoint_unknown_commit_sleep",
        "_exact_batch_row_key",
        "_read_exact_batch_rows",
        "_insert_exact_batch",
        "_insert_exact_rows",
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
        "_rehydrate_compact_execution_graph_v2",
        "load_execution_result_v2",
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
        "_encrypted_transport_attempts_v2",
        "_is_checkpoint_bootstrap_scope",
        "_resolve_parent_ancestry_transport_v2",
        "_validate_output_ancestry_checkpoint_v2",
        "_validate_checkpointed_graph_proof_v2",
        "_persist_ancestry_checkpoint_after_graph_v2",
        "_persist_graph_then_ancestry_checkpoint_v2",
        "_compact_parent_graphs_for_transport",
        "execute_scoring_v2",
    ),
    "gateway/research_lab/attested_weight_inputs_v2.py": (
        "_compact_weight_ancestry",
        "build_gateway_weight_inputs_v2",
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
        "DEFAULT_LEG1_ALPHA_PERCENT",
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
        "_DOCUMENTATION_HTTP_400_MAX_ATTEMPTS",
        "build_automatic_probe_config",
        "process_source_add_work_item",
        "_process_provenance",
        "_process_functional_probe",
        "_process_provisioning_smoke",
        "_current_builtin_disabled_provision_row",
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
    "gateway/research_lab/provider_preflight.py": (
        "ProviderPreflight.check",
        "_cached_attested_preflight",
        "preflight_gate",
    ),
    "gateway/research_lab/provider_evidence_proxy.py": (
        "EvidenceStore.acquire_or_wait",
    ),
    "gateway/tee/disable_gateway_miner_submissions_secret.py": (
        "__module__",
        "GATEWAY_SECRET_ID",
        "EXPECTED_AWS_ACCOUNT_ID",
        "EXPECTED_AWS_REGION",
        "EXPECTED_GATEWAY_ROLE_NAME",
        "TARGET_ENV_NAME",
        "TARGET_ENV_VALUE",
        "_LEGACY_IDENTICAL_SHELL_DUPLICATE_NAME",
        "_LEGACY_PRESERVED_AWS_ENV_NAMES",
        "RECOVERY_JOURNAL_SCHEMA_VERSION",
        "DEFAULT_RECOVERY_JOURNAL_PATH",
        "MAX_RECOVERY_JOURNAL_BYTES",
        "_TRUE_VALUES",
        "_FALSE_VALUES",
        "_VERSION_ID_RE",
        "_CUSTOM_STAGE_PREFIX",
        "_CUSTOM_STAGE_RE",
        "_STAGE_LABEL_RE",
        "_ENVIRONMENT_NAME_RE",
        "_FORBIDDEN_RESTART_AUTHORITY_NAMES",
        "_N_MINUS_ONE_HYDRATION_SKIP_NAMES",
        "_EC2_ROLE_ARN_RE",
        "_FORBIDDEN_AWS_ENV_NAMES",
        "_EXPECTED_AWS_ENDPOINTS",
        "GatewayMinerSubmissionsDisableError",
        "_json_object_without_duplicates",
        "_decode_shell_target_value",
        "_decode_legacy_duplicate_value",
        "_shell_records",
        "_parse_shell_environment",
        "_parse_environment",
        "_render_shell_environment",
        "_render_environment",
        "_secret_string",
        "_version_id",
        "_version_stages",
        "_stage_holders",
        "_require_unique_current",
        "_validate_initial_topology",
        "_read_current_secret",
        "_read_exact_secret",
        "_document_commitment",
        "_n_minus_one_hydrated_environment",
        "_n_minus_one_hydrated_environment_commitment",
        "_topology_commitment",
        "_recovery_journal_body",
        "_validate_recovery_journal",
        "_open_recovery_journal_parent_fd",
        "_ensure_recovery_journal_parent",
        "_write_recovery_journal",
        "_read_recovery_journal",
        "_remove_recovery_journal",
        "_validated_candidate",
        "_custom_stage_label",
        "_expected_staged_topology",
        "_expected_promoted_topology",
        "_remove_custom_stage",
        "_cleanup_candidate_and_verify_original_topology",
        "_fail_before_promotion",
        "_restore_original_topology",
        "_nontransaction_topology",
        "_set_previous_version",
        "_recover_orphan_transaction",
        "_disable_gateway_miner_submissions_secret",
        "_apply_gateway_miner_submissions_secret",
        "disable_gateway_miner_submissions_secret",
        "_instance_role_aws_clients",
        "_instance_role_secrets_client",
        "_verify_protected_source",
        "main",
    ),
    "gateway/tee/gateway_miner_maintenance_restart_v1.py": (
        "__module__",
        "SCHEMA_VERSION",
        "CANONICAL_GATEWAY_RESTART_LOCK_PATH",
        "CANONICAL_GATEWAY_ENV_PATH",
        "PROOF_FD_ENV_NAME",
        "PROOF_FD_NUMBER",
        "CONTROLLER_WRAPPER_FD_NUMBER",
        "CONTROLLER_GIT_HELPER_FD_NUMBER",
        "CONTROLLER_EXACT_COMMIT_HELPER_FD_NUMBER",
        "CONTROLLER_MEMORY_GUARD_FD_NUMBER",
        "MAX_PROOF_BYTES",
        "MAX_RUNTIME_STATUS_BYTES",
        "DEFAULT_RUNTIME_STATUS_URL",
        "SOURCE_ADD_PAUSE_RPC",
        "SOURCE_ADD_ADMISSION_CONTRACT_RPC",
        "SOURCE_ADD_CLAIM_CONTROL_CONTRACT_RPC",
        "SOURCE_ADD_RESTART_QUIESCENCE_RPC",
        "SOURCE_ADD_RESTART_GUARD_STATE_RPC",
        "SOURCE_ADD_ACQUIRE_RESTART_GUARD_RPC",
        "SOURCE_ADD_RELEASE_RESTART_GUARD_RPC",
        "SOURCE_ADD_CONTROL_TABLE",
        "SOURCE_ADD_PAUSE_REASON",
        "SOURCE_ADD_CONTROL_MAX_BYTES",
        "SOURCE_ADD_CONTROL_TIMEOUT_SECONDS",
        "SOURCE_ADD_CANONICAL_COORDINATION_DEADLINE_SECONDS",
        "SOURCE_ADD_RESTART_GUARD_SAFETY_MARGIN_SECONDS",
        "SOURCE_ADD_RESTART_GUARD_LEASE_SECONDS",
        "SOURCE_ADD_RESTART_GUARD_AUTHORITY",
        "SOURCE_ADD_QUIESCENCE_TIMEOUT_SECONDS",
        "SOURCE_ADD_QUIESCENCE_POLL_SECONDS",
        "SUPPORTED_N_MINUS_ONE_CONTROLLER_COMMITS",
        "LEGACY_N_MINUS_ONE_CONTROLLER_COMMIT",
        "LEGACY_N_MINUS_ONE_GATEWAY_GIT_HELPER",
        "RUNTIME_BUILD_IDENTITY_NAMES",
        "_COMMIT_RE",
        "_TREE_RE",
        "_VERSION_ID_RE",
        "_SHA256_RE",
        "_UNSAFE_GIT_ENV_NAMES",
        "_RESTART_AUTHORITY_NAMES",
        "_PROOF_FIELDS",
        "GatewayMinerMaintenanceRestartError",
        "_SourceAddAuthorityRejected",
        "_SourceAddClaimControlContractResponse",
        "_decode_secret_environment_value",
        "_source_add_pause_credentials",
        "_source_add_control_request",
        "_normalized_source_add_control",
        "_source_add_control_commitment",
        "_read_source_add_control",
        "_require_source_add_admission_control_contract",
        "_require_source_add_claim_control_contract",
        "_source_add_restart_guard_identity",
        "_source_add_owner_generation_commitment",
        "_source_add_guard_generation",
        "_source_add_expected_guard_generation",
        "_source_add_expected_restore_paused",
        "_source_add_guard_expiry",
        "_normalized_source_add_restart_guard_state",
        "_normalized_source_add_restart_guard",
        "_normalized_source_add_quiescence",
        "_read_source_add_restart_guard_state",
        "_read_source_add_restart_quiescence",
        "_require_owned_source_add_guard_state",
        "_source_add_quiescence_commitment",
        "_wait_for_source_add_quiescence",
        "_require_source_add_quiescent",
        "_acquire_source_add_restart_guard",
        "_pause_source_add_for_restart",
        "_renew_source_add_restart_guard",
        "_normalized_source_add_restart_guard_release",
        "_release_source_add_restart_guard",
        "_require_source_add_paused",
        "_require_source_add_state",
        "_force_source_add_paused_after_restart_failure",
        "_require_runtime_source_add_closed",
        "_require_runtime_source_add_restored",
        "_require_pre_activation_runtime_source_add_closed",
        "_require_pre_hydration_runtime_source_add_closed",
        "_candidate_root",
        "_require_fixed_bootstrap_authority",
        "_resolve_bootstrap_secrets_client",
        "_require_canonical_restart_lock_fd",
        "_read_bounded_proc_file",
        "_require_restart_authority_absent_from_environment_payload",
        "_live_gateway_restart_authority_commitment",
        "_pre_hydration_live_process_commitment",
        "_utc_now",
        "_json_object_without_duplicates",
        "_load_json",
        "_load_json_bytes",
        "_safe_git_environment",
        "_run_git",
        "_run_git_bytes",
        "_git_commit_exists",
        "_git_is_ancestor",
        "_canonical_remote",
        "_require_unmodified_git_object_authority",
        "_open_private_parent_fd",
        "_read_private_regular_file",
        "_read_hydrated_gateway_environment",
        "_require_hydrated_environment_commitment",
        "_replace_private_regular_file",
        "_read_exact_installed_file",
        "_harden_installed_controller_directory",
        "_verified_installed_controller_release_directory",
        "_verified_installed_controller_bundle",
        "_verify_installed_controller",
        "_validate_candidate_identity",
        "_proof_body",
        "_validate_proof_document",
        "_serialized_proof",
        "_required_memfd_seals",
        "_require_reserved_memfd_numbers_available",
        "_seal_payload_at_fd_number",
        "_read_sealed_payload_fd",
        "_proof_from_fd",
        "_proof_fd_from_environment",
        "_require_disabled_parent_environment",
        "_require_disabled_secret_readback",
        "_verify_proof_against_state",
        "prepare_gateway_miner_maintenance_restart",
        "verify_gateway_miner_maintenance_state",
        "_read_handoff_marker",
        "_wait_for_handoff_marker",
        "_close_bootstrap_tree",
        "_leave_and_close_bootstrap_tree",
        "_install_controller_bundle_memfds",
        "_controller_exec_environment",
        "bootstrap_gateway_miner_maintenance_restart",
        "_fetch_runtime_status",
        "verify_gateway_miner_maintenance_shutdown_quiescence",
        "_require_runtime_miner_disabled",
        "verify_gateway_miner_maintenance_runtime_state",
        "_active_tree_hash",
        "main",
    ),
    "scripts/verify_installed_gateway_controller_v1.py": (
        "__module__",
        "SUPPORTED_CONTROLLER_COMMITS",
        "RECOVERY_HOST_CONTROLLER_COMMITS",
        "CONTROLLER_FILES",
        "_COMMIT_RE",
        "_UNSAFE_GIT_ENV_NAMES",
        "InstalledGatewayControllerError",
        "_safe_git_environment",
        "_git",
        "_git_commit_exists",
        "_git_is_ancestor",
        "_require_unmodified_git_authority",
        "_open_parent_fd",
        "_read_exact_file",
        "_verify_directory",
        "_reviewed_controller_parent_paths",
        "verify_candidate_bound_controller_lineage",
        "verify_installed_controller_bundle",
        "_exec_verified_helper",
        "_recover_exact_controller_checkout_drift",
        "main",
    ),
    "gateway/tee/restart_preflight_v2.py": (
        "__module__",
        "FULL_TOPOLOGY_INSTANCE_TYPE",
        "REQUIRED_BOOT_ENVELOPE_FILES",
        "_COMMIT_RE",
        "_PROTECTED_PARENT_PLAINTEXT_SLOTS",
        "GatewayRestartPreflightV2Error",
        "_json",
        "_imds_instance_type",
        "_configured_processor_count",
        "_observed_capacity",
        "load_parent_environment",
        "_reject_parent_provider_plaintext",
        "verify_artifact_bucket_lock_v2",
        "verify_gateway_restart_preflight_v2",
        "main",
    ),
    "leadpoet_canonical/weight_computation.py": (
        "compute_final_weights_with_lab_arena",
        "weight_config_document",
        "normalize_to_u16_with_uids_pure",
        "research_lab_uid_weights_from_allocation",
        "compute_final_weights",
    ),
    "leadpoet_canonical/lab_arena_rewards.py": (
        "validate_reward_constants",
        "validate_reward_basis",
        "signing_key_from_document",
        "verify_reward_basis_signature",
        "signing_key_hash_from_environment",
        "rewards_enabled_from_environment",
        "reward_week_index",
        "champion_share_for_week",
        "governing_reward_basis",
        "epoch_eligible",
        "champion_uid_for_hotkey",
        "champion_values",
        "check_snapshot_champion_triple",
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
        "CHAIN_MAX_RUNTIME_METADATA_BYTES",
        "_REVEAL_PERIOD_METADATA_DEFAULTS_V2",
        "configure_chain_source_boundary_v2",
        "chain_source_policy_document",
        "chain_source_policy_hash",
        "chain_source_boundary_for_profile_v2",
        "reveal_period_epochs_storage_key",
        "decode_reveal_period_epochs_storage",
        "decode_runtime_metadata_commitment",
        "resolve_reveal_period_metadata_default_v2",
        "system_events_storage_key",
        "system_event_count_storage_key",
        "timelocked_weight_commits_storage_key",
        "decode_timelocked_weight_commits",
        "weights_storage_key",
        "decode_weights_storage",
        "last_update_storage_key",
        "decode_last_update_storage",
        "validate_arweave_checkpoint_event",
    ),
    "leadpoet_canonical/subtensor_events_v2.py": (
        "__module__",
    ),
    "leadpoet_canonical/production_parity_boundary_v2.py": (
        "PRODUCTION_SUPABASE_ORIGIN",
        "PRODUCTION_CHAIN_HOST",
        "PRODUCTION_CHAIN_ARCHIVE_HOST",
        "PRODUCTION_PARITY_ENV_NAMES",
        "_parity_configuration",
        "validate_production_parity_boundary_document_v2",
        "validate_production_parity_boundary_v2",
        "configured_boundary_document_v2",
        "configured_supabase_origin_v2",
        "production_parity_enabled_v2",
        "configured_chain_source_boundary_v2",
        "configured_chain_signing_profile_path_v2",
        "configured_rebenchmark_now_v2",
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
        "_canonical_source_add_created_at",
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
    "gateway/tee/active_release_requirements_v2.py": (
        "__module__",
        "ACTIVE_RELEASE_REQUIREMENTS_SCHEMA_VERSION",
        "MAX_ACTIVE_RELEASE_COMMITS",
        "ActiveReleaseRequirementsV2Error",
        "build_active_release_requirements_v2",
        "validate_active_release_requirements_v2",
    ),
    "gateway/tee/prepare_active_release_lineage_v2.py": (
        "__module__",
        "RESULT_SCHEMA_VERSION",
        "_FALLBACK_CONTEXTS",
        "PrepareActiveReleaseLineageV2Error",
        "_atomic_json_documents",
        "_fetch_exact_release_lineage_v2",
        "_load_active_source_add_graphs_v2",
        "prepare_validator_initial_active_lineage_v2",
        "prepare_gateway_final_active_lineage_v2",
        "prepare_validator_final_active_lineage_v2",
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
    "gateway/tee/coordinator_chain_source_v2.py": (
        "CoordinatorChainSourceV2",
    ),
    "gateway/tee/research_lab_runtime_config_v2.py": (
        "SCHEMA_VERSION",
        "_default_chain_signing_profile",
        "_normalized_epoch_authority",
        "build_research_lab_execution_config",
        "validate_research_lab_execution_config",
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
    "research_lab/employee_buckets.py": (
        "LINKEDIN_EMPLOYEE_BUCKETS",
        "_OBSERVED_EMPLOYEE_COUNT_INTERVALS",
        "normalize_employee_count_bucket",
        "normalize_observed_employee_count_bucket",
    ),
    "qualification/scoring/lead_scorer.py": (
        "_decision_from_observed_employee_size",
        "_reverify_decision",
        "_llm_reverify_company",
    ),
    "gateway/tee/provider_broker_v2.py": (
        "MEASURED_TRANSPORT_REQUEST_HEADERS",
        "PROVIDER_TRANSPORT_HEALTH_SCHEMA_VERSION",
        "PROVIDER_TRANSPORT_FAILURE_DIAGNOSTIC_SCHEMA_VERSION",
        "EGRESS_POLICY_DIRECT_ONLY",
        "_EGRESS_POLICIES",
        "_PROVIDER_TERMINAL_STATUSES",
        "_CHAIN_WEIGHT_OBSERVATION_PURPOSE",
        "_SAFE_ERROR_TYPE_RE",
        "_PROVIDER_ID_RE",
        "_PROVIDER_TRANSPORT_FAILURE_DIAGNOSTIC_FIELDS",
        "_PROVIDER_TRANSPORT_FAILURE_STAGES",
        "_CLEANUP_RESOURCE_KIND_BY_STAGE",
        "_MAX_DIAGNOSTIC_ERRNO",
        "_EXPLICIT_HTTP_TRANSPORT_ATTRIBUTE",
        "_BROKER_OWNED_HTTPX_CLIENTS_LOCK",
        "_BROKER_OWNED_HTTPX_CLIENTS",
        "_BROKER_OWNED_HTTPX_SEND_GRANT",
        "_register_broker_owned_httpx_client",
        "is_broker_owned_httpx_client",
        "_broker_owned_httpx_send_scope",
        "_extract_tls_metadata",
        "_local_resource_failure",
        "_safe_error_type",
        "_failure_code",
        "validate_provider_transport_failure_diagnostic",
        "_provider_transport_failure_diagnostic",
        "_force_close_response_network_stream",
        "_ExplicitProviderTransportCloseFailure",
        "_close_client_transports",
        "ProviderTransportCleanupError",
        "ProviderRouteV2",
        "HTTPXProviderTransport",
        "ProviderBrokerV2",
        "ProviderBrokerV2.reseal_transport_failure_diagnostic",
        "provider_registry_document",
    ),
    "gateway/tee/provider_client_v2.py": (
        "_headers_without_credentials",
        "_ExecutionScope",
        "BrokeredProviderTransportV2.execute_attested_local_http",
        "BrokeredProviderTransportV2._execute_request",
        "BrokeredProviderTransportV2.install",
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
        "MAX_PROC_TCP_HEALTH_BYTES",
        "MAX_PROC_TCP_HEALTH_ROWS",
        "_parse_proxy_request",
        "_proc_tcp_address_is_loopback",
        "_loopback_tcp_state_counts",
        "_process_transport_resource_health",
        "_shutdown_and_close_socket",
        "EnclaveEgressProxyCleanupError",
        "_FramedParentBridge",
        "_ManagedProxyStream",
        "EnclaveEgressProxy",
    ),
    "gateway/tee/proxy_transport_preflight_v2.py": (
        "WorkerProxyTransportCleanupV2Error",
        "_RETIRED_CLEANUP_LOCK",
        "_RETIRED_CLEANUP_RECOVERY_LOCK",
        "_RETIRED_CLEANUP_RESOURCES",
        "_retain_cleanup_resource",
        "_retry_retired_cleanup",
        "verify_tls_proxy_connect_v2",
    ),
    "gateway/tee/artifact_persistence_v2.py": (
        "ArtifactTransportCleanupError",
        "_ArtifactHTTPSProxyTransport",
        "_ArtifactVerificationTransportPool",
        "_ArtifactVerificationTransportSession",
        "ArtifactPersistenceVerifierV2",
    ),
    "gateway/utils/tee_egress_forwarder.py": (
        "TEEEgressForwarderCleanupError",
        "_shutdown_and_close_socket",
        "_connect_public_destination",
        "_handle_connection",
        "TEEEgressForwarder",
        "main",
    ),
    "gateway/tee/provider_semantics_v2.py": (
        "_FAIL_CLOSED_REQUEST_SCHEMA_VERSION",
        "_SEMANTICS_HEALTH_STAGES",
        "ProviderSemanticsAuthorityV2",
    ),
    "gateway/tee/mtls_identity.py": (
        "ATTESTED_TLS_CERTIFICATE_LIFETIME",
        "ATTESTED_TLS_CERTIFICATE_CLOCK_SKEW",
        "_atomic_private_write",
        "generate_ephemeral_tls_identity",
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
        "TRANSPORT_HEALTH_SCHEMA_VERSION",
        "MAX_TRANSPORT_CLEANUP_EVENT_COUNT",
        "TRANSPORT_SUPERVISOR_POLL_SECONDS",
        "TRANSPORT_CLEANUP_ATTEMPTS_PER_RECOVERY_CYCLE",
        "MAX_TRANSPORT_CLEANUP_ATTEMPT_COUNT",
        "_TRANSIENT_ACCEPT_ERRNOS",
        "InterEnclaveTLSError",
        "_RetryableInterEnclaveTransportError",
        "InterEnclaveTransportCleanupError",
        "_ExplicitCloseFailure",
        "_close_transport_required",
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
        "VSOCK_RPC_TRANSPORT_HEALTH_SCHEMA_VERSION",
        "INTER_ENCLAVE_ROLE_TRANSPORT_HEALTH_SCHEMA_VERSION",
        "INTER_ENCLAVE_RPC_TRANSPORT_HEALTH_SCHEMA_VERSION",
        "MAX_VSOCK_RPC_CLEANUP_EVENT_COUNT",
        "VSOCK_RPC_SUPERVISOR_POLL_SECONDS",
        "VSOCK_RPC_CLEANUP_ATTEMPTS_PER_RECOVERY_CYCLE",
        "MAX_VSOCK_RPC_CLEANUP_ATTEMPT_COUNT",
        "_VSOCK_RPC_TRANSIENT_ACCEPT_ERRNOS",
        "vsock_rpc_transport_health_lock",
        "vsock_rpc_pending_cleanup_failures",
        "vsock_rpc_terminal_failure_event",
        "vsock_rpc_cleanup_recovery_lock",
        "acknowledge_checkpoint",
        "build_checkpoint",
        "handle_v2_runtime_rpc",
        "get_v2_provider_broker",
        "get_v2_coordinator_job_manager",
        "get_v2_scoring_job_manager",
        "get_v2_inter_enclave_client",
        "execute_v2_provider_request",
        "handle_inter_enclave_rpc",
        "start_v2_tls_service",
        "_inter_enclave_transport_health",
        "VSOCKRPCCleanupError",
        "_ExplicitVSOCKCloseFailure",
        "_close_vsock_rpc_required",
        "_record_vsock_rpc_cleanup",
        "_retain_vsock_rpc_cleanup_failure",
        "_recover_vsock_rpc_cleanup_failures",
        "vsock_rpc_transport_health",
        "_handle_vsock_connection",
        "_serve_vsock_connections",
    ),
    "gateway/tee/verify_topology.py": (
        "_REQUIRED_TRANSPORT_HEALTH_SCHEMAS_BY_ROLE",
        "_INTER_ENCLAVE_CHILD_TRANSPORT_HEALTH_SCHEMA",
        "_V2_RUNTIME_CONFIG_SCHEMA",
        "TopologyHealthError",
        "verify_roles",
    ),
    "gateway/tee/rpc_authority.py": (
        "COORDINATOR_ROLE",
        "active_enclave_role",
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
        "__module__",
        "validate_compact_release_lineage_v2",
        "build_compact_release_lineage_boot_verifier_v2",
        "load_approved_release_lineage_v2",
        "build_release_lineage_boot_verifier_v2",
    ),
    "gateway/tee/release_manifest_v2.py": (
        "__module__",
    ),
    "gateway/tee/release_channel_v2.py": (
        "__module__",
        "build_release_channel_v2",
        "validate_release_channel_v2",
        "fetch_release_channel_v2",
        "build_release_lineage_v2",
        "fetch_release_lineage_v2",
        "git_ancestor_commits_v2",
    ),
    "gateway/tee/scoring_executor_v2.py": (
        "SCORING_OPERATIONS_V2",
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
