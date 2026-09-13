-- Retire confirmed-empty Research Lab periphery without touching active or historical data.
-- The two empty v1 attestation tables stay because the shared epoch-fence trigger function
-- still names research_lab_attested_execution_receipts. Shared v2 attestation tables contain
-- surviving SOURCE_ADD history and are outside every allowlist below.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

CREATE TEMP TABLE _retire_empty_research_lab_tables (
    table_name TEXT PRIMARY KEY
) ON COMMIT DROP;

INSERT INTO _retire_empty_research_lab_tables (table_name) VALUES
    ('engine_generated_candidates'),
    ('engine_generated_datasets'),
    ('engine_generated_evaluators'),
    ('engine_issue_events'),
    ('engine_issues'),
    ('research_lab_autoresearch_frontier_commitments'),
    ('research_lab_autoresearch_operation_settlements'),
    ('research_lab_autoresearch_tree_events'),
    ('research_lab_autoresearch_tree_handoffs'),
    ('research_lab_autoresearch_tree_nodes'),
    ('research_lab_autoresearch_trees'),
    ('research_lab_candidate_model_unit_terminals'),
    ('research_lab_candidate_waterfall_metrics'),
    ('research_lab_candidate_waterfall_receipts'),
    ('research_lab_conditional_validation_events'),
    ('research_lab_routing_adapter_failures_v2'),
    ('research_lab_routing_budget_events_v2'),
    ('research_lab_routing_decision_receipts_v2'),
    ('research_lab_routing_evaluation_receipts_v2'),
    ('research_lab_routing_execution_request_leases_v2'),
    ('research_lab_routing_execution_requests_v2'),
    ('research_lab_routing_experiment_claim_closures_v2'),
    ('research_lab_routing_experiment_claim_closures_v3'),
    ('research_lab_routing_experiment_claim_heartbeats_v2'),
    ('research_lab_routing_experiment_claim_heartbeats_v3'),
    ('research_lab_routing_experiment_claims_v2'),
    ('research_lab_routing_experiment_claims_v3'),
    ('research_lab_routing_experiment_events_v2'),
    ('research_lab_routing_experiments_v2'),
    ('research_lab_routing_lab_references_v2'),
    ('research_lab_routing_provider_attempts_v2'),
    ('research_lab_scoring_job_candidate'),
    ('research_lab_scoring_job_queue');

CREATE TEMP TABLE _retire_empty_research_lab_views (
    view_name TEXT PRIMARY KEY
) ON COMMIT DROP;

INSERT INTO _retire_empty_research_lab_views (view_name) VALUES
    ('research_lab_autoresearch_operation_current'),
    ('research_lab_autoresearch_run_tree_current'),
    ('research_lab_autoresearch_tree_current'),
    ('research_lab_autoresearch_tree_node_current');

CREATE TEMP TABLE _retire_empty_research_lab_routines (
    routine_name TEXT NOT NULL,
    identity_arguments TEXT NOT NULL,
    expected_body_md5 TEXT NOT NULL,
    PRIMARY KEY (routine_name, identity_arguments)
) ON COMMIT DROP;

INSERT INTO _retire_empty_research_lab_routines (routine_name, identity_arguments, expected_body_md5) VALUES
    ('append_research_lab_autoresearch_tree_event', 'requested_tree_id text, requested_event_type text, requested_node_id text, requested_previous_event_hash text, requested_event_doc jsonb, requested_event_hash text', 'd8e8b3cdfe6a7f8669b4d2720b64a9ab'),
    ('commit_research_lab_autoresearch_frontier', 'requested_tree_id text, requested_round_index integer, requested_expected_previous_hash text, requested_frontier_hash text, requested_frontier_doc jsonb, requested_commitment_hash text', '9b5b624d146dbb64f49f505755bf4167'),
    ('create_research_lab_autoresearch_tree', 'requested_tree_id text, requested_run_id uuid, requested_root_artifact_hash text, requested_root_manifest_hash text, requested_root_source_tree_hash text, requested_root_git_commit text, requested_root_image_digest text, requested_policy_hash text, requested_evaluator_commitment_hash text, requested_tree_doc jsonb, requested_identity_hash text', '0ad88553ec7fcdadaa139a537bc0f732'),
    ('create_research_lab_git_tree_candidate_handoff', 'requested_candidate_doc jsonb, requested_tree_id text, requested_run_id uuid, requested_candidate_id text, requested_node_id text, requested_root_git_commit text, requested_node_git_commit text, requested_lineage_hash text, requested_handoff_doc jsonb, requested_handoff_hash text, requested_previous_event_hash text, requested_completed_event_hash text', 'd32e4e7b2f529273a8ce44ff6d3b434b'),
    ('guard_research_lab_git_tree_handoff_active_root', '', '298402069fa98ae9ee13e9198fa8569f'),
    ('plan_research_lab_autoresearch_tree_node', 'requested_tree_id text, requested_node_id text, requested_parent_node_id text, requested_root_branch_id text, requested_depth integer, requested_child_ordinal integer, requested_generation_operation_id text, requested_generation_request_hash text, requested_generation_transition_hash text, requested_node_doc jsonb, requested_identity_hash text', '95d9d2ee8c5c790833b10580d0864e24'),
    ('prevent_engine_issue_event_mutation', '', '7504af6f7123075c3de27cf3d49cf072'),
    ('prevent_research_lab_candidate_waterfall_mutation', '', '3fffea0072be65d57f3888e43ec7a8f2'),
    ('record_research_lab_autoresearch_tree_handoff', 'requested_tree_id text, requested_run_id uuid, requested_candidate_id text, requested_node_id text, requested_root_git_commit text, requested_node_git_commit text, requested_lineage_hash text, requested_handoff_doc jsonb, requested_handoff_hash text, requested_previous_event_hash text, requested_completed_event_hash text', '707a62a3871116d238d828ec305bed92'),
    ('research_lab_autoresearch_run_evaluation_usage', 'requested_run_id uuid', 'a4bb69ac2f5f568bb3228a61ce9c3f23'),
    ('research_lab_cancel_conditional_generation', 'target_queue_generation_id uuid, expected_claimed_by text, expected_attempt_count integer, target_failure_class text', '51eec238ab028be4c433e99e82e74564'),
    ('research_lab_candidate_append_model_unit_terminal_v1', 'p_receipt_id text, p_receipt_hash text, p_experiment_hash text, p_claim_key text, p_claim_generation bigint, p_terminal_doc jsonb', 'ad8b3be2b082136a47fe2e0dfced07ce'),
    ('research_lab_candidate_append_waterfall_metric_v1', 'p_metric_id text, p_metric_hash text, p_experiment_hash text, p_claim_key text, p_claim_generation bigint, p_metric_doc jsonb', 'c915b28548e14d71befc7ec072a97246'),
    ('research_lab_candidate_append_waterfall_receipt_v1', 'p_receipt_id text, p_receipt_hash text, p_experiment_hash text, p_claim_key text, p_claim_generation bigint, p_receipt_doc jsonb', '4b162ff3f999a4a66e6b87692b9da930'),
    ('research_lab_candidate_assert_model_unit_terminal_v1', 'p_experiment_hash text, p_variant_id text, p_unit_ref text', 'c20ab92e288636500ff7cf98b9203482'),
    ('research_lab_candidate_assert_model_waterfall_authority_v1', 'p_experiment_hash text', '1c6934e343fdf790591fa1e6342d2638'),
    ('research_lab_candidate_metric_projection_v1', 'p_experiment_hash text, p_evaluation_receipt_id text, p_variant_id text, p_split text, p_target_verified_qualified_count integer', '4928c2e27a4985f4c0b8efda9d077f2b'),
    ('research_lab_candidate_promotion_sidecars_guard_v1', '', '87e7b4a5236cab75febb1f4d97e1e604'),
    ('research_lab_claim_conditional_preliminary_gate', 'target_queue_generation_id uuid, target_worker_ref text, target_lease_seconds integer', '4f41430cd3a63ac3f2a17da3ded95711'),
    ('research_lab_claim_scoring_queue_failure_projection', 'target_worker_ref text, target_lease_seconds integer', '52a41adf616be6c5d793d90948b77b7b'),
    ('research_lab_complete_scoring_queue_failure_projection', 'target_queue_generation_id uuid, expected_claimed_by text, expected_attempt_count integer', '568b5e579507c791d1237cfd0fc636de'),
    ('research_lab_conditional_validation_event_hash', 'target_candidate_id text, target_event_type text, target_assignment_hash text, target_source_ref text, target_event_doc jsonb', '98e32edecbd9e640c93afd1f8473078f'),
    ('research_lab_decide_conditional_preliminary_gate', 'target_queue_generation_id uuid, candidate_preliminary_score double precision, target_preliminary_proof jsonb, expected_claimed_by text, expected_attempt_count integer', '7d03c42f8078a9daf19600bac6555784'),
    ('research_lab_decide_conditional_public_gate', 'target_queue_generation_id uuid, candidate_public_score double precision', 'e608e6302596ee26a6dab81514c61f92'),
    ('research_lab_fail_scoring_queue_generation', 'target_job_id uuid, expected_claimed_by text, expected_attempt_count integer, target_failure_class text', '5701ed63a4a083d511d1ea170f834749'),
    ('research_lab_requeue_conditional_scoring_job', 'target_job_id uuid, expected_claimed_by text, expected_attempt_count integer, target_failure_class text', '9dc32d04213a85247d7dfdb20df3d9f8'),
    ('research_lab_routing_append_adapter_failure_v3', 'p_failure_key text, p_experiment_hash text, p_provider_receipt_ref text, p_binding_id text, p_tool_id text, p_variant_id text, p_unit_ref text, p_claim_key text, p_claim_generation bigint, p_request_fingerprint text, p_latency_ms bigint, p_execution_mode text, p_failure_doc jsonb', '90b61b452181842d6e5e47bca9aa5c34'),
    ('research_lab_routing_append_decision_receipt_v2', 'p_receipt_id text, p_experiment_hash text, p_variant_id text, p_unit_ref text, p_plan_hash text, p_route_hash text, p_claim_key text, p_claim_generation bigint, p_claim_nonce text, p_decision_doc jsonb', 'dcfaa5bd1469e65f45bde70b121652bf'),
    ('research_lab_routing_append_decision_receipt_v3', 'p_receipt_id text, p_experiment_hash text, p_variant_id text, p_unit_ref text, p_plan_hash text, p_route_hash text, p_claim_key text, p_claim_generation bigint, p_decision_doc jsonb', 'd5c92cc0f8e675e9dde75c326e14750e'),
    ('research_lab_routing_append_evaluation_v2', 'p_receipt_id text, p_experiment_hash text, p_evaluation_hash text, p_selected_variant_id text, p_claim_key text, p_claim_generation bigint, p_claim_nonce text, p_evaluation_doc jsonb', 'ac5506455a224f98083d61a675e1da6d'),
    ('research_lab_routing_append_evaluation_v3', 'p_receipt_id text, p_experiment_hash text, p_evaluation_hash text, p_selected_variant_id text, p_claim_key text, p_claim_generation bigint, p_evaluation_doc jsonb', '2116b8d6110c2bb8b15882deadb954ac'),
    ('research_lab_routing_append_event_v2', 'p_event_hash text, p_experiment_hash text, p_event_type text, p_event_doc jsonb', '44120e09900b4b9febf43efb9d5104d4'),
    ('research_lab_routing_append_fenced_event_v2', 'p_event_hash text, p_experiment_hash text, p_event_type text, p_claim_key text, p_claim_generation bigint, p_claim_nonce text, p_event_doc jsonb', '7a1e50b298f5ecee87b40e6ea22b38f3'),
    ('research_lab_routing_append_fenced_event_v3', 'p_event_hash text, p_experiment_hash text, p_event_type text, p_claim_key text, p_claim_generation bigint, p_event_doc jsonb', '2e9c9b22bc754ecbcecbb1d34b170a6b'),
    ('research_lab_routing_append_only_v2', '', 'b04eeef622d84114f4d5bac79fb82fe2'),
    ('research_lab_routing_append_provider_attempt_v2', 'p_attempt_key text, p_experiment_hash text, p_provider_receipt_ref text, p_binding_id text, p_tool_id text, p_variant_id text, p_unit_ref text, p_reservation_id text, p_action_id text, p_binding_catalog_manifest_hash text, p_authorization_hash text, p_authorization_proof_hash text, p_request_body_hash text, p_claim_key text, p_claim_generation bigint, p_claim_nonce text, p_request_fingerprint text, p_outcome text, p_credit_microunits bigint, p_latency_ms bigint, p_execution_mode text, p_billing_state text, p_authoritative_billed_credit_microunits bigint, p_terminal_receipt_hash text, p_protected_release_receipt_hash text, p_admission_bundle_hash text, p_terminal_provider_record_hash text, p_terminal_billing_projection_hash text, p_attempt_doc jsonb', '62bdeb8153b858718461d39d2bd75fea'),
    ('research_lab_routing_append_provider_attempt_v3', 'p_attempt_key text, p_experiment_hash text, p_provider_receipt_ref text, p_binding_id text, p_tool_id text, p_variant_id text, p_unit_ref text, p_reservation_id text, p_action_id text, p_binding_catalog_manifest_hash text, p_authorization_hash text, p_authorization_request_hash text, p_authorization_proof_hash text, p_request_body_hash text, p_claim_key text, p_claim_generation bigint, p_request_fingerprint text, p_outcome text, p_credit_microunits bigint, p_latency_ms bigint, p_execution_mode text, p_billing_state text, p_authoritative_billed_credit_microunits bigint, p_terminal_receipt_hash text, p_protected_release_receipt_hash text, p_admission_bundle_hash text, p_terminal_request_hash text, p_terminal_result_hash text, p_terminal_provider_record_hash text, p_terminal_billing_projection_hash text, p_attempt_doc jsonb', 'd110683e59ff7675d9e9aeebf5587df9'),
    ('research_lab_routing_assert_claim_v2', 'p_experiment_hash text, p_claim_key text, p_claim_generation bigint, p_claim_nonce text', 'dcd653762971633b36097d9150a80024'),
    ('research_lab_routing_assert_claim_v3', 'p_experiment_hash text, p_claim_key text, p_claim_generation bigint', 'e75198ef7252af6eabe8ee58cdbb06e5'),
    ('research_lab_routing_assert_promotion_receipt_chain_v2', 'p_experiment_hash text', 'a5c5591384d45a8a8e5f9f44f3fc3d6d'),
    ('research_lab_routing_assert_promotion_receipt_chain_v3', 'p_experiment_hash text', 'aecd6bf262eec7e4246f3a60a78e9aa8'),
    ('research_lab_routing_assert_promotion_reconciliation_v3', 'p_experiment_hash text, p_evaluation_receipt_id text, p_evaluation_hash text, p_selected_variant_id text, p_reconciliation_doc jsonb', '2ff9533ba48661a17733faeb9667fa18'),
    ('research_lab_routing_assert_provider_receipt_chain_v2', 'p_experiment_hash text, p_binding_id text, p_tool_id text, p_variant_id text, p_unit_ref text, p_action_id text, p_authorization_hash text, p_authorization_proof_hash text, p_terminal_receipt_hash text, p_protected_release_receipt_hash text, p_admission_bundle_hash text, p_terminal_provider_record_hash text, p_terminal_billing_projection_hash text, p_outcome text, p_credit_microunits bigint, p_latency_ms bigint, p_billing_state text, p_authoritative_billed_credit_microunits bigint, p_attempt_doc jsonb', 'd747213d7492347e4965fc5101f5d0e3'),
    ('research_lab_routing_assert_provider_receipt_chain_v3', 'p_experiment_hash text, p_binding_id text, p_tool_id text, p_variant_id text, p_unit_ref text, p_action_id text, p_authorization_hash text, p_authorization_request_hash text, p_authorization_proof_hash text, p_terminal_receipt_hash text, p_protected_release_receipt_hash text, p_admission_bundle_hash text, p_terminal_request_hash text, p_terminal_result_hash text, p_terminal_provider_record_hash text, p_terminal_billing_projection_hash text, p_outcome text, p_credit_microunits bigint, p_latency_ms bigint, p_billing_state text, p_authoritative_billed_credit_microunits bigint, p_attempt_doc jsonb', '1861be05ec004ee56448d2eadfef2747'),
    ('research_lab_routing_claim_capability_commitment_v2', 'p_claim_nonce text', '34dfffd9f4840c6f3ba0ed1202dd1a46'),
    ('research_lab_routing_claim_execution_requests_v2', 'p_worker_ref text, p_batch_size integer, p_lease_seconds integer', '0acbc76078d0ee99cfdda9b8fbb59b50'),
    ('research_lab_routing_claim_execution_v3', 'p_request_hash text, p_lease_hash text, p_lease_generation bigint, p_worker_ref text, p_claim_key text, p_claim_lease_seconds integer, p_claim_doc jsonb, p_event_hash text, p_event_doc jsonb', 'fc627cde8a3afe17d32a65b958390c53'),
    ('research_lab_routing_claim_experiment_v2', 'p_experiment_hash text, p_claim_key text, p_claim_nonce text, p_worker_ref text, p_lease_seconds integer, p_claim_doc jsonb, p_event_hash text, p_event_doc jsonb', '8b1fee8c85358b0697c45de0c28619ab'),
    ('research_lab_routing_claim_experiment_v3', 'p_experiment_hash text, p_request_hash text, p_lease_hash text, p_lease_generation bigint, p_claim_key text, p_worker_ref text, p_lease_seconds integer, p_claim_doc jsonb, p_event_hash text, p_event_doc jsonb', '05d4e0202a2d195549a854e151014b6e'),
    ('research_lab_routing_close_claim_v2', 'p_close_key text, p_experiment_hash text, p_claim_key text, p_claim_generation bigint, p_claim_nonce text, p_close_reason text, p_close_doc jsonb', '1adf099a252b21089ab8fd811ac27756'),
    ('research_lab_routing_close_claim_v3', 'p_close_key text, p_experiment_hash text, p_claim_key text, p_claim_generation bigint, p_close_reason text, p_close_doc jsonb', '0dde4f409cf116a2b93c047cd870440d'),
    ('research_lab_routing_close_execution_request_lease_v2', 'p_request_hash text, p_worker_ref text, p_lease_hash text, p_lease_generation bigint, p_close_reason text', 'a6add3a0cc1af47200feef8eb8893306'),
    ('research_lab_routing_exact_model_transition_contract_v1', '', 'd567fb5c020a85e9e8a2ccc8747a5e9a'),
    ('research_lab_routing_list_expired_budget_reservations_v2', 'p_experiment_hash text, p_claim_key text, p_claim_generation bigint, p_claim_nonce text', '67883bce564f248d322c34570e3664d4'),
    ('research_lab_routing_list_expired_budget_reservations_v3', 'p_experiment_hash text, p_claim_key text, p_claim_generation bigint', 'de44f7cb02fc3819b7d15d5c7fa59357'),
    ('research_lab_routing_list_unresolved_budget_reservations_v2', 'p_experiment_hash text, p_claim_key text, p_claim_generation bigint, p_claim_nonce text', '3a98d96a434294dcf17f05e9f8f4d9a6'),
    ('research_lab_routing_list_unresolved_budget_reservations_v3', 'p_experiment_hash text, p_claim_key text, p_claim_generation bigint', 'b7c4c1492de3c12271d1ac318244b057'),
    ('research_lab_routing_mark_budget_uncertain_v2', 'p_event_key text, p_reservation_id text, p_claim_key text, p_claim_generation bigint, p_claim_nonce text, p_event_doc jsonb', '5cd3a04fccca343ebded930722770f60'),
    ('research_lab_routing_mark_budget_uncertain_v3', 'p_event_key text, p_reservation_id text, p_claim_key text, p_claim_generation bigint, p_event_doc jsonb', 'b0d57f82399536192c5771dd85fbffc4'),
    ('research_lab_routing_promote_v2', 'p_reference_hash text, p_experiment_hash text, p_evaluation_receipt_id text, p_evaluation_hash text, p_selected_variant_id text, p_reconciliation_doc jsonb, p_event_hash text, p_event_doc jsonb', '3ee9af525a78cc90c5eb425b88da22f1'),
    ('research_lab_routing_promote_v3', 'p_reference_hash text, p_experiment_hash text, p_evaluation_receipt_id text, p_evaluation_hash text, p_selected_variant_id text, p_reconciliation_doc jsonb, p_event_hash text, p_event_doc jsonb', 'bef9cb7f8ca0f1fcd400b115602f6985'),
    ('research_lab_routing_recover_budget_v2', 'p_event_key text, p_reservation_id text, p_claim_key text, p_claim_generation bigint, p_claim_nonce text, p_event_doc jsonb', 'a5d4974b28431c2672d863ebccb7fdf5'),
    ('research_lab_routing_recover_budget_v3', 'p_event_key text, p_reservation_id text, p_claim_key text, p_claim_generation bigint, p_event_doc jsonb', 'b17737e4a177b6586449dcab70846478'),
    ('research_lab_routing_recover_claim_v2', 'p_experiment_hash text, p_recovery_key text, p_worker_ref text, p_recovery_doc jsonb, p_event_hash text, p_event_doc jsonb', '2268c3056bc4a98ff149c226213369e1'),
    ('research_lab_routing_recover_claim_v3', 'p_experiment_hash text, p_recovery_key text, p_worker_ref text, p_recovery_doc jsonb, p_event_hash text, p_event_doc jsonb', '80a2642776a0bdd6155458d1eafd2831'),
    ('research_lab_routing_reject_secret_doc_v2', 'p_doc jsonb, p_name text', '68e3df3d01d3f5f924f5f65cfb52e944'),
    ('research_lab_routing_renew_claim_v2', 'p_heartbeat_key text, p_experiment_hash text, p_claim_key text, p_claim_generation bigint, p_claim_nonce text, p_lease_seconds integer, p_heartbeat_doc jsonb', 'a7da3fb08cc81c71fec32cdf961eb3d1'),
    ('research_lab_routing_renew_claim_v3', 'p_heartbeat_key text, p_experiment_hash text, p_claim_key text, p_claim_generation bigint, p_lease_seconds integer, p_heartbeat_doc jsonb', '134332fbb13d1627430afecbe0d603a5'),
    ('research_lab_routing_renew_execution_request_lease_v2', 'p_request_hash text, p_worker_ref text, p_lease_hash text, p_lease_generation bigint, p_lease_seconds integer', '967fd848ce5173b56c530c45fd292c37'),
    ('research_lab_routing_request_execution_v2', 'p_request_hash text, p_experiment_hash text, p_request_doc jsonb', 'c9f4819fa7c3a3e93f0074e8faba7fcc'),
    ('research_lab_routing_reserve_budget_v2', 'p_event_key text, p_reservation_id text, p_experiment_hash text, p_binding_id text, p_claim_key text, p_claim_generation bigint, p_claim_nonce text, p_credit_microunits bigint, p_lease_seconds integer, p_event_doc jsonb', '82184de50bb38a5a1d16abca247aaaf5'),
    ('research_lab_routing_reserve_budget_v3', 'p_event_key text, p_reservation_id text, p_experiment_hash text, p_binding_id text, p_claim_key text, p_claim_generation bigint, p_credit_microunits bigint, p_lease_seconds integer, p_event_doc jsonb', '95b4441e63f7e7a6d24529564eb3944b'),
    ('research_lab_routing_settle_budget_v2', 'p_event_key text, p_reservation_id text, p_attempt_key text, p_claim_key text, p_claim_generation bigint, p_claim_nonce text, p_event_doc jsonb', '481044372a65371c9aec0bf503dc77c3'),
    ('research_lab_routing_settle_budget_v3', 'p_event_key text, p_reservation_id text, p_attempt_key text, p_claim_key text, p_claim_generation bigint, p_event_doc jsonb', '50553cbf4276ff6f5d5d8ece14a23eab'),
    ('research_lab_routing_submit_experiment_v2', 'p_experiment_hash text, p_experiment_id text, p_spec_doc jsonb, p_receipt_execution_mode text, p_allow_live_credit_spend boolean, p_event_hash text, p_event_doc jsonb, p_execution_envelope_hash text, p_execution_envelope_doc jsonb', '1d22ac2f4b8748d198024fe623c4a7e5'),
    ('select_research_lab_autoresearch_tree_final', 'requested_tree_id text, requested_node_id text, requested_selection_hash text, requested_selection_doc jsonb, requested_previous_event_hash text, requested_event_hash text', '72ec123131c02d5abcfd2fc44dbb21d0'),
    ('transition_research_lab_autoresearch_operation', 'requested_logical_operation_id text, requested_tree_id text, requested_node_id text, requested_operation_kind text, requested_operation_status text, requested_request_hash text, requested_result_hash text, requested_settled_cost_microusd bigint, requested_provider_call_count integer, requested_settlement_doc jsonb, requested_transition_hash text, expected_current_status text', 'ccfdb777f2b1503062191ddfeea65afb');

DO $retirement_guard$
DECLARE
    target RECORD;
    has_rows BOOLEAN;
    unexpected_name TEXT;
    protected_tables CONSTANT TEXT[] := ARRAY[
        'research_lab_attested_ancestry_activations_v2',
        'research_lab_attested_ancestry_checkpoints_v2',
        'research_lab_attested_artifact_links_v2',
        'research_lab_attested_boot_identities_v2',
        'research_lab_attested_business_artifact_links_v2',
        'research_lab_attested_execution_receipts_v2',
        'research_lab_attested_execution_results_v2',
        'research_lab_attested_host_operations_v2',
        'research_lab_attested_receipt_edges_v2',
        'research_lab_attested_receipt_transport_v2',
        'research_lab_attested_transport_attempts_v2'
    ];
BEGIN
    IF (SELECT count(*) FROM _retire_empty_research_lab_tables) <> 33 THEN
        RAISE EXCEPTION 'retirement allowlist must contain exactly 33 tables';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM _retire_empty_research_lab_tables
        WHERE table_name ILIKE '%source_add%'
           OR table_name = ANY (protected_tables)
    ) THEN
        RAISE EXCEPTION 'retirement allowlist intersects SOURCE_ADD or shared v2 attestation history';
    END IF;

    FOR target IN
        SELECT allowlist.table_name, relation.relkind
        FROM _retire_empty_research_lab_tables AS allowlist
        LEFT JOIN pg_catalog.pg_namespace AS namespace
          ON namespace.nspname = 'public'
        LEFT JOIN pg_catalog.pg_class AS relation
          ON relation.relnamespace = namespace.oid
         AND relation.relname = allowlist.table_name
        ORDER BY allowlist.table_name
    LOOP
        IF target.relkind IS NULL THEN
            CONTINUE;
        END IF;
        IF target.relkind NOT IN ('r', 'p') THEN
            RAISE EXCEPTION 'refusing to retire public.% because it is not a table', target.table_name;
        END IF;
        EXECUTE format('LOCK TABLE public.%I IN ACCESS EXCLUSIVE MODE', target.table_name);
        EXECUTE format('SELECT EXISTS (SELECT FROM public.%I LIMIT 1)', target.table_name)
           INTO has_rows;
        IF has_rows THEN
            RAISE EXCEPTION 'refusing to retire nonempty table public.%', target.table_name;
        END IF;
    END LOOP;

    SELECT format('%I.%I', view_schema, view_name)
      INTO unexpected_name
      FROM (
          SELECT schemaname AS view_schema, viewname AS view_name, definition
          FROM pg_catalog.pg_views
          UNION ALL
          SELECT schemaname, matviewname, definition
          FROM pg_catalog.pg_matviews
      ) AS candidate_view
     WHERE view_schema <> 'information_schema'
       AND view_schema !~ '^pg_'
       AND (
           EXISTS (
               SELECT 1
               FROM _retire_empty_research_lab_tables AS candidate
               WHERE candidate_view.definition ~* ('\m' || candidate.table_name || '\M')
           )
           OR EXISTS (
               SELECT 1
               FROM _retire_empty_research_lab_views AS retired_view
               WHERE candidate_view.definition ~* ('\m' || retired_view.view_name || '\M')
           )
           OR EXISTS (
               SELECT 1
               FROM _retire_empty_research_lab_routines AS retired_routine
               WHERE candidate_view.definition ~* ('\m' || retired_routine.routine_name || '\s*\(')
           )
       )
       AND NOT (
           view_schema = 'public'
           AND EXISTS (
               SELECT 1
               FROM _retire_empty_research_lab_views AS allowed
               WHERE allowed.view_name = candidate_view.view_name
           )
       )
     LIMIT 1;
    IF unexpected_name IS NOT NULL THEN
        RAISE EXCEPTION 'unexpected view depends on retirement table: %', unexpected_name;
    END IF;

    SELECT format('%I.%I(%s)', namespace.nspname, routine.proname,
                  pg_catalog.pg_get_function_identity_arguments(routine.oid))
      INTO unexpected_name
      FROM pg_catalog.pg_proc AS routine
      JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = routine.pronamespace
     WHERE namespace.nspname <> 'information_schema'
       AND namespace.nspname !~ '^pg_'
       AND routine.prokind IN ('f', 'p')
       AND (
           EXISTS (
               SELECT 1
               FROM _retire_empty_research_lab_tables AS candidate
               WHERE pg_catalog.pg_get_functiondef(routine.oid) ~* ('\m' || candidate.table_name || '\M')
           )
           OR EXISTS (
               SELECT 1
               FROM _retire_empty_research_lab_views AS retired_view
               WHERE pg_catalog.pg_get_functiondef(routine.oid) ~* ('\m' || retired_view.view_name || '\M')
           )
           OR EXISTS (
               SELECT 1
               FROM _retire_empty_research_lab_routines AS retired
               WHERE pg_catalog.pg_get_functiondef(routine.oid) ~* ('\m' || retired.routine_name || '\s*\(')
           )
       )
       AND NOT (
           namespace.nspname = 'public'
           AND EXISTS (
               SELECT 1
               FROM _retire_empty_research_lab_routines AS allowed
               WHERE allowed.routine_name = routine.proname
                 AND allowed.identity_arguments = pg_catalog.pg_get_function_identity_arguments(routine.oid)
           )
       )
     LIMIT 1;
    IF unexpected_name IS NOT NULL THEN
        RAISE EXCEPTION 'unexpected routine depends on retirement closure: %', unexpected_name;
    END IF;

    SELECT format('%I on public.%I', trigger.tgname, relation.relname)
      INTO unexpected_name
      FROM pg_catalog.pg_trigger AS trigger
      JOIN pg_catalog.pg_class AS relation ON relation.oid = trigger.tgrelid
      JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
      JOIN pg_catalog.pg_proc AS routine ON routine.oid = trigger.tgfoid
      JOIN _retire_empty_research_lab_routines AS retired
        ON retired.routine_name = routine.proname
       AND retired.identity_arguments = pg_catalog.pg_get_function_identity_arguments(routine.oid)
     WHERE namespace.nspname = 'public'
       AND NOT trigger.tgisinternal
       AND NOT EXISTS (
           SELECT 1
           FROM _retire_empty_research_lab_tables AS candidate
           WHERE candidate.table_name = relation.relname
       )
     LIMIT 1;
    IF unexpected_name IS NOT NULL THEN
        RAISE EXCEPTION 'retirement routine is used by a retained trigger: %', unexpected_name;
    END IF;

    SELECT format('%I.%I(%s)', namespace.nspname, routine.proname,
                  pg_catalog.pg_get_function_identity_arguments(routine.oid))
      INTO unexpected_name
      FROM pg_catalog.pg_proc AS routine
      JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = routine.pronamespace
      JOIN _retire_empty_research_lab_routines AS allowed
        ON allowed.routine_name = routine.proname
       AND allowed.identity_arguments = pg_catalog.pg_get_function_identity_arguments(routine.oid)
     WHERE namespace.nspname = 'public'
       AND pg_catalog.md5(routine.prosrc) <> allowed.expected_body_md5
     LIMIT 1;
    IF unexpected_name IS NOT NULL THEN
        RAISE EXCEPTION 'retirement routine changed after review: %', unexpected_name;
    END IF;

    IF to_regclass('cron.job') IS NOT NULL THEN
        EXECUTE $cron$
            SELECT jobname
            FROM cron.job
            WHERE EXISTS (
                SELECT 1
                FROM _retire_empty_research_lab_tables AS candidate
                WHERE command ~* ('\m' || candidate.table_name || '\M')
            )
               OR EXISTS (
                SELECT 1
                FROM _retire_empty_research_lab_views AS candidate
                WHERE command ~* ('\m' || candidate.view_name || '\M')
            )
               OR EXISTS (
                SELECT 1
                FROM _retire_empty_research_lab_routines AS candidate
                WHERE command ~* ('\m' || candidate.routine_name || '\s*\(')
            )
            LIMIT 1
        $cron$ INTO unexpected_name;
        IF unexpected_name IS NOT NULL THEN
            RAISE EXCEPTION 'scheduled job depends on retirement closure: %', unexpected_name;
        END IF;
    END IF;
END
$retirement_guard$;

DO $retirement_drop$
DECLARE
    drop_targets TEXT;
BEGIN
    SELECT string_agg(format('public.%I', view_name), ', ' ORDER BY view_name)
      INTO drop_targets
      FROM _retire_empty_research_lab_views
     WHERE to_regclass(format('public.%I', view_name)) IS NOT NULL;
    IF drop_targets IS NOT NULL THEN
        EXECUTE 'DROP VIEW IF EXISTS ' || drop_targets;
    END IF;

    FOR drop_targets IN
        SELECT format(
                   'DROP TRIGGER %I ON public.%I',
                   trigger.tgname,
                   relation.relname
               )
        FROM pg_catalog.pg_trigger AS trigger
        JOIN pg_catalog.pg_class AS relation ON relation.oid = trigger.tgrelid
        JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
        JOIN _retire_empty_research_lab_tables AS candidate
          ON candidate.table_name = relation.relname
        WHERE namespace.nspname = 'public'
          AND NOT trigger.tgisinternal
        ORDER BY relation.relname, trigger.tgname
    LOOP
        EXECUTE drop_targets;
    END LOOP;

    SELECT string_agg(format('public.%I(%s)', routine.proname,
                             pg_catalog.pg_get_function_identity_arguments(routine.oid)),
                      ', ' ORDER BY routine.proname, pg_catalog.pg_get_function_identity_arguments(routine.oid))
      INTO drop_targets
      FROM pg_catalog.pg_proc AS routine
      JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = routine.pronamespace
      JOIN _retire_empty_research_lab_routines AS allowed
        ON allowed.routine_name = routine.proname
       AND allowed.identity_arguments = pg_catalog.pg_get_function_identity_arguments(routine.oid)
     WHERE namespace.nspname = 'public';
    IF drop_targets IS NOT NULL THEN
        EXECUTE 'DROP FUNCTION IF EXISTS ' || drop_targets;
    END IF;

    SELECT string_agg(format('public.%I', table_name), ', ' ORDER BY table_name)
      INTO drop_targets
      FROM _retire_empty_research_lab_tables
     WHERE to_regclass(format('public.%I', table_name)) IS NOT NULL;
    IF drop_targets IS NOT NULL THEN
        EXECUTE 'DROP TABLE IF EXISTS ' || drop_targets;
    END IF;
END
$retirement_drop$;

DO $retirement_verify$
DECLARE
    survivor TEXT;
BEGIN
    SELECT table_name
      INTO survivor
      FROM _retire_empty_research_lab_tables
     WHERE to_regclass(format('public.%I', table_name)) IS NOT NULL
     LIMIT 1;
    IF survivor IS NOT NULL THEN
        RAISE EXCEPTION 'retirement table survived: public.%', survivor;
    END IF;
END
$retirement_verify$;

NOTIFY pgrst, 'reload schema';

COMMIT;
