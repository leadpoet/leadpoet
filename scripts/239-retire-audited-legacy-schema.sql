-- Migration 239: retire the exact 65-table legacy schema approved after
-- the complete 95-table production audit. Preserve all 30 retained public
-- tables, SOURCE_ADD history, the provider cache, and current Arena behavior.

BEGIN;

SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

CREATE TEMP TABLE _retire_239_tables (
    table_name TEXT PRIMARY KEY
) ON COMMIT DROP;
INSERT INTO _retire_239_tables (table_name) VALUES
    ('banned_hotkeys'),
    ('contributor_attestations'),
    ('dashboard_miner_stats'),
    ('dashboard_precalc'),
    ('engine_trace_mappings'),
    ('epoch_audit_logs'),
    ('evidence_bundles'),
    ('execution_traces'),
    ('merkle_checkpoints'),
    ('miner_rate_limits'),
    ('ops_alert_current'),
    ('ops_alert_delivery_events'),
    ('ops_alert_events'),
    ('ops_alert_monitor_state'),
    ('ops_validator_registry'),
    ('qualification_baselines'),
    ('qualification_model_rate_limits'),
    ('qualification_models'),
    ('qualification_payments'),
    ('research_evaluation_score_bundle_events'),
    ('research_evaluation_score_bundles'),
    ('research_island_participation_snapshots'),
    ('research_lab_allocator_selection_records'),
    ('research_lab_attested_artifact_links'),
    ('research_lab_attested_execution_receipts'),
    ('research_lab_candidate_artifacts'),
    ('research_lab_candidate_claim'),
    ('research_lab_candidate_evaluation_events'),
    ('research_lab_candidate_promotion_events'),
    ('research_lab_company_label_examples'),
    ('research_lab_corpus_complete'),
    ('research_lab_gateway_control_events'),
    ('research_lab_inner_loop_activation_events'),
    ('research_lab_maintenance_lease'),
    ('research_lab_openrouter_key_refs'),
    ('research_lab_openrouter_privacy_proof_events'),
    ('research_lab_private_model_benchmark_bundles'),
    ('research_lab_private_model_benchmark_events'),
    ('research_lab_private_model_version_events'),
    ('research_lab_private_model_versions'),
    ('research_lab_private_repo_commit_events'),
    ('research_lab_provider_credential_envelopes_v2'),
    ('research_lab_provider_registry'),
    ('research_lab_provider_usage_ledger'),
    ('research_lab_public_benchmark_report_events'),
    ('research_lab_public_benchmark_reports'),
    ('research_lab_rejected_companies'),
    ('research_lab_results_ledger'),
    ('research_lab_rolling_icp_windows'),
    ('research_lab_score_calibration'),
    ('research_lab_shadow_monitor_windows'),
    ('research_lab_signed_transition_commands_v2'),
    ('research_lab_trace_pointer_quarantine'),
    ('research_loop_receipt_events'),
    ('research_loop_receipts'),
    ('research_loop_run_claim'),
    ('research_loop_run_queue_events'),
    ('research_loop_start_credit_events'),
    ('research_loop_start_payments'),
    ('research_loop_ticket_events'),
    ('research_loop_tickets'),
    ('research_trajectories'),
    ('research_trajectory_events'),
    ('validator_attestations'),
    ('validator_sourcing_epoch_inputs_v2');

CREATE TEMP TABLE _retain_239_relations (
    schema_name TEXT NOT NULL,
    table_name TEXT NOT NULL,
    PRIMARY KEY (schema_name, table_name)
) ON COMMIT DROP;
INSERT INTO _retain_239_relations (schema_name, table_name) VALUES
    ('public', 'lab_arena_accepted_weight_states'),
    ('public', 'lab_arena_chain_outcomes'),
    ('public', 'lab_arena_company_judgment_reservations'),
    ('public', 'lab_arena_company_judgments'),
    ('public', 'lab_arena_judgment_cache'),
    ('public', 'lab_arena_ledger'),
    ('public', 'lab_arena_restart_claim_control'),
    ('public', 'lab_arena_rounds'),
    ('public', 'lab_arena_runs'),
    ('public', 'lab_arena_submission_credentials'),
    ('public', 'lab_arena_submissions'),
    ('public', 'published_weight_bundles'),
    ('public', 'qualification_private_icp_sets'),
    ('public', 'research_lab_attested_ancestry_activations_v2'),
    ('public', 'research_lab_attested_ancestry_checkpoints_v2'),
    ('public', 'research_lab_attested_artifact_links_v2'),
    ('public', 'research_lab_attested_boot_identities_v2'),
    ('public', 'research_lab_attested_business_artifact_links_v2'),
    ('public', 'research_lab_attested_execution_receipts_v2'),
    ('public', 'research_lab_attested_execution_results_v2'),
    ('public', 'research_lab_attested_host_operations_v2'),
    ('public', 'research_lab_attested_receipt_edges_v2'),
    ('public', 'research_lab_attested_receipt_transport_v2'),
    ('public', 'research_lab_attested_transport_attempts_v2'),
    ('public', 'research_lab_provider_evidence_cache_v2'),
    ('public', 'research_lab_stateful_subnet_epoch_boundaries_v1'),
    ('public', 'research_lab_stateful_subnet_epoch_candidates_v1'),
    ('public', 'research_lab_stateful_subnet_epoch_cutover_state_v1'),
    ('public', 'research_lab_stateful_subnet_epoch_cutovers_v1'),
    ('public', 'research_lab_stateful_subnet_epoch_snapshots_v1'),
    ('source_add_history', 'archive_manifests'),
    ('source_add_history', 'legacy_audit_rows');

CREATE TEMP TABLE _retire_239_views (
    view_name TEXT PRIMARY KEY,
    expected_definition_md5 TEXT NOT NULL
) ON COMMIT DROP;
INSERT INTO _retire_239_views (view_name, expected_definition_md5) VALUES
    ('qualification_champion_history', '6ab7b4cc4da587511f2ab22ea94ef142'),
    ('qualification_current_champion', '4d1364c45e5bf4030576e95a89071e15'),
    ('qualification_leaderboard', 'f5b1454397fb53f5ba27210740fbe5f3'),
    ('research_evaluation_score_bundle_current', '87172e4bc39659df87447ab26c85dfb7'),
    ('research_lab_candidate_evaluation_current', 'c0f67c08c554e0282e4ab5a7e0092400'),
    ('research_lab_daily_noise_budget_report_current', '8740672af869c6485ae600a1b0a94ea5'),
    ('research_lab_gateway_control_current', 'cf73a2c9e319e7aadfe2c7194d9e5140'),
    ('research_lab_icp_churn_reversal_report', '41588d237bf8ec5e83d8b1b938c9914c'),
    ('research_lab_inner_loop_activation_current', '8e0b1c474eaa31f021fe5fcf8cfc44bd'),
    ('research_lab_private_model_benchmark_current', '034f590c024e7d75401172feb738d872'),
    ('research_lab_private_model_version_current', '84e5a501e77208129cd7bd72ca79f0de'),
    ('research_lab_provider_registry_current', '82b1304829a30c9d19a92192cab18406'),
    ('research_lab_public_benchmark_report_current', 'd588c07cb98ba1d9cb0734b644d24ff2'),
    ('research_lab_sales_llm_corpus_metadata_current', '1027949a317f95099ceb0130a0e165a4'),
    ('research_lab_unpaid_ticket_expiry_candidates', 'a07f3218b8fa123b20505be61991ec17'),
    ('research_loop_available_credits', '81f3a6eead897db7714f83390ee7e0b3'),
    ('research_loop_receipt_current', '9aac7a3855429c48078abfbde9715a13'),
    ('research_loop_run_queue_current', '068d476207fae423fd832fa26cb651f5'),
    ('research_loop_start_credit_current', 'e4ce99017d8cf442e561bc5af0a99370'),
    ('research_loop_ticket_current', '1a5c112503fd98e77520bb3845abccdd');

CREATE TEMP TABLE _retire_239_routines (
    routine_name TEXT NOT NULL,
    identity_arguments TEXT NOT NULL,
    expected_body_md5 TEXT NOT NULL,
    PRIMARY KEY (routine_name, identity_arguments)
) ON COMMIT DROP;
INSERT INTO _retire_239_routines (
    routine_name, identity_arguments, expected_body_md5
) VALUES
    ('append_research_lab_inner_loop_activation_event', 'requested_event_type text, requested_phase text, requested_run_id uuid, requested_evidence_doc jsonb, requested_event_hash text, expected_current_phase text', '8b35638a95ce9e842ccbaa29158c8d8e'),
    ('claim_next_research_lab_candidate', 'p_holder_ref text, p_ttl_seconds integer, p_baseline_not_ready_retry_seconds integer, p_retryable_failure_retry_seconds integer', '3b5da998facce249dcdb8301c7082c94'),
    ('claim_next_research_loop_run', 'p_holder_ref text, p_ttl_seconds integer, p_allowed_run_ids uuid[]', '45b2442e66563bba10fd64227432cbc2'),
    ('claim_ops_alert_monitor_lease', 'p_monitor_id text, p_owner text, p_lease_seconds integer', '2515063d6924d0cb1351b667607bbb02'),
    ('guard_research_lab_candidate_artifact_image_build', '', '60d9e0b6b3627f86823f4476d0fb7a56'),
    ('guard_research_lab_candidate_claim', '', '4a6460ab19a7aaffce2d7de814241382'),
    ('guard_research_lab_credit_consume', '', '141abbcfc0b4e77962dccbbb0074b388'),
    ('guard_research_lab_loop_start_payment_expiry', '', 'c2393ca895b7bb6462efea183e295a0f'),
    ('guard_research_lab_one_active_private_model_version', '', 'cafa696426fa2264ca35794c24c5778f'),
    ('guard_research_lab_queue_capacity', '', '9e475e80d334c00b02c7f63cb54f6d6c'),
    ('guard_research_lab_run_claim', '', 'a3ddb6d71c21e81931606daf65a0fef9'),
    ('guard_research_lab_ticket_lifecycle', '', 'aa17d88503a87d9080eef28aaee60c95'),
    ('insert_research_lab_provider_usage_ledger_rows', 'rows jsonb', '1af3416168232e8263e6fa6f28ce88d8'),
    ('nuke_banned_hotkey_models', '', '4c698726d019e94db88e88f3dde0e069'),
    ('prevent_research_lab_allocator_selection_mutation', '', '91e19db2ff3affe32c40b7ea89d8e37e'),
    ('prevent_research_lab_append_only_mutation', '', '0a3eff15a5f2d0d8b6c1db5c08d9ef9d'),
    ('prevent_research_lab_attested_receipt_mutation', '', '5286a440aabbb6dbb41ba7216b40791c'),
    ('prevent_research_lab_provider_registry_mutation', '', '50faa7809fb0d4fdfcc09ec25c78cd84'),
    ('prevent_research_lab_provider_usage_mutation', '', '6bba2630db06781572f73a7552824075'),
    ('prevent_research_lab_score_calibration_mutation', '', '822608494ca17df3e0feabec559556f0'),
    ('prevent_research_trajectory_event_mutation', '', 'e9c25acc063c612f8273b1141cf93be4'),
    ('refresh_qualification_model_scores', '', '3ed889b28f0062b184f9e5474e97bd36'),
    ('research_lab_acquire_maintenance_lease', 'p_lease_name text, p_holder_ref text, p_ttl_seconds integer', 'edc3c97cec9d7533d4a816196a14d30b'),
    ('research_lab_missing_trajectory_ids', 'candidate_ids uuid[]', '1a277e6fa46df2c37873d01f133f800c'),
    ('research_lab_next_unprojected_terminal_runs', 'p_limit integer, p_newest_first boolean', '4ac01216122dc68413e22a2a1d9d7184'),
    ('research_lab_private_benchmark_schema_contract_v1', '', 'd221bd96e54322ce243783f133266364'),
    ('research_lab_private_model_lineage_generation', '', '5697937ce37043aa0040812aac019160'),
    ('research_lab_terminal_runs_missing_traces', 'p_limit integer, p_newest_first boolean', '40c3345f7c71f59727111cbc58f15847'),
    ('research_lab_ticket_has_unpaid_lifecycle_evidence', 'target_ticket_id uuid', '12321f60dfacab18ead8342c9656a9e8'),
    ('reset_daily_stats', '', '2b22bd66d5c4a47ad0fba040361a8e4f'),
    ('reset_miner_rate_limits', '', 'd06d34414b1138ee9d23f35aeb545ce8'),
    ('reset_miner_rate_limits_daily', '', '90e9a7ccc90e0056ea2974abdd56a6e7'),
    ('resume_research_lab_credit_blocked_run_v1', 'p_run_id uuid, p_ticket_id uuid, p_expected_event_seq integer, p_expected_event_hash text, p_event_id uuid, p_anchored_hash text, p_queue_priority integer, p_worker_ref text, p_reason text, p_event_doc jsonb', 'd16171dfe82f41ef872e5f7eacb12b9b'),
    ('update_qual_rate_limits_updated_at', '', '06bcf30ac3d0a7f279a54cbf228a7bec'),
    ('update_qualification_model_status', '', '0d6bc1311098839e4b4204038863d12f'),
    ('update_qualification_timestamp', '', '06bcf30ac3d0a7f279a54cbf228a7bec'),
    ('update_updated_at_column', '', 'da5ac28a58c8b4bb30209bf0d3d7082c');

CREATE TEMP TABLE _retire_239_triggers (
    table_name TEXT NOT NULL,
    trigger_name TEXT NOT NULL,
    expected_definition_md5 TEXT NOT NULL,
    PRIMARY KEY (table_name, trigger_name)
) ON COMMIT DROP;
INSERT INTO _retire_239_triggers (
    table_name, trigger_name, expected_definition_md5
) VALUES
    ('banned_hotkeys', 'auto_nuke_on_ban', '9e82b3a21e8b2f25991df95f8cf38f8f'),
    ('contributor_attestations', 'update_contributor_attestations_updated_at', '20fcd5e0f941047944ed879d264b5172'),
    ('epoch_audit_logs', 'enforce_research_lab_stateful_epoch_fence_v1', '2cfcfc021b9a39b602d94706bb819561'),
    ('qualification_model_rate_limits', 'qual_rate_limits_updated_at', '16bd34d94f4c925b59e12e495165328d'),
    ('qualification_models', 'tr_qm_updated', '40b1f7911abe09a4e5673473dd071508'),
    ('research_evaluation_score_bundle_events', 'prevent_research_eval_score_bundle_events_mutation', 'de498433ea1c85ea8168254a00064ce2'),
    ('research_evaluation_score_bundles', 'enforce_research_lab_stateful_epoch_fence_v1', '42d51ca82d1426ef788616e81211003a'),
    ('research_evaluation_score_bundles', 'prevent_research_eval_score_bundles_mutation', '77ee1543fa579c782a6dea818e6567ce'),
    ('research_island_participation_snapshots', 'prevent_research_island_participation_snapshots_mutation', 'b7d964dd4004d9fd573116466dde7257'),
    ('research_lab_allocator_selection_records', 'prevent_research_lab_allocator_selection_mutation', '79a0d5692504d586b5fb45dc6b7853c5'),
    ('research_lab_attested_artifact_links', 'prevent_research_lab_attested_artifact_links_mutation', 'e231d67da3261789af10ccd8e997e367'),
    ('research_lab_attested_execution_receipts', 'enforce_research_lab_stateful_epoch_fence_v1', '64289e3461ed0392eeff4cfc34cf4ece'),
    ('research_lab_attested_execution_receipts', 'prevent_research_lab_attested_execution_receipts_mutation', 'ae15ebded5901253f84fe96649a80ceb'),
    ('research_lab_candidate_artifacts', 'guard_research_lab_candidate_artifact_image_build_insert', '4404e64f098ea030c52f712a38931fd8'),
    ('research_lab_candidate_artifacts', 'prevent_research_lab_candidate_artifacts_mutation', 'b96088c3e07aebbd0c801f574873a642'),
    ('research_lab_candidate_evaluation_events', 'guard_research_lab_candidate_claim_insert', 'f9f63682bb668f52db0026820862db8a'),
    ('research_lab_candidate_evaluation_events', 'prevent_research_lab_candidate_eval_events_mutation', '477f7a2edaf25ec10aa4809f3803efb8'),
    ('research_lab_candidate_promotion_events', 'prevent_research_lab_candidate_promotion_events_mutation', 'c979145c52945a4070f2876904784770'),
    ('research_lab_company_label_examples', 'prevent_research_lab_company_label_examples_mutation', '61b18acc78b874cb123e31ffefb42cbc'),
    ('research_lab_gateway_control_events', 'prevent_research_lab_gateway_control_events_mutation', '4497d0398b49ad78a5a024ae16e594e1'),
    ('research_lab_inner_loop_activation_events', 'prevent_research_lab_inner_loop_activation_mutation', '66605b32ab707b436b4b9e57042ad876'),
    ('research_lab_openrouter_key_refs', 'prevent_research_lab_openrouter_key_refs_mutation', '0e008a1b8e8559d958b587a8aa3e4e70'),
    ('research_lab_openrouter_privacy_proof_events', 'prevent_research_lab_openrouter_privacy_proof_events_mutation', 'e55679d3d192b12cbd7a1775d9ecf70b'),
    ('research_lab_private_model_benchmark_bundles', 'enforce_research_lab_stateful_epoch_fence_v1', '76bcf417c070fbc6c56200a4bf8d1e5c'),
    ('research_lab_private_model_benchmark_bundles', 'prevent_research_lab_private_model_benchmark_bundles_mutation', 'f218b0fb95eeba483cd87863de8a921e'),
    ('research_lab_private_model_benchmark_events', 'prevent_research_lab_private_model_benchmark_events_mutation', 'd10b24eaee0b3e6a74305d030148b1b0'),
    ('research_lab_private_model_version_events', 'guard_research_lab_one_active_version_insert', '1343805bd1879ae1074328d804b1daeb'),
    ('research_lab_private_model_version_events', 'prevent_research_lab_private_model_version_events_mutation', '0ee44e39f8688828acb4057959ebd211'),
    ('research_lab_private_model_versions', 'prevent_research_lab_private_model_versions_mutation', '37948bc64d7286a2401a929cff267e72'),
    ('research_lab_private_repo_commit_events', 'prevent_research_lab_private_repo_commit_events_mutation', '8c801d344252b4ad37f068d5cb5bbf89'),
    ('research_lab_provider_credential_envelopes_v2', 'prevent_research_lab_provider_credential_envelopes_v2_mutation', '0d254d6fed605baea0c4a78d3a0cf1ff'),
    ('research_lab_provider_registry', 'trg_research_lab_provider_registry_no_mutation', '0fd69cc5be6d5d15c7cc9c59ee181857'),
    ('research_lab_provider_usage_ledger', 'trg_research_lab_provider_usage_no_mutation', '6f0ef2643be763f9ea8f87d26168dcdd'),
    ('research_lab_public_benchmark_report_events', 'prevent_research_lab_public_benchmark_report_events_mutation', 'c57b94b1db2355b59bbb4fbd63c8cd89'),
    ('research_lab_public_benchmark_reports', 'prevent_research_lab_public_benchmark_reports_mutation', '44d94f837e3c9b9303ac8175a05f2772'),
    ('research_lab_rolling_icp_windows', 'prevent_research_lab_rolling_icp_windows_mutation', '3122882cc0d19b3b5935f5bee1999717'),
    ('research_lab_score_calibration', 'prevent_research_lab_score_calibration_mutation', 'd22b244dc1be3868ca23fba15ed02bed'),
    ('research_lab_signed_transition_commands_v2', 'prevent_research_lab_signed_transition_commands_v2_mutation', '3d4795459634ce5b28118d8dff8b577a'),
    ('research_loop_receipt_events', 'prevent_research_loop_receipt_events_mutation', '42ba1b0f1236376a4dc356934f9c93ec'),
    ('research_loop_receipts', 'prevent_research_loop_receipts_mutation', '81204426959b8402995ecd58a9ab5bfd'),
    ('research_loop_run_queue_events', 'guard_research_loop_queue_capacity_insert', '9738b943e4098ce768046c1c432fedaf'),
    ('research_loop_run_queue_events', 'guard_research_loop_run_claim_insert', '538ccbdde4a440ddad0a3887907efb1f'),
    ('research_loop_run_queue_events', 'prevent_research_loop_run_queue_events_mutation', '6174f44b8c9427b37a163def88ba3ada'),
    ('research_loop_start_credit_events', 'guard_research_loop_start_credit_consume_insert', '228cf0636c9b93796a5d3818ec5da675'),
    ('research_loop_start_credit_events', 'prevent_research_loop_start_credit_events_mutation', '385fe49d148d93daabff3989d07b9ede'),
    ('research_loop_start_payments', 'guard_research_lab_loop_start_payment_expiry_insert', 'ab1ae4bf0442a83a01be715efd256d10'),
    ('research_loop_start_payments', 'prevent_research_loop_start_payments_mutation', 'd91db7d23f5772e595e8eeb48d46e621'),
    ('research_loop_ticket_events', 'guard_research_lab_ticket_lifecycle_insert', 'e68902193952efff2927e4994f3e05d1'),
    ('research_loop_ticket_events', 'prevent_research_loop_ticket_events_mutation', '28e00910e181b01d3020a0a138a651e7'),
    ('research_loop_tickets', 'prevent_research_loop_tickets_mutation', '11a09768a16a174ecefd9264fb5c9c63'),
    ('research_trajectory_events', 'prevent_research_trajectory_event_mutation', 'c84431183f5d9a09ae9d64b8af869fa6'),
    ('validator_attestations', 'enforce_research_lab_stateful_epoch_fence_v1', '2b46fd38c80fe23d0bd02a5ce327aae7'),
    ('validator_sourcing_epoch_inputs_v2', 'enforce_research_lab_stateful_epoch_fence_v1', 'c70bd8e85b3c26cc6dc95470f6306af3'),
    ('validator_sourcing_epoch_inputs_v2', 'prevent_validator_sourcing_epoch_inputs_v2_mutation', 'b733bc916cb7a0afb1c99992d7d51689');

CREATE TEMP TABLE _retire_239_cron (
    jobid BIGINT PRIMARY KEY,
    jobname TEXT UNIQUE NOT NULL,
    schedule TEXT NOT NULL,
    expected_command_md5 TEXT NOT NULL
) ON COMMIT DROP;
INSERT INTO _retire_239_cron (
    jobid, jobname, schedule, expected_command_md5
) VALUES
    (8, 'reset-daily-stats', '0 5 * * *', '7c11e1d7df370081d38da280ce2f2d1f'),
    (15, 'process-rep-scores', '* * * * *', 'bde11b582626efd03d99fa6a58b39ed5'),
    (47, 'reset-miner-rate-limits-daily', '0 0 * * *', '0c8ce7b8847b7bf6329d24b8ea4dcb8e');

CREATE OR REPLACE FUNCTION pg_temp._retire_239_relation_signature(
    relation_oid OID
) RETURNS TEXT
LANGUAGE sql
STABLE
SET search_path = ''
AS $signature$
SELECT pg_catalog.md5(
    pg_catalog.jsonb_build_object(
        'relkind', relation.relkind,
        'owner', pg_catalog.pg_get_userbyid(relation.relowner),
        'acl', relation.relacl::TEXT,
        'row_security', relation.relrowsecurity,
        'force_row_security', relation.relforcerowsecurity,
        'replica_identity', relation.relreplident,
        'options', relation.reloptions,
        'tablespace', relation.reltablespace,
        'columns', (
            SELECT COALESCE(
                pg_catalog.jsonb_agg(
                    pg_catalog.jsonb_build_object(
                        'number', attribute.attnum,
                        'name', attribute.attname,
                        'type', pg_catalog.format_type(
                            attribute.atttypid, attribute.atttypmod
                        ),
                        'not_null', attribute.attnotnull,
                        'identity', attribute.attidentity,
                        'generated', attribute.attgenerated,
                        'collation', attribute.attcollation,
                        'default', pg_catalog.pg_get_expr(
                            default_row.adbin, default_row.adrelid
                        )
                    ) ORDER BY attribute.attnum
                ),
                '[]'::JSONB
            )
            FROM pg_catalog.pg_attribute AS attribute
            LEFT JOIN pg_catalog.pg_attrdef AS default_row
              ON default_row.adrelid = attribute.attrelid
             AND default_row.adnum = attribute.attnum
            WHERE attribute.attrelid = relation.oid
              AND attribute.attnum > 0
              AND NOT attribute.attisdropped
        ),
        'indexes', (
            SELECT COALESCE(
                pg_catalog.jsonb_agg(
                    pg_catalog.jsonb_build_object(
                        'name', index_relation.relname,
                        'definition', pg_catalog.pg_get_indexdef(
                            index_row.indexrelid
                        ),
                        'valid', index_row.indisvalid,
                        'ready', index_row.indisready,
                        'live', index_row.indislive
                    ) ORDER BY index_relation.relname
                ),
                '[]'::JSONB
            )
            FROM pg_catalog.pg_index AS index_row
            JOIN pg_catalog.pg_class AS index_relation
              ON index_relation.oid = index_row.indexrelid
            WHERE index_row.indrelid = relation.oid
        ),
        'constraints', (
            SELECT COALESCE(
                pg_catalog.jsonb_agg(
                    pg_catalog.jsonb_build_object(
                        'name', constraint_row.conname,
                        'type', constraint_row.contype,
                        'definition', pg_catalog.pg_get_constraintdef(
                            constraint_row.oid, TRUE
                        ),
                        'validated', constraint_row.convalidated,
                        'deferrable', constraint_row.condeferrable,
                        'deferred', constraint_row.condeferred
                    ) ORDER BY constraint_row.conname
                ),
                '[]'::JSONB
            )
            FROM pg_catalog.pg_constraint AS constraint_row
            WHERE constraint_row.conrelid = relation.oid
        ),
        'triggers', (
            SELECT COALESCE(
                pg_catalog.jsonb_agg(
                    pg_catalog.jsonb_build_object(
                        'name', trigger_row.tgname,
                        'enabled', trigger_row.tgenabled,
                        'definition', pg_catalog.pg_get_triggerdef(
                            trigger_row.oid, TRUE
                        )
                    ) ORDER BY trigger_row.tgname
                ),
                '[]'::JSONB
            )
            FROM pg_catalog.pg_trigger AS trigger_row
            WHERE trigger_row.tgrelid = relation.oid
              AND NOT trigger_row.tgisinternal
        ),
        'policies', (
            SELECT COALESCE(
                pg_catalog.jsonb_agg(
                    pg_catalog.jsonb_build_object(
                        'name', policy.polname,
                        'permissive', policy.polpermissive,
                        'command', policy.polcmd,
                        'roles', policy.polroles::TEXT,
                        'using', pg_catalog.pg_get_expr(
                            policy.polqual, policy.polrelid
                        ),
                        'check', pg_catalog.pg_get_expr(
                            policy.polwithcheck, policy.polrelid
                        )
                    ) ORDER BY policy.polname
                ),
                '[]'::JSONB
            )
            FROM pg_catalog.pg_policy AS policy
            WHERE policy.polrelid = relation.oid
        ),
        'publications', (
            SELECT COALESCE(
                pg_catalog.jsonb_agg(
                    publication.pubname ORDER BY publication.pubname
                ),
                '[]'::JSONB
            )
            FROM pg_catalog.pg_publication_rel AS membership
            JOIN pg_catalog.pg_publication AS publication
              ON publication.oid = membership.prpubid
            WHERE membership.prrelid = relation.oid
        )
    )::TEXT
)
FROM pg_catalog.pg_class AS relation
WHERE relation.oid = relation_oid;
$signature$;

DO $preflight$
DECLARE
    expected RECORD;
    actual_md5 TEXT;
    object_oid OID;
    present BOOLEAN;
    unexpected TEXT;
    closure_pattern TEXT;
BEGIN
    IF (SELECT count(*) FROM _retire_239_tables) <> 65 THEN
        RAISE EXCEPTION 'migration 239 table allowlist must contain 65 names';
    END IF;
    IF (SELECT count(*) FROM _retain_239_relations
        WHERE schema_name = 'public') <> 30
       OR (SELECT count(*) FROM _retain_239_relations) <> 32 THEN
        RAISE EXCEPTION 'migration 239 retained relation allowlist differs';
    END IF;
    IF EXISTS (
        SELECT 1 FROM _retire_239_tables AS retired
        JOIN _retain_239_relations AS retained
          ON retained.schema_name = 'public'
         AND retained.table_name = retired.table_name
    ) THEN
        RAISE EXCEPTION 'migration 239 retire and retain allowlists overlap';
    END IF;
    IF (SELECT count(*) FROM _retire_239_views) <> 20
       OR (SELECT count(*) FROM _retire_239_routines) <> 37
       OR (SELECT count(*) FROM _retire_239_triggers) <> 54
       OR (SELECT count(*) FROM _retire_239_cron) <> 3 THEN
        RAISE EXCEPTION 'migration 239 object closure cardinality differs';
    END IF;

    FOR expected IN SELECT * FROM _retain_239_relations LOOP
        SELECT relation.oid
          INTO object_oid
          FROM pg_catalog.pg_class AS relation
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = relation.relnamespace
         WHERE namespace.nspname = expected.schema_name
           AND relation.relname = expected.table_name
           AND relation.relkind IN ('r', 'p');
        IF object_oid IS NULL THEN
            RAISE EXCEPTION 'required retained table %.% is missing',
                expected.schema_name, expected.table_name;
        END IF;
        EXECUTE pg_catalog.format(
            'LOCK TABLE %I.%I IN ACCESS SHARE MODE',
            expected.schema_name, expected.table_name
        );
        object_oid := NULL;
    END LOOP;

    FOR expected IN SELECT * FROM _retire_239_tables LOOP
        SELECT relation.oid
          INTO object_oid
          FROM pg_catalog.pg_class AS relation
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = relation.relnamespace
         WHERE namespace.nspname = 'public'
           AND relation.relname = expected.table_name;
        IF object_oid IS NOT NULL AND (
            SELECT relation.relkind FROM pg_catalog.pg_class AS relation
            WHERE relation.oid = object_oid
        ) NOT IN ('r', 'p') THEN
            RAISE EXCEPTION 'public.% is not a table', expected.table_name;
        END IF;
        object_oid := NULL;
    END LOOP;

    FOR expected IN SELECT * FROM _retire_239_views LOOP
        SELECT relation.oid, pg_catalog.md5(
                   pg_catalog.pg_get_viewdef(relation.oid)
               )
          INTO object_oid, actual_md5
          FROM pg_catalog.pg_class AS relation
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = relation.relnamespace
         WHERE namespace.nspname = 'public'
           AND relation.relname = expected.view_name;
        IF object_oid IS NULL THEN
            CONTINUE;
        END IF;
        IF (SELECT relkind FROM pg_catalog.pg_class WHERE oid = object_oid) <> 'v'
           OR actual_md5 IS DISTINCT FROM expected.expected_definition_md5 THEN
            RAISE EXCEPTION 'reviewed view public.% changed after audit',
                expected.view_name;
        END IF;
        object_oid := NULL;
        actual_md5 := NULL;
    END LOOP;

    FOR expected IN SELECT * FROM _retire_239_routines LOOP
        SELECT routine.oid, pg_catalog.md5(routine.prosrc)
          INTO object_oid, actual_md5
          FROM pg_catalog.pg_proc AS routine
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = routine.pronamespace
         WHERE namespace.nspname = 'public'
           AND routine.proname = expected.routine_name
           AND pg_catalog.pg_get_function_identity_arguments(routine.oid) =
               expected.identity_arguments;
        IF object_oid IS NULL THEN
            CONTINUE;
        END IF;
        IF (SELECT prokind FROM pg_catalog.pg_proc WHERE oid = object_oid) <> 'f'
           OR actual_md5 IS DISTINCT FROM expected.expected_body_md5 THEN
            RAISE EXCEPTION 'reviewed routine public.%(%) changed after audit',
                expected.routine_name, expected.identity_arguments;
        END IF;
        object_oid := NULL;
        actual_md5 := NULL;
    END LOOP;

    FOR expected IN SELECT * FROM _retire_239_triggers LOOP
        SELECT trigger_row.oid,
               pg_catalog.md5(pg_catalog.pg_get_triggerdef(trigger_row.oid))
          INTO object_oid, actual_md5
          FROM pg_catalog.pg_trigger AS trigger_row
          JOIN pg_catalog.pg_class AS relation
            ON relation.oid = trigger_row.tgrelid
          JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = relation.relnamespace
         WHERE namespace.nspname = 'public'
           AND relation.relname = expected.table_name
           AND trigger_row.tgname = expected.trigger_name
           AND NOT trigger_row.tgisinternal;
        IF object_oid IS NULL THEN
            CONTINUE;
        END IF;
        IF actual_md5 IS DISTINCT FROM expected.expected_definition_md5 THEN
            RAISE EXCEPTION 'reviewed trigger %.% changed after audit',
                expected.table_name, expected.trigger_name;
        END IF;
        object_oid := NULL;
        actual_md5 := NULL;
    END LOOP;

    SELECT pg_catalog.format('%I.%I', namespace.nspname, relation.relname)
      INTO unexpected
      FROM pg_catalog.pg_trigger AS trigger_row
      JOIN pg_catalog.pg_class AS relation ON relation.oid = trigger_row.tgrelid
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = relation.relnamespace
      JOIN _retire_239_tables AS retired
        ON namespace.nspname = 'public'
       AND retired.table_name = relation.relname
      LEFT JOIN _retire_239_triggers AS allowed
        ON allowed.table_name = relation.relname
       AND allowed.trigger_name = trigger_row.tgname
     WHERE NOT trigger_row.tgisinternal
       AND allowed.table_name IS NULL
     LIMIT 1;
    IF unexpected IS NOT NULL THEN
        RAISE EXCEPTION 'unexpected trigger exists on a migration 239 table: %',
            unexpected;
    END IF;

    SELECT pg_catalog.format('%I.%I: %s', child_namespace.nspname,
                             child.relname, constraint_row.conname)
      INTO unexpected
      FROM pg_catalog.pg_constraint AS constraint_row
      JOIN pg_catalog.pg_class AS parent ON parent.oid = constraint_row.confrelid
      JOIN pg_catalog.pg_namespace AS parent_namespace
        ON parent_namespace.oid = parent.relnamespace
      JOIN _retire_239_tables AS retired
        ON parent_namespace.nspname = 'public'
       AND retired.table_name = parent.relname
      JOIN pg_catalog.pg_class AS child ON child.oid = constraint_row.conrelid
      JOIN pg_catalog.pg_namespace AS child_namespace
        ON child_namespace.oid = child.relnamespace
      LEFT JOIN _retire_239_tables AS retired_child
        ON child_namespace.nspname = 'public'
       AND retired_child.table_name = child.relname
     WHERE constraint_row.contype = 'f'
       AND retired_child.table_name IS NULL
     LIMIT 1;
    IF unexpected IS NOT NULL THEN
        RAISE EXCEPTION 'retained table has an FK to a migration 239 target: %',
            unexpected;
    END IF;

    SELECT '(^|[^a-zA-Z0-9_])(' || pg_catalog.string_agg(name, '|') ||
           ')([^a-zA-Z0-9_]|$)'
      INTO closure_pattern
      FROM (
          SELECT table_name AS name FROM _retire_239_tables
          UNION SELECT view_name FROM _retire_239_views
          UNION SELECT routine_name FROM _retire_239_routines
      ) AS closure_names;

    SELECT pg_catalog.format('%I.%I', candidate.schemaname, candidate.viewname)
      INTO unexpected
      FROM (
          SELECT schemaname, viewname, definition FROM pg_catalog.pg_views
          UNION ALL
          SELECT schemaname, matviewname, definition FROM pg_catalog.pg_matviews
      ) AS candidate
     WHERE candidate.schemaname <> 'information_schema'
       AND candidate.schemaname !~ '^pg_'
       AND candidate.definition ~* closure_pattern
       AND NOT (
           candidate.schemaname = 'public'
           AND EXISTS (
               SELECT 1 FROM _retire_239_views AS allowed
               WHERE allowed.view_name = candidate.viewname
           )
       )
     LIMIT 1;
    IF unexpected IS NOT NULL THEN
        RAISE EXCEPTION 'unexpected view depends on migration 239 closure: %',
            unexpected;
    END IF;

    SELECT pg_catalog.format(
               '%I.%I(%s)', namespace.nspname, routine.proname,
               pg_catalog.pg_get_function_identity_arguments(routine.oid)
           )
      INTO unexpected
      FROM pg_catalog.pg_proc AS routine
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = routine.pronamespace
     WHERE routine.prokind IN ('f', 'p')
       AND namespace.nspname <> 'information_schema'
       AND namespace.nspname !~ '^pg_'
       AND pg_catalog.pg_get_functiondef(routine.oid) ~* closure_pattern
       AND NOT (
           namespace.nspname = 'public'
           AND (
               routine.proname = 'enforce_research_lab_stateful_epoch_fence_v1'
               OR EXISTS (
                   SELECT 1 FROM _retire_239_routines AS allowed
                   WHERE allowed.routine_name = routine.proname
                     AND allowed.identity_arguments =
                         pg_catalog.pg_get_function_identity_arguments(routine.oid)
               )
           )
       )
     LIMIT 1;
    IF unexpected IS NOT NULL THEN
        RAISE EXCEPTION 'unexpected routine depends on migration 239 closure: %',
            unexpected;
    END IF;

    SELECT pg_catalog.format('%I on %I.%I', trigger_row.tgname,
                             namespace.nspname, relation.relname)
      INTO unexpected
      FROM pg_catalog.pg_trigger AS trigger_row
      JOIN pg_catalog.pg_class AS relation ON relation.oid = trigger_row.tgrelid
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = relation.relnamespace
      JOIN pg_catalog.pg_proc AS routine ON routine.oid = trigger_row.tgfoid
      JOIN _retire_239_routines AS retired
        ON retired.routine_name = routine.proname
       AND retired.identity_arguments =
           pg_catalog.pg_get_function_identity_arguments(routine.oid)
      LEFT JOIN _retire_239_tables AS retired_table
        ON namespace.nspname = 'public'
       AND retired_table.table_name = relation.relname
     WHERE NOT trigger_row.tgisinternal
       AND retired_table.table_name IS NULL
     LIMIT 1;
    IF unexpected IS NOT NULL THEN
        RAISE EXCEPTION 'retired routine is used by a retained trigger: %',
            unexpected;
    END IF;

    IF pg_catalog.to_regclass('cron.job') IS NOT NULL THEN
        FOR expected IN SELECT * FROM _retire_239_cron LOOP
            EXECUTE $cron_actual$
                SELECT pg_catalog.md5(command)
                FROM cron.job
                WHERE jobid = $1
                  AND jobname = $2
                  AND schedule = $3
            $cron_actual$
            INTO actual_md5
            USING expected.jobid, expected.jobname, expected.schedule;
            IF actual_md5 IS NULL THEN
                EXECUTE 'SELECT EXISTS (SELECT 1 FROM cron.job WHERE jobid = $1 OR jobname = $2)'
                INTO present
                USING expected.jobid, expected.jobname;
                IF present THEN
                    RAISE EXCEPTION 'reviewed cron identity changed: %:%',
                        expected.jobid, expected.jobname;
                END IF;
            ELSIF actual_md5 <> expected.expected_command_md5 THEN
                RAISE EXCEPTION 'reviewed cron command changed: %:%',
                    expected.jobid, expected.jobname;
            END IF;
            actual_md5 := NULL;
            present := NULL;
        END LOOP;
    END IF;

    -- These two empty candidate tables are the only candidate FKs into the
    -- protected receipt graph. Refuse any row written after the audit.
    IF pg_catalog.to_regclass(
           'public.research_lab_signed_transition_commands_v2'
       ) IS NOT NULL THEN
        LOCK TABLE public.research_lab_signed_transition_commands_v2
            IN ACCESS EXCLUSIVE MODE;
        EXECUTE 'SELECT EXISTS (SELECT 1 FROM public.research_lab_signed_transition_commands_v2)'
        INTO present;
        IF present THEN
            RAISE EXCEPTION 'signed transition rows appeared after SOURCE_ADD audit';
        END IF;
    END IF;
    IF pg_catalog.to_regclass(
           'public.validator_sourcing_epoch_inputs_v2'
       ) IS NOT NULL THEN
        LOCK TABLE public.validator_sourcing_epoch_inputs_v2
            IN ACCESS EXCLUSIVE MODE;
        EXECUTE 'SELECT EXISTS (SELECT 1 FROM public.validator_sourcing_epoch_inputs_v2)'
        INTO present;
        IF present THEN
            RAISE EXCEPTION 'validator sourcing rows appeared after SOURCE_ADD audit';
        END IF;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM source_add_history.archive_manifests
        WHERE source_relation = 'public.transparency_log'
          AND archive_floor = 41960931
          AND historical_ceiling = 41982623
          AND historical_row_count = 21660
          AND archived_row_count = 21660
          AND min_source_pk = 41960931
          AND max_source_pk = 41982623
          AND source_row_fingerprint = '5331decb81b3e4a11e0ebcc3d78ae606'
          AND audit_row_count = 324
          AND source_add_marker_row_count = 152
          AND missing_nonzero_parent_count = 4
          AND missing_parent_fingerprint = 'e8fb898746467eab12a07bdc296bfc18'
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD archive manifest differs from migration 233';
    END IF;

    SELECT pg_catalog.md5(routine.prosrc)
      INTO actual_md5
      FROM pg_catalog.pg_proc AS routine
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = routine.pronamespace
     WHERE namespace.nspname = 'public'
       AND routine.proname = 'enforce_research_lab_stateful_epoch_fence_v1'
       AND pg_catalog.pg_get_function_identity_arguments(routine.oid) = '';
    IF actual_md5 IS NULL OR actual_md5 NOT IN (
        '5b48baac95474f877b66f84deb78d8f4',
        '5b97ae04b866110b43b6cf8ec159463b'
    ) THEN
        RAISE EXCEPTION 'shared epoch fence changed after audit: %', actual_md5;
    END IF;
END;
$preflight$;

CREATE TEMP TABLE _retain_239_before ON COMMIT DROP AS
SELECT retained.schema_name,
       retained.table_name,
       relation.oid AS relation_oid,
       relation.relfilenode,
       pg_temp._retire_239_relation_signature(relation.oid) AS signature
FROM _retain_239_relations AS retained
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.nspname = retained.schema_name
JOIN pg_catalog.pg_class AS relation
  ON relation.relnamespace = namespace.oid
 AND relation.relname = retained.table_name;

CREATE TEMP TABLE _fence_239_before ON COMMIT DROP AS
SELECT routine.oid,
       routine.proowner,
       routine.proacl,
       routine.prosecdef,
       routine.proleakproof,
       routine.provolatile,
       routine.proparallel,
       routine.proconfig
FROM pg_catalog.pg_proc AS routine
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = routine.pronamespace
WHERE namespace.nspname = 'public'
  AND routine.proname = 'enforce_research_lab_stateful_epoch_fence_v1'
  AND pg_catalog.pg_get_function_identity_arguments(routine.oid) = '';

CREATE TEMP TABLE _archive_routine_239_before ON COMMIT DROP AS
SELECT routine.oid,
       pg_catalog.md5(routine.prosrc) AS body_md5,
       routine.proowner,
       routine.proacl,
       routine.prosecdef,
       routine.proconfig
FROM pg_catalog.pg_proc AS routine
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = routine.pronamespace
WHERE namespace.nspname = 'source_add_history';

-- Remove the obsolete V1 receipt table from the shared retained fence. No
-- other branch or permission changes.
CREATE OR REPLACE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1()
 RETURNS trigger
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO ''
AS $function$
DECLARE
    state_row public.research_lab_stateful_subnet_epoch_cutover_state_v1%ROWTYPE;
    row_doc JSONB := pg_catalog.to_jsonb(NEW);
    identity_key TEXT;
    identity_text TEXT;
    identity_value BIGINT;
    payload JSONB;
    authority_value JSONB;
    has_stateful_receipt BOOLEAN;
    linked_bundle RECORD;
BEGIN
    SELECT * INTO state_row
    FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1
    WHERE singleton;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'stateful epoch cutover singleton state is missing';
    END IF;
    IF state_row.lifecycle_state IN ('cutover_fenced', 'stateful_staged') THEN
        PERFORM pg_catalog.pg_advisory_xact_lock(7100, 0);
        SELECT * INTO state_row
        FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1
        WHERE singleton;
    END IF;

    IF TG_TABLE_NAME = 'research_lab_stateful_subnet_epoch_candidates_v1' THEN
        IF state_row.lifecycle_state = 'legacy_open' THEN
            RAISE EXCEPTION
                'stateful epoch candidate requires the durable pre-boundary fence';
        END IF;
        IF state_row.lifecycle_state = 'cutover_fenced'
           AND state_row.mapping_hash IS NULL
           AND row_doc->>'network_genesis_hash' = state_row.network_genesis_hash
           AND (row_doc->>'netuid')::INTEGER = state_row.netuid
           AND (row_doc->>'proposed_settlement_epoch_id')::INTEGER =
               state_row.first_settlement_epoch_id THEN
            RETURN NEW;
        END IF;
        IF row_doc->>'mapping_hash' IS NOT DISTINCT FROM state_row.mapping_hash
           AND row_doc->>'snapshot_hash' IS NOT DISTINCT FROM
              state_row.candidate_snapshot_hash
           AND row_doc->>'chain_state_receipt_hash' IS NOT DISTINCT FROM
              state_row.candidate_receipt_hash THEN
            RETURN NEW;
        END IF;
        RAISE EXCEPTION 'stateful epoch fence rejects an unbound candidate';
    ELSIF TG_TABLE_NAME = 'research_lab_stateful_subnet_epoch_cutovers_v1' THEN
        IF state_row.cutover_receipt_hash IS NOT NULL
           AND state_row.initialization_nonce IS NOT NULL
           AND state_row.initialization_payload_hash IS NOT NULL
           AND row_doc->>'mapping_hash' IS NOT DISTINCT FROM state_row.mapping_hash
           AND row_doc->>'cutover_authority_hash' IS NOT DISTINCT FROM
              state_row.cutover_authority_hash
           AND row_doc->>'cutover_receipt_hash' IS NOT DISTINCT FROM
              state_row.cutover_receipt_hash
           AND (row_doc->>'first_settlement_epoch_id')::BIGINT IS NOT DISTINCT FROM
              state_row.first_settlement_epoch_id::BIGINT THEN
            RETURN NEW;
        END IF;
        RAISE EXCEPTION 'stateful epoch fence requires the atomic staged cutover RPC';
    ELSIF TG_TABLE_NAME IN (
        'research_lab_stateful_subnet_epoch_boundaries_v1',
        'research_lab_stateful_subnet_epoch_snapshots_v1'
    ) THEN
        IF state_row.lifecycle_state = 'stateful_active' THEN
            RETURN NEW;
        END IF;
        RAISE EXCEPTION 'stateful epoch fence rejects boundary/snapshot writes before activation';
    END IF;

    IF state_row.lifecycle_state = 'legacy_open' THEN
        RETURN NEW;
    END IF;

    IF state_row.lifecycle_state = 'stateful_active' THEN
        FOR authority_value IN
            SELECT value
            FROM pg_catalog.jsonb_path_query(
                row_doc,
                'strict $.**.cutover_mapping_hash'
            ) AS authority_item(value)
        LOOP
            IF pg_catalog.jsonb_typeof(authority_value) <> 'string'
               OR authority_value #>> '{}' <> state_row.mapping_hash THEN
                RAISE EXCEPTION
                    'stateful epoch active mapping authority differs on %',
                    TG_TABLE_NAME;
            END IF;
        END LOOP;

        IF TG_TABLE_NAME IN (
            'published_weight_bundles',
            'research_lab_attested_weight_bundles',
            'research_lab_legacy_finalized_allocation_migrations_v2'
        ) THEN
            FOREACH identity_key IN ARRAY ARRAY[
                'epoch', 'epoch_id', 'evaluation_epoch'
            ]
            LOOP
                IF row_doc ? identity_key
                   AND pg_catalog.jsonb_typeof(row_doc->identity_key) <> 'null'
                   AND row_doc->>identity_key ~ '^[0-9]+$'
                   AND (row_doc->>identity_key)::BIGINT >=
                       state_row.first_settlement_epoch_id::BIGINT THEN
                    RAISE EXCEPTION
                        'stateful epoch active namespace rejects legacy %.% identity %',
                        TG_TABLE_NAME,
                        identity_key,
                        row_doc->>identity_key;
                END IF;
            END LOOP;
            RETURN NEW;
        END IF;

        IF TG_TABLE_NAME = 'research_lab_attested_execution_receipts_v2'
           AND row_doc->>'epoch_id' ~ '^[0-9]+$'
           AND (row_doc->>'epoch_id')::BIGINT >=
               state_row.first_settlement_epoch_id::BIGINT
           AND row_doc->>'purpose' IN (
               'validator.weight_snapshot.v2',
               'validator.weights.computed.v2',
               'validator.weights.finalized.v2',
               'gateway.weights.publication.v2'
           )
           AND NOT EXISTS (
               SELECT 1
               FROM public.research_lab_attested_execution_receipts_v2 AS receipt
               WHERE receipt.epoch_id = (row_doc->>'epoch_id')::INTEGER
                 AND receipt.role = 'validator_weights'
                 AND receipt.purpose = 'validator.subnet_epoch_snapshot.v2'
                 AND receipt.receipt_status = 'succeeded'
           ) THEN
            RAISE EXCEPTION
                'stateful epoch active V2 receipt lacks subnet authority %',
                row_doc->>'epoch_id';
        END IF;

        IF TG_TABLE_NAME = 'research_lab_attested_weight_bundles_v2'
           AND row_doc->>'epoch_id' ~ '^[0-9]+$'
           AND (row_doc->>'epoch_id')::BIGINT >=
               state_row.first_settlement_epoch_id::BIGINT THEN
            payload := row_doc->'bundle_doc'->'receipt_graph'->'receipts';
            SELECT EXISTS (
                SELECT 1
                FROM pg_catalog.jsonb_array_elements(payload) AS graph_receipt(doc)
                JOIN public.research_lab_attested_execution_receipts_v2 AS receipt
                  ON receipt.receipt_hash = graph_receipt.doc->>'receipt_hash'
                WHERE graph_receipt.doc->>'role' = 'validator_weights'
                  AND graph_receipt.doc->>'purpose' =
                      'validator.subnet_epoch_snapshot.v2'
                  AND graph_receipt.doc->>'epoch_id' ~ '^[0-9]+$'
                  AND (graph_receipt.doc->>'epoch_id')::BIGINT =
                      (row_doc->>'epoch_id')::BIGINT
                  AND receipt.epoch_id = (row_doc->>'epoch_id')::INTEGER
                  AND receipt.role = 'validator_weights'
                  AND receipt.purpose = 'validator.subnet_epoch_snapshot.v2'
                  AND receipt.receipt_status = 'succeeded'
            ) INTO has_stateful_receipt
            WHERE pg_catalog.jsonb_typeof(payload) = 'array';
            IF NOT COALESCE(has_stateful_receipt, FALSE) THEN
                RAISE EXCEPTION
                    'stateful epoch active V2 bundle lacks subnet authority %',
                    row_doc->>'epoch_id';
            END IF;
        END IF;

        IF TG_TABLE_NAME = 'research_lab_attested_weight_finalizations_v2' THEN
            SELECT bundle.epoch_id, bundle.bundle_doc
            INTO linked_bundle
            FROM public.research_lab_attested_weight_bundles_v2 AS bundle
            WHERE bundle.bundle_hash = row_doc->>'bundle_hash';
            IF FOUND
               AND linked_bundle.epoch_id >=
                   state_row.first_settlement_epoch_id THEN
                payload := linked_bundle.bundle_doc->'receipt_graph'->'receipts';
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_catalog.jsonb_array_elements(payload)
                         AS graph_receipt(doc)
                    JOIN public.research_lab_attested_execution_receipts_v2 AS receipt
                      ON receipt.receipt_hash = graph_receipt.doc->>'receipt_hash'
                    WHERE graph_receipt.doc->>'purpose' =
                          'validator.subnet_epoch_snapshot.v2'
                      AND graph_receipt.doc->>'epoch_id' ~ '^[0-9]+$'
                      AND (graph_receipt.doc->>'epoch_id')::INTEGER =
                          linked_bundle.epoch_id
                      AND receipt.epoch_id = linked_bundle.epoch_id
                      AND receipt.purpose = 'validator.subnet_epoch_snapshot.v2'
                      AND receipt.receipt_status = 'succeeded'
                ) INTO has_stateful_receipt
                WHERE pg_catalog.jsonb_typeof(payload) = 'array';
                IF NOT COALESCE(has_stateful_receipt, FALSE) THEN
                    RAISE EXCEPTION
                        'stateful epoch active V2 finalization lacks subnet authority %',
                        linked_bundle.epoch_id;
                END IF;
            END IF;
        END IF;

        RETURN NEW;
    END IF;

    IF TG_TABLE_NAME = 'research_lab_attested_execution_receipts_v2'
       AND row_doc ? 'epoch_id'
       AND row_doc->>'epoch_id' ~ '^[0-9]+$'
       AND (row_doc->>'epoch_id')::BIGINT >=
           state_row.first_settlement_epoch_id::BIGINT THEN
        IF row_doc->>'role' = 'validator_weights'
           AND row_doc->>'purpose' = 'validator.subnet_epoch_snapshot.v2'
           AND row_doc->>'receipt_status' = 'succeeded'
           AND row_doc->>'output_root' ~ '^sha256:[0-9a-f]{64}$'
           AND (row_doc->>'epoch_id')::BIGINT =
               state_row.first_settlement_epoch_id::BIGINT
           AND (
               state_row.candidate_receipt_hash IS NULL
               OR (
                   row_doc->>'receipt_hash' = state_row.candidate_receipt_hash
                   AND row_doc->>'output_root' = state_row.candidate_snapshot_hash
               )
           ) THEN
            RETURN NEW;
        END IF;
        IF row_doc->>'role' = 'gateway_coordinator'
           AND row_doc->>'purpose' = 'research_lab.subnet_epoch_cutover.v2'
           AND row_doc->>'receipt_status' = 'succeeded'
           AND row_doc->>'output_root' = state_row.cutover_authority_hash
           AND (row_doc->>'epoch_id')::BIGINT =
               state_row.first_settlement_epoch_id::BIGINT
           AND pg_catalog.jsonb_typeof(
                  row_doc->'receipt_doc'->'parent_receipt_hashes'
               ) = 'array'
           AND pg_catalog.jsonb_array_length(
                  row_doc->'receipt_doc'->'parent_receipt_hashes'
               ) = 2
           AND row_doc->'receipt_doc'->'parent_receipt_hashes'
               @> pg_catalog.jsonb_build_array(
                   state_row.candidate_receipt_hash,
                   state_row.last_legacy_finalization_receipt_hash
               ) THEN
            RETURN NEW;
        END IF;
        RAISE EXCEPTION 'stateful epoch fence rejects receipt epoch identity %',
            row_doc->>'epoch_id';
    END IF;

    FOREACH identity_key IN ARRAY ARRAY['epoch', 'epoch_id', 'evaluation_epoch']
    LOOP
        IF NOT (row_doc ? identity_key)
           OR pg_catalog.jsonb_typeof(row_doc->identity_key) = 'null' THEN
            CONTINUE;
        END IF;
        identity_text := row_doc->>identity_key;
        IF identity_text !~ '^-?[0-9]+$' THEN
            RAISE EXCEPTION 'stateful epoch fence rejects malformed %.% identity',
                TG_TABLE_NAME, identity_key;
        END IF;
        identity_value := identity_text::BIGINT;
        IF identity_value >= state_row.first_settlement_epoch_id::BIGINT THEN
            RAISE EXCEPTION 'stateful epoch fence rejects %.% identity %',
                TG_TABLE_NAME, identity_key, identity_value;
        END IF;
    END LOOP;
    RETURN NEW;
END;
$function$;

DO $drop_target_triggers$
DECLARE
    expected RECORD;
BEGIN
    FOR expected IN SELECT * FROM _retire_239_triggers LOOP
        IF pg_catalog.to_regclass('public.' || expected.table_name) IS NOT NULL THEN
            EXECUTE pg_catalog.format(
                'DROP TRIGGER IF EXISTS %I ON public.%I',
                expected.trigger_name, expected.table_name
            );
        END IF;
    END LOOP;
END;
$drop_target_triggers$;

-- This dependent view calls a routine in the function drop set.
DROP VIEW IF EXISTS public.research_lab_unpaid_ticket_expiry_candidates RESTRICT;

DROP FUNCTION IF EXISTS
    public.append_research_lab_inner_loop_activation_event(requested_event_type text, requested_phase text, requested_run_id uuid, requested_evidence_doc jsonb, requested_event_hash text, expected_current_phase text),
    public.claim_next_research_lab_candidate(p_holder_ref text, p_ttl_seconds integer, p_baseline_not_ready_retry_seconds integer, p_retryable_failure_retry_seconds integer),
    public.claim_next_research_loop_run(p_holder_ref text, p_ttl_seconds integer, p_allowed_run_ids uuid[]),
    public.claim_ops_alert_monitor_lease(p_monitor_id text, p_owner text, p_lease_seconds integer),
    public.guard_research_lab_candidate_artifact_image_build(),
    public.guard_research_lab_candidate_claim(),
    public.guard_research_lab_credit_consume(),
    public.guard_research_lab_loop_start_payment_expiry(),
    public.guard_research_lab_one_active_private_model_version(),
    public.guard_research_lab_queue_capacity(),
    public.guard_research_lab_run_claim(),
    public.guard_research_lab_ticket_lifecycle(),
    public.insert_research_lab_provider_usage_ledger_rows(rows jsonb),
    public.nuke_banned_hotkey_models(),
    public.prevent_research_lab_allocator_selection_mutation(),
    public.prevent_research_lab_append_only_mutation(),
    public.prevent_research_lab_attested_receipt_mutation(),
    public.prevent_research_lab_provider_registry_mutation(),
    public.prevent_research_lab_provider_usage_mutation(),
    public.prevent_research_lab_score_calibration_mutation(),
    public.prevent_research_trajectory_event_mutation(),
    public.refresh_qualification_model_scores(),
    public.research_lab_acquire_maintenance_lease(p_lease_name text, p_holder_ref text, p_ttl_seconds integer),
    public.research_lab_missing_trajectory_ids(candidate_ids uuid[]),
    public.research_lab_next_unprojected_terminal_runs(p_limit integer, p_newest_first boolean),
    public.research_lab_private_benchmark_schema_contract_v1(),
    public.research_lab_private_model_lineage_generation(),
    public.research_lab_terminal_runs_missing_traces(p_limit integer, p_newest_first boolean),
    public.research_lab_ticket_has_unpaid_lifecycle_evidence(target_ticket_id uuid),
    public.reset_daily_stats(),
    public.reset_miner_rate_limits(),
    public.reset_miner_rate_limits_daily(),
    public.resume_research_lab_credit_blocked_run_v1(p_run_id uuid, p_ticket_id uuid, p_expected_event_seq integer, p_expected_event_hash text, p_event_id uuid, p_anchored_hash text, p_queue_priority integer, p_worker_ref text, p_reason text, p_event_doc jsonb),
    public.update_qual_rate_limits_updated_at(),
    public.update_qualification_model_status(),
    public.update_qualification_timestamp(),
    public.update_updated_at_column()
RESTRICT;

DROP VIEW IF EXISTS
    public.qualification_champion_history,
    public.qualification_current_champion,
    public.qualification_leaderboard,
    public.research_evaluation_score_bundle_current,
    public.research_lab_candidate_evaluation_current,
    public.research_lab_daily_noise_budget_report_current,
    public.research_lab_gateway_control_current,
    public.research_lab_icp_churn_reversal_report,
    public.research_lab_inner_loop_activation_current,
    public.research_lab_private_model_benchmark_current,
    public.research_lab_private_model_version_current,
    public.research_lab_provider_registry_current,
    public.research_lab_public_benchmark_report_current,
    public.research_lab_sales_llm_corpus_metadata_current,
    public.research_loop_available_credits,
    public.research_loop_receipt_current,
    public.research_loop_run_queue_current,
    public.research_loop_start_credit_current,
    public.research_loop_ticket_current
RESTRICT;

DO $unschedule_legacy_cron$
DECLARE
    expected RECORD;
    present BOOLEAN;
BEGIN
    IF pg_catalog.to_regclass('cron.job') IS NULL THEN
        RETURN;
    END IF;
    FOR expected IN SELECT * FROM _retire_239_cron ORDER BY jobid LOOP
        EXECUTE 'SELECT EXISTS (SELECT 1 FROM cron.job WHERE jobid = $1)'
        INTO present USING expected.jobid;
        IF present THEN
            PERFORM cron.unschedule(expected.jobid);
        END IF;
    END LOOP;
END;
$unschedule_legacy_cron$;

DROP TABLE IF EXISTS public.banned_hotkeys RESTRICT;
DROP TABLE IF EXISTS public.contributor_attestations RESTRICT;
DROP TABLE IF EXISTS public.dashboard_miner_stats RESTRICT;
DROP TABLE IF EXISTS public.dashboard_precalc RESTRICT;
DROP TABLE IF EXISTS public.engine_trace_mappings RESTRICT;
DROP TABLE IF EXISTS public.epoch_audit_logs RESTRICT;
DROP TABLE IF EXISTS public.evidence_bundles RESTRICT;
DROP TABLE IF EXISTS public.execution_traces RESTRICT;
DROP TABLE IF EXISTS public.merkle_checkpoints RESTRICT;
DROP TABLE IF EXISTS public.miner_rate_limits RESTRICT;
DROP TABLE IF EXISTS public.ops_alert_delivery_events RESTRICT;
DROP TABLE IF EXISTS public.ops_alert_events RESTRICT;
DROP TABLE IF EXISTS public.ops_alert_current RESTRICT;
DROP TABLE IF EXISTS public.ops_alert_monitor_state RESTRICT;
DROP TABLE IF EXISTS public.ops_validator_registry RESTRICT;
DROP TABLE IF EXISTS public.qualification_baselines RESTRICT;
DROP TABLE IF EXISTS public.qualification_model_rate_limits RESTRICT;
DROP TABLE IF EXISTS public.qualification_payments RESTRICT;
DROP TABLE IF EXISTS public.qualification_models RESTRICT;
DROP TABLE IF EXISTS public.research_evaluation_score_bundle_events RESTRICT;
DROP TABLE IF EXISTS public.research_island_participation_snapshots RESTRICT;
DROP TABLE IF EXISTS public.research_lab_allocator_selection_records RESTRICT;
DROP TABLE IF EXISTS public.research_lab_attested_artifact_links RESTRICT;
DROP TABLE IF EXISTS public.research_lab_attested_execution_receipts RESTRICT;
DROP TABLE IF EXISTS public.research_lab_candidate_claim RESTRICT;
DROP TABLE IF EXISTS public.research_lab_candidate_evaluation_events RESTRICT;
DROP TABLE IF EXISTS public.research_lab_candidate_promotion_events RESTRICT;
DROP TABLE IF EXISTS public.research_lab_company_label_examples RESTRICT;
DROP TABLE IF EXISTS public.research_lab_corpus_complete RESTRICT;
DROP TABLE IF EXISTS public.research_lab_gateway_control_events RESTRICT;
DROP TABLE IF EXISTS public.research_lab_inner_loop_activation_events RESTRICT;
DROP TABLE IF EXISTS public.research_lab_maintenance_lease RESTRICT;
DROP TABLE IF EXISTS public.research_lab_openrouter_key_refs RESTRICT;
DROP TABLE IF EXISTS public.research_lab_openrouter_privacy_proof_events RESTRICT;
DROP TABLE IF EXISTS public.research_lab_private_model_benchmark_events RESTRICT;
DROP TABLE IF EXISTS public.research_lab_private_model_version_events RESTRICT;
DROP TABLE IF EXISTS public.research_lab_private_repo_commit_events RESTRICT;
DROP TABLE IF EXISTS public.research_lab_private_model_versions RESTRICT;
DROP TABLE IF EXISTS public.research_evaluation_score_bundles RESTRICT;
DROP TABLE IF EXISTS public.research_lab_candidate_artifacts RESTRICT;
DROP TABLE IF EXISTS public.research_lab_provider_credential_envelopes_v2 RESTRICT;
DROP TABLE IF EXISTS public.research_lab_provider_registry RESTRICT;
DROP TABLE IF EXISTS public.research_lab_provider_usage_ledger RESTRICT;
DROP TABLE IF EXISTS public.research_lab_public_benchmark_report_events RESTRICT;
DROP TABLE IF EXISTS public.research_lab_public_benchmark_reports RESTRICT;
DROP TABLE IF EXISTS public.research_lab_private_model_benchmark_bundles RESTRICT;
DROP TABLE IF EXISTS public.research_lab_rejected_companies RESTRICT;
DROP TABLE IF EXISTS public.research_lab_results_ledger RESTRICT;
DROP TABLE IF EXISTS public.research_lab_rolling_icp_windows RESTRICT;
DROP TABLE IF EXISTS public.research_lab_score_calibration RESTRICT;
DROP TABLE IF EXISTS public.research_lab_shadow_monitor_windows RESTRICT;
DROP TABLE IF EXISTS public.research_lab_signed_transition_commands_v2 RESTRICT;
DROP TABLE IF EXISTS public.research_lab_trace_pointer_quarantine RESTRICT;
DROP TABLE IF EXISTS public.research_loop_receipt_events RESTRICT;
DROP TABLE IF EXISTS public.research_loop_receipts RESTRICT;
DROP TABLE IF EXISTS public.research_loop_run_claim RESTRICT;
DROP TABLE IF EXISTS public.research_loop_run_queue_events RESTRICT;
DROP TABLE IF EXISTS public.research_loop_start_credit_events RESTRICT;
DROP TABLE IF EXISTS public.research_loop_start_payments RESTRICT;
DROP TABLE IF EXISTS public.research_loop_ticket_events RESTRICT;
DROP TABLE IF EXISTS public.research_loop_tickets RESTRICT;
DROP TABLE IF EXISTS public.research_trajectory_events RESTRICT;
DROP TABLE IF EXISTS public.research_trajectories RESTRICT;
DROP TABLE IF EXISTS public.validator_attestations RESTRICT;
DROP TABLE IF EXISTS public.validator_sourcing_epoch_inputs_v2 RESTRICT;

DO $postflight$
DECLARE
    unexpected TEXT;
    closure_pattern TEXT;
    actual_md5 TEXT;
BEGIN
    SELECT pg_catalog.format('%I.%I', namespace.nspname, relation.relname)
      INTO unexpected
      FROM pg_catalog.pg_class AS relation
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = relation.relnamespace
      JOIN _retire_239_tables AS retired
        ON namespace.nspname = 'public'
       AND retired.table_name = relation.relname
     LIMIT 1;
    IF unexpected IS NOT NULL THEN
        RAISE EXCEPTION 'migration 239 target relation remains: %', unexpected;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_proc AS routine
        JOIN pg_catalog.pg_namespace AS namespace
          ON namespace.oid = routine.pronamespace
        JOIN _retire_239_routines AS retired
          ON retired.routine_name = routine.proname
         AND retired.identity_arguments =
             pg_catalog.pg_get_function_identity_arguments(routine.oid)
        WHERE namespace.nspname = 'public'
    ) OR EXISTS (
        SELECT 1
        FROM pg_catalog.pg_class AS relation
        JOIN pg_catalog.pg_namespace AS namespace
          ON namespace.oid = relation.relnamespace
        JOIN _retire_239_views AS retired
          ON retired.view_name = relation.relname
        WHERE namespace.nspname = 'public'
    ) THEN
        RAISE EXCEPTION 'migration 239 reviewed routine or view remains';
    END IF;

    SELECT '(^|[^a-zA-Z0-9_])(' || pg_catalog.string_agg(name, '|') ||
           ')([^a-zA-Z0-9_]|$)'
      INTO closure_pattern
      FROM (
          SELECT table_name AS name FROM _retire_239_tables
          UNION SELECT view_name FROM _retire_239_views
          UNION SELECT routine_name FROM _retire_239_routines
      ) AS closure_names;

    SELECT pg_catalog.format('%I.%I', candidate.schemaname, candidate.viewname)
      INTO unexpected
      FROM (
          SELECT schemaname, viewname, definition FROM pg_catalog.pg_views
          UNION ALL
          SELECT schemaname, matviewname, definition FROM pg_catalog.pg_matviews
      ) AS candidate
     WHERE candidate.schemaname <> 'information_schema'
       AND candidate.schemaname !~ '^pg_'
       AND candidate.definition ~* closure_pattern
     LIMIT 1;
    IF unexpected IS NOT NULL THEN
        RAISE EXCEPTION 'remaining view references migration 239 closure: %',
            unexpected;
    END IF;

    SELECT pg_catalog.format(
               '%I.%I(%s)', namespace.nspname, routine.proname,
               pg_catalog.pg_get_function_identity_arguments(routine.oid)
           )
      INTO unexpected
      FROM pg_catalog.pg_proc AS routine
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = routine.pronamespace
     WHERE routine.prokind IN ('f', 'p')
       AND namespace.nspname <> 'information_schema'
       AND namespace.nspname !~ '^pg_'
       AND pg_catalog.pg_get_functiondef(routine.oid) ~* closure_pattern
     LIMIT 1;
    IF unexpected IS NOT NULL THEN
        RAISE EXCEPTION 'remaining routine references migration 239 closure: %',
            unexpected;
    END IF;

    IF pg_catalog.to_regclass('cron.job') IS NOT NULL THEN
        EXECUTE 'SELECT format(''%s:%s'', jobid, jobname) FROM cron.job WHERE jobid IN (8,15,47) OR jobname IN (''reset-daily-stats'',''process-rep-scores'',''reset-miner-rate-limits-daily'') LIMIT 1'
        INTO unexpected;
        IF unexpected IS NOT NULL THEN
            RAISE EXCEPTION 'migration 239 cron remains: %', unexpected;
        END IF;
        EXECUTE 'SELECT format(''%s:%s'', jobid, jobname) FROM cron.job WHERE command ~* $1 LIMIT 1'
        INTO unexpected USING closure_pattern;
        IF unexpected IS NOT NULL THEN
            RAISE EXCEPTION 'remaining cron references migration 239 closure: %',
                unexpected;
        END IF;
    END IF;

    IF EXISTS (
        (SELECT before.schema_name, before.table_name, before.relation_oid,
                before.relfilenode, before.signature
         FROM _retain_239_before AS before
         EXCEPT
         SELECT retained.schema_name, retained.table_name, relation.oid,
                relation.relfilenode,
                pg_temp._retire_239_relation_signature(relation.oid)
         FROM _retain_239_relations AS retained
         JOIN pg_catalog.pg_namespace AS namespace
           ON namespace.nspname = retained.schema_name
         JOIN pg_catalog.pg_class AS relation
           ON relation.relnamespace = namespace.oid
          AND relation.relname = retained.table_name)
        UNION ALL
        (SELECT retained.schema_name, retained.table_name, relation.oid,
                relation.relfilenode,
                pg_temp._retire_239_relation_signature(relation.oid)
         FROM _retain_239_relations AS retained
         JOIN pg_catalog.pg_namespace AS namespace
           ON namespace.nspname = retained.schema_name
         JOIN pg_catalog.pg_class AS relation
           ON relation.relnamespace = namespace.oid
          AND relation.relname = retained.table_name
         EXCEPT
         SELECT before.schema_name, before.table_name, before.relation_oid,
                before.relfilenode, before.signature
         FROM _retain_239_before AS before)
    ) THEN
        RAISE EXCEPTION 'retained relation metadata changed during migration 239';
    END IF;

    IF (SELECT count(*) FROM pg_catalog.pg_class AS relation
        JOIN pg_catalog.pg_namespace AS namespace
          ON namespace.oid = relation.relnamespace
        WHERE namespace.nspname = 'public'
          AND relation.relkind IN ('r','p')) <> 30
       OR EXISTS (
           SELECT relation.relname
           FROM pg_catalog.pg_class AS relation
           JOIN pg_catalog.pg_namespace AS namespace
             ON namespace.oid = relation.relnamespace
           WHERE namespace.nspname = 'public'
             AND relation.relkind IN ('r','p')
           EXCEPT
           SELECT table_name FROM _retain_239_relations
           WHERE schema_name = 'public'
       ) THEN
        RAISE EXCEPTION 'public table set differs from the 30-table retain set';
    END IF;

    SELECT pg_catalog.md5(routine.prosrc)
      INTO actual_md5
      FROM pg_catalog.pg_proc AS routine
      JOIN pg_catalog.pg_namespace AS namespace
        ON namespace.oid = routine.pronamespace
     WHERE namespace.nspname = 'public'
       AND routine.proname = 'enforce_research_lab_stateful_epoch_fence_v1'
       AND pg_catalog.pg_get_function_identity_arguments(routine.oid) = '';
    IF actual_md5 <> '5b97ae04b866110b43b6cf8ec159463b'
       OR EXISTS (
           (SELECT oid, proowner, proacl, prosecdef, proleakproof,
                   provolatile, proparallel, proconfig
            FROM _fence_239_before
            EXCEPT
            SELECT routine.oid, routine.proowner, routine.proacl,
                   routine.prosecdef, routine.proleakproof,
                   routine.provolatile, routine.proparallel, routine.proconfig
            FROM pg_catalog.pg_proc AS routine
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = routine.pronamespace
            WHERE namespace.nspname = 'public'
              AND routine.proname =
                  'enforce_research_lab_stateful_epoch_fence_v1'
              AND pg_catalog.pg_get_function_identity_arguments(routine.oid) = '')
       ) THEN
        RAISE EXCEPTION 'shared epoch fence body or metadata differs after rewrite';
    END IF;

    IF EXISTS (
        (SELECT * FROM _archive_routine_239_before
         EXCEPT
         SELECT routine.oid, pg_catalog.md5(routine.prosrc), routine.proowner,
                routine.proacl, routine.prosecdef, routine.proconfig
         FROM pg_catalog.pg_proc AS routine
         JOIN pg_catalog.pg_namespace AS namespace
           ON namespace.oid = routine.pronamespace
         WHERE namespace.nspname = 'source_add_history')
        UNION ALL
        (SELECT routine.oid, pg_catalog.md5(routine.prosrc), routine.proowner,
                routine.proacl, routine.prosecdef, routine.proconfig
         FROM pg_catalog.pg_proc AS routine
         JOIN pg_catalog.pg_namespace AS namespace
           ON namespace.oid = routine.pronamespace
         WHERE namespace.nspname = 'source_add_history'
         EXCEPT SELECT * FROM _archive_routine_239_before)
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD archive routine metadata changed';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM source_add_history.archive_manifests
        WHERE source_relation = 'public.transparency_log'
          AND archived_row_count = 21660
          AND source_row_fingerprint = '5331decb81b3e4a11e0ebcc3d78ae606'
          AND source_add_marker_row_count = 152
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD archive manifest changed during migration 239';
    END IF;
END;
$postflight$;

NOTIFY pgrst, 'reload schema';

COMMIT;
