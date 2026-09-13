# Confirmed-unused Supabase cleanup, September 12, 2026

Migration 231 removes 33 confirmed-empty tables, four dependent views, and 77
exclusive routines from the subnet production project. It also removes the
indexes, constraints, policies, and triggers owned by those tables. It does not
change application code, credentials, active table columns, or runtime services.

The review covered current subnet/Arena, dashboard, website and supporting
worker source, all five accessible project catalogs, SQL dependencies, scheduled
jobs, and 24 hours of subnet REST traffic. The subnet scan classified every one
of its 155 public tables, 34 views, and 404 public routines. Every target had an
exact zero row count, no active application reference, no matching REST request,
no outside foreign key, and only retired SQL consumers. Historical statistics
were not used as proof of an empty or inactive table.

## Preserved data and services

All 11 Arena tables and 102 Arena routines remain. Gateway epoch, qualification,
provider evidence/cache, baseline/rebenchmark, scoring, rewards, weight state,
validator, dashboard and website dependencies remain. Shared routing JSON/hash
helpers remain because official-baseline functions call them. The candidate
claim and two v1 attestation tables remain because retained guards name them.

No relation named `source_add` exists in any accessible project. Earlier
migration 198 removed the dedicated subnet SOURCE_ADD schema. This cleanup
preserves every surviving SOURCE_ADD record, all 11 shared v2 attestation graph
tables, transparency history, and their dependencies. The before snapshot found
3,510 SOURCE_ADD receipts, 1,435 execution results, and 9,994 transport attempts.
The website still uses a SOURCE_ADD tool namespace and is unchanged.

The website, staging, and canary projects have active or unresolved dependencies;
none has a positively proven deletion target. Their objects remain. Populated
legacy backups and history also remain where retirement or retention cannot be
proved. In particular, `test_leads_for_miners` contains 50,000 rows despite an
old statistics estimate of zero.

## Safety and validation

The exact migration takes table locks before checking for any row. It aborts if
it finds data, a changed routine body, an unexpected function/view/trigger, or a
scheduled job that depends on the removal set. It uses explicit names and
signatures and no `CASCADE`. PostgreSQL also rejects unlisted foreign keys and
other tracked dependencies. All removals commit together, then PostgREST reloads
its schema. A second application is safe.

Eight disposable PostgreSQL contract tests cover successful removal, preserved
history and baseline helpers, repeated application, nonempty tables, changed
routine bodies, private-schema callers/views, cron callers and foreign keys.
A separate structural rehearsal reconstructs the current public table layouts,
non-extension routines, views, indexes, constraints, triggers and policies and
applies the exact migration. It is a schema compatibility rehearsal, not a full
production-data restore. Complete current schema metadata and completed daily
backup metadata were captured before deletion; target tables contain no rows.

Production before/after checks cover gateway health and build, Arena proxy and
sidecar agreement, published baseline outputs for 20 ICPs, validator service and
finalized weight progression, dashboard readiness and competition/dashboard
APIs, retained database object identity, and SOURCE_ADD fingerprints/counts.
No paid scoring job, new submission, restart, or forced chain write is needed.

Existing dashboard failures that predate this cleanup are recorded separately:
retired Research Lab/Fulfillment routes still query some previously removed
objects. They are not evidence that another object is safe to delete, and this
migration does not restore them.

This cleanup reduces schema clutter. Its targets are empty and occupy only
about 1 MB, so it does not materially reduce the Supabase bill. Large active,
historical, and unresolved tables still require a separate retention or AWS
migration decision.

## Table decisions

The following is the complete subnet public-table decision list at the reviewed
source revision `f5f60197d1c038fb0397df837bc34bd45519c73f`.

| Table | Decision | Reason |
| --- | --- | --- |
| `banned_hotkeys` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `company_information_table` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `contributor_attestations` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `dashboard_miner_stats` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `dashboard_precalc` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `early_access_emails` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `engine_generated_candidates` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `engine_generated_datasets` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `engine_generated_evaluators` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `engine_issue_events` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `engine_issues` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `engine_trace_mappings` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `epoch_audit_logs` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `evidence_bundles` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `execution_traces` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `lab_arena_accepted_weight_states` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `lab_arena_chain_outcomes` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `lab_arena_company_judgment_reservations` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `lab_arena_company_judgments` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `lab_arena_judgment_cache` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `lab_arena_ledger` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `lab_arena_restart_claim_control` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `lab_arena_rounds` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `lab_arena_runs` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `lab_arena_submission_credentials` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `lab_arena_submissions` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `leads_private` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `leads_private_backup` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `merkle_checkpoints` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `miner_rate_limits` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `miner_test_leads` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `ops_alert_current` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `ops_alert_delivery_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `ops_alert_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `ops_alert_monitor_state` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `ops_research_lab_event_monitor_state` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `ops_research_lab_event_notifications` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `ops_validator_registry` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `outreach_email_verifications` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `published_weight_bundles` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `qualification_baselines` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `qualification_model_rate_limits` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `qualification_models` | retain shared dependency | A retained trigger/function or current shared control path depends on this relation. |
| `qualification_payments` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `qualification_private_icp_sets` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `research_evaluation_score_bundle_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_evaluation_score_bundles` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_island_participation_snapshots` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_allocator_selection_records` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_attested_ancestry_activations_v2` | retain source add attestation history | Shared v2 receipt graph can contain surviving SOURCE_ADD provenance and is also shared by current attested execution. |
| `research_lab_attested_ancestry_checkpoints_v2` | retain source add attestation history | Shared v2 receipt graph can contain surviving SOURCE_ADD provenance and is also shared by current attested execution. |
| `research_lab_attested_artifact_links` | retain shared dependency | A retained trigger/function or current shared control path depends on this relation. |
| `research_lab_attested_artifact_links_v2` | retain source add attestation history | Shared v2 receipt graph can contain surviving SOURCE_ADD provenance and is also shared by current attested execution. |
| `research_lab_attested_boot_identities_v2` | retain source add attestation history | Shared v2 receipt graph can contain surviving SOURCE_ADD provenance and is also shared by current attested execution. |
| `research_lab_attested_business_artifact_links_v2` | retain source add attestation history | Shared v2 receipt graph can contain surviving SOURCE_ADD provenance and is also shared by current attested execution. |
| `research_lab_attested_execution_receipts` | retain shared dependency | A retained trigger/function or current shared control path depends on this relation. |
| `research_lab_attested_execution_receipts_v2` | retain source add attestation history | Shared v2 receipt graph can contain surviving SOURCE_ADD provenance and is also shared by current attested execution. |
| `research_lab_attested_execution_results_v2` | retain source add attestation history | Shared v2 receipt graph can contain surviving SOURCE_ADD provenance and is also shared by current attested execution. |
| `research_lab_attested_host_operations_v2` | retain source add attestation history | Shared v2 receipt graph can contain surviving SOURCE_ADD provenance and is also shared by current attested execution. |
| `research_lab_attested_receipt_edges_v2` | retain source add attestation history | Shared v2 receipt graph can contain surviving SOURCE_ADD provenance and is also shared by current attested execution. |
| `research_lab_attested_receipt_transport_v2` | retain source add attestation history | Shared v2 receipt graph can contain surviving SOURCE_ADD provenance and is also shared by current attested execution. |
| `research_lab_attested_transport_attempts_v2` | retain source add attestation history | Shared v2 receipt graph can contain surviving SOURCE_ADD provenance and is also shared by current attested execution. |
| `research_lab_auto_research_loop_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_autoresearch_frontier_commitments` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_autoresearch_operation_settlements` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_autoresearch_tree_events` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_autoresearch_tree_handoffs` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_autoresearch_tree_nodes` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_autoresearch_trees` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_candidate_artifacts` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_candidate_claim` | retain shared dependency | A retained trigger/function or current shared control path depends on this relation. |
| `research_lab_candidate_evaluation_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_candidate_model_unit_terminals` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_candidate_promotion_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_candidate_waterfall_metrics` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_candidate_waterfall_receipts` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_company_label_examples` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_conditional_validation_events` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_corpus_complete` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_gateway_control_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_inner_loop_activation_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_maintenance_lease` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_official_baseline_action_attempts_v1` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_official_baseline_action_terminals_v1` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_official_baseline_runs_v1` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_official_baseline_unit_closures_v1` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_openrouter_key_refs` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_openrouter_privacy_proof_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_private_model_benchmark_bundles` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_private_model_benchmark_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_private_model_version_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_private_model_versions` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_private_repo_commit_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_provider_cost_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_provider_credential_envelopes_v2` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_provider_evidence_cache_v2` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `research_lab_provider_outcome_checkpoints_v2` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_provider_registry` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `research_lab_provider_usage_ledger` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_public_benchmark_report_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_public_benchmark_reports` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_public_loop_card_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_public_loop_cards` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_rejected_companies` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_results_ledger` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_rolling_icp_windows` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_routing_adapter_failures_v2` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_routing_budget_events_v2` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_routing_decision_receipts_v2` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_routing_evaluation_receipts_v2` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_routing_execution_request_leases_v2` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_routing_execution_requests_v2` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_routing_experiment_claim_closures_v2` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_routing_experiment_claim_closures_v3` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_routing_experiment_claim_heartbeats_v2` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_routing_experiment_claim_heartbeats_v3` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_routing_experiment_claims_v2` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_routing_experiment_claims_v3` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_routing_experiment_events_v2` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_routing_experiments_v2` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_routing_lab_references_v2` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_routing_provider_attempts_v2` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_score_calibration` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_scoring_category_results` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_scoring_dispatch_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_scoring_icp_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_scoring_icp_executions` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_scoring_job_candidate` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_scoring_job_queue` | delete confirmed empty | Live exact count and mutation counters are zero; no active source reference; migration removes its reviewed dependency closure. |
| `research_lab_scoring_run_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_scoring_runs` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_shadow_monitor_windows` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_signed_transition_commands_v2` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_lab_stateful_subnet_epoch_boundaries_v1` | retain shared dependency | A retained trigger/function or current shared control path depends on this relation. |
| `research_lab_stateful_subnet_epoch_candidates_v1` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `research_lab_stateful_subnet_epoch_cutover_state_v1` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `research_lab_stateful_subnet_epoch_cutovers_v1` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `research_lab_stateful_subnet_epoch_snapshots_v1` | retain shared dependency | A retained trigger/function or current shared control path depends on this relation. |
| `research_lab_trace_pointer_quarantine` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_loop_receipt_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_loop_receipts` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_loop_run_claim` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_loop_run_queue_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_loop_start_credit_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_loop_start_payments` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_loop_ticket_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_loop_tickets` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_trajectories` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `research_trajectory_events` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `suppression_ledger` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `test_leads_for_miners` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `transparency_log` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `validation_evidence_private` | retain active | Current runtime, startup, preflight, Arena store, gateway route, or TEE source references this relation. |
| `validator_attestations` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
| `validator_sourcing_epoch_inputs_v2` | retain unproven or historical | Not in the confirmed-empty allowlist; preserve until content, activity, and full dependency evidence prove deletion safe. |
