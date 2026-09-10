"""Text-level gates for scripts/179-lab-arena-v1.sql (no database needed)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
MIGRATION = SCRIPTS / "179-lab-arena-v1.sql"
SQL = MIGRATION.read_text(encoding="utf-8")
DAILY_SQL = (SCRIPTS / "180-lab-arena-daily-competition.sql").read_text(
    encoding="utf-8"
)
SOURCE_SQL = (SCRIPTS / "181-lab-arena-source-submissions.sql").read_text(
    encoding="utf-8"
)
SOURCE_EXECUTION_SQL = (SCRIPTS / "182-lab-arena-source-execution.sql").read_text(
    encoding="utf-8"
)
MINER_REWARD_SQL = (SCRIPTS / "183-lab-arena-miner-reward-basis.sql").read_text(
    encoding="utf-8"
)
CREDENTIAL_SQL = (SCRIPTS / "185-lab-arena-miner-credentials.sql").read_text(
    encoding="utf-8"
)
PROMOTION_THRESHOLD_SQL = (SCRIPTS / "187-lab-arena-promotion-threshold.sql").read_text(
    encoding="utf-8"
)
NETWORK_SCOPE_SQL = (SCRIPTS / "189-lab-arena-round-network-scope.sql").read_text(
    encoding="utf-8"
)
SCORER_REFRESH_SQL = (SCRIPTS / "194-lab-arena-open-scorer-refresh.sql").read_text(
    encoding="utf-8"
)
REWARD_CHAIN_SCOPE_SQL = (
    SCRIPTS / "197-lab-arena-reward-chain-scope.sql"
).read_text(encoding="utf-8")
WEIGHT_STATE_SQL = (
    SCRIPTS / "202-arena-accepted-weight-state.sql"
).read_text(encoding="utf-8")
RETIRED_INCENTIVE_BRIDGE_SQL = (
    SCRIPTS / "203-retire-legacy-incentive-weight-bridge.sql"
).read_text(encoding="utf-8")
HISTORICAL_UPLOAD_MIGRATION = SCRIPTS / "191-lab-arena-upload-recovery.sql"
HISTORICAL_UPLOAD_SHA256 = (
    "42913cf44d0d1f69a465731e75045af634c1b2600ab0e8fba24530ada979f8d7"
)

SERVICE_FUNCTIONS = (
    "lab_arena_whoami",
    "lab_arena_create_round",
    "lab_arena_transition_round",
    "lab_arena_activate_reward",
    "lab_arena_register_submission",
    "lab_arena_update_submission",
    "lab_arena_open_stage",
    "lab_arena_claim_assignment",
    "lab_arena_reserve_call",
    "lab_arena_mark_dispatched",
    "lab_arena_settle_call",
    "lab_arena_mark_uncertain",
    "lab_arena_complete_attempt",
    "lab_arena_expire_leases",
    "lab_arena_close_stage",
    "lab_arena_open_scoring",
    "lab_arena_close_scoring",
    "lab_arena_cancel_round",
    "lab_arena_record_run_scores",
)
TABLES = (
    "lab_arena_rounds",
    "lab_arena_submissions",
    "lab_arena_runs",
    "lab_arena_ledger",
)


def test_arena_migrations_are_uniquely_numbered():
    numbered = {}
    for path in SCRIPTS.glob("*.sql"):
        # This exact path was applied before the Arena migration moved to 193.
        # Keep its blob for snapshot history, but never select it as a forward migration.
        if path == HISTORICAL_UPLOAD_MIGRATION:
            continue
        match = re.match(r"^(\d+)-", path.name)
        if match and int(match.group(1)) >= 100:
            numbered.setdefault(int(match.group(1)), []).append(path.name)
    assert numbered[178] == ["178-research-lab-source-add-miner-status.sql"]
    assert numbered[179] == ["179-lab-arena-v1.sql"]
    assert numbered[180] == ["180-lab-arena-daily-competition.sql"]
    assert numbered[181] == ["181-lab-arena-source-submissions.sql"]
    assert numbered[182] == ["182-lab-arena-source-execution.sql"]
    assert numbered[183] == ["183-lab-arena-miner-reward-basis.sql"]
    assert numbered[184] == ["184-lab-arena-scoring-failure-isolation.sql"]
    assert numbered[185] == ["185-lab-arena-miner-credentials.sql"]
    assert numbered[186] == ["186-research-lab-source-add-provisioned-status.sql"]
    assert numbered[187] == ["187-lab-arena-promotion-threshold.sql"]
    assert numbered[188] == ["188-lab-arena-baseline-promotion.sql"]
    assert numbered[189] == ["189-lab-arena-round-network-scope.sql"]
    assert numbered[190] == ["190-lab-arena-restart-claim-drain.sql"]
    assert numbered[191] == ["191-fresh-network-subnet-epoch-authority.sql"]
    assert numbered[193] == ["193-lab-arena-upload-recovery.sql"]
    assert numbered[194] == ["194-lab-arena-open-scorer-refresh.sql"]
    assert numbered[197] == ["197-lab-arena-reward-chain-scope.sql"]
    assert numbered[199] == ["199-lab-arena-source-disclosure-time.sql"]
    assert numbered[200] == ["200-lab-arena-next-day-icp-disclosure.sql"]
    assert numbered[201] == ["201-lab-arena-daily-capacity.sql"]
    assert numbered[202] == ["202-arena-accepted-weight-state.sql"]
    assert numbered[203] == ["203-retire-legacy-incentive-weight-bridge.sql"]
    assert all(len(paths) == 1 for paths in numbered.values()), numbered


def test_reward_chain_scope_migration_scopes_every_reward_history_read():
    assert "lab_arena_rounds_reward_chain_epoch_uq" in REWARD_CHAIN_SCOPE_SQL
    assert REWARD_CHAIN_SCOPE_SQL.count(
        "arena_network_name = v_round.arena_network_name"
    ) >= 3
    assert REWARD_CHAIN_SCOPE_SQL.count("arena_netuid = v_round.arena_netuid") >= 3
    assert "published_at, arena_network_name, arena_netuid" in REWARD_CHAIN_SCOPE_SQL
    assert "'version', 197" in REWARD_CHAIN_SCOPE_SQL


def test_weight_state_migration_keeps_core_schema_rollback_compatible():
    assert "CREATE OR REPLACE FUNCTION public.lab_arena_schema_version_v1()" not in WEIGHT_STATE_SQL
    assert "CREATE OR REPLACE FUNCTION public.lab_arena_weight_state_schema_v1()" in WEIGHT_STATE_SQL
    assert "'leadpoet.lab_arena.weight_state_schema.v1', 'version', 202" in WEIGHT_STATE_SQL
    assert "research_lab_emission_allocation_snapshots" not in WEIGHT_STATE_SQL
    assert "fulfillment_score_consensus" not in WEIGHT_STATE_SQL


def test_retired_incentive_schema_is_removed_without_scoring_scope_growth():
    assert "DROP FUNCTION IF EXISTS public.lab_arena_weight_inputs_v1" in RETIRED_INCENTIVE_BRIDGE_SQL
    for table in (
        "research_reimbursement_awards",
        "research_lab_champion_reward_obligations",
        "research_lab_emission_allocation_snapshots",
        "research_lab_attested_weight_bundles_v2",
        "research_lab_chain_realized_epoch_settlements_v1",
        "research_lab_compact_weight_authorities_v2",
    ):
        assert f"DROP TABLE IF EXISTS public.{table}" in RETIRED_INCENTIVE_BRIDGE_SQL
    assert "DROP TABLE IF EXISTS public.research_lab_scoring_runs" not in RETIRED_INCENTIVE_BRIDGE_SQL
    assert "DROP TABLE IF EXISTS public.lab_arena_" not in RETIRED_INCENTIVE_BRIDGE_SQL
    assert "DROP TABLE IF EXISTS public.fulfillment_score_consensus" not in RETIRED_INCENTIVE_BRIDGE_SQL
    assert "lab_arena_incentive_retirement_schema_v1" in RETIRED_INCENTIVE_BRIDGE_SQL
    assert "DROP TABLE IF EXISTS public.research_lab_stateful_subnet_epoch_cutovers_v1" not in RETIRED_INCENTIVE_BRIDGE_SQL
    assert "DROP TABLE IF EXISTS public.research_lab_stateful_subnet_epoch_cutover_state_v1" not in RETIRED_INCENTIVE_BRIDGE_SQL
    assert "detach_retired_weight_history" in RETIRED_INCENTIVE_BRIDGE_SQL
    assert "DROP VIEW IF EXISTS public.research_lab_stateful_subnet_epoch_mapping_v1" in RETIRED_INCENTIVE_BRIDGE_SQL
    assert "research_lab_stateful_subnet_epoch_stage_v2" in RETIRED_INCENTIVE_BRIDGE_SQL
    assert "research_lab_stateful_subnet_epoch_cutover_public_state_v1'" not in RETIRED_INCENTIVE_BRIDGE_SQL
    assert "research_lab_champion_lifetime_credit_contract_v1" in RETIRED_INCENTIVE_BRIDGE_SQL
    assert "enforce_temporary_testnet401_execution_result_epoch_scope_v1" in RETIRED_INCENTIVE_BRIDGE_SQL
    assert "enforce_temporary_testnet401_weight_submission_epoch_scope_v1" in RETIRED_INCENTIVE_BRIDGE_SQL


def test_historical_reward_and_weight_function_inventory_is_classified():
    create_function = re.compile(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+(?:public\.)?([a-zA-Z0-9_]+)",
        re.I | re.S,
    )
    historical = {
        name
        for path in SCRIPTS.glob("*.sql")
        for name in create_function.findall(path.read_text(encoding="utf-8"))
    }
    retired = {
        name for name in historical
        if (
            name.startswith("persist_research_lab_allocation_")
            or name.startswith("persist_research_lab_chain_realized_")
            or name.startswith("research_lab_allocation_frontier_")
            or name.startswith("research_lab_compact_weight_")
            or name == "research_lab_champion_lifetime_credit_contract_v1"
            or name.startswith("enforce_temporary_testnet401_")
        )
    }
    assert retired == {
        "persist_research_lab_allocation_frontier_bootstrap_v2",
        "persist_research_lab_allocation_settlement_frontier_v2",
        "persist_research_lab_chain_realized_lifetime_settlement_v2",
        "persist_research_lab_chain_realized_settlement_v1",
        "persist_research_lab_chain_realized_unattributed_v2",
        "research_lab_allocation_frontier_bootstrap_contract_v2",
        "research_lab_allocation_frontier_historical_source_contract_v1",
        "research_lab_champion_lifetime_credit_contract_v1",
        "research_lab_compact_weight_settlement_contract_v1",
        "enforce_temporary_testnet401_execution_result_epoch_scope_v1",
        "enforce_temporary_testnet401_weight_submission_epoch_scope_v1",
    }
    for pattern in (
        "persist_research_lab_chain_realized_%",
        "persist_research_lab_allocation_%",
        "research_lab_allocation_frontier_%",
        "research_lab_compact_weight_%",
    ):
        assert pattern in RETIRED_INCENTIVE_BRIDGE_SQL
    # These similarly named functions remain by design: current Arena state,
    # generic epoch-table integrity, and the one public namespace reader.
    assert "lab_arena_publish_weight_state_v1" not in retired
    assert "validate_research_lab_stateful_subnet_epoch_v1" not in retired
    assert "research_lab_stateful_subnet_epoch_cutover_public_state_v1" not in retired


def test_historical_upload_migration_is_retained_byte_for_byte():
    assert hashlib.sha256(HISTORICAL_UPLOAD_MIGRATION.read_bytes()).hexdigest() == (
        HISTORICAL_UPLOAD_SHA256
    )


def test_network_scope_migration_keeps_legacy_finney_defaults_queryable():
    assert "COALESCE(configuration_doc ->> 'network_name', 'finney')" in NETWORK_SCOPE_SQL
    assert "COALESCE((configuration_doc ->> 'netuid')::BIGINT, 71)" in NETWORK_SCOPE_SQL
    assert "arena_network_name, arena_netuid, status, created_at DESC" in NETWORK_SCOPE_SQL
    assert "(configuration_doc ? 'network_name') = (configuration_doc ? 'netuid')" in NETWORK_SCOPE_SQL
    assert "'version', 189" in NETWORK_SCOPE_SQL
    assert "NOTIFY pgrst, 'reload schema';" in NETWORK_SCOPE_SQL


def test_scorer_refresh_is_limited_to_atomic_open_commit():
    assert "p_expected_status = 'open' AND p_next_status = 'committed'" in SCORER_REFRESH_SQL
    assert "ARRAY['scorer_image_digest', 'scorer_image_reference']" in SCORER_REFRESH_SQL
    assert "v_round.configuration_doc || pg_catalog.jsonb_build_object" in SCORER_REFRESH_SQL
    assert "OLD.status = 'open'" in SCORER_REFRESH_SQL
    assert "AND NEW.status = 'committed'" in SCORER_REFRESH_SQL
    assert "NEW.configuration_doc - 'scorer_image_digest' - 'scorer_image_reference'" in SCORER_REFRESH_SQL
    assert "lab_arena_scorer_image_invalid" in SCORER_REFRESH_SQL
    assert "'version', 194" in SCORER_REFRESH_SQL


def test_promotion_threshold_migration_uses_exact_numeric_one_point_gate():
    assert PROMOTION_THRESHOLD_SQL.lstrip().startswith(
        "-- 187-lab-arena-promotion-threshold.sql"
    )
    assert "v_winner_score < v_baseline_score + 1" in PROMOTION_THRESHOLD_SQL
    assert "::NUMERIC >= v_baseline_score + 1" in PROMOTION_THRESHOLD_SQL
    assert "OLD.status = 'published'" in MINER_REWARD_SQL
    assert "UPDATE public.lab_arena_rounds" not in PROMOTION_THRESHOLD_SQL


def test_migration_transaction_and_reload_shape():
    assert SQL.lstrip().startswith("-- 179-lab-arena-v1.sql")
    assert "\nBEGIN;\n" in SQL
    assert SQL.rstrip().endswith("COMMIT;")
    assert "NOTIFY pgrst, 'reload schema';" in SQL
    assert "REVOKE TRUNCATE" in SQL
    assert "SET search_path = pg_catalog, public" in SQL
    # COALESCE/NULLIF/GREATEST/LEAST are keyword expressions, not catalog functions.
    assert not re.search(r"pg_catalog\.(coalesce|nullif|greatest|least)\(", SQL, re.I)


def test_roles_follow_the_migration_156_pattern():
    assert "CREATE ROLE lab_arena_owner NOLOGIN" in SQL
    assert "CREATE ROLE lab_arena_service NOLOGIN" in SQL
    assert "pg_advisory_xact_lock" in SQL
    assert "ALTER ROLE lab_arena_service WITH NOCREATEDB NOCREATEROLE NOINHERIT;" in SQL
    code_lines = [line for line in SQL.splitlines() if not line.lstrip().startswith("--")]
    assert not any("NOSUPERUSER" in line or "NOBYPASSRLS" in line for line in code_lines)
    assert "rolsuper OR rolbypassrls OR rolcanlogin OR rolreplication" in SQL
    assert "rolname = 'authenticator'" in SQL
    assert "GRANT lab_arena_service TO authenticator" in SQL
    assert "REVOKE CREATE ON SCHEMA public FROM lab_arena_service" in SQL


def test_every_table_has_rls_and_service_select_policy():
    for table in TABLES:
        assert f"CREATE TABLE IF NOT EXISTS public.{table} (" in SQL
        assert f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY;" in SQL
        assert f"ALTER TABLE public.{table} OWNER TO lab_arena_owner;" in SQL
    assert "FOR SELECT TO lab_arena_service USING (TRUE)" in SQL
    assert "GRANT SELECT ON TABLE public.%I TO lab_arena_service" in SQL
    assert "GRANT SELECT ON TABLE public.%I TO service_role" not in SQL


def test_append_only_and_write_once_triggers_exist():
    assert "CREATE TRIGGER lab_arena_ledger_append_only" in SQL
    assert "BEFORE UPDATE OR DELETE ON public.lab_arena_ledger" in SQL
    assert "CREATE TRIGGER lab_arena_rounds_write_once" in SQL
    assert "CREATE TRIGGER lab_arena_submissions_frozen" in SQL
    assert "CREATE TRIGGER lab_arena_runs_terminal" in SQL


def test_every_service_function_is_definer_owned_and_granted_only_to_service():
    for function in SERVICE_FUNCTIONS:
        assert f"CREATE OR REPLACE FUNCTION public.{function}(" in SQL, function
        assert f"'public.{function}(" in SQL, function
        assert f"ALTER FUNCTION public.{function}(" in SQL, function
    definer_count = SQL.count("SECURITY DEFINER")
    assert definer_count >= len(SERVICE_FUNCTIONS) - 1
    assert "SECURITY INVOKER" in SQL  # whoami runs as the caller
    assert "GRANT EXECUTE ON FUNCTION %s TO lab_arena_service" in SQL
    assert "REVOKE ALL ON FUNCTION %s FROM lab_arena_service" in SQL  # internal helpers


def test_state_vocabularies_match_contracts():
    from lab_arena import contracts

    for status in contracts.ROUND_STATUSES:
        assert f"'{status}'" in SQL
    for outcome in contracts.KING_OUTCOMES:
        assert f"'{outcome}'" in SQL
    for cause in contracts.TERMINAL_CAUSES:
        assert f"'{cause}'" in SQL + CREDENTIAL_SQL
    for kind in contracts.LEDGER_ENTRY_KINDS:
        assert f"'{kind}'" in SQL


def test_unique_indexes_enforce_plan_invariants():
    assert "lab_arena_runs_one_active_attempt_uq" in SQL
    assert "WHERE status IN ('pending', 'leased', 'submitted')" in SQL
    assert "lab_arena_runs_one_accepted_uq" in SQL
    assert "lab_arena_runs_claim_request_uq" in SQL
    assert "lab_arena_ledger_reservation_uq" in SQL
    assert "lab_arena_ledger_terminal_uq" in SQL
    assert "lab_arena_rounds_effective_reward_epoch_uq" in SQL
    assert "ON public.lab_arena_rounds (effective_reward_epoch)\n  WHERE effective_reward_epoch IS NOT NULL" in SQL
    assert "funding_source IS NULL OR funding_source = 'host'" in SQL
    assert "balance_microusd" not in SQL and "credit_deposit" not in SQL
    assert "CREATE TABLE IF NOT EXISTS public.lab_arena_accounts" not in SQL
    assert "CREATE OR REPLACE FUNCTION public.lab_arena__aggregate_preflight" not in SQL
    assert "lab_arena__run_consumed(p_run_id TEXT, p_provider TEXT)" in SQL
    assert "FOR UPDATE SKIP LOCKED" in SQL
    assert "'lab_arena.runner:' || p_round_id || ':' || p_runner_hotkey" in SQL
    assert "DROP INDEX IF EXISTS public.lab_arena_submissions_image_digest_uq" in SQL
    assert "CREATE UNIQUE INDEX IF NOT EXISTS lab_arena_submissions_image_digest_uq" not in SQL


def test_miner_credentials_are_ciphertext_only_and_miner_calls_cannot_use_host_funds():
    assert "CREATE TABLE IF NOT EXISTS public.lab_arena_submission_credentials" in CREDENTIAL_SQL
    assert "ciphertext BYTEA NOT NULL" in CREDENTIAL_SQL
    assert "openrouter_management" not in CREDENTIAL_SQL
    assert "funding_source IN ('host', 'miner_key')" in CREDENTIAL_SQL
    assert "lab_arena_funding_source_mismatch" in CREDENTIAL_SQL
    assert "lab_arena_submission_credentials_required" in CREDENTIAL_SQL
    assert "pg_catalog.replace(" in CREDENTIAL_SQL


def test_attempt_completion_stores_only_the_result_and_output():
    assert "result_doc JSONB" in SQL
    # These names can occur only in DROP statements that clean up an earlier
    # development draft. They are not part of the installed run shape.
    for removed in ("receipt_doc", "receipt_hash", "provider_call_root", "private_event_root", "cost_root"):
        assert f"  {removed} " not in SQL
        assert f"SET {removed} =" not in SQL
    assert "(p_result ->> 'terminal_status') IS DISTINCT FROM p_terminal_cause" in SQL
    assert "ALTER FUNCTION public.lab_arena_complete_attempt(TEXT, TEXT, JSONB, TEXT, TEXT)" in SQL


def test_competition_identity_hash_chains_are_not_installed():
    assert "generation_attempts JSONB" not in SQL
    assert "CREATE OR REPLACE FUNCTION public.lab_arena_append_generation_attempt(" not in SQL
    assert "CREATE TABLE IF NOT EXISTS public.lab_arena_events" not in SQL
    assert "CREATE OR REPLACE FUNCTION public.lab_arena_append_events(" not in SQL
    assert "CREATE OR REPLACE FUNCTION public.lab_arena_append_journal_entry(" not in SQL
    for removed in (
        "journal_head_hash",
        "configuration_hash",
        "commitment_hash",
        "commitment_doc",
        "participant_set_hash",
        "stage1_scoring_plan_hash",
        "final_score_bundle_hash",
        "result_bundle_hash",
        "event_cursor",
        "event_head_hash",
        "submitted_digest",
        "entry_command",
        "image_environment",
        "working_dir",
        "icp_hash",
        "work_item_id",
        "output_hash",
        "score_ref",
        "score_doc",
    ):
        assert f"  {removed} " not in SQL
        assert f"SET {removed} =" not in SQL


def test_simple_stage_and_scoring_linkage_has_no_hash_gate():
    assert "ALTER FUNCTION public.lab_arena_open_stage(TEXT, SMALLINT, JSONB, INTEGER[])" in SQL
    assert "p_icp_hashes" not in SQL
    assert "v_item ->> 'scored_run_id'" in SQL
    assert "v_item ->> 'submission_id'" in SQL
    assert "v_item ->> 'icp_position'" in SQL
    assert "v_item ->> 'output_ref'" in SQL


def test_two_stage_cut_uses_plain_finalist_ids_and_fixed_position_ranges():
    assert "'stage1_scored'" in SQL
    assert "'stage2_scoring_plan_doc'" in SQL
    assert "'finalists'" in SQL
    assert "v_patch -> 'stage1_scores_ref'" not in SQL
    assert "v_patch -> 'final_scores_ref'" not in SQL
    assert "p_stage NOT IN (1, 2)" in SQL
    assert "ARRAY[0,1,2,3,4,5,6,7,8,9]" in SQL
    assert "ARRAY[10,11,12,13,14,15,16,17,18,19]" in DAILY_SQL
    assert "pg_catalog.jsonb_array_length(v_patch -> 'finalists') <> LEAST(10, v_challenger_count)" in SQL
    assert "AVG(" not in SQL


def test_daily_migration_has_one_private_input_and_no_parallel_baseline_state():
    assert "qualification_private_icp_sets" in DAILY_SQL
    assert "jsonb_array_length(source.icps) = 20" in DAILY_SQL
    assert "research_lab_daily_rebenchmarks" not in DAILY_SQL
    assert "receipt" not in DAILY_SQL.lower()
    assert "manifest" not in DAILY_SQL.lower()
    assert "hash_chain" not in DAILY_SQL.lower()


def test_source_intake_migration_has_one_active_slot_and_no_image_identity():
    assert "source_ref TEXT" in SOURCE_SQL
    assert "source_cache_key" not in SOURCE_SQL
    assert "ADD COLUMN IF NOT EXISTS source_sha256" not in SOURCE_SQL
    assert "source_size_bytes BIGINT" in SOURCE_SQL
    assert "status IN ('uploading', 'accepted', 'frozen')" in SOURCE_SQL
    assert "lab_arena_submissions_one_active_per_miner_uq" in SOURCE_SQL
    assert "submitted_reference" not in SOURCE_SQL
    assert "image_digest" not in SOURCE_SQL
    assert "manifest" not in SOURCE_SQL.lower()
    assert "receipt" not in SOURCE_SQL.lower()
    assert "hash_chain" not in SOURCE_SQL.lower()


def test_source_execution_leases_source_and_removes_miner_image_columns():
    assert SOURCE_EXECUTION_SQL.lstrip().startswith(
        "-- 182-lab-arena-source-execution.sql"
    )
    assert "\nBEGIN;\n" in SOURCE_EXECUTION_SQL
    assert SOURCE_EXECUTION_SQL.rstrip().endswith("COMMIT;")
    assert "NOTIFY pgrst, 'reload schema';" in SOURCE_EXECUTION_SQL
    assert "CREATE OR REPLACE FUNCTION public.lab_arena_claim_assignment(" in SOURCE_EXECUTION_SQL
    assert "'source_ref', CASE WHEN v_run.kind = 'execute'" in SOURCE_EXECUTION_SQL
    assert "source_cache_key" not in SOURCE_EXECUTION_SQL
    assert "'source_sha256', CASE WHEN v_run.kind = 'execute'" not in SOURCE_EXECUTION_SQL
    assert "'source_size_bytes', CASE WHEN v_run.kind = 'execute'" in SOURCE_EXECUTION_SQL
    assert "'image_digest'," not in SOURCE_EXECUTION_SQL
    assert "source_submission.status = 'frozen'" in SOURCE_EXECUTION_SQL
    assert "PERFORM public.lab_arena_cancel_round(v_round_id, 'source_bundle_cutover')" in SOURCE_EXECUTION_SQL
    assert "DISABLE TRIGGER lab_arena_submissions_frozen" in SOURCE_EXECUTION_SQL
    assert "ENABLE TRIGGER lab_arena_submissions_frozen" in SOURCE_EXECUTION_SQL
    assert "baseline_submission.is_king" in SOURCE_EXECUTION_SQL
    assert "(v_round.configuration_doc ->> 'baseline_hotkey')" in SOURCE_EXECUTION_SQL
    for column in (
        "submitted_reference",
        "image_reference",
        "image_digest",
        "image_size_bytes",
    ):
        assert "DROP COLUMN IF EXISTS %s" % column in SOURCE_EXECUTION_SQL
    assert "CASCADE" not in SOURCE_EXECUTION_SQL
    assert "manifest" not in SOURCE_EXECUTION_SQL.lower()
    assert "receipt" not in SOURCE_EXECUTION_SQL.lower()
    assert "hash_chain" not in SOURCE_EXECUTION_SQL.lower()


def test_host_openrouter_money_caps_are_atomic_and_round_wide_per_submission():
    assert "lab_arena__submission_kind_spend(TEXT, TEXT)" in SQL
    assert "configuration_doc ->> 'execution_cap_microusd'" in SQL
    assert "configuration_doc ->> 'scoring_cap_microusd'" in SQL
    assert "v_spent > v_money_cap - p_amount_microusd" in SQL
    assert "v_reason := 'money_cap'" in SQL
    assert "WHERE ledger.submission_id = p_submission_id" in SQL
    assert "head.entry_kind IN ('reservation', 'dispatch', 'settlement', 'uncertain')" in SQL


def test_reward_view_is_live_only_and_score_leases_use_round_scorer():
    assert "configuration_doc ->> 'mode' = 'live'" in SQL
    assert "v_round.configuration_doc ->> 'scorer_image_digest'" in SQL
