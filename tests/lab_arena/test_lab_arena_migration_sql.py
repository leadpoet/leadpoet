"""Text-level gates for scripts/174-lab-arena-v1.sql (no database needed)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
MIGRATION = SCRIPTS / "174-lab-arena-v1.sql"
SQL = MIGRATION.read_text(encoding="utf-8")

SERVICE_FUNCTIONS = (
    "lab_arena_whoami",
    "lab_arena_create_round",
    "lab_arena_transition_round",
    "lab_arena_append_journal_entry",
    "lab_arena_register_submission",
    "lab_arena_update_submission",
    "lab_arena_upsert_account_credential",
    "lab_arena_record_preflight",
    "lab_arena_credit_deposit",
    "lab_arena_open_stage",
    "lab_arena_claim_assignment",
    "lab_arena_reserve_call",
    "lab_arena_mark_dispatched",
    "lab_arena_settle_call",
    "lab_arena_mark_uncertain",
    "lab_arena_append_events",
    "lab_arena_complete_attempt",
    "lab_arena_expire_leases",
    "lab_arena_close_stage",
    "lab_arena_cancel_round",
    "lab_arena_record_run_scores",
)
TABLES = (
    "lab_arena_rounds",
    "lab_arena_submissions",
    "lab_arena_runs",
    "lab_arena_events",
    "lab_arena_accounts",
    "lab_arena_ledger",
)


def test_migration_is_the_frontier_and_uniquely_numbered():
    numbered = {}
    for path in SCRIPTS.glob("*.sql"):
        match = re.match(r"^(\d+)-", path.name)
        if match and int(match.group(1)) >= 100:
            numbered.setdefault(int(match.group(1)), []).append(path.name)
    assert numbered[174] == ["174-lab-arena-v1.sql"]
    tracked = [n for n in numbered if n != 174]
    assert max(tracked) == 173, "174 must sit directly above the production frontier"


def test_migration_transaction_and_reload_shape():
    assert SQL.lstrip().startswith("-- 174-lab-arena-v1.sql")
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
    assert "ALTER ROLE lab_arena_service WITH NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION" in SQL
    code_lines = [line for line in SQL.splitlines() if not line.lstrip().startswith("--")]
    assert not any("NOSUPERUSER" in line or "NOBYPASSRLS" in line for line in code_lines)
    assert "rolsuper OR rolbypassrls OR rolcanlogin" in SQL
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
    assert "CREATE TRIGGER lab_arena_events_append_only" in SQL
    assert "CREATE TRIGGER lab_arena_ledger_append_only" in SQL
    assert "BEFORE UPDATE OR DELETE ON public.lab_arena_events" in SQL
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
        assert f"'{cause}'" in SQL
    for kind in contracts.LEDGER_ENTRY_KINDS:
        assert f"'{kind}'" in SQL


def test_unique_indexes_enforce_plan_invariants():
    assert "lab_arena_runs_one_active_attempt_uq" in SQL
    assert "WHERE status IN ('pending', 'leased', 'submitted')" in SQL
    assert "lab_arena_runs_one_accepted_uq" in SQL
    assert "lab_arena_runs_claim_request_uq" in SQL
    assert "lab_arena_ledger_payment_reference_uq" in SQL
    assert "lab_arena_ledger_reservation_uq" in SQL
    assert "lab_arena_ledger_terminal_uq" in SQL
    assert "lab_arena_rounds_effective_reward_epoch_uq" in SQL
    assert "CHECK (balance_microusd >= 0)" in SQL
    assert "FOR UPDATE SKIP LOCKED" in SQL
