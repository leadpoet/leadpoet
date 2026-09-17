from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).parents[1]
TEMPLATE = ROOT / "scripts/285-arena-2026-09-17-baseline-recovery.sql.template"
MIGRATION = TEMPLATE.with_suffix("")


def placeholders(value: str) -> set[str]:
    return set(re.findall(r"__[A-Z0-9_]+__", value))


def test_template_is_one_minimal_private_transactional_rpc():
    sql = TEMPLATE.read_text()
    assert placeholders(sql) == {
        "__TERMINAL_BASELINE_LEDGER_COUNT__",
        "__TERMINAL_CONFIGURATION_JSON__",
        "__TERMINAL_PARTICIPANTS_JSON__",
    }
    assert sql.count("CREATE OR REPLACE FUNCTION") == 1
    assert "CREATE TABLE" not in sql
    assert "CREATE TRIGGER" not in sql
    assert "baseline_recovery285_authority" not in sql
    assert "baseline_recovery285_audit" not in sql
    assert "archive_guard" not in sql
    assert "SECURITY DEFINER" in sql
    assert "SET search_path = pg_catalog, public, extensions" in sql
    assert "SET LOCAL lock_timeout = '5s'" in sql
    assert "SET LOCAL statement_timeout = '120s'" in sql
    assert "IN ACCESS EXCLUSIVE MODE" in sql
    assert sql.count("DISABLE TRIGGER USER") == 4
    assert sql.count("ENABLE TRIGGER USER") == 4
    assert "FROM PUBLIC, anon, authenticated, service_role, lab_arena_service" in sql
    assert ") TO lab_arena_service;" in sql


def test_template_preserves_prior_history_and_creates_only_fresh_work():
    sql = TEMPLATE.read_text()
    assert "lab_arena_sep17_recovery284_archive_valid_v1() IS NOT TRUE" in sql
    assert "lab_arena_sep17_recovery284_nonbaseline_ledger_valid_v1()" in sql
    assert "arena-2026-09-17-rerun284archive" in sql
    assert "baseline-2026-09-17-rerun284archive" in sql
    assert "authorized_sep17_recovery284_baseline_archive" in sql
    assert "assignment_id LIKE '%:rerun284'" in sql
    assert "assignment_id LIKE '%:rerun285'" in sql
    assert "FOR v_position IN 0 .. 19 LOOP" in sql
    assert "status = 'pending'" in sql
    assert "output_ref IS NULL" in sql
    assert "v_protected_after IS DISTINCT FROM v_protected_before" in sql
    assert "v_runs_after IS DISTINCT FROM v_runs_before" in sql
    assert "v_ledger_after IS DISTINCT FROM v_ledger_before" in sql
    assert "status_generation = 12" in sql
    assert "stage_generation = 11" in sql
    assert "rewards_enabled', FALSE" in sql


def test_sealed_migration_has_no_placeholders_when_present():
    if not MIGRATION.is_file():
        return
    assert not placeholders(MIGRATION.read_text())
