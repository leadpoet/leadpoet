"""Structural gates for SOURCE_ADD restart-state restoration."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = "scripts/174-research-lab-source-add-restart-state-restore.sql"
SQL = (ROOT / MIGRATION).read_text(encoding="utf-8")


def test_migration_is_transactional_quiet_and_fail_closed() -> None:
    assert SQL.startswith("-- Preserve SOURCE_ADD operator state")
    assert SQL.rstrip().endswith("COMMIT;")
    assert "SET LOCAL lock_timeout = '5s';" in SQL
    assert "IN ACCESS EXCLUSIVE MODE NOWAIT" in SQL
    assert "IN SHARE ROW EXCLUSIVE MODE NOWAIT" in SQL
    assert "SOURCE_ADD must be paused before restart-state migration" in SQL
    assert "SOURCE_ADD restart guard is active during restart-state migration" in SQL
    assert "SOURCE_ADD work is leased during restart-state migration" in SQL
    assert "WHERE work.work_status = 'leased'" in SQL
    assert "NOTIFY pgrst, 'reload schema';" in SQL


def test_guard_bound_restore_state_and_trigger_are_declared() -> None:
    assert (
        "ADD COLUMN IF NOT EXISTS restart_guard_restore_paused BOOLEAN" in SQL
    )
    assert (
        "CREATE OR REPLACE FUNCTION "
        "public.enforce_source_add_restart_restore_pause_v2()" in SQL
    )
    assert "CREATE TRIGGER trg_source_add_restart_restore_pause_v2" in SQL
    assert "NEW.restart_guard_restore_paused := OLD.paused;" in SQL
    assert "NEW.restart_guard_restore_paused := TRUE;" in SQL
    assert "OLD.restart_guard_restore_paused" in SQL
    assert "research_lab_source_add_control_restart_restore_check" in SQL
    assert "restart_guard_commitment = ''" in SQL
    assert "restart_guard_restore_paused IS NULL" in SQL
    assert "restart_guard_restore_paused IS NOT NULL" in SQL


def test_v2_guard_rpcs_restore_only_on_exact_release() -> None:
    required_functions = {
        "research_lab_source_add_restart_guard_state_v2": "()",
        "research_lab_source_add_acquire_restart_guard_v2": "(",
        "research_lab_source_add_release_restart_guard_v2": "(",
        "research_lab_source_add_claim_control_contract_v2": "()",
    }
    for function, suffix in required_functions.items():
        assert (
            f"CREATE OR REPLACE FUNCTION public.{function}{suffix}" in SQL
        )
        assert f"GRANT EXECUTE ON FUNCTION public.{function}" in SQL

    release = SQL.split(
        "CREATE OR REPLACE FUNCTION "
        "public.research_lab_source_add_release_restart_guard_v2(",
        1,
    )[1].split(
        "CREATE OR REPLACE FUNCTION "
        "public.research_lab_source_add_claim_control_contract_v2()",
        1,
    )[0]
    assert "v_existing_commitment <> v_guard_commitment" in release
    assert "v_existing_owner_commitment <> v_owner_commitment" in release
    assert "v_existing_generation <> p_guard_generation" in release
    assert "v_restore_paused IS NULL" in release
    assert "v_final_paused := v_restore_paused" in release
    assert "restart_guard_restore_paused = NULL" in release
    assert "'restored_pre_restart_state', TRUE" in release


def test_contract_documents_all_restart_restore_invariants() -> None:
    for contract_fragment in (
        "'acquire_captures_pre_restart_paused', TRUE",
        "'renewal_preserves_restore_state', TRUE",
        "'expired_takeover_preserves_restore_state', TRUE",
        "'operator_pause_wins', TRUE",
        "'release_restores_pre_restart_state', TRUE",
        "'failed_restart_keeps_paused', TRUE",
        "'migration_requires_paused', TRUE",
        "'migration_requires_zero_leased', TRUE",
        "'migration_requires_guard_clear', TRUE",
    ):
        assert contract_fragment in SQL
    assert "'restore_state_column', 'restart_guard_restore_paused'" in SQL
    assert "'restore_paused'" in SQL
    assert "'service_role_callable'" in SQL
    assert "'anon_callable'" in SQL
    assert "'authenticated_callable'" in SQL


def test_submission_credential_check_is_non_retroactive_defense_in_depth() -> None:
    assert (
        "research_lab_source_add_submission_no_credential_material_v2" in SQL
    )
    assert "submission_doc::TEXT !~*" in SQL
    assert "api[[:space:]_-]*key" in SQL
    assert "client[[:space:]_-]*secret" in SQL
    assert "(authorization|proxy-authorization)" in SQL
    assert "(bearer|basic|api([[:space:]_-]*key)?)" in SQL
    assert ") NOT VALID;" in SQL
