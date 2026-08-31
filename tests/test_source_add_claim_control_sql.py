"""Static release checks for serialized SOURCE_ADD claim control."""

from pathlib import Path

from gateway.tee.supabase_schema_preflight_v2 import REQUIRED_SUPABASE_V2_RPCS


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = "scripts/172-research-lab-source-add-claim-control.sql"
SQL = (ROOT / MIGRATION).read_text(encoding="utf-8")


def _function_body(name: str, next_name: str) -> str:
    return SQL.split(f"FUNCTION public.{name}", 1)[1].split(
        f"CREATE OR REPLACE FUNCTION public.{next_name}", 1
    )[0]


def test_claim_control_migration_is_transactional_and_nondestructive() -> None:
    assert SQL.startswith("-- Linearize SOURCE_ADD work claims")
    assert "BEGIN;" in SQL
    assert SQL.rstrip().endswith("COMMIT;")
    assert "SET LOCAL lock_timeout = '5s'" in SQL
    assert "DROP TABLE" not in SQL.upper()
    assert "TRUNCATE" not in SQL.upper()
    assert "DELETE FROM" not in SQL.upper()


def test_application_requires_paused_and_every_lease_drained() -> None:
    handoff = SQL.split("DO $quiet_pause$", 1)[1].split(
        "$quiet_pause$;", 1
    )[0]
    assert "IN ACCESS EXCLUSIVE MODE NOWAIT" in handoff
    assert "IN SHARE ROW EXCLUSIVE MODE NOWAIT" in handoff
    assert (
        "SOURCE_ADD must be paused before claim-control migration" in handoff
    )
    assert "WHERE work.work_status = 'leased'" in handoff
    assert "work_kind" not in handoff
    assert "lease_expires_at" not in handoff


def test_claim_locks_control_before_reading_pause_or_work() -> None:
    claim = _function_body(
        "research_lab_source_add_claim_work",
        "research_lab_source_add_restart_quiescence_v1",
    )
    control_lock = claim.index("hashtextextended('source-add-control', 0)")
    paused_read = claim.index(
        "SELECT paused\n        FROM public.research_lab_source_add_control"
    )
    work_read = claim.index(
        "SELECT w.* INTO v_row\n    FROM public.research_lab_source_add_work_items"
    )
    lease_write = claim.index(
        "UPDATE public.research_lab_source_add_work_items\n"
        "    SET work_status = 'leased'"
    )
    assert control_lock < paused_read < work_read < lease_write
    assert "COALESCE((" in claim
    assert "), TRUE)" in claim


def test_quiescence_uses_same_lock_and_counts_expired_leases() -> None:
    quiescence = _function_body(
        "research_lab_source_add_restart_quiescence_v1",
        "research_lab_source_add_claim_control_contract_v1",
    )
    control_lock = quiescence.index(
        "hashtextextended('source-add-control', 0)"
    )
    paused_read = quiescence.index(
        "SELECT\n        control.paused,"
    )
    leased_read = quiescence.index("SELECT COUNT(*)::INTEGER")
    result = quiescence.index("RETURN pg_catalog.jsonb_build_object(")
    assert control_lock < paused_read < leased_read < result
    assert "WHERE work.work_status = 'leased'" in quiescence
    assert "lease_expires_at" not in quiescence
    for field in (
        "'schema_version', 'leadpoet.source_add_restart_quiescence.v1'",
        "'paused'",
        "'guard_active'",
        "'guard_matches'",
        "'guard_commitment'",
        "'guard_expires_at'",
        "'leased_work_count'",
        "'quiescent'",
    ):
        assert field in quiescence


def test_exact_authority_and_private_rpc_surface_are_restart_required() -> None:
    for function_name in (
        "research_lab_source_add_claim_work",
        "research_lab_source_add_set_paused",
        "research_lab_source_add_acquire_restart_guard_v1",
        "research_lab_source_add_restart_quiescence_v1",
        "research_lab_source_add_release_restart_guard_v1",
        "research_lab_source_add_claim_control_contract_v1",
    ):
        assert (MIGRATION, function_name) in REQUIRED_SUPABASE_V2_RPCS
    assert "function_authority_sha256" in SQL
    assert "pg_get_function_identity_arguments" in SQL
    assert "'all_leased_regardless_of_expiry'" in SQL
    assert "'lock_before_paused_read', TRUE" in SQL
    assert "FROM PUBLIC, anon, authenticated" in SQL
    assert "TO service_role" in SQL
    for authority in (
        "admission_guard",
        "acquire_restart_guard_v1",
        "claim_work",
        "contract_v1",
        "pause",
        "release_restart_guard_v1",
        "restart_quiescence_v1",
    ):
        assert f"'{authority}'" in SQL


def test_restart_guard_is_bounded_exact_and_never_auto_resumes() -> None:
    assert "restart_guard_commitment TEXT NOT NULL DEFAULT ''" in SQL
    assert "research_lab_source_add_control_restart_guard_check" in SQL
    assert "NOT BETWEEN 60 AND 3600" in SQL
    assert "^source_add_restart_guard:[0-9a-f]{64}$" in SQL
    assert "sha256_utf8_guard_id" in SQL
    assert "active_guard_replay_extends_lease', FALSE" in SQL
    assert "explicit_reacquire_then_exact_release" in SQL
    assert "release_keeps_paused', TRUE" in SQL
    assert (
        "IF p_paused IS NOT TRUE AND v_guard_commitment <> '' THEN" in SQL
    )
    assert "restart guard identity does not match" in SQL
    release = _function_body(
        "research_lab_source_add_release_restart_guard_v1",
        "research_lab_source_add_restart_quiescence_v1",
    )
    assert "SET paused = TRUE" in release
    assert "restart_guard_commitment = ''" in release
