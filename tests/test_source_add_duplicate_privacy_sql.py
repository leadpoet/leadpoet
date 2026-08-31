"""Static release checks for duplicate-private SOURCE_ADD admission."""

from pathlib import Path

from gateway.tee.supabase_schema_preflight_v2 import REQUIRED_SUPABASE_V2_RPCS


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = "scripts/170-research-lab-source-add-duplicate-privacy.sql"
SQL = (ROOT / MIGRATION).read_text(encoding="utf-8")


def _admission_function() -> str:
    return SQL.split("FUNCTION public.research_lab_source_add_admit_v3", 1)[1].split(
        "CREATE OR REPLACE FUNCTION public.research_lab_source_add_admit_v2",
        1,
    )[0]


def test_migration_is_append_only_and_keeps_169_unchanged():
    assert SQL.startswith("-- Classify durable SOURCE_ADD duplicates")
    assert "BEGIN;" in SQL
    assert SQL.rstrip().endswith("COMMIT;")
    assert "CREATE OR REPLACE FUNCTION public.research_lab_source_add_admit_v3" in SQL
    assert "CREATE OR REPLACE FUNCTION public.research_lab_source_add_admit_v2" in SQL
    assert "SOURCE_ADD must be paused before duplicate-privacy migration" in SQL
    assert "DROP TABLE" not in SQL.upper()
    assert "TRUNCATE" not in SQL.upper()
    assert "DELETE FROM" not in SQL.upper()


def test_duplicate_keys_lock_and_classify_before_hotkey_cooldown():
    function = _admission_function()
    lock_loop = function.index("FOR v_lock_key IN")
    hotkey_lock = function.index("'source-add-hotkey:'")
    duplicate_check = function.index("IF EXISTS (")
    cooldown_check = function.index("'status', 'route_cooldown'")
    delegate = function.index("public.research_lab_source_add_admit(")

    assert lock_loop < hotkey_lock < duplicate_check < cooldown_check < delegate
    assert "ORDER BY lock_rank, lock_key" in function
    assert "pg_advisory_xact_lock" in function
    for lock_scope in (
        "source-add-provider-origin:",
        "source-add-identity:",
        "source-add-hotkey:",
        "source-add-submission:",
        "source-add-work:",
    ):
        assert lock_scope in function
    for durable_source in (
        "research_lab_source_add_submission_current",
        "research_lab_source_add_work_items",
        "research_lab_source_add_provider_origin_current",
        "research_lab_source_add_identity_current",
        "research_lab_source_catalog",
    ):
        assert durable_source in function
    assert "jsonb_build_object('status', 'duplicate')" in function


def test_distinct_source_cooldown_uses_durable_provenance_work():
    function = _admission_function()
    assert "MAX(work.created_at)" in function
    assert "work.work_kind = 'provenance'" in function
    assert "work.job_doc->>'admission_kind' = 'miner_submission'" in function
    assert "p_cooldown_seconds * INTERVAL '1 second'" in function
    assert "pg_catalog.clock_timestamp()" in function
    assert "NOW()" not in function
    assert "'cooldown_seconds', p_cooldown_seconds" in function
    assert "'wait_seconds', v_wait_seconds" in function


def test_admission_rpc_is_private_and_restart_required():
    assert "FROM PUBLIC, anon, authenticated" in SQL
    assert "TO service_role" in SQL
    assert (
        MIGRATION,
        "research_lab_source_add_admit_v3",
    ) in REQUIRED_SUPABASE_V2_RPCS
    assert (
        MIGRATION,
        "research_lab_source_add_duplicate_privacy_contract_v1",
    ) in REQUIRED_SUPABASE_V2_RPCS


def test_v2_rolling_compatibility_and_exact_contract_are_bound():
    assert "RETURN public.research_lab_source_add_admit_v3(" in SQL
    assert "p_max_30d,\n        20" in SQL
    assert (
        "FUNCTION public.research_lab_source_add_duplicate_privacy_contract_v1" in SQL
    )
    assert "function_authority_sha256" in SQL
    assert "pg_get_function_identity_arguments" in SQL
    assert "admit_v2_compatibility" in SQL
    assert "clock_timestamp_after_advisory_locks" in SQL
