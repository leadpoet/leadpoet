"""Structural gates for the current SOURCE_ADD Leg 1 release economics."""

from pathlib import Path

from gateway.tee.supabase_schema_preflight_v2 import REQUIRED_SUPABASE_V2_RPCS


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = "scripts/173-research-lab-source-add-leg1-release-policy.sql"
SQL = (ROOT / MIGRATION).read_text(encoding="utf-8")


def test_migration_is_transactional_quiet_and_non_retroactive():
    assert SQL.startswith("-- Set the production SOURCE_ADD Leg 1 release policy")
    assert SQL.rstrip().endswith("COMMIT;")
    assert "IN ACCESS EXCLUSIVE MODE NOWAIT" in SQL
    assert "IN SHARE ROW EXCLUSIVE MODE NOWAIT" in SQL
    assert "SOURCE_ADD must be paused before Leg 1 policy migration" in SQL
    assert "SOURCE_ADD work is leased during Leg 1 policy migration" in SQL
    assert "WHERE work_status = 'leased'" in SQL
    assert "NOTIFY pgrst, 'reload schema';" in SQL
    upper = SQL.upper()
    assert "UPDATE PUBLIC.RESEARCH_LAB_SOURCE_ADD_REWARD_OBLIGATIONS" not in upper
    assert "DELETE FROM PUBLIC.RESEARCH_LAB_SOURCE_ADD_REWARD_OBLIGATIONS" not in upper
    assert "TRUNCATE" not in upper


def test_database_owns_point_two_percent_and_fifty_slot_policy():
    reserve = SQL.split(
        "CREATE OR REPLACE FUNCTION public.research_lab_source_add_reserve_leg1_slot_v3(",
        1,
    )[1].split(
        "CREATE OR REPLACE FUNCTION public.research_lab_source_add_finalize_leg1_v3(",
        1,
    )[0]
    finalize = SQL.split(
        "CREATE OR REPLACE FUNCTION public.research_lab_source_add_finalize_leg1_v3(",
        1,
    )[1].split(
        "CREATE OR REPLACE FUNCTION public.research_lab_source_add_post_accept_leg1_contract_v2()",
        1,
    )[0]
    assert reserve.count("\n            50,") == 1
    assert "\n        50," in reserve
    assert "p_daily_cap" in reserve
    assert "<> 0.2" in finalize
    assert "\n        50," in finalize
    assert "<> 1.0" not in finalize
    assert "'daily_cap', 50" in SQL
    assert "'leg1_alpha_percent', 0.2" in SQL
    assert "'leg1_reward_epochs', 20" in SQL


def test_current_policy_functions_are_attributed_to_migration_173():
    functions = {
        function
        for migration, function in REQUIRED_SUPABASE_V2_RPCS
        if migration == MIGRATION
    }
    assert functions == {
        "research_lab_source_add_reserve_leg1_slot_v3",
        "research_lab_source_add_finalize_leg1_v3",
        "research_lab_source_add_post_accept_leg1_contract_v2",
    }
    for function in functions:
        assert f"CREATE OR REPLACE FUNCTION public.{function}(" in SQL
        assert f"GRANT EXECUTE ON FUNCTION public.{function}(" in SQL
