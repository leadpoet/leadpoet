"""Fail-closed SQL checks for the SOURCE_ADD LLM-only Leg 2 migration."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SQL = (ROOT / "scripts" / "82-research-lab-source-add-llm-only-leg2.sql").read_text(
    encoding="utf-8"
)


def test_migration_aborts_on_legacy_leg2_before_dropping_constraints():
    precheck = SQL.index("legacy_leg2_count")
    abort = SQL.index("SOURCE_ADD Leg 2 migration aborted")
    constraint_drop = SQL.index("DROP CONSTRAINT")
    assert precheck < abort < constraint_drop
    assert "WHERE leg = 2" in SQL
    assert "llm_judge_passed" in SQL
    assert "RAISE EXCEPTION" in SQL
    assert "DELETE FROM" not in SQL.upper()
    assert "UPDATE public.research_lab_source_add_reward_obligations" not in SQL


def test_replacement_constraint_is_strictly_llm_only():
    strict_section = SQL.split(
        "ADD CONSTRAINT research_lab_source_add_reward_leg2_llm_only_check",
        1,
    )[1]
    strict_check = strict_section.split("NOT VALID", 1)[0]
    assert "leg = 1" in strict_check
    assert "llm_judge_passed" in strict_check
    assert "shadow_window_passed" not in strict_check
    assert "ablation_passed" not in strict_check
    assert "VALIDATE CONSTRAINT research_lab_source_add_reward_leg2_llm_only_check" in SQL
