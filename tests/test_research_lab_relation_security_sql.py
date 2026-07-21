from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT
    / "scripts"
    / "113-restrict-research-lab-observability-and-source-rewards.sql"
)


def test_internal_reward_tables_are_service_role_only():
    sql = MIGRATION.read_text(encoding="utf-8")

    for relation in (
        "research_lab_source_add_reward_obligations",
        "research_lab_source_add_reward_events",
    ):
        assert f"ALTER TABLE public.{relation}\n    ENABLE ROW LEVEL SECURITY;" in sql
        assert f"ON public.{relation};\nCREATE POLICY service_role_all" in sql

    assert "FROM PUBLIC, anon, authenticated;" in sql
    assert "TO service_role;" in sql


def test_internal_views_are_security_invoker_and_service_role_only():
    sql = MIGRATION.read_text(encoding="utf-8")

    for relation in (
        "research_lab_source_add_reward_current",
        "research_lab_icp_churn_reversal_report",
    ):
        assert (
            f"ALTER VIEW public.{relation}\n"
            "    SET (security_invoker = true);"
        ) in sql
        assert f"public.{relation}" in sql

    assert "FROM PUBLIC, anon, authenticated;" in sql
    assert "GRANT SELECT ON TABLE" in sql
