"""Static safety contract for the additive inner-loop activation migration."""

from pathlib import Path


SQL = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "93-research-lab-inner-loop-activation.sql"
).read_text(encoding="utf-8")


def test_activation_state_is_append_only_private_and_restart_stable():
    assert "CREATE TABLE IF NOT EXISTS public.research_lab_inner_loop_activation_events" in SQL
    assert "BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY" in SQL
    assert "event_hash          TEXT NOT NULL UNIQUE" in SQL
    assert "prevent_research_lab_inner_loop_activation_mutation" in SQL
    assert "BEFORE UPDATE OR DELETE" in SQL
    assert "ENABLE ROW LEVEL SECURITY" in SQL
    assert "REVOKE ALL ON TABLE public.research_lab_inner_loop_activation_events" in SQL
    assert "FROM anon, authenticated" in SQL
    assert "uq_research_lab_inner_loop_run_observed" in SQL
    assert "WHERE event_type = 'run_observed' AND run_id IS NOT NULL" in SQL


def test_transition_rpc_serializes_workers_and_compares_expected_phase():
    assert "append_research_lab_inner_loop_activation_event" in SQL
    assert "pg_catalog.pg_advisory_xact_lock" in SQL
    assert "expected_current_phase TEXT DEFAULT NULL" in SQL
    assert "research_lab_inner_loop_phase_conflict" in SQL
    assert "USING ERRCODE = '40001'" in SQL
    assert "SECURITY DEFINER" in SQL
    assert "TO service_role" in SQL
    assert "NOTIFY pgrst, 'reload schema'" in SQL
