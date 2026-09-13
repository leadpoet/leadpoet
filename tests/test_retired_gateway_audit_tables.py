"""Regression checks for the retired relational gateway audit path."""

from __future__ import annotations

from pathlib import Path

from gateway.tee.supabase_source_v2 import QUERY_POLICIES


ROOT = Path(__file__).resolve().parents[1]

RETIRED_TABLES = {
    "transparency_log",
    "validation_evidence_private",
    "company_information_table",
    "early_access_emails",
    "outreach_email_verifications",
    "suppression_ledger",
    "research_lab_public_loop_cards",
    "research_lab_public_loop_card_events",
    "research_lab_official_baseline_action_attempts_v1",
    "research_lab_official_baseline_action_terminals_v1",
    "research_lab_official_baseline_runs_v1",
    "research_lab_official_baseline_unit_closures_v1",
}

ACTIVE_GATEWAY_SOURCES = (
    "gateway/main.py",
    "gateway/tasks/hourly_batch.py",
    "gateway/tasks/icp_generator.py",
    "gateway/utils/logger.py",
)


def test_active_gateway_audit_and_icp_paths_do_not_call_retired_tables() -> None:
    for relative_path in ACTIVE_GATEWAY_SOURCES:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        for table in RETIRED_TABLES:
            assert f'.table("{table}")' not in source
            assert f".table('{table}')" not in source


def test_historical_source_add_query_graph_stays_measured() -> None:
    assert QUERY_POLICIES["qualification_epoch_assignment"].table == "transparency_log"
    assert QUERY_POLICIES["qualification_leads_by_ids"].table == "leads_private"
    gateway_bootstrap = (ROOT / "gateway/main.py").read_text(encoding="utf-8")
    assert "qualification_admission_v2" not in gateway_bootstrap
    assert "supabase_source_v2" not in gateway_bootstrap


def test_arena_and_validator_authorities_do_not_use_relational_event_logging() -> None:
    current_sources = [ROOT / "gateway/api/arena_proxy.py"]
    current_sources.extend((ROOT / "lab_arena").rglob("*.py"))
    current_sources.extend((ROOT / "validator_tee").rglob("*.py"))
    for path in current_sources:
        source = path.read_text(encoding="utf-8")
        assert "gateway.utils.logger" not in source
        assert ".table(\"transparency_log\")" not in source
        assert ".table('transparency_log')" not in source
