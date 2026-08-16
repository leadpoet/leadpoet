"""Private Research Lab failure-funnel contract tests."""

from __future__ import annotations

from pathlib import Path

from gateway.research_lab import failure_funnel


MIGRATION = Path("scripts/150-research-lab-failure-funnel-reporting.sql")


def test_failure_funnel_migration_is_service_only_and_read_only():
    sql = MIGRATION.read_text(encoding="utf-8")

    assert "CREATE OR REPLACE FUNCTION public.get_research_lab_failure_funnel" in sql
    assert "SECURITY INVOKER" in sql
    assert (
        "REVOKE ALL ON FUNCTION public.get_research_lab_failure_funnel(UUID, TEXT)"
        in sql
    )
    assert "FROM PUBLIC, anon, authenticated" in sql
    assert (
        "GRANT EXECUTE ON FUNCTION public.get_research_lab_failure_funnel(UUID, TEXT)"
        in sql
    )
    assert "TO service_role" in sql
    assert "CREATE TABLE" not in sql
    assert "DELETE FROM" not in sql
    assert "TRUNCATE" not in sql
    assert "raw provider" not in sql.lower()


async def test_failure_funnel_loader_returns_rpc_report(monkeypatch):
    expected = {
        "schema_version": "research_lab_failure_funnel.v1",
        "ticket_id": "ticket-1",
        "candidate_id": None,
        "stages": [{"stage": "sourcing", "reviewed": 2}],
        "rejections": [],
        "model_revisions": [],
        "telemetry": {"status": "complete"},
    }
    captured = {}

    async def fake_call_rpc(name, params):
        captured.update({"name": name, "params": params})
        return expected

    monkeypatch.setattr(failure_funnel, "call_rpc", fake_call_rpc)
    report = await failure_funnel.build_ticket_failure_funnel("ticket-1")

    assert report == expected
    assert captured == {
        "name": "get_research_lab_failure_funnel",
        "params": {"p_ticket_id": "ticket-1", "p_candidate_id": None},
    }


async def test_failure_funnel_loader_degrades_to_explicit_missing(monkeypatch):
    async def failed_call_rpc(*_args, **_kwargs):
        raise RuntimeError("migration not applied")

    monkeypatch.setattr(failure_funnel, "call_rpc", failed_call_rpc)
    report = await failure_funnel.build_ticket_failure_funnel("ticket-1", "candidate-1")

    assert report["stages"] == []
    assert report["rejections"] == []
    assert report["telemetry"] == {
        "status": "missing",
        "report_available": False,
    }


async def test_failure_funnel_loader_rejects_malformed_rpc_result(monkeypatch):
    async def malformed_call_rpc(*_args, **_kwargs):
        return {"telemetry": None}

    monkeypatch.setattr(failure_funnel, "call_rpc", malformed_call_rpc)
    report = await failure_funnel.build_ticket_failure_funnel("ticket-1")

    assert report["telemetry"] == {
        "status": "missing",
        "report_available": True,
    }
