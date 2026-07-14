"""Failure isolation tests for the read-only inner-loop monitor."""

from __future__ import annotations

import asyncio

from gateway.research_lab import store
from scripts import research_lab_inner_loop_monitor as monitor


def test_monitor_reports_one_unavailable_source_without_losing_other_data(
    monkeypatch,
):
    async def select_all(table, **_kwargs):
        if table == "research_lab_inner_loop_activation_events":
            raise RuntimeError("relation does not exist")
        if table == "research_lab_results_ledger":
            return [
                {
                    "ledger_row_id": "ledger-1",
                    "status": "keep",
                    "delta_vs_parent": 0.5,
                    "cost_usd": 2.0,
                    "created_at": "2026-07-13T00:00:00+00:00",
                }
            ]
        return []

    monkeypatch.setattr(store, "select_all", select_all)
    monkeypatch.delenv("RESEARCH_LAB_DEV_SNAPSHOT_URI", raising=False)
    result = asyncio.run(monitor._collect(14))

    assert "activation_events" in result["data_source_errors"]
    assert "monitor_data_source_error:activation_events" in result["alerts"]
    assert result["ledger_yield"]["status_counts"] == {"keep": 1}
    assert result["ledger_yield"]["kept_delta_sum"] == 0.5
    assert result["ledger_yield"]["total_cost_usd"] == 2.0
