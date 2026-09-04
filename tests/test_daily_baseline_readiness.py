from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gateway.research_lab import daily_baseline_readiness as readiness
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab.public_baseline_runner import BASELINE_ID


NOW = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)


def _config(*, enabled: bool = True) -> ResearchLabGatewayConfig:
    return ResearchLabGatewayConfig(public_baseline_rebenchmark_enabled=enabled)


def _completed_row() -> dict:
    summaries = [
        {"icp_ref": f"icp-{index}", "score": 62.5, "company_count": 1}
        for index in range(20)
    ]
    return {
        "run_id": "run-1",
        "status": "completed",
        "aggregate_score": 62.5,
        "expected_icp_count": 20,
        "completed_icp_count": 20,
        "per_icp_results": [
            {"icp_ref": item["icp_ref"], "status": "completed", "summary": item}
            for item in summaries
        ],
        "score_summary_doc": {
            "baseline": {"id": BASELINE_ID},
            "aggregate_score": 62.5,
            "per_icp_summaries": summaries,
        },
        "public_report_doc": {
            "schema_version": "public.v1",
            "baseline": {"id": BASELINE_ID},
            "aggregate_score": 62.5,
            "completed_icp_count": 20,
            "per_icp": [
                {
                    "icp_ref": item["icp_ref"],
                    "score": item["score"],
                    "company_count": item["company_count"],
                }
                for item in summaries
            ],
        },
    }


@pytest.mark.asyncio
async def test_complete_public_baseline_releases_daily_gate(monkeypatch):
    calls = []

    async def load_completed_run(**kwargs):
        calls.append(kwargs)
        return _completed_row()

    monkeypatch.setattr(readiness, "load_completed_run", load_completed_run)
    result = await readiness.daily_public_baseline_readiness(
        _config(), now=NOW
    )

    assert calls == [{"benchmark_date": "2026-09-03", "baseline_id": BASELINE_ID}]
    assert result == {
        "available": True,
        "reason": "daily_baseline_published",
        "benchmark_date": "2026-09-03",
        "baseline_run_id": "run-1",
        "baseline_id": BASELINE_ID,
        "aggregate_score": 62.5,
        "completed_icp_count": 20,
    }


@pytest.mark.asyncio
async def test_incomplete_public_baseline_keeps_daily_gate_closed(monkeypatch):
    row = _completed_row()
    row["completed_icp_count"] = 1

    async def load_completed_run(**_kwargs):
        return row

    monkeypatch.setattr(readiness, "load_completed_run", load_completed_run)
    result = await readiness.daily_public_baseline_readiness(_config(), now=NOW)

    assert result == {
        "available": False,
        "reason": "daily_baseline_not_published",
        "benchmark_date": "2026-09-03",
    }


@pytest.mark.asyncio
async def test_malformed_summary_keeps_daily_gate_closed(monkeypatch):
    row = _completed_row()
    row["score_summary_doc"]["per_icp_summaries"] = ["bad", "data"]

    async def load_completed_run(**_kwargs):
        return row

    monkeypatch.setattr(readiness, "load_completed_run", load_completed_run)
    result = await readiness.daily_public_baseline_readiness(_config(), now=NOW)

    assert result["available"] is False
    assert result["reason"] == "daily_baseline_not_published"


@pytest.mark.asyncio
async def test_inconsistent_aggregate_keeps_daily_gate_closed(monkeypatch):
    row = _completed_row()
    row["score_summary_doc"]["aggregate_score"] = 80.0

    async def load_completed_run(**_kwargs):
        return row

    monkeypatch.setattr(readiness, "load_completed_run", load_completed_run)
    result = await readiness.daily_public_baseline_readiness(_config(), now=NOW)

    assert result["available"] is False
    assert result["reason"] == "daily_baseline_not_published"


@pytest.mark.asyncio
async def test_disabled_public_baseline_does_not_query_store(monkeypatch):
    async def forbidden(**_kwargs):
        pytest.fail("disabled daily baseline must not query the store")

    monkeypatch.setattr(readiness, "load_completed_run", forbidden)
    result = await readiness.daily_public_baseline_readiness(
        _config(enabled=False), now=NOW
    )

    assert result == {"available": True, "reason": "daily_baseline_disabled"}
