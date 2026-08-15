from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from fastapi import HTTPException
import pytest

from gateway.research_lab import api
from gateway.research_lab import daily_baseline_readiness as readiness_mod
from gateway.research_lab.config import ResearchLabGatewayConfig
from research_lab.eval.conditional_validation import (
    build_conditional_category_assignment,
)


NOW = datetime(2026, 8, 11, 6, 0, tzinfo=timezone.utc)
WINDOW_HASH = "sha256:" + "c" * 64


def _config(*, baseline_enabled: bool = True) -> ResearchLabGatewayConfig:
    return ResearchLabGatewayConfig(
        private_baseline_rebenchmark_enabled=baseline_enabled,
    )


def _active_model() -> SimpleNamespace:
    return SimpleNamespace(
        artifact=SimpleNamespace(
            model_artifact_hash="sha256:" + "a" * 64,
            manifest_hash="sha256:" + "b" * 64,
        )
    )


def _durable_rows(config: ResearchLabGatewayConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = config.conditional_validation_policy()
    items = [
        {
            "icp_ref": f"icp-{index:02d}",
            "icp_hash": "sha256:" + f"{index:064x}",
            "industry": f"industry-{index:02d}",
            "sub_industry": f"sub-{index:02d}",
            "product_service": f"service-{index:02d}",
            "intent_signals": [f"signal-{index:02d}"],
            "set_id": index,
            "day_index": 1,
            "day_rank": index,
            "cohort": "fresh" if index <= 20 else "retained",
            "icp": {
                "industry": f"industry-{index:02d}",
                "sub_industry": f"sub-{index:02d}",
                "product_service": f"service-{index:02d}",
                "intent_signals": [f"signal-{index:02d}"],
            },
        }
        for index in range(1, policy.total_icps + 1)
    ]
    summaries = [
        {
            "icp_ref": item["icp_ref"],
            "icp_hash": item["icp_hash"],
            "score": float(index),
        }
        for index, item in enumerate(items, start=1)
    ]
    assignment = build_conditional_category_assignment(
        rolling_window_hash=WINDOW_HASH,
        benchmark_items=items,
        per_icp_summaries=summaries,
        policy=policy,
        baseline_serving_model_version_hash="sha256:" + "d" * 64,
    )
    active = _active_model().artifact
    benchmark = {
        "benchmark_bundle_id": "bundle-1",
        "benchmark_date": "2026-08-11",
        "private_model_artifact_hash": active.model_artifact_hash,
        "private_model_manifest_hash": active.manifest_hash,
        "rolling_window_hash": WINDOW_HASH,
        "aggregate_score": 20.5,
        "benchmark_quality": "passed",
        "current_benchmark_status": "completed",
        "score_summary_doc": {
            "per_icp_summaries": summaries,
            "category_assignment": assignment,
        },
        "benchmark_attempt": 1,
    }
    report = {
        "report_id": "report-1",
        "benchmark_date": "2026-08-11",
        "benchmark_bundle_id": "bundle-1",
        "private_model_artifact_hash": active.model_artifact_hash,
        "private_model_manifest_hash": active.manifest_hash,
        "rolling_window_hash": WINDOW_HASH,
        "benchmark_quality": "passed",
        "current_report_status": "published",
        "benchmark_attempt": 1,
    }
    return benchmark, report


@pytest.mark.asyncio
async def test_operator_pause_precedes_daily_baseline_lookup(monkeypatch):
    async def paused_state():
        return {"paused": True, "reason": "operator_pause", "status_at": "now"}

    async def forbidden_readiness(_config):
        pytest.fail("operator pause must fail before baseline lookup")

    monkeypatch.setattr(api, "get_autoresearch_maintenance_state", paused_state)
    monkeypatch.setattr(api, "autoresearch_daily_baseline_readiness", forbidden_readiness)

    with pytest.raises(HTTPException) as raised:
        await api._require_autoresearch_not_paused(_config())

    assert raised.value.status_code == 503
    assert raised.value.detail["code"] == "research_lab_maintenance_paused"


@pytest.mark.asyncio
async def test_resume_intent_stays_effectively_held_until_daily_publication(monkeypatch):
    async def resumed_state():
        return {"paused": False, "reason": "operator_resume"}

    async def held_readiness(_config):
        return {
            "available": False,
            "reason": "daily_baseline_not_published",
            "benchmark_date": "2026-08-11",
        }

    monkeypatch.setattr(api, "get_autoresearch_maintenance_state", resumed_state)
    monkeypatch.setattr(api, "autoresearch_daily_baseline_readiness", held_readiness)

    with pytest.raises(HTTPException) as raised:
        await api._require_autoresearch_not_paused(_config())

    assert raised.value.status_code == 503
    assert raised.value.detail == {
        "code": "research_lab_daily_baseline_not_ready",
        "message": (
            "Research Lab auto-research is waiting for the current daily "
            "baseline publication"
        ),
        "reason": "daily_baseline_not_published",
        "benchmark_date": "2026-08-11",
    }


@pytest.mark.asyncio
async def test_complete_exact_active_model_baseline_releases_admission(monkeypatch):
    config = _config()
    active = _active_model()
    benchmark, report = _durable_rows(config)

    async def load_active(_config, *, register_bootstrap):
        assert register_bootstrap is False
        return active

    async def select_rows(table, *, columns, filters, order_by, limit):
        assert ("benchmark_date", "2026-08-11") in filters
        assert (
            "private_model_artifact_hash",
            active.artifact.model_artifact_hash,
        ) in filters
        assert (
            "private_model_manifest_hash",
            active.artifact.manifest_hash,
        ) in filters
        assert order_by == (("benchmark_attempt", True), ("created_at", True))
        assert limit == 10
        if table == "research_lab_private_model_benchmark_current":
            assert ("current_benchmark_status", "completed") in filters
            return [benchmark]
        if table == "research_lab_public_benchmark_report_current":
            assert ("current_report_status", "published") in filters
            return [report]
        raise AssertionError(table)

    monkeypatch.setattr(readiness_mod, "load_active_private_model", load_active)
    monkeypatch.setattr(readiness_mod, "select_many", select_rows)

    readiness = await readiness_mod.autoresearch_daily_baseline_readiness(
        config,
        now=NOW,
    )

    assert readiness == {
        "available": True,
        "reason": "daily_baseline_published",
        "benchmark_date": "2026-08-11",
        "report_id": "report-1",
        "benchmark_bundle_id": "bundle-1",
        "rolling_window_hash": WINDOW_HASH,
    }


@pytest.mark.asyncio
async def test_staging_evidence_commits_every_icp_and_exact_category_assignment(
    monkeypatch,
):
    config = _config()
    benchmark, report = _durable_rows(config)

    async def load_active(_config, *, register_bootstrap):
        assert register_bootstrap is False
        return _active_model()

    async def select_rows(table, **_kwargs):
        if table == "research_lab_private_model_benchmark_current":
            return [benchmark]
        if table == "research_lab_public_benchmark_report_current":
            return [report]
        raise AssertionError(table)

    monkeypatch.setattr(readiness_mod, "load_active_private_model", load_active)
    monkeypatch.setattr(readiness_mod, "select_many", select_rows)

    readiness = await readiness_mod.autoresearch_daily_baseline_readiness(
        config,
        now=NOW,
        include_commitments=True,
    )

    commitments = readiness["completion_commitments"]
    policy = config.conditional_validation_policy().to_dict()
    assert commitments["all_icp_count"] == policy["total_icps"]
    assert commitments["category_counts"] == {
        "public": policy["public_total_icps"],
        "private": policy["private_total_icps"],
        "conditional": policy["conditional_total_icps"],
    }
    assert commitments["category_strength_counts"] == {
        "public": {
            "weak": policy["public_weak_total"],
            "strong": policy["public_strong_total"],
        },
        "private": {
            "weak": policy["private_weak_total"],
            "strong": policy["private_strong_total"],
        },
        "conditional": {"center": policy["conditional_total_icps"]},
    }
    assert commitments["category_assignment_hash"].startswith("sha256:")
    assert commitments["per_icp_summaries_hash"].startswith("sha256:")
    assert commitments["conditional_policy_hash"] == policy["policy_hash"]


@pytest.mark.asyncio
async def test_assignment_score_divergence_remains_fail_closed(monkeypatch):
    config = _config()
    benchmark, report = _durable_rows(config)
    benchmark["score_summary_doc"]["per_icp_summaries"][0]["score"] = 99.0

    async def load_active(_config, *, register_bootstrap):
        return _active_model()

    async def select_rows(table, **_kwargs):
        if table == "research_lab_private_model_benchmark_current":
            return [benchmark]
        if table == "research_lab_public_benchmark_report_current":
            return [report]
        raise AssertionError(table)

    monkeypatch.setattr(readiness_mod, "load_active_private_model", load_active)
    monkeypatch.setattr(readiness_mod, "select_many", select_rows)

    readiness = await readiness_mod.autoresearch_daily_baseline_readiness(
        config,
        now=NOW,
        include_commitments=True,
    )

    assert readiness == {
        "available": False,
        "reason": "daily_baseline_not_published",
        "benchmark_date": "2026-08-11",
    }


@pytest.mark.asyncio
async def test_staging_commitments_do_not_change_normal_admission(monkeypatch):
    config = _config()
    benchmark, report = _durable_rows(config)
    benchmark["score_summary_doc"]["per_icp_summaries"][0]["score"] = 99.0

    async def load_active(_config, *, register_bootstrap):
        return _active_model()

    async def select_rows(table, **_kwargs):
        if table == "research_lab_private_model_benchmark_current":
            return [benchmark]
        if table == "research_lab_public_benchmark_report_current":
            return [report]
        raise AssertionError(table)

    monkeypatch.setattr(readiness_mod, "load_active_private_model", load_active)
    monkeypatch.setattr(readiness_mod, "select_many", select_rows)

    readiness = await readiness_mod.autoresearch_daily_baseline_readiness(
        config,
        now=NOW,
    )

    assert readiness["available"] is True
    assert "completion_commitments" not in readiness


@pytest.mark.asyncio
async def test_partial_assignment_remains_fail_closed(monkeypatch):
    config = _config()
    benchmark, report = _durable_rows(config)
    benchmark["score_summary_doc"]["category_assignment"]["items"].pop()

    async def load_active(_config, *, register_bootstrap):
        assert register_bootstrap is False
        return _active_model()

    async def select_rows(table, **_kwargs):
        if table == "research_lab_private_model_benchmark_current":
            return [benchmark]
        if table == "research_lab_public_benchmark_report_current":
            return [report]
        raise AssertionError(table)

    monkeypatch.setattr(readiness_mod, "load_active_private_model", load_active)
    monkeypatch.setattr(readiness_mod, "select_many", select_rows)

    readiness = await readiness_mod.autoresearch_daily_baseline_readiness(
        config,
        now=NOW,
    )

    assert readiness == {
        "available": False,
        "reason": "daily_baseline_not_published",
        "benchmark_date": "2026-08-11",
    }


@pytest.mark.asyncio
async def test_gate_unavailability_fails_closed_without_error_detail(monkeypatch):
    async def unavailable(*_args, **_kwargs):
        raise RuntimeError("credential-like upstream detail")

    monkeypatch.setattr(readiness_mod, "load_active_private_model", unavailable)

    readiness = await readiness_mod.autoresearch_daily_baseline_readiness(
        _config(),
        now=NOW,
    )

    assert readiness == {
        "available": False,
        "reason": "daily_baseline_gate_unavailable",
        "benchmark_date": "2026-08-11",
    }


@pytest.mark.asyncio
async def test_disabled_daily_baseline_preserves_existing_admission_behavior(monkeypatch):
    async def forbidden(*_args, **_kwargs):
        pytest.fail("disabled baseline must not perform readiness I/O")

    monkeypatch.setattr(readiness_mod, "load_active_private_model", forbidden)
    monkeypatch.setattr(readiness_mod, "select_many", forbidden)

    readiness = await readiness_mod.autoresearch_daily_baseline_readiness(
        _config(baseline_enabled=False),
        now=NOW,
    )

    assert readiness == {
        "available": True,
        "reason": "daily_baseline_disabled",
    }
