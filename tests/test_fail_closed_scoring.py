"""Fail-closed scoring: health-gate quarantine, baseline publication gates,
and the day-jump default.

Covers the priority-#5 hardening:
- the scoring health gate honors ``scoring_health_gate_enabled`` and returns a
  ``quarantine`` decision on threshold violations instead of observe-only,
- a quarantined candidate records a ``scoring_health_quarantined`` promotion
  event and is never handed to the promotion controller,
- a baseline whose own health gate failed refuses publication, and the
  day-over-day jump limit is enforced by default.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest

from gateway.research_lab import scoring_worker as sw


def _worker(**config_overrides) -> sw.ResearchLabGatewayScoringWorker:
    worker = object.__new__(sw.ResearchLabGatewayScoringWorker)
    defaults = dict(
        scoring_health_gate_enabled=True,
        scoring_health_max_reference_runtime_failure_rate=0.25,
        scoring_health_max_candidate_runtime_failure_rate=0.25,
        scoring_health_max_reference_zero_company_rate=1.0,
        scoring_health_max_candidate_zero_company_rate=1.0,
        scoring_health_max_provider_error_rate=0.10,
        scoring_health_max_timeout_rate=0.10,
        improvement_threshold_points=1.0,
    )
    defaults.update(config_overrides)
    worker.config = SimpleNamespace(**defaults)
    worker.worker_ref = "test-scorer-0"
    return worker


def _bundle(provider_error_rate: float = 0.0, **health) -> dict:
    doc = {"provider_error_rate": provider_error_rate, **health}
    return {"scoring_health": doc, "icp_set_hash": "sha256:w", "aggregates": {}}


# ---------------------------------------------------------------------------
# _scoring_health_gate_result decisions
# ---------------------------------------------------------------------------


def test_gate_quarantines_on_violation_when_enabled():
    result = _worker()._scoring_health_gate_result(_bundle(provider_error_rate=0.5))
    assert result["enabled"] is True
    assert result["decision"] == "quarantine"
    assert result["would_quarantine"] is True
    assert any(v["metric"] == "provider_error_rate" for v in result["violations"])


def test_gate_observes_only_when_disabled():
    result = _worker(scoring_health_gate_enabled=False)._scoring_health_gate_result(
        _bundle(provider_error_rate=0.5)
    )
    assert result["enabled"] is False
    assert result["decision"] == "observe_only"
    assert result["would_quarantine"] is True  # still recorded for audit


def test_gate_observes_only_without_violations():
    result = _worker()._scoring_health_gate_result(_bundle(provider_error_rate=0.0))
    assert result["decision"] == "observe_only"
    assert result["violations"] == []


def test_gate_zero_company_rate_not_a_violation_at_default_threshold():
    # Legitimate zero-company outcomes must not quarantine at the defaults.
    result = _worker()._scoring_health_gate_result(
        _bundle(candidate_zero_company_rate=0.9, reference_zero_company_rate=0.9)
    )
    assert result["decision"] == "observe_only"


def test_gate_timeout_violation_quarantines():
    result = _worker()._scoring_health_gate_result(_bundle(timeout_rate=0.5))
    assert result["decision"] == "quarantine"


# ---------------------------------------------------------------------------
# quarantine recorder: event written, promotion withheld
# ---------------------------------------------------------------------------


def test_quarantine_recorder_writes_promotion_event_and_skips_promotion():
    worker = _worker()
    candidate = {
        "candidate_id": "cand-1",
        "parent_artifact_hash": "sha256:parent",
        "candidate_kind": "auto_research",
    }
    gate = {"decision": "quarantine", "violations": [{"metric": "provider_error_rate"}]}
    recorded = {}

    async def fake_promotion_event(**kwargs):
        recorded.update(kwargs)
        return {"event_id": "e1"}

    with mock.patch.object(sw, "create_candidate_promotion_event", fake_promotion_event), \
            mock.patch.object(
                sw.ResearchLabGatewayScoringWorker,
                "_maybe_promote_scored_candidate",
                side_effect=AssertionError("promotion must not run for quarantined candidates"),
            ):
        result = asyncio.run(
            worker._record_scoring_health_quarantined(
                candidate=candidate,
                score_bundle_row={"score_bundle_id": "sb-1"},
                score_bundle=_bundle(),
                scoring_health_gate=gate,
            )
        )
    assert result == {"status": "scoring_health_quarantined"}
    assert recorded["event_type"] == "scoring_health_quarantined"
    assert recorded["promotion_status"] == "rejected"
    assert recorded["candidate_id"] == "cand-1"
    assert recorded["event_doc"]["scoring_health_gate"] == gate


# ---------------------------------------------------------------------------
# baseline publication gates
# ---------------------------------------------------------------------------


def _health(gate_passed: bool, unresolved: int = 0) -> dict:
    return {
        "gate_passed": gate_passed,
        "unresolved_provider_errors": unresolved,
        "max_unresolved_icps": 2,
        "decision": "observe_only",
    }


def test_baseline_gate_blocks_degraded_publication():
    with pytest.raises(sw.BaselineHealthGateFailure) as excinfo:
        sw._enforce_baseline_publication_gates(
            baseline_health=_health(False, unresolved=5),
            aggregate_score=24.46,
            day_jump_points=None,
            health_gate_enforced=True,
            max_day_jump=15.0,
        )
    assert "unresolved_provider_errors_gate_failed" in str(excinfo.value)
    assert excinfo.value.baseline_health["unresolved_provider_errors"] == 5


def test_baseline_gate_allows_healthy_publication():
    sw._enforce_baseline_publication_gates(
        baseline_health=_health(True),
        aggregate_score=34.35,
        day_jump_points=2.0,
        health_gate_enforced=True,
        max_day_jump=15.0,
    )


def test_baseline_gate_observe_only_when_disabled():
    sw._enforce_baseline_publication_gates(
        baseline_health=_health(False, unresolved=5),
        aggregate_score=24.46,
        day_jump_points=None,
        health_gate_enforced=False,
        max_day_jump=None,
    )


def test_day_jump_gate_blocks_large_swing():
    with pytest.raises(sw.BaselineHealthGateFailure) as excinfo:
        sw._enforce_baseline_publication_gates(
            baseline_health=_health(True),
            aggregate_score=10.0,
            day_jump_points=-24.0,
            health_gate_enforced=True,
            max_day_jump=15.0,
        )
    assert "day_over_day_jump_gate_failed" in str(excinfo.value)
    assert excinfo.value.baseline_health["gate_passed"] is False


def test_day_jump_gate_allows_swing_within_limit():
    sw._enforce_baseline_publication_gates(
        baseline_health=_health(True),
        aggregate_score=34.35,
        day_jump_points=9.9,
        health_gate_enforced=True,
        max_day_jump=15.0,
    )


# ---------------------------------------------------------------------------
# day-jump threshold default
# ---------------------------------------------------------------------------


def test_day_jump_threshold_defaults_enforced(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_BASELINE_MAX_DAY_JUMP_POINTS", raising=False)
    assert sw._baseline_max_day_jump_points() == sw.DEFAULT_BASELINE_MAX_DAY_JUMP_POINTS


@pytest.mark.parametrize("raw", ["0", "0.0", "off", "none", "disabled", "OFF"])
def test_day_jump_threshold_explicit_disable(monkeypatch, raw):
    monkeypatch.setenv("RESEARCH_LAB_BASELINE_MAX_DAY_JUMP_POINTS", raw)
    assert sw._baseline_max_day_jump_points() is None


def test_day_jump_threshold_custom_value(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_BASELINE_MAX_DAY_JUMP_POINTS", "-25.5")
    assert sw._baseline_max_day_jump_points() == 25.5


def test_day_jump_threshold_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_BASELINE_MAX_DAY_JUMP_POINTS", "not-a-number")
    assert sw._baseline_max_day_jump_points() == sw.DEFAULT_BASELINE_MAX_DAY_JUMP_POINTS


# ---------------------------------------------------------------------------
# config defaults
# ---------------------------------------------------------------------------


def test_config_defaults_fail_closed(monkeypatch):
    from gateway.research_lab.config import ResearchLabGatewayConfig

    for name in (
        "RESEARCH_LAB_SCORING_HEALTH_GATE_ENABLED",
        "RESEARCH_LAB_BASELINE_HEALTH_GATE_ENFORCED",
        "RESEARCH_LAB_SCORING_HEALTH_MAX_REFERENCE_ZERO_COMPANY_RATE",
        "RESEARCH_LAB_SCORING_HEALTH_MAX_CANDIDATE_ZERO_COMPANY_RATE",
    ):
        monkeypatch.delenv(name, raising=False)
    config = ResearchLabGatewayConfig.from_env()
    assert config.scoring_health_gate_enabled is True
    assert config.baseline_health_gate_enforced is True
    assert config.scoring_health_max_reference_zero_company_rate == 1.0
    assert config.scoring_health_max_candidate_zero_company_rate == 1.0


# ---------------------------------------------------------------------------
# provider-recovery rescore (quarantine requeue)
# ---------------------------------------------------------------------------


def _recovery_worker(monkeypatch, **overrides):
    worker = _worker(**overrides)
    monkeypatch.setenv("RESEARCH_LAB_QUARANTINE_RECOVERY_INTERVAL_SECONDS", "0")
    return worker


def test_recovery_requeues_quarantined_scored_candidate(monkeypatch):
    worker = _recovery_worker(monkeypatch)
    writes = []

    async def fake_select_many(table, **kwargs):
        filters = dict(kwargs.get("filters") or ())
        if table == "research_lab_candidate_promotion_events":
            if filters.get("event_type") == "scoring_health_quarantined":
                return [{"candidate_id": "cand-q", "created_at": "2026-07-10T10:00:00"}]
            # latest promotion for the candidate is still the quarantine
            return [{"event_type": "scoring_health_quarantined", "created_at": "2026-07-10T10:00:00"}]
        if table == "research_lab_candidate_evaluation_events":
            return []  # no prior recovery attempts
        raise AssertionError(f"unexpected table {table}")

    async def fake_select_one(table, **kwargs):
        assert table == "research_lab_candidate_evaluation_current"
        return {
            "candidate_id": "cand-q",
            "current_candidate_status": "scored",
            "run_id": "run-1",
            "ticket_id": "tick-1",
        }

    async def fake_create_event(**kwargs):
        writes.append(kwargs)
        return {"event_id": "e1"}

    with mock.patch.object(sw, "select_many", fake_select_many), \
            mock.patch.object(sw, "select_one", fake_select_one), \
            mock.patch.object(sw, "create_candidate_evaluation_event", fake_create_event):
        requeued = asyncio.run(worker._requeue_quarantined_candidates())
    assert requeued == 1
    assert writes[0]["event_type"] == "queued"
    assert writes[0]["candidate_status"] == "queued"
    assert writes[0]["reason"] == "provider_recovery_rescore"
    assert writes[0]["candidate_id"] == "cand-q"


def test_recovery_skips_candidate_already_requeued(monkeypatch):
    worker = _recovery_worker(monkeypatch)
    writes = []

    async def fake_select_many(table, **kwargs):
        filters = dict(kwargs.get("filters") or ())
        if filters.get("event_type") == "scoring_health_quarantined":
            return [{"candidate_id": "cand-q", "created_at": "t"}]
        return []

    async def fake_select_one(table, **kwargs):
        return {"candidate_id": "cand-q", "current_candidate_status": "queued"}

    async def fake_create_event(**kwargs):
        writes.append(kwargs)

    with mock.patch.object(sw, "select_many", fake_select_many), \
            mock.patch.object(sw, "select_one", fake_select_one), \
            mock.patch.object(sw, "create_candidate_evaluation_event", fake_create_event):
        requeued = asyncio.run(worker._requeue_quarantined_candidates())
    assert requeued == 0
    assert writes == []


def test_recovery_skips_when_quarantine_superseded(monkeypatch):
    worker = _recovery_worker(monkeypatch)
    writes = []

    async def fake_select_many(table, **kwargs):
        filters = dict(kwargs.get("filters") or ())
        if filters.get("event_type") == "scoring_health_quarantined":
            return [{"candidate_id": "cand-q", "created_at": "t"}]
        if table == "research_lab_candidate_promotion_events":
            return [{"event_type": "merged", "created_at": "t2"}]  # superseded
        return []

    async def fake_select_one(table, **kwargs):
        return {"candidate_id": "cand-q", "current_candidate_status": "scored"}

    async def fake_create_event(**kwargs):
        writes.append(kwargs)

    with mock.patch.object(sw, "select_many", fake_select_many), \
            mock.patch.object(sw, "select_one", fake_select_one), \
            mock.patch.object(sw, "create_candidate_evaluation_event", fake_create_event):
        requeued = asyncio.run(worker._requeue_quarantined_candidates())
    assert requeued == 0
    assert writes == []


def test_recovery_respects_attempt_cap(monkeypatch):
    worker = _recovery_worker(monkeypatch)
    monkeypatch.setenv("RESEARCH_LAB_QUARANTINE_RECOVERY_MAX_ATTEMPTS", "2")
    writes = []

    async def fake_select_many(table, **kwargs):
        filters = dict(kwargs.get("filters") or ())
        if filters.get("event_type") == "scoring_health_quarantined":
            return [{"candidate_id": "cand-q", "created_at": "t"}]
        if table == "research_lab_candidate_promotion_events":
            return [{"event_type": "scoring_health_quarantined", "created_at": "t"}]
        if table == "research_lab_candidate_evaluation_events":
            return [{"event_id": "r1"}, {"event_id": "r2"}]  # cap reached
        return []

    async def fake_select_one(table, **kwargs):
        return {"candidate_id": "cand-q", "current_candidate_status": "scored"}

    async def fake_create_event(**kwargs):
        writes.append(kwargs)

    with mock.patch.object(sw, "select_many", fake_select_many), \
            mock.patch.object(sw, "select_one", fake_select_one), \
            mock.patch.object(sw, "create_candidate_evaluation_event", fake_create_event):
        requeued = asyncio.run(worker._requeue_quarantined_candidates())
    assert requeued == 0
    assert writes == []


def test_recovery_disabled_with_gate(monkeypatch):
    worker = _recovery_worker(monkeypatch, scoring_health_gate_enabled=False)

    async def boom(*a, **k):
        raise AssertionError("must not query when gate disabled")

    with mock.patch.object(sw, "select_many", boom):
        assert asyncio.run(worker._requeue_quarantined_candidates()) == 0


def test_recovery_interval_throttles(monkeypatch):
    worker = _worker()
    monkeypatch.setenv("RESEARCH_LAB_QUARANTINE_RECOVERY_INTERVAL_SECONDS", "3600")
    calls = {"n": 0}

    async def fake_select_many(table, **kwargs):
        calls["n"] += 1
        return []

    with mock.patch.object(sw, "select_many", fake_select_many):
        asyncio.run(worker._requeue_quarantined_candidates())
        asyncio.run(worker._requeue_quarantined_candidates())
    assert calls["n"] == 1  # second call throttled by the interval


def test_reusable_bundle_skips_quarantine_worthy_bundle():
    worker = _worker()
    degraded_doc = {
        "execution_trace_ref": "trace:cand-1",
        "score_bundle_hash": "sha256:h",
        "signature_ref": "s3://sig",
        "scoring_health": {"provider_error_rate": 0.9},
    }

    async def fake_select_many(table, **kwargs):
        return [
            {
                "score_bundle_id": "sb-degraded",
                "score_bundle_doc": degraded_doc,
                "signature_ref": "s3://sig",
            }
        ]

    with mock.patch.object(sw, "select_many", fake_select_many):
        row = asyncio.run(
            worker._find_reusable_scored_bundle(
                candidate_id="cand-1",
                run_id="run-1",
                candidate_artifact_hash="sha256:a",
                evaluation_epoch=1,
            )
        )
    assert row is None  # degraded bundle must not be reused


def test_reusable_bundle_returns_healthy_bundle():
    worker = _worker()
    healthy_doc = {
        "execution_trace_ref": "trace:cand-1",
        "score_bundle_hash": "sha256:h",
        "signature_ref": "s3://sig",
        "scoring_health": {"provider_error_rate": 0.0},
    }

    async def fake_select_many(table, **kwargs):
        return [
            {
                "score_bundle_id": "sb-healthy",
                "score_bundle_doc": healthy_doc,
                "signature_ref": "s3://sig",
            }
        ]

    with mock.patch.object(sw, "select_many", fake_select_many):
        row = asyncio.run(
            worker._find_reusable_scored_bundle(
                candidate_id="cand-1",
                run_id="run-1",
                candidate_artifact_hash="sha256:a",
                evaluation_epoch=1,
            )
        )
    assert row is not None and row["score_bundle_id"] == "sb-healthy"
