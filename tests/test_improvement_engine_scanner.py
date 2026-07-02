"""Regression tests for the Improvement Engine passive scanner (plan §28.4).

Synthetic canonical rows (loop events / candidate rows / score bundles)
exercise clustering, category derivation from the live taxonomy, priority
stability, the infra-failure exclusion, miner-opportunity redaction, the
auto-apply contract, and the reopen policy.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import research_lab.improvement_engine.scanner as scanner
from research_lab.improvement_engine.clusterer import cluster_events
from research_lab.improvement_engine.config import ImprovementEngineConfig
from research_lab.improvement_engine.fingerprints import issue_fingerprint, normalize_reason
from research_lab.improvement_engine.fix_generator import draft_fix_spec, sanitized_miner_opportunity
from research_lab.improvement_engine.issue_reopener import should_reopen
from research_lab.improvement_engine.models import EngineTraceEvent
from research_lab.improvement_engine.prioritizer import priority_for_score, severity_score
from research_lab.improvement_engine.scanner import (
    _candidate_event_to_engine,
    _loop_event_to_engine,
    _score_bundle_events,
    issue_from_events,
    scan_for_issues,
)
from research_lab.observability.redaction import assert_langfuse_safe


def _now_iso(minutes_ago: int = 0) -> str:
    return (
        (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago))
        .isoformat()
        .replace("+00:00", "Z")
    )


def _event(
    category: str = "candidate_build_failed",
    *,
    stage: str = "candidate_image_build_failed",
    reason: str = "docker build exited 1",
    component: str = "code_build",
    run_id: str = "run-1",
    minutes_ago: int = 5,
    severity_hint: float = 2.0,
    score_delta: float | None = None,
) -> EngineTraceEvent:
    return EngineTraceEvent(
        failure_category=category,
        runtime_stage=stage,
        normalized_failure_reason=normalize_reason(reason),
        component=component,
        run_id=run_id,
        event_at=_now_iso(minutes_ago),
        severity_hint=severity_hint,
        score_delta=score_delta,
    )


# ---------------------------------------------------------------------------
# Fingerprints and clustering
# ---------------------------------------------------------------------------


def test_same_failure_clusters_together_across_runs() -> None:
    events = [_event(run_id=f"run-{i}") for i in range(4)]
    clusters = cluster_events(events)
    assert len(clusters) == 1
    (members,) = clusters.values()
    assert len(members) == 4


def test_different_categories_do_not_collide() -> None:
    events = [
        _event(),
        _event(category="patch_scope_violation", stage="patch_validation_failed", component="code_edit"),
    ]
    assert len(cluster_events(events)) == 2


def test_fingerprint_ignores_run_identity_and_normalizes_ids() -> None:
    a = _event(run_id="run-a", reason="failed for 11111111-1111-4111-8111-111111111111")
    b = _event(run_id="run-b", reason="failed for 99999999-9999-4999-8999-999999999999")
    assert issue_fingerprint(a) == issue_fingerprint(b)


def test_normalize_reason_strips_uuids_and_hashes() -> None:
    text = normalize_reason(
        "hash sha256:" + "a" * 64 + " run 11111111-1111-4111-8111-111111111111 FAILED"
    )
    assert "<uuid>" in text and "sha256:<hash>" in text
    assert "a" * 64 not in text


# ---------------------------------------------------------------------------
# Severity / priority
# ---------------------------------------------------------------------------


def test_priority_bands_are_stable() -> None:
    assert priority_for_score(8.0, category="x") == "critical"
    assert priority_for_score(4.0, category="x") == "high"
    assert priority_for_score(1.5, category="x") == "medium"
    assert priority_for_score(1.49, category="x") == "low"
    assert priority_for_score(0.1, category="secret_leak") == "critical"


def test_severity_is_deterministic_for_same_events() -> None:
    events = [_event(run_id=f"run-{i}") for i in range(6)]
    assert severity_score(events) == severity_score(list(events))


def test_severity_scales_with_occurrences_and_component() -> None:
    few = [_event(run_id=f"r{i}") for i in range(2)]
    many = [_event(run_id=f"r{i}") for i in range(20)]
    assert severity_score(many) > severity_score(few)
    non_critical = [_event(component="auto_research_loop", run_id=f"r{i}") for i in range(20)]
    assert severity_score(many) > severity_score(non_critical)


# ---------------------------------------------------------------------------
# Live-taxonomy derivation
# ---------------------------------------------------------------------------


def test_infra_build_failures_are_ops_noise_not_issues() -> None:
    row = {
        "event_type": "candidate_build_infra_failed",
        "run_id": "run-1",
        "created_at": _now_iso(),
        "event_doc": {"reason": "registry timeout"},
    }
    assert _loop_event_to_engine(row) is None


@pytest.mark.parametrize(
    "event_type",
    ["candidate_patch_apply_failed", "candidate_test_failed", "candidate_image_build_failed", "candidate_build_failed"],
)
def test_typed_build_stages_map_to_candidate_build_failed(event_type: str) -> None:
    event = _loop_event_to_engine(
        {"event_type": event_type, "run_id": "run-1", "created_at": _now_iso(), "event_doc": {}}
    )
    assert event is not None
    assert event.failure_category == "candidate_build_failed"
    assert event.component == "code_build"


def test_score_bundle_signals_derive_three_categories() -> None:
    row = {
        "run_id": "run-1",
        "bundle_status": "rejected",
        "score_bundle_hash": "sha256:" + "c" * 64,
        "created_at": _now_iso(),
        "score_bundle_doc": {
            "aggregates": {"mean_delta": -3.0},
            "scoring_health": {"sourced_zero_no_error_count": 2},
            "baseline_health": {"gate_passed": False},
        },
    }
    categories = {event.failure_category for event in _score_bundle_events(row)}
    assert categories == {
        "candidate_model_zero_companies",
        "score_bundle_gateway_rejected",
        "provider_error",
    }


def test_cost_reason_maps_to_cost_budget_exceeded() -> None:
    event = _candidate_event_to_engine(
        {"candidate_status": "failed", "reason": "cost budget exceeded", "run_id": "run-1", "created_at": _now_iso()}
    )
    assert event is not None and event.failure_category == "cost_budget_exceeded"


# ---------------------------------------------------------------------------
# End-to-end dry scan over faked canonical stores
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_for_issues_dry_run_clusters_and_respects_min_size(monkeypatch) -> None:
    loop_rows = [
        {
            "event_type": "candidate_image_build_failed",
            "run_id": f"run-{i}",
            "created_at": _now_iso(i),
            "event_doc": {"reason": "docker build exited 1"},
        }
        for i in range(4)
    ] + [
        {
            "event_type": "candidate_build_infra_failed",
            "run_id": "run-x",
            "created_at": _now_iso(),
            "event_doc": {"reason": "registry flake"},
        },
        # A singleton below min_cluster_size must NOT open an issue.
        {
            "event_type": "patch_validation_failed",
            "run_id": "run-y",
            "created_at": _now_iso(),
            "event_doc": {"reason": "path outside allowlist"},
        },
    ]

    async def fake_fetch(table: str, **kwargs):
        if table == "research_lab_auto_research_loop_events":
            return loop_rows
        return []

    monkeypatch.setattr(scanner, "fetch_recent_rows", fake_fetch)
    cfg = ImprovementEngineConfig(min_cluster_size=3)
    result = await scan_for_issues(config=cfg, dry_run=True, persist=False)
    assert result["issue_count"] == 1
    issue = result["issues"][0]
    assert issue["category"] == "candidate_build_failed"
    assert issue["occurrence_count"] == 4
    assert issue["status"] == "open"
    assert issue["suggested_fix_doc"]["requires_human_review"] is True
    assert issue["suggested_fix_doc"]["auto_apply_allowed"] is False
    assert result["persisted"] == []


# ---------------------------------------------------------------------------
# Auto-apply contract + miner-opportunity redaction
# ---------------------------------------------------------------------------


def _issue(**overrides):
    events = [_event(run_id=f"run-{i}") for i in range(3)]
    issue = issue_from_events(issue_fingerprint(events[0]), events)
    if overrides:
        issue = type(issue)(**{**issue.__dict__, **overrides})
    return issue


def test_auto_apply_patches_config_is_rejected() -> None:
    cfg = ImprovementEngineConfig(auto_apply_patches=True)
    with pytest.raises(ValueError, match="never auto-applied"):
        draft_fix_spec(_issue(), cfg)


def test_auto_apply_flag_defaults_false_from_env(monkeypatch) -> None:
    monkeypatch.delenv("RESEARCH_LAB_IMPROVEMENT_ENGINE_AUTO_APPLY_PATCHES", raising=False)
    assert ImprovementEngineConfig.from_env().auto_apply_patches is False


def test_miner_opportunity_exposes_only_public_fields() -> None:
    issue = _issue(
        root_cause_doc={
            "root_cause_summary": "query expansion over-narrowed",
            "likely_components": ["sourcing_model"],
            "evidence": ["zero scoreable companies", "expansion returned empty set"],
        },
    )
    opportunity = sanitized_miner_opportunity(issue)
    assert set(opportunity) == {
        "opportunity_id",
        "title",
        "category",
        "priority",
        "public_problem_summary",
        "affected_component_hint",
        "safe_failure_examples",
        "suggested_research_directions",
        "acceptance_signal",
        "engine_confidence",
        "created_from_trace_count",
    }
    # No private linkage or fingerprint material leaks, and the payload is
    # Langfuse-safe by the same redaction contract.
    assert "fingerprint" not in str(opportunity)
    assert "linked_trace_ids" not in opportunity
    assert_langfuse_safe(opportunity)


# ---------------------------------------------------------------------------
# Reopen policy
# ---------------------------------------------------------------------------


def test_reopen_policy() -> None:
    fixed_high = _issue(status="fixed", priority="high")
    assert should_reopen(fixed_high, recurrence_count=1) is True
    ignored_low = _issue(status="ignored", priority="low")
    assert should_reopen(ignored_low, recurrence_count=5) is False
    open_high = _issue(status="open", priority="high")
    assert should_reopen(open_high, recurrence_count=3) is False
