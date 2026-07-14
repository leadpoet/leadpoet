"""Production gates for automatic Research Lab inner-loop activation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from gateway.research_lab.inner_loop_activation import (
    CANDIDATE_ELIGIBILITY_GATE,
    HISTORICAL_PAIR_GATE,
    INNER_LOOP_EVENT_TABLE,
    OBSERVE_RUN_GATE,
    SHADOW_RUN_GATE,
    InnerLoopEvidence,
    build_inner_loop_evidence,
    configured_inner_loop_mode,
    decide_inner_loop_policy,
    resolve_inner_loop_policy,
)


def _readiness(**overrides: Any) -> dict[str, Any]:
    return {
        "ready": True,
        "reason": "ready",
        "manifest_hash": "sha256:" + "a" * 64,
        "dev_set_hash": "sha256:" + "b" * 64,
        "recorded_at": "2026-07-01T00:00:00+00:00",
        "snapshot_age_seconds": 3600,
        "dev_set_size": 8,
        **overrides,
    }


def _observations(
    phase: str,
    count: int,
    *,
    hours_between: int,
    first_seq: int = 1,
    evidence_overrides: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    start = datetime(2026, 7, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(count):
        candidate_count = 1 if phase == "observe" else 3
        evidence = {
            "run_eligible": True,
            "candidate_count": candidate_count,
            "eligible_candidate_count": candidate_count,
            "unclassified_error_count": 0,
            "silent_miss_count": 0,
            "candidate_width_mismatch": False,
            "paid_finalist_invariant_violations": 0,
            "protected_workflow_invariant_violations": 0,
            **dict(evidence_overrides or {}),
        }
        rows.append(
            {
                "seq": first_seq + index,
                "event_type": "run_observed",
                "phase": phase,
                "run_id": f"00000000-0000-0000-0000-{index:012d}",
                "evidence_doc": evidence,
                "created_at": (start + timedelta(hours=index * hours_between)).isoformat(),
            }
        )
    return sorted(rows, key=lambda row: int(row["seq"]), reverse=True)


def _calibrations(count: int = HISTORICAL_PAIR_GATE) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": f"candidate-{index}",
            "dev_score": float(index),
            "realized_mean_delta": float(index),
            "created_at": f"2026-07-{(index % 28) + 1:02d}T00:00:00Z",
        }
        for index in range(count)
    ]


def _transition(phase: str, seq: int) -> dict[str, Any]:
    return {
        "seq": seq,
        "event_type": "phase_transition",
        "phase": phase,
        "run_id": None,
        "evidence_doc": {},
        "created_at": "2026-07-12T00:00:00+00:00",
    }


def _policy(mode: str, evidence: InnerLoopEvidence, **overrides: Any):
    values = {
        "requested_mode": mode,
        "snapshot_readiness": _readiness(),
        "evidence": evidence,
        "dev_eval_kill_switch_enabled": True,
        "configured_candidate_width": 3,
        "configured_paid_finalist_count": 1,
        "stop_at_candidate_cap_enabled": True,
    }
    values.update(overrides)
    return decide_inner_loop_policy(**values)


def test_mode_defaults_and_invalid_values_fail_closed(monkeypatch):
    monkeypatch.delenv("RESEARCH_LAB_INNER_LOOP_MODE", raising=False)
    assert configured_inner_loop_mode() == "off"
    assert configured_inner_loop_mode("unexpected") == "off"
    policy = _policy("off", InnerLoopEvidence())
    assert policy.effective_phase == "off"
    assert policy.candidate_width == 1
    assert policy.paid_finalist_count == 1
    assert not policy.evaluator_enabled


def test_nonautomatic_modes_never_query_activation_state():
    class _UnexpectedStore:
        async def select_all(self, *_args: Any, **_kwargs: Any):
            raise AssertionError("non-auto modes must not query activation evidence")

    for mode in ("off", "observe", "shadow", "rank"):
        policy = asyncio.run(
            resolve_inner_loop_policy(
                requested_mode=mode,
                snapshot_readiness=_readiness(),
                dev_eval_kill_switch_enabled=True,
                configured_candidate_width=3,
                configured_paid_finalist_count=1,
                stop_at_candidate_cap_enabled=True,
                store=_UnexpectedStore(),
            )
        )
        assert policy.effective_phase == mode


def test_observe_gate_requires_ten_consecutive_runs_over_24_hours():
    short = build_inner_loop_evidence(
        events=_observations("observe", OBSERVE_RUN_GATE, hours_between=2),
        calibrations=[],
    )
    assert _policy("auto", short).effective_phase == "observe"

    eligible = build_inner_loop_evidence(
        events=_observations("observe", OBSERVE_RUN_GATE, hours_between=3),
        calibrations=[],
    )
    policy = _policy("auto", eligible)
    assert policy.effective_phase == "shadow"
    assert policy.candidate_width == 3
    assert policy.shadow_enabled
    assert not policy.ranking_enabled
    assert policy.sequential_chain_enabled


def test_rank_gate_requires_full_health_and_positive_historical_lift():
    observations = _observations(
        "shadow",
        SHADOW_RUN_GATE,
        hours_between=9,
    )
    events = [_transition("shadow", 1000), *observations]
    evidence = build_inner_loop_evidence(events=events, calibrations=_calibrations())
    assert evidence.historical_pair_count == HISTORICAL_PAIR_GATE
    assert evidence.candidate_eligibility_rate >= CANDIDATE_ELIGIBILITY_GATE
    policy = _policy("auto", evidence)
    assert policy.effective_phase == "rank"
    assert policy.ranking_enabled
    assert policy.candidate_width == 3
    assert policy.sequential_chain_enabled


def test_rank_downgrades_on_silent_miss_and_protected_failures_stop_it():
    healthy = InnerLoopEvidence(
        current_phase="rank",
        shadow_healthy_runs=SHADOW_RUN_GATE,
        shadow_span_seconds=8 * 86400,
        historical_pair_count=HISTORICAL_PAIR_GATE,
        spearman_rho=0.8,
        top_quartile_lift=0.2,
        candidate_eligibility_rate=1.0,
    )
    missed = _policy("auto", InnerLoopEvidence(**{**healthy.__dict__, "silent_miss_count": 1}))
    assert missed.effective_phase == "shadow"
    assert not missed.ranking_enabled

    protected = _policy(
        "auto",
        InnerLoopEvidence(
            **{**healthy.__dict__, "protected_workflow_invariant_violations": 1}
        ),
    )
    assert protected.effective_phase == "off"
    assert protected.candidate_width == 1

    broken_chain = _policy(
        "auto",
        InnerLoopEvidence(
            **{**healthy.__dict__, "sequential_chain_invariant_violations": 1}
        ),
    )
    assert broken_chain.effective_phase == "observe"
    assert broken_chain.candidate_width == 1
    assert not broken_chain.sequential_chain_enabled


def test_retried_run_observation_counts_once():
    rows = _observations("observe", OBSERVE_RUN_GATE, hours_between=3)
    duplicate = {
        **rows[0],
        "seq": int(rows[0]["seq"]) + 100,
        "created_at": "2026-07-20T00:00:00+00:00",
    }

    evidence = build_inner_loop_evidence(
        events=[duplicate, *rows],
        calibrations=[],
    )

    assert evidence.observe_eligible_runs == OBSERVE_RUN_GATE


def test_historical_gate_counts_each_candidate_once_and_rejects_nonfinite_pairs():
    calibrations = _calibrations()
    calibrations.extend(
        [
            {
                **calibrations[0],
                "dev_score": -1000.0,
                "created_at": "2026-06-01T00:00:00Z",
            },
            {
                "candidate_id": "candidate-invalid",
                "dev_score": float("nan"),
                "realized_mean_delta": 1.0,
                "created_at": "2026-07-30T00:00:00Z",
            },
        ]
    )

    evidence = build_inner_loop_evidence(events=[], calibrations=calibrations)

    assert evidence.historical_pair_count == HISTORICAL_PAIR_GATE
    assert evidence.spearman_rho == 1.0


def test_snapshot_and_paid_finalist_preflight_force_ordinary_single_candidate():
    evidence = InnerLoopEvidence(current_phase="rank")
    stale = _policy(
        "rank",
        evidence,
        snapshot_readiness=_readiness(snapshot_age_seconds=15 * 86400),
    )
    assert stale.effective_phase == "observe"
    assert stale.candidate_width == 1
    assert stale.fallback_reason

    invalid_paid = _policy(
        "rank",
        evidence,
        configured_paid_finalist_count=2,
    )
    assert invalid_paid.effective_phase == "observe"
    assert invalid_paid.candidate_width == 1
    assert invalid_paid.paid_finalist_count == 1


class _ConcurrentStore:
    def __init__(self, events: list[dict[str, Any]], calibrations: list[dict[str, Any]]):
        self.events = list(events)
        self.calibrations = list(calibrations)
        self.lock = None
        self.rpc_calls = 0

    async def select_all(self, table: str, **_kwargs: Any):
        await asyncio.sleep(0)
        rows = self.events if table == INNER_LOOP_EVENT_TABLE else self.calibrations
        return [dict(row) for row in rows]

    async def call_rpc(self, _name: str, params: Mapping[str, Any]):
        if self.lock is None:
            self.lock = asyncio.Lock()
        async with self.lock:
            self.rpc_calls += 1
            transitions = [
                row for row in self.events if row.get("event_type") == "phase_transition"
            ]
            current = str((transitions[0] if transitions else {}).get("phase") or "observe")
            expected = str(params.get("expected_current_phase") or "observe")
            if current != expected:
                raise RuntimeError("40001 research_lab_inner_loop_phase_conflict")
            row = {
                "seq": max([int(item.get("seq") or 0) for item in self.events] or [0]) + 1,
                "event_type": str(params["requested_event_type"]),
                "phase": str(params["requested_phase"]),
                "run_id": params.get("requested_run_id"),
                "evidence_doc": dict(params.get("requested_evidence_doc") or {}),
                "created_at": "2026-07-13T00:00:00+00:00",
            }
            self.events.insert(0, row)
            return row


class _UncommittedTransitionStore(_ConcurrentStore):
    async def call_rpc(self, _name: str, params: Mapping[str, Any]):
        del params
        raise RuntimeError("database transport failed before commit")


def test_uncommitted_transition_never_widens_candidate_generation():
    store = _UncommittedTransitionStore(
        _observations("observe", OBSERVE_RUN_GATE, hours_between=3),
        [],
    )

    policy = asyncio.run(
        resolve_inner_loop_policy(
            requested_mode="auto",
            snapshot_readiness=_readiness(),
            dev_eval_kill_switch_enabled=True,
            configured_candidate_width=3,
            configured_paid_finalist_count=1,
            stop_at_candidate_cap_enabled=True,
            store=store,
        )
    )

    assert policy.effective_phase == "observe"
    assert policy.candidate_width == 1
    assert policy.ranking_enabled is False
    assert policy.fallback_reason.startswith("activation_transition_uncommitted:")


def test_ten_concurrent_workers_create_one_restart_stable_transition():
    store = _ConcurrentStore(
        _observations("observe", OBSERVE_RUN_GATE, hours_between=3),
        [],
    )

    async def _run():
        return await asyncio.gather(
            *(
                resolve_inner_loop_policy(
                    requested_mode="auto",
                    snapshot_readiness=_readiness(),
                    dev_eval_kill_switch_enabled=True,
                    configured_candidate_width=3,
                    configured_paid_finalist_count=1,
                    stop_at_candidate_cap_enabled=True,
                    store=store,
                )
                for _index in range(10)
            )
        )

    policies = asyncio.run(_run())
    transitions = [
        row for row in store.events if row.get("event_type") == "phase_transition"
    ]
    assert len(transitions) == 1
    assert transitions[0]["phase"] == "shadow"
    assert all(policy.effective_phase == "shadow" for policy in policies)
    assert all(policy.candidate_width == 3 for policy in policies)
    assert all(policy.sequential_chain_enabled for policy in policies)
