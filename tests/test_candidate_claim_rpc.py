"""Regression tests: server-side candidate selection (egress reduction).

_claim_next_candidate previously pulled up to 50 full candidate current rows
(SELECT *, so every large patch artifact) on every poll and filtered them
client-side. It now asks the DB for the single next eligible candidate via
claim_next_research_lab_candidate. The assigned-event append stays in Python
(anchored_hash parity) under the guard_research_lab_candidate_claim trigger, so
concurrency safety is unchanged -- only the selection read shrinks.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import gateway.research_lab.scoring_worker as scoring


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT / "scripts" / "118-research-lab-claim-next-candidate.sql"
).read_text(encoding="utf-8")


def _worker() -> scoring.ResearchLabGatewayScoringWorker:
    w = object.__new__(scoring.ResearchLabGatewayScoringWorker)
    w.config = SimpleNamespace(
        scoring_worker_baseline_not_ready_retry_seconds=900,
        scoring_worker_retryable_failure_retry_seconds=300,
    )
    w.worker_ref = "scoring-worker-test"
    w.proxy_ref_hash = None
    w._candidate_start_hold_logged_key = None
    w._last_candidate_start_gate = None
    return w


@pytest.mark.asyncio
async def test_claim_uses_selection_rpc_with_retry_params(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_call_rpc(fn: str, params: dict[str, Any]) -> Any:
        calls.append((fn, params))
        return [{"candidate_id": "cand-1", "run_id": "run-1", "ticket_id": "tk-1"}]

    # A SELECT * over 50 rows must NOT happen anymore.
    async def forbidden_select_many(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("candidate claim must not scan 50 full rows")

    monkeypatch.setattr(scoring, "call_rpc", fake_call_rpc)
    monkeypatch.setattr(scoring, "select_many", forbidden_select_many)

    w = _worker()
    # Hold at the start-gate so the method returns right after selection without
    # attempting the (separately tested) event append.
    async def held_gate() -> dict[str, Any]:
        return {"available": False, "reason": "daily_baseline_hold"}

    w._candidate_scoring_start_gate = held_gate

    async def forbidden_append(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not append while start-gate holds")

    monkeypatch.setattr(scoring, "create_candidate_evaluation_event", forbidden_append)

    result = await w._claim_next_candidate()
    assert result is None  # start-gate held
    assert len(calls) == 1
    assert calls[0][0] == "claim_next_research_lab_candidate"
    assert calls[0][1] == {
        "p_baseline_not_ready_retry_seconds": 900,
        "p_retryable_failure_retry_seconds": 300,
    }


@pytest.mark.asyncio
async def test_claim_returns_none_when_rpc_empty(monkeypatch) -> None:
    async def empty_rpc(fn: str, params: dict[str, Any]) -> Any:
        return []

    gate_calls = {"n": 0}

    async def gate() -> dict[str, Any]:
        gate_calls["n"] += 1
        return {"available": True}

    monkeypatch.setattr(scoring, "call_rpc", empty_rpc)
    w = _worker()
    w._candidate_scoring_start_gate = gate

    assert await w._claim_next_candidate() is None
    # No eligible candidate -> we never even evaluate the (costly) start gate.
    assert gate_calls["n"] == 0


def test_migration_is_a_locked_down_single_row_selector() -> None:
    assert "claim_next_research_lab_candidate" in MIGRATION
    assert "RETURNS SETOF public.research_lab_candidate_evaluation_current" in MIGRATION
    assert "LIMIT 1" in MIGRATION
    assert "ORDER BY v.current_status_at ASC" in MIGRATION
    # Staleness filter mirrors _status_is_stale(max(60, threshold)).
    assert "baseline_not_ready" in MIGRATION
    assert "candidate_scoring_retryable_failure" in MIGRATION
    assert "conditional_validation_retryable_failure" in MIGRATION
    assert "GREATEST(60," in MIGRATION
    assert "make_interval" in MIGRATION
    # Locked down.
    assert "SECURITY DEFINER" in MIGRATION
    assert "SET search_path = ''" in MIGRATION
    assert "FROM PUBLIC, anon, authenticated" in MIGRATION
    assert "TO service_role" in MIGRATION
    # GREATEST/COALESCE are SQL expressions, not pg_catalog functions -- must not
    # be schema-qualified under search_path=''.
    assert "pg_catalog.greatest" not in MIGRATION.lower()
    assert "pg_catalog.coalesce" not in MIGRATION.lower()
