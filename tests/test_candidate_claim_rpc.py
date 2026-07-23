"""Regression tests: atomic single-owner candidate claim (egress reduction).

_claim_next_candidate previously pulled up to 50 full candidate rows per poll.
It now checks the global start-gate first, then makes ONE atomic claim RPC that
reserves the single next eligible candidate for this worker and returns only the
columns scoring needs. Concurrent no-double-assign is proven against real
Postgres in tests/test_migrations_postgres_integration.py; here we pin the
client wiring and the migration's shape.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import gateway.research_lab.scoring_worker as scoring


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT / "scripts" / "121-research-lab-atomic-candidate-claim.sql"
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
async def test_claim_checks_gate_first_then_claims_with_holder_and_ttl(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_call_rpc(fn: str, params: dict[str, Any]) -> Any:
        calls.append((fn, params))
        return [{"candidate_id": "cand-1", "run_id": "run-1", "ticket_id": "tk-1"}]

    async def forbidden_select_many(*a: Any, **k: Any) -> Any:
        raise AssertionError("claim must not scan candidate rows client-side")

    monkeypatch.setattr(scoring, "call_rpc", fake_call_rpc)
    monkeypatch.setattr(scoring, "select_many", forbidden_select_many)
    monkeypatch.setattr(scoring, "create_candidate_evaluation_event", _fail_append)
    monkeypatch.setenv("RESEARCH_LAB_CANDIDATE_CLAIM_TTL_SECONDS", "120")

    w = _worker()

    async def open_gate() -> dict[str, Any]:
        return {"available": True}

    w._candidate_scoring_start_gate = open_gate
    # Stop right after the claim so we only assert the claim call (append is
    # exercised elsewhere); make the append raise a claim-race to bail cleanly.
    result = await w._claim_next_candidate()

    assert result is None  # append raced (stubbed) -> None, but the claim happened
    assert len(calls) == 1
    fn, params = calls[0]
    assert fn == "claim_next_research_lab_candidate"
    assert params == {
        "p_holder_ref": "scoring-worker-test",
        "p_ttl_seconds": 120,
        "p_baseline_not_ready_retry_seconds": 900,
        "p_retryable_failure_retry_seconds": 300,
    }


async def _fail_append(**kwargs: Any) -> Any:
    # A claim-race marker _is_candidate_claim_race_error recognizes, so
    # _claim_next_candidate returns None cleanly after the claim RPC ran.
    raise RuntimeError("research_lab_candidate_claim_conflict")


@pytest.mark.asyncio
async def test_held_gate_takes_no_claim(monkeypatch) -> None:
    calls: list[Any] = []

    async def fake_call_rpc(fn: str, params: dict[str, Any]) -> Any:
        calls.append(fn)
        return []

    monkeypatch.setattr(scoring, "call_rpc", fake_call_rpc)
    w = _worker()

    async def held_gate() -> dict[str, Any]:
        return {"available": False, "reason": "daily_baseline_hold"}

    w._candidate_scoring_start_gate = held_gate
    assert await w._claim_next_candidate() is None
    # Gate held -> we never reserve a candidate.
    assert calls == []


@pytest.mark.asyncio
async def test_empty_claim_returns_none(monkeypatch) -> None:
    async def empty_rpc(fn: str, params: dict[str, Any]) -> Any:
        return []

    monkeypatch.setattr(scoring, "call_rpc", empty_rpc)
    w = _worker()

    async def open_gate() -> dict[str, Any]:
        return {"available": True}

    w._candidate_scoring_start_gate = open_gate
    assert await w._claim_next_candidate() is None


def test_ttl_helper_default_and_floor(monkeypatch) -> None:
    monkeypatch.delenv("RESEARCH_LAB_CANDIDATE_CLAIM_TTL_SECONDS", raising=False)
    assert scoring._candidate_claim_ttl_seconds() == 120
    monkeypatch.setenv("RESEARCH_LAB_CANDIDATE_CLAIM_TTL_SECONDS", "5")
    assert scoring._candidate_claim_ttl_seconds() == 30  # floored
    monkeypatch.setenv("RESEARCH_LAB_CANDIDATE_CLAIM_TTL_SECONDS", "nope")
    assert scoring._candidate_claim_ttl_seconds() == 120


def test_migration_is_a_locked_down_atomic_claim() -> None:
    assert "claim_next_research_lab_candidate" in MIGRATION
    assert "research_lab_candidate_claim" in MIGRATION           # claim-lease table
    assert "pg_advisory_xact_lock" in MIGRATION                  # serialize claimers
    assert "ON CONFLICT (candidate_id) DO UPDATE" in MIGRATION   # reserve
    assert "cl.claimed_at >= v.current_status_at" in MIGRATION   # requeue-safe guard
    assert "ORDER BY v.current_status_at ASC" in MIGRATION
    assert "LIMIT 1" in MIGRATION
    # Returns the minimal scoring column set, not SELECT *.
    for col in ("private_model_manifest_doc", "candidate_patch_manifest",
                "candidate_build_doc", "hypothesis_doc", "redacted_public_summary"):
        assert col in MIGRATION
    assert "v.*" not in MIGRATION
    # Staleness filter mirrors _status_is_stale(max(60, threshold)).
    assert "GREATEST(60," in MIGRATION
    assert "baseline_not_ready" in MIGRATION
    # Locked down.
    assert "SECURITY DEFINER" in MIGRATION
    assert "SET search_path = ''" in MIGRATION
    assert "ENABLE ROW LEVEL SECURITY" in MIGRATION
    assert "FROM PUBLIC, anon, authenticated" in MIGRATION
    assert "TO service_role" in MIGRATION
    assert "pg_catalog.greatest" not in MIGRATION.lower()
    assert "pg_catalog.coalesce" not in MIGRATION.lower()
