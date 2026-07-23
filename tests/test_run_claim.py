"""Regression tests: atomic single-owner hosted-run claim (egress reduction).

_next_queued_run previously downloaded the queued-run window on every worker and
picked locally by a client hash partition. It now makes ONE atomic claim RPC
that reserves the next queued run by priority for this worker and returns only
the consumed columns; concurrent workers get different runs. A not-yet-applied
migration falls back to the legacy trimmed scan + partition. Concurrent
no-double-assign is proven against real Postgres in the migration integration
test; here we pin the client wiring, the fallback, and the migration shape.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import gateway.research_lab.worker as hosted


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT / "scripts" / "122-research-lab-atomic-run-claim.sql"
).read_text(encoding="utf-8")

_QUEUE_ROW_COLUMNS = {
    "run_id", "ticket_id", "queue_priority", "current_event_hash", "current_status_at",
}


def _worker(total_workers: int = 1, worker_index: int = 0) -> hosted.ResearchLabHostedWorker:
    w = object.__new__(hosted.ResearchLabHostedWorker)
    w.config = SimpleNamespace(
        hosted_worker_queue_fetch_limit=20,
        hosted_worker_total_workers=total_workers,
        hosted_worker_index=worker_index,
    )
    w.worker_ref = "research-lab-worker-1"
    return w


@pytest.mark.asyncio
async def test_next_queued_run_uses_atomic_claim_rpc(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_call_rpc(fn: str, params: dict[str, Any]) -> Any:
        seen["fn"] = fn
        seen["params"] = params
        return [{"run_id": "run-1", "ticket_id": "tk-1", "queue_priority": 3,
                 "current_event_hash": "h", "current_status_at": "t"}]

    async def forbidden_select_many(*a: Any, **k: Any) -> Any:
        raise AssertionError("atomic claim must not scan the queue window")

    monkeypatch.setattr(hosted, "call_rpc", fake_call_rpc)
    monkeypatch.setattr(hosted, "select_many", forbidden_select_many)
    monkeypatch.setenv("RESEARCH_LAB_RUN_CLAIM_TTL_SECONDS", "120")

    row = await _worker()._next_queued_run()
    assert row is not None and row["run_id"] == "run-1"
    assert seen["fn"] == "claim_next_research_loop_run"
    assert seen["params"] == {
        "p_holder_ref": "research-lab-worker-1",
        "p_ttl_seconds": 120,
        "p_allowed_run_ids": [],
    }
    # Only the consumed columns come back.
    assert set(row) == _QUEUE_ROW_COLUMNS


@pytest.mark.asyncio
async def test_empty_claim_returns_none(monkeypatch) -> None:
    async def empty_rpc(fn: str, params: dict[str, Any]) -> Any:
        return []

    async def forbidden_select_many(*a: Any, **k: Any) -> Any:
        raise AssertionError("empty claim is a definitive no-run, not a fallback")

    monkeypatch.setattr(hosted, "call_rpc", empty_rpc)
    monkeypatch.setattr(hosted, "select_many", forbidden_select_many)
    assert await _worker()._next_queued_run() is None


@pytest.mark.asyncio
async def test_falls_back_to_scan_partition_when_rpc_unavailable(monkeypatch) -> None:
    # Migration 122 not applied yet -> RPC errors -> legacy trimmed scan + partition.
    async def boom_rpc(fn: str, params: dict[str, Any]) -> Any:
        raise RuntimeError("function claim_next_research_loop_run does not exist")

    captured: dict[str, Any] = {}

    async def fake_select_many(table: str, *, columns: str = "*", **kwargs: Any) -> Any:
        captured["columns"] = columns
        return [{"run_id": "run-legacy"}]

    monkeypatch.setattr(hosted, "call_rpc", boom_rpc)
    monkeypatch.setattr(hosted, "select_many", fake_select_many)

    row = await _worker()._next_queued_run()
    assert row is not None and row["run_id"] == "run-legacy"
    assert captured["columns"] != "*"
    assert set(captured["columns"].split(",")) == _QUEUE_ROW_COLUMNS


def test_ttl_helper_default_and_floor(monkeypatch) -> None:
    monkeypatch.delenv("RESEARCH_LAB_RUN_CLAIM_TTL_SECONDS", raising=False)
    assert hosted._run_claim_ttl_seconds() == 120
    monkeypatch.setenv("RESEARCH_LAB_RUN_CLAIM_TTL_SECONDS", "5")
    assert hosted._run_claim_ttl_seconds() == 30
    monkeypatch.setenv("RESEARCH_LAB_RUN_CLAIM_TTL_SECONDS", "nope")
    assert hosted._run_claim_ttl_seconds() == 120


def test_migration_is_a_locked_down_atomic_run_claim() -> None:
    assert "claim_next_research_loop_run" in MIGRATION
    assert "research_loop_run_claim" in MIGRATION
    assert "pg_advisory_xact_lock" in MIGRATION
    assert "ON CONFLICT (run_id) DO UPDATE" in MIGRATION
    assert "cl.claimed_at >= q.current_status_at" in MIGRATION
    assert "q.run_id = ANY (p_allowed_run_ids)" in MIGRATION
    assert "ORDER BY q.queue_priority ASC, q.current_status_at ASC" in MIGRATION
    assert "LIMIT 1" in MIGRATION
    assert "SECURITY DEFINER" in MIGRATION
    assert "SET search_path = ''" in MIGRATION
    assert "ENABLE ROW LEVEL SECURITY" in MIGRATION
    assert "FROM PUBLIC, anon, authenticated" in MIGRATION
    assert "TO service_role" in MIGRATION


@pytest.mark.asyncio
async def test_atomic_claim_passes_hosted_run_allowlist(monkeypatch) -> None:
    allowed = "33333333-3333-4333-8333-333333333333"
    seen: dict[str, Any] = {}

    async def fake_call_rpc(fn: str, params: dict[str, Any]) -> Any:
        seen["params"] = params
        return [{"run_id": allowed, "ticket_id": "tk-1"}]

    monkeypatch.setenv(hosted.HOSTED_RUN_ALLOWLIST_ENV, allowed)
    monkeypatch.setattr(hosted, "call_rpc", fake_call_rpc)

    row = await _worker()._next_queued_run()

    assert row is not None and row["run_id"] == allowed
    assert seen["params"]["p_allowed_run_ids"] == [allowed]


@pytest.mark.asyncio
async def test_atomic_claim_rejects_out_of_allowlist_response(monkeypatch) -> None:
    allowed = "33333333-3333-4333-8333-333333333333"

    async def bad_call_rpc(fn: str, params: dict[str, Any]) -> Any:
        return [{
            "run_id": "55555555-5555-4555-8555-555555555555",
            "ticket_id": "tk-1",
        }]

    monkeypatch.setenv(hosted.HOSTED_RUN_ALLOWLIST_ENV, allowed)
    monkeypatch.setattr(hosted, "call_rpc", bad_call_rpc)

    with pytest.raises(
        hosted.HostedResearchLabWorkerError,
        match="outside the hosted allowlist",
    ):
        await _worker()._next_queued_run()
