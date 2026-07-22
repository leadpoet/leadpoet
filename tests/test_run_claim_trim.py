"""Regression test: hosted-run selection reads only the columns it uses.

_next_queued_run partitioned queued runs across the fleet by pulling up to
queue_fetch_limit FULL run rows (SELECT *) every poll on every worker. The
partition needs only run_id, and the claim path downstream consumes just
ticket_id, queue_priority, current_event_hash and current_status_at. Atomicity
of the actual claim is enforced DB-side by guard_research_lab_run_claim (script
42), not by this read, so trimming the read changes nothing but egress.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import gateway.research_lab.worker as hosted


# Exactly the fields queue_row.* is read for anywhere downstream.
_EXPECTED_COLUMNS = {
    "run_id",
    "ticket_id",
    "queue_priority",
    "current_event_hash",
    "current_status_at",
}


def _worker(total_workers: int = 1, worker_index: int = 0) -> hosted.ResearchLabHostedWorker:
    w = object.__new__(hosted.ResearchLabHostedWorker)
    w.config = SimpleNamespace(
        hosted_worker_queue_fetch_limit=20,
        hosted_worker_total_workers=total_workers,
        hosted_worker_index=worker_index,
    )
    return w


@pytest.mark.asyncio
async def test_next_queued_run_selects_only_needed_columns(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_select_many(table: str, *, columns: str = "*", **kwargs: Any) -> Any:
        seen["table"] = table
        seen["columns"] = columns
        seen["kwargs"] = kwargs
        return [{"run_id": "run-1", "ticket_id": "tk-1", "queue_priority": 5}]

    monkeypatch.setattr(hosted, "select_many", fake_select_many)

    row = await _worker()._next_queued_run()
    assert row is not None
    assert seen["table"] == "research_loop_run_queue_current"
    # No SELECT *; exactly the consumed column set (order-independent).
    assert seen["columns"] != "*"
    assert set(seen["columns"].split(",")) == _EXPECTED_COLUMNS
    # Still filtered/ordered/limited server-side.
    assert seen["kwargs"]["filters"] == (("current_queue_status", "queued"),)
    assert seen["kwargs"]["limit"] == 20


def test_single_worker_takes_head_of_priority_order() -> None:
    rows = [{"run_id": "a"}, {"run_id": "b"}]
    assert _worker(total_workers=1)._select_preferred_queued_row(rows)["run_id"] == "a"


def test_partition_is_stable_and_covers_the_fleet() -> None:
    # Every queued run is claimable by exactly one shard, and the fallback keeps
    # a worker from starving when its own shard is momentarily empty.
    rows = [{"run_id": f"run-{i}"} for i in range(12)]
    total = 3
    for idx in range(total):
        picked = _worker(total_workers=total, worker_index=idx)._select_preferred_queued_row(rows)
        assert picked in rows
    assert _worker(total_workers=total, worker_index=0)._select_preferred_queued_row([]) is None
