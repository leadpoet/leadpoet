"""Tests for the allocator-priors nightly refresh (inner-loop activation Phase 2).

Covers:
- refresh persists the deterministic daily selection record (row shape,
  selection_doc round trip),
- idempotency: a second refresh over the same (day, window) is a no-op,
- empty-ledger short circuit,
- the engine read path (load_cell_yield_priors) prefers the persisted record
  and falls back to on-demand computation when none exists (including when
  the selection table itself is missing).
"""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

from gateway.research_lab import allocator_priors


class _FakeStore:
    """Stand-in for gateway.research_lab.store select/insert helpers."""

    def __init__(self, rows_by_table: Mapping[str, list[dict[str, Any]]]):
        self.rows_by_table = {name: list(rows) for name, rows in rows_by_table.items()}
        self.inserted: list[tuple[str, dict[str, Any]]] = []
        self.select_errors: dict[str, Exception] = {}

    async def select_many(self, table, *, columns="*", filters, order_by=(), limit=100):
        if table in self.select_errors:
            raise self.select_errors[table]
        rows = [dict(row) for row in self.rows_by_table.get(table, [])]
        for spec in filters:
            field, value = spec[0], spec[-1]
            rows = [row for row in rows if str(row.get(field)) == str(value)]
        for field, desc in reversed(list(order_by or ())):
            rows.sort(key=lambda row: str(row.get(field) or ""), reverse=bool(desc))
        return rows[:limit]

    async def insert_row(self, table, row):
        self.inserted.append((table, dict(row)))
        stored = dict(row)
        stored.setdefault("created_at", f"2026-07-06T00:00:0{len(self.inserted)}Z")
        self.rows_by_table.setdefault(table, []).append(stored)
        return stored


def _ledger_row(
    *,
    ledger_row_id: str,
    lane: str = "provider",
    status: str = "discard",
    delta: float | None = None,
    cost_usd: float = 1.0,
    created_at: str = "2026-07-01T00:00:00Z",
) -> dict[str, Any]:
    return {
        "ledger_row_id": ledger_row_id,
        "island": "generalist",
        "targeted_metric": "candidate_delta_vs_daily_baseline",
        "status": status,
        "delta_vs_parent": delta,
        "cost_usd": cost_usd,
        "description": (
            f"CODE_EDIT on {lane} targeted candidate_delta_vs_daily_baseline; "
            f"decision={status}; delta=n/a."
        ),
        "created_at": created_at,
    }


def _ledger_rows() -> list[dict[str, Any]]:
    rows = []
    for index in range(6):
        rows.append(
            _ledger_row(
                ledger_row_id=f"aaaa000{index}-0000-0000-0000-000000000000",
                lane="provider",
                status="keep" if index < 5 else "discard",
                delta=1.5 if index < 5 else -0.5,
            )
        )
    for index in range(6):
        rows.append(
            _ledger_row(
                ledger_row_id=f"bbbb000{index}-0000-0000-0000-000000000000",
                lane="intent",
                status="crash" if index % 2 else "discard",
            )
        )
    return rows


def test_refresh_persists_daily_selection_record():
    store = _FakeStore({allocator_priors.RESULTS_LEDGER_TABLE: _ledger_rows()})
    result = asyncio.run(
        allocator_priors.refresh_allocator_priors(
            store=store, day="2026-07-06", created_by="test-worker"
        )
    )
    assert result["status"] == "persisted"
    assert result["day"] == "2026-07-06"
    assert result["selection_id"].startswith("meta_allocator_selection:cell-yield:")
    assert len(store.inserted) == 1
    table, row = store.inserted[0]
    assert table == allocator_priors.ALLOCATOR_SELECTION_RECORDS_TABLE
    assert row["schema_version"] == "1.0"
    assert row["day"] == "2026-07-06"
    assert row["window_hash"] == result["window_hash"]
    assert row["created_by"] == "test-worker"
    doc = row["selection_doc"]
    assert doc["ranked_cells"]
    assert doc["exploration_floor"] == allocator_priors.EXPLORATION_FLOOR


def test_refresh_is_idempotent_per_day_and_window():
    store = _FakeStore({allocator_priors.RESULTS_LEDGER_TABLE: _ledger_rows()})
    first = asyncio.run(
        allocator_priors.refresh_allocator_priors(store=store, day="2026-07-06")
    )
    second = asyncio.run(
        allocator_priors.refresh_allocator_priors(store=store, day="2026-07-06")
    )
    assert first["status"] == "persisted"
    assert second["status"] == "already_persisted"
    assert second["window_hash"] == first["window_hash"]
    assert len(store.inserted) == 1


def test_refresh_short_circuits_on_empty_ledger():
    store = _FakeStore({allocator_priors.RESULTS_LEDGER_TABLE: []})
    result = asyncio.run(allocator_priors.refresh_allocator_priors(store=store))
    assert result["status"] == "empty_ledger"
    assert not store.inserted


def test_load_prefers_persisted_record_over_recompute():
    store = _FakeStore({allocator_priors.RESULTS_LEDGER_TABLE: _ledger_rows()})
    asyncio.run(allocator_priors.refresh_allocator_priors(store=store, day="2026-07-06"))
    persisted_doc = store.inserted[0][1]["selection_doc"]

    # Mutate the ledger afterwards: the persisted record must still win.
    store.rows_by_table[allocator_priors.RESULTS_LEDGER_TABLE].append(
        _ledger_row(
            ledger_row_id="cccc0000-0000-0000-0000-000000000000",
            lane="query",
            status="keep",
            delta=9.0,
            created_at="2026-07-06T12:00:00Z",
        )
    )
    loaded = asyncio.run(
        allocator_priors.load_cell_yield_priors(store=store, day="2026-07-06")
    )
    assert loaded == persisted_doc


def test_load_falls_back_to_compute_without_persisted_record():
    store = _FakeStore({allocator_priors.RESULTS_LEDGER_TABLE: _ledger_rows()})
    loaded = asyncio.run(
        allocator_priors.load_cell_yield_priors(store=store, day="2026-07-06")
    )
    computed = asyncio.run(
        allocator_priors.build_cell_yield_priors(store=store, day="2026-07-06")
    )
    assert loaded == computed
    assert loaded is not None and loaded["ranked_cells"]


def test_load_falls_back_when_selection_table_is_missing():
    store = _FakeStore({allocator_priors.RESULTS_LEDGER_TABLE: _ledger_rows()})
    store.select_errors[allocator_priors.ALLOCATOR_SELECTION_RECORDS_TABLE] = RuntimeError(
        "relation does not exist"
    )
    loaded = asyncio.run(
        allocator_priors.load_cell_yield_priors(store=store, day="2026-07-06")
    )
    assert loaded is not None and loaded["ranked_cells"]
