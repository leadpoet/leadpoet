"""Regression tests for the conflict-safe batched provider-usage ledger insert.

Egress reduction: the projector previously issued one existence SELECT (project
path) or two (backfill path) plus one INSERT per provider-usage ledger row —
the dominant weekly PostgREST volume. These tests pin that:
  * the write path issues exactly ONE server-side RPC for a whole batch,
  * the removed per-row existence reads are gone,
  * behavior degrades to the old per-row path only for injected fakes without
    call_rpc, and
  * the SQL is conflict-safe (ON CONFLICT DO NOTHING), locked-down, and bounded.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import pytest

import gateway.research_lab.trajectory_projector as tp


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT / "scripts" / "118-research-lab-provider-usage-batch-insert.sql"
).read_text(encoding="utf-8")


def _row(row_id: str, **overrides: Any) -> dict[str, Any]:
    row = {
        "usage_row_id": row_id,
        "schema_version": "1.0",
        "utc_day": "2026-07-22",
        "recorded_at": "2026-07-22T00:00:00+00:00",
        "provider_id": "exa",
        "endpoint_class": "search",
        "request_fingerprint": "fp",
        "evidence": "hit",
        "status": 0,
        "est_cost_microusd": 12,
        "caller_doc": {"worker": "scorer-1"},
    }
    row.update(overrides)
    return row


class _RpcStore:
    """Fake store exposing call_rpc; records the single batch call."""

    def __init__(self, inserted: int) -> None:
        self.inserted = inserted
        self.rpc_calls: list[tuple[str, dict[str, Any]]] = []
        self.select_one_calls = 0
        self.insert_calls = 0

    async def call_rpc(self, function_name: str, params: dict[str, Any]) -> Any:
        self.rpc_calls.append((function_name, params))
        return {"requested": len(params["rows"]), "inserted": self.inserted}

    async def select_one(self, *args: Any, **kwargs: Any) -> Any:
        self.select_one_calls += 1
        return None

    async def insert_row(self, *args: Any, **kwargs: Any) -> Any:
        self.insert_calls += 1
        return {}


class _PerRowStore:
    """Fake store WITHOUT call_rpc: must fall back to per-row inserts."""

    def __init__(self) -> None:
        self.select_one_calls = 0
        self.insert_calls = 0

    async def select_one(self, *args: Any, **kwargs: Any) -> Any:
        self.select_one_calls += 1
        return None

    async def insert_row(self, *args: Any, **kwargs: Any) -> Any:
        self.insert_calls += 1
        return {}


@pytest.mark.asyncio
async def test_batch_insert_issues_one_rpc_for_the_whole_batch() -> None:
    store = _RpcStore(inserted=3)
    rows = [_row(f"{i:032d}") for i in range(3)]

    written = await tp._insert_provider_usage_ledger_rows_batch(store, rows)

    assert written == 3
    # Exactly one server-side call for the whole batch — not one per row.
    assert len(store.rpc_calls) == 1
    name, params = store.rpc_calls[0]
    assert name == "insert_research_lab_provider_usage_ledger_rows"
    assert len(params["rows"]) == 3
    # The removed per-row existence reads must not happen on the RPC path.
    assert store.select_one_calls == 0
    assert store.insert_calls == 0


@pytest.mark.asyncio
async def test_batch_insert_skips_rows_without_a_primary_key() -> None:
    store = _RpcStore(inserted=1)
    rows = [_row("a" * 32), {"provider_id": "exa"}]  # second has no usage_row_id

    await tp._insert_provider_usage_ledger_rows_batch(store, rows)

    assert len(store.rpc_calls) == 1
    assert len(store.rpc_calls[0][1]["rows"]) == 1


@pytest.mark.asyncio
async def test_batch_insert_empty_is_a_noop() -> None:
    store = _RpcStore(inserted=0)
    assert await tp._insert_provider_usage_ledger_rows_batch(store, []) == 0
    assert store.rpc_calls == []


@pytest.mark.asyncio
async def test_batch_insert_falls_back_to_per_row_without_call_rpc() -> None:
    store = _PerRowStore()
    rows = [_row(f"{i:032d}") for i in range(4)]

    written = await tp._insert_provider_usage_ledger_rows_batch(store, rows)

    # Fallback preserves the old per-row behavior exactly for injected fakes.
    assert written == 4
    assert store.insert_calls == 4


@pytest.mark.asyncio
async def test_batch_insert_returns_none_on_rpc_error_not_zero() -> None:
    # A failed insert must be distinguishable from "0 rows already present", so
    # a transient provider-ledger failure is never mistaken for completeness.
    class _Boom:
        async def call_rpc(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("edge blip")

    written = await tp._insert_provider_usage_ledger_rows_batch(
        _Boom(), [_row("f" * 32)]
    )
    assert written is None  # failure, NOT 0

    # All-already-present is a real success and returns 0 (not None).
    class _AllPresent:
        async def call_rpc(self, *args: Any, **kwargs: Any) -> Any:
            return {"requested": 1, "inserted": 0}

    assert await tp._insert_provider_usage_ledger_rows_batch(_AllPresent(), [_row("a" * 32)]) == 0


class _LedgerRpcStore:
    """Store modeling the real RPC: dedup by usage_row_id (ON CONFLICT DO
    NOTHING), with a one-shot transient failure to prove recovery."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}  # usage_row_id -> row
        self.fail_next = False
        self.rpc_calls = 0

    async def call_rpc(self, function_name: str, params: dict[str, Any]) -> Any:
        self.rpc_calls += 1
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("transient supabase 503")
        inserted = 0
        for row in params["rows"]:
            rid = row["usage_row_id"]
            if rid not in self.rows:            # ON CONFLICT DO NOTHING
                self.rows[rid] = row
                inserted += 1
        return {"requested": len(params["rows"]), "inserted": inserted}


@pytest.mark.asyncio
async def test_batch_insert_helper_is_idempotent_and_signals_failure() -> None:
    # Helper-level contract (the end-to-end recovery through the real backfill
    # dispatcher is proven in test_trajectory_backfill_recovery.py):
    # a transient failure returns None and persists nothing; a later pass
    # restores every row exactly once; re-running is a no-op.
    store = _LedgerRpcStore()
    rows = [_row(f"{i:032d}") for i in range(5)]

    store.fail_next = True
    assert await tp._insert_provider_usage_ledger_rows_batch(store, rows) is None
    assert store.rows == {}  # nothing persisted during the outage

    assert await tp._insert_provider_usage_ledger_rows_batch(store, rows) == 5
    assert set(store.rows) == {r["usage_row_id"] for r in rows}

    assert await tp._insert_provider_usage_ledger_rows_batch(store, rows) == 0
    assert len(store.rows) == 5


def test_json_safe_row_projects_columns_and_coerces_datetime() -> None:
    row = _row(
        "d" * 32,
        recorded_at=datetime.datetime(2026, 7, 22, tzinfo=datetime.timezone.utc),
        extra_unknown_field="dropped",
    )
    safe = tp._json_safe_ledger_row(row)
    assert set(safe) == set(tp._PROVIDER_USAGE_LEDGER_COLUMNS)
    assert "extra_unknown_field" not in safe
    assert isinstance(safe["recorded_at"], str)
    assert safe["recorded_at"].startswith("2026-07-22T00:00:00")


def test_batch_inserted_count_handles_dict_list_and_garbage() -> None:
    assert tp._batch_inserted_count({"inserted": 5}) == 5
    assert tp._batch_inserted_count([{"inserted": 2}]) == 2
    assert tp._batch_inserted_count([]) == 0
    assert tp._batch_inserted_count(None) == 0
    assert tp._batch_inserted_count({"inserted": "bad"}) == 0


def test_migration_is_conflict_safe_and_locked_down() -> None:
    assert "ON CONFLICT (usage_row_id) DO NOTHING" in MIGRATION
    assert "SECURITY DEFINER" in MIGRATION
    assert "SET search_path = ''" in MIGRATION
    assert (
        "GRANT EXECUTE ON FUNCTION "
        "public.insert_research_lab_provider_usage_ledger_rows(JSONB)\n"
        "    TO service_role;" in MIGRATION
    )
    assert "FROM PUBLIC, anon, authenticated;" in MIGRATION
    # Bounded batch + fail-closed on malformed input (no silent row drops).
    assert "exceeds 5000 rows" in MIGRATION
    assert "must be a JSON array" in MIGRATION
    # Never UPDATEs (the table is append-only): the only write is a plain INSERT.
    assert "UPDATE public.research_lab_provider_usage_ledger" not in MIGRATION
