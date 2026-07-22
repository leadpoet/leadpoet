"""Regression tests: single-owner maintenance lease (egress reduction).

Every hosted/scoring worker used to run the global maintenance sweeps (stale-run
and stale-candidate recovery, projection reconciles, champion-reward
reconciliation, corpus projection) on every pass -- N workers doing N x the same
scans and writes. A DB-backed lease now grants "ownership" to one worker at a
time; only the holder runs the sweeps, and the lease replaces the fragile
worker_index==0 gating (a dead worker 0 no longer starves reward reconciles).

These tests pin the lease acquisition contract (parse + fail-closed), the TTL
helpers, and that the migration is a locked-down, atomically-acquired lease.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

import gateway.research_lab.maintenance as maint
import gateway.research_lab.store as store
import gateway.research_lab.worker as hosted
import gateway.research_lab.scoring_worker as scoring


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT / "scripts" / "117-research-lab-maintenance-lease.sql"
).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_acquire_returns_true_only_when_rpc_reports_acquired(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_call_rpc(fn: str, params: dict[str, Any]) -> Any:
        captured["fn"] = fn
        captured["params"] = params
        # PostgREST returns the JSONB payload directly for a scalar RPC.
        return {"acquired": True, "holder_ref": "worker-a", "expires_at": "2026-01-01T00:00:00Z"}

    monkeypatch.setattr(store, "call_rpc", fake_call_rpc)
    got = await maint.try_acquire_maintenance_lease(
        lease_name=maint.MAINTENANCE_LEASE_HOSTED, holder_ref="worker-a", ttl_seconds=180
    )
    assert got is True
    assert captured["fn"] == "research_lab_acquire_maintenance_lease"
    assert captured["params"] == {
        "p_lease_name": "hosted_worker_maintenance",
        "p_holder_ref": "worker-a",
        "p_ttl_seconds": 180,
    }


@pytest.mark.asyncio
async def test_acquire_false_when_held_by_another(monkeypatch) -> None:
    async def fake_call_rpc(fn: str, params: dict[str, Any]) -> Any:
        # A live holder that is not us -> acquired is False.
        return [{"acquired": False, "holder_ref": "worker-b"}]

    monkeypatch.setattr(store, "call_rpc", fake_call_rpc)
    got = await maint.try_acquire_maintenance_lease(
        lease_name=maint.MAINTENANCE_LEASE_SCORING, holder_ref="worker-a", ttl_seconds=180
    )
    assert got is False


@pytest.mark.asyncio
async def test_acquire_fail_closed_on_error(monkeypatch) -> None:
    async def boom(fn: str, params: dict[str, Any]) -> Any:
        raise RuntimeError("supabase blip")

    monkeypatch.setattr(store, "call_rpc", boom)
    got = await maint.try_acquire_maintenance_lease(
        lease_name=maint.MAINTENANCE_LEASE_HOSTED, holder_ref="worker-a", ttl_seconds=180
    )
    # Fail-closed: on error a worker simply skips the sweep this pass.
    assert got is False


@pytest.mark.asyncio
async def test_acquire_false_on_unexpected_shape(monkeypatch) -> None:
    async def fake_call_rpc(fn: str, params: dict[str, Any]) -> Any:
        return None

    monkeypatch.setattr(store, "call_rpc", fake_call_rpc)
    got = await maint.try_acquire_maintenance_lease(
        lease_name=maint.MAINTENANCE_LEASE_HOSTED, holder_ref="w", ttl_seconds=1
    )
    assert got is False


def test_lease_names_are_distinct_scopes() -> None:
    assert maint.MAINTENANCE_LEASE_HOSTED != maint.MAINTENANCE_LEASE_SCORING


def test_lease_holder_ref_is_unique_per_process_for_same_worker_name() -> None:
    # Finding: worker_ref (e.g. research-lab-worker-1) is a stable name reused by
    # every replica/restart. The lease holder token must be unique per process so
    # two overlapping processes with the same name do not both hold the lease.
    name = "research-lab-worker-1"
    a = maint.make_lease_holder_ref(name)
    b = maint.make_lease_holder_ref(name)
    assert a != b
    assert a.startswith(name + "#") and b.startswith(name + "#")
    assert str(os.getpid()) in a  # instance identity present


class _LeaseModel:
    """In-process model of migration 117's acquire semantics, keyed by token.

    Grants the lease iff it is unheld, expired, or already held by the SAME
    token (renewal). Mirrors the ON CONFLICT ... WHERE expires_at < now OR
    holder_ref = EXCLUDED.holder_ref clause so we can prove the identity rule
    without a live database (the real SQL is exercised by the Postgres
    integration test).
    """

    def __init__(self) -> None:
        self._rows: dict[str, tuple[str, float]] = {}  # lease -> (holder, expires)
        self._now = 1000.0

    def acquire(self, lease: str, holder: str, ttl: float) -> bool:
        held = self._rows.get(lease)
        if held is None or held[1] < self._now or held[0] == holder:
            self._rows[lease] = (holder, self._now + ttl)
            return True
        return False


def test_two_processes_same_name_different_tokens_only_one_acquires() -> None:
    model = _LeaseModel()
    lease = maint.MAINTENANCE_LEASE_HOSTED
    # Two live processes, identical human-readable name, distinct lease tokens.
    tok_a = maint.make_lease_holder_ref("research-lab-worker-1")
    tok_b = maint.make_lease_holder_ref("research-lab-worker-1")

    assert model.acquire(lease, tok_a, ttl=180) is True   # first wins
    assert model.acquire(lease, tok_b, ttl=180) is False  # second is locked out
    assert model.acquire(lease, tok_a, ttl=180) is True   # holder renews freely

    # Regression guard: if both processes had (incorrectly) used the bare
    # worker_ref as the holder, the model would grant BOTH — proving the bug the
    # unique token fixes.
    bare = "research-lab-worker-1"
    model2 = _LeaseModel()
    assert model2.acquire(lease, bare, ttl=180) is True
    assert model2.acquire(lease, bare, ttl=180) is True  # both "acquire" -> the bug


def test_ttl_helpers_default_and_floor(monkeypatch) -> None:
    monkeypatch.delenv("RESEARCH_LAB_MAINTENANCE_LEASE_TTL_SECONDS", raising=False)
    assert hosted._maintenance_lease_ttl_seconds() == 180
    assert scoring._scoring_maintenance_lease_ttl_seconds() == 180
    # Floor at 60s so a live holder renewing each pass never lapses mid-backoff.
    monkeypatch.setenv("RESEARCH_LAB_MAINTENANCE_LEASE_TTL_SECONDS", "5")
    assert hosted._maintenance_lease_ttl_seconds() == 60
    assert scoring._scoring_maintenance_lease_ttl_seconds() == 60
    monkeypatch.setenv("RESEARCH_LAB_MAINTENANCE_LEASE_TTL_SECONDS", "not-an-int")
    assert hosted._maintenance_lease_ttl_seconds() == 180


def test_ttl_exceeds_idle_backoff_cap(monkeypatch) -> None:
    # The lease must outlast the longest idle-backoff gap or a live holder could
    # lose ownership between passes.
    monkeypatch.delenv("RESEARCH_LAB_MAINTENANCE_LEASE_TTL_SECONDS", raising=False)
    monkeypatch.delenv("RESEARCH_LAB_WORKER_IDLE_BACKOFF_MAX_SECONDS", raising=False)
    assert hosted._maintenance_lease_ttl_seconds() > hosted._idle_backoff_max_seconds(15.0)


def test_migration_is_a_locked_down_atomic_lease() -> None:
    assert "research_lab_maintenance_lease" in MIGRATION
    assert "research_lab_acquire_maintenance_lease" in MIGRATION
    # Atomic acquire/renew: advisory lock + upsert that only overwrites an
    # expired lease or the same holder (renewal).
    assert "pg_advisory_xact_lock" in MIGRATION
    assert "ON CONFLICT (lease_name) DO UPDATE" in MIGRATION
    assert "l.expires_at < v_now" in MIGRATION
    assert "l.holder_ref = EXCLUDED.holder_ref" in MIGRATION
    # Locked down.
    assert "SECURITY DEFINER" in MIGRATION
    assert "SET search_path = ''" in MIGRATION
    assert "ENABLE ROW LEVEL SECURITY" in MIGRATION
    assert "FROM PUBLIC, anon, authenticated" in MIGRATION
    assert "TO service_role" in MIGRATION
