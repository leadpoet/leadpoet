"""Regression tests: the egress changes preserve reward/weight outputs.

The single-owner maintenance lease only changes WHICH worker (and thus which
actor_ref/worker_ref) runs the reward reconcilers -- never what they compute.
These tests pin that identity-invariance for the champion-reward reconcilers,
and that the lease-held maintenance cadence stays independent of the idle poll
backoff (so recovery + reward maintenance schedules are not coupled to polling).

The no-double-assign proof for the atomic claims lives in
tests/test_migrations_postgres_integration.py; the batched-ledger no-row-loss
proof lives in tests/test_provider_usage_batch_insert.py.
"""

from __future__ import annotations

import pytest

import gateway.research_lab.worker as hosted


@pytest.mark.asyncio
async def test_champion_status_reconcile_is_actor_identity_invariant(monkeypatch):
    from gateway.research_lab import maintenance
    from gateway.research_lab import allocations

    reward_id = "champion_reward:sha256:" + "7" * 64
    reward = {
        "champion_reward_id": reward_id,
        "miner_uid": 7,
        "desired_alpha_percent": 5.0,
        "epoch_count": 2,
        "current_reward_status": "active",
    }

    async def select_all(_table, *, filters=(), **_kwargs):
        status = next((v for f, v in filters if f == "current_reward_status"), "")
        return [reward] if status == "active" else []

    async def fully_settled(**_kwargs):
        return {reward_id: 99.0}

    async def forbid_write(**kwargs):  # dry-run must not write
        raise AssertionError("dry-run reconcile must not write")

    monkeypatch.setattr(maintenance, "select_all", select_all)
    monkeypatch.setattr(allocations, "_champion_finalized_paid_alpha_to_date", fully_settled)

    # Two DIFFERENT callers (as the lease can hand ownership to any worker).
    plan_a = await maintenance.reconcile_champion_reward_statuses(
        epoch=102, netuid=71, actor_ref="research-lab-worker-1#hostA#11#aaa", dry_run=True)
    plan_b = await maintenance.reconcile_champion_reward_statuses(
        epoch=102, netuid=71, actor_ref="research-lab-worker-1#hostB#22#bbb", dry_run=True)

    # The reconciliation decision (which rewards are settled/held) is identical
    # regardless of who ran it -- the lease change cannot alter reward outcomes.
    assert plan_a.get("planned") == plan_b.get("planned")
    assert plan_a.get("planned_count") == plan_b.get("planned_count")
    assert plan_a.get("ok") == plan_b.get("ok")


def test_maintenance_cadence_is_independent_of_idle_backoff(monkeypatch):
    # Item 9: the lease-held sweep interval must always be >= the idle-backoff
    # cap, so a poll that has backed off to 60s never widens the maintenance gap
    # (the sweeps run on their own wall-clock cadence, not once per poll).
    monkeypatch.delenv("RESEARCH_LAB_WORKER_MAINTENANCE_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("RESEARCH_LAB_WORKER_IDLE_BACKOFF_MAX_SECONDS", raising=False)
    interval = hosted._lease_maintenance_interval_seconds()
    assert interval >= hosted._idle_backoff_max_seconds(15.0)
    assert interval == 150  # default

    # Even if an operator sets a tiny interval, it is floored to the backoff cap
    # so it can never become coupled to (smaller than) the poll gap.
    monkeypatch.setenv("RESEARCH_LAB_WORKER_MAINTENANCE_INTERVAL_SECONDS", "5")
    monkeypatch.setenv("RESEARCH_LAB_WORKER_IDLE_BACKOFF_MAX_SECONDS", "60")
    assert hosted._lease_maintenance_interval_seconds() == 60


def test_idle_request_rate_stays_bounded_as_worker_count_grows():
    # Item 12: idle request volume must not multiply with worker count. Each
    # worker's idle poll rate is capped by the backoff cap (not the base poll),
    # and the global maintenance sweeps run on exactly ONE worker (the lease
    # holder), so N idle workers issue ~N cheap polls -- never N x the maintenance
    # scan volume. Here we pin the per-worker cap component.
    base, cap = 15.0, 60.0
    interval = base
    for _ in range(10):
        interval = hosted._idle_backoff_next(interval, base, cap)
    assert interval == cap  # converges to the cap, so per-worker rate is 1/cap
    # The maintenance sweeps are lease-gated to a single holder (proven in
    # test_migrations_postgres_integration + test_maintenance_lease), so their
    # request volume is O(1) in the fleet size, not O(workers).
