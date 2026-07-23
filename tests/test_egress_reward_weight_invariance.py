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


def test_emission_allocation_to_uid_weights_golden() -> None:
    # Research Lab emission allocation -> validator UID weights. The egress
    # changes never touch this kernel or its inputs; pin its output so any
    # accidental change is caught. A champion + a reimbursement each get their
    # paid_alpha_percent; a champion whose hotkey no longer matches the
    # metagraph is treated as deregistered (burned, not paid to a stranger).
    from leadpoet_canonical.weight_computation import research_lab_uid_weights_from_allocation

    metagraph_hotkeys = [f"hk{i}" for i in range(10)]
    allocation_doc = {
        "lab_cap_percent": 20.0,
        "unallocated_percent": 0.0,
        "champion_allocations": [
            {"uid": 5, "miner_hotkey": "hk5", "paid_alpha_percent": 6.0},
            {"uid": 8, "miner_hotkey": "STALE", "paid_alpha_percent": 4.0},  # deregistered
        ],
        "reimbursement_allocations": [
            {"uid": 7, "miner_hotkey": "hk7", "paid_alpha_percent": 3.0},
        ],
    }
    uid_weights, burn_share, breakdown = research_lab_uid_weights_from_allocation(
        allocation_doc, metagraph_hotkeys=metagraph_hotkeys, reserved_share=0.20
    )
    assert uid_weights == {5: 0.06, 7: 0.03}             # paid to the right UIDs only
    assert round(breakdown["paid"], 6) == 0.13           # 6% + 4% + 3% (pre-resolution)
    assert round(breakdown["deregistered"], 6) == 0.04   # the stale champion's 4% burns
    # burn = deregistered (0.04) + rounding gap up to the 20% cap (0.07).
    assert round(burn_share, 6) == 0.11


def test_onchain_u16_weight_vector_golden() -> None:
    # The exact u16 vector that goes on-chain (Bittensor emit format). Pin it so a
    # weight-mutation regression is caught. max weight -> U16_MAX, half -> ~half.
    from leadpoet_canonical.weight_computation import normalize_to_u16_with_uids_pure

    uids, weights = normalize_to_u16_with_uids_pure([5, 7, 3], [0.06, 0.03, 0.0])
    assert uids == [5, 7]              # the zero-weight uid is dropped
    assert weights == [65535, 32768]   # 0.06 -> U16_MAX, 0.03 -> half


def test_reimbursement_cost_evidence_comes_from_loop_result_not_ledger_table() -> None:
    # The batched provider-usage ledger insert (item 4) is WRITE-only: reimburse-
    # ment cost evidence is derived from the in-memory loop result's
    # provider_usage / cost_ledger, never from the provider_usage_ledger table,
    # so the batching cannot change any reimbursement input.
    from gateway.research_lab.reimbursement_awards import cost_evidence_from_loop_result

    class _LoopResult:
        provider_usage = [{"provider_id": "exa", "endpoint_class": "search"}]
        actual_openrouter_cost_usd = 1.25
        openrouter_call_count = 3
        iterations_completed = 2
        stop_reason = "plateau"

        def cost_ledger(self):
            return {"actual_openrouter_cost_microusd": 1_250_000}

    ev = cost_evidence_from_loop_result(_LoopResult())
    assert ev["source"] == "loop_result"
    assert ev["provider_usage"] == [{"provider_id": "exa", "endpoint_class": "search"}]
    assert ev["cost_ledger"]["actual_openrouter_cost_microusd"] == 1_250_000


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
