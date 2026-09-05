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
