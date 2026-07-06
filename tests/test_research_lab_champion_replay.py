from __future__ import annotations

import pytest

from gateway.research_lab.allocations import (
    _champion_paid_alpha_to_date_from_snapshots,
    _champion_replay_obligation,
)
from leadpoet_verifier.economics import allocate_research_lab_epoch


def _policy(*, lab_cap: float = 20.0) -> dict[str, object]:
    return {
        "policy_id": "test-champion-replay",
        "enabled": True,
        "research_lab_emission_percent": lab_cap,
        "reward_epochs": 20,
        "reimbursement_epochs": 20,
        "champion_placeholder_alpha_percent": 0.0001,
    }


def _champion(uid: int, *, start_epoch: int, desired: float, remaining: float | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "uid": uid,
        "miner_hotkey": f"5Fchampion{uid}",
        "source_id": f"champion_reward:{uid}",
        "island": "generalist",
        "start_epoch": start_epoch,
        "epoch_count": 20,
        "improvement_points": 1.0,
        "desired_alpha_percent": desired,
    }
    if remaining is not None:
        total_due = desired * 20
        row.update(
            {
                "total_due_alpha_percent": total_due,
                "paid_alpha_percent_to_date": total_due - remaining,
                "remaining_alpha_percent": remaining,
                "replay_status": "extended_replay",
            }
        )
    return row


def _paid_for_uid(allocation: dict[str, object], uid: int) -> float:
    rows = list(allocation["champion_allocations"]) + list(allocation["queued_champion_allocations"])
    return sum(float(row["paid_alpha_percent"]) for row in rows if int(row["uid"]) == uid)


def test_champion_capacity_flows_chronologically_to_first_unpaid_reward():
    champions = [
        _champion(1, start_epoch=10, desired=5.0),
        _champion(2, start_epoch=11, desired=5.0),
        _champion(3, start_epoch=12, desired=5.0),
        _champion(4, start_epoch=13, desired=5.0),
        _champion(5, start_epoch=14, desired=5.0),
    ]

    allocation = allocate_research_lab_epoch(20, _policy(lab_cap=18.0), [], champions)

    assert _paid_for_uid(allocation, 1) == pytest.approx(5.0)
    assert _paid_for_uid(allocation, 2) == pytest.approx(5.0)
    assert _paid_for_uid(allocation, 3) == pytest.approx(5.0)
    assert _paid_for_uid(allocation, 4) == pytest.approx(3.0)
    assert _paid_for_uid(allocation, 5) == pytest.approx(0.0)
    assert allocation["queued_champion_allocations"][0]["uid"] == 4
    assert allocation["queued_champion_allocations"][0]["reason"] == "queued_with_partial_capacity"


def test_champion_replay_state_sums_prior_paid_alpha_from_snapshots():
    paid = _champion_paid_alpha_to_date_from_snapshots(
        [
            {
                "epoch": 100,
                "allocation_doc": {
                    "champion_allocations": [{"source_id": "champion_reward:abc", "paid_alpha_percent": 4.0}],
                    "queued_champion_allocations": [{"source_id": "champion_reward:def", "paid_alpha_percent": 0.25}],
                },
            },
            {
                "epoch": 101,
                "allocation_doc": {
                    "champion_allocations": [{"source_id": "champion_reward:abc", "paid_alpha_percent": 5.0}],
                    "queued_champion_allocations": [{"source_id": "champion_reward:def", "paid_alpha_percent": 1.75}],
                },
            },
        ]
    )

    assert paid["champion_reward:abc"] == pytest.approx(9.0)
    assert paid["champion_reward:def"] == pytest.approx(2.0)

    replay = _champion_replay_obligation(
        {
            "champion_reward_id": "champion_reward:def",
            "start_epoch": 100,
            "epoch_count": 20,
            "improvement_points": 2.0,
            "threshold_points": 1.0,
            "desired_alpha_percent": 4.0,
        },
        paid_by_reward=paid,
        epoch=121,
    )
    assert replay is not None
    assert replay["replay_status"] == "extended_replay"
    assert replay["total_due_alpha_percent"] == pytest.approx(80.0)
    assert replay["paid_alpha_percent_to_date"] == pytest.approx(2.0)
    assert replay["remaining_alpha_percent"] == pytest.approx(78.0)


def test_replay_tracked_champion_final_epoch_cannot_overpay_remaining_balance():
    champions = [
        _champion(1, start_epoch=10, desired=5.0, remaining=1.0),
        _champion(2, start_epoch=11, desired=5.0, remaining=5.0),
    ]

    allocation = allocate_research_lab_epoch(50, _policy(lab_cap=20.0), [], champions)

    assert _paid_for_uid(allocation, 1) == pytest.approx(1.0)
    assert _paid_for_uid(allocation, 2) == pytest.approx(5.0)
    assert allocation["unallocated_percent"] == pytest.approx(14.0)
    first = allocation["champion_allocations"][0]
    assert first["remaining_alpha_percent_before_epoch"] == pytest.approx(1.0)
    assert first["remaining_alpha_percent_after_epoch"] == pytest.approx(0.0)
