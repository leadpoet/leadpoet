from __future__ import annotations

import pytest

from gateway.research_lab.allocations import (
    _champion_paid_alpha_to_date_from_snapshots,
    _champion_replay_obligation,
    _source_add_paid_alpha_to_date_from_snapshots,
)
from gateway.research_lab.config import ResearchLabGatewayConfig
from leadpoet_verifier.economics import (
    DEFAULT_RESEARCH_LAB_CHAMPION_QUEUE_TRIGGER_RATIO,
    allocate_research_lab_epoch,
)


def _policy(*, lab_cap: float = 20.0) -> dict[str, object]:
    return {
        "policy_id": "test-champion-replay",
        "enabled": True,
        "research_lab_emission_percent": lab_cap,
        "reward_epochs": 20,
        "reimbursement_epochs": 20,
        "reimbursement_allow_overpay_without_champions": True,
        "reimbursement_max_cost_multiplier_with_champions": 1.0,
        "champion_placeholder_alpha_percent": 0.0001,
        "champion_queue_trigger_ratio": 0.50,
        "usd_per_0_1_percent_epoch": 0.6666666667,
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


def _reimbursement(uid: int, *, spend_usd: float, weight: float = 1.0) -> dict[str, object]:
    return {
        "uid": uid,
        "miner_hotkey": f"5Freimburse{uid}",
        "source_id": f"reimbursement:{uid}",
        "island": "generalist",
        "start_epoch": 10,
        "epoch_count": 20,
        "target_reimbursement_microusd": int(spend_usd * 1_000_000),
        "island_weight": weight,
    }


def _reimbursement_paid_for_uid(allocation: dict[str, object], uid: int) -> float:
    return sum(
        float(row["paid_alpha_percent"])
        for row in allocation["reimbursement_allocations"]
        if int(row["uid"]) == uid
    )


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


def test_source_add_replay_counts_only_first_class_snapshot_sections():
    paid = _source_add_paid_alpha_to_date_from_snapshots(
        [
            {
                "allocation_doc": {
                    "champion_allocations": [
                        {
                            "source_id": "source_add_reward:legacy",
                            "reward_kind": "source_acceptance",
                            "paid_alpha_percent": 1.0,
                        }
                    ]
                }
            },
            {
                "allocation_doc": {
                    "source_add_allocations": [
                        {
                            "source_id": "source_add_reward:legacy",
                            "reward_kind": "source_acceptance",
                            "paid_alpha_percent": 1.0,
                        },
                        {
                            "source_id": "source_add_reward:new",
                            "reward_kind": "source_implementation",
                            "paid_alpha_percent": 5.0,
                        },
                    ]
                }
            },
        ]
    )

    assert paid["source_add_reward:legacy"] == pytest.approx(1.0)
    assert paid["source_add_reward:new"] == pytest.approx(5.0)


def test_source_add_replay_does_not_settle_legacy_champion_rail_rows():
    paid = _source_add_paid_alpha_to_date_from_snapshots(
        [
            {
                "allocation_doc": {
                    "champion_allocations": [
                        {
                            "source_id": "source_add_reward:unpaid",
                            "reward_kind": "source_acceptance",
                            "paid_alpha_percent": 1.0,
                        }
                    ]
                }
            }
        ]
    )

    assert paid == {}


def test_replay_tracked_champion_final_epoch_due_capped_surplus_still_flows():
    champions = [
        _champion(1, start_epoch=10, desired=5.0, remaining=1.0),
        _champion(2, start_epoch=11, desired=5.0, remaining=5.0),
    ]

    allocation = allocate_research_lab_epoch(50, _policy(lab_cap=20.0), [], champions)

    # Dues stay capped by each reward's remaining balance (1.0 and 5.0), and
    # the 14.0 surplus splits across active champions by improvement points
    # (equal here) instead of burning as unallocated emission.
    assert _paid_for_uid(allocation, 1) == pytest.approx(8.0)
    assert _paid_for_uid(allocation, 2) == pytest.approx(12.0)
    assert allocation["unallocated_percent"] == pytest.approx(0.0)
    first = allocation["champion_allocations"][0]
    assert first["remaining_alpha_percent_before_epoch"] == pytest.approx(1.0)
    assert first["remaining_alpha_percent_after_epoch"] == pytest.approx(0.0)


def test_active_champion_absorbs_full_lab_slice_no_burn():
    champions = [_champion(1, start_epoch=10, desired=4.0, remaining=70.0)]

    allocation = allocate_research_lab_epoch(50, _policy(lab_cap=20.0), [], champions)

    assert _paid_for_uid(allocation, 1) == pytest.approx(20.0)
    assert allocation["unallocated_percent"] == pytest.approx(0.0)


def test_reimbursements_keep_full_target_until_half_lab_cap_is_exhausted():
    champion = [_champion(99, start_epoch=10, desired=15.0)]

    four_miners = [_reimbursement(uid, spend_usd=500.0) for uid in range(1, 5)]
    allocation = allocate_research_lab_epoch(12, _policy(lab_cap=30.0), four_miners, champion)

    assert allocation["reimbursement_alpha_percent"] == pytest.approx(15.0)
    assert allocation["champion_alpha_percent"] == pytest.approx(15.0)
    for row in allocation["reimbursement_allocations"]:
        assert row["paid_alpha_percent"] == pytest.approx(3.75)
        assert row["intended_alpha_percent"] == pytest.approx(3.75)
        assert row["reason"] == "full_reimbursement"

    five_miners = [_reimbursement(uid, spend_usd=500.0) for uid in range(1, 6)]
    allocation = allocate_research_lab_epoch(12, _policy(lab_cap=30.0), five_miners, champion)

    assert allocation["reimbursement_alpha_percent"] == pytest.approx(15.0)
    assert allocation["champion_alpha_percent"] == pytest.approx(15.0)
    for row in allocation["reimbursement_allocations"]:
        assert row["paid_alpha_percent"] == pytest.approx(3.0)
        assert row["intended_alpha_percent"] == pytest.approx(3.75)
        assert row["reason"] == "scaled_by_lab_capacity"


def test_low_desired_champion_still_caps_reimbursements_at_queue_trigger_ratio():
    reimbursements = [_reimbursement(uid, spend_usd=500.0) for uid in range(1, 20)]
    champion = [_champion(99, start_epoch=10, desired=7.0)]

    allocation = allocate_research_lab_epoch(12, _policy(lab_cap=30.0), reimbursements, champion)

    assert allocation["reimbursement_alpha_percent"] == pytest.approx(15.0)
    assert allocation["champion_alpha_percent"] == pytest.approx(15.0)
    assert _reimbursement_paid_for_uid(allocation, 1) == pytest.approx(15.0 / 19.0)
    assert allocation["unallocated_percent"] == pytest.approx(0.0)


def test_champion_queue_trigger_default_is_shared_by_gateway_and_verifier():
    assert ResearchLabGatewayConfig().lab_champion_queue_trigger_ratio == pytest.approx(
        float(DEFAULT_RESEARCH_LAB_CHAMPION_QUEUE_TRIGGER_RATIO)
    )
    policy = _policy(lab_cap=30.0)
    policy.pop("champion_queue_trigger_ratio")
    reimbursements = [_reimbursement(uid, spend_usd=500.0) for uid in range(1, 20)]
    champion = [_champion(99, start_epoch=10, desired=7.0)]

    allocation = allocate_research_lab_epoch(12, policy, reimbursements, champion)

    assert allocation["reimbursement_alpha_percent"] == pytest.approx(15.0)
    assert allocation["champion_alpha_percent"] == pytest.approx(15.0)


def test_no_champion_reimbursements_can_use_full_lab_cap_pro_rata():
    reimbursements = [_reimbursement(uid, spend_usd=500.0) for uid in range(1, 6)]

    allocation = allocate_research_lab_epoch(12, _policy(lab_cap=30.0), reimbursements, [])

    assert allocation["reimbursement_alpha_percent"] == pytest.approx(30.0)
    assert allocation["champion_alpha_percent"] == pytest.approx(0.0)
    for row in allocation["reimbursement_allocations"]:
        assert row["paid_alpha_percent"] == pytest.approx(6.0)
        assert row["overpaid_alpha_percent"] == pytest.approx(2.25)
        assert row["reason"] == "overpay_no_active_champions"


def test_crowded_reimbursements_scale_by_spend_and_island_weight():
    champion = [_champion(99, start_epoch=10, desired=15.0)]
    reimbursements = [
        _reimbursement(1, spend_usd=400.0),
        _reimbursement(2, spend_usd=500.0),
        _reimbursement(3, spend_usd=600.0),
        _reimbursement(4, spend_usd=700.0),
        _reimbursement(5, spend_usd=800.0),
    ]

    allocation = allocate_research_lab_epoch(12, _policy(lab_cap=30.0), reimbursements, champion)

    assert allocation["reimbursement_alpha_percent"] == pytest.approx(15.0)
    assert [_reimbursement_paid_for_uid(allocation, uid) for uid in range(1, 6)] == pytest.approx(
        [2.0, 2.5, 3.0, 3.5, 4.0]
    )

    weighted = [
        _reimbursement(1, spend_usd=2_000.0, weight=1.0),
        _reimbursement(2, spend_usd=2_000.0, weight=2.0),
    ]
    allocation = allocate_research_lab_epoch(12, _policy(lab_cap=30.0), weighted, champion)

    assert _reimbursement_paid_for_uid(allocation, 1) == pytest.approx(5.0)
    assert _reimbursement_paid_for_uid(allocation, 2) == pytest.approx(10.0)


def test_multiple_champions_still_queue_when_half_lab_cap_is_tight():
    champion = [
        _champion(1, start_epoch=10, desired=15.0),
        _champion(2, start_epoch=11, desired=15.0),
        _champion(3, start_epoch=12, desired=15.0),
    ]
    reimbursements = [_reimbursement(uid, spend_usd=500.0) for uid in range(1, 20)]

    allocation = allocate_research_lab_epoch(12, _policy(lab_cap=30.0), reimbursements, champion)

    assert allocation["reimbursement_alpha_percent"] == pytest.approx(14.9998)
    assert allocation["champion_allocations"][0]["uid"] == 1
    assert allocation["champion_allocations"][0]["paid_alpha_percent"] == pytest.approx(15.0)
    assert allocation["queued_champion_allocations"]
    assert allocation["queued_champion_allocations"][0]["paid_alpha_percent"] > 0
