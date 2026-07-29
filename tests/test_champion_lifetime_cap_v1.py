from __future__ import annotations

from copy import deepcopy
from decimal import Decimal

import pytest

from gateway.research_lab.allocations import (
    _champion_lifetime_credit_ledger_from_snapshots,
)
from leadpoet_canonical.weight_computation import (
    normalize_to_u16_with_uids_pure,
    research_lab_uid_weights_from_allocation,
)
from leadpoet_verifier.economics import (
    CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1,
    allocate_research_lab_epoch,
)


def _policy(*, lab_cap: float = 30.0) -> dict[str, object]:
    return {
        "policy_id": "champion-lifetime-cap-test",
        "enabled": True,
        "research_lab_emission_percent": lab_cap,
        "reward_epochs": 20,
        "reimbursement_epochs": 20,
        "reimbursement_max_cost_multiplier_with_champions": 1.0,
        "champion_placeholder_alpha_percent": 0.0001,
        "champion_queue_trigger_ratio": 0.50,
        "usd_per_0_1_percent_epoch": 0.6666666667,
    }


def _champion(
    uid: int,
    *,
    desired: float,
    remaining: float,
    improvement_points: float = 1.0,
) -> dict[str, object]:
    total_due = desired * 20
    return {
        "uid": uid,
        "miner_hotkey": f"champion-hotkey-{uid}",
        "source_id": f"champion_reward:{uid}",
        "champion_reward_id": f"champion_reward:{uid}",
        "island": "generalist",
        "start_epoch": 100,
        "epoch_count": 20,
        "improvement_points": improvement_points,
        "desired_alpha_percent": desired,
        "total_due_alpha_percent": total_due,
        "paid_alpha_percent_to_date": total_due - remaining,
        "remaining_alpha_percent": remaining,
        "replay_status": "extended_replay",
    }


def _champion_rows(allocation: dict[str, object]) -> list[dict[str, object]]:
    return [
        *allocation["champion_allocations"],
        *allocation["queued_champion_allocations"],
    ]


def _paid(allocation: dict[str, object], uid: int) -> float:
    return sum(
        float(row["paid_alpha_percent"])
        for row in _champion_rows(allocation)
        if int(row["uid"]) == uid
    )


def test_73_percent_champion_receives_exact_146_percent_lifetime_cap():
    remaining = 146.0
    paid_by_epoch: list[float] = []
    burn_by_epoch: list[float] = []

    for epoch in range(100, 106):
        allocation = allocate_research_lab_epoch(
            epoch,
            _policy(),
            [],
            [_champion(7, desired=7.3, remaining=remaining)],
        )
        assert allocation["champion_credit_policy"] == (
            CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
        )
        paid = _paid(allocation, 7)
        paid_by_epoch.append(paid)
        uid_weights, burn_share, _breakdown = (
            research_lab_uid_weights_from_allocation(
                allocation,
                metagraph_hotkeys=[
                    "burn-hotkey",
                    *[f"unused-{uid}" for uid in range(1, 7)],
                    "champion-hotkey-7",
                ],
                reserved_share=0.30,
            )
        )
        burn_by_epoch.append(burn_share)
        assert uid_weights.get(7, 0.0) == pytest.approx(paid / 100.0)
        remaining = max(0.0, remaining - paid)

    assert paid_by_epoch == pytest.approx([30.0, 30.0, 30.0, 30.0, 26.0, 0.0])
    assert sum(paid_by_epoch) == pytest.approx(146.0)
    assert burn_by_epoch == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.04, 0.30])


def test_surplus_redistributes_after_one_champion_reaches_cap():
    allocation = allocate_research_lab_epoch(
        200,
        _policy(),
        [],
        [
            _champion(
                1,
                desired=2.0,
                remaining=4.0,
                improvement_points=1.0,
            ),
            _champion(
                2,
                desired=2.0,
                remaining=26.0,
                improvement_points=3.0,
            ),
        ],
    )

    assert _paid(allocation, 1) == pytest.approx(4.0)
    assert _paid(allocation, 2) == pytest.approx(26.0)
    assert allocation["unallocated_percent"] == pytest.approx(0.0)


def test_zero_point_fallback_redistributes_without_exceeding_caps():
    allocation = allocate_research_lab_epoch(
        200,
        _policy(lab_cap=10.0),
        [],
        [
            _champion(
                1,
                desired=1.0,
                remaining=3.0,
                improvement_points=0.0,
            ),
            _champion(
                2,
                desired=1.0,
                remaining=7.0,
                improvement_points=0.0,
            ),
        ],
    )

    assert _paid(allocation, 1) == pytest.approx(3.0)
    assert _paid(allocation, 2) == pytest.approx(7.0)
    assert allocation["unallocated_percent"] == pytest.approx(0.0)


def test_chronological_queue_payment_never_exceeds_lifetime_balance():
    older = _champion(
        1,
        desired=8.0,
        remaining=8.0,
        improvement_points=1.0,
    )
    newer = _champion(
        2,
        desired=8.0,
        remaining=16.0,
        improvement_points=3.0,
    )
    newer["start_epoch"] = 101
    allocation = allocate_research_lab_epoch(
        200,
        _policy(lab_cap=10.0),
        [],
        [older, newer],
    )

    assert _paid(allocation, 1) == pytest.approx(8.0)
    assert _paid(allocation, 2) == pytest.approx(2.0)
    assert [row["uid"] for row in allocation["champion_allocations"]] == [1]
    assert [row["uid"] for row in allocation["queued_champion_allocations"]] == [2]
    assert allocation["queued_champion_allocations"][0][
        "remaining_alpha_percent_after_epoch"
    ] == pytest.approx(14.0)
    assert allocation["unallocated_percent"] == pytest.approx(0.0)


def test_six_decimal_rounding_conserves_pool_and_emits_valid_u16():
    allocation = allocate_research_lab_epoch(
        200,
        _policy(lab_cap=1.000001),
        [],
        [
            _champion(
                1,
                desired=0.1,
                remaining=2.0,
                improvement_points=1.0,
            ),
            _champion(
                2,
                desired=0.1,
                remaining=2.0,
                improvement_points=2.0,
            ),
            _champion(
                3,
                desired=0.1,
                remaining=2.0,
                improvement_points=3.0,
            ),
        ],
    )
    rows = _champion_rows(allocation)
    paid = sum(
        (Decimal(str(row["paid_alpha_percent"])) for row in rows),
        Decimal("0"),
    )
    unallocated = Decimal(str(allocation["unallocated_percent"]))

    assert paid + unallocated == Decimal("1.000001")
    uid_weights, burn_share, _breakdown = (
        research_lab_uid_weights_from_allocation(
            allocation,
            metagraph_hotkeys=["burn", "champion-hotkey-1", "champion-hotkey-2", "champion-hotkey-3"],
            reserved_share=0.30,
        )
    )
    float_weights = [
        0.70 + burn_share,
        uid_weights[1],
        uid_weights[2],
        uid_weights[3],
    ]
    assert sum(float_weights) == pytest.approx(1.0, abs=1e-15)
    sparse_uids, sparse_weights = normalize_to_u16_with_uids_pure(
        [0, 1, 2, 3],
        float_weights,
    )
    assert sparse_uids == [0, 1, 2, 3]
    assert all(0 < value <= 65535 for value in sparse_weights)
    assert max(sparse_weights) == 65535


def test_exhausted_champion_remains_zero_for_100_epochs():
    remaining = 146.0
    total_paid = 0.0

    for epoch in range(100, 200):
        allocation = allocate_research_lab_epoch(
            epoch,
            _policy(),
            [],
            [_champion(7, desired=7.3, remaining=remaining)],
        )
        paid = _paid(allocation, 7)
        total_paid += paid
        remaining = max(0.0, remaining - paid)
        if epoch >= 105:
            assert paid == 0.0
            assert allocation["unallocated_percent"] == pytest.approx(30.0)

    assert total_paid == pytest.approx(146.0)
    assert remaining == 0.0


def test_inconsistent_lifetime_balance_fails_closed():
    champion = _champion(1, desired=7.3, remaining=146.0)
    champion["total_due_alpha_percent"] = 147.0

    with pytest.raises(
        ValueError,
        match="total_due_alpha_percent differs",
    ):
        allocate_research_lab_epoch(100, _policy(), [], [champion])

    champion = _champion(1, desired=7.3, remaining=100.0)
    champion["paid_alpha_percent_to_date"] = 40.0
    with pytest.raises(ValueError, match="lifetime balance is inconsistent"):
        allocate_research_lab_epoch(100, _policy(), [], [champion])


def test_marked_realized_credit_retires_lifetime_and_exposes_stale_excess(
    monkeypatch,
):
    monkeypatch.setenv(
        "RESEARCH_LAB_CHAMPION_SCHEDULE_CAP_START_EPOCH",
        "24100",
    )
    reward_id = "champion_reward:7"
    rows = [
        {
            "epoch": 24099,
            "allocation_doc": {
                "champion_allocations": [
                    {
                        "source_id": reward_id,
                        "paid_alpha_percent": 30.0,
                        "base_desired_alpha_percent": 7.3,
                    }
                ],
                "queued_champion_allocations": [],
            },
        },
        {
            "epoch": 24100,
            "allocation_doc": {
                "champion_allocations": [
                    {
                        "source_id": reward_id,
                        "paid_alpha_percent": 30.0,
                        "base_desired_alpha_percent": 7.3,
                    }
                ],
                "queued_champion_allocations": [],
            },
        },
        {
            "epoch": 24101,
            "allocation_doc": {
                "epoch": 24101,
                "champion_credit_policy": (
                    CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
                ),
                "champion_allocations": [
                    {
                        "source_id": reward_id,
                        "paid_alpha_percent": 100.0,
                    }
                ],
                "queued_champion_allocations": [],
            },
        },
        {
            "epoch": 24102,
            "allocation_doc": {
                "epoch": 24102,
                "champion_credit_policy": (
                    CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
                ),
                "champion_allocations": [
                    {
                        "source_id": reward_id,
                        "paid_alpha_percent": 100.0,
                    }
                ],
                "queued_champion_allocations": [],
            },
        },
    ]

    ledger = _champion_lifetime_credit_ledger_from_snapshots(
        rows,
        obligation_caps={reward_id: 146.0},
    )

    # Pre-cutover legacy surplus remains grandfathered. Post-cutover unmarked
    # rows retain scheduled-credit behavior. Marked rows use actual emissions.
    assert ledger["realized_by_reward"][reward_id] == pytest.approx(237.3)
    assert ledger["applied_by_reward"][reward_id] == pytest.approx(146.0)
    assert ledger["excess_by_reward"][reward_id] == pytest.approx(91.3)


def test_marked_replay_is_order_independent_and_rejects_duplicate_epochs():
    marker = CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
    rows = [
        {
            "epoch": epoch,
            "allocation_doc": {
                "epoch": epoch,
                "champion_credit_policy": marker,
                "champion_allocations": [
                    {
                        "source_id": "champion_reward:7",
                        "paid_alpha_percent": paid,
                    }
                ],
                "queued_champion_allocations": [],
            },
        }
        for epoch, paid in ((24102, 30.0), (24100, 30.0), (24101, 30.0))
    ]
    expected = _champion_lifetime_credit_ledger_from_snapshots(
        rows,
        obligation_caps={"champion_reward:7": 146.0},
    )

    assert expected == _champion_lifetime_credit_ledger_from_snapshots(
        list(reversed(rows)),
        obligation_caps={"champion_reward:7": 146.0},
    )
    with pytest.raises(ValueError, match="epoch is duplicated"):
        _champion_lifetime_credit_ledger_from_snapshots(
            [rows[0], deepcopy(rows[0])],
            obligation_caps={"champion_reward:7": 146.0},
        )
    mismatched = deepcopy(rows[0])
    mismatched["allocation_doc"]["epoch"] = int(mismatched["epoch"]) + 1
    with pytest.raises(ValueError, match="allocation epoch differs"):
        _champion_lifetime_credit_ledger_from_snapshots(
            [mismatched],
            obligation_caps={"champion_reward:7": 146.0},
        )


@pytest.mark.parametrize(
    "allocation_doc",
    (
        {"champion_credit_policy": "unknown_policy"},
        {
            "epoch": 24100,
            "champion_credit_policy": (
                CHAMPION_CREDIT_POLICY_ACCELERATED_LIFETIME_CAP_V1
            ),
            "champion_allocations": [
                {
                    "source_id": "champion_reward:7",
                    "paid_alpha_percent": 1.0,
                    "champion_credit_policy": "other_policy",
                }
            ],
        },
    ),
)
def test_unknown_or_mixed_policy_evidence_fails_closed(allocation_doc):
    with pytest.raises(ValueError, match="policy"):
        _champion_lifetime_credit_ledger_from_snapshots(
            [{"epoch": 24100, "allocation_doc": allocation_doc}],
            obligation_caps={"champion_reward:7": 146.0},
        )


def test_only_champion_and_burn_weights_change_when_lifetime_cap_binds():
    baseline = {
        "lab_cap_percent": 30.0,
        "unallocated_percent": 0.0,
        "source_add_allocations": [
            {
                "uid": 3,
                "miner_hotkey": "source-add",
                "paid_alpha_percent": 2.0,
            }
        ],
        "reimbursement_allocations": [
            {
                "uid": 4,
                "miner_hotkey": "reimbursement",
                "paid_alpha_percent": 3.0,
            }
        ],
        "champion_allocations": [
            {
                "uid": 5,
                "miner_hotkey": "champion",
                "paid_alpha_percent": 25.0,
            }
        ],
        "queued_champion_allocations": [],
    }
    capped = deepcopy(baseline)
    capped["champion_allocations"][0]["paid_alpha_percent"] = 21.0
    capped["unallocated_percent"] = 4.0
    hotkeys = ["burn", "unused-1", "unused-2", "source-add", "reimbursement", "champion"]

    baseline_weights, baseline_burn, _ = (
        research_lab_uid_weights_from_allocation(
            baseline,
            metagraph_hotkeys=hotkeys,
            reserved_share=0.30,
        )
    )
    capped_weights, capped_burn, _ = research_lab_uid_weights_from_allocation(
        capped,
        metagraph_hotkeys=hotkeys,
        reserved_share=0.30,
    )

    assert capped_weights[3] == baseline_weights[3]
    assert capped_weights[4] == baseline_weights[4]
    assert capped_weights[5] == pytest.approx(baseline_weights[5] - 0.04)
    assert capped_burn == pytest.approx(baseline_burn + 0.04)
    assert sum(capped_weights.values()) + capped_burn == pytest.approx(0.30)
