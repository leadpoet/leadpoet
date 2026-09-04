from __future__ import annotations

import random

import pytest

from gateway.research_lab.allocations import (
    _champion_paid_alpha_to_date_from_snapshots,
    _champion_replay_obligation,
    _source_add_paid_alpha_to_date_from_snapshots,
)
from gateway.research_lab.config import ResearchLabGatewayConfig
from leadpoet_canonical.weight_computation import (
    WEIGHT_SNAPSHOT_SCHEMA_VERSION,
    compute_final_weights,
    research_lab_uid_weights_from_allocation,
    weight_config_hash,
)
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
                            "base_desired_alpha_percent": 5.0,
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


def test_source_add_chain_quantization_retires_the_nominal_epoch_schedule():
    reward_ref = "source_add_reward:quantized"
    snapshot_rows = [
        {
            "allocation_doc": {
                "source": "chain_realized_obligation_credits",
                "authority_type": "chain_realized_emission_v1",
                "source_add_allocations": [
                    {
                        "source_id": reward_ref,
                        "source_add_reward_id": reward_ref,
                        "paid_alpha_percent": "0.199494411111",
                        "base_desired_alpha_percent": "0.2",
                        "lab_attributed_alpha_percent": "0.199494411111",
                        "observed_chain_alpha_percent": "0.199494411111",
                    }
                ],
            }
        }
        for _epoch in range(20)
    ]

    paid = _source_add_paid_alpha_to_date_from_snapshots(snapshot_rows)

    assert paid[reward_ref] == pytest.approx(4.0)
    assert (
        _champion_replay_obligation(
            {
                "champion_reward_id": reward_ref,
                "start_epoch": 100,
                "epoch_count": 20,
                "desired_alpha_percent": 0.2,
            },
            paid_by_reward=paid,
            epoch=120,
        )
        is None
    )


def test_source_add_chain_underpayment_remains_due_after_nominal_window():
    reward_ref = "source_add_reward:underpaid"
    paid = _source_add_paid_alpha_to_date_from_snapshots(
        [
            {
                "allocation_doc": {
                    "source": "chain_realized_obligation_credits",
                    "authority_type": "chain_realized_emission_v1",
                    "source_add_allocations": [
                        {
                            "source_id": reward_ref,
                            "paid_alpha_percent": "0.19",
                            "base_desired_alpha_percent": "0.2",
                            "lab_attributed_alpha_percent": "0.19",
                            "observed_chain_alpha_percent": "0.19",
                        }
                    ],
                }
            }
            for _epoch in range(20)
        ]
    )

    replay = _champion_replay_obligation(
        {
            "champion_reward_id": reward_ref,
            "start_epoch": 100,
            "epoch_count": 20,
            "desired_alpha_percent": 0.2,
        },
        paid_by_reward=paid,
        epoch=120,
    )

    assert paid[reward_ref] == pytest.approx(3.8)
    assert replay is not None
    assert replay["remaining_alpha_percent"] == pytest.approx(0.2)


def test_source_add_chain_credit_mismatch_fails_closed():
    with pytest.raises(
        ValueError, match="SOURCE_ADD chain settlement credit is invalid"
    ):
        _source_add_paid_alpha_to_date_from_snapshots(
            [
                {
                    "allocation_doc": {
                        "source": "chain_realized_obligation_credits",
                        "authority_type": "chain_realized_emission_v1",
                        "source_add_allocations": [
                            {
                                "source_id": "source_add_reward:invalid",
                                "paid_alpha_percent": "0.19",
                                "base_desired_alpha_percent": "0.2",
                                "lab_attributed_alpha_percent": "0.2",
                                "observed_chain_alpha_percent": "0.2",
                            }
                        ],
                    }
                }
            ]
        )


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


def test_replay_tracked_champion_final_epoch_caps_surplus_at_lifetime_balance():
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


def test_active_champion_absorbs_full_lab_slice_no_burn():
    champions = [_champion(1, start_epoch=10, desired=4.0, remaining=70.0)]

    allocation = allocate_research_lab_epoch(50, _policy(lab_cap=20.0), [], champions)

    assert _paid_for_uid(allocation, 1) == pytest.approx(20.0)
    assert allocation["unallocated_percent"] == pytest.approx(0.0)


def test_champion_final_balance_and_later_epochs_flow_to_reimbursement_no_burn():
    reimbursement = [_reimbursement(2, spend_usd=13.333333334)]
    champion = _champion(
        99,
        start_epoch=10,
        desired=7.25,
        remaining=145.0,
    )
    champion_paid_by_epoch = []
    reimbursement_paid_by_epoch = []
    remaining = 145.0

    for epoch in range(10, 30):
        active_champions = [champion] if remaining > 0 else []
        allocation = allocate_research_lab_epoch(
            epoch,
            _policy(lab_cap=30.0),
            reimbursement,
            active_champions,
        )
        champion_paid = _paid_for_uid(allocation, 99)
        reimbursement_paid = _reimbursement_paid_for_uid(allocation, 2)
        champion_paid_by_epoch.append(champion_paid)
        reimbursement_paid_by_epoch.append(reimbursement_paid)
        assert allocation["unallocated_percent"] == pytest.approx(0.0)

        remaining = max(0.0, remaining - champion_paid)
        champion["paid_alpha_percent_to_date"] = 145.0 - remaining
        champion["remaining_alpha_percent"] = remaining

    assert champion_paid_by_epoch[:5] == pytest.approx(
        [29.9, 29.9, 29.9, 29.9, 25.4]
    )
    assert champion_paid_by_epoch[5:] == pytest.approx([0.0] * 15)
    assert sum(champion_paid_by_epoch) == pytest.approx(145.0)
    assert reimbursement_paid_by_epoch[:5] == pytest.approx(
        [0.1, 0.1, 0.1, 0.1, 4.6]
    )
    assert reimbursement_paid_by_epoch[5:] == pytest.approx([30.0] * 15)


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


def test_no_champion_reimbursements_use_full_lab_cap_pro_rata():
    reimbursements = [_reimbursement(uid, spend_usd=500.0) for uid in range(1, 6)]

    policy = _policy(lab_cap=30.0)
    policy["reimbursement_allow_overpay_without_champions"] = True
    allocation = allocate_research_lab_epoch(12, policy, reimbursements, [])

    assert allocation["reimbursement_alpha_percent"] == pytest.approx(30.0)
    assert allocation["champion_alpha_percent"] == pytest.approx(0.0)
    assert allocation["unallocated_percent"] == pytest.approx(0.0)
    for row in allocation["reimbursement_allocations"]:
        assert row["paid_alpha_percent"] == pytest.approx(6.0)
        assert row["overpaid_alpha_percent"] == pytest.approx(2.25)
        assert row["reason"] == "surplus_reimbursement_no_burn"


def test_reimbursement_surplus_is_proportional_to_compute_weight():
    reimbursements = [
        _reimbursement(1, spend_usd=100.0, weight=1.0),
        _reimbursement(2, spend_usd=100.0, weight=3.0),
    ]

    allocation = allocate_research_lab_epoch(
        12,
        _policy(lab_cap=30.0),
        reimbursements,
        [],
    )

    rows = allocation["reimbursement_allocations"]
    assert allocation["reimbursement_alpha_percent"] == pytest.approx(30.0)
    assert allocation["unallocated_percent"] == pytest.approx(0.0)
    assert rows[1]["overpaid_alpha_percent"] == pytest.approx(
        rows[0]["overpaid_alpha_percent"] * 3
    )


def test_legacy_overpay_environment_switch_is_inert(monkeypatch):
    monkeypatch.setenv(
        "RESEARCH_LAB_REIMBURSEMENT_ALLOW_OVERPAY_WITHOUT_CHAMPIONS",
        "true",
    )

    config = ResearchLabGatewayConfig.from_env()

    assert (
        config.reimbursement_policy_doc(enabled=True)[
            "reimbursement_allow_overpay_without_champions"
        ]
        is False
    )


def test_no_champion_reimbursements_scale_down_when_set_rates_exceed_lab_cap():
    reimbursements = [_reimbursement(uid, spend_usd=500.0) for uid in range(1, 6)]

    allocation = allocate_research_lab_epoch(12, _policy(lab_cap=10.0), reimbursements, [])

    assert allocation["reimbursement_alpha_percent"] == pytest.approx(10.0)
    assert allocation["unallocated_percent"] == pytest.approx(0.0)
    for row in allocation["reimbursement_allocations"]:
        assert row["paid_alpha_percent"] == pytest.approx(2.0)
        assert row["paid_alpha_percent"] < row["intended_alpha_percent"]
        assert row["overpaid_alpha_percent"] == pytest.approx(0.0)
        assert row["reason"] == "scaled_by_lab_capacity"


def test_no_champion_reimbursement_eliminates_final_weight_burn():
    reimbursement = [_reimbursement(2, spend_usd=500.0)]
    allocation = allocate_research_lab_epoch(
        12,
        _policy(lab_cap=30.0),
        reimbursement,
        [],
    )
    snapshot = {
        "schema_version": WEIGHT_SNAPSHOT_SCHEMA_VERSION,
        "netuid": 71,
        "epoch_id": 12,
        "block": 4679,
        "commit_sha": "a" * 40,
        "config_hash": "",
        "parent_receipt_hashes": [],
        "research_lab_allocation_receipt_hash": "",
        "burn_target_uid": 0,
        "expected_burn_target_hotkey": "burn-hotkey",
        "metagraph_hotkeys": [
            "burn-hotkey",
            "fulfillment-hotkey",
            "5Freimburse2",
        ],
        "banned_hotkeys": [],
        "banned_lookup_ok": True,
        "ff_enabled": True,
        "base_burn_share": 0.0,
        "champion_share": 0.0,
        "champion_uid": None,
        "effective_champion_share": 0.0,
        "research_lab_fallback_share": 0.30,
        "research_lab_allocation_doc": allocation,
        "leaderboard_bonus_share": 0.0,
        "leaderboard_rank_shares": [],
        "leaderboard_entries": [],
        "leaderboard_fetch_ok": True,
        "fulfillment_share": 0.70,
        "fulfillment_rows": [
            {"hotkey": "fulfillment-hotkey", "share": 0.70}
        ],
        "fulfillment_fetch_ok": True,
        "rolling_lead_count": 0,
        "rolling_scores": [],
        "sourcing_floor_threshold": 125_000,
        "min_total_rep_for_distribution": 100,
    }
    snapshot["config_hash"] = weight_config_hash(snapshot)

    result = compute_final_weights(snapshot)

    assert allocation["reimbursement_alpha_percent"] == pytest.approx(30.0)
    assert allocation["unallocated_percent"] == pytest.approx(0.0)
    assert result["uids"] == [0, 1, 2]
    assert result["weights"] == pytest.approx([0.0, 0.70, 0.30])
    assert result["components"]["research_lab_burn"] == pytest.approx(0.0)
    assert result["components"]["research_lab_paid"] == pytest.approx(0.30)


def test_no_champion_reimbursement_uses_full_cap_across_randomized_inputs():
    rng = random.Random(710_2026_07_28)
    for case in range(1_000):
        lab_cap = rng.uniform(0.0001, 30.0)
        reimbursements = [
            _reimbursement(
                uid,
                spend_usd=rng.uniform(0.01, 5_000.0),
                weight=rng.uniform(0.01, 4.0),
            )
            for uid in range(1, rng.randint(2, 35))
        ]
        policy = _policy(lab_cap=lab_cap)
        policy["reimbursement_allow_overpay_without_champions"] = bool(case % 2)

        allocation = allocate_research_lab_epoch(
            12,
            policy,
            reimbursements,
            [],
        )

        paid_total = sum(
            float(row["paid_alpha_percent"])
            for row in allocation["reimbursement_allocations"]
        )
        assert paid_total == pytest.approx(lab_cap, abs=0.000002), case
        assert allocation["unallocated_percent"] == pytest.approx(
            0.0,
            abs=0.000002,
        ), case


def test_no_champion_reimbursements_conserve_100_accelerated_epochs_without_burn():
    policy = _policy(lab_cap=30.0)
    policy["reimbursement_allow_overpay_without_champions"] = True
    for epoch in range(100, 200):
        miner_count = 1 + epoch % 20
        reimbursements = []
        hotkeys = ["burn-hotkey"]
        for uid in range(1, miner_count + 1):
            obligation = _reimbursement(
                uid,
                spend_usd=float((epoch % 13 + 1) * uid * 25),
                weight=1.0 + (uid % 3) * 0.25,
            )
            obligation["start_epoch"] = epoch
            reimbursements.append(obligation)
            hotkeys.append(str(obligation["miner_hotkey"]))

        allocation = allocate_research_lab_epoch(
            epoch,
            policy,
            reimbursements,
            [],
        )
        uid_weights, burn_share, breakdown = (
            research_lab_uid_weights_from_allocation(
                allocation,
                metagraph_hotkeys=hotkeys,
                reserved_share=0.30,
            )
        )

        assert allocation["reimbursement_alpha_percent"] == pytest.approx(
            30.0,
            abs=0.000002,
        ), epoch
        assert allocation["unallocated_percent"] == pytest.approx(
            0.0,
            abs=0.000002,
        ), epoch
        assert burn_share == pytest.approx(0.0, abs=0.000002), epoch
        assert sum(uid_weights.values()) + burn_share == pytest.approx(
            0.30,
            abs=0.000002,
        ), epoch
        assert breakdown["deregistered"] == pytest.approx(0.0), epoch


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
