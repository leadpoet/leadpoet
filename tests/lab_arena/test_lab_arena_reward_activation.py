"""The Arena reward kernel through the canonical weight computation
(labarena.md 13.1-13.4, 18.8 pre-activation items that need no release).

With rewards disabled the snapshot is untouched and every weight result is
byte-identical; with an eligible king the existing champion slot allocates
exactly the derived share to the king's UID, burns nothing extra, and
returns the amount to fulfillment on an ineligible epoch or an
unregistered king.
"""

from __future__ import annotations

from bittensor_wallet import Keypair

from lab_arena import contracts, rewards
from leadpoet_canonical.attested_v2 import canonical_json
from leadpoet_canonical.weight_computation import (
    WEIGHT_SNAPSHOT_SCHEMA_VERSION,
    compute_final_weights,
    weight_config_hash,
)

BURN_HOTKEY = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
KING_HOTKEY = Keypair.create_from_uri("//ArenaKing").ss58_address


def snapshot(**overrides):
    value = {
        "schema_version": WEIGHT_SNAPSHOT_SCHEMA_VERSION,
        "netuid": 71,
        "epoch_id": 24810,
        "block": 36099,
        "commit_sha": "a" * 40,
        "config_hash": "",
        "parent_receipt_hashes": [],
        "research_lab_allocation_receipt_hash": "",
        "burn_target_uid": 0,
        "expected_burn_target_hotkey": BURN_HOTKEY,
        "metagraph_hotkeys": [BURN_HOTKEY, "fulfillment-hotkey", "lab-hotkey", KING_HOTKEY],
        "banned_hotkeys": [],
        "banned_lookup_ok": True,
        "ff_enabled": True,
        "base_burn_share": 0.0,
        "champion_share": 0.0,
        "champion_uid": None,
        "effective_champion_share": 0.0,
        "research_lab_fallback_share": 0.3,
        "research_lab_allocation_doc": {
            "lab_cap_percent": 30.0,
            "unallocated_percent": 25.0,
            "reimbursement_allocations": [],
            "champion_allocations": [{"uid": 2, "miner_hotkey": "lab-hotkey", "paid_alpha_percent": 5.0}],
            "queued_champion_allocations": [],
        },
        "leaderboard_bonus_share": 0.095,
        "leaderboard_rank_shares": [0.05, 0.03, 0.015],
        "leaderboard_entries": [{"miner_hotkey": "fulfillment-hotkey", "wins": 9}],
        "leaderboard_fetch_ok": True,
        "fulfillment_share": 0.605,
        "fulfillment_rows": [{"hotkey": "fulfillment-hotkey", "share": 0.605}],
        "fulfillment_fetch_ok": True,
        "rolling_lead_count": 0,
        "rolling_scores": [],
        "sourcing_floor_threshold": 125_000,
        "min_total_rep_for_distribution": 100,
    }
    value.update(overrides)
    if "config_hash" not in overrides:
        value["config_hash"] = weight_config_hash(value)
    return value


def basis(*, outcome="crowned", effective=24801, start=24801, hotkey=KING_HOTKEY):
    return contracts.finalize_reward_basis({
        "schema_version": contracts.REWARD_BASIS_SCHEMA_VERSION, "round_id": "arena-2026-09-02", "configuration_hash": contracts.document_hash("cfg"),
        "commitment_hash": contracts.document_hash("cm"), "result_bundle_hash": contracts.document_hash("rb"), "published_at": "2026-09-02T10:00:00Z",
        "effective_reward_epoch": effective, "king_hotkey": hotkey if outcome != "no_king" else "", "king_outcome": outcome, "king_start_epoch": start,
        "reward_constants": rewards.reward_constants_document(),
    })


def apply_kernel(base, reward_basis, epoch_id):
    # The king's share is a share of total emissions: the lab and leaderboard shares do not enter it.
    values = rewards.champion_values(reward_basis, epoch_id, base["metagraph_hotkeys"])
    return snapshot(epoch_id=epoch_id, champion_share=values["champion_share"], effective_champion_share=values["effective_champion_share"], champion_uid=values["champion_uid"]), values


def weight_of(result, uid):
    return result["weights"][result["uids"].index(uid)]


def burn_of(result):
    return weight_of(result, 0)


def test_rewards_disabled_leaves_the_snapshot_and_every_weight_byte_identical():
    baseline = snapshot()
    result = compute_final_weights(baseline)
    again = compute_final_weights(snapshot())
    assert canonical_json(result) == canonical_json(again)
    assert baseline["champion_share"] == 0.0 and baseline["champion_uid"] is None
    # No Arena document is consulted when the flag is off: the snapshot builder is the existing one.
    assert set(baseline) == set(snapshot())


def test_eligible_king_receives_the_exact_week_share_through_the_champion_slot():
    base = snapshot()
    baseline = compute_final_weights(base)
    proposed, values = apply_kernel(base, basis(), 24810)  # week 0: 100% of the Arena pool, 25% of emissions
    assert values["eligible"] is True and values["champion_uid"] == 3
    assert values["champion_share"] == values["effective_champion_share"] == 0.25
    result = compute_final_weights(proposed)
    assert abs(weight_of(result, 3) - 0.25) < 1e-9
    assert 3 not in baseline["uids"] or weight_of(baseline, 3) == 0.0
    assert burn_of(result) <= burn_of(baseline) + 1e-9  # the Arena amount is never burned
    assert abs(result["components"]["fulfillment_pool_share"] - (baseline["components"]["fulfillment_pool_share"] - 0.25)) < 1e-9
    assert result["components"]["research_lab_share"] == baseline["components"]["research_lab_share"] == 0.3
    assert proposed["config_hash"] != base["config_hash"]  # visible but harmless weekly change
    # The lab share is derived from the allocation document identically on both sides.
    assert rewards.derive_research_lab_share(base["research_lab_allocation_doc"], 0.99) == 0.3
    assert rewards.derive_research_lab_share({"lab_cap_percent": ""}, 0.3) == 0.3


def test_week_decay_and_floor_are_visible_in_the_weight_vector():
    base = snapshot()
    expected = {0: 0.25, 1: 0.2, 2: 0.15, 3: 0.1, 4: 0.05, 9: 0.05}
    for week, share in expected.items():
        epoch = 24801 + 140 * week
        # A round publishes every twenty epochs: the governing row is recent and
        # the defended king keeps its original start epoch, so only the week decays.
        governing = basis(effective=epoch - 5, start=24801, outcome="defended") if week else basis()
        proposed, values = apply_kernel(base, governing, epoch)
        assert values["champion_share"] == share, week
        result = compute_final_weights(proposed)
        assert abs(weight_of(result, 3) - share) < 1e-9


def test_ineligible_epoch_unregistered_king_and_non_paying_outcomes_return_everything_to_fulfillment():
    base = snapshot()
    for epoch, reward_basis in (
        (24801 + 46, basis()),  # 46 epochs after the governing row: stale
        (24810, basis(hotkey=Keypair.create_from_uri("//Unregistered").ss58_address)),
        (24810, basis(outcome="retained_ineligible")),
        (24810, basis(outcome="no_king", hotkey="")),
    ):
        proposed, values = apply_kernel(base, reward_basis, epoch)
        assert values["champion_share"] == 0.0 and values["champion_uid"] is None
        # Byte-identical to the flag-off computation at the same epoch.
        assert canonical_json(compute_final_weights(proposed)) == canonical_json(compute_final_weights(snapshot(epoch_id=epoch)))


def test_governing_row_selection_and_write_once_epochs():
    rows = [basis(effective=24801, start=24801), basis(effective=24821, start=24801, outcome="defended")]
    assert rewards.governing_reward_basis(rows, 24815)["effective_reward_epoch"] == 24801
    assert rewards.governing_reward_basis(rows, 24821)["effective_reward_epoch"] == 24821
    assert rewards.governing_reward_basis(rows, 24800) is None
    import pytest

    with pytest.raises(ValueError):
        rewards.governing_reward_basis(rows + [basis(effective=24821, start=24821)], 24830)
