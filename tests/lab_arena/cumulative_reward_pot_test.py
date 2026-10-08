"""Signed slot shares must fit the miner pot without changing old bases."""

from copy import deepcopy
from fractions import Fraction

import pytest

from lab_arena import contracts, rewards, signing, weight_state
from leadpoet_canonical import arena_weights, lab_arena_rewards as kernel


BURN, ALICE, BOB, CAROL = ["5" + letter * 47 for letter in "ABCD"]
HOTKEYS = [BURN, ALICE, BOB, CAROL]


def _win(hotkey, delta, day, name):
    return {
        "round_id": "arena-2026-10-%02d" % day,
        "submission_id": name,
        "miner_hotkey": hotkey,
        "baseline_submission_id": "baseline-%02d" % day,
        "baseline_score": 40,
        "winner_score": 40 + delta,
        "start_epoch": 100,
    }


def _basis(slots, *, policy=None, king=ALICE, factor=1_000_000):
    return rewards.reward_basis_document(
        round_id="arena-2026-10-08",
        published_at="2026-10-09T00:00:00Z",
        finalized_epoch=99,
        king_outcome="crowned",
        king_hotkey=king,
        champion_reward_factor_ppm=factor,
        slot_policy=deepcopy(policy or rewards.REWARD_SLOT_POLICY),
        reward_slots=slots,
    )


def _vector(basis, hotkeys=HOTKEYS):
    signer = signing.LocalSigner.generate()
    signed = signing.sign_document(signer, basis, hash_field="reward_basis_hash")
    state = weight_state.build_accepted_weight_state(
        signer, network="finney", genesis_hash="1" * 64, netuid=71,
        epoch=100, valid_from_block=1000, valid_until_block=1100,
        reward_basis=signed, burn_hotkey=BURN, issued_at=basis["published_at"],
    )
    arena_weights.verify_accepted_weight_state_signature(
        state, public_key_der=signer.public_key_der,
        expected_public_key_hash=signer.public_key_hash,
    )
    return arena_weights.derive_arena_weights(state, hotkeys)


def test_policy_centralizes_cumulative_30_percent_pot():
    assert rewards.reward_slot_policy_document() == {
        "assignment_mode": "all_qualifying",
        "pool_percent": 30,
        "decay": {"epochs_per_halving": 140, "max_halvings": 4},
        "tiers": [
            {"minimum_improvement": 10, "allocation_percent": 50},
            {"minimum_improvement": 5, "allocation_percent": 30},
            {"minimum_improvement": 1, "allocation_percent": 20},
        ],
    }


def test_one_plus_11_winner_takes_whole_pot_and_weight_bundle_keeps_burn():
    winner = _win(ALICE, 11, 1, "alice")
    basis = _basis([winner, winner, winner])
    assert basis["slot_policy"]["pool_percent"] == 30
    assert kernel.slot_allocations(basis, 100, HOTKEYS) == {ALICE: Fraction(3, 10)}
    vector = _vector(basis)
    assert vector["champion_share_ppb"] == 300_000_000
    assert vector["burned_residual_ppb"] == 700_000_000
    assert vector["sparse_uids"] == [0, 1]
    assert vector["sparse_weights_u16"] == [65535, 28086]


def test_later_plus_6_and_plus_2_winners_replace_only_qualified_slots():
    alice = _win(ALICE, 11, 1, "alice")
    bob = _win(BOB, 6, 2, "bob")
    carol = _win(CAROL, 2, 3, "carol")
    after_bob = _basis([alice, bob, bob], king=BOB)
    assert kernel.slot_allocations(after_bob, 100, HOTKEYS) == {
        ALICE: Fraction(15, 100), BOB: Fraction(15, 100),
    }
    after_carol = _basis([alice, bob, carol], king=CAROL)
    assert kernel.slot_allocations(after_carol, 100, HOTKEYS) == {
        ALICE: Fraction(15, 100), BOB: Fraction(9, 100),
        CAROL: Fraction(6, 100),
    }
    vector = _vector(after_carol)
    assert vector["champion_share_ppb"] == 300_000_000
    assert vector["burned_residual_ppb"] == 700_000_000
    assert vector["sparse_uids"] == [0, 1, 2, 3]


def test_empty_and_unregistered_slots_keep_residual_in_burn():
    alice = _win(ALICE, 11, 1, "alice")
    bob = _win(BOB, 6, 2, "bob")
    basis = _basis([alice, bob, None])
    assert kernel.slot_allocations(basis, 100, [BURN, ALICE]) == {
        ALICE: Fraction(15, 100),
    }
    vector = _vector(basis, [BURN, ALICE])
    assert vector["champion_share_ppb"] == 150_000_000
    assert vector["burned_residual_ppb"] == 850_000_000
    assert kernel.slot_allocations(_basis([None, None, None]), 100, HOTKEYS) == {}


def test_existing_funding_factor_applies_only_to_current_king_slots():
    alice = _win(ALICE, 11, 1, "alice")
    bob = _win(BOB, 6, 2, "bob")
    basis = _basis([alice, bob, bob], king=BOB, factor=500_000)
    assert kernel.slot_allocations(basis, 100, HOTKEYS) == {
        ALICE: Fraction(15, 100), BOB: Fraction(75, 1000),
    }
    vector = _vector(basis)
    assert vector["champion_share_ppb"] == 225_000_000
    assert vector["burned_residual_ppb"] == 775_000_000


def test_policy_pool_is_configurable_without_changing_historical_metadata():
    policy = deepcopy(rewards.REWARD_SLOT_POLICY)
    policy["pool_percent"] = 20
    alice = _win(ALICE, 11, 1, "alice")
    basis = _basis([alice, alice, alice], policy=policy)
    assert basis["reward_constants"]["pool_percent"] == 30
    assert kernel.slot_allocations(basis, 100, HOTKEYS) == {ALICE: Fraction(1, 5)}


@pytest.mark.parametrize("bad", [True, False, None, -1, 101, 30.0, "30"])
def test_invalid_signed_pool_percent_fails_closed(bad):
    policy = deepcopy(rewards.REWARD_SLOT_POLICY)
    policy["pool_percent"] = bad
    with pytest.raises(kernel.LabArenaRewardError, match="pool_percent"):
        kernel.validate_slot_policy(policy)


def test_legacy_signed_v2_without_pool_retains_exact_hash_and_arithmetic():
    policy = deepcopy(rewards.REWARD_SLOT_POLICY)
    policy.pop("pool_percent")
    policy.pop("decay")
    winner = _win(ALICE, 11, 1, "old")
    winner.pop("start_epoch")
    winner.update(round_id="arena-2026-10-01", baseline_submission_id="base",
                  baseline_score=40, winner_score=51)
    basis = rewards.reward_basis_document(
        round_id="arena-2026-10-01", published_at="2026-10-02T00:00:00Z",
        finalized_epoch=99, king_outcome="crowned", king_hotkey=ALICE,
        slot_policy=policy, reward_slots=[winner, winner, winner],
    )
    assert basis["reward_basis_hash"] == (
        "sha256:5bc40913ea6080253e2218b3169647b1c41581875f1120abd6f14a149b27ca7b"
    )
    assert "pool_percent" not in basis["slot_policy"]
    assert contracts.validate_reward_basis(basis) == basis
    assert kernel.slot_allocations(basis, 100, HOTKEYS) == {ALICE: Fraction(1)}
    assert _vector(basis)["champion_share_ppb"] == 1_000_000_000
