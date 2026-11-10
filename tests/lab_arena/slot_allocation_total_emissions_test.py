"""Explicit total-emissions tiers retain exact shares and legacy signed formats."""

from copy import deepcopy
from fractions import Fraction
import json

import pytest

from lab_arena import signing, weight_state
from leadpoet_canonical import arena_weights, lab_arena_rewards as kernel
from tests.lab_arena.slot_decay_kernel_test import (
    ALICE, BOB, BURN, CAROL, HOTKEYS, _basis, _win,
)


START = 100
POLICY = {
    "assignment_mode": "all_qualifying",
    "allocation_basis": "total_emissions",
    "pool_percent": 75,
    "decay": {"epochs_per_halving": 140, "max_halvings": 4},
    "tiers": [
        {"minimum_improvement": 15, "allocation_percent": 35},
        {"minimum_improvement": 10, "allocation_percent": 25},
        {"minimum_improvement": 5, "allocation_percent": 15},
    ],
}


def _state(basis, epoch, *, starts=None):
    signer = signing.LocalSigner.generate()
    signed = signing.sign_document(signer, basis, hash_field="reward_basis_hash")
    kwargs = {}
    if starts is not None:
        kwargs["weekly_slot_decay"] = {
            "epochs_per_halving": 140,
            "slots": [None if slot is None else {
                "start_epoch": start,
                "source_normalized_sha256": "sha256:" + str(index + 1) * 64,
            } for index, (slot, start) in enumerate(zip(basis["reward_slots"], starts))],
        }
    state = weight_state.build_accepted_weight_state(
        signer, network="finney", genesis_hash="1" * 64, netuid=71,
        epoch=epoch, valid_from_block=1000, valid_until_block=1100,
        reward_basis=signed, burn_hotkey=BURN, issued_at=basis["published_at"],
        slot_eligibility="until_replaced", **kwargs,
    )
    assert arena_weights.verify_accepted_weight_state_signature(
        state, public_key_der=signer.public_key_der,
        expected_public_key_hash=signer.public_key_hash,
    ) == state["state_hash"]
    return state, signer


@pytest.mark.parametrize("delta,occupied,share", [
    (4.999, [False, False, False], Fraction(0)),
    (5, [False, False, True], Fraction(15, 100)),
    (9.999, [False, False, True], Fraction(15, 100)),
    (10, [False, True, True], Fraction(40, 100)),
    (14.999, [False, True, True], Fraction(40, 100)),
    (15, [True, True, True], Fraction(75, 100)),
])
def test_exact_prospective_threshold_boundaries(delta, occupied, share):
    winner = _win(delta=delta, start=START)
    slots = [deepcopy(winner) if present else None for present in occupied]
    basis = _basis(slots, policy=POLICY)
    assert kernel.slot_allocations(basis, START, HOTKEYS) == ({ALICE: share} if share else {})
    for index, present in enumerate(occupied):
        if not present:
            invalid = deepcopy(slots)
            invalid[index] = winner
            with pytest.raises(kernel.LabArenaRewardError, match="threshold"):
                kernel.validate_reward_slots(invalid, POLICY, effective_reward_epoch=START)
    state, _ = _state(basis, START)
    vector = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert vector["champion_share_ppb"] == int(share * 1_000_000_000)
    assert vector["burned_residual_ppb"] == int((1 - share) * 1_000_000_000)


def test_separate_owners_receive_exact_total_shares_and_quarter_burn():
    basis = _basis([_win(ALICE, delta=15), _win(BOB, delta=10), _win(CAROL, delta=5)], policy=POLICY)
    assert kernel.slot_allocations(basis, START, HOTKEYS) == {
        ALICE: Fraction(35, 100), BOB: Fraction(25, 100), CAROL: Fraction(15, 100),
    }
    state, _ = _state(basis, START)
    vector = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert vector["champion_share_ppb"] == 750_000_000
    assert vector["burned_residual_ppb"] == 250_000_000
    assert vector["sparse_weights_u16"] == [46811, 65535, 46811, 28086]


@pytest.mark.parametrize("age,divisor", [(139, 1), (140, 2), (280, 4), (700, 32)])
def test_weekly_clock_continues_exact_halving(age, divisor):
    winner = _win(delta=15)
    basis = _basis([winner] * 3, policy=POLICY)
    state, _ = _state(basis, START + age, starts=[START] * 3)
    expected = Fraction(3, 4 * divisor)
    vector = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert vector["champion_share_ppb"] == int(expected * 1_000_000_000)
    assert vector["burned_residual_ppb"] == int((1 - expected) * 1_000_000_000)


def test_independent_clocks_funding_missing_owner_and_empty_slot_go_to_burn():
    basis = _basis([
        _win(ALICE, delta=15, start=100), _win(BOB, delta=10, start=240), None,
    ], epoch=380, policy=POLICY, king=BOB, factor=500_000)
    state, _ = _state(basis, 380, starts=[100, 240, None])
    # High has aged twice, mid once; funding halves only the current king.
    vector = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert vector["champion_share_ppb"] == 150_000_000
    assert vector["burned_residual_ppb"] == 850_000_000
    absent = arena_weights.derive_arena_weights(state, [BURN, ALICE])
    assert absent["champion_share_ppb"] == 87_500_000
    assert absent["burned_residual_ppb"] == 912_500_000
    empty = _basis([None] * 3, policy=POLICY)
    empty_state, _ = _state(empty, START, starts=[None] * 3)
    assert arena_weights.derive_arena_weights(empty_state, HOTKEYS)["burned_residual_ppb"] == 1_000_000_000


@pytest.mark.parametrize("change", [
    {"allocation_basis": None}, {"allocation_basis": "within_pool"},
    {"allocation_basis": True}, {"allocation_basis": "total_emission"},
    {"pool_percent": None}, {"pool_percent": True}, {"pool_percent": 75.0},
    {"pool_percent": 76}, {"unknown": "total_emissions"},
])
def test_total_emissions_rejects_invalid_shape_or_pool(change):
    policy = {**deepcopy(POLICY), **change}
    with pytest.raises(kernel.LabArenaRewardError):
        kernel.validate_slot_policy(policy)


def test_explicit_basis_requires_pool_and_integer_tiers():
    policy = deepcopy(POLICY)
    policy.pop("pool_percent")
    with pytest.raises(kernel.LabArenaRewardError):
        kernel.validate_slot_policy(policy)
    for bad in [35.0, True, "35", 0]:
        policy = deepcopy(POLICY)
        policy["tiers"][0]["allocation_percent"] = bad
        with pytest.raises(kernel.LabArenaRewardError):
            kernel.validate_slot_policy(policy)


@pytest.mark.parametrize("pool,expected", [(None, Fraction(1)), (30, Fraction(3, 10)), (50, Fraction(1, 2))])
def test_legacy_signed_policy_basis_and_state_keep_exact_bytes(pool, expected):
    legacy = deepcopy(POLICY)
    legacy.pop("allocation_basis")
    legacy.pop("pool_percent")
    if pool is not None:
        legacy["pool_percent"] = pool
    for tier, allocation in zip(legacy["tiers"], [40, 30, 30]):
        tier["allocation_percent"] = allocation
    winner = _win(delta=15)
    basis = _basis([winner] * 3, policy=legacy)
    state, signer = _state(basis, START)
    original = kernel.canonical_json(state)
    loaded = arena_weights.validate_accepted_weight_state(json.loads(original))
    assert kernel.canonical_json(loaded) == original
    assert "allocation_basis" not in loaded["reward_basis"]["slot_policy"]
    assert kernel.slot_allocations(loaded["reward_basis"], START, HOTKEYS) == {ALICE: expected}
    assert arena_weights.verify_accepted_weight_state_signature(
        loaded, public_key_der=signer.public_key_der,
        expected_public_key_hash=signer.public_key_hash,
    ) == state["state_hash"]
    invalid = deepcopy(legacy)
    invalid["tiers"][0]["allocation_percent"] = 35
    with pytest.raises(kernel.LabArenaRewardError, match="sum to 100"):
        kernel.validate_slot_policy(invalid)
