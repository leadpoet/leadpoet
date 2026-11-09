"""Signed model starts continue decay every existing 140-epoch reward week."""

from copy import deepcopy
from fractions import Fraction
import hashlib
import json

import pytest

from lab_arena import signing, weight_state
from leadpoet_canonical import arena_weights, lab_arena_rewards as kernel
from tests.lab_arena.slot_decay_kernel_test import (
    ALICE, BOB, BURN, CAROL, CURRENT_ROUND, HOTKEYS, _basis, _win,
)


WEEK = 140
START = 100
BLOCK = 10000
POLICY = {
    "assignment_mode": "all_qualifying", "pool_percent": 50,
    "tiers": [
        {"minimum_improvement": 20, "allocation_percent": 40},
        {"minimum_improvement": 15, "allocation_percent": 30},
        {"minimum_improvement": 8, "allocation_percent": 30},
    ],
    "decay": {"epochs_per_halving": 140, "max_halvings": 4},
}


def _clock(achievement, *, start_epoch=START):
    return {
        "start_epoch": start_epoch,
        "source_normalized_sha256": "sha256:" + hashlib.sha256(
            achievement["submission_id"].encode("utf-8")).hexdigest(),
    }


def _fixture(*, age=0, slots=None, starts=None, factor=1_000_000, king=ALICE):
    if slots is None:
        winner = _win(delta=20)
        slots = [deepcopy(winner) for _ in range(3)]
    basis = _basis(slots, epoch=START + age, policy=deepcopy(POLICY), king=king, factor=factor)
    clock = {
        "epochs_per_halving": WEEK,
        "slots": [None if slot is None else _clock(
            slot, start_epoch=START if starts is None else starts[index],
        ) for index, slot in enumerate(slots)],
    }
    return basis, clock


def _signed_state(basis, clock=None, *, epoch=None, valid_from=BLOCK):
    signer = signing.LocalSigner.generate()
    signed = signing.sign_document(signer, basis, hash_field="reward_basis_hash")
    state = weight_state.build_accepted_weight_state(
        signer, network="finney", genesis_hash="1" * 64, netuid=71,
        epoch=basis["effective_reward_epoch"] if epoch is None else epoch, valid_from_block=valid_from, valid_until_block=valid_from + 100,
        reward_basis=signed, burn_hotkey=BURN, issued_at=basis["published_at"],
        weekly_slot_decay=clock,
    )
    assert arena_weights.verify_accepted_weight_state_signature(
        state, public_key_der=signer.public_key_der,
        expected_public_key_hash=signer.public_key_hash,
    ) == state["state_hash"]
    return state, signer


@pytest.mark.parametrize("age", [0] + [
    weeks * WEEK + offset
    for weeks in [1, 2, 4, 5, 12, 100, 1100] for offset in [-1, 0, 1]
])
def test_140_epoch_week_boundaries_continue_beyond_legacy_floor(age):
    basis, clock = _fixture(age=age)
    before = deepcopy(basis)
    expected = Fraction(1, 2 ** (age // WEEK + 1))
    assert kernel.slot_allocations(
        basis, basis["effective_reward_epoch"], HOTKEYS, until_replaced=True, weekly_slot_decay=clock,
    ) == {ALICE: expected}
    assert basis == before
    state, _ = _signed_state(basis, clock)
    vector = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert vector["champion_share_ppb"] == int(expected * 1_000_000_000)
    assert vector["burned_residual_ppb"] == int((1 - expected) * 1_000_000_000)
    if age >= 100 * WEEK:
        assert vector["sparse_uids"] == [0]
        assert vector["sparse_weights_u16"] == [65535]


def test_same_hotkey_models_keep_independent_clocks_and_fresh_models_reset_only_their_slots():
    slots = [_win(delta=20, start=100, name="high"),
             _win(delta=15, start=240, name="mid"),
             _win(delta=8, start=380, name="low")]
    basis, clock = _fixture(age=6 * WEEK, slots=slots, starts=[START, START + WEEK, START + 2 * WEEK])
    expected = Fraction(1, 5 * 64) + Fraction(3, 20 * 32) + Fraction(3, 20 * 16)
    assert kernel.slot_allocations(
        basis, basis["effective_reward_epoch"], HOTKEYS, until_replaced=True, weekly_slot_decay=clock,
    ) == {ALICE: expected}
    # A new model can retain the same miner hotkey and obtain a fresh age.
    replacement = _win(delta=8, start=None, name="new-low", round_id=CURRENT_ROUND)
    slots[2] = replacement
    replaced, new_clock = _fixture(age=6 * WEEK, slots=slots, starts=[START, START + WEEK, START + 6 * WEEK])
    new_clock["slots"][2] = _clock(replacement, start_epoch=START + 6 * WEEK)
    new_expected = Fraction(1, 5 * 64) + Fraction(3, 20 * 32) + Fraction(3, 20)
    state, _ = _signed_state(replaced, new_clock)
    assert arena_weights.derive_arena_weights(state, HOTKEYS)["champion_share_ppb"] == int(new_expected * 1_000_000_000)


def test_funding_missing_uid_and_empty_slots_leave_residual_for_burn():
    slots = [_win(ALICE, delta=20), _win(BOB, delta=15), _win(CAROL, delta=8)]
    basis, clock = _fixture(age=5 * WEEK, slots=slots, factor=500_000, king=BOB)
    registered = [BURN, ALICE, BOB]
    expected = {ALICE: Fraction(1, 160), BOB: Fraction(3, 1280)}
    assert kernel.slot_allocations(
        basis, basis["effective_reward_epoch"], registered, until_replaced=True, weekly_slot_decay=clock,
    ) == expected
    state, _ = _signed_state(basis, clock)
    vector = arena_weights.derive_arena_weights(state, registered)
    total = sum(expected.values(), Fraction(0))
    assert vector["champion_share_ppb"] == int(total * 1_000_000_000)
    assert vector["burned_residual_ppb"] == int((1 - total) * 1_000_000_000)
    reordered = arena_weights.derive_arena_weights(state, list(reversed(registered)))
    assert reordered["champion_share_ppb"] == vector["champion_share_ppb"]
    assert reordered["sparse_uids"] == [0, 1, 2]
    empty_basis, empty_clock = _fixture(slots=[None] * 3)
    empty, _ = _signed_state(empty_basis, empty_clock)
    assert arena_weights.derive_arena_weights(empty, registered)["sparse_weights_u16"] == [65535]


def test_legacy_policy_and_state_bytes_are_unchanged_when_clock_is_absent():
    basis, _ = _fixture(age=5 * WEEK)
    before = kernel.canonical_json(basis)
    state, _ = _signed_state(basis)
    assert "weekly_slot_decay" not in state
    assert arena_weights.accepted_weight_state_body(state) == {
        key: value for key, value in state.items() if key not in ("state_hash", "signature")
    }
    assert arena_weights.derive_arena_weights(state, HOTKEYS)["champion_share_ppb"] == 31_250_000
    assert kernel.canonical_json(kernel.validate_reward_basis(json.loads(before))) == before
    policy = deepcopy(POLICY)
    policy["decay"]["max_halvings"] = None
    with pytest.raises(kernel.LabArenaRewardError, match="max_halvings"):
        kernel.validate_slot_policy(policy)


def test_clock_is_copied_and_covered_by_state_hash_and_signature():
    basis, clock = _fixture(age=WEEK)
    state, signer = _signed_state(basis, clock)
    clock["slots"][0]["start_epoch"] -= 1
    assert state["weekly_slot_decay"]["slots"][0]["start_epoch"] == START
    mutations = []
    removed = deepcopy(state)
    removed.pop("weekly_slot_decay")
    mutations.append(removed)
    changed = deepcopy(state)
    changed["weekly_slot_decay"]["slots"][0]["start_epoch"] -= 1
    mutations.append(changed)
    forged_source = deepcopy(state)
    forged_source["weekly_slot_decay"]["slots"][0]["source_normalized_sha256"] = "sha256:" + "f" * 64
    mutations.append(forged_source)
    for mutated in mutations:
        with pytest.raises(arena_weights.ArenaWeightError, match="state_hash"):
            arena_weights.verify_accepted_weight_state_signature(
                mutated, public_key_der=signer.public_key_der,
                expected_public_key_hash=signer.public_key_hash,
            )
        mutated["state_hash"] = kernel.sha256_json(arena_weights.accepted_weight_state_body(mutated))
        with pytest.raises(arena_weights.ArenaWeightError, match="signature"):
            arena_weights.verify_accepted_weight_state_signature(
                mutated, public_key_der=signer.public_key_der,
                expected_public_key_hash=signer.public_key_hash,
            )


@pytest.mark.parametrize("path,value", [
    ((), None), ((), {}), ((), []), ((), True),
    (("epochs_per_halving",), 604800), (("epochs_per_halving",), None),
    (("epochs_per_halving",), True), (("epochs_per_halving",), 140.0),
    (("epochs_per_halving",), 0), (("epochs_per_halving",), "140"),
    (("as_of_time",), 1_700_000_000), (("seconds_per_halving",), 604800),
    (("slots",), None), (("slots",), []), (("slots",), [None] * 3),
    (("slots", 0), None), (("slots", 0), {}),
    (("slots", 0, "start_time"), 1_700_000_000),
    (("slots", 0, "start_block"), BLOCK),
    (("slots", 0, "start_epoch"), True), (("slots", 0, "start_epoch"), -1),
    (("slots", 0, "start_epoch"), None), (("slots", 0, "start_epoch"), 100.0),
    (("slots", 0, "start_epoch"), 101),
    (("slots", 0, "source_normalized_sha256"), None),
    (("slots", 0, "source_normalized_sha256"), "sha256:" + "z" * 64),
    (("slots", 0, "source_normalized_sha256"), "sha256:" + "a" * 63),
    (("extra",), 0), (("slots", 0, "extra"), 0),
])
def test_invalid_clock_and_source_bindings_fail_before_signing(path, value):
    basis, clock = _fixture()
    if not path:
        clock = value
        if clock is None:
            # An explicit null field is invalid even though omission is valid.
            state, _ = _signed_state(basis)
            state["weekly_slot_decay"] = None
            with pytest.raises(arena_weights.ArenaWeightError, match="weekly_slot_decay"):
                arena_weights.validate_accepted_weight_state(state)
            return
    else:
        target = clock
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
    with pytest.raises(kernel.LabArenaRewardError):
        _signed_state(basis, clock)


def test_occupied_clock_cannot_be_attached_to_empty_slot():
    basis, clock = _fixture(slots=[None] * 3)
    clock["slots"][0] = _clock(_win(delta=20))
    with pytest.raises(kernel.LabArenaRewardError, match="empty slot"):
        _signed_state(basis, clock)


def test_current_source_null_epoch_resolves_to_basis_and_allows_older_identical_model_start():
    winner = _win(delta=20, start=None, round_id=CURRENT_ROUND)
    basis, clock = _fixture(age=2 * WEEK, slots=[winner] * 3)
    clock["slots"] = [_clock(winner, start_epoch=basis["effective_reward_epoch"]) for _ in range(3)]
    state, _ = _signed_state(basis, clock)
    assert arena_weights.derive_arena_weights(state, HOTKEYS)["champion_share_ppb"] == 500_000_000
    for item in clock["slots"]:
        item["start_epoch"] = START
    state, _ = _signed_state(basis, clock)
    assert arena_weights.derive_arena_weights(state, HOTKEYS)["champion_share_ppb"] == 125_000_000
    clock["slots"][0]["start_epoch"] = basis["effective_reward_epoch"] + 1
    with pytest.raises(kernel.LabArenaRewardError, match="source epoch"):
        _signed_state(basis, clock)


def test_clock_can_preserve_older_identical_model_age_but_cannot_postdate_source():
    basis, clock = _fixture()
    clock["slots"][0]["start_epoch"] = 99
    _signed_state(basis, clock)
    clock["slots"][0]["start_epoch"] = 101
    with pytest.raises(kernel.LabArenaRewardError, match="source epoch"):
        _signed_state(basis, clock)


def test_exact_arithmetic_bound_keeps_positive_fraction_before_float32_conversion():
    maximum = kernel.MAX_EXACT_SLOT_DECAY_HALVINGS
    basis, clock = _fixture(age=WEEK * maximum)
    assert kernel.slot_allocations(
        basis, basis["effective_reward_epoch"], HOTKEYS, until_replaced=True, weekly_slot_decay=clock,
    ) == {ALICE: Fraction(1, 2 ** (maximum + 1))}
    vector = arena_weights.derive_arena_weights(_signed_state(basis, clock)[0], HOTKEYS)
    assert vector["sparse_uids"] == [0]
    assert vector["sparse_weights_u16"] == [65535]
    assert vector["champion_share_ppb"] == 0


@pytest.mark.parametrize("halvings", [4097, 10 ** 100])
def test_huge_epoch_age_fails_before_unbounded_exponentiation(halvings):
    basis, clock = _fixture(age=WEEK * halvings)
    with pytest.raises(kernel.LabArenaRewardError, match="exact-arithmetic bounds"):
        kernel.slot_allocations(basis, basis["effective_reward_epoch"], HOTKEYS, until_replaced=True, weekly_slot_decay=clock)


def test_v1_basis_cannot_accept_weekly_slot_clocks():
    basis, clock = _fixture()
    body = {key: value for key, value in basis.items()
            if key not in ("reward_basis_hash", "slot_policy", "reward_slots")}
    body["schema_version"] = kernel.REWARD_BASIS_SCHEMA_VERSION
    body["reward_basis_hash"] = kernel.sha256_json(body)
    with pytest.raises(kernel.LabArenaRewardError, match="v2 reward basis"):
        _signed_state(body, clock)
