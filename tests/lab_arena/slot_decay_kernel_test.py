"""Per-achievement age remains signed across daily publication and UID changes."""

from copy import deepcopy
from fractions import Fraction
import json

import pytest

from lab_arena import contracts, rewards, signing, weight_state
from leadpoet_canonical import arena_weights, lab_arena_rewards as kernel


BURN, ALICE, BOB, CAROL = ["5" + letter * 47 for letter in "ABCD"]
HOTKEYS = [BURN, ALICE, BOB, CAROL]
CURRENT_ROUND = "arena-2026-10-08"


def _win(hotkey=ALICE, delta=11, start=100, name="old", round_id="arena-2026-09-01"):
    return {
        "round_id": round_id, "submission_id": name, "miner_hotkey": hotkey,
        "baseline_submission_id": "baseline-" + name,
        "baseline_score": 40, "winner_score": 40 + delta, "start_epoch": start,
    }


def _basis(slots, *, epoch=100, policy=None, king=ALICE, factor=1_000_000,
           round_id=CURRENT_ROUND):
    return rewards.reward_basis_document(
        round_id=round_id, published_at="2026-10-09T00:00:00Z",
        finalized_epoch=epoch - 1, king_hotkey=king, king_outcome="crowned",
        champion_reward_factor_ppm=factor,
        slot_policy=deepcopy(policy if policy is not None else rewards.REWARD_SLOT_POLICY),
        reward_slots=slots,
    )


def _body(basis):
    return {key: value for key, value in basis.items() if key not in ("reward_basis_hash", "signature")}


def _state(basis, epoch):
    signer = signing.LocalSigner.generate()
    signed = signing.sign_document(signer, basis, hash_field="reward_basis_hash")
    state = weight_state.build_accepted_weight_state(
        signer, network="finney", genesis_hash="1" * 64, netuid=71,
        epoch=epoch, valid_from_block=1000, valid_until_block=1100,
        reward_basis=signed, burn_hotkey=BURN, issued_at=basis["published_at"],
    )
    assert arena_weights.verify_accepted_weight_state_signature(
        state, public_key_der=signer.public_key_der,
        expected_public_key_hash=signer.public_key_hash,
    ) == state["state_hash"]
    return state, signer


@pytest.mark.parametrize("age,divisor", [
    (0, 1), (139, 1), (140, 2), (279, 2), (280, 4), (419, 4),
    (420, 8), (559, 8), (560, 16), (699, 16), (700, 16), (5000, 16),
])
def test_complete_weeks_halve_each_slot_then_keep_one_sixteenth(age, divisor):
    winner = _win()
    epoch = 100 + age
    basis = _basis([winner, winner, winner], epoch=epoch)
    assert kernel.slot_allocations(basis, epoch, HOTKEYS) == {
        ALICE: Fraction(3, 10 * divisor),
    }
    state, _ = _state(basis, epoch)
    vector = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert vector["champion_share_ppb"] == 300_000_000 // divisor
    assert vector["burned_residual_ppb"] == 1_000_000_000 - 300_000_000 // divisor


def test_mixed_ages_are_independent_even_for_one_hotkey():
    slots = [_win(start=100, name="high"), _win(delta=6, start=240, name="mid"),
             _win(delta=2, start=380, name="low")]
    basis = _basis(slots, epoch=660)
    assert kernel.slot_allocations(basis, 660, HOTKEYS) == {
        ALICE: Fraction(15, 100 * 16) + Fraction(9, 100 * 8) + Fraction(6, 100 * 4),
    }
    assert [slot["start_epoch"] for slot in basis["reward_slots"]] == [100, 240, 380]


def test_mixed_owners_funding_and_missing_uid_leave_decay_residual_for_burn():
    basis = _basis([_win(ALICE, start=100), _win(BOB, delta=6, start=240),
                    _win(CAROL, delta=2, start=380)], epoch=660, king=BOB, factor=500_000)
    assert kernel.slot_allocations(basis, 660, HOTKEYS) == {
        ALICE: Fraction(15, 1600), BOB: Fraction(9, 1600), CAROL: Fraction(6, 400),
    }
    present = [BURN, ALICE, BOB]
    assert kernel.slot_allocations(basis, 660, present) == {
        ALICE: Fraction(15, 1600), BOB: Fraction(9, 1600),
    }
    state, _ = _state(basis, 660)
    vector = arena_weights.derive_arena_weights(state, present)
    assert vector["champion_share_ppb"] == 15_000_000
    assert vector["burned_residual_ppb"] == 985_000_000
    assert vector["sparse_uids"] == [0, 1, 2]


def test_holder_at_burn_uid_merges_with_decay_residual():
    basis = _basis([_win(BURN), None, None], epoch=660)
    state, _ = _state(basis, 660)
    vector = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert vector["champion_share_ppb"] == 9_375_000
    assert vector["burned_residual_ppb"] == 990_625_000
    assert vector["sparse_uids"] == [0]
    assert vector["sparse_weights_u16"] == [65535]


def test_daily_bases_preserve_old_slot_age_and_reset_only_replaced_slots():
    winner = _win()
    original = _basis([winner, winner, winner], epoch=100)
    later = _basis(deepcopy(original["reward_slots"]), epoch=240)
    assert later["reward_slots"] == original["reward_slots"]
    assert kernel.slot_allocations(later, 240, HOTKEYS) == {ALICE: Fraction(3, 20)}
    replacement = _win(BOB, delta=6, start=None, name="fresh", round_id=CURRENT_ROUND)
    replaced = _basis([winner, replacement, replacement], epoch=240, king=BOB)
    assert replaced["reward_slots"][1]["start_epoch"] is None
    assert kernel.slot_allocations(replaced, 240, HOTKEYS) == {
        ALICE: Fraction(3, 40), BOB: Fraction(3, 20),
    }
    # After activation, the next day's snapshot carries the source round's
    # original effective epoch, rather than the new daily basis's epoch.
    historical = deepcopy(replaced["reward_slots"])
    for slot in historical[1:]:
        slot["start_epoch"] = 240
    next_day = _basis(historical, epoch=380, king=BOB, round_id="arena-2026-10-09")
    assert kernel.slot_allocations(next_day, 380, HOTKEYS) == {
        ALICE: Fraction(3, 80), BOB: Fraction(3, 40),
    }


def test_same_hotkey_replacement_resets_only_new_achievements():
    old = _win()
    fresh = _win(delta=6, start=None, name="fresh", round_id=CURRENT_ROUND)
    basis = _basis([old, fresh, fresh], epoch=660)
    assert kernel.slot_allocations(basis, 660, HOTKEYS) == {
        ALICE: Fraction(15, 1600) + Fraction(15, 100),
    }


def test_own_round_null_resolves_to_effective_epoch_without_mutating_signed_bytes():
    current = _win(start=None, round_id=CURRENT_ROUND)
    basis = _basis([current, current, current], epoch=500)
    before = deepcopy(basis)
    assert kernel.slot_allocations(basis, 500, HOTKEYS) == {ALICE: Fraction(3, 10)}
    assert kernel.validate_reward_basis(json.loads(json.dumps(basis))) == before
    assert basis == before
    assert kernel.slot_allocations(basis, 545, HOTKEYS) == {ALICE: Fraction(3, 10)}
    assert kernel.slot_allocations(basis, 546, HOTKEYS) == {}
    state, signer = _state(basis, 500)
    assert kernel.verify_reward_basis_signature(
        state["reward_basis"], public_key_der=signer.public_key_der,
        expected_public_key_hash=signer.public_key_hash,
    ) == basis["reward_basis_hash"]
    assert arena_weights.derive_arena_weights(state, HOTKEYS)["champion_share_ppb"] == 300_000_000


def test_decay_uses_signed_week_length_and_floor():
    policy = deepcopy(rewards.REWARD_SLOT_POLICY)
    policy["decay"] = {"epochs_per_halving": 10, "max_halvings": 1}
    winner = _win()
    basis = _basis([winner, winner, winner], epoch=120, policy=policy)
    assert kernel.slot_allocations(basis, 120, HOTKEYS) == {ALICE: Fraction(3, 20)}


@pytest.mark.parametrize("bad", [None, [], {}, True, "weekly",
    {"epochs_per_halving": 140}, {"max_halvings": 4},
    {"epochs_per_halving": 140, "max_halvings": 4, "floor": 16},
])
def test_bad_decay_shapes_fail_closed(bad):
    policy = deepcopy(rewards.REWARD_SLOT_POLICY)
    policy["decay"] = bad
    with pytest.raises(kernel.LabArenaRewardError, match="decay"):
        kernel.validate_slot_policy(policy)


@pytest.mark.parametrize("field,bad", [
    ("epochs_per_halving", True), ("epochs_per_halving", False),
    ("epochs_per_halving", None), ("epochs_per_halving", 0),
    ("epochs_per_halving", -1), ("epochs_per_halving", 140.0),
    ("epochs_per_halving", "140"), ("epochs_per_halving", 1_000_001),
    ("max_halvings", True), ("max_halvings", False), ("max_halvings", None),
    ("max_halvings", -1), ("max_halvings", 4.0), ("max_halvings", "4"),
    ("max_halvings", 13),
])
def test_bad_decay_types_and_bounds_fail_closed(field, bad):
    policy = deepcopy(rewards.REWARD_SLOT_POLICY)
    policy["decay"][field] = bad
    with pytest.raises(kernel.LabArenaRewardError, match=field):
        kernel.validate_slot_policy(policy)


@pytest.mark.parametrize("weeks,max_halvings", [(1, 0), (1_000_000, 12)])
def test_decay_validation_accepts_bounded_extremes(weeks, max_halvings):
    policy = deepcopy(rewards.REWARD_SLOT_POLICY)
    policy["decay"] = {"epochs_per_halving": weeks, "max_halvings": max_halvings}
    assert kernel.validate_slot_policy(policy) == policy


@pytest.mark.parametrize("bad", [None, True, False, -1, 101, 100.0, "100", [], {}])
def test_historical_start_epoch_must_be_nonnegative_integer_not_future(bad):
    with pytest.raises(kernel.LabArenaRewardError, match="start_epoch"):
        _basis([_win(start=bad), None, None])


@pytest.mark.parametrize("mutation", ["missing", "extra", "own_round_wrong_epoch"])
def test_start_epoch_shape_and_own_round_provenance_fail_closed(mutation):
    slot = _win()
    if mutation == "missing":
        slot.pop("start_epoch")
    elif mutation == "extra":
        slot["started_at"] = "2026-10-01"
    else:
        slot.update(round_id=CURRENT_ROUND, start_epoch=99)
    with pytest.raises(kernel.LabArenaRewardError):
        _basis([slot, None, None])


def test_valid_explicit_own_round_start_and_zero_historical_start():
    slots = [_win(start=100, round_id=CURRENT_ROUND),
             _win(delta=6, start=0), None]
    assert kernel.validate_reward_basis(_basis(slots))


@pytest.mark.parametrize("bad", [None, True, -1, 101, 100.0, "100"])
def test_both_signed_basis_readers_reject_invalid_age_even_when_basis_is_stale(bad):
    basis = _body(_basis([_win(), None, None]))
    basis["reward_slots"][0]["start_epoch"] = bad
    with pytest.raises(kernel.LabArenaRewardError, match="start_epoch"):
        kernel.validate_reward_basis(basis)
    with pytest.raises(contracts.ArenaContractError, match="start_epoch"):
        contracts.validate_reward_basis(basis)
    with pytest.raises(kernel.LabArenaRewardError, match="start_epoch"):
        kernel.slot_allocations(basis, 6000, HOTKEYS)


@pytest.mark.parametrize("field", ["start_epoch", "epochs_per_halving", "max_halvings"])
def test_decay_age_and_policy_are_bound_by_hash_and_signature(field):
    basis = _basis([_win(), None, None], epoch=240)
    state, signer = _state(basis, 240)
    signed = deepcopy(state["reward_basis"])
    if field == "start_epoch":
        signed["reward_slots"][0][field] = 101
    else:
        signed["slot_policy"]["decay"][field] += 1
    with pytest.raises(kernel.LabArenaRewardError, match="hash"):
        kernel.verify_reward_basis_signature(
            signed, public_key_der=signer.public_key_der,
            expected_public_key_hash=signer.public_key_hash,
        )
    signed["reward_basis_hash"] = kernel.sha256_json(_body(signed))
    with pytest.raises(kernel.LabArenaRewardError, match="signature invalid"):
        kernel.verify_reward_basis_signature(
            signed, public_key_der=signer.public_key_der,
            expected_public_key_hash=signer.public_key_hash,
        )


@pytest.mark.parametrize("pool", [None, 30])
def test_legacy_no_decay_keeps_six_field_shape_and_no_holder_age(pool):
    policy = deepcopy(rewards.REWARD_SLOT_POLICY)
    policy.pop("decay")
    if pool is None:
        policy.pop("pool_percent")
    slot = _win()
    slot.pop("start_epoch")
    basis = _basis([slot, slot, slot], epoch=6000, policy=policy)
    assert len(basis["reward_slots"][0]) == 6
    assert "decay" not in basis["slot_policy"]
    assert kernel.slot_allocations(basis, 6000, HOTKEYS) == {
        ALICE: Fraction(pool if pool is not None else 100, 100),
    }
    invalid = _body(basis)
    invalid["reward_slots"][0]["start_epoch"] = 100
    with pytest.raises(kernel.LabArenaRewardError, match="fields"):
        kernel.validate_reward_basis(invalid)


def test_central_decay_policy_document_returns_independent_nested_values():
    first = rewards.reward_slot_policy_document()
    first["decay"]["max_halvings"] = 1
    assert rewards.reward_slot_policy_document()["decay"] == {
        "epochs_per_halving": contracts.EPOCHS_PER_REWARD_WEEK, "max_halvings": 4,
    }
