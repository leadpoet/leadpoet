"""Retained legacy awards keep old shares; qualifying re-wins get new shares."""

from copy import deepcopy
from datetime import date
from fractions import Fraction

import pytest

from lab_arena import reward_decay, rewards
from leadpoet_canonical import arena_weights, lab_arena_rewards as kernel
from tests.lab_arena.slot_decay_kernel_test import BOB, BURN, _basis, _win
from tests.lab_arena.weekly_slot_decay_kernel_test import _signed_state


LEGACY = "5HDmNaBQNVQoyApQbFQdSx6pD4ajcJdaYDJePufpWvoxCsrq"
MODEL = "sha256:1047b383f4b32dbc38d19e5f052384174d4a9a89a7c48b4620cecb18714ed06d"
NEW_MODEL = "sha256:" + "2" * 64
START = 26183
EFFECTIVE = 26315
HOTKEYS = [BURN, LEGACY, BOB]


def _source(legacy=(True, True, True), *, same_model_new_submission=False):
    old = _win(LEGACY, delta=20, start=START, name="nov03",
               round_id="arena-2026-11-03")
    if same_model_new_submission:
        old = _win(LEGACY, delta=20, start=None, name="nov11-same-model",
                   round_id="arena-2026-11-11")
    fresh = _win(BOB, delta=20, start=None, name="nov11-new-model",
                 round_id="arena-2026-11-11")
    slots = [deepcopy(old if keep else fresh) for keep in legacy]
    source = _basis(slots, epoch=EFFECTIVE, policy=rewards.REWARD_SLOT_POLICY_NOV11,
                    king=LEGACY, round_id="arena-2026-11-11")
    clock = {"epochs_per_halving": 140, "slots": [
        {"start_epoch": START if keep else EFFECTIVE,
         "source_normalized_sha256": MODEL if keep else NEW_MODEL}
        for keep in legacy
    ]}
    return source, clock


def test_november_tenth_activation_stays_on_old_policy_even_at_early_epoch():
    assert rewards.reward_slot_policy_document(
        date(2026, 11, 10), effective_reward_epoch=EFFECTIVE,
    ) == rewards.REWARD_SLOT_POLICY_OCT23
    assert rewards.reward_slot_policy_document(
        effective_reward_epoch=EFFECTIVE,
    ) == rewards.REWARD_SLOT_POLICY_OCT23
    assert rewards.reward_slot_policy_document(date(2026, 11, 11)) == rewards.REWARD_SLOT_POLICY_NOV11


def test_legacy_award_keeps_original_age_and_source_bytes():
    source, clock = _source()
    before = deepcopy(source)
    projected = rewards.legacy_slot_allocation_projection_document(source, clock)
    assert source == before
    assert projected["reward_slots"] == source["reward_slots"]
    assert projected["slot_policy"]["pool_percent"] == 50
    assert [tier["allocation_percent"] for tier in projected["slot_policy"]["tiers"]] == [20, 15, 15]
    assert kernel.slot_allocations(projected, START + 139, HOTKEYS,
                                   until_replaced=True, weekly_slot_decay=clock) == {LEGACY: Fraction(1, 2)}
    assert kernel.slot_allocations(projected, START + 140, HOTKEYS,
                                   until_replaced=True, weekly_slot_decay=clock) == {LEGACY: Fraction(1, 4)}
    assert rewards.reward_basis_matches_policy_projection(
        projected, source, START + 140, weekly_slot_decay=clock,
    )
    assert not rewards.reward_basis_matches_policy_projection(projected, source, START + 140)
    assert rewards.reward_basis_matches_policy_projection(source, source, START + 140)


@pytest.mark.parametrize("legacy,expected", [
    ((True, True, True), (20, 15, 15)),
    ((True, True, False), (20, 15, 15)),
    ((True, False, False), (20, 25, 15)),
    ((False, True, True), (35, 15, 15)),
    ((False, False, False), (35, 25, 15)),
])
def test_only_replaced_slots_get_new_award(legacy, expected):
    source, clock = _source(legacy)
    projected = rewards.legacy_slot_allocation_projection_document(source, clock)
    assert tuple(tier["allocation_percent"] for tier in projected["slot_policy"]["tiers"]) == expected
    assert projected["slot_policy"]["pool_percent"] == sum(expected)
    assert projected["reward_slots"] == source["reward_slots"]


def test_historical_signed_same_model_clock_still_validates_its_old_award():
    source, clock = _source(same_model_new_submission=True)
    projected = rewards.legacy_slot_allocation_projection_document(source, clock)
    assert projected["slot_policy"]["pool_percent"] == 50
    assert projected["reward_slots"][0]["submission_id"] == "nov11-same-model"


@pytest.mark.parametrize("mutation", ["digest", "owner", "tenure"])
def test_different_model_owner_or_tenure_does_not_get_old_award(mutation):
    source, clock = _source()
    if mutation == "digest":
        for slot in clock["slots"]:
            slot["source_normalized_sha256"] = NEW_MODEL
    elif mutation == "owner":
        for slot in source["reward_slots"]:
            slot["miner_hotkey"] = BOB
        source = rewards.finalize_reward_basis({
            key: value for key, value in source.items() if key != "reward_basis_hash"
        })
    else:
        for slot in source["reward_slots"]:
            slot["start_epoch"] = START + 1
        for slot in clock["slots"]:
            slot["start_epoch"] = START + 1
        source = rewards.finalize_reward_basis({
            key: value for key, value in source.items() if key != "reward_basis_hash"
        })
    projected = rewards.legacy_slot_allocation_projection_document(source, clock)
    assert projected == source


def test_legacy_owner_requires_a_valid_clock():
    source, clock = _source()
    with pytest.raises(kernel.LabArenaRewardError):
        rewards.legacy_slot_allocation_projection_document(source, None)
    clock["slots"][0]["start_epoch"] = EFFECTIVE + 1
    with pytest.raises(kernel.LabArenaRewardError):
        rewards.legacy_slot_allocation_projection_document(source, clock)


def test_no_legacy_owner_needs_no_clock_or_projection():
    source, _ = _source((False, False, False))
    assert rewards.legacy_slot_allocation_projection_document(source, None) == source


@pytest.mark.parametrize("delta,replaced", [(5, [False, False, True]),
                                         (10, [False, True, True]),
                                         (15, [True, True, True])])
@pytest.mark.parametrize("age", [0, 139, 140, 279, 280, 700])
def test_same_hotkey_same_source_qualifying_rewin_gets_full_new_tiers(delta, replaced, age):
    original, old_clock = _source()
    previous = {"reward_basis": original, "weekly_slot_decay": old_clock}
    before = deepcopy(previous)
    fresh = _win(LEGACY, delta=delta, start=None, name="qualifying-rewin",
                 round_id=original["round_id"])
    slots = [deepcopy(fresh if won else old)
             for won, old in zip(replaced, original["reward_slots"])]
    current = _basis(slots, epoch=EFFECTIVE, policy=rewards.REWARD_SLOT_POLICY_NOV11,
                     king=LEGACY, round_id=original["round_id"])
    epoch = EFFECTIVE + age
    clock = reward_decay.weekly_slot_decay(
        current, epoch=epoch, source_identity=lambda _: MODEL, history=[original],
        previous_state=previous,
    )
    assert [item["start_epoch"] for item in clock["slots"]] == [
        EFFECTIVE if won else START for won in replaced]
    projected = rewards.legacy_slot_allocation_projection_document(current, clock)
    expected_shares = [new if won else old for won, new, old in
                       zip(replaced, [35, 25, 15], [20, 15, 15])]
    assert [tier["allocation_percent"] for tier in projected["slot_policy"]["tiers"]] == expected_shares
    expected = sum((Fraction(share, 100 * 2 ** ((epoch - slot["start_epoch"]) // 140))
                    for share, slot in zip(expected_shares, clock["slots"])), Fraction(0))
    assert kernel.slot_allocations(projected, epoch, HOTKEYS, until_replaced=True,
                                   weekly_slot_decay=clock) == {LEGACY: expected}
    state, _ = _signed_state(projected, clock, epoch=epoch)
    vector = arena_weights.derive_arena_weights(state, HOTKEYS)
    assert vector["champion_share_ppb"] == int(expected * 1_000_000_000)
    assert vector["burned_residual_ppb"] == int((1 - expected) * 1_000_000_000)
    assert previous == before
