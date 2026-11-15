"""Each verified qualifying award owns its slot decay clock."""
from copy import deepcopy
import hashlib

import pytest

from lab_arena.reward_decay import weekly_slot_decay
from lab_arena.rewards import finalize_reward_basis
from tests.lab_arena.slot_decay_kernel_test import ALICE, BOB, _basis, _win
from tests.lab_arena.weekly_slot_decay_kernel_test import POLICY


def basis(epoch, model, owner=ALICE, *, empty=False, legacy=False):
    round_id = f"arena-2026-{epoch // 100:02d}-01"
    achievement = _win(owner, delta=20, name=model, start=None, round_id=round_id)
    policy = deepcopy(POLICY)
    if legacy:
        policy.pop("decay")
        achievement.pop("start_epoch")
    return _basis([None if empty else deepcopy(achievement) for _ in range(3)],
                  epoch=epoch, round_id=round_id, policy=policy, king=owner)


def identity(slot):
    return "sha256:" + hashlib.sha256(slot["submission_id"].split("-copy")[0].encode()).hexdigest()


def prepare(current, history, previous=None, **kwargs):
    epoch = current["effective_reward_epoch"]
    return weekly_slot_decay(current, epoch=epoch, source_identity=identity,
                             history=history, previous_state=previous, **kwargs)


def state(current, history):
    return {"reward_basis": current, "weekly_slot_decay": prepare(current, history)}


def starts(result):
    return [None if slot is None else slot["start_epoch"] for slot in result["slots"]]


def test_new_model_on_same_hotkey_resets_only_earned_slots():
    first = basis(100, "A")
    second = basis(200, "B")
    old = deepcopy(first["reward_slots"][0]); old["start_epoch"] = 100
    second["reward_slots"][0] = old
    second = finalize_reward_basis({k: v for k, v in second.items() if k != "reward_basis_hash"})
    assert starts(prepare(second, [first], state(first, []))) == [100, 200, 200]


def test_new_qualifying_award_with_repacked_same_model_resets_age():
    first, second = basis(100, "A"), basis(200, "A-copy")
    assert starts(prepare(second, [first], state(first, []))) == [200] * 3


@pytest.mark.parametrize("middle,latest,expected", [("B", "A", 300), ("B", "B-copy", 300)])
def test_skipped_weight_epochs_do_not_skip_intervening_promotions(middle, latest, expected):
    first, second, third = basis(100, "A"), basis(200, middle), basis(300, latest)
    assert starts(prepare(third, [first, second], state(first, []))) == [expected] * 3


def test_same_source_after_another_owner_resets_age():
    first, second, third = basis(100, "A"), basis(200, "A-copy", BOB), basis(300, "A-copy2")
    assert starts(prepare(third, [first, second], state(first, []))) == [300] * 3


def test_same_source_after_empty_slot_resets_age():
    first, empty, third = basis(100, "A"), basis(200, "A", empty=True), basis(300, "A-copy")
    assert starts(prepare(third, [first, empty], state(first, []))) == [300] * 3


def test_bootstrap_does_not_merge_distinct_same_source_awards():
    first, second, third = basis(100, "A"), basis(200, "A-copy"), basis(300, "A-copy2")
    assert starts(prepare(third, [second, first])) == [300] * 3


def test_old_no_decay_award_does_not_age_a_new_same_source_win():
    first, second = basis(100, "A", legacy=True), basis(200, "A-copy")
    assert starts(prepare(second, [first])) == [200] * 3


def test_new_award_has_its_own_signed_epoch_despite_truncated_history():
    assert starts(prepare(basis(300, "A"), [], history_complete=False)) == [300] * 3


def test_changed_model_can_stop_at_proven_previous_occupant_despite_truncated_older_history():
    first, second = basis(100, "A"), basis(200, "B")
    assert starts(prepare(second, [first], history_complete=False)) == [200] * 3


def test_daily_basis_restart_and_funding_change_use_signed_checkpoint_without_new_source_reads():
    first = basis(100, "A")
    previous = state(first, [])
    current = basis(300, "ignored")
    current["reward_slots"] = deepcopy(first["reward_slots"])
    for slot in current["reward_slots"]:
        slot["start_epoch"] = 100
    current["champion_reward_factor_ppm"] = 500_000
    current = finalize_reward_basis({k: v for k, v in current.items() if k != "reward_basis_hash"})
    def no_source(_):
        raise AssertionError("unchanged source fetched again")
    result = weekly_slot_decay(current, epoch=300, source_identity=no_source,
                               history=[], previous_state=previous, history_complete=False)
    assert starts(result) == [100] * 3
    assert previous["weekly_slot_decay"]["slots"] == result["slots"]


def test_no_occupied_slots_require_no_source_or_history():
    current = basis(100, "A", empty=True)
    assert starts(prepare(current, [], history_complete=False)) == [None] * 3


@pytest.mark.parametrize("with_checkpoint", [False, True])
def test_replaying_an_old_award_after_an_intervening_win_keeps_original_epoch(with_checkpoint):
    first, second = basis(100, "A"), basis(200, "B")
    replay = basis(300, "unused")
    replay["reward_slots"] = deepcopy(first["reward_slots"])
    for slot in replay["reward_slots"]:
        slot["start_epoch"] = 100
    replay = finalize_reward_basis({key: value for key, value in replay.items()
                                   if key != "reward_basis_hash"})
    previous = state(second, [first]) if with_checkpoint else None
    assert starts(prepare(replay, [second, first], previous)) == [100] * 3


def test_legacy_carried_award_without_start_uses_its_own_source_round():
    first = basis(100, "A", legacy=True)
    carried = basis(300, "unused", legacy=True)
    carried["reward_slots"] = deepcopy(first["reward_slots"])
    carried = finalize_reward_basis({key: value for key, value in carried.items()
                                    if key != "reward_basis_hash"})
    assert starts(prepare(carried, [first])) == [100] * 3
    with pytest.raises(ValueError, match="source promotion is missing"):
        prepare(carried, [])
    with pytest.raises(ValueError, match="history is incomplete"):
        prepare(carried, [], history_complete=False)


def test_previously_signed_wrong_model_clock_is_repaired_for_a_future_state():
    current = basis(300, "A-copy")
    previous = {"reward_basis": current, "weekly_slot_decay": {
        "epochs_per_halving": 140, "slots": [
            {"start_epoch": 100, "source_normalized_sha256": identity(slot)}
            for slot in current["reward_slots"]],
    }}
    before = deepcopy(previous)
    assert starts(prepare(current, [], previous)) == [300] * 3
    assert previous == before
