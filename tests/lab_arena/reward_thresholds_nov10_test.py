"""The prospective 75-percent policy carries signed slots without requalification."""

from copy import deepcopy
from datetime import date, timedelta
from fractions import Fraction

import pytest

from lab_arena import reward_decay, rewards, signing, weight_state
from lab_arena.service import ServiceError
from lab_arena.store import ArenaStoreError
from leadpoet_canonical import lab_arena_rewards as kernel
from tests.lab_arena.slot_decay_kernel_test import ALICE, BOB, BURN, CAROL, HOTKEYS, _basis, _win
from tests.lab_arena.reward_thresholds_oct23_test import _service


def _policy():
    return deepcopy(rewards.REWARD_SLOT_POLICY_NOV11)


def _source():
    epoch = rewards.REWARD_SLOT_POLICY_NOV11_EFFECTIVE_EPOCH - 1
    slots = [_win(ALICE, delta=20, start=epoch - 280, name="high"),
             _win(BOB, delta=15, start=epoch - 140, name="mid"),
             _win(CAROL, delta=8, start=epoch, name="low")]
    return _basis(slots, epoch=epoch, policy=rewards.REWARD_SLOT_POLICY_OCT23)


def test_registered_date_and_epoch_boundaries_preserve_earlier_policy():
    cutover = rewards.REWARD_SLOT_POLICY_NOV11_EFFECTIVE_EPOCH
    day = rewards.REWARD_SLOT_POLICY_NOV11_START
    assert rewards.reward_slot_policy_document(day - timedelta(days=1)) == rewards.REWARD_SLOT_POLICY_OCT23
    assert rewards.reward_slot_policy_document(effective_reward_epoch=cutover - 1) == rewards.REWARD_SLOT_POLICY_OCT23
    assert rewards.reward_slot_policy_document(day) == _policy()
    assert rewards.reward_slot_policy_document(date(2026, 11, 10), effective_reward_epoch=cutover + 100) == rewards.REWARD_SLOT_POLICY_OCT23
    assert rewards.reward_slot_policy_document(effective_reward_epoch=cutover + 100) == rewards.REWARD_SLOT_POLICY_OCT23
    assert _policy()["allocation_basis"] == "total_emissions"
    assert _policy()["pool_percent"] == 75
    assert [(tier["minimum_improvement"], tier["allocation_percent"]) for tier in _policy()["tiers"]] == [(15, 35), (10, 25), (5, 15)]


def test_projection_carries_each_current_ordinal_and_original_tenure():
    source = _source()
    original = kernel.canonical_json(source)
    projected = rewards.reward_basis_policy_projection_document(source, _policy())
    assert projected["reward_slots"] == source["reward_slots"]
    assert projected["slot_policy"] == _policy()
    assert projected["reward_basis_hash"] != source["reward_basis_hash"]
    assert kernel.canonical_json(source) == original
    assert kernel.slot_allocations(projected, source["effective_reward_epoch"], HOTKEYS) == {
        ALICE: Fraction(35, 400), BOB: Fraction(25, 200), CAROL: Fraction(15, 100),
    }
    # A repeated projection must not let a newer lower holder fill upper tiers.
    assert rewards.reward_basis_policy_projection_document(projected, _policy()) == projected


def test_pre_oct23_source_is_normalized_before_lowering_without_resurrection():
    epoch = rewards.REWARD_SLOT_POLICY_NOV11_EFFECTIVE_EPOCH - 1
    high = _win(ALICE, delta=11, start=epoch - 140, name="displaced-high")
    low = _win(BOB, delta=6, start=epoch, name="latest-low")
    source = _basis([high, low, deepcopy(low)], epoch=epoch,
                    policy=rewards.REWARD_SLOT_POLICY_50_PERCENT)
    projected = rewards.reward_basis_policy_projection_document(source, _policy())
    assert projected["reward_slots"] == [None, None, high]
    # The six-point holder lost eligibility at the old Oct23 boundary. It
    # cannot return just because the prospective minimum is now five points.
    assert all(slot is None or slot["submission_id"] != low["submission_id"]
               for slot in projected["reward_slots"])
    assert kernel.canonical_json(source) == kernel.canonical_json(kernel.validate_reward_basis(source))


def test_pre_oct23_newer_eight_does_not_revive_older_eleven():
    epoch = rewards.REWARD_SLOT_POLICY_NOV11_EFFECTIVE_EPOCH - 1
    high = _win(ALICE, delta=11, start=epoch - 140, name="old-high")
    newest = _win(BOB, delta=8, start=epoch, name="new-eight")
    source = _basis([high, newest, deepcopy(newest)], epoch=epoch,
                    policy=rewards.REWARD_SLOT_POLICY_50_PERCENT)
    assert rewards.reward_basis_policy_projection_document(source, _policy())["reward_slots"] == [None, None, newest]


def test_initial_legacy_policy_without_pool_can_establish_transition_authority():
    epoch = rewards.REWARD_SLOT_POLICY_NOV11_EFFECTIVE_EPOCH - 1
    policy = deepcopy(rewards.REWARD_SLOT_POLICY)
    policy.pop("pool_percent")
    winner = _win(ALICE, delta=11, start=epoch)
    source = _basis([winner] * 3, epoch=epoch, policy=policy)
    assert rewards.reward_basis_policy_projection_document(source, _policy())["reward_slots"] == [None, None, winner]


@pytest.mark.parametrize("mutation", ["threshold", "allocation", "decay"])
def test_only_registered_lowering_with_unchanged_decay_is_permitted(mutation):
    policy = _policy()
    if mutation == "threshold":
        policy["tiers"][0]["minimum_improvement"] = 14
    elif mutation == "allocation":
        policy["tiers"][0]["allocation_percent"] -= 1
        policy["tiers"][1]["allocation_percent"] += 1
    else:
        policy["decay"]["epochs_per_halving"] = 141
    with pytest.raises(kernel.LabArenaRewardError):
        rewards.reward_basis_policy_projection_document(_source(), policy)


def test_preexisting_oct23_projection_remains_accepted_after_new_cutover():
    cutover = rewards.REWARD_SLOT_POLICY_NOV11_EFFECTIVE_EPOCH
    service, original, saved, calls, signer = _service(epoch=cutover)
    old_projection = rewards.reward_basis_policy_projection_document(original, rewards.REWARD_SLOT_POLICY_OCT23)
    signed = signing.sign_document(signer, old_projection, hash_field="reward_basis_hash")
    accepted = weight_state.build_accepted_weight_state(
        signer, network="finney", netuid=71, genesis_hash="1" * 64,
        epoch=cutover, valid_from_block=1000, valid_until_block=1100,
        reward_basis=signed, burn_hotkey=BURN, issued_at=original["published_at"],
    )
    before = kernel.canonical_json(accepted)
    saved[cutover] = {"state_doc": accepted}
    assert rewards.reward_basis_matches_policy_projection(old_projection, original, cutover)
    assert kernel.canonical_json(service.public_weight_state(cutover)["state"]) == before
    assert calls == []


def test_new_projection_is_not_valid_before_cutover_and_preserves_old_epoch():
    source = _source()
    projected = rewards.reward_basis_policy_projection_document(source, _policy())
    cutover = rewards.REWARD_SLOT_POLICY_NOV11_EFFECTIVE_EPOCH
    assert not rewards.reward_basis_matches_policy_projection(projected, source, cutover - 1)
    assert rewards.reward_basis_matches_policy_projection(projected, source, cutover)
    assert projected["effective_reward_epoch"] == source["effective_reward_epoch"]


def test_same_epoch_latest_race_preserves_preexisting_oct23_projection_bytes():
    cutover = rewards.REWARD_SLOT_POLICY_NOV11_EFFECTIVE_EPOCH
    service, original, saved, calls, signer = _service(epoch=cutover)
    body = {key: value for key, value in original.items() if key not in ("reward_basis_hash", "signature")}
    body["effective_reward_epoch"] = cutover
    original.clear()
    original.update(signing.sign_document(
        signer, rewards.finalize_reward_basis(body), hash_field="reward_basis_hash",
    ))
    old_projection = rewards.reward_basis_policy_projection_document(original, rewards.REWARD_SLOT_POLICY_OCT23)
    signed = signing.sign_document(signer, old_projection, hash_field="reward_basis_hash")
    accepted = weight_state.build_accepted_weight_state(
        signer, network="finney", netuid=71, genesis_hash="1" * 64,
        epoch=cutover, valid_from_block=1000, valid_until_block=1100,
        reward_basis=signed, burn_hotkey=BURN, issued_at=original["published_at"],
    )
    record = {key: accepted[key] for key in ("network", "netuid", "epoch", "state_hash")}
    record["state_doc"] = accepted
    service._config.weekly_slot_decay_enabled = True
    service._store.latest_weight_state = lambda *_args: record
    before = kernel.canonical_json(accepted)
    assert kernel.canonical_json(service.public_weight_state(cutover)["state"]) == before
    assert saved == {}


def _publish_race(*, earlier=True, mutation=None, weekly=False):
    cutover = rewards.REWARD_SLOT_POLICY_NOV11_EFFECTIVE_EPOCH
    service, original, saved, calls, signer = _service(epoch=cutover)
    policy = rewards.REWARD_SLOT_POLICY_OCT23 if earlier else _policy()
    winner_basis = rewards.reward_basis_policy_projection_document(original, policy)
    if mutation == "source":
        winner_basis["reward_slots"][2]["submission_id"] = "unrelated-source"
        winner_basis = rewards.finalize_reward_basis({
            key: value for key, value in winner_basis.items() if key != "reward_basis_hash"})
    elif mutation == "policy":
        winner_basis["slot_policy"]["tiers"][0]["allocation_percent"] += 1
        winner_basis["slot_policy"]["tiers"][1]["allocation_percent"] -= 1
        winner_basis = rewards.finalize_reward_basis({
            key: value for key, value in winner_basis.items() if key != "reward_basis_hash"})
    winner_basis = signing.sign_document(signer, winner_basis, hash_field="reward_basis_hash")
    clock = {"epochs_per_halving": 140, "slots": [None, None, {
        "start_epoch": 25846, "source_normalized_sha256": "sha256:" + "1" * 64,
    }]}
    if weekly:
        service._config.weekly_slot_decay_enabled = True
        service._store.latest_weight_state = lambda *_args: None
        service._weekly_reward_clock = lambda *_args, **_kwargs: deepcopy(clock)
    def publish(network, netuid, epoch, digest, candidate):
        kwargs = {}
        if weekly:
            winner_clock = deepcopy(clock)
            if mutation == "clock":
                winner_clock["slots"][2]["start_epoch"] -= 1
            kwargs["weekly_slot_decay"] = winner_clock
        winner = weight_state.build_accepted_weight_state(
            signer, network=network, netuid=netuid, genesis_hash="1" * 64,
            epoch=epoch, valid_from_block=1000,
            valid_until_block=1101 if mutation == "validity" else 1100,
            reward_basis=winner_basis, burn_hotkey=BOB if mutation == "burn" else BURN,
            issued_at=original["published_at"], **kwargs,
        )
        if mutation == "signature":
            winner["signature"]["signature_b64"] = "AAAA"
        saved[epoch] = {"state_doc": winner}
        raise ArenaStoreError("lab_arena_weight_state_conflict")
    service._store.publish_weight_state = publish
    return service, saved, cutover


@pytest.mark.parametrize("weekly", [False, True])
def test_old_policy_first_publisher_wins_cutover_race_on_first_request(weekly):
    service, saved, epoch = _publish_race(weekly=weekly)
    returned = service.public_weight_state(epoch)["state"]
    assert returned["reward_basis"]["slot_policy"] == rewards.REWARD_SLOT_POLICY_OCT23
    assert returned == saved[epoch]["state_doc"]
    assert service.public_weight_state(epoch)["state"] == returned


@pytest.mark.parametrize("earlier", [False, True])
@pytest.mark.parametrize("mutation", ["source", "policy", "clock", "burn", "validity", "signature"])
def test_publish_race_still_rejects_unrelated_authority_or_changed_body(earlier, mutation):
    service, saved, epoch = _publish_race(earlier=earlier, mutation=mutation, weekly=True)
    with pytest.raises(ServiceError, match="accepted_weight_state_(reward_conflict|conflict|invalid)"):
        service.public_weight_state(epoch)


@pytest.mark.parametrize("delta,occupied,expected", [
    (5, [False, False, True], Fraction(15, 100)),
    (10, [False, True, True], Fraction(40, 100)),
    (15, [True, True, True], Fraction(75, 100)),
])
def test_new_promotions_use_new_cumulative_thresholds(delta, occupied, expected):
    epoch = rewards.REWARD_SLOT_POLICY_NOV11_EFFECTIVE_EPOCH
    winner = _win(delta=delta, start=None, round_id="arena-2026-10-08")
    basis = _basis([winner if present else None for present in occupied], epoch=epoch, policy=_policy())
    assert kernel.slot_allocations(basis, epoch, HOTKEYS) == {ALICE: expected}


@pytest.mark.parametrize("decay", [None, {"epochs_per_halving": 140, "max_halvings": 4}])
def test_history_identity_projection_preserves_original_decay_shape(decay):
    source = _source()
    body = {key: value for key, value in source.items() if key != "reward_basis_hash"}
    policy = deepcopy(source["slot_policy"])
    target = _policy()
    if decay is None:
        policy.pop("decay")
        target.pop("decay")
        body["reward_slots"] = [{key: value for key, value in slot.items() if key != "start_epoch"}
                                for slot in source["reward_slots"]]
    body["slot_policy"] = policy
    historical = rewards.finalize_reward_basis(body)
    projected = rewards.reward_basis_policy_projection_document(historical, target)
    assert projected["reward_slots"] == historical["reward_slots"]
    assert projected["slot_policy"].get("decay") == decay


def test_weekly_reconstruction_keeps_each_incumbent_model_clock_across_transition():
    old = _source()
    projected = rewards.reward_basis_policy_projection_document(old, _policy())
    epoch = rewards.REWARD_SLOT_POLICY_NOV11_EFFECTIVE_EPOCH
    identity = lambda slot: "sha256:" + {ALICE: "1", BOB: "2", CAROL: "3"}[slot["miner_hotkey"]] * 64
    clock = reward_decay.weekly_slot_decay(projected, epoch=epoch, source_identity=identity, history=[old])
    assert [item["start_epoch"] for item in clock["slots"]] == [slot["start_epoch"] for slot in old["reward_slots"]]


@pytest.mark.parametrize("changed", [False, True])
def test_same_hotkey_new_promotion_uses_projected_history_without_resetting_other_tiers(changed):
    old = _source()
    epoch = rewards.REWARD_SLOT_POLICY_NOV11_EFFECTIVE_EPOCH
    prior_clocks = [{"start_epoch": slot["start_epoch"],
                     "source_normalized_sha256": "sha256:" + str(index + 1) * 64}
                    for index, slot in enumerate(old["reward_slots"])]
    slots = deepcopy(old["reward_slots"])
    slots[2] = _win(CAROL, delta=5, start=None, name="fresh-low", round_id="arena-2026-10-08")
    current = _basis(slots, epoch=epoch, policy=_policy())
    digest = "sha256:" + ("4" if changed else "3") * 64
    clock = reward_decay.weekly_slot_decay(
        current, epoch=epoch, source_identity=lambda _slot: digest,
        history=[old], previous_state={"reward_basis": old,
            "weekly_slot_decay": {"epochs_per_halving": 140, "slots": prior_clocks}},
    )
    assert clock["slots"][:2] == prior_clocks[:2]
    assert clock["slots"][2]["start_epoch"] == (epoch if changed else prior_clocks[2]["start_epoch"])
