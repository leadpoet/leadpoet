"""Correct only new reward states; keep the published November 3 award."""

from copy import deepcopy

import pytest

from lab_arena import rewards, signing, weight_state
from lab_arena.service import ServiceError
from lab_arena.store import ArenaStoreError
from leadpoet_canonical import arena_weights, lab_arena_rewards as kernel
from tests.lab_arena.reward_thresholds_oct23_test import _service
from tests.lab_arena.reward_shares_oct22_test import _activation_service
from tests.lab_arena.slot_decay_kernel_test import BURN, _basis, _win


OWNER = "5HDmNaBQNVQoyApQbFQdSx6pD4ajcJdaYDJePufpWvoxCsrq"
SOURCE = "sha256:1047b383f4b32dbc38d19e5f052384174d4a9a89a7c48b4620cecb18714ed06d"
START = 26183


@pytest.mark.parametrize("evaluation_date,policy", [
    ("2026-11-10", rewards.REWARD_SLOT_POLICY_OCT23),
    ("2026-11-11", rewards.REWARD_SLOT_POLICY_NOV11),
])
def test_actual_activation_uses_round_date_not_late_chain_epoch(evaluation_date, policy):
    service, row, captured = _activation_service(evaluation_date)
    service._config.chain.current_settlement_epoch = lambda: 26318
    assert service.activate_reward(row["round_id"])["status"] == "activated"
    assert captured["basis"]["slot_policy"] == policy


def test_old_source_never_upgrades_merely_because_chain_epoch_advanced():
    service, _, _, _, _ = _service(epoch=26320)
    state = service.public_weight_state(26320)["state"]
    assert state["reward_basis"]["slot_policy"] == rewards.REWARD_SLOT_POLICY_OCT23


def _harness(epoch):
    service, original, saved, calls, signer = _service(epoch=epoch)
    winner = _win(OWNER, delta=27.125, start=START,
                  name="sub-28594d902080fb5213f4344c09ffb531",
                  round_id="arena-2026-11-03")
    original.clear()
    original.update(signing.sign_document(signer, _basis(
        [deepcopy(winner) for _ in range(3)], epoch=26315,
        policy=rewards.REWARD_SLOT_POLICY_NOV11, king=OWNER,
        round_id="arena-2026-11-10",
    ), hash_field="reward_basis_hash"))
    clock = {"epochs_per_halving": 140, "slots": [
        {"start_epoch": START, "source_normalized_sha256": SOURCE} for _ in range(3)
    ]}
    previous = weight_state.build_accepted_weight_state(
        signer, network="finney", netuid=71, genesis_hash="1" * 64,
        epoch=epoch - 1, valid_from_block=900, valid_until_block=1000,
        reward_basis=original, burn_hotkey=BURN,
        issued_at=original["published_at"], weekly_slot_decay=clock,
    )
    record = {key: previous[key] for key in ("network", "netuid", "epoch", "state_hash")}
    record["state_doc"] = previous
    service._config.weekly_slot_decay_enabled = True
    service._store.latest_weight_state = lambda *_: record
    service._store.get_submission = lambda *_: pytest.fail("verified unchanged clock needs no source reload")
    return service, original, saved, signer, clock


@pytest.mark.parametrize("epoch,share", [(26318, 500_000_000), (26322, 500_000_000),
                                        (26323, 250_000_000), (26463, 125_000_000)])
def test_actual_legacy_award_corrects_new_state_and_preserves_tenure(epoch, share):
    service, original, saved, signer, clock = _harness(epoch)
    before = kernel.canonical_json(original)
    state = service.public_weight_state(epoch)["state"]
    assert state["weekly_slot_decay"] == clock
    assert state["reward_basis"]["reward_slots"] == original["reward_slots"]
    assert state["reward_basis"]["slot_policy"]["pool_percent"] == 50
    assert kernel.canonical_json(original) == before
    vector = arena_weights.derive_arena_weights(state, [BURN, OWNER])
    assert vector["champion_share_ppb"] == share
    assert vector["burned_residual_ppb"] == 1_000_000_000 - share
    arena_weights.verify_accepted_weight_state_signature(
        state, public_key_der=signer.public_key_der,
        expected_public_key_hash=signer.public_key_hash,
    )
    # Subsequent reads/restarts and the dashboard use the identical accepted state.
    assert service.public_weight_state(epoch)["state"] == saved[epoch]["state_doc"] == state
    assert service.public_current()["reward_slot_policy"] == state["reward_basis"]["slot_policy"]


def test_already_issued_early_seventy_five_state_is_not_rewritten():
    epoch = 26318
    service, original, saved, signer, clock = _harness(epoch)
    issued = weight_state.build_accepted_weight_state(
        signer, network="finney", netuid=71, genesis_hash="1" * 64,
        epoch=epoch, valid_from_block=1000, valid_until_block=1100,
        reward_basis=original, burn_hotkey=BURN, issued_at=original["published_at"],
        weekly_slot_decay=clock,
    )
    saved[epoch] = {"state_doc": issued}
    before = kernel.canonical_json(issued)
    assert kernel.canonical_json(service.public_weight_state(epoch)["state"]) == before
    assert arena_weights.derive_arena_weights(issued, [BURN, OWNER])["champion_share_ppb"] == 750_000_000


@pytest.mark.parametrize("mutation", [None, "burn", "clock", "source"])
def test_old_gateway_first_writer_is_preserved_only_with_same_authority(mutation):
    epoch = 26323
    service, original, saved, signer, clock = _harness(epoch)
    def publish(_network, _netuid, _epoch, _digest, _candidate):
        raced_clock = deepcopy(clock)
        raced_basis = deepcopy(original)
        if mutation == "clock":
            raced_clock["slots"][0]["start_epoch"] -= 1
        if mutation == "source":
            raced_basis["reward_slots"][0]["submission_id"] = "another-source"
            raced_basis = signing.sign_document(signer, rewards.finalize_reward_basis({
                key: value for key, value in raced_basis.items() if key not in ("signature", "reward_basis_hash")
            }), hash_field="reward_basis_hash")
        issued = weight_state.build_accepted_weight_state(
            signer, network="finney", netuid=71, genesis_hash="1" * 64,
            epoch=epoch, valid_from_block=1000, valid_until_block=1100,
            reward_basis=raced_basis, burn_hotkey=OWNER if mutation == "burn" else BURN,
            issued_at=original["published_at"], weekly_slot_decay=raced_clock,
        )
        saved[epoch] = {"state_doc": issued}
        raise ArenaStoreError("lab_arena_weight_state_conflict")
    service._store.publish_weight_state = publish
    if mutation:
        with pytest.raises(ServiceError):
            service.public_weight_state(epoch)
    else:
        assert service.public_weight_state(epoch)["state"] == saved[epoch]["state_doc"]


def test_legacy_award_cannot_be_increased_when_verified_clock_is_disabled():
    service, _, _, _, _ = _harness(26323)
    service._config.weekly_slot_decay_enabled = False
    with pytest.raises(ServiceError, match="reward.*clock|reward.*projection"):
        service.public_weight_state(26323)
