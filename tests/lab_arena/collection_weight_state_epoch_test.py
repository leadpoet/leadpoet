"""A collection weight-state read outside its epoch refuses 409, never 503.

Validators poll ``/arena/v1/weight-state`` continuously. Once per tempo one poll
lands on the far side of the epoch boundary and no state exists for it yet. The
legacy reward path answers that with ``accepted_weight_epoch_not_current`` (409);
after the collection cutover the same request was answered
``collection_weight_state_unavailable`` (503), because the catch-all around
``collection_rewards.epoch_state`` swallowed the refusal.
"""
from types import SimpleNamespace

import pytest

from lab_arena import collection_rewards
from lab_arena.service import ArenaService, ServiceError

GENESIS = "a" * 64
BURN = "5BurnHotkeyAddressForCollectionRewardTests000000000000"
EPOCH = 26600


def _chain(scope_epoch):
    return SimpleNamespace(accepted_weight_epoch_scope=lambda: {
        "epoch": scope_epoch, "genesis_hash": GENESIS,
        "valid_from_block": 50, "valid_until_block": 409,
    })


def _store():
    return SimpleNamespace(
        get_weight_state=lambda *_args: None,
        collection_reward_quotes=lambda _epoch: pytest.fail(
            "an out-of-epoch collection read must refuse before quoting"),
    )


def test_epoch_state_raises_epoch_not_current_outside_its_epoch():
    with pytest.raises(collection_rewards.EpochNotCurrent):
        collection_rewards.epoch_state(
            _store(), _chain(EPOCH + 1), object(),
            epoch=EPOCH, burn_hotkey=BURN, valuation_source=lambda _block: None,
        )


def test_public_weight_state_refuses_409_not_503_at_the_epoch_boundary():
    service = object.__new__(ArenaService)
    service._store = _store()
    service._config = SimpleNamespace(
        mode="live", chain=_chain(EPOCH + 1), accepted_burn_hotkey=BURN,
        collection_emission_valuation=lambda _block: None,
    )
    service._chain_scope = lambda: ("finney", 71)
    service._collection_reward_active = lambda _epoch: True
    service._reward_signer = lambda: object()

    with pytest.raises(ServiceError) as refusal:
        service.public_weight_state(EPOCH)
    assert refusal.value.status == 409
    assert refusal.value.code == "accepted_weight_epoch_not_current"


def test_collection_chain_failure_remains_unavailable():
    def unreadable_scope():
        raise OSError('finalized chain read failed')

    service = object.__new__(ArenaService)
    service._store = _store()
    service._config = SimpleNamespace(
        mode='live', chain=SimpleNamespace(accepted_weight_epoch_scope=unreadable_scope),
        accepted_burn_hotkey=BURN, collection_emission_valuation=lambda _: None,
    )
    service._chain_scope = lambda: ('finney', 71)
    service._collection_reward_active = lambda _: True
    service._reward_signer = lambda: object()
    with pytest.raises(ServiceError) as unavailable:
        service.public_weight_state(EPOCH)
    assert (unavailable.value.status, unavailable.value.code) == (503, 'collection_weight_state_unavailable')
