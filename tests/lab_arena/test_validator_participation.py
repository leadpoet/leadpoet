"""Weight access remains permit-based, independent of recent Arena work."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lab_arena.service import ArenaService, ServiceError
from lab_arena.store import ArenaStore, ArenaStoreError


def weight_service(*, stake=100_000, eligible=False, network="finney"):
    snapshot = SimpleNamespace(
        netuid=71, hotkeys=("validator",), validator_permit=(True,), S=(stake,)
    )
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(
        network_name=network,
        netuid=71,
        chain=SimpleNamespace(metagraph=Mock(return_value=snapshot)),
    )
    service._store = SimpleNamespace(
        has_recent_participation=Mock(return_value=eligible)
    )
    service.validate_request = Mock(
        return_value={
            "hotkey": "validator",
            "round_id": "weight-state",
            "body": {"network": network, "netuid": 71, "epoch": 123},
        }
    )
    service.public_weight_state = Mock(
        return_value={"lookup_ok": True, "state": {"epoch": 123}}
    )
    return service, snapshot


@pytest.mark.parametrize("network", ["finney", "test"])
@pytest.mark.parametrize("stake", [0, 74_999.99, 75_000, 75_000.01, 100_000])
@pytest.mark.parametrize("eligible", [False, True])
def test_permitted_validators_receive_state_without_recent_work(network, stake, eligible):
    service, _ = weight_service(stake=stake, eligible=eligible, network=network)
    assert service.handle_weight_state({})["state"] == {"epoch": 123}
    service.store.has_recent_participation.assert_not_called()
    service.config.chain.metagraph.assert_called_once_with(finalized=True)
    service.store.has_recent_participation.return_value = False
    assert service.handle_weight_state({})["state"] == {"epoch": 123}
    service.store.has_recent_participation.assert_not_called()


@pytest.mark.parametrize(
    "stake", [None, True, -1, float("nan"), float("inf"), "100000"]
)
def test_weight_access_does_not_depend_on_scoring_stake_data(stake):
    service, _ = weight_service(stake=stake, eligible=True)
    assert service.handle_weight_state({})["lookup_ok"] is True
    service.store.has_recent_participation.assert_not_called()


def test_missing_stake_does_not_block_permitted_weight_access():
    service, snapshot = weight_service(eligible=True)
    del snapshot.S
    assert service.handle_weight_state({})["lookup_ok"] is True
    snapshot.stake = (100_000,)
    assert service.handle_weight_state({})["lookup_ok"] is True
    snapshot.stake = ()
    assert service.handle_weight_state({})["lookup_ok"] is True


def test_participation_database_failure_does_not_block_weight_state():
    service, _ = weight_service(eligible=True)
    service.store.has_recent_participation.side_effect = ArenaStoreError("private data")
    assert service.handle_weight_state({})["lookup_ok"] is True
    service.store.has_recent_participation.assert_not_called()


def test_chain_failure_still_blocks_weight_state():
    service, _ = weight_service()
    service.config.chain.metagraph.side_effect = RuntimeError("private data")
    with pytest.raises(ServiceError, match="validator_snapshot_unavailable") as caught:
        service.handle_weight_state({})
    assert caught.value.status == 503
    service.public_weight_state.assert_not_called()


def test_permission_and_scope_checks_still_precede_state_delivery():
    service, snapshot = weight_service(eligible=True)
    snapshot.validator_permit = (False,)
    with pytest.raises(ServiceError, match="runner_validator_required"):
        service.handle_weight_state({})
    service.store.has_recent_participation.assert_not_called()
    snapshot.validator_permit = (True,)
    service.validate_request.return_value["body"]["netuid"] = 72
    with pytest.raises(ServiceError, match="weight_state_scope_mismatch"):
        service.handle_weight_state({})
    service.store.has_recent_participation.assert_not_called()


@pytest.mark.parametrize(
    "response",
    [None, {}, {"eligible": 1}, {"eligible": "false"}, {"eligible": True, "extra": 1}],
)
def test_store_rejects_malformed_participation_response(response):
    store = ArenaStore(SimpleNamespace(rpc=Mock(return_value=response)))
    with pytest.raises(ArenaStoreError):
        store.has_recent_participation("finney", 71, "validator")


@pytest.mark.parametrize("eligible", [True, False])
def test_store_binds_lookup_to_scope_and_hotkey(eligible):
    rpc = Mock(return_value={"eligible": eligible})
    store = ArenaStore(SimpleNamespace(rpc=rpc))
    assert store.has_recent_participation("finney", 71, "validator") is eligible
    rpc.assert_called_once_with(
        "lab_arena_has_recent_participation_v1",
        {
            "p_network": "finney",
            "p_netuid": 71,
            "p_runner_hotkey": "validator",
        },
    )
