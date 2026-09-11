from types import SimpleNamespace

import pytest

from gateway.utils.hotkey_roles import classify_hotkey_from_metagraph


def _metagraph():
    return SimpleNamespace(
        hotkeys=("primary", "stake-validator", "miner"),
        active=(True, False, True),
        validator_permit=(True, True, False),
        S=(1.0, 500_001.0, 900_000.0),
    )


def test_shared_gateway_role_classifier_preserves_mainnet_policy():
    metagraph = _metagraph()

    assert classify_hotkey_from_metagraph(
        "primary", metagraph, network_name="finney"
    ) == (True, "validator")
    assert classify_hotkey_from_metagraph(
        "stake-validator", metagraph, network_name="finney"
    ) == (True, "validator")
    assert classify_hotkey_from_metagraph(
        "miner", metagraph, network_name="finney"
    ) == (True, "miner")
    assert classify_hotkey_from_metagraph(
        "absent", metagraph, network_name="finney"
    ) == (False, None)


def test_shared_gateway_role_classifier_accepts_inactive_testnet_permit():
    metagraph = SimpleNamespace(
        hotkeys=("active", "permitted", "miner"),
        active=(True, False, False),
        validator_permit=(False, True, False),
        S=(0.0, 0.018, 900_000.0),
    )

    assert classify_hotkey_from_metagraph(
        "active", metagraph, network_name="test"
    ) == (True, "validator")
    assert classify_hotkey_from_metagraph(
        "permitted", metagraph, network_name="test"
    ) == (True, "validator")
    assert classify_hotkey_from_metagraph(
        "miner", metagraph, network_name="test"
    ) == (True, "miner")
    assert classify_hotkey_from_metagraph(
        "permitted", metagraph, network_name="finney"
    ) == (True, "miner")


@pytest.mark.parametrize("field", ["active", "validator_permit", "S"])
def test_shared_gateway_role_classifier_rejects_missing_role_field(field):
    metagraph = _metagraph()
    setattr(metagraph, field, None)

    with pytest.raises((TypeError, ValueError)):
        classify_hotkey_from_metagraph(
            "primary", metagraph, network_name="finney"
        )
