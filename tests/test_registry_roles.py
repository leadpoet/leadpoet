import asyncio
from types import SimpleNamespace

import pytest

from gateway.utils.hotkey_roles import (
    MIN_VALIDATOR_STAKE_WEIGHT,
    ValidatorIneligible,
    classify_hotkey_from_metagraph,
    classify_hotkey_role,
    permitted_validator_uid,
    validator_hotkeys_from_metagraph,
    validator_uid,
)


def _metagraph():
    return SimpleNamespace(
        netuid=71,
        hotkeys=("below", "boundary", "above", "no-permit"),
        active=(True, False, False, True),
        validator_permit=(True, True, True, False),
        S=(74_999.0, 75_000.0, 75_001.0, 900_000.0),
    )


def test_shared_gateway_role_classifier_uses_permit_without_stake():
    metagraph = _metagraph()

    assert classify_hotkey_from_metagraph(
        "below", metagraph, network_name="finney"
    ) == (True, "validator")
    assert classify_hotkey_from_metagraph(
        "boundary", metagraph, network_name="finney"
    ) == (True, "validator")
    assert classify_hotkey_from_metagraph(
        "above", metagraph, network_name="finney"
    ) == (True, "validator")
    assert classify_hotkey_from_metagraph(
        "absent", metagraph, network_name="finney"
    ) == (False, None)
    assert validator_hotkeys_from_metagraph(
        metagraph, network_name="finney"
    ) == ["below", "boundary", "above"]
    assert validator_hotkeys_from_metagraph(
        metagraph, network_name="finney", require_stake=True
    ) == ["boundary", "above"]


@pytest.mark.parametrize(
    "stake, eligible",
    [
        (MIN_VALIDATOR_STAKE_WEIGHT - 1e-9, False),
        (MIN_VALIDATOR_STAKE_WEIGHT, True),
        (MIN_VALIDATOR_STAKE_WEIGHT + 1e-9, True),
    ],
)
def test_mainnet_stake_boundary_is_inclusive_and_activity_independent(stake, eligible):
    metagraph = SimpleNamespace(
        netuid=71,
        hotkeys=("validator",),
        active=None,
        validator_permit=(True,),
        S=(stake,),
    )

    assert (
        classify_hotkey_role(
            None,
            True,
            stake,
            network_name="finney",
            require_stake=True,
        )[0]
        == ("validator" if eligible else "miner")
    )
    if eligible:
        assert validator_uid(
            metagraph,
            "validator",
            netuid=71,
            network_name="finney",
        ) == 0
    else:
        with pytest.raises(
            ValidatorIneligible, match="runner_stake_below_minimum"
        ):
            validator_uid(
                metagraph,
                "validator",
                netuid=71,
                network_name="finney",
            )


def test_shared_gateway_role_classifier_accepts_inactive_testnet_permit():
    metagraph = SimpleNamespace(
        netuid=401,
        hotkeys=("active", "permitted", "miner"),
        active=(True, False, False),
        validator_permit=(False, True, False),
        S=None,
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
    assert validator_hotkeys_from_metagraph(
        metagraph, network_name="test"
    ) == ["active", "permitted"]
    assert validator_hotkeys_from_metagraph(
        metagraph, network_name="test", require_stake=True
    ) == ["active", "permitted"]


def test_scalar_mainnet_role_defaults_to_permit_only():
    assert classify_hotkey_role(
        None, True, None, network_name="finney"
    ) == ("validator", "permit=True")
    assert classify_hotkey_role(
        None, False, None, network_name="finney"
    ) == ("miner", "permit=False")


def test_shared_policy_accepts_bittensor_105_numpy_field_types():
    import numpy as np

    metagraph = SimpleNamespace(
        netuid=np.int64(401),
        hotkeys=("active", "permitted", "miner"),
        active=np.array([1, 0, 0], dtype=np.int64),
        validator_permit=np.array([False, True, False], dtype=bool),
        S=np.array([0.018, 75_000, 1_000_000], dtype=np.float32),
    )
    assert validator_hotkeys_from_metagraph(
        metagraph, network_name="test", require_stake=True
    ) == [
        "active", "permitted",
    ]
    assert validator_uid(metagraph, "active", netuid=401, network_name="test") == 0
    assert classify_hotkey_role(np.int64(1), np.bool_(False), np.float32(0.018),
                               network_name="test")[0] == "validator"
    assert validator_hotkeys_from_metagraph(
        metagraph, network_name="finney", require_stake=True
    ) == ["permitted"]


@pytest.mark.parametrize("active", [-1, 2, 0.0, "1", "false", None])
def test_testnet_activity_rejects_non_boolean_sdk_values(active):
    with pytest.raises(ValueError, match="active"):
        classify_hotkey_role(active, True, 0, network_name="test")


def test_identity_only_uid_keeps_permit_and_testnet_rules_without_mainnet_stake():
    mainnet = SimpleNamespace(
        netuid=71,
        hotkeys=("permitted", "miner"),
        validator_permit=(True, False),
        active=None,
        S=None,
    )
    assert validator_uid(
        mainnet,
        "permitted",
        netuid=71,
        network_name="finney",
        require_stake=False,
    ) == 0
    with pytest.raises(ValidatorIneligible, match="runner_validator_required"):
        validator_uid(
            mainnet,
            "miner",
            netuid=71,
            network_name="finney",
            require_stake=False,
        )

    testnet = SimpleNamespace(
        netuid=401,
        hotkeys=("active",),
        validator_permit=(False,),
        active=(True,),
        S=None,
    )
    assert validator_uid(
        testnet,
        "active",
        netuid=401,
        network_name="test",
        require_stake=False,
    ) == 0
    with pytest.raises(ValidatorIneligible, match="runner_validator_required"):
        permitted_validator_uid(testnet, "active", netuid=401)


def test_permitted_validator_uid_is_strict_and_does_not_read_stake_or_activity():
    metagraph = SimpleNamespace(
        netuid=401,
        hotkeys=("active", "permitted"),
        validator_permit=(False, True),
        active=None,
        S=None,
    )
    assert permitted_validator_uid(metagraph, "permitted", netuid=401) == 1
    with pytest.raises(ValidatorIneligible, match="runner_validator_required"):
        permitted_validator_uid(metagraph, "active", netuid=401)
    with pytest.raises(ValidatorIneligible, match="runner_hotkey_unregistered"):
        permitted_validator_uid(metagraph, "absent", netuid=401)


@pytest.mark.parametrize("field", ["validator_permit", "S"])
def test_mainnet_classifier_rejects_missing_required_field(field):
    metagraph = _metagraph()
    setattr(metagraph, field, None)

    with pytest.raises((TypeError, ValueError)):
        classify_hotkey_from_metagraph(
            "boundary", metagraph, network_name="finney", require_stake=True
        )


@pytest.mark.parametrize(
    "patch",
    [
        {"netuid": 72},
        {"netuid": True},
        {"hotkeys": ("validator", "validator")},
        {"validator_permit": ()},
        {"validator_permit": (1,)},
        {"S": ()},
        {"S": (True,)},
        {"S": ("75000",)},
        {"S": (-1,)},
        {"S": (float("nan"),)},
        {"S": (float("inf"),)},
    ],
)
def test_validator_uid_fails_closed_for_malformed_mainnet_snapshot(patch):
    metagraph = SimpleNamespace(
        netuid=71,
        hotkeys=("validator",),
        active=None,
        validator_permit=(True,),
        S=(75_000,),
    )
    metagraph.__dict__.update(patch)

    with pytest.raises(ValueError):
        validator_uid(
            metagraph,
            "validator",
            netuid=71,
            network_name="finney",
        )


def test_validator_uid_reports_stable_ineligibility_codes():
    metagraph = _metagraph()
    with pytest.raises(ValidatorIneligible, match="runner_hotkey_unregistered"):
        validator_uid(
            metagraph, "absent", netuid=71, network_name="finney"
        )
    with pytest.raises(ValidatorIneligible, match="runner_validator_required"):
        validator_uid(
            metagraph, "no-permit", netuid=71, network_name="finney"
        )


@pytest.mark.parametrize(
    "network, expected",
    [
        ("finney", ["below", "boundary", "above"]),
        ("test", ["below", "boundary", "above", "no-permit"]),
    ],
)
def test_registry_counts_and_assignment_set_match_shared_policy(
    monkeypatch, network, expected
):
    from gateway import config
    from gateway.utils import assignment, registry

    metagraph = _metagraph()
    monkeypatch.setattr(config, "BITTENSOR_NETWORK", network)
    monkeypatch.setattr(assignment, "BITTENSOR_NETWORK", network)
    monkeypatch.setattr(registry, "get_metagraph", lambda: metagraph)

    async def get_metagraph_async(*, cache_epoch_id=None):
        return metagraph

    monkeypatch.setattr(registry, "get_metagraph_async", get_metagraph_async)

    assert registry.get_validator_count() == len(expected)
    assert registry.get_miner_count() == len(metagraph.hotkeys) - len(expected)
    assert asyncio.run(registry.get_validator_count_async()) == len(expected)
    assert asyncio.run(registry.get_miner_count_async()) == (
        len(metagraph.hotkeys) - len(expected)
    )
    assert asyncio.run(
        assignment.get_validator_set(1, fail_closed=True)
    ) == expected
