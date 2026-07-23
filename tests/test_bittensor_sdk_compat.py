from types import SimpleNamespace

import pytest

from Leadpoet.utils.bittensor_sdk import weight_hyperparameters_compat


class _Substrate:
    def __init__(self):
        self.queries = []

    def get_block_hash(self, block):
        return f"hash-{block}"

    def query(self, **kwargs):
        self.queries.append(kwargs)
        values = {"Tempo": 360, "RevealPeriodEpochs": 1}
        return SimpleNamespace(value=values[kwargs["storage_function"]])


class _Subtensor:
    def __init__(self):
        self.substrate = _Substrate()

    def get_subnet_hyperparameters(self, netuid, block=None):
        return ("native", netuid, block)


def test_v9_weight_hyperparameters_use_exact_chain_storage_and_restore():
    subtensor = _Subtensor()
    original = subtensor.get_subnet_hyperparameters

    with weight_hyperparameters_compat(
        subtensor,
        netuid=71,
        sdk_version="9.12.2",
    ):
        result = subtensor.get_subnet_hyperparameters(71, block=8_681_018)
        assert result.tempo == 360
        assert result.commit_reveal_period == 1

    assert subtensor.get_subnet_hyperparameters == original
    assert subtensor.substrate.queries == [
        {
            "module": "SubtensorModule",
            "storage_function": "Tempo",
            "params": [71],
            "block_hash": "hash-8681018",
        },
        {
            "module": "SubtensorModule",
            "storage_function": "RevealPeriodEpochs",
            "params": [71],
            "block_hash": "hash-8681018",
        },
    ]


def test_v10_weight_hyperparameters_leave_native_sdk_unchanged():
    subtensor = _Subtensor()
    original = subtensor.get_subnet_hyperparameters

    with weight_hyperparameters_compat(
        subtensor,
        netuid=71,
        sdk_version="10.5.0",
    ):
        assert subtensor.get_subnet_hyperparameters(71, block=10) == (
            "native",
            71,
            10,
        )

    assert subtensor.get_subnet_hyperparameters == original
    assert subtensor.substrate.queries == []


def test_v9_weight_hyperparameters_fail_closed_on_invalid_storage():
    subtensor = _Subtensor()
    subtensor.substrate.query = lambda **_kwargs: SimpleNamespace(value=0)

    with weight_hyperparameters_compat(
        subtensor,
        netuid=71,
        sdk_version="9.12.2",
    ):
        with pytest.raises(RuntimeError, match="chain storage are invalid"):
            subtensor.get_subnet_hyperparameters(71, block=8_681_018)
