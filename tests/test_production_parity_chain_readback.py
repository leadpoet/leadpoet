from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "read_production_parity_chain_state.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "production_parity_chain_readback", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Substrate:
    def get_block_hash(self, block: int) -> str:
        assert block == 0
        return "0x" + "1" * 64

    def get_chain_finalised_head(self) -> str:
        return "0x" + "2" * 64

    def get_block_number(self, block_hash: str) -> int:
        assert block_hash == "0x" + "2" * 64
        return 987

    def query(self, *, module, storage_function, params, block_hash):
        assert module == "SubtensorModule"
        assert block_hash == "0x" + "2" * 64
        if storage_function == "LastUpdate":
            assert params == [1]
            return SimpleNamespace(value=[0, 900, 901, 902])
        if storage_function == "Uids":
            assert params[0] == 1
            return SimpleNamespace(
                value={"4" * 48: 1, "5" * 48: 2, "6" * 48: 3}[params[1]]
            )
        assert storage_function == "Weights"
        return SimpleNamespace(value=[(0, 32768), (7, 32767)])


class _Subtensor:
    def __init__(self, *, network: str) -> None:
        assert network == "wss://test.finney.opentensor.ai:443"
        self.substrate = _Substrate()
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_chain_readback_uses_finalized_testnet_storage() -> None:
    module = _module()
    created = []

    def factory(*, network):
        value = _Subtensor(network=network)
        created.append(value)
        return value

    result = module.read_finalized_state(
        endpoint="wss://test.finney.opentensor.ai:443",
        netuid=1,
        hotkeys=["4" * 48, "5" * 48, "6" * 48],
        expected_genesis_hash="0x" + "1" * 64,
        subtensor_factory=factory,
    )

    assert result["finalized_block"] == 987
    assert [item["last_update"] for item in result["validators"]] == [
        900,
        901,
        902,
    ]
    assert all(item["weights"] == [[0, 32768], [7, 32767]] for item in result["validators"])
    assert result["readback_hash"].startswith("sha256:")
    assert created[0].closed is True


def test_chain_readback_rejects_non_testnet_authority() -> None:
    module = _module()

    with pytest.raises(
        module.ProductionParityChainReadbackError, match="official testnet"
    ):
        module.read_finalized_state(
            endpoint="wss://finney.opentensor.ai:443",
            netuid=1,
            hotkeys=["4" * 48, "5" * 48, "6" * 48],
            expected_genesis_hash="0x" + "1" * 64,
            subtensor_factory=_Subtensor,
        )
