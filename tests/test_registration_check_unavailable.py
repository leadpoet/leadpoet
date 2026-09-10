"""Registration read failures must not look like miner deregistration."""

import pytest

from gateway.qualification.utils import chain as chain_utils
from gateway.qualification.utils.chain import ChainRegistrationUnavailable


class _Metagraph:
    hotkeys = ["hk-registered"]
    validator_permit = [False]


async def _raise_unreadable():
    raise RuntimeError("metagraph is unavailable")


@pytest.mark.asyncio
async def test_strict_registration_check_raises_when_metagraph_is_unreadable(
    monkeypatch,
):
    monkeypatch.setattr(chain_utils, "get_metagraph", _raise_unreadable)

    with pytest.raises(ChainRegistrationUnavailable):
        await chain_utils.check_hotkey_registration("hk-registered")


@pytest.mark.asyncio
async def test_strict_registration_check_keeps_genuine_absence(monkeypatch):
    async def get_metagraph():
        return _Metagraph()

    monkeypatch.setattr(chain_utils, "get_metagraph", get_metagraph)

    assert await chain_utils.check_hotkey_registration("hk-registered") == (
        True,
        "miner",
    )
    assert await chain_utils.check_hotkey_registration("hk-absent") == (
        False,
        None,
    )


@pytest.mark.asyncio
async def test_compatibility_registration_check_still_returns_false(monkeypatch):
    monkeypatch.setattr(chain_utils, "get_metagraph", _raise_unreadable)

    assert await chain_utils.is_hotkey_registered("hk-registered") == (False, None)
