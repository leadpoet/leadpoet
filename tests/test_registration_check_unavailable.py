"""Registration read failures must not look like miner deregistration."""

import pytest
from fastapi import HTTPException

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


class _Payload:
    signature = "signature"

    def __init__(self, hotkey: str):
        self.miner_hotkey = hotkey

    def signed_payload(self):
        return b"payload"


async def _not_banned(_hotkey):
    return False, None


@pytest.mark.asyncio
async def test_signed_miner_gets_503_when_registration_is_unknown(monkeypatch):
    from gateway.research_lab import api

    async def unavailable(_hotkey):
        raise ChainRegistrationUnavailable("metagraph is unavailable")

    monkeypatch.setattr(api, "verify_hotkey_signature", lambda **_kwargs: True)
    monkeypatch.setattr(api, "is_hotkey_banned", _not_banned)
    monkeypatch.setattr(api, "chain_is_hotkey_registered", unavailable)

    with pytest.raises(HTTPException) as exc_info:
        await api._verify_signed_miner(_Payload("hk-registered"))

    assert exc_info.value.status_code == 503
    assert exc_info.value.headers == {"Retry-After": "30"}


@pytest.mark.asyncio
async def test_signed_miner_keeps_403_for_genuine_absence(monkeypatch):
    from gateway.research_lab import api

    async def absent(_hotkey):
        return False, None

    monkeypatch.setattr(api, "verify_hotkey_signature", lambda **_kwargs: True)
    monkeypatch.setattr(api, "is_hotkey_banned", _not_banned)
    monkeypatch.setattr(api, "chain_is_hotkey_registered", absent)

    with pytest.raises(HTTPException) as exc_info:
        await api._verify_signed_miner(_Payload("hk-absent"))

    assert exc_info.value.status_code == 403
