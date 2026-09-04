"""
An unreadable metagraph must not be reported to a miner as deregistration.

Regression cover for the 2026-09-04 lockout: every metagraph refresh failed,
the registration check reported that as "not registered", and the whole subnet
was refused with 403 for ninety minutes.
"""

import pytest
from fastapi import HTTPException

from gateway.qualification.utils import chain as chain_utils
from gateway.qualification.utils.chain import ChainRegistrationUnavailable


class _Metagraph:
    hotkeys = ["hk-registered"]
    validator_permit = [False]


async def _raise_unreadable():
    raise RuntimeError("Failed to fetch metagraph and no cache available")


@pytest.mark.asyncio
async def test_check_raises_when_metagraph_unreadable(monkeypatch):
    monkeypatch.setattr(chain_utils, "get_metagraph", _raise_unreadable)
    with pytest.raises(ChainRegistrationUnavailable):
        await chain_utils.check_hotkey_registration("hk-registered")


@pytest.mark.asyncio
async def test_check_still_reports_a_genuinely_absent_hotkey(monkeypatch):
    async def _ok():
        return _Metagraph()

    monkeypatch.setattr(chain_utils, "get_metagraph", _ok)
    assert await chain_utils.check_hotkey_registration("hk-registered") == (True, "miner")
    assert await chain_utils.check_hotkey_registration("hk-absent") == (False, None)


@pytest.mark.asyncio
async def test_legacy_wrapper_keeps_its_swallowing_contract(monkeypatch):
    monkeypatch.setattr(chain_utils, "get_metagraph", _raise_unreadable)
    assert await chain_utils.is_hotkey_registered("hk-registered") == (False, None)


@pytest.mark.asyncio
async def test_signed_miner_gets_503_not_403_when_registration_is_unknown(monkeypatch):
    from gateway.research_lab import api as research_lab_api

    class _Payload:
        miner_hotkey = "hk-registered"
        signature = "sig"

        def signed_payload(self):
            return b"payload"

    async def _not_banned(_hotkey):
        return False, None

    async def _unavailable(_hotkey):
        raise ChainRegistrationUnavailable("metagraph unreadable")

    monkeypatch.setattr(research_lab_api, "verify_hotkey_signature", lambda **_: True)
    monkeypatch.setattr(research_lab_api, "is_hotkey_banned", _not_banned)
    monkeypatch.setattr(research_lab_api, "chain_check_hotkey_registration", _unavailable)

    with pytest.raises(HTTPException) as excinfo:
        await research_lab_api._verify_signed_miner(_Payload())

    assert excinfo.value.status_code == 503
    assert excinfo.value.headers.get("Retry-After") == "30"


@pytest.mark.asyncio
async def test_signed_miner_still_gets_403_when_genuinely_unregistered(monkeypatch):
    from gateway.research_lab import api as research_lab_api

    class _Payload:
        miner_hotkey = "hk-absent"
        signature = "sig"

        def signed_payload(self):
            return b"payload"

    async def _not_banned(_hotkey):
        return False, None

    async def _absent(_hotkey):
        return False, None

    monkeypatch.setattr(research_lab_api, "verify_hotkey_signature", lambda **_: True)
    monkeypatch.setattr(research_lab_api, "is_hotkey_banned", _not_banned)
    monkeypatch.setattr(research_lab_api, "chain_check_hotkey_registration", _absent)

    with pytest.raises(HTTPException) as excinfo:
        await research_lab_api._verify_signed_miner(_Payload())

    assert excinfo.value.status_code == 403
