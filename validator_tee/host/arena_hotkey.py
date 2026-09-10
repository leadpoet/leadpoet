"""Minimal public Arena keypair backed by narrow protected signing RPCs."""

from __future__ import annotations

from typing import Any

from bittensor_wallet import Keypair


class ArenaProtectedHotkeyError(RuntimeError):
    pass


def _message_bytes(value: Any) -> bytes:
    if hasattr(value, "data"):
        value = value.data
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    try:
        return bytes(value)
    except Exception as exc:
        raise ArenaProtectedHotkeyError("Arena signing input is not bytes") from exc


class ArenaProtectedKeypair(Keypair):
    """Public keypair that exposes only the Arena application signer."""

    def __new__(cls, *, ss58_address: str, client: Any, state: Any = None):
        del client, state
        return super().__new__(cls, ss58_address=str(ss58_address))

    def __init__(self, *, ss58_address: str, client: Any, state: Any = None) -> None:
        self._arena_client = client
        self._arena_state = dict(state or {})

    @property
    def arena_state(self):
        return dict(self._arena_state)

    def sign(self, data: Any) -> bytes:
        result = self._arena_client.sign_arena_application_v1(_message_bytes(data))
        try:
            signature = bytes.fromhex(str(result["signature"]).removeprefix("0x"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ArenaProtectedHotkeyError("protected Arena signature is invalid") from exc
        if len(signature) != 64 or result.get("scope") not in {
            "validator.arena_claim.v1", "validator.arena_complete.v1",
        }:
            raise ArenaProtectedHotkeyError("protected Arena signature scope is invalid")
        return signature


def build_arena_protected_keypair(*, client: Any) -> ArenaProtectedKeypair:
    state = client.get_arena_hotkey_state_v1()
    if state.get("provisioned") is not True:
        raise ArenaProtectedHotkeyError("protected Arena hotkey is not provisioned")
    keypair = ArenaProtectedKeypair(
        ss58_address=str(state.get("validator_hotkey") or ""), client=client, state=state
    )
    if keypair.public_key.hex() != str(state.get("hotkey_public_key") or "").lower():
        raise ArenaProtectedHotkeyError("protected Arena hotkey identity differs")
    return keypair
