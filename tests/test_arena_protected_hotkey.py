import pytest

from validator_tee.host.arena_hotkey import (
    ArenaProtectedHotkeyError,
    build_arena_protected_keypair,
)


class _Client:
    def __init__(self):
        self.messages = []

    def get_arena_hotkey_state_v1(self):
        # Alice's standard Substrate development identity.
        return {
            "provisioned": True,
            "validator_hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "hotkey_public_key": "d43593c715fdd31c61141abd04a99fd6822c8558854ccde39a5684e7a56da27d",
            "policy_hash": "sha256:" + "1" * 64,
            "policy": {},
        }

    def sign_arena_application_v1(self, message):
        self.messages.append(message)
        return {"signature": "ab" * 64, "scope": "validator.arena_claim.v1"}


def test_minimal_arena_keypair_uses_protected_scoped_signer():
    client = _Client()
    keypair = build_arena_protected_keypair(client=client)
    assert keypair.sign(b"canonical claim") == bytes.fromhex("ab" * 64)
    assert client.messages == [b"canonical claim"]


def test_minimal_arena_keypair_rejects_unexpected_scope():
    client = _Client()
    client.sign_arena_application_v1 = lambda _message: {
        "signature": "ab" * 64, "scope": "validator.weights.v2"
    }
    keypair = build_arena_protected_keypair(client=client)
    with pytest.raises(ArenaProtectedHotkeyError, match="scope"):
        keypair.sign(b"not an Arena request")
