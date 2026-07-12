from types import SimpleNamespace

import pytest
from bittensor_wallet import Wallet

from leadpoet_canonical.hotkey_authority_v2 import signed_extrinsic_hash_v2
from validator_tee.host.enclave_hotkey_v2 import (
    AuthoritativeServeAxonContextV2,
    AuthoritativeSetWeightsContextV2,
    EnclaveBackedKeypairV2,
    EnclaveBackedWalletV2,
    EnclaveHotkeyV2Error,
    _weight_extrinsic_module,
    build_enclave_backed_wallet_v2,
)


HOTKEY = "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"
PUBLIC_KEY = "a6bfe69c29bf9e4db65c63ac6f6d1e23c252ca871744afb6edc5623d9bc39004"


class FakeClient:
    def __init__(self):
        self.requests = []

    def get_hotkey_state_v2(self):
        return {
            "validator_hotkey": HOTKEY,
            "hotkey_public_key": PUBLIC_KEY,
            "provisioned": True,
        }

    def sign_application_message_v2(self, message):
        self.requests.append(("application", message))
        if message == b"unknown":
            raise RuntimeError("Enclave error: application message domain is not authorized")
        return {
            "signature": "11" * 64,
            "receipt": {"receipt_hash": "sha256:" + "1" * 64},
        }

    def prepare_weight_commit_v2(self, request):
        self.requests.append(("commit", dict(request)))
        return {
            "commit_authorization_id": "sha256:" + "2" * 64,
            "commitment_hex": "abcd",
            "reveal_round": 99,
        }

    def sign_weight_extrinsic_v2(self, request):
        self.requests.append(("extrinsic", dict(request)))
        return {
            "signature": "33" * 64,
            "receipt": {"receipt_hash": "sha256:" + "4" * 64},
            "authorization": {"authorization_hash": "sha256:" + "5" * 64},
            "extrinsic_hash": signed_extrinsic_hash_v2(b"fake-extrinsic"),
        }

    def sign_serve_axon_extrinsic_v2(self, request):
        self.requests.append(("serve_axon", dict(request)))
        return {
            "signature": "44" * 64,
            "receipt": {"receipt_hash": "sha256:" + "8" * 64},
            "authorization": {
                "authorization_hash": "sha256:" + "9" * 64,
                "signed_message_hex": request["signature_payload_hex"],
                "call_data_hex": "07044700",
            },
        }


class FakeEra:
    def encode(self, era):
        self.era = dict(era)

    def birth(self, current):
        return int(current) - (int(current) % int(self.era["period"]))


class FakeRuntimeConfig:
    def create_scale_object(self, name):
        assert name == "Era"
        return FakeEra()


class FakeSubstrate:
    def __init__(self):
        self.runtime_config = FakeRuntimeConfig()
        self.original_calls = []
        self.create_signed_extrinsic = self.original_create_signed_extrinsic

    def init_runtime(self):
        return None

    def get_account_nonce(self, address):
        assert address == HOTKEY
        return 7

    def get_chain_finalised_head(self):
        return "0xfinal"

    def get_block_number(self, head):
        assert head == "0xfinal"
        return 123

    def get_block_hash(self, block_id):
        assert block_id == 120
        return "0x" + "bb" * 32

    def generate_signature_payload(self, **kwargs):
        assert kwargs["era"] == {"period": 8, "current": 123}
        assert kwargs["nonce"] == 7
        return SimpleNamespace(data=b"canonical-scale-payload")

    def original_create_signed_extrinsic(self, **kwargs):
        self.original_calls.append(dict(kwargs))
        return SimpleNamespace(
            data=SimpleNamespace(data=b"fake-extrinsic"),
            signature=kwargs["signature"],
        )


class FakeServeCall:
    def __init__(self):
        self.data = SimpleNamespace(data=bytes.fromhex("07044700"))
        self.value = {
            "call_module": "SubtensorModule",
            "call_function": "serve_axon",
            "call_args": {
                "version": 9012000,
                "ip": 2130706433,
                "port": 8093,
                "ip_type": 4,
                "netuid": 71,
                "hotkey": HOTKEY,
                "coldkey": HOTKEY,
                "protocol": 4,
                "placeholder1": 0,
                "placeholder2": 0,
            },
        }


def _wallet(client):
    return build_enclave_backed_wallet_v2(
        name="validator_72",
        hotkey_name="default",
        path="/nonexistent-public-wallet",
        client=client,
    )


def test_public_only_keypair_routes_only_application_domains_to_enclave():
    client = FakeClient()
    keypair = EnclaveBackedKeypairV2(ss58_address=HOTKEY, client=client)
    assert keypair.public_key.hex() == PUBLIC_KEY
    assert keypair.sign(b"1234567890") == bytes.fromhex("11" * 64)
    assert keypair.last_application_receipt_v2["receipt_hash"].startswith("sha256:")
    with pytest.raises(RuntimeError, match="not authorized"):
        keypair.sign(b"unknown")
    assert not hasattr(keypair, "private_key")


def test_enclave_wallet_is_a_real_wallet_subtype_without_hotkey_file_unlock():
    client = FakeClient()
    wallet = _wallet(client)
    assert isinstance(wallet, Wallet)
    assert isinstance(wallet, EnclaveBackedWalletV2)
    assert wallet.hotkey.ss58_address == HOTKEY
    assert wallet.unlock_hotkey() is None
    with pytest.raises(EnclaveHotkeyV2Error, match="mutation"):
        wallet.set_hotkey(None)


def test_wallet_creation_rejects_unprovisioned_or_mismatched_state():
    client = FakeClient()
    client.get_hotkey_state_v2 = lambda: {
        "validator_hotkey": HOTKEY,
        "hotkey_public_key": PUBLIC_KEY,
        "provisioned": False,
    }
    with pytest.raises(EnclaveHotkeyV2Error, match="not provisioned"):
        _wallet(client)

    client.get_hotkey_state_v2 = lambda: {
        "validator_hotkey": HOTKEY,
        "hotkey_public_key": "00" * 32,
        "provisioned": True,
    }
    with pytest.raises(EnclaveHotkeyV2Error, match="differ"):
        _wallet(client)


def test_set_weights_context_uses_enclave_commit_and_exact_sdk_payload(monkeypatch):
    mechanism = _weight_extrinsic_module()

    client = FakeClient()
    wallet = _wallet(client)
    substrate = FakeSubstrate()
    original_drand = mechanism.get_encrypted_commit
    original_signer = substrate.create_signed_extrinsic
    with AuthoritativeSetWeightsContextV2(
        substrate=substrate,
        wallet=wallet,
        weight_authorization_id="sha256:" + "6" * 64,
        weight_submission_event_hash="sha256:" + "7" * 64,
    ) as context:
        commitment, reveal_round = mechanism.get_encrypted_commit(
            uids=[0, 14],
            weights=[65535, 123],
            version_key=9012000,
            tempo=360,
            current_block=122,
            netuid=71,
            subnet_reveal_period_epochs=1,
            block_time=12.0,
            hotkey=wallet.hotkey.public_key,
        )
        assert commitment == bytes.fromhex("abcd")
        assert reveal_round == 99
        result = substrate.create_signed_extrinsic(
            call=object(),
            keypair=wallet.hotkey,
            era={"period": 8},
            nonce=None,
        )
        assert result.signature == bytes.fromhex("33" * 64)
        assert context.extrinsic_signature_results[0]["receipt"][
            "receipt_hash"
        ].startswith("sha256:")
    assert mechanism.get_encrypted_commit is original_drand
    assert substrate.create_signed_extrinsic == original_signer
    extrinsic_request = [item for item in client.requests if item[0] == "extrinsic"][0][1]
    assert extrinsic_request == {
        "commit_authorization_id": "sha256:" + "2" * 64,
        "era_current": 123,
        "nonce": 7,
        "block_hash": "bb" * 32,
        "signature_payload_hex": b"canonical-scale-payload".hex(),
    }
    assert substrate.original_calls[0]["signature"] == bytes.fromhex("33" * 64)


def test_set_weights_context_restores_sdk_functions_after_failure():
    mechanism = _weight_extrinsic_module()

    client = FakeClient()
    wallet = _wallet(client)
    substrate = FakeSubstrate()
    original_drand = mechanism.get_encrypted_commit
    original_signer = substrate.create_signed_extrinsic
    with pytest.raises(RuntimeError, match="fixture failure"):
        with AuthoritativeSetWeightsContextV2(
            substrate=substrate,
            wallet=wallet,
            weight_authorization_id="sha256:" + "6" * 64,
            weight_submission_event_hash="sha256:" + "7" * 64,
        ):
            raise RuntimeError("fixture failure")
    assert mechanism.get_encrypted_commit is original_drand
    assert substrate.create_signed_extrinsic == original_signer


def test_set_weights_context_fsync_callback_runs_before_sdk_can_broadcast():
    mechanism = _weight_extrinsic_module()
    client = FakeClient()
    wallet = _wallet(client)
    substrate = FakeSubstrate()
    observed = []

    def fail_journal(result):
        observed.append(dict(result))
        raise RuntimeError("journal fsync failed")

    with AuthoritativeSetWeightsContextV2(
        substrate=substrate,
        wallet=wallet,
        weight_authorization_id="sha256:" + "6" * 64,
        weight_submission_event_hash="sha256:" + "7" * 64,
        on_signed_extrinsic=fail_journal,
    ) as context:
        mechanism.get_encrypted_commit(
            uids=[0, 14],
            weights=[65535, 123],
            version_key=9012000,
            tempo=360,
            current_block=122,
            netuid=71,
            subnet_reveal_period_epochs=1,
            block_time=12.0,
            hotkey=wallet.hotkey.public_key,
        )
        with pytest.raises(RuntimeError, match="journal fsync failed"):
            substrate.create_signed_extrinsic(
                call=object(),
                keypair=wallet.hotkey,
                era={"period": 8},
                nonce=None,
            )
        assert len(observed) == 1
        assert context.extrinsic_signature_results == []


def test_set_weights_context_rejects_generic_chain_signing():
    client = FakeClient()
    wallet = _wallet(client)
    substrate = FakeSubstrate()
    with AuthoritativeSetWeightsContextV2(
        substrate=substrate,
        wallet=wallet,
        weight_authorization_id="sha256:" + "6" * 64,
        weight_submission_event_hash="sha256:" + "7" * 64,
    ):
        with pytest.raises(EnclaveHotkeyV2Error, match="no enclave commitment"):
            substrate.create_signed_extrinsic(
                call=object(),
                keypair=wallet.hotkey,
                era={"period": 8},
            )


def test_serve_axon_context_signs_only_exact_measured_call():
    client = FakeClient()
    wallet = _wallet(client)
    substrate = FakeSubstrate()
    original_signer = substrate.create_signed_extrinsic
    with AuthoritativeServeAxonContextV2(
        substrate=substrate,
        wallet=wallet,
    ) as context:
        result = substrate.create_signed_extrinsic(
            call=FakeServeCall(),
            keypair=wallet.hotkey,
            era={"period": 8},
            nonce=None,
        )
        assert result.signature == bytes.fromhex("44" * 64)
        assert context.extrinsic_signature_results[0]["receipt"][
            "receipt_hash"
        ].startswith("sha256:")
    assert substrate.create_signed_extrinsic == original_signer
    request = [item for item in client.requests if item[0] == "serve_axon"][0][1]
    assert request["netuid"] == 71
    assert request["port"] == 8093
    assert request["signature_payload_hex"] == b"canonical-scale-payload".hex()


def test_serve_axon_context_rejects_any_other_chain_call():
    client = FakeClient()
    wallet = _wallet(client)
    substrate = FakeSubstrate()
    call = FakeServeCall()
    call.value["call_function"] = "add_stake"
    with AuthoritativeServeAxonContextV2(substrate=substrate, wallet=wallet):
        with pytest.raises(EnclaveHotkeyV2Error, match="authorized serve_axon"):
            substrate.create_signed_extrinsic(
                call=call,
                keypair=wallet.hotkey,
                era={"period": 8},
            )
