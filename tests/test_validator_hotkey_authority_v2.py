import base64
from datetime import datetime, timezone
import hashlib

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding

from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_canonical.binding import create_binding_message
from leadpoet_canonical.hotkey_authority_v2 import (
    CHAIN_SIGNING_PROFILE_SCHEMA_VERSION,
    subnet_epoch_candidate_authorization_message_v1,
)
from validator_tee.enclave import hotkey_authority_v2 as module
from validator_tee.enclave.hotkey_authority_v2 import (
    HOTKEY_ENCRYPTION_ALGORITHM,
    HOTKEY_RECIPIENT_SCHEMA_VERSION,
    ValidatorHotkeyAuthorityV2,
    ValidatorHotkeyAuthorityV2Error,
)


HOTKEY = "5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"
SEED = b"s" * 32
HOTKEY_PUBLIC = b"p" * 32
HOTKEY_SECRET = b"k" * 64
NOW = datetime(2026, 7, 10, 20, 0, tzinfo=timezone.utc)


class _Sr25519:
    def pair_from_seed(self, seed):
        assert seed == SEED
        return HOTKEY_PUBLIC, HOTKEY_SECRET

    def sign(self, keypair, message):
        assert keypair == (HOTKEY_PUBLIC, HOTKEY_SECRET)
        return hashlib.sha512(HOTKEY_PUBLIC + bytes(message)).digest()

    def verify(self, signature, message, public_key):
        return bytes(signature) == hashlib.sha512(
            bytes(public_key) + bytes(message)
        ).digest()


class _Drand:
    def __init__(self):
        self.calls = []

    def generate_commit(self, **kwargs):
        self.calls.append(kwargs)
        return b"timelocked-commitment" * 20, 998877


def _profile():
    return {
        "schema_version": CHAIN_SIGNING_PROFILE_SCHEMA_VERSION,
        "network": "finney",
        "chain_endpoint": "wss://entrypoint-finney.opentensor.ai:443",
        "genesis_hash": "0" * 64,
        "spec_version": 432,
        "transaction_version": 1,
        "version_key": 10005000,
        "commit_call_index": "0776",
        "serve_axon_call_index": "0704",
        "commit_reveal_version": 4,
        "mechid": 0,
        "tempo": 360,
        "subnet_reveal_period_epochs": 1,
        "block_time_millis": 12000,
        "max_snapshot_block_drift": 64,
        "extrinsic_period": 8,
        "signed_extensions": [
            "CheckMortality",
            "CheckNonce",
            "ChargeTransactionPayment",
            "CheckMetadataHash",
            "CheckSpecVersion",
            "CheckTxVersion",
            "CheckGenesis",
            "CheckMortalityAdditionalSigned",
            "CheckMetadataHashAdditionalSigned",
        ],
    }


def _boot(signing_key):
    public = signing_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    return {
        "boot_identity_hash": "sha256:" + "1" * 64,
        "commit_sha": "2" * 40,
        "pcr0": "3" * 96,
        "build_manifest_hash": "sha256:" + "4" * 64,
        "dependency_lock_hash": "sha256:" + "5" * 64,
        "config_hash": "sha256:" + "6" * 64,
        "signing_pubkey": public,
    }


def _weight_response(boot, signing_key):
    result = {
        "netuid": 71,
        "epoch_id": 23860,
        "block": 8596805,
        "weights_hash": "7" * 64,
        "sparse_uids": [0, 14, 213],
        "sparse_weights_u16": [65535, 3210, 2600],
    }
    root = {
        "receipt_hash": "sha256:" + "8" * 64,
        "role": "validator_weights",
        "purpose": "validator.weights.computed.v2",
        "output_root": sha256_json(result),
        "boot_identity_hash": boot["boot_identity_hash"],
        "enclave_pubkey": boot["signing_pubkey"],
    }
    return {
        "weight_snapshot": {
            "validator_hotkey": HOTKEY,
            "snapshot_hash": "sha256:" + "9" * 64,
        },
        "weight_result": result,
        "weights_signature": signing_key.sign(bytes.fromhex(result["weights_hash"])).hex(),
        "receipt_graph": {
            "root_receipt_hash": root["receipt_hash"],
            "receipts": [root],
        },
        "boot_identity": boot,
    }


def _authority(monkeypatch):
    signing_key = ed25519.Ed25519PrivateKey.generate()
    boot = _boot(signing_key)
    drand = _Drand()
    monkeypatch.setattr(module, "validate_receipt_graph", lambda _graph: None)
    monkeypatch.setattr(
        module,
        "validate_weight_snapshot_v2",
        lambda _snapshot: _weight_response(boot, signing_key)["weight_result"],
    )
    observed_attestation = []

    def attest(*, user_data, public_key):
        observed_attestation.append((user_data, public_key))
        return b"nitro-hotkey-recipient"

    authority = ValidatorHotkeyAuthorityV2(
        boot_identity_supplier=lambda: boot,
        validator_hotkey=HOTKEY,
        hotkey_public_key_hex=HOTKEY_PUBLIC.hex(),
        chain_profile=_profile(),
        sign_receipt_digest=signing_key.sign,
        attestation_supplier=attest,
        drand_backend=drand,
        sr25519_backend=_Sr25519(),
        clock=lambda: NOW,
    )
    return authority, boot, signing_key, drand, observed_attestation


def _provision(authority):
    request = authority.recipient_request()
    public_key = serialization.load_der_public_key(
        base64.b64decode(request["recipient_public_key_der_b64"])
    )
    ciphertext = public_key.encrypt(
        SEED,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return authority.provision_seed(
        ciphertext_for_recipient_b64=base64.b64encode(ciphertext).decode()
    )


def test_hotkey_seed_is_unsealed_only_to_attested_recipient(monkeypatch):
    authority, _boot_value, _signing_key, _drand, observed = _authority(monkeypatch)
    request = authority.recipient_request()
    assert request["schema_version"] == HOTKEY_RECIPIENT_SCHEMA_VERSION
    assert request["key_encryption_algorithm"] == HOTKEY_ENCRYPTION_ALGORITHM
    assert base64.b64decode(request["attestation_document_b64"]) == b"nitro-hotkey-recipient"
    assert observed[0][1] == base64.b64decode(
        request["recipient_public_key_der_b64"]
    )
    state = _provision(authority)
    assert state["provisioned"] is True
    assert state["hotkey_public_key"] == HOTKEY_PUBLIC.hex()
    with pytest.raises(ValidatorHotkeyAuthorityV2Error, match="already provisioned"):
        _provision(authority)


def test_wrong_seed_cannot_provision_validator_hotkey(monkeypatch):
    authority, *_rest = _authority(monkeypatch)
    authority._sr25519 = type(
        "WrongSeedBackend",
        (),
        {
            "pair_from_seed": lambda self, seed: (b"x" * 32, b"y" * 64),
            "sign": lambda self, keypair, message: b"z" * 64,
        },
    )()
    request = authority.recipient_request()
    public_key = serialization.load_der_public_key(
        base64.b64decode(request["recipient_public_key_der_b64"])
    )
    ciphertext = public_key.encrypt(
        b"x" * 32,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    with pytest.raises(
        ValidatorHotkeyAuthorityV2Error,
        match="does not match validator hotkey",
    ):
        authority.provision_seed(
            ciphertext_for_recipient_b64=base64.b64encode(ciphertext).decode()
        )


def test_gateway_binding_signature_requires_exact_weight_parent_and_boot(monkeypatch):
    authority, boot, signing_key, _drand, _observed = _authority(monkeypatch)
    _provision(authority)
    response = _weight_response(boot, signing_key)
    weight_id = authority.register_weight_result(response)
    root = response["receipt_graph"]["root_receipt_hash"]
    message = create_binding_message(
        netuid=71,
        chain=_profile()["chain_endpoint"],
        enclave_pubkey=boot["signing_pubkey"],
        validator_code_hash=boot["build_manifest_hash"],
        version=boot["commit_sha"],
    )
    result = authority.sign_application_message(
        message_hex=message.encode().hex(),
        parent_receipt_hash=root,
    )
    assert result["purpose"] == "validator.gateway_binding.v2"
    assert result["receipt"]["parent_receipt_hashes"] == [root]
    assert len(bytes.fromhex(result["signature"])) == 64
    assert weight_id.startswith("sha256:")

    with pytest.raises(ValidatorHotkeyAuthorityV2Error, match="parent"):
        authority.sign_application_message(message_hex=message.encode().hex())


def test_application_signer_rejects_unknown_message_or_unwanted_parent(monkeypatch):
    authority, *_rest = _authority(monkeypatch)
    _provision(authority)
    with pytest.raises(Exception, match="not authorized"):
        authority.sign_application_message(
            message_hex=b"generic signing oracle".hex()
        )
    with pytest.raises(ValidatorHotkeyAuthorityV2Error, match="parent is not allowed"):
        authority.sign_application_message(
            message_hex=("GET_EPOCH_LEADS:23860:%s" % HOTKEY).encode().hex(),
            parent_receipt_hash="sha256:" + "a" * 64,
        )


def test_application_signer_authorizes_exact_subnet_epoch_candidate(monkeypatch):
    authority, *_rest = _authority(monkeypatch)
    _provision(authority)
    message = subnet_epoch_candidate_authorization_message_v1(
        validator_hotkey=HOTKEY,
        candidate_payload_hash="sha256:" + "7" * 64,
    )

    result = authority.sign_application_message(
        message_hex=message.encode("utf-8").hex()
    )

    assert result["purpose"] == "validator.subnet_epoch_candidate.v2"
    assert result["receipt"]["purpose"] == "validator.hotkey_signature.v2"
    assert result["receipt"]["parent_receipt_hashes"] == []
    assert len(bytes.fromhex(result["signature"])) == 64


def _prepare(monkeypatch):
    authority, boot, signing_key, drand, _observed = _authority(monkeypatch)
    _provision(authority)
    response = _weight_response(boot, signing_key)
    weight_id = authority.register_weight_result(response)
    prepared = authority.prepare_weight_commit(
        weight_authorization_id=weight_id,
        weight_submission_event_hash="sha256:" + "a" * 64,
        uids=response["weight_result"]["sparse_uids"],
        weights_u16=response["weight_result"]["sparse_weights_u16"],
        version_key=10005000,
        last_epoch_block=response["weight_result"]["block"] - 126,
        pending_epoch_at=0,
        subnet_epoch_index=23860,
        tempo=360,
        blocks_since_last_step=127,
        current_block=response["weight_result"]["block"] + 1,
        subnet_reveal_period_epochs=1,
        block_time=12.0,
        hotkey_public_key_hex=HOTKEY_PUBLIC.hex(),
    )
    return authority, response, prepared, drand


def test_commit_and_extrinsic_signature_bind_exact_enclave_vector(monkeypatch):
    authority, response, prepared, drand = _prepare(monkeypatch)
    assert drand.calls[0]["uids"] == response["weight_result"]["sparse_uids"]
    assert drand.calls[0]["weights_u16"] == response["weight_result"][
        "sparse_weights_u16"
    ]
    assert drand.calls[0]["last_epoch_block"] == 8596679
    assert drand.calls[0]["pending_epoch_at"] == 0
    assert drand.calls[0]["subnet_epoch_index"] == 23860
    assert drand.calls[0]["blocks_since_last_step"] == 127
    # Build once with a placeholder payload to obtain the canonical bytes the
    # SDK must present. The authority independently rebuilds this document.
    from leadpoet_canonical.hotkey_authority_v2 import build_weight_extrinsic_authorization_v2

    expected = build_weight_extrinsic_authorization_v2(
        profile=_profile(),
        validator_hotkey=HOTKEY,
        hotkey_public_key_hex=HOTKEY_PUBLIC.hex(),
        epoch_id=response["weight_result"]["epoch_id"],
        netuid=71,
        weight_receipt_hash=response["receipt_graph"]["root_receipt_hash"],
        weight_submission_event_hash="sha256:" + "a" * 64,
        weights_hash=response["weight_result"]["weights_hash"],
        sparse_uids=response["weight_result"]["sparse_uids"],
        sparse_weights_u16=response["weight_result"]["sparse_weights_u16"],
        commitment=bytes.fromhex(prepared["commitment_hex"]),
        reveal_round=prepared["reveal_round"],
        era_current=response["weight_result"]["block"] + 1,
        nonce=7,
        block_hash="b" * 64,
    )
    result = authority.sign_weight_extrinsic(
        commit_authorization_id=prepared["commit_authorization_id"],
        era_current=response["weight_result"]["block"] + 1,
        nonce=7,
        block_hash="b" * 64,
        signature_payload_hex=expected["signed_message_hex"],
    )
    assert result["authorization"] == expected
    assert result["receipt"]["purpose"] == "validator.set_weights_extrinsic.v2"
    assert result["receipt"]["parent_receipt_hashes"] == [
        response["receipt_graph"]["root_receipt_hash"]
    ]
    with pytest.raises(ValidatorHotkeyAuthorityV2Error, match="already used"):
        authority.sign_weight_extrinsic(
            commit_authorization_id=prepared["commit_authorization_id"],
            era_current=response["weight_result"]["block"] + 1,
            nonce=7,
            block_hash="b" * 64,
            signature_payload_hex=expected["signed_message_hex"],
        )


def test_extrinsic_signer_rejects_modified_sdk_payload(monkeypatch):
    authority, response, prepared, _drand = _prepare(monkeypatch)
    with pytest.raises(ValidatorHotkeyAuthorityV2Error, match="differs"):
        authority.sign_weight_extrinsic(
            commit_authorization_id=prepared["commit_authorization_id"],
            era_current=response["weight_result"]["block"] + 1,
            nonce=7,
            block_hash="b" * 64,
            signature_payload_hex="00" * 32,
        )


def test_restart_recovery_revalidates_bundle_binding_and_signed_extrinsic(
    monkeypatch,
):
    authority, response, prepared, _drand = _prepare(monkeypatch)
    from leadpoet_canonical.hotkey_authority_v2 import (
        build_weight_extrinsic_authorization_v2,
    )

    event_hash = "sha256:" + "a" * 64
    expected = build_weight_extrinsic_authorization_v2(
        profile=_profile(),
        validator_hotkey=HOTKEY,
        hotkey_public_key_hex=HOTKEY_PUBLIC.hex(),
        epoch_id=response["weight_result"]["epoch_id"],
        netuid=71,
        weight_receipt_hash=response["receipt_graph"]["root_receipt_hash"],
        weight_submission_event_hash=event_hash,
        weights_hash=response["weight_result"]["weights_hash"],
        sparse_uids=response["weight_result"]["sparse_uids"],
        sparse_weights_u16=response["weight_result"]["sparse_weights_u16"],
        commitment=bytes.fromhex(prepared["commitment_hex"]),
        reveal_round=prepared["reveal_round"],
        era_current=response["weight_result"]["block"] + 1,
        nonce=7,
        block_hash="b" * 64,
    )
    signed = authority.sign_weight_extrinsic(
        commit_authorization_id=prepared["commit_authorization_id"],
        era_current=response["weight_result"]["block"] + 1,
        nonce=7,
        block_hash="b" * 64,
        signature_payload_hex=expected["signed_message_hex"],
    )
    boot = dict(authority._boot_identity_supplier())
    old_boot = {**boot, "physical_role": "validator_weights"}
    binding_message = create_binding_message(
        netuid=71,
        chain=_profile()["chain_endpoint"],
        enclave_pubkey=boot["signing_pubkey"],
        validator_code_hash=boot["build_manifest_hash"],
        version=boot["commit_sha"],
    )
    binding = authority.sign_application_message(
        message_hex=binding_message.encode().hex(),
        parent_receipt_hash=response["receipt_graph"]["root_receipt_hash"],
    )
    bundle = {
        "schema_version": "leadpoet.published_weight_bundle.v2",
        "validator_hotkey": HOTKEY,
        "binding_message": binding_message,
        "validator_hotkey_signature": binding["signature"],
        "weight_snapshot": response["weight_snapshot"],
        "weight_result": response["weight_result"],
        "weights_signature": response["weights_signature"],
        "receipt_graph": {
            "root_receipt_hash": binding["receipt"]["receipt_hash"],
            "boot_identities": [old_boot],
            "receipts": [
                response["receipt_graph"]["receipts"][0],
                binding["receipt"],
            ],
            "transport_attempts": [],
            "host_operations": [],
        },
    }
    monkeypatch.setattr(
        module,
        "validate_published_weight_bundle_v2",
        lambda _bundle: {
            "bundle_hash": "sha256:" + "d" * 64,
            "validator_hotkey": HOTKEY,
        },
    )
    monkeypatch.setattr(module, "verify_boot_identity_nitro", lambda *_a, **_k: {})

    authority._weights.clear()
    authority._commits.clear()
    recovered = authority.recover_weight_publication(
        published_bundle=bundle,
        weight_submission_event_hash=event_hash,
        extrinsic_signature_results=[signed],
    )
    assert recovered["weight_authorization_id"].startswith("sha256:")
    assert recovered["signed_extrinsics"] == [
        {
            "authorization_hash": signed["authorization_hash"],
            "extrinsic_hash": signed["extrinsic_hash"],
            "extrinsic_hex": next(iter(authority._commits.values()))[
                "signed_result"
            ]["extrinsic_hex"],
        }
    ]

    with pytest.raises(
        ValidatorHotkeyAuthorityV2Error,
        match="authorization differs",
    ):
        authority.recover_weight_publication(
            published_bundle=bundle,
            weight_submission_event_hash="sha256:" + "f" * 64,
            extrinsic_signature_results=[signed],
        )


def test_serve_axon_signer_rebuilds_and_limits_the_exact_call(monkeypatch):
    authority, _boot_value, _signing_key, _drand, _observed = _authority(
        monkeypatch
    )
    _provision(authority)
    from leadpoet_canonical.hotkey_authority_v2 import (
        build_serve_axon_extrinsic_authorization_v2,
    )

    request = {
        "netuid": 71,
        "version": 10005000,
        "ip": 2130706433,
        "port": 8093,
        "ip_type": 4,
        "protocol": 4,
        "placeholder1": 0,
        "placeholder2": 0,
        "era_current": 8596806,
        "nonce": 7,
        "block_hash": "b" * 64,
    }
    expected = build_serve_axon_extrinsic_authorization_v2(
        profile=_profile(),
        validator_hotkey=HOTKEY,
        hotkey_public_key_hex=HOTKEY_PUBLIC.hex(),
        **request,
    )
    result = authority.sign_serve_axon_extrinsic(
        **request,
        signature_payload_hex=expected["signed_message_hex"],
    )
    assert result["authorization"] == expected
    assert result["receipt"]["purpose"] == "validator.serve_axon_extrinsic.v2"
    with pytest.raises(ValidatorHotkeyAuthorityV2Error, match="differs"):
        authority.sign_serve_axon_extrinsic(
            **request,
            signature_payload_hex="00" * 32,
        )


def test_commit_preparation_rejects_changed_weights_or_chain_profile(monkeypatch):
    authority, boot, signing_key, _drand, _observed = _authority(monkeypatch)
    _provision(authority)
    response = _weight_response(boot, signing_key)
    weight_id = authority.register_weight_result(response)
    common = {
        "weight_authorization_id": weight_id,
        "weight_submission_event_hash": "sha256:" + "a" * 64,
        "uids": response["weight_result"]["sparse_uids"],
        "weights_u16": response["weight_result"]["sparse_weights_u16"],
        "version_key": 10005000,
        "last_epoch_block": response["weight_result"]["block"] - 126,
        "pending_epoch_at": 0,
        "subnet_epoch_index": 23860,
        "tempo": 360,
        "blocks_since_last_step": 127,
        "current_block": response["weight_result"]["block"] + 1,
        "subnet_reveal_period_epochs": 1,
        "block_time": 12.0,
        "hotkey_public_key_hex": HOTKEY_PUBLIC.hex(),
    }
    changed = dict(common)
    changed["weights_u16"] = [65535, 3210, 2599]
    with pytest.raises(ValidatorHotkeyAuthorityV2Error, match="vector differs"):
        authority.prepare_weight_commit(**changed)
    changed = dict(common)
    changed["tempo"] = 359
    with pytest.raises(ValidatorHotkeyAuthorityV2Error, match="measured profile"):
        authority.prepare_weight_commit(**changed)


def test_commit_retry_budget_covers_the_full_measured_block_window(monkeypatch):
    authority, boot, signing_key, _drand, _observed = _authority(monkeypatch)
    _provision(authority)
    response = _weight_response(boot, signing_key)
    weight_id = authority.register_weight_result(response)
    common = {
        "weight_authorization_id": weight_id,
        "weight_submission_event_hash": "sha256:" + "a" * 64,
        "uids": response["weight_result"]["sparse_uids"],
        "weights_u16": response["weight_result"]["sparse_weights_u16"],
        "version_key": 10005000,
        "last_epoch_block": response["weight_result"]["block"] - 126,
        "pending_epoch_at": 0,
        "subnet_epoch_index": 23860,
        "tempo": 360,
        "blocks_since_last_step": 127,
        "current_block": response["weight_result"]["block"] + 1,
        "subnet_reveal_period_epochs": 1,
        "block_time": 12.0,
        "hotkey_public_key_hex": HOTKEY_PUBLIC.hex(),
    }
    max_attempts = (_profile()["max_snapshot_block_drift"] + 1) * 5
    for _ in range(max_attempts):
        authority.prepare_weight_commit(**common)
    with pytest.raises(ValidatorHotkeyAuthorityV2Error, match="attempt limit"):
        authority.prepare_weight_commit(**common)
