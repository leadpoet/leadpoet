import base64

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from gateway.tee.kms_recipient_v2 import KMSRecipientV2, KMSRecipientV2Error
from gateway.tee.provider_broker_v2 import (
    credential_reference_hash,
    credential_value_hash,
)
from gateway.tee.source_add_runtime_v2 import source_add_job_credential_slot
from leadpoet_canonical.attested_v2 import sha256_json


def _manager(secret="provider-secret"):
    observed = {}

    def attest(*, user_data, signing_pubkey):
        observed["user_data"] = user_data
        observed["public_key"] = signing_pubkey
        return b"nitro-attestation"

    manager = KMSRecipientV2(
        boot_identity_supplier=lambda: {"boot_identity_hash": "sha256:" + "a" * 64},
        expected_credential_ref_hashes={
            "openrouter": credential_reference_hash(secret)
        },
        attestation_supplier=attest,
    )
    return manager, observed, secret


def _encrypt(request, plaintext):
    public_key = serialization.load_der_public_key(
        base64.b64decode(request["recipient_public_key_der_b64"])
    )
    return base64.b64encode(
        public_key.encrypt(
            plaintext.encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
    ).decode()


def test_kms_recipient_unwraps_only_expected_credential_inside_enclave():
    manager, observed, secret = _manager()
    request = manager.recipient_request("openrouter")
    assert request["attestation_document_b64"] == base64.b64encode(
        b"nitro-attestation"
    ).decode()
    assert observed["public_key"] == base64.b64decode(
        request["recipient_public_key_der_b64"]
    )
    assert manager.unwrap_credential(
        slot="openrouter",
        ciphertext_for_recipient_b64=_encrypt(request, secret),
    ) == secret
    assert manager.provisioned_slots() == ("openrouter",)


def test_kms_recipient_refreshes_attestation_for_each_kms_request():
    attestations = []

    def attest(*, user_data, signing_pubkey):
        attestations.append((user_data, signing_pubkey))
        return f"nitro-attestation-{len(attestations)}".encode()

    manager = KMSRecipientV2(
        boot_identity_supplier=lambda: {
            "boot_identity_hash": "sha256:" + "a" * 64
        },
        expected_credential_ref_hashes={
            "openrouter": credential_reference_hash("provider-secret")
        },
        attestation_supplier=attest,
    )

    first = manager.recipient_request("openrouter")
    second = manager.recipient_request("openrouter")

    assert first["attestation_document_b64"] != second["attestation_document_b64"]
    assert first["request_nonce"] != second["request_nonce"]
    assert (
        first["recipient_public_key_der_b64"]
        == second["recipient_public_key_der_b64"]
    )
    assert len(attestations) == 2


def test_kms_recipient_rejects_wrong_plaintext_and_reprovisioning():
    manager, _, secret = _manager()
    request = manager.recipient_request("openrouter")
    with pytest.raises(KMSRecipientV2Error, match="reference mismatch"):
        manager.unwrap_credential(
            slot="openrouter",
            ciphertext_for_recipient_b64=_encrypt(request, "attacker-secret"),
        )
    assert manager.unwrap_credential(
        slot="openrouter",
        ciphertext_for_recipient_b64=_encrypt(request, secret),
    ) == secret
    with pytest.raises(KMSRecipientV2Error, match="already provisioned"):
        manager.unwrap_credential(
            slot="openrouter",
            ciphertext_for_recipient_b64=_encrypt(request, secret),
        )


def test_kms_recipient_has_no_unknown_slot_or_plaintext_fallback():
    manager, _, _ = _manager()
    with pytest.raises(KMSRecipientV2Error, match="not measured"):
        manager.recipient_request("unknown")
    with pytest.raises(KMSRecipientV2Error, match="ciphertext"):
        manager.unwrap_credential(
            slot="openrouter",
            ciphertext_for_recipient_b64="not-base64",
        )


def test_job_kms_recipient_is_single_use_and_binds_job_key_hash():
    manager, _, _ = _manager()
    secret = "miner-specific-key"
    request = manager.job_recipient_request(
        job_id="autoresearch-v2:job-1",
        slot="openrouter",
        credential_value_hash_expected=credential_value_hash(secret),
        key_ref_hash=sha256_json({"key_ref": "encrypted_ref:openrouter:abc"}),
    )
    lease = manager.unwrap_job_credential(
        request_id=request["request_id"],
        ciphertext_for_recipient_b64=_encrypt(request, secret),
    )
    assert lease["job_id"] == "autoresearch-v2:job-1"
    assert lease["credential"] == secret
    with pytest.raises(KMSRecipientV2Error, match="already used"):
        manager.unwrap_job_credential(
            request_id=request["request_id"],
            ciphertext_for_recipient_b64=_encrypt(request, secret),
        )


def test_job_only_slot_does_not_become_a_boot_global_credential():
    manager = KMSRecipientV2(
        boot_identity_supplier=lambda: {
            "boot_identity_hash": "sha256:" + "a" * 64
        },
        expected_credential_ref_hashes={
            "openrouter": credential_reference_hash("provider-secret")
        },
        expected_job_slot_ref_hashes={
            "openrouter_management": sha256_json(
                {"slot": "openrouter_management"}
            )
        },
        attestation_supplier=lambda **_kwargs: b"nitro-attestation",
    )
    secret = "management-key"
    request = manager.job_recipient_request(
        job_id="autoresearch-v2:job-2",
        slot="openrouter_management",
        credential_value_hash_expected=credential_value_hash(secret),
        key_ref_hash=sha256_json({"key_ref": "management"}),
    )
    assert manager.unwrap_job_credential(
        request_id=request["request_id"],
        ciphertext_for_recipient_b64=_encrypt(request, secret),
    )["credential_slot"] == "openrouter_management"
    with pytest.raises(KMSRecipientV2Error, match="not measured"):
        manager.recipient_request("openrouter_management")


def test_dynamic_source_add_slot_is_job_only_and_hash_bound():
    manager, _, _ = _manager()
    slot = source_add_job_credential_slot("source_one")
    secret = "source-add-job-secret"
    request = manager.job_recipient_request(
        job_id="autoresearch-v2:source-one",
        slot=slot,
        credential_value_hash_expected=credential_value_hash(secret),
        key_ref_hash=sha256_json(
            {"key_ref": "encrypted_ref:source_add:" + "a" * 32}
        ),
    )
    lease = manager.unwrap_job_credential(
        request_id=request["request_id"],
        ciphertext_for_recipient_b64=_encrypt(request, secret),
    )
    assert lease["credential_slot"] == slot
    assert lease["credential"] == secret
    with pytest.raises(KMSRecipientV2Error, match="not measured"):
        manager.recipient_request(slot)
