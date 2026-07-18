import base64

import pytest
from cryptography.hazmat.primitives import hashes, padding as symmetric_padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

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


def _length(value):
    if value < 128:
        return bytes([value])
    encoded = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(encoded)]) + encoded


def _tlv(tag, value):
    return bytes([tag]) + _length(len(value)) + value


def _indefinite(tag, *children):
    return bytes([tag, 0x80]) + b"".join(children) + b"\x00\x00"


def _oid(value):
    parts = [int(part) for part in value.split(".")]
    encoded = bytearray([parts[0] * 40 + parts[1]])
    for part in parts[2:]:
        octets = [part & 0x7F]
        part >>= 7
        while part:
            octets.append(0x80 | (part & 0x7F))
            part >>= 7
        encoded.extend(reversed(octets))
    return _tlv(0x06, bytes(encoded))


def _algorithm(identifier, parameters):
    return _tlv(0x30, _oid(identifier) + parameters)


def _kms_cms_encrypt(request, plaintext, *, constructed_content=False):
    public_key = serialization.load_der_public_key(
        base64.b64decode(request["recipient_public_key_der_b64"])
    )
    aes_key = bytes(range(32))
    iv = bytes(range(16))
    padder = symmetric_padding.PKCS7(128).padder()
    padded = padder.update(plaintext.encode()) + padder.finalize()
    encryptor = Cipher(algorithms.AES(aes_key), modes.CBC(iv)).encryptor()
    encrypted_content = encryptor.update(padded) + encryptor.finalize()
    encrypted_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    sha256 = _algorithm("2.16.840.1.101.3.4.2.1", _tlv(0x05, b""))
    oaep_parameters = _tlv(
        0x30,
        _tlv(0xA0, sha256)
        + _tlv(
            0xA1,
            _algorithm("1.2.840.113549.1.1.8", sha256),
        ),
    )
    recipient = _tlv(
        0x30,
        _tlv(0x02, b"\x02")
        + _tlv(0x80, b"\x11" * 32)
        + _algorithm("1.2.840.113549.1.1.7", oaep_parameters)
        + _tlv(0x04, encrypted_key),
    )
    content = (
        _indefinite(
            0xA0,
            _tlv(0x04, encrypted_content[:16]),
            _tlv(0x04, encrypted_content[16:]),
        )
        if constructed_content
        else _tlv(0x80, encrypted_content)
    )
    encrypted_content_info = _indefinite(
        0x30,
        _oid("1.2.840.113549.1.7.1"),
        _algorithm("2.16.840.1.101.3.4.1.42", _tlv(0x04, iv)),
        content,
    )
    enveloped_data = _indefinite(
        0x30,
        _tlv(0x02, b"\x02"),
        _tlv(0x31, recipient),
        encrypted_content_info,
    )
    return base64.b64encode(
        _indefinite(
            0x30,
            _oid("1.2.840.113549.1.7.3"),
            _indefinite(0xA0, enveloped_data),
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


def test_kms_recipient_unwraps_aws_cms_enveloped_data():
    manager, _, secret = _manager()
    request = manager.recipient_request("openrouter")

    assert manager.unwrap_credential(
        slot="openrouter",
        ciphertext_for_recipient_b64=_kms_cms_encrypt(request, secret),
    ) == secret


def test_kms_recipient_unwraps_chunked_ber_encrypted_content():
    manager, _, secret = _manager()
    request = manager.recipient_request("openrouter")

    assert manager.unwrap_credential(
        slot="openrouter",
        ciphertext_for_recipient_b64=_kms_cms_encrypt(
            request,
            secret,
            constructed_content=True,
        ),
    ) == secret


def test_kms_recipient_rejects_cms_with_unmeasured_content_algorithm():
    manager, _, secret = _manager()
    request = manager.recipient_request("openrouter")
    ciphertext = base64.b64decode(_kms_cms_encrypt(request, secret))
    ciphertext = ciphertext.replace(
        _oid("2.16.840.1.101.3.4.1.42"),
        _oid("2.16.840.1.101.3.4.1.2"),
        1,
    )

    with pytest.raises(KMSRecipientV2Error, match="unwrap failed"):
        manager.unwrap_credential(
            slot="openrouter",
            ciphertext_for_recipient_b64=base64.b64encode(ciphertext).decode(),
        )


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
