import base64
import hashlib

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_canonical import credential_recipient_v2 as recipient_module


MINER = "5" + "M" * 47
VERIFIED_BOOT = {
    "boot_identity_hash": "sha256:" + "b" * 64,
    "pcr0": "c" * 96,
}


def _recipient(kind="runtime"):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_der = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    claim = {
        "schema_version": recipient_module.OPENROUTER_INGRESS_RECIPIENT_SCHEMA_VERSION,
        "purpose": recipient_module.OPENROUTER_INGRESS_RECIPIENT_PURPOSE,
        "boot_identity_hash": "sha256:" + "b" * 64,
        "miner_hotkey_hash": "sha256:" + hashlib.sha256(MINER.encode()).hexdigest(),
        "credential_kind": kind,
        "credential_slot": (
            "openrouter" if kind == "runtime" else "openrouter_management"
        ),
        "recipient_public_key_hash": "sha256:" + hashlib.sha256(public_der).hexdigest(),
        "request_nonce": "a" * 32,
    }
    request_id = sha256_json(claim)
    recipient = {
        **claim,
        "request_id": request_id,
        "recipient_public_key_der_b64": base64.b64encode(public_der).decode(),
        "attestation_document_b64": base64.b64encode(b"attestation").decode(),
        "key_encryption_algorithm": recipient_module.KMS_KEY_ENCRYPTION_ALGORITHM,
    }
    attestation = {
        "attestation_public_key": public_der.hex(),
        "user_data": {
            "schema_version": recipient_module.OPENROUTER_INGRESS_RECIPIENT_SCHEMA_VERSION,
            "purpose": recipient_module.OPENROUTER_INGRESS_RECIPIENT_PURPOSE,
            "claim_hash": request_id,
        },
    }
    return private_key, recipient, attestation


def test_verified_recipient_encrypts_credential_locally(monkeypatch):
    private_key, recipient, attestation = _recipient()
    monkeypatch.setattr(
        recipient_module,
        "verify_nitro_attestation_full",
        lambda **_kwargs: (True, attestation),
    )

    encrypted = recipient_module.verify_and_encrypt_openrouter_credential_v2(
        recipient,
        "sk-or-v1-secret",
        miner_hotkey=MINER,
        credential_kind="runtime",
        verified_coordinator_boot_identity=VERIFIED_BOOT,
    )

    plaintext = private_key.decrypt(
        base64.b64decode(encrypted["ciphertext_b64"], validate=True),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    assert plaintext == b"sk-or-v1-secret"
    assert encrypted["request_id"] == recipient["request_id"]


def test_client_policy_matches_gateway_recipient_policy():
    from gateway.tee import kms_recipient_v2 as gateway_recipient

    assert (
        recipient_module.OPENROUTER_INGRESS_RECIPIENT_SCHEMA_VERSION
        == gateway_recipient.OPENROUTER_INGRESS_RECIPIENT_SCHEMA_VERSION
    )
    assert (
        recipient_module.OPENROUTER_INGRESS_RECIPIENT_PURPOSE
        == gateway_recipient.OPENROUTER_INGRESS_RECIPIENT_PURPOSE
    )
    assert (
        recipient_module.KMS_KEY_ENCRYPTION_ALGORITHM
        == gateway_recipient.KMS_KEY_ENCRYPTION_ALGORITHM
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(miner_hotkey_hash="sha256:" + "c" * 64), "miner binding"),
        (lambda value: value.update(credential_slot="openrouter_management"), "policy"),
        (lambda value: value.update(request_id="sha256:" + "d" * 64), "claim hash"),
    ],
)
def test_recipient_rejects_rebinding(monkeypatch, mutation, message):
    _private_key, recipient, attestation = _recipient()
    mutation(recipient)
    monkeypatch.setattr(
        recipient_module,
        "verify_nitro_attestation_full",
        lambda **_kwargs: (True, attestation),
    )
    with pytest.raises(recipient_module.CredentialRecipientV2Error, match=message):
        recipient_module.verify_and_encrypt_openrouter_credential_v2(
            recipient,
            "sk-or-v1-secret",
            miner_hotkey=MINER,
            credential_kind="runtime",
            verified_coordinator_boot_identity=VERIFIED_BOOT,
        )


def test_recipient_requires_valid_nitro_attestation(monkeypatch):
    _private_key, recipient, _attestation = _recipient()
    monkeypatch.setattr(
        recipient_module,
        "verify_nitro_attestation_full",
        lambda **_kwargs: (False, {"error": "invalid"}),
    )
    with pytest.raises(
        recipient_module.CredentialRecipientV2Error,
        match="Nitro attestation",
    ):
        recipient_module.verify_and_encrypt_openrouter_credential_v2(
            recipient,
            "sk-or-v1-secret",
            miner_hotkey=MINER,
            credential_kind="runtime",
            verified_coordinator_boot_identity=VERIFIED_BOOT,
        )


def _release_url(commit, version, *, host=None):
    host = host or (
        "leadpoet-attested-v2-artifacts-493765492819."
        "s3.us-east-1.amazonaws.com"
    )
    return (
        f"https://{host}/attested-v2/releases/{commit}/release-channel-v2.json"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
        "&X-Amz-Credential=credential"
        "&X-Amz-Date=20260721T000000Z"
        "&X-Amz-Expires=300"
        "&X-Amz-SignedHeaders=host"
        "&X-Amz-Signature=signature"
        f"&versionId={version}"
    )


class _Response:
    def __init__(self, url, headers, body=b""):
        self._url = url
        self.headers = headers
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def geturl(self):
        return self._url

    def read(self, _limit):
        return self._body


def _release_evidence():
    commit = "d" * 40
    version = "release-version-1"
    boot = {
        "boot_identity_hash": "sha256:" + "e" * 64,
        "physical_role": "gateway_coordinator",
        "commit_sha": commit,
        "pcr0": "f" * 96,
        "build_manifest_hash": "sha256:" + "1" * 64,
        "dependency_lock_hash": "sha256:" + "2" * 64,
    }
    return {
        "schema_version": recipient_module.OPENROUTER_RELEASE_EVIDENCE_SCHEMA_VERSION,
        "coordinator_boot_identity": boot,
        "release_channel_version_id": version,
        "release_channel_get_url": _release_url(commit, version),
        "release_channel_head_url": _release_url(commit, version),
    }


def test_release_evidence_binds_object_locked_channel_and_boot(monkeypatch):
    evidence = _release_evidence()
    boot = evidence["coordinator_boot_identity"]
    headers = {
        "x-amz-object-lock-mode": "COMPLIANCE",
        "x-amz-object-lock-retain-until-date": "2027-07-21T00:00:00Z",
        "x-amz-version-id": evidence["release_channel_version_id"],
    }

    def open_url(url, *, method):
        return _Response(
            url,
            headers,
            b'{}' if method == "GET" else b"",
        )

    monkeypatch.setattr(
        "gateway.tee.release_channel_v2.validate_release_channel_v2",
        lambda _value, expected_commit: {
            "commit_sha": expected_commit,
            "gateway_release_manifest": {},
        },
    )
    monkeypatch.setattr(
        "gateway.tee.release_manifest_v2.role_expectation",
        lambda _manifest, _role: {
            key: boot[key]
            for key in (
                "physical_role",
                "commit_sha",
                "pcr0",
                "build_manifest_hash",
                "dependency_lock_hash",
            )
        },
    )
    monkeypatch.setattr(
        "leadpoet_canonical.attested_v2.validate_boot_identity",
        lambda _boot: None,
    )
    monkeypatch.setattr(
        "leadpoet_canonical.attested_v2.verify_boot_identity_nitro",
        lambda identity, **_kwargs: identity,
    )

    assert recipient_module.verify_openrouter_credential_release_v2(
        evidence,
        http_open=open_url,
    ) == boot


def test_release_evidence_rejects_noncompliance_object():
    evidence = _release_evidence()
    headers = {
        "x-amz-object-lock-mode": "GOVERNANCE",
        "x-amz-object-lock-retain-until-date": "2027-07-21T00:00:00Z",
        "x-amz-version-id": evidence["release_channel_version_id"],
    }

    with pytest.raises(
        recipient_module.CredentialRecipientV2Error,
        match="Object-Locked",
    ):
        recipient_module.verify_openrouter_credential_release_v2(
            evidence,
            http_open=lambda url, *, method: _Response(url, headers, b"{}"),
        )


def test_release_evidence_rejects_another_host():
    evidence = _release_evidence()
    evidence["release_channel_get_url"] = _release_url(
        evidence["coordinator_boot_identity"]["commit_sha"],
        evidence["release_channel_version_id"],
        host="attacker.example.com",
    )
    with pytest.raises(
        recipient_module.CredentialRecipientV2Error,
        match="violates policy",
    ):
        recipient_module.verify_openrouter_credential_release_v2(evidence)
