import base64

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from pydantic import ValidationError

from gateway.research_lab import admin
from gateway.research_lab.models import ResearchLabSourceAddProbeSpec
from gateway.tee.kms_recipient_v2 import (
    KMS_KEY_ENCRYPTION_ALGORITHM,
    SOURCE_ADD_INGRESS_RECIPIENT_PURPOSE,
    SOURCE_ADD_INGRESS_RECIPIENT_SCHEMA_VERSION,
)
from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json


def _recipient(monkeypatch):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_der = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    claim = {
        "schema_version": SOURCE_ADD_INGRESS_RECIPIENT_SCHEMA_VERSION,
        "purpose": SOURCE_ADD_INGRESS_RECIPIENT_PURPOSE,
        "boot_identity_hash": "sha256:" + "1" * 64,
        "miner_hotkey_hash": "sha256:" + "2" * 64,
        "adapter_ref_hash": "sha256:" + "3" * 64,
        "credential_ref": "encrypted_ref:source_add:" + "4" * 32,
        "key_ref_hash": "sha256:" + "5" * 64,
        "recipient_public_key_hash": sha256_bytes(public_der),
        "request_nonce": "6" * 32,
    }
    request_id = sha256_json(claim)
    recipient = {
        **claim,
        "request_id": request_id,
        "recipient_public_key_der_b64": base64.b64encode(public_der).decode("ascii"),
        "attestation_document_b64": base64.b64encode(b"attestation").decode("ascii"),
        "key_encryption_algorithm": KMS_KEY_ENCRYPTION_ALGORITHM,
    }

    def fake_verify(**_kwargs):
        return True, {
            "attestation_public_key": public_der.hex(),
            "user_data": {
                "schema_version": SOURCE_ADD_INGRESS_RECIPIENT_SCHEMA_VERSION,
                "purpose": SOURCE_ADD_INGRESS_RECIPIENT_PURPOSE,
                "claim_hash": request_id,
            },
        }

    monkeypatch.setattr(
        "leadpoet_canonical.nitro.verify_nitro_attestation_full",
        fake_verify,
    )
    return recipient, private_key


@pytest.mark.asyncio
async def test_source_add_configure_dry_run_uses_gateway_model_validation():
    args = admin.build_parser().parse_args(
        [
            "source-add",
            "configure-test",
            "--submission-id",
            "source_add_submission:0123456789abcdef",
            "--base-url",
            "https://api.example.com/v1",
            "--probe-json",
            '{"method":"GET","path":"/records","query":{"limit":1},"body_json":null}',
        ]
    )

    result = await admin._run_source_add_admin(args)

    assert result["dry_run"] is True
    assert result["configuration"]["base_url"] == "https://api.example.com/v1"
    assert result["configuration"]["probes"][0]["method"] == "GET"
    assert "api_credential" not in str(result)


@pytest.mark.asyncio
async def test_source_add_configure_dry_run_rejects_unsafe_header():
    args = admin.build_parser().parse_args(
        [
            "source-add",
            "configure-test",
            "--submission-id",
            "source_add_submission:0123456789abcdef",
            "--base-url",
            "https://api.example.com",
            "--header",
            "Host=metadata.internal",
            "--probe-json",
            '{"method":"GET","path":"/records","query":{},"body_json":null}',
        ]
    )

    with pytest.raises(ValueError, match="request header is unsafe"):
        await admin._run_source_add_admin(args)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        ["X-Client=safe\r\nX-Injected: true"],
        ["X-Client=one", "x-client=two"],
    ],
)
async def test_source_add_configure_rejects_header_injection_and_case_duplicates(
    headers,
):
    argv = [
        "source-add",
        "configure-test",
        "--submission-id",
        "source_add_submission:0123456789abcdef",
        "--base-url",
        "https://api.example.com",
        "--probe-json",
        '{"method":"GET","path":"/records","query":{},"body_json":null}',
    ]
    for header in headers:
        argv.extend(("--header", header))

    with pytest.raises(ValueError, match="request header is unsafe"):
        await admin._run_source_add_admin(admin.build_parser().parse_args(argv))


def test_source_add_probe_body_enforces_size_depth_and_secret_limits():
    with pytest.raises(ValidationError, match="exceeds structural limits"):
        ResearchLabSourceAddProbeSpec(
            method="POST",
            path="/records",
            body_json={"nested": {"next": {"next": {"next": {"next": {"next": {"next": {"next": {"next": {"next": {"next": {"next": {"next": {"next": 1}}}}}}}}}}}}}},
        )
    with pytest.raises(ValidationError, match="exceeds 64 KiB"):
        ResearchLabSourceAddProbeSpec(
            method="POST",
            path="/records",
            body_json={"records": ["x" * 4_096 for _ in range(17)]},
        )
    with pytest.raises(ValidationError, match="raw secret field"):
        ResearchLabSourceAddProbeSpec(
            method="POST",
            path="/records",
            body_json={"api_key": "not-a-real-key"},
        )


def test_source_add_operator_credential_is_bound_to_verified_attestation(monkeypatch):
    recipient, private_key = _recipient(monkeypatch)
    encrypted = admin._verify_and_encrypt_source_add_credential(
        recipient,
        b"operator-secret",
    )

    plaintext = private_key.decrypt(
        base64.b64decode(encrypted["ciphertext_b64"]),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    assert plaintext == b"operator-secret"
    assert encrypted["request_id"] == recipient["request_id"]
    assert "operator-secret" not in str(encrypted)


def test_source_add_operator_credential_rejects_tampered_or_oversized_input(monkeypatch):
    recipient, _private_key = _recipient(monkeypatch)
    with pytest.raises(ValueError, match="claim hash differs"):
        admin._verify_and_encrypt_source_add_credential(
            {**recipient, "request_id": "sha256:" + "f" * 64},
            b"secret",
        )
    with pytest.raises(ValueError, match="RSA-OAEP limit"):
        admin._verify_and_encrypt_source_add_credential(recipient, b"x" * 191)
