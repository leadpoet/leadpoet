from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from gateway.tee.artifact_vault_v2 import EncryptedArtifactVaultV2
from gateway.tee.kms_recipient_v2 import KMSRecipientV2, KMSRecipientV2Error
from gateway.tee.provider_broker_v2 import credential_value_hash
from gateway.tee.source_add_credential_ingress_v2 import (
    seal_source_add_ingress_credential_v2,
    unseal_source_add_job_credential_v2,
)
from gateway.tee.source_add_runtime_v2 import (
    build_source_add_job_envelope_v2,
    build_source_add_runtime_route_v2,
    validate_source_add_credential_envelope_v2,
)
from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json


HASH = "sha256:" + "a" * 64
MASTER_KEY = bytes(range(32))
SECRET = "source-add-secret-value"


def _recipient() -> KMSRecipientV2:
    return KMSRecipientV2(
        boot_identity_supplier=lambda: {"boot_identity_hash": HASH},
        expected_credential_ref_hashes={"artifact_master_key": HASH},
        attestation_supplier=lambda **_kwargs: b"nsm-attestation",
    )


def _vault() -> EncryptedArtifactVaultV2:
    return EncryptedArtifactVaultV2(
        master_key=MASTER_KEY,
        boot_identity_hash=HASH,
        retention_days=30,
        clock=lambda: datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc),
    )


def _sealed_envelope():
    recipient = _recipient()
    credential_ref = "encrypted_ref:source_add:" + "1" * 32
    request = recipient.source_add_ingress_recipient_request(
        miner_hotkey="miner-hotkey",
        adapter_ref="source_add:adapter:test-source",
        credential_ref=credential_ref,
    )
    public_key = serialization.load_der_public_key(
        base64.b64decode(request["recipient_public_key_der_b64"])
    )
    ciphertext = public_key.encrypt(
        SECRET.encode(),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    lease = recipient.unwrap_source_add_ingress_credential(
        request_id=request["request_id"],
        ciphertext_b64=base64.b64encode(ciphertext).decode(),
    )
    envelope = seal_source_add_ingress_credential_v2(lease, vault=_vault())
    return recipient, request, envelope


def _source_row(envelope):
    return {
        "adapter_id": "adapter:test-source",
        "miner_hotkey": "miner-hotkey",
        "registry_provider_id": "test_source",
        "provision_status": "provisioned_autoresearch_eligible",
        "credential_envelope": envelope,
        "provision_doc": {
            "provider_registry_entry": {
                "id": "test_source",
                "base_url": "https://api.test-source.example",
                "auth_kind": "header",
                "auth_name": "x-api-key",
                "credential_ref": [envelope["credential_ref"]],
                "cost_model": {"est_cost_microusd_per_call": 500},
                "capability_policy": {
                    "routes": [{"method": "GET", "path": "/search"}]
                },
            },
            "probe_endpoints": [
                {
                    "endpoint_id": "test_source.search",
                    "provider_id": "test_source",
                    "method": "GET",
                    "path": "/search",
                    "params": [],
                }
            ],
        },
    }


def test_client_secret_is_only_plaintext_inside_coordinator_and_survives_restart() -> None:
    _recipient_authority, request, envelope = _sealed_envelope()
    normalized = validate_source_add_credential_envelope_v2(envelope)

    assert normalized["envelope_kind"] == "coordinator_sealed"
    assert normalized["credential_value_hash"] == credential_value_hash(SECRET)
    assert SECRET not in repr(request)
    assert SECRET not in repr(envelope)

    row = _source_row(envelope)
    route = build_source_add_runtime_route_v2(row)
    job_envelope = build_source_add_job_envelope_v2(row, job_id="model-job-1")
    restarted_vault = _vault()
    lease = unseal_source_add_job_credential_v2(
        job_envelope,
        vault=restarted_vault,
    )
    assert route["credential_value_hash"] == credential_value_hash(SECRET)
    assert lease["credential"] == SECRET
    assert lease["credential_slot"] == route["credential_slot"]


def test_ingress_recipient_is_one_use_and_ciphertext_is_scope_bound() -> None:
    recipient, request, _envelope = _sealed_envelope()
    public_key = serialization.load_der_public_key(
        base64.b64decode(request["recipient_public_key_der_b64"])
    )
    ciphertext = public_key.encrypt(
        SECRET.encode(),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    with pytest.raises(KMSRecipientV2Error, match="already used"):
        recipient.unwrap_source_add_ingress_credential(
            request_id=request["request_id"],
            ciphertext_b64=base64.b64encode(ciphertext).decode(),
        )


def test_source_add_seal_rpc_retry_is_idempotent_and_rejects_rebinding(
    monkeypatch,
) -> None:
    from gateway.tee import source_add_credential_ingress_v2 as ingress
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "gateway" / "tee")
    )
    from gateway.tee import tee_service

    class _Recipient:
        calls = 0

        def unwrap_source_add_ingress_credential(self, **kwargs):
            self.calls += 1
            return {"request_id": kwargs["request_id"]}

    recipient = _Recipient()
    monkeypatch.setattr(tee_service, "get_v2_runtime_identity", object)
    monkeypatch.setattr(tee_service, "get_v2_kms_recipient", lambda: recipient)
    monkeypatch.setattr(tee_service, "get_v2_artifact_vault", object)
    monkeypatch.setattr(
        ingress,
        "seal_source_add_ingress_credential_v2",
        lambda lease, **_kwargs: {"sealed_request_id": lease["request_id"]},
    )
    tee_service.v2_ingress_seal_cache.clear()
    params = {
        "request_id": HASH,
        "ciphertext_b64": base64.b64encode(b"ciphertext-one").decode("ascii"),
    }
    first = tee_service.handle_v2_runtime_rpc(
        "v2_seal_source_add_ingress_credential", params
    )
    second = tee_service.handle_v2_runtime_rpc(
        "v2_seal_source_add_ingress_credential", params
    )
    assert first == second
    assert recipient.calls == 1

    changed = {
        **params,
        "ciphertext_b64": base64.b64encode(b"ciphertext-two").decode("ascii"),
    }
    with pytest.raises(ValueError, match="ciphertext changed"):
        tee_service.handle_v2_runtime_rpc(
            "v2_seal_source_add_ingress_credential", changed
        )
    tee_service.v2_ingress_seal_cache.clear()


def test_sealed_job_envelope_rejects_ciphertext_tampering() -> None:
    _recipient_authority, _request, envelope = _sealed_envelope()
    job_envelope = build_source_add_job_envelope_v2(
        _source_row(envelope),
        job_id="model-job-1",
    )
    raw = base64.b64decode(job_envelope["ciphertext_blob_b64"])
    tampered = {
        **job_envelope,
        "ciphertext_blob_b64": base64.b64encode(raw + b"x").decode(),
    }
    with pytest.raises(Exception, match="hash differs"):
        unseal_source_add_job_credential_v2(tampered, vault=_vault())


def test_sealed_job_envelope_rejects_credential_rescoping() -> None:
    _recipient_authority, _request, envelope = _sealed_envelope()
    job_envelope = build_source_add_job_envelope_v2(
        _source_row(envelope),
        job_id="model-job-1",
    )
    changed_context = {
        **job_envelope["encryption_context"],
        "miner_hotkey": "different-miner-hotkey",
    }
    rescoped = {
        **job_envelope,
        "encryption_context": changed_context,
        "encryption_context_hash": sha256_json(changed_context),
    }
    with pytest.raises(Exception, match="scope differs"):
        unseal_source_add_job_credential_v2(rescoped, vault=_vault())


@pytest.mark.asyncio
async def test_job_provision_dispatches_sealed_envelope_without_kms_plaintext() -> None:
    from gateway.utils.tee_kms_provision_v2 import (
        provision_job_credential_envelope_v2,
    )

    _recipient_authority, _request, envelope = _sealed_envelope()
    job_envelope = build_source_add_job_envelope_v2(
        _source_row(envelope),
        job_id="model-job-1",
    )

    class _Client:
        observed = None

        async def v2_provision_job_sealed_source_add_secret(self, *, envelope):
            self.observed = dict(envelope)
            return {
                "status": "ready",
                "job_id": envelope["job_id"],
                "credential_slot": envelope["credential_slot"],
                "credential_ref_hash": envelope["credential_value_hash"],
            }

    client = _Client()
    result = await provision_job_credential_envelope_v2(
        job_envelope,
        client=client,
    )
    assert result["status"] == "ready"
    assert SECRET not in repr(client.observed)
    assert sha256_bytes(SECRET.encode()) == job_envelope["credential_value_hash"]
