import base64
import json

import pytest

from gateway.tee.kms_recipient_v2 import (
    KMS_KEY_ENCRYPTION_ALGORITHM,
    KMS_RECIPIENT_SCHEMA_VERSION,
)
from gateway.utils.tee_kms_provision_v2 import (
    JOB_PROVIDER_ENVELOPE_SCHEMA_VERSION,
    PROVIDER_ENVELOPE_SCHEMA_VERSION,
    TEEKMSProvisionV2Error,
    build_provider_envelope_v2,
    kms_key_reference_hash,
    load_provider_envelopes,
    provider_reference_hashes_from_envelopes,
    provision_provider_envelope_v2,
    provision_job_provider_envelope_v2,
    provision_job_credential_envelope_v2,
)
from gateway.tee.artifact_vault_v2 import EncryptedArtifactVaultV2
from gateway.tee.openrouter_credential_v2 import (
    build_openrouter_sealed_job_envelope_v2,
    seal_openrouter_persistent_credential_v2,
)
from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json


def _envelope():
    ciphertext = b"kms-encrypted-provider-secret"
    context = {"service": "leadpoet", "slot": "openrouter"}
    return {
        "schema_version": PROVIDER_ENVELOPE_SCHEMA_VERSION,
        "credential_slot": "openrouter",
        "credential_ref_hash": "sha256:" + "c" * 64,
        "ciphertext_blob_b64": base64.b64encode(ciphertext).decode(),
        "ciphertext_blob_hash": sha256_bytes(ciphertext),
        "kms_key_id_hash": kms_key_reference_hash("kms-key-1"),
        "encryption_context": context,
        "encryption_context_hash": sha256_json(context),
    }


class _Client:
    def __init__(self):
        self.provisioned = None

    async def v2_get_kms_recipient(self, slot):
        return {
            "schema_version": KMS_RECIPIENT_SCHEMA_VERSION,
            "credential_slot": slot,
            "credential_ref_hash": "sha256:" + "c" * 64,
            "key_encryption_algorithm": KMS_KEY_ENCRYPTION_ALGORITHM,
            "attestation_document_b64": base64.b64encode(b"nitro").decode(),
        }


    async def v2_provision_encrypted_secret(
        self, *, credential_slot, ciphertext_for_recipient_b64
    ):
        self.provisioned = (
            credential_slot,
            base64.b64decode(ciphertext_for_recipient_b64),
        )
        return {
            "status": "ready",
            "credential_slots": [credential_slot],
            "missing_credential_slots": [],
        }

    async def v2_get_job_kms_recipient(
        self,
        *,
        job_id,
        credential_slot,
        credential_value_hash,
        key_ref_hash,
    ):
        return {
            "schema_version": "leadpoet.kms_job_recipient.v2",
            "request_id": sha256_json({"job_id": job_id}),
            "job_id": job_id,
            "credential_slot": credential_slot,
            "credential_value_hash": credential_value_hash,
            "key_ref_hash": key_ref_hash,
            "key_encryption_algorithm": KMS_KEY_ENCRYPTION_ALGORITHM,
            "attestation_document_b64": base64.b64encode(b"nitro-job").decode(),
        }

    async def v2_provision_job_encrypted_secret(
        self, *, request_id, ciphertext_for_recipient_b64
    ):
        self.job_provisioned = (
            request_id,
            base64.b64decode(ciphertext_for_recipient_b64),
        )
        return {
            "status": "ready",
            "job_id": "autoresearch-v2:job-1",
            "credential_slot": "openrouter",
            "credential_ref_hash": "sha256:" + "a" * 64,
        }

    async def v2_provision_job_sealed_openrouter_secret(self, *, envelope):
        self.sealed_openrouter = dict(envelope)
        return {
            "status": "ready",
            "job_id": envelope["job_id"],
            "credential_slot": envelope["credential_slot"],
            "credential_ref_hash": envelope["credential_value_hash"],
        }


class _EncryptKMS:
    def __init__(self):
        self.request = None

    def encrypt(self, **request):
        self.request = request
        return {
            "KeyId": "arn:aws:kms:us-east-1:123:key/provider-v2",
            "CiphertextBlob": b"provider-ciphertext",
        }


def test_operator_builds_provider_envelope_without_persisting_plaintext():
    kms = _EncryptKMS()
    context = {
        "leadpoet:purpose": "gateway-provider-v2",
        "leadpoet:slot": "openrouter",
    }
    envelope = build_provider_envelope_v2(
        credential_slot="openrouter",
        plaintext=b"secret-provider-key",
        credential_ref_hash="sha256:" + "c" * 64,
        kms_key_id="alias/leadpoet-provider-v2",
        encryption_context=context,
        kms_client=kms,
    )

    assert kms.request == {
        "KeyId": "alias/leadpoet-provider-v2",
        "Plaintext": b"secret-provider-key",
        "EncryptionContext": context,
    }
    assert "secret-provider-key" not in str(envelope)
    assert envelope["credential_slot"] == "openrouter"
    assert envelope["kms_key_id_hash"] == kms_key_reference_hash(
        "arn:aws:kms:us-east-1:123:key/provider-v2"
    )


def test_binary_plaintext_requires_explicit_artifact_key_mode():
    with pytest.raises(TEEKMSProvisionV2Error):
        build_provider_envelope_v2(
            credential_slot="openrouter",
            plaintext=b"secret\x00value",
            credential_ref_hash="sha256:" + "c" * 64,
            kms_key_id="alias/leadpoet-provider-v2",
            encryption_context={"purpose": "credential"},
            kms_client=_EncryptKMS(),
        )

    envelope = build_provider_envelope_v2(
        credential_slot="artifact_master_key",
        plaintext=b"\x00" * 32,
        credential_ref_hash="sha256:" + "c" * 64,
        kms_key_id="alias/leadpoet-provider-v2",
        encryption_context={"purpose": "artifact-key"},
        kms_client=_EncryptKMS(),
        allow_binary=True,
    )
    assert envelope["credential_slot"] == "artifact_master_key"


class _KMS:
    def __init__(self, response=None):
        self.request = None
        self.response = response or {
            "KeyId": "kms-key-1",
            "CiphertextForRecipient": b"rsa-encrypted-for-enclave",
        }

    def decrypt(self, **request):
        self.request = request
        return self.response


@pytest.mark.asyncio
async def test_parent_relays_only_kms_and_recipient_ciphertext():
    client = _Client()
    kms = _KMS()
    result = await provision_provider_envelope_v2(
        _envelope(),
        client=client,
        kms_client=kms,
    )
    assert result["status"] == "ready"
    assert kms.request["CiphertextBlob"] == b"kms-encrypted-provider-secret"
    assert kms.request["Recipient"] == {
        "KeyEncryptionAlgorithm": KMS_KEY_ENCRYPTION_ALGORITHM,
        "AttestationDocument": b"nitro",
    }
    assert client.provisioned == (
        "openrouter",
        b"rsa-encrypted-for-enclave",
    )
    assert provider_reference_hashes_from_envelopes([_envelope()]) == {
        "openrouter": "sha256:" + "c" * 64
    }


@pytest.mark.asyncio
async def test_loaded_envelope_can_be_provisioned_without_field_drift(tmp_path):
    path = tmp_path / "openrouter.json"
    path.write_text(json.dumps(_envelope()), encoding="utf-8")

    loaded = load_provider_envelopes([path])
    assert "ciphertext_blob" not in loaded[0]

    result = await provision_provider_envelope_v2(
        loaded[0],
        client=_Client(),
        kms_client=_KMS(),
    )
    assert result["status"] == "ready"


@pytest.mark.asyncio
async def test_parent_rejects_plaintext_kms_response():
    with pytest.raises(TEEKMSProvisionV2Error, match="plaintext"):
        await provision_provider_envelope_v2(
            _envelope(),
            client=_Client(),
            kms_client=_KMS(
                {
                    "KeyId": "kms-key-1",
                    "Plaintext": b"must-never-reach-parent",
                    "CiphertextForRecipient": b"ciphertext",
                }
            ),
        )


@pytest.mark.asyncio
async def test_parent_rejects_wrong_kms_key_metadata():
    with pytest.raises(TEEKMSProvisionV2Error, match="key differs"):
        await provision_provider_envelope_v2(
            _envelope(),
            client=_Client(),
            kms_client=_KMS(
                {
                    "KeyId": "attacker-key",
                    "CiphertextForRecipient": b"ciphertext",
                }
            ),
        )


@pytest.mark.asyncio
async def test_job_key_is_reencrypted_directly_to_attested_coordinator():
    envelope = {
        **_envelope(),
        "schema_version": JOB_PROVIDER_ENVELOPE_SCHEMA_VERSION,
        "job_id": "autoresearch-v2:job-1",
        "credential_value_hash": "sha256:" + "a" * 64,
        "key_ref_hash": "sha256:" + "b" * 64,
    }
    client = _Client()
    kms = _KMS()
    result = await provision_job_provider_envelope_v2(
        envelope,
        client=client,
        kms_client=kms,
    )
    assert result["job_id"] == "autoresearch-v2:job-1"
    assert kms.request["Recipient"]["AttestationDocument"] == b"nitro-job"
    assert client.job_provisioned[1] == b"rsa-encrypted-for-enclave"


@pytest.mark.asyncio
async def test_enclave_sealed_openrouter_job_never_calls_parent_kms():
    vault = EncryptedArtifactVaultV2(
        master_key=b"v" * 32,
        boot_identity_hash="sha256:" + "d" * 64,
    )
    stored = seal_openrouter_persistent_credential_v2(
        credential="sk-or-v1-" + "r" * 32,
        credential_kind="runtime",
        key_ref="encrypted_ref:openrouter:" + "e" * 32,
        miner_hotkey="miner-hotkey",
        job_id="registration-job",
        purpose="research_lab.openrouter_credential.v2",
        vault=vault,
    )
    envelope = build_openrouter_sealed_job_envelope_v2(
        stored,
        job_id="autoresearch-v2:job-1",
    )
    client = _Client()
    result = await provision_job_credential_envelope_v2(
        envelope,
        client=client,
        kms_client=pytest.fail,
    )
    assert result["status"] == "ready"
    assert client.sealed_openrouter["job_id"] == "autoresearch-v2:job-1"
