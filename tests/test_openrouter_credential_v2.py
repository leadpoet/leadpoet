import base64
import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from gateway.research_lab.models import ResearchLabOpenRouterKeyRegisterRequest
from gateway.research_lab.models import (
    AttestedCredentialCiphertextV2,
    ResearchLabOpenRouterCredentialRecipientRequest,
)
from gateway.research_lab import api
from gateway.research_lab import maintenance
from types import SimpleNamespace
import time
from gateway.tee.artifact_vault_v2 import EncryptedArtifactVaultV2
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.kms_recipient_v2 import KMSRecipientV2, KMSRecipientV2Error
from gateway.tee.openrouter_credential_v2 import (
    OPENROUTER_REGISTRATION_REQUEST_SCHEMA_VERSION,
    OpenRouterRegistrationAuthorityV2,
    build_openrouter_sealed_job_envelope_v2,
    seal_openrouter_ingress_credential_v2,
    unseal_openrouter_job_credential_v2,
    validate_openrouter_ingress_envelope_v2,
)
from gateway.tee.provider_broker_v2 import (
    ProviderBrokerV2,
    credential_value_hash,
)
from gateway.tee.provider_client_v2 import BrokeredProviderTransportV2
from leadpoet_canonical.attested_v2 import canonical_json


HASH = "sha256:" + "a" * 64
BOOT_HASH = "sha256:" + "b" * 64
MINER = "5" + "M" * 47
RUNTIME_KEY = "sk-or-v1-" + "r" * 32
MANAGEMENT_KEY = "sk-or-v1-" + "m" * 32


def _vault():
    return EncryptedArtifactVaultV2(
        master_key=b"v" * 32,
        boot_identity_hash=BOOT_HASH,
    )


def _recipient():
    return KMSRecipientV2(
        boot_identity_supplier=lambda: {"boot_identity_hash": BOOT_HASH},
        expected_credential_ref_hashes={"openrouter": HASH},
        expected_job_slot_ref_hashes={"openrouter_management": HASH},
        attestation_supplier=lambda **_kwargs: b"attestation",
    )


def _client_encrypt(request, plaintext):
    public_key = serialization.load_der_public_key(
        base64.b64decode(request["recipient_public_key_der_b64"], validate=True)
    )
    ciphertext = public_key.encrypt(
        plaintext.encode("utf-8"),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return base64.b64encode(ciphertext).decode("ascii")


def _ingress_envelope(recipient, vault, kind, credential):
    request = recipient.openrouter_ingress_recipient_request(
        miner_hotkey=MINER,
        credential_kind=kind,
    )
    lease = recipient.unwrap_openrouter_ingress_credential(
        request_id=request["request_id"],
        ciphertext_b64=_client_encrypt(request, credential),
    )
    return seal_openrouter_ingress_credential_v2(lease, vault=vault)


def test_openrouter_ingress_is_one_use_and_parent_envelope_has_no_plaintext():
    recipient = _recipient()
    vault = _vault()
    request = recipient.openrouter_ingress_recipient_request(
        miner_hotkey=MINER,
        credential_kind="runtime",
    )
    ciphertext = _client_encrypt(request, RUNTIME_KEY)
    lease = recipient.unwrap_openrouter_ingress_credential(
        request_id=request["request_id"],
        ciphertext_b64=ciphertext,
    )
    envelope = seal_openrouter_ingress_credential_v2(lease, vault=vault)
    validate_openrouter_ingress_envelope_v2(envelope)
    assert RUNTIME_KEY not in canonical_json(envelope)
    assert envelope["credential_value_hash"] == credential_value_hash(RUNTIME_KEY)
    with pytest.raises(KMSRecipientV2Error, match="already used"):
        recipient.unwrap_openrouter_ingress_credential(
            request_id=request["request_id"],
            ciphertext_b64=ciphertext,
        )


def test_openrouter_seal_rpc_retry_is_idempotent_and_rejects_rebinding(
    monkeypatch,
):
    from gateway.tee import openrouter_credential_v2 as credential_module
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "gateway" / "tee")
    )
    from gateway.tee import tee_service

    class _Recipient:
        calls = 0

        def unwrap_openrouter_ingress_credential(self, **kwargs):
            self.calls += 1
            return {"request_id": kwargs["request_id"]}

    recipient = _Recipient()
    monkeypatch.setattr(tee_service, "get_v2_runtime_identity", object)
    monkeypatch.setattr(tee_service, "get_v2_kms_recipient", lambda: recipient)
    monkeypatch.setattr(tee_service, "get_v2_artifact_vault", object)
    monkeypatch.setattr(
        credential_module,
        "seal_openrouter_ingress_credential_v2",
        lambda lease, **_kwargs: {"sealed_request_id": lease["request_id"]},
    )
    tee_service.v2_ingress_seal_cache.clear()
    params = {
        "request_id": HASH,
        "ciphertext_b64": base64.b64encode(b"ciphertext-one").decode("ascii"),
    }
    first = tee_service.handle_v2_runtime_rpc(
        "v2_seal_openrouter_ingress_credential", params
    )
    second = tee_service.handle_v2_runtime_rpc(
        "v2_seal_openrouter_ingress_credential", params
    )
    assert first == second
    assert recipient.calls == 1

    changed = {
        **params,
        "ciphertext_b64": base64.b64encode(b"ciphertext-two").decode("ascii"),
    }
    with pytest.raises(ValueError, match="ciphertext changed"):
        tee_service.handle_v2_runtime_rpc(
            "v2_seal_openrouter_ingress_credential", changed
        )
    tee_service.v2_ingress_seal_cache.clear()


def test_registration_executes_unchanged_provider_sequence_and_seals_jobs():
    recipient = _recipient()
    vault = _vault()
    runtime_envelope = _ingress_envelope(
        recipient, vault, "runtime", RUNTIME_KEY
    )
    management_envelope = _ingress_envelope(
        recipient, vault, "management", MANAGEMENT_KEY
    )
    observed = []

    def transport(**request):
        parsed = urlsplit(request["url"])
        observed.append(
            (
                request["method"],
                parsed.path,
                parsed.query,
                request["headers"].get("Authorization"),
            )
        )
        if parsed.path == "/api/v1/key":
            body = {
                "data": {
                    "hash": "runtime-provider-hash",
                    "label": "miner-runtime",
                    "creator_user_id": "user-1",
                    "limit": 100,
                    "limit_remaining": 80,
                    "usage": 20,
                    "disabled": False,
                }
            }
        elif parsed.path == "/api/v1/workspaces" and request["method"] == "GET":
            body = {"data": [{"id": "workspace-1"}]}
        elif parsed.path == "/api/v1/keys":
            body = {
                "data": [
                    {
                        "label": "miner-runtime",
                        "creator_user_id": "user-1",
                    }
                ]
            }
        elif request["method"] == "PATCH":
            body = {"data": {"id": "workspace-1"}}
        else:
            body = {
                "data": {
                    "id": "workspace-1",
                    "is_observability_io_logging_enabled": False,
                    "is_data_discount_logging_enabled": False,
                    "is_observability_broadcast_enabled": False,
                    "io_logging_api_key_ids": [],
                }
            }
        return {
            "http_status": 200,
            "headers": {"content-type": "application/json"},
            "body": json.dumps(body, sort_keys=True).encode("utf-8"),
            "tls_peer_chain_hash": "sha256:" + "c" * 64,
            "tls_protocol": "TLSv1.3",
        }

    broker = ProviderBrokerV2(
        credential_ref_hashes={},
        retry_policy_hashes={
            "openrouter": HASH,
            "openrouter_management": HASH,
        },
        transport=transport,
        artifact_sink=lambda plaintext, **scope: vault.seal(plaintext, **scope),
    )
    intercepted = BrokeredProviderTransportV2(broker.execute)
    authority = OpenRouterRegistrationAuthorityV2(
        broker=broker,
        transport=intercepted,
        retry_policy_hashes={
            "openrouter": HASH,
            "openrouter_management": HASH,
        },
        vault=vault,
    )
    context = ExecutionContextV2(
        job_id="openrouter-registration-test",
        purpose="research_lab.openrouter_credential.v2",
        epoch_id=1,
        provider_credential_ref_hashes={
            "openrouter": credential_value_hash(RUNTIME_KEY),
            "openrouter_management": credential_value_hash(MANAGEMENT_KEY),
        },
    )
    try:
        result = authority.execute(
            {
                "schema_version": OPENROUTER_REGISTRATION_REQUEST_SCHEMA_VERSION,
                "miner_hotkey": MINER,
                "key_label": "miner key",
                "runtime_credential": runtime_envelope,
                "management_credential": management_envelope,
            },
            context,
        )
    finally:
        intercepted.restore()

    assert [(method, path) for method, path, _query, _auth in observed] == [
        ("GET", "/api/v1/key"),
        ("GET", "/api/v1/key"),
        ("GET", "/api/v1/workspaces"),
        ("GET", "/api/v1/keys"),
        ("PATCH", "/api/v1/workspaces/workspace-1"),
        ("GET", "/api/v1/workspaces/workspace-1"),
    ]
    assert observed[0][3] == "Bearer " + RUNTIME_KEY
    assert observed[2][3] == "Bearer " + MANAGEMENT_KEY
    assert len(result.transport_attempts) == 6
    assert broker.health()["job_credential_lease_count"] == 0
    assert RUNTIME_KEY not in canonical_json(result.output)
    assert MANAGEMENT_KEY not in canonical_json(result.output)
    envelopes = {
        item["credential_kind"]: item
        for item in result.output["credential_envelopes"]
    }
    job_envelope = build_openrouter_sealed_job_envelope_v2(
        envelopes["runtime"],
        job_id="later-job",
    )
    lease = unseal_openrouter_job_credential_v2(job_envelope, vault=vault)
    assert lease["credential"] == RUNTIME_KEY
    assert lease["job_id"] == "later-job"


def test_registration_request_signature_never_contains_plaintext_keys():
    now = __import__("time").time()
    payload = ResearchLabOpenRouterKeyRegisterRequest(
        miner_hotkey=MINER,
        signature="s" * 64,
        timestamp=int(now),
        idempotency_key="openrouter-registration-1",
        openrouter_api_key=RUNTIME_KEY,
        openrouter_management_key=MANAGEMENT_KEY,
    )
    signed = canonical_json(payload.signed_payload())
    assert RUNTIME_KEY not in signed
    assert MANAGEMENT_KEY not in signed


@pytest.mark.asyncio
async def test_openrouter_recipient_route_returns_two_scoped_attestations(monkeypatch):
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                miner_submissions_enabled=True,
            )
        ),
    )
    monkeypatch.setattr(api, "_verify_signed_miner", lambda _payload: _async_none())

    async def recipient(*, miner_hotkey, credential_kind):
        return {
            "schema_version": "leadpoet.openrouter_ingress_recipient.v2",
            "purpose": "leadpoet.openrouter_credential_ingress.v2",
            "request_id": "sha256:" + ("1" if credential_kind == "runtime" else "2") * 64,
            "boot_identity_hash": BOOT_HASH,
            "miner_hotkey_hash": "sha256:" + "3" * 64,
            "credential_kind": credential_kind,
            "credential_slot": (
                "openrouter"
                if credential_kind == "runtime"
                else "openrouter_management"
            ),
            "recipient_public_key_hash": "sha256:" + "4" * 64,
            "request_nonce": "5" * 32,
            "recipient_public_key_der_b64": base64.b64encode(b"public").decode(),
            "attestation_document_b64": base64.b64encode(b"attestation").decode(),
            "key_encryption_algorithm": "RSAES_OAEP_SHA_256",
        }

    monkeypatch.setattr(api, "_openrouter_credential_recipient_v2", recipient)
    payload = ResearchLabOpenRouterCredentialRecipientRequest(
        miner_hotkey=MINER,
        signature="s" * 64,
        timestamp=int(time.time()),
        idempotency_key="openrouter-recipient-1",
    )
    response = await api.create_openrouter_credential_recipients(payload)
    assert response.runtime.credential_slot == "openrouter"
    assert response.management.credential_slot == "openrouter_management"


@pytest.mark.asyncio
async def test_openrouter_registration_route_rejects_plaintext_before_network(monkeypatch):
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                miner_submissions_enabled=True,
            )
        ),
    )
    monkeypatch.setattr(api, "_verify_signed_miner", lambda _payload: _async_none())
    monkeypatch.setattr(
        api,
        "_enforce_openrouter_key_registration_rate_limit",
        lambda _hotkey: None,
    )
    payload = ResearchLabOpenRouterKeyRegisterRequest(
        miner_hotkey=MINER,
        signature="s" * 64,
        timestamp=int(time.time()),
        idempotency_key="openrouter-registration-plaintext",
        openrouter_api_key=RUNTIME_KEY,
        openrouter_management_key=MANAGEMENT_KEY,
    )
    with pytest.raises(api.HTTPException, match="plaintext OpenRouter"):
        await api.register_research_lab_openrouter_key(payload)


@pytest.mark.asyncio
async def test_openrouter_registration_route_persists_only_attested_envelopes(
    monkeypatch,
):
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                miner_submissions_enabled=True,
                evaluation_epoch=123,
            )
        ),
    )
    monkeypatch.setattr(api, "_verify_signed_miner", lambda _payload: _async_none())
    monkeypatch.setattr(
        api,
        "_enforce_openrouter_key_registration_rate_limit",
        lambda _hotkey: None,
    )
    monkeypatch.setattr(
        api,
        "resolve_research_lab_evaluation_epoch",
        lambda _configured: _async_value((123, 456, "test")),
    )
    sealed = {}

    async def seal(*, encrypted, miner_hotkey, credential_kind):
        sealed[credential_kind] = encrypted.request_id
        return {
            "credential_kind": credential_kind,
            "credential_slot": (
                "openrouter"
                if credential_kind == "runtime"
                else "openrouter_management"
            ),
            "credential_value_hash": "sha256:" + (
                "6" if credential_kind == "runtime" else "7"
            ) * 64,
            "miner_hotkey_hash": "sha256:" + "8" * 64,
            "envelope_hash": "sha256:" + (
                "9" if credential_kind == "runtime" else "a"
            ) * 64,
            "ciphertext_blob_hash": "sha256:" + (
                "b" if credential_kind == "runtime" else "c"
            ) * 64,
        }

    monkeypatch.setattr(api, "_seal_openrouter_credential_v2", seal)
    key_ref = "encrypted_ref:openrouter:" + "d" * 32
    envelopes = [
        {
            "schema_version": "leadpoet.provider_credential_envelope.enclave.v2",
            "envelope_hash": "sha256:" + "e" * 64,
            "key_ref": key_ref,
            "key_ref_hash": "sha256:" + "f" * 64,
            "miner_hotkey_hash": "sha256:" + "1" * 64,
            "credential_kind": kind,
            "credential_slot": slot,
            "credential_value_hash": "sha256:" + digit * 64,
            "ciphertext_blob_b64": base64.b64encode((kind + "-sealed").encode()).decode(),
            "ciphertext_blob_hash": "sha256:" + ("4" if kind == "runtime" else "5") * 64,
            "kms_key_id_hash": "sha256:" + "6" * 64,
            "encryption_context": {"purpose": "sealed"},
            "encryption_context_hash": "sha256:" + "7" * 64,
        }
        for kind, slot, digit in (
            ("runtime", "openrouter", "2"),
            ("management", "openrouter_management", "3"),
        )
    ]

    async def register(**kwargs):
        assert kwargs["miner_hotkey"] == MINER
        return {
            "result": {
                "schema_version": "leadpoet.openrouter_credential_registration_result.v2",
                "key_ref": key_ref,
                "key_hash": "provider-key-hash",
                "management_key_hash": "management-key-hash",
                "preflight_doc": {"limit_remaining": 10, "limit_reset": None},
                "privacy_proof_doc": {
                    "workspace_id_hash": "workspace-hash",
                    "verified_at": "2026-07-10T00:00:00+00:00",
                },
                "credential_envelopes": envelopes,
            }
        }

    import gateway.research_lab.attested_coordinator_v2 as coordinator_bridge

    monkeypatch.setattr(
        coordinator_bridge,
        "register_openrouter_credentials_v2",
        register,
    )
    stored = {}
    persisted = []

    async def create_ref(**kwargs):
        stored.update(kwargs)
        return kwargs

    async def persist(envelope):
        persisted.append(dict(envelope))
        return dict(envelope)

    monkeypatch.setattr(api, "create_openrouter_key_ref", create_ref)
    monkeypatch.setattr(api, "persist_openrouter_credential_envelope_v2", persist)
    encrypted = AttestedCredentialCiphertextV2(
        request_id="sha256:" + "8" * 64,
        ciphertext_b64=base64.b64encode(b"x" * 384).decode(),
    )
    payload = ResearchLabOpenRouterKeyRegisterRequest(
        miner_hotkey=MINER,
        signature="s" * 64,
        timestamp=int(time.time()),
        idempotency_key="openrouter-registration-v2",
        openrouter_api_key_v2=encrypted,
        openrouter_management_key_v2=encrypted.model_copy(
            update={"request_id": "sha256:" + "9" * 64}
        ),
    )
    response = await api.register_research_lab_openrouter_key(payload)
    assert response.key_ref == key_ref
    assert sealed == {
        "runtime": "sha256:" + "8" * 64,
        "management": "sha256:" + "9" * 64,
    }
    assert len(persisted) == 2
    assert stored["encrypted_key_ciphertext"] == envelopes[0]["ciphertext_blob_b64"]
    assert stored["encrypted_management_key_ciphertext"] == envelopes[1]["ciphertext_blob_b64"]
    assert RUNTIME_KEY not in canonical_json(stored)
    assert MANAGEMENT_KEY not in canonical_json(stored)


async def _async_none():
    return None


async def _async_value(value):
    return value


def test_attested_credit_preflight_uses_leased_key_and_one_authenticated_call():
    vault = _vault()
    observed = []

    def transport(**request):
        observed.append(dict(request))
        body = {
            "data": {
                "hash": "runtime-provider-hash",
                "limit": 100,
                "limit_remaining": 25,
                "disabled": False,
            }
        }
        return {
            "http_status": 200,
            "headers": {"content-type": "application/json"},
            "body": json.dumps(body).encode("utf-8"),
            "tls_peer_chain_hash": "sha256:" + "c" * 64,
            "tls_protocol": "TLSv1.3",
        }

    broker = ProviderBrokerV2(
        credential_ref_hashes={},
        retry_policy_hashes={"openrouter": HASH},
        transport=transport,
        artifact_sink=lambda plaintext, **scope: vault.seal(plaintext, **scope),
    )
    value_hash = credential_value_hash(RUNTIME_KEY)
    broker.provision_job_credential(
        job_id="credit-preflight-job",
        slot="openrouter",
        credential=RUNTIME_KEY,
        credential_value_hash_expected=value_hash,
    )
    intercepted = BrokeredProviderTransportV2(broker.execute)
    authority = OpenRouterRegistrationAuthorityV2(
        broker=broker,
        transport=intercepted,
        retry_policy_hashes={"openrouter": HASH},
        vault=vault,
    )
    context = ExecutionContextV2(
        job_id="credit-preflight-job",
        purpose="research_lab.openrouter_credit_preflight.v2",
        epoch_id=1,
        provider_credential_ref_hashes={"openrouter": value_hash},
    )
    try:
        result = authority.preflight(
            {
                "schema_version": "leadpoet.openrouter_credit_preflight_request.v2",
                "key_ref_hash": "sha256:" + "d" * 64,
                "miner_hotkey_hash": "sha256:" + "e" * 64,
                "credential_value_hash": value_hash,
            },
            context,
        )
    finally:
        intercepted.restore()
        broker.release_job_credentials("credit-preflight-job")
    assert len(result.transport_attempts) == 1
    assert result.output["preflight_doc"]["limit_remaining"] == 25
    assert observed[0]["headers"]["Authorization"] == "Bearer " + RUNTIME_KEY
    assert RUNTIME_KEY not in canonical_json(result.output)


@pytest.mark.asyncio
async def test_maintenance_credit_preflight_uses_attested_coordinator(monkeypatch):
    monkeypatch.setattr(
        maintenance,
        "select_one",
        lambda *_args, **_kwargs: _async_value(
            {
                "ticket_id": "ticket-1",
                "miner_hotkey": MINER,
                "miner_openrouter_key_ref": "encrypted_ref:openrouter:" + "f" * 32,
                "miner_openrouter_key_handling": "encrypted_ref",
            }
        ),
    )
    monkeypatch.setattr(
        maintenance.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(lambda: SimpleNamespace(evaluation_epoch=123)),
    )
    import gateway.research_lab.chain as chain
    import gateway.research_lab.attested_coordinator_v2 as coordinator_bridge

    monkeypatch.setattr(
        chain,
        "resolve_research_lab_evaluation_epoch",
        lambda _configured: _async_value((123, 456, "test")),
    )
    observed = {}

    async def preflight(**kwargs):
        observed.update(kwargs)
        return {
            "result": {
                "preflight_doc": {
                    "key_hash": "provider-key-hash",
                    "limit": 100,
                    "limit_remaining": 12,
                    "limit_reset": None,
                }
            }
        }

    monkeypatch.setattr(
        coordinator_bridge,
        "preflight_openrouter_key_ref_v2",
        preflight,
    )
    result = await maintenance._preflight_openrouter_key_for_run("ticket-1")
    assert result["ok"] is True
    assert result["limit_remaining"] == 12
    assert observed["miner_hotkey"] == MINER
