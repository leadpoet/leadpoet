from __future__ import annotations

import base64

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.tee.provider_evidence_v2 import (
    REQUEST_SCHEMA_VERSION,
    ProviderEvidenceAuthorityV2,
    ProviderEvidenceV2Error,
    create_signed_provider_evidence_record,
    validate_signed_provider_evidence_record,
)
from gateway.tee.autoresearch_executor_v2 import AutoresearchExecutorV2
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.source_add_runtime_v2 import (
    build_source_add_runtime_catalog_v2,
    source_add_dynamic_retry_policy_hash,
)
from leadpoet_canonical.attested_v2 import (
    build_transport_attempt,
    sha256_bytes,
    sha256_json,
)


HASH = "sha256:" + "a" * 64
REQUEST_ARTIFACT = "sha256:" + "b" * 64
RESPONSE_ARTIFACT = "sha256:" + "c" * 64
ATTEMPT_HASH = "sha256:" + "d" * 64


def _identity(private_key: Ed25519PrivateKey):
    pubkey = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    return {
        "boot_identity_hash": HASH,
        "signing_pubkey": pubkey,
    }


def _request(*, live_enabled: bool = True):
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "caller_job_id": "autoresearch-v2:probe-job",
        "purpose": "research_lab.candidate_decision.v2",
        "endpoint": {
            "endpoint_id": "exa.search",
            "provider_id": "exa",
            "method": "POST",
            "path": "/search",
            "params": [
                {
                    "name": "query",
                    "type": "string",
                    "required": True,
                    "location": "body",
                    "max_length": 300,
                }
            ],
            "est_cost_microusd": 5000,
            "description": "probe",
        },
        "upstream_base_url": "https://api.exa.ai",
        "query_params": {},
        "body_params": {"query": "developer tools"},
        "live_enabled": live_enabled,
        "timeout_seconds": 60,
    }


class _Broker:
    retry_policy_hashes = {"exa": HASH}

    def __init__(self):
        self.calls = []

    def execute(self, request):
        self.calls.append(dict(request))
        return {
            "terminal_status": "authenticated_response",
            "http_status": 200,
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(b'{"results":[]}').decode("ascii"),
            "encrypted_request_artifact_id": REQUEST_ARTIFACT,
            "encrypted_artifact_id": RESPONSE_ARTIFACT,
            "transport_attempt": {"attempt_hash": ATTEMPT_HASH},
        }


def test_provider_evidence_records_once_then_replays_signed_source():
    key = Ed25519PrivateKey.generate()
    identity = _identity(key)
    broker = _Broker()
    authority = ProviderEvidenceAuthorityV2(
        broker=broker,
        boot_identity_supplier=lambda: identity,
        sign_digest=key.sign,
        clock=lambda: "2026-07-10T00:00:00Z",
    )

    live = authority.resolve(_request())
    hit = authority.resolve(_request())

    assert len(broker.calls) == 1
    assert live["evidence"] == "recorded"
    assert hit["evidence"] == "hit"
    assert hit["source_record"]["record_hash"] == live["record"]["record_hash"]
    assert hit["record"]["source_record_hash"] == live["record"]["record_hash"]
    assert hit["body_b64"] == live["body_b64"]
    validate_signed_provider_evidence_record(live["record"], boot_identity=identity)
    validate_signed_provider_evidence_record(hit["record"], boot_identity=identity)


def test_provider_evidence_replay_miss_never_calls_provider():
    key = Ed25519PrivateKey.generate()
    identity = _identity(key)
    broker = _Broker()
    authority = ProviderEvidenceAuthorityV2(
        broker=broker,
        boot_identity_supplier=lambda: identity,
        sign_digest=key.sign,
    )

    result = authority.resolve(_request(live_enabled=False))

    assert result["status"] == 409
    assert result["evidence"] == "replay_miss"
    assert result["transport_attempts"] == []
    assert broker.calls == []


def test_provider_evidence_rejects_tampered_record_and_route():
    key = Ed25519PrivateKey.generate()
    identity = _identity(key)
    authority = ProviderEvidenceAuthorityV2(
        broker=_Broker(),
        boot_identity_supplier=lambda: identity,
        sign_digest=key.sign,
    )
    result = authority.resolve(_request())
    tampered = {**result["record"], "status": 201}
    with pytest.raises(ProviderEvidenceV2Error, match="hash mismatch"):
        validate_signed_provider_evidence_record(tampered, boot_identity=identity)

    request = _request()
    request["upstream_base_url"] = "https://attacker.invalid"
    with pytest.raises(ProviderEvidenceV2Error, match="base URL differs"):
        authority.resolve(request)


def test_provider_evidence_rejects_params_moved_between_query_and_body():
    key = Ed25519PrivateKey.generate()
    identity = _identity(key)
    authority = ProviderEvidenceAuthorityV2(
        broker=_Broker(),
        boot_identity_supplier=lambda: identity,
        sign_digest=key.sign,
    )
    request = _request()
    request["body_params"] = {}
    request["query_params"] = {"query": "developer tools"}
    with pytest.raises(ProviderEvidenceV2Error, match="params differ"):
        authority.resolve(request)


def test_provider_evidence_routes_dynamic_source_only_with_measured_route():
    key = Ed25519PrivateKey.generate()
    identity = _identity(key)
    broker = _Broker()
    authority = ProviderEvidenceAuthorityV2(
        broker=broker,
        boot_identity_supplier=lambda: identity,
        sign_digest=key.sign,
        clock=lambda: "2026-07-10T00:00:00Z",
    )
    row = {
        "adapter_id": "adapter:public-source",
        "miner_hotkey": "miner-one",
        "provision_status": "provisioned_autoresearch_eligible",
        "registry_provider_id": "public_source",
        "credential_envelope": {},
        "provision_doc": {
            "provider_registry_entry": {
                "id": "public_source",
                "base_url": "https://api.public-source.example",
                "auth_kind": "none",
                "auth_name": "",
                "credential_ref": [],
                "per_day_quota": 10,
                "cost_model": {"est_cost_microusd_per_call": 0},
                "capability_policy": {
                    "routes": [{"method": "POST", "path": "/search"}]
                },
            },
            "probe_endpoints": [
                {
                    "endpoint_id": "public_source.search",
                    "provider_id": "public_source",
                    "method": "POST",
                    "path": "/search",
                    "params": [
                        {
                            "name": "query",
                            "type": "string",
                            "required": True,
                            "location": "body",
                            "max_length": 300,
                        }
                    ],
                }
            ],
        },
    }
    route = build_source_add_runtime_catalog_v2([row])["routes"][0]
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "caller_job_id": "autoresearch-v2:dynamic-probe",
        "purpose": "research_lab.candidate_decision.v2",
        "endpoint": row["provision_doc"]["probe_endpoints"][0],
        "upstream_base_url": route["base_url"],
        "query_params": {},
        "body_params": {"query": "developer tools"},
        "live_enabled": True,
        "timeout_seconds": 60,
        "dynamic_route": route,
    }

    result = authority.resolve(request)

    assert result["evidence"] == "recorded"
    assert broker.calls[0]["provider_id"] == "public_source"
    assert broker.calls[0]["dynamic_route"]["route_hash"] == route["route_hash"]
    assert broker.calls[0]["retry_policy_hash"] == (
        source_add_dynamic_retry_policy_hash(route)
    )

    tampered = {
        **route,
        "allowed_routes": [{"method": "POST", "path": "/unlisted"}],
    }
    with pytest.raises(ProviderEvidenceV2Error, match="route is invalid"):
        authority.resolve({**request, "dynamic_route": tampered})


def test_autoresearch_accepts_restored_cache_lineage_and_multiple_cache_attempts():
    key = Ed25519PrivateKey.generate()
    identity = _identity(key)
    probe_value = {
        key: value
        for key, value in _request().items()
        if key not in {"schema_version", "caller_job_id", "purpose"}
    }
    full_request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "caller_job_id": "autoresearch-v2:probe-job",
        "purpose": "research_lab.candidate_decision.v2",
        **probe_value,
    }
    body = b'{"results":[]}'
    fingerprint = "f" * 64
    cache_entry_hash = "sha256:" + "9" * 64
    source = create_signed_provider_evidence_record(
        body={
            "schema_version": "leadpoet.provider_evidence_resolution.v2",
            "coordinator_boot_identity_hash": identity["boot_identity_hash"],
            "request_hash": HASH,
            "request_fingerprint": fingerprint,
            "evidence": "restored",
            "status": 200,
            "body_hash": sha256_bytes(body),
            "encrypted_request_artifact_id": REQUEST_ARTIFACT,
            "encrypted_response_artifact_id": RESPONSE_ARTIFACT,
            "transport_attempt_hash": "",
            "source_record_hash": cache_entry_hash,
            "issued_at": "2026-07-10T00:00:00Z",
        },
        coordinator_pubkey=identity["signing_pubkey"],
        sign_digest=key.sign,
    )
    hit = create_signed_provider_evidence_record(
        body={
            "schema_version": "leadpoet.provider_evidence_resolution.v2",
            "coordinator_boot_identity_hash": identity["boot_identity_hash"],
            "request_hash": sha256_json(full_request),
            "request_fingerprint": fingerprint,
            "evidence": "hit",
            "status": 200,
            "body_hash": sha256_bytes(body),
            "encrypted_request_artifact_id": "",
            "encrypted_response_artifact_id": "",
            "transport_attempt_hash": "",
            "source_record_hash": source["record_hash"],
            "issued_at": "2026-07-10T00:00:01Z",
        },
        coordinator_pubkey=identity["signing_pubkey"],
        sign_digest=key.sign,
    )
    attempts = []
    for ordinal, method in enumerate(("GET", "POST")):
        attempts.append(
            build_transport_attempt(
                request_id=("1" if ordinal == 0 else "2") * 32,
                logical_operation_id="cache:%d" % ordinal,
                job_id="autoresearch-v2:probe-job",
                purpose="research_lab.candidate_decision.v2",
                provider_id="supabase",
                attempt_number=ordinal,
                method=method,
                destination_host="qplwoislplkcegvdmbim.supabase.co",
                destination_port=443,
                path_hash=HASH,
                nonsecret_headers_hash=HASH,
                body_hash=HASH,
                credential_ref_hash=HASH,
                retry_policy_hash=HASH,
                timeout_ms=45000,
                started_at="2026-07-10T00:00:00Z",
                terminal_status="authenticated_response",
                http_status=200,
                response_hash=HASH,
                request_artifact_hash=HASH,
                response_artifact_hash=HASH,
                tls_peer_chain_hash=HASH,
                tls_protocol="TLSv1.3",
                failure_code=None,
                completed_at="2026-07-10T00:00:01Z",
            )
        )
    result = {
        "status": 200,
        "body_b64": base64.b64encode(body).decode("ascii"),
        "evidence": "hit",
        "transport_attempts": attempts,
        "evidence_artifact_hashes": [
            hit["record_hash"],
            source["record_hash"],
            identity["boot_identity_hash"],
            cache_entry_hash,
        ],
        "record": hit,
        "source_record": source,
        "source_boot_identity": identity,
        "coordinator_boot_identity": identity,
    }
    executor = AutoresearchExecutorV2(
        provider_execute=lambda _request: {},
        retry_policy_hashes={},
        probe_execute=lambda _request: result,
        coordinator_boot_verifier=lambda value: value,
        artifact_seal=lambda **_kwargs: {},
    )
    context = ExecutionContextV2(
        job_id="autoresearch-v2:probe-job",
        purpose="research_lab.candidate_decision.v2",
        epoch_id=1,
    )
    try:
        resolved = executor._probe_resolver(context)(probe_value)
    finally:
        executor.close()

    assert resolved["evidence"] == "hit"
    assert resolved["body"] == body
    assert len(context.transport_attempts) == 2
    assert cache_entry_hash in context.artifact_hashes
