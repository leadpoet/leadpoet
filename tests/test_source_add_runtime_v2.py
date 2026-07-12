from __future__ import annotations

import base64

import pytest

from gateway.tee.provider_broker_v2 import (
    BUILTIN_PROVIDER_ROUTES,
    PROVIDER_BROKER_SCHEMA_VERSION,
    ProviderBrokerV2,
    ProviderBrokerV2Error,
    credential_reference_hash,
    credential_value_hash,
)
from gateway.tee.source_add_runtime_v2 import (
    SourceAddRuntimeV2Error,
    build_source_add_job_envelope_v2,
    build_source_add_runtime_catalog_v2,
    source_add_dynamic_retry_policy_hash,
    source_add_route_for_url_v2,
    validate_source_add_runtime_catalog_v2,
    validate_source_add_runtime_route_v2,
)
from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json


NOW = "2026-07-10T20:00:00Z"


def _source_row(*, auth_kind: str = "header") -> dict:
    credential_ref = "encrypted_ref:source_add:" + "a" * 32
    credential = "source-add-secret"
    ciphertext = b"encrypted-source-add-secret"
    context = {
        "adapter_ref": "source_add:adapter:source-one",
        "miner_hotkey": "miner-hotkey-one",
        "purpose": "leadpoet_research_lab_source_add_credential",
    }
    envelope = {
        "schema_version": "leadpoet.source_add_credential_envelope.v2",
        "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
        "ciphertext_blob_hash": sha256_bytes(ciphertext),
        "kms_key_id": "arn:aws:kms:us-east-1:111122223333:key/test-key",
        "kms_key_id_hash": "sha256:" + "b" * 64,
        "encryption_context": context,
        "encryption_context_hash": sha256_json(context),
        "credential_ref": credential_ref,
        "credential_value_hash": credential_value_hash(credential),
        "key_ref_hash": sha256_bytes(credential_ref.encode("utf-8")),
    }
    return {
        "adapter_id": "adapter:source-one",
        "miner_hotkey": "miner-hotkey-one",
        "provision_status": "provisioned_autoresearch_eligible",
        "registry_provider_id": "source_one",
        "credential_envelope": envelope if auth_kind != "none" else {},
        "provision_doc": {
            "provider_registry_entry": {
                "id": "source_one",
                "base_url": "https://api.source-one.example/v1",
                "auth_kind": auth_kind,
                "auth_name": "x-source-key" if auth_kind != "none" else "",
                "credential_ref": (
                    ["SOURCE_ONE_API_KEY"] if auth_kind != "none" else []
                ),
                "per_day_quota": 7,
                "cost_model": {"est_cost_microusd_per_call": 1250},
                "capability_policy": {
                    "routes": [
                        {"method": "POST", "path": "/search"},
                        {"method": "GET", "path": "/status"},
                    ]
                },
            },
            "probe_endpoints": [
                {
                    "endpoint_id": "source_one.search",
                    "provider_id": "source_one",
                    "method": "POST",
                    "path": "/search",
                    "params": [],
                },
                {
                    "endpoint_id": "source_one.status",
                    "provider_id": "source_one",
                    "method": "GET",
                    "path": "/status",
                    "params": [],
                },
            ],
        },
    }


def _broker(transport):
    credentials = {
        "openrouter": "openrouter-secret",
        "exa": "exa-secret",
        "scrapingdog": "scrapingdog-secret",
        "deepline": "deepline-secret",
        "supabase_service_role": "supabase-secret",
        "truelist": "truelist-secret",
    }
    broker = ProviderBrokerV2(
        credential_ref_hashes={
            slot: credential_reference_hash(value)
            for slot, value in credentials.items()
        },
        retry_policy_hashes={
            provider: "sha256:" + "c" * 64
            for provider in BUILTIN_PROVIDER_ROUTES
        },
        transport=transport,
        artifact_sink=lambda body, **_: {
            "artifact_id": sha256_bytes(b"artifact:" + body),
            "plaintext_hash": sha256_bytes(body),
        },
        clock=lambda: NOW,
    )
    broker.provision_credentials(credentials)
    return broker


def test_runtime_catalog_binds_exact_route_and_credential_context():
    catalog = build_source_add_runtime_catalog_v2([_source_row()])
    normalized = validate_source_add_runtime_catalog_v2(catalog)
    route = normalized["routes"][0]

    assert route["provider_id"] == "source_one"
    assert route["credential_value_hash"] == credential_value_hash(
        "source-add-secret"
    )
    assert source_add_route_for_url_v2(
        catalog,
        "POST",
        "https://api.source-one.example/v1/search?q=allowed",
    ) == route
    assert source_add_route_for_url_v2(
        catalog,
        "GET",
        "https://api.source-one.example/v1/search",
    ) is None
    assert source_add_route_for_url_v2(
        catalog,
        "POST",
        "https://api.source-one.example/v1/unlisted",
    ) is None


def test_runtime_route_rejects_row_context_and_route_tampering():
    row = _source_row()
    row["credential_envelope"]["encryption_context"] = {
        **row["credential_envelope"]["encryption_context"],
        "miner_hotkey": "different-miner",
    }
    row["credential_envelope"]["encryption_context_hash"] = sha256_json(
        row["credential_envelope"]["encryption_context"]
    )
    with pytest.raises(SourceAddRuntimeV2Error, match="context differs"):
        build_source_add_runtime_catalog_v2([row])

    route = build_source_add_runtime_catalog_v2([_source_row()])["routes"][0]
    changed = {**route, "per_day_quota": route["per_day_quota"] + 1}
    with pytest.raises(SourceAddRuntimeV2Error, match="route hash differs"):
        validate_source_add_runtime_route_v2(changed)


def test_runtime_route_rejects_unsafe_auth_header():
    row = _source_row()
    row["provision_doc"]["provider_registry_entry"]["auth_name"] = "Host"
    with pytest.raises(SourceAddRuntimeV2Error, match="auth header is unsafe"):
        build_source_add_runtime_catalog_v2([row])


def test_dynamic_provider_uses_only_job_credential_and_exact_route():
    calls = []

    def transport(**request):
        calls.append(request)
        return {
            "http_status": 200,
            "headers": {"content-type": "application/json"},
            "body": b'{"ok":true}',
            "tls_peer_chain_hash": "sha256:" + "d" * 64,
            "tls_protocol": "TLSv1.3",
        }

    broker = _broker(transport)
    route = build_source_add_runtime_catalog_v2([_source_row()])["routes"][0]
    broker.provision_job_credential(
        job_id="job-source-one",
        slot=route["credential_slot"],
        credential="source-add-secret",
        credential_value_hash_expected=route["credential_value_hash"],
    )
    request = {
        "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
        "logical_operation_id": "source-one-operation",
        "job_id": "job-source-one",
        "purpose": "research_lab.provider_evidence.v2",
        "provider_id": "source_one",
        "attempt_number": 0,
        "method": "POST",
        "url": "https://api.source-one.example/v1/search?q=leadpoet",
        "headers": {"content-type": "application/json"},
        "body_b64": base64.b64encode(b'{"query":"leadpoet"}').decode(
            "ascii"
        ),
        "timeout_ms": 30000,
        "retry_policy_hash": source_add_dynamic_retry_policy_hash(route),
        "dynamic_route": route,
    }
    result = broker.execute(request)
    assert result["terminal_status"] == "authenticated_response"
    assert calls[0]["headers"]["x-source-key"] == "source-add-secret"
    assert "source-add-secret" not in str(result)

    with pytest.raises(ProviderBrokerV2Error, match="method/path"):
        broker.execute(
            {
                **request,
                "logical_operation_id": "source-one-wrong-path",
                "url": "https://api.source-one.example/v1/unlisted",
            }
        )
    with pytest.raises(ProviderBrokerV2Error, match="retry policy"):
        broker.execute(
            {
                **request,
                "logical_operation_id": "source-one-wrong-retry",
                "retry_policy_hash": "sha256:" + "e" * 64,
            }
        )


def test_job_envelope_uses_dynamic_slot_and_exact_commitments():
    row = _source_row()
    route = build_source_add_runtime_catalog_v2([row])["routes"][0]
    envelope = build_source_add_job_envelope_v2(row, job_id="job-source-one")
    assert envelope is not None
    assert envelope["credential_slot"] == route["credential_slot"]
    assert envelope["credential_value_hash"] == route["credential_value_hash"]
    assert envelope["key_ref_hash"] == route["key_ref_hash"]
