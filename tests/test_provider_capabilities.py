from __future__ import annotations

import http.client
import json

import pytest

from gateway.research_lab.provider_capabilities import (
    EffectiveProviderCapabilities,
    LiveTextModelCatalog,
    load_effective_provider_capabilities_sync,
    provider_request_allowed,
    validate_capability_provider_doc,
)
from gateway.research_lab.provider_evidence_proxy import (
    ProviderRegistryEntry,
    ProviderRegistryState,
    serve_evidence_proxy,
)
from research_lab.canonical import sha256_json
from research_lab.eval.provider_evidence_cache import canonical_request_fingerprint



def _provider_doc(
    provider_id: str = "synthetic_feed",
    *,
    base_url: str = "https://api.synthetic-feed.invalid",
    origin: str = "builtin",
    policy: dict | None = None,
) -> dict:
    provider = {
        "id": provider_id,
        "base_url": base_url,
        "auth_kind": "none",
        "auth_name": "",
        "credential_ref": [],
        "per_day_quota": 0,
        "cost_model": {"est_cost_microusd_per_call": 5000},
        "active": True,
        "origin": origin,
        "reward_eligible": False,
        "capability_policy": policy
        or {
            "routes": [{"method": "GET", "path_prefix": "/"}],
            "blocked_routes": [],
            "allow_unlisted_paths": False,
            "model_policy": {"kind": "none"},
        },
        "planner_summary": {
            "provider_alias": "synthetic discovery",
            "endpoint_families": [{"family": "search", "description": "Synthetic search"}],
            "model_policy": "",
            "probe_metadata": [],
        },
        "probe_endpoints": [],
    }
    return provider
























def _private_row(*providers: dict) -> dict:
    doc = {"schema_version": "1.0", "providers": list(providers)}
    return {
        "registry_hash": sha256_json(doc),
        "provider_count": len(providers),
        "registry_doc": doc,
    }


def _capabilities(*providers: dict, private_loaded: bool = True) -> EffectiveProviderCapabilities:
    provider_tuple = tuple(dict(item, credential_ready=True) for item in providers)
    return EffectiveProviderCapabilities(
        providers=provider_tuple,
        capability_hash=sha256_json({"providers": provider_tuple}),
        private_registry_hash="sha256:" + "1" * 64 if private_loaded else "",
        private_snapshot_loaded=private_loaded,
    )













def test_registry_rejects_secret_and_malformed_policy_docs():
    secret = _provider_doc()
    secret["credential_value"] = "sk" + "-or-synthetic"
    assert "provider_doc_contains_forbidden_material" in validate_capability_provider_doc(secret)

    malformed = _provider_doc()
    malformed["capability_policy"] = {
        "routes": [{"method": "DELETE", "path": "/search"}],
        "model_policy": {"kind": "unknown"},
    }
    errors = validate_capability_provider_doc(malformed)
    assert "provider_routes_invalid" in errors
    assert "provider_model_policy_kind_invalid" in errors



def test_route_policy_allows_unlisted_safe_paths_and_blocks_admin_paths():
    policy = {
        "routes": [],
        "blocked_routes": [{"method": "GET", "path_prefix": "/admin"}],
        "allow_unlisted_paths": True,
        "unlisted_methods": ["GET", "POST"],
        "model_policy": {"kind": "none"},
    }
    provider = _provider_doc(policy=policy)
    assert provider_request_allowed(provider, "GET", "/new-surface?q=x")[:2] == (
        True,
        "allowed_unlisted_route",
    )
    assert provider_request_allowed(provider, "GET", "/admin/keys")[:2] == (
        False,
        "blocked_route",
    )
    assert provider_request_allowed(provider, "GET", "/safe/../admin")[:2] == (
        False,
        "unsafe_route",
    )
    assert provider_request_allowed(provider, "GET", "/new-surface?q=x%0d%0aheader")[:2] == (
        False,
        "unsafe_route",
    )


def test_live_text_model_catalog_caches_and_keeps_last_known_good():
    calls = []
    should_fail = {"value": False}

    def fetch_json(url, _headers):
        calls.append(url)
        if should_fail["value"]:
            raise RuntimeError("temporary catalog failure")
        return {
            "data": [
                {"id": "vendor-a/text-one", "architecture": {"output_modalities": ["text"]}},
                {"id": "vendor-b/text-two", "architecture": {"output_modalities": ["text"]}},
                {"id": "openai/test-text", "architecture": {"output_modalities": ["text"]}},
                {"id": "anthropic/test-text", "architecture": {"output_modalities": ["text"]}},
                {"id": "perplexity/test-text", "architecture": {"output_modalities": ["text"]}},
                {"id": "deepseek/test-text", "architecture": {"output_modalities": ["text"]}},
                {"id": "google/test-text", "architecture": {"output_modalities": ["text"]}},
                {"id": "moonshotai/test-text", "architecture": {"output_modalities": ["text"]}},
                {"id": "vendor-c/image-only", "architecture": {"output_modalities": ["image"]}},
            ]
        }

    provider = _provider_doc(
        "model_hub",
        base_url="https://models.invalid",
        policy={
            "routes": [{"method": "POST", "path": "/v1/chat"}],
            "blocked_routes": [{"method": "GET", "path_prefix": "/admin"}],
            "allow_unlisted_paths": False,
            "model_policy": {
                "kind": "live_text_catalog",
                "catalog_path": "/v1/models?output_modalities=text",
                "lookup_path_template": "/v1/model/{model_id}",
            },
        },
    )
    catalog = LiveTextModelCatalog(ttl_seconds=900, fetch_json=fetch_json)
    assert catalog.validate_model(provider, "vendor-a/text-one") == (True, "live")
    assert catalog.validate_model(provider, "vendor-b/text-two") == (True, "live")
    for family in ("openai", "anthropic", "perplexity", "deepseek", "google", "moonshotai"):
        assert catalog.validate_model(provider, f"{family}/test-text")[0] is True
    assert catalog.validate_model(provider, "vendor-c/image-only")[0] is False
    assert len(calls) == 2  # one catalog fetch plus one single-model lookup

    should_fail["value"] = True
    models, status = catalog.refresh(provider, force=True)
    assert status == "last_known_good"
    assert "vendor-a/text-one" in models


def test_registry_state_retains_last_known_good_on_refresh_failure():
    first = _capabilities(_provider_doc("first_feed"))
    second = _capabilities(_provider_doc("second_feed"))
    first_entry = ProviderRegistryEntry.from_mapping(first.providers[0])
    second_entry = ProviderRegistryEntry.from_mapping(second.providers[0])
    outcomes = [RuntimeError("db unavailable"), ([second_entry], second)]

    def loader():
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    state = ProviderRegistryState(
        entries=[first_entry],
        capabilities=first,
        loader=loader,
    )
    assert state.refresh_once() is False
    assert state.resolve("first_feed") is not None
    assert state.refresh_once() is True
    assert state.resolve("first_feed") is None
    assert state.resolve("second_feed") is not None


def _post(port: int, path: str, body: dict) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        payload = json.dumps(body).encode("utf-8")
        connection.request(
            "POST",
            path,
            body=payload,
            headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
        )
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_proxy_enforces_text_catalog_before_replaying_cached_response():
    policy = {
        "routes": [{"method": "POST", "path": "/v1/chat"}],
        "blocked_routes": [{"method": "POST", "path_prefix": "/admin"}],
        "allow_unlisted_paths": False,
        "model_policy": {
            "kind": "live_text_catalog",
            "catalog_path": "/v1/models?output_modalities=text",
        },
    }
    entry = ProviderRegistryEntry.from_mapping(
        _provider_doc("model_hub", base_url="http://127.0.0.1:9", policy=policy)
    )
    catalog = LiveTextModelCatalog(
        fetch_json=lambda _url, _headers: {
            "data": [{"id": "vendor-a/text-one", "architecture": {"output_modalities": ["text"]}}]
        }
    )
    server, store, _thread = serve_evidence_proxy(
        host="127.0.0.1",
        port=0,
        registry=[entry],
        enforcement_mode="enforce",
        model_catalog=catalog,
    )
    try:
        body = {"model": "vendor-a/text-one", "messages": []}
        encoded = json.dumps(body).encode("utf-8")
        fingerprint = canonical_request_fingerprint(
            "POST",
            "http://127.0.0.1:9/v1/chat",
            encoded,
        )
        store.record(fingerprint, 200, b'{"choices":[]}')
        status, _response = _post(server.server_address[1], "/model_hub/v1/chat", body)
        assert status == 200

        status, response = _post(
            server.server_address[1],
            "/model_hub/v1/chat",
            {"model": "vendor-z/missing", "messages": []},
        )
        assert status == 403
        assert b"text model not allowed" in response
    finally:
        server.shutdown()
        server.server_close()


def test_migration_is_service_only_and_contains_no_provider_inventory():
    sql = open("scripts/81-research-lab-private-provider-capabilities.sql", encoding="utf-8").read()
    assert "research_lab_provider_registry_current" in sql
    assert "DROP CONSTRAINT IF EXISTS research_lab_provider_registry_registry_hash_key" in sql
    assert "ENABLE ROW LEVEL SECURITY" in sql
    assert "FROM PUBLIC, anon, authenticated" in sql
    assert "api.synthetic-feed.invalid" not in sql
