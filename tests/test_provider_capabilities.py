from __future__ import annotations

import http.client
import json

from gateway.research_lab.provider_capabilities import (
    EffectiveProviderCapabilities,
    LiveTextModelCatalog,
    load_effective_provider_capabilities_sync,
    provider_request_allowed,
    summary_mentions_private_capability,
    validate_candidate_provider_diff,
    validate_capability_provider_doc,
)
from gateway.research_lab.provider_evidence_proxy import (
    ProviderRegistryEntry,
    ProviderRegistryState,
    serve_evidence_proxy,
)
from research_lab.canonical import sha256_json
from research_lab.code_editing import build_loop_direction_planner_messages
from research_lab.eval.provider_evidence_cache import canonical_request_fingerprint


def _provider_doc(
    provider_id: str = "synthetic_feed",
    *,
    base_url: str = "https://api.synthetic-feed.invalid",
    origin: str = "builtin",
    policy: dict | None = None,
) -> dict:
    return {
        "id": provider_id,
        "base_url": base_url,
        "auth_kind": "none",
        "auth_name": "",
        "credential_ref": [],
        "per_day_quota": 0,
        "cost_model": {"est_cost_microusd_per_call": 5000},
        "active": True,
        "origin": origin,
        "reward_eligible": origin == "source_add",
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


def test_private_snapshot_merges_ready_source_add_and_continuity_fallback():
    private = _provider_doc("private_feed")
    source_row = {
        "adapter_id": "adapter:synthetic",
        "miner_hotkey": "hk-synthetic",
        "provision_status": "provisioned_autoresearch_eligible",
        "credential_envelope": {},
        "provision_doc": {
            "provider_registry_entry": {
                "id": "community_feed",
                "base_url": "https://community-feed.invalid",
                "auth_kind": "none",
                "credential_ref": [],
            },
            "probe_endpoints": [
                {
                    "endpoint_id": "community_feed.search",
                    "provider_id": "community_feed",
                    "method": "POST",
                    "path": "/search",
                    "params": [],
                    "description": "Synthetic community search",
                }
            ],
        },
    }
    capabilities = load_effective_provider_capabilities_sync(
        [_provider_doc("legacy_feed", origin="legacy_fallback")],
        private_row_loader=lambda: _private_row(private),
        source_row_loader=lambda: [source_row],
    )

    assert capabilities.private_snapshot_loaded is True
    assert {item["id"] for item in capabilities.providers} == {
        "private_feed",
        "community_feed",
        "legacy_feed",
    }
    assert capabilities.source_add_provider_count == 1
    summary = capabilities.prompt_summary()
    assert summary["provider_count"] == 3
    diagnostic_text = json.dumps(capabilities.diagnostic(), sort_keys=True)
    assert "private_feed" not in diagnostic_text
    assert "community-feed.invalid" not in diagnostic_text


def test_source_add_cannot_replace_reserved_provider_id():
    private = _provider_doc("reserved_feed")
    source_row = {
        "adapter_id": "adapter:collision",
        "miner_hotkey": "hk-collision",
        "provision_status": "provisioned_autoresearch_eligible",
        "credential_envelope": {},
        "provision_doc": {
            "provider_registry_entry": {
                "id": "reserved_feed",
                "base_url": "https://collision.invalid",
                "auth_kind": "none",
                "credential_ref": [],
            },
            "probe_endpoints": [],
        },
    }
    capabilities = load_effective_provider_capabilities_sync(
        [],
        private_row_loader=lambda: _private_row(private),
        source_row_loader=lambda: [source_row],
    )
    assert [item["base_url"] for item in capabilities.providers] == [private["base_url"]]
    assert "source_add_provider_id_collision" in capabilities.warning_codes


def test_invalid_newest_snapshot_falls_back_to_prior_valid_snapshot():
    valid_row = _private_row(_provider_doc("prior_valid_feed"))
    invalid_row = dict(_private_row(_provider_doc("invalid_newest_feed")))
    invalid_row["registry_hash"] = "sha256:" + "0" * 64

    capabilities = load_effective_provider_capabilities_sync(
        [],
        private_row_loader=lambda: [invalid_row, valid_row],
        source_row_loader=lambda: [],
    )
    assert capabilities.private_snapshot_loaded is True
    assert [item["id"] for item in capabilities.providers] == ["prior_valid_feed"]
    assert "private_snapshot_invalid_skipped" in capabilities.warning_codes


def test_unresolved_authenticated_source_add_is_omitted(monkeypatch):
    monkeypatch.delenv("SYNTHETIC_FEED_KEY", raising=False)
    source_row = {
        "adapter_id": "adapter:unready",
        "miner_hotkey": "hk-unready",
        "provision_status": "provisioned_autoresearch_eligible",
        "credential_envelope": {},
        "provision_doc": {
            "provider_registry_entry": {
                "id": "unready_feed",
                "base_url": "https://unready.invalid",
                "auth_kind": "header",
                "auth_name": "x-synthetic-key",
                "credential_ref": ["SYNTHETIC_FEED_KEY"],
            },
            "probe_endpoints": [],
        },
    }
    capabilities = load_effective_provider_capabilities_sync(
        [],
        private_row_loader=lambda: None,
        source_row_loader=lambda: [source_row],
    )
    assert capabilities.providers == ()
    assert "source_add_provider_not_runtime_ready" in capabilities.warning_codes


def test_pending_and_disabled_source_add_rows_are_not_advertised():
    def source_row(provider_id: str, status: str) -> dict:
        return {
            "adapter_id": f"adapter:{provider_id}",
            "miner_hotkey": "hk-status",
            "provision_status": status,
            "credential_envelope": {},
            "provision_doc": {
                "provider_registry_entry": {
                    "id": provider_id,
                    "base_url": f"https://{provider_id}.invalid",
                    "auth_kind": "none",
                    "credential_ref": [],
                },
                "probe_endpoints": [],
            },
        }

    capabilities = load_effective_provider_capabilities_sync(
        [],
        private_row_loader=lambda: None,
        source_row_loader=lambda: [
            source_row("pending_feed", "approved_pending_provision"),
            source_row("disabled_feed", "disabled"),
        ],
    )
    assert capabilities.providers == ()
    assert capabilities.source_add_provider_count == 0


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


def test_private_capability_summary_reaches_hidden_planner_context():
    capabilities = _capabilities(_provider_doc())
    summary = capabilities.prompt_summary()
    messages = build_loop_direction_planner_messages(
        ticket={},
        artifact_manifest={},
        component_registry={},
        benchmark_public_summary={},
        runtime_source_index={"editable_files": ["sourcing_model/provider.py"]},
        budget_context={},
        provider_capability_summary=summary,
    )
    prompt = messages[-1]["content"]
    assert "approved_provider_capabilities" in prompt
    assert "synthetic discovery" in prompt
    assert "new_credentials_forbidden" in prompt
    assert summary_mentions_private_capability(
        "Switch the synthetic discovery provider to another route",
        capabilities,
    ) is True
    assert summary_mentions_private_capability(
        "Improve bounded source routing while preserving output checks",
        capabilities,
    ) is False


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


def test_candidate_static_guard_rejects_unknown_hosts_blocked_routes_and_new_clients():
    policy = {
        "routes": [{"method": "POST", "path": "/v1/generate"}],
        "blocked_routes": [{"method": "GET", "path_prefix": "/admin"}],
        "allow_unlisted_paths": False,
        "model_policy": {"kind": "none"},
    }
    capabilities = _capabilities(_provider_doc(policy=policy))
    diff = """diff --git a/sourcing_model/provider.py b/sourcing_model/provider.py
--- a/sourcing_model/provider.py
+++ b/sourcing_model/provider.py
@@ -1 +1,4 @@
+import httpx
+GOOD = 'https://api.synthetic-feed.invalid/v1/generate'
+BAD = 'https://unknown-provider.invalid/search'
+ADMIN = '/admin/keys'
+KEY = os.getenv('SYNTHETIC_KEY')
"""
    errors = validate_candidate_provider_diff(diff, capabilities)
    assert "candidate_adds_new_network_client_import" in errors
    assert any(item.startswith("candidate_adds_unknown_provider_host:") for item in errors)
    assert "candidate_adds_blocked_provider_route" in errors
    assert "candidate_adds_new_credential_or_env_reference" in errors


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
