from __future__ import annotations

import http.client
import json

import pytest

from gateway.research_lab.provider_capabilities import (
    EffectiveProviderCapabilities,
    LiveTextModelCatalog,
    _normalize_source_add_v8_registration,
    load_effective_provider_capabilities_sync,
    normalize_source_add_planner_contract,
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
    if origin == "source_add":
        provider["source_add_provisioning_provenance_sha256"] = "a" * 64
    return provider


def _builtwith_execution_plan() -> dict:
    return {
        "schema_version": "source-add-signal-bound-json-intent-plan:v1",
        "provider_id": "builtwith_trends",
        "tool_id": "intent.source_add.builtwith_trends",
        "request": {
            "method": "GET",
            "path": "/api.json",
            "query": {
                "TECH": {
                    "source": "signal_temporal_parameter",
                    "name": "technology",
                    "max_length": 120,
                }
            },
        },
        "response_projection": {
            "kind": "technology_context",
            "category": "TECHSTACK",
            "object_field": "Tech",
            "identity_field": "name",
            "expected_identity_query_key": "TECH",
            "canonical_url_field": "trends_link",
            "canonical_source_domain": "trends.builtwith.com",
            "canonical_url_path_prefix": "/shop/",
            "excerpt_fields": ["description"],
        },
    }


def _builtwith_static_execution_plan() -> dict:
    plan = _builtwith_execution_plan()
    plan["schema_version"] = "source-add-static-json-intent-plan:v1"
    plan["request"]["query"] = {"TECH": "Shopify"}
    projection = plan["response_projection"]
    projection.pop("expected_identity_query_key")
    projection["expected_identity"] = "Shopify"
    return plan


def _builtwith_probe_endpoint() -> dict:
    return {
        "endpoint_id": "builtwith_trends.technology",
        "provider_id": "builtwith_trends",
        "method": "GET",
        "path": "/api.json",
        "params": [
            {
                "name": "TECH",
                "type": "string",
                "required": True,
                "location": "query",
                "max_length": 120,
            }
        ],
    }


def _builtwith_tested_probe() -> dict:
    return {
        "method": "GET",
        "path": "/api.json",
        "query": {"TECH": "Shopify"},
        "body_json": None,
    }


def test_source_add_v8_normalizer_matches_upstream_golden_manifest():
    normalized = _normalize_source_add_v8_registration(
        {
            "provider_id": "approved_accounts",
            "stage": "candidate_acquisition",
            "priority": 80,
            "capabilities": ("candidate.provider_discovery",),
            "cost_class": "metered",
            "unit_cost": 0.005,
            "max_calls": 2,
            "max_results": 75,
            "timeout_seconds": 45,
            "evidence_types": ("provider_database",),
        }
    )
    assert normalized["manifest_sha256"] == (
        "e914fee915b37739edc34f964406e3935841e00ff6d2baba45b6ad4b64f32897"
    )
    assert normalized["revision"] == "source-add-e914fee915b3"


def test_source_add_execution_plan_is_bound_to_tested_provisioned_route():
    contract = {
        "stage": "intent_evidence",
        "execution_mode": "invoke",
        "priority": 35,
        "capabilities": ["intent.provider_evidence"],
        "idempotency": "idempotent",
        "cost_class": "free",
        "unit_cost": 0.0,
        "max_calls": 1,
        "max_results": 1,
        "timeout_seconds": 30.0,
        "intent_categories": ["TECHSTACK"],
        "evidence_types": ["reputable_publisher"],
        "category_contracts": [],
        "binding_requirements": [],
        "best_for": ["intent.techstack"],
        "avoid_when": [],
        "execution_plan_identity": _builtwith_execution_plan(),
    }
    normalized = normalize_source_add_planner_contract(
        "builtwith_trends",
        contract,
        probe_endpoints=[_builtwith_probe_endpoint()],
        tested_probes=[_builtwith_tested_probe()],
    )

    assert normalized["execution_plan_identity"] == _builtwith_execution_plan()

    source_row = {
        "adapter_id": "adapter:builtwith-trends",
        "miner_hotkey": "hk-builtwith-trends",
        "provision_status": "provisioned_autoresearch_eligible",
        "credential_envelope": {},
        "provision_doc": {
            "provider_registry_entry": {
                "id": "builtwith_trends",
                "base_url": "https://api.builtwith.com/trends/v6",
                "auth_kind": "none",
                "credential_ref": [],
                "planner_summary": {
                    **normalized,
                    "provider_alias": "BuiltWith trends",
                },
            },
            "probe_endpoints": [_builtwith_probe_endpoint()],
        },
    }
    capabilities = load_effective_provider_capabilities_sync(
        [],
        private_row_loader=lambda: None,
        source_row_loader=lambda: [source_row],
    )
    provider = capabilities.providers[0]
    assert provider["planner_summary"]["execution_plan_identity"] == (
        _builtwith_execution_plan()
    )


def test_source_add_execution_plan_canonicalizes_legacy_host_field():
    legacy_plan = json.loads(json.dumps(_builtwith_execution_plan()))
    projection = legacy_plan["response_projection"]
    projection["canonical_url_host"] = projection.pop(
        "canonical_source_domain"
    )

    normalized = normalize_source_add_planner_contract(
        "builtwith_trends",
        {
            "stage": "intent_evidence",
            "cost_class": "free",
            "unit_cost": 0.0,
            "max_calls": 1,
            "max_results": 1,
            "intent_categories": ["TECHSTACK"],
            "execution_plan_identity": legacy_plan,
        },
        probe_endpoints=[_builtwith_probe_endpoint()],
        tested_probes=[_builtwith_tested_probe()],
    )

    canonical_projection = normalized["execution_plan_identity"][
        "response_projection"
    ]
    assert canonical_projection["canonical_source_domain"] == (
        "trends.builtwith.com"
    )
    assert "canonical_url_host" not in canonical_projection


def test_source_add_execution_plan_rejects_ambiguous_domain_fields():
    plan = _builtwith_execution_plan()
    plan["response_projection"]["canonical_url_host"] = (
        "trends.builtwith.com"
    )

    with pytest.raises(ValueError, match="fields differ from the contract"):
        normalize_source_add_planner_contract(
            "builtwith_trends",
            {
                "stage": "intent_evidence",
                "cost_class": "free",
                "unit_cost": 0.0,
                "max_calls": 1,
                "max_results": 1,
                "intent_categories": ["TECHSTACK"],
                "execution_plan_identity": plan,
            },
            probe_endpoints=[_builtwith_probe_endpoint()],
            tested_probes=[_builtwith_tested_probe()],
        )


@pytest.mark.parametrize(
    "probe_endpoints,tested_probes,error",
    (
        (
            [],
            [_builtwith_tested_probe()],
            "must match one provisioned endpoint",
        ),
        (
            [{**_builtwith_probe_endpoint(), "method": "POST"}],
            [_builtwith_tested_probe()],
            "must match one provisioned endpoint",
        ),
        (
            [_builtwith_probe_endpoint()],
            [],
            "must match one successful test probe",
        ),
        (
            [_builtwith_probe_endpoint()],
            [{**_builtwith_tested_probe(), "query": {"OTHER": "React"}}],
            "must match one successful test probe",
        ),
        (
            [
                {
                    **_builtwith_probe_endpoint(),
                    "params": [
                        {
                            **_builtwith_probe_endpoint()["params"][0],
                            "max_length": 119,
                        }
                    ],
                }
            ],
            [_builtwith_tested_probe()],
            "query differs from the provisioned endpoint",
        ),
        (
            [_builtwith_probe_endpoint()],
            [
                {
                    **_builtwith_tested_probe(),
                    "query": {"TECH": "x" * 121},
                }
            ],
            "must match one successful test probe",
        ),
        (
            [
                {
                    **_builtwith_probe_endpoint(),
                    "params": [
                        *_builtwith_probe_endpoint()["params"],
                        {
                            "name": "required_extra",
                            "required": True,
                            "location": "query",
                        },
                    ],
                }
            ],
            [_builtwith_tested_probe()],
            "query differs from the provisioned endpoint",
        ),
    ),
)
def test_source_add_execution_plan_rejects_untested_route_drift(
    probe_endpoints,
    tested_probes,
    error,
):
    with pytest.raises(ValueError, match=error):
        normalize_source_add_planner_contract(
            "builtwith_trends",
            {
                "stage": "intent_evidence",
                "cost_class": "free",
                "unit_cost": 0.0,
                "max_calls": 1,
                "max_results": 1,
                "intent_categories": ["TECHSTACK"],
                "execution_plan_identity": _builtwith_execution_plan(),
            },
            probe_endpoints=probe_endpoints,
            tested_probes=tested_probes,
        )


def test_source_add_signal_bound_plan_accepts_another_tested_value():
    normalized = normalize_source_add_planner_contract(
        "builtwith_trends",
        {
            "stage": "intent_evidence",
            "cost_class": "free",
            "unit_cost": 0.0,
            "max_calls": 1,
            "max_results": 1,
            "intent_categories": ["TECHSTACK"],
            "execution_plan_identity": _builtwith_execution_plan(),
        },
        probe_endpoints=[_builtwith_probe_endpoint()],
        tested_probes=[
            {**_builtwith_tested_probe(), "query": {"TECH": "WooCommerce"}}
        ],
    )

    assert normalized["execution_plan_identity"] == _builtwith_execution_plan()


def test_source_add_static_plan_remains_supported():
    normalized = normalize_source_add_planner_contract(
        "builtwith_trends",
        {
            "stage": "intent_evidence",
            "cost_class": "free",
            "unit_cost": 0.0,
            "max_calls": 1,
            "max_results": 1,
            "intent_categories": ["TECHSTACK"],
            "execution_plan_identity": _builtwith_static_execution_plan(),
        },
        probe_endpoints=[_builtwith_probe_endpoint()],
        tested_probes=[_builtwith_tested_probe()],
    )

    assert normalized["execution_plan_identity"] == (
        _builtwith_static_execution_plan()
    )


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
    community = next(
        item
        for item in capabilities.providers
        if item["id"] == "community_feed"
    )
    assert len(
        community["source_add_provisioning_provenance_sha256"]
    ) == 64
    assert "source_add_manifest_sha256" not in community
    diagnostic = capabilities.diagnostic()
    assert diagnostic["provider_count"] == 3
    diagnostic_text = json.dumps(diagnostic, sort_keys=True)
    assert "private_feed" not in diagnostic_text
    assert "community-feed.invalid" not in diagnostic_text


def test_source_add_projection_preserves_v8_routing_metadata_losslessly():
    planner_summary = {
        "provider_alias": "Reviewed community signals",
        "stage": "intent_evidence",
        "execution_mode": "observe",
        "priority": 0,
        "capabilities": [
            "intent.techstack_observation",
            "evidence.technology_observation",
        ],
        "idempotency": "resume_safe",
        "cost_class": "paid",
        "unit_cost": 0.1234567,
        "max_calls": 10_000,
        "max_results": 100_000,
        "timeout_seconds": 3_600.0,
        "intent_categories": ["TECHSTACK"],
        "evidence_types": ["technology_observation"],
        "category_contracts": [
            {
                "category": "TECHSTACK",
                "capabilities": [
                    "intent.techstack_observation",
                    "evidence.technology_observation",
                ],
                "evidence_types": ["technology_observation"],
                "requirements": ["observation_only"],
            }
        ],
        "binding_requirements": ["receipt_only"],
        "best_for_features": ["intent.techstack"],
        "avoid_when_features": ["intent.publication"],
        "best_for": "Reviewed observation-only signal.",
        "avoid_when": "Never treat it as publication evidence.",
    }
    source_row = {
        "adapter_id": "adapter:v8-shaped",
        "miner_hotkey": "hk-v8-shaped",
        "provision_status": "provisioned_autoresearch_eligible",
        "credential_envelope": {},
        "provision_doc": {
            "provider_registry_entry": {
                "id": "community_signals",
                "base_url": "https://community-signals.invalid",
                "auth_kind": "none",
                "credential_ref": [],
                "planner_summary": planner_summary,
            },
            "probe_endpoints": [],
        },
    }
    capabilities = load_effective_provider_capabilities_sync(
        [],
        private_row_loader=lambda: None,
        source_row_loader=lambda: [source_row],
    )
    assert capabilities.source_add_provider_count == 1
    provider = capabilities.providers[0]
    for field_name, expected in planner_summary.items():
        assert provider["planner_summary"][field_name] == expected

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
