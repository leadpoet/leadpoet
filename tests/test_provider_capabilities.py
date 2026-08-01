from __future__ import annotations

import difflib
import http.client
import json

import pytest

from gateway.research_lab.provider_capabilities import (
    EffectiveProviderCapabilities,
    LiveTextModelCatalog,
    _normalize_source_add_v8_registration,
    approved_source_router_suggestions,
    load_effective_provider_capabilities_sync,
    provider_request_allowed,
    summary_mentions_private_capability,
    validate_candidate_provider_diff,
    validate_capability_provider_doc,
    validate_source_add_registration_diff,
)
from gateway.research_lab.provider_evidence_proxy import (
    ProviderRegistryEntry,
    ProviderRegistryState,
    serve_evidence_proxy,
)
from research_lab.canonical import sha256_json
from research_lab.code_editing import build_loop_direction_planner_messages
from research_lab.eval.provider_evidence_cache import canonical_request_fingerprint


_EMPTY_ROUTER_RUNTIME = """class SourceAddRoutingRegistration:
    pass

SOURCE_ADD_ROUTING_REGISTRATIONS = (
)
"""
_EMPTY_V8_ROUTER_RUNTIME = """SOURCE_ADD_BINDING_MANIFEST_SCHEMA_VERSION = (
    "leadpoet.intent-source-binding-manifest:v1"
)

class SourceAddCategoryContract:
    pass

class SourceAddRoutingRegistration:
    pass

SOURCE_ADD_ROUTING_REGISTRATIONS = ()
"""


def _router_registration_source(
    *,
    provider_id: str = "community_accounts",
    stage: str = "candidate_acquisition",
    manifest: str = "a" * 64,
    priority: int | None = None,
    max_calls: int = 1,
    unknown_field: str = "",
    outside_registry: str = "",
) -> str:
    candidate_stage = stage == "candidate_acquisition"
    priority = priority if priority is not None else (80 if candidate_stage else 35)
    capabilities = (
        ("candidate.provider_discovery",)
        if candidate_stage
        else ("intent.provider_evidence",)
    )
    evidence_types = (
        ("provider_database",) if candidate_stage else ("external",)
    )
    best_for = (
        ("icp.structured_eligible",)
        if candidate_stage
        else ("intent.general",)
    )
    best_for_description = (
        "Approved SOURCE_ADD company-discovery provider for structured ICP "
        "acquisition."
        if candidate_stage
        else "Approved SOURCE_ADD provider for company-scoped intent-evidence "
        "discovery."
    )
    return f"""class SourceAddRoutingRegistration:
    pass

SOURCE_ADD_ROUTING_REGISTRATIONS = (
    SourceAddRoutingRegistration(
        provider_id={provider_id!r},
        stage={stage!r},
        revision={f"source-add-{manifest[:12]}"!r},
        manifest_sha256={manifest!r},
        priority={priority},
        capabilities={capabilities!r},
        idempotency="idempotent",
        cost_class="metered",
        unit_cost=0.005,
        max_calls={max_calls},
        max_results={100 if candidate_stage else 1},
        timeout_seconds={60.0 if candidate_stage else 30.0},
        intent_categories=(),
        evidence_types={evidence_types!r},
        best_for={best_for!r},
        avoid_when=(),
        best_for_description={best_for_description!r},
        avoid_when_description="Avoid when the consumer binding is unavailable, unhealthy, outside its approved categories, or over budget.",
{unknown_field}    ),
)
{outside_registry}"""


def _router_runtime_diff(before: str, after: str) -> str:
    body = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="a/sourcing_model/routing/runtime.py",
            tofile="b/sourcing_model/routing/runtime.py",
        )
    )
    return (
        "diff --git a/sourcing_model/routing/runtime.py "
        "b/sourcing_model/routing/runtime.py\n"
        f"{body}"
    )


def _v8_router_registration_source(request: dict) -> str:
    contracts = []
    for contract in request["category_contracts"]:
        contracts.append(
            """        SourceAddCategoryContract(
            category={category!r},
            capabilities={capabilities!r},
            evidence_types={evidence_types!r},
            requirements={requirements!r},
        ),""".format(
                category=contract["category"],
                capabilities=tuple(contract["capabilities"]),
                evidence_types=tuple(contract["evidence_types"]),
                requirements=tuple(contract["requirements"]),
            )
        )
    category_contracts = "\n".join(contracts)
    if category_contracts:
        category_contracts = "(\n" + category_contracts + "\n        )"
    else:
        category_contracts = "()"
    return _EMPTY_V8_ROUTER_RUNTIME.replace(
        "SOURCE_ADD_ROUTING_REGISTRATIONS = ()",
        f"""SOURCE_ADD_ROUTING_REGISTRATIONS = (
    SourceAddRoutingRegistration(
        provider_id={request['provider_id']!r},
        stage={request['stage']!r},
        revision={request['revision']!r},
        manifest_sha256={request['manifest_sha256']!r},
        execution_mode={request['execution_mode']!r},
        priority={request['priority']!r},
        capabilities={tuple(request['capabilities'])!r},
        idempotency={request['idempotency']!r},
        cost_class={request['cost_class']!r},
        unit_cost={request['unit_cost']!r},
        max_calls={request['max_calls']!r},
        max_results={request['max_results']!r},
        timeout_seconds={request['timeout_seconds']!r},
        intent_categories={tuple(request['intent_categories'])!r},
        evidence_types={tuple(request['evidence_types'])!r},
        category_contracts={category_contracts},
        binding_requirements={tuple(request['binding_requirements'])!r},
        best_for={tuple(request['best_for'])!r},
        avoid_when={tuple(request['avoid_when'])!r},
        best_for_description={request['best_for_description']!r},
        avoid_when_description={request['avoid_when_description']!r},
    ),
)""",
    )
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
    summary = capabilities.prompt_summary()
    assert summary["provider_count"] == 3
    diagnostic_text = json.dumps(capabilities.diagnostic(), sort_keys=True)
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

    context = approved_source_router_suggestions(
        "Use community signals for intent discovery.",
        capabilities.providers,
    )
    request = context["requests"][0]
    assert request["priority"] == 0
    assert request["execution_mode"] == "observe"
    assert request["idempotency"] == "resume_safe"
    assert request["cost_class"] == "paid"
    assert request["unit_cost"] == 0.123457
    assert request["max_calls"] == 10_000
    assert request["max_results"] == 100_000
    assert request["timeout_seconds"] == 3_600.0
    assert request["binding_manifest"]["binding_requirements"] == [
        "receipt_only"
    ]
    assert request["manifest_sha256"] != request[
        "provisioning_provenance_sha256"
    ]
    valid_source = _v8_router_registration_source(request)
    assert validate_source_add_registration_diff(
        "",
        context,
        existing_runtime_source=valid_source,
    ) == []
    for direct_constructor in (
        "SourceAddRoutingRegistration(",
        "SourceAddCategoryContract(",
    ):
        qualified_source = valid_source.replace(
            direct_constructor,
            f"unreviewed.{direct_constructor}",
            1,
        )
        assert validate_source_add_registration_diff(
            _router_runtime_diff(valid_source, qualified_source),
            context,
            existing_runtime_source=valid_source,
        ) == ["source_add_registration_patched_source_invalid"]


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


def test_approved_source_mention_creates_company_router_registration_request():
    provider = _provider_doc(
        "community_accounts",
        origin="source_add",
    )
    capabilities = _capabilities(provider, private_loaded=False)
    summary = capabilities.prompt_summary(
        miner_focus=(
            "Incorporate community accounts for company discovery."
        )
    )
    incorporation = summary["routerverse_source_incorporation"]
    request = incorporation["requests"][0]

    assert request["provider_id"] == "community_accounts"
    assert request["stage"] == "candidate_acquisition"
    assert request["tool_id"] == (
        "candidate.source_add.community_accounts"
    )
    assert request["runtime_binding_id"] == "community_accounts"
    assert request["provisioning_provenance_sha256"] == "a" * 64
    assert request["legacy_v7_manifest_sha256"] == "a" * 64
    assert request["manifest_sha256"] != "a" * 64
    assert request["binding_manifest"]["provider_id"] == (
        "community_accounts"
    )
    assert request["registration_symbol"].endswith(
        "::SOURCE_ADD_ROUTING_REGISTRATIONS"
    )
    assert request["schema_version"] == (
        "leadpoet.routerverse_source_incorporation.v3"
    )
    assert request["best_for"] == ["icp.structured_eligible"]
    assert request["intent_categories"] == []
    assert request["avoid_when"] == []
    assert "company-discovery" in request["best_for_description"]
    assert "consumer binding" in request["avoid_when_description"]
    assert incorporation["clarifications"] == []


def test_approved_source_mention_creates_intent_router_registration_request():
    provider = _provider_doc(
        "community_signals",
        origin="source_add",
    )
    context = approved_source_router_suggestions(
        "Use community signals for intent discovery.",
        _capabilities(provider, private_loaded=False).providers,
    )

    assert [item["stage"] for item in context["requests"]] == [
        "intent_evidence"
    ]
    assert context["requests"][0]["tool_id"] == (
        "intent.source_add.community_signals"
    )
    assert context["requests"][0]["runtime_binding_id"] == "community_signals"
    assert context["requests"][0]["best_for"] == ["intent.general"]
    assert context["requests"][0]["intent_categories"] == []


def test_approved_source_guidance_rejects_invalid_v8_planner_metadata():
    provider = _provider_doc(
        "community_signals",
        origin="source_add",
    )
    provider["planner_summary"].update(
        {
            "best_for": "Fresh funding and hiring evidence.",
            "avoid_when": "The request requires first-party proof.",
            "best_for_features": [
                "intent.funding",
                "intent.hiring",
                "NOT VALID",
            ],
            "avoid_when_features": [
                "evidence.first_party_required",
                "also invalid",
            ],
        }
    )

    context = approved_source_router_suggestions(
        "Use community signals for intent discovery.",
        _capabilities(provider, private_loaded=False).providers,
    )

    assert context["requests"] == []
    assert context["clarifications"] == [
        {
            "provider_id": "community_signals",
            "provider_alias": "synthetic discovery",
            "reason_code": "approved_source_routing_contract_invalid",
        }
    ]


def test_category_scoped_intent_registration_matches_approved_guidance():
    provider = _provider_doc(
        "community_signals",
        origin="source_add",
    )
    provider["planner_summary"].update(
        {
            "best_for_features": ["intent.funding"],
            "best_for": "Fresh funding evidence.",
        }
    )
    context = approved_source_router_suggestions(
        "Use community signals for intent discovery.",
        _capabilities(provider, private_loaded=False).providers,
    )
    expected = _router_registration_source(
        provider_id="community_signals",
        stage="intent_evidence",
    ).replace(
        "intent_categories=()",
        "intent_categories=('FUNDING',)",
    ).replace(
        "best_for=('intent.general',)",
        "best_for=('intent.funding',)",
    ).replace(
        "Approved SOURCE_ADD provider for company-scoped intent-evidence discovery.",
        "Fresh funding evidence.",
    )

    assert validate_source_add_registration_diff(
        _router_runtime_diff(_EMPTY_ROUTER_RUNTIME, expected),
        context,
        existing_runtime_source=_EMPTY_ROUTER_RUNTIME,
    ) == []


def test_source_mention_without_discovery_stage_fails_closed_for_clarification():
    provider = _provider_doc(
        "community_signals",
        origin="source_add",
    )
    context = approved_source_router_suggestions(
        "Please incorporate community signals into Routerverse.",
        _capabilities(provider, private_loaded=False).providers,
    )

    assert context["requests"] == []
    assert context["clarifications"] == [
        {
            "provider_id": "community_signals",
            "provider_alias": "synthetic discovery",
            "reason_code": "discovery_stage_not_explicit",
        }
    ]


def test_unapproved_or_unmentioned_source_cannot_create_router_request():
    source_add = _provider_doc(
        "community_signals",
        origin="source_add",
    )
    builtin = _provider_doc("built_in_search")

    assert approved_source_router_suggestions(
        "Use built in search for company discovery.",
        _capabilities(source_add, builtin).providers,
    )["requests"] == []
    assert approved_source_router_suggestions(
        "Improve company discovery without changing providers.",
        _capabilities(source_add).providers,
    )["requests"] == []


def test_colliding_approved_source_aliases_require_exact_provider_id():
    first = _provider_doc("first_source", origin="source_add")
    second = _provider_doc("second_source", origin="source_add")
    capabilities = _capabilities(first, second, private_loaded=False)

    ambiguous = approved_source_router_suggestions(
        "Use synthetic discovery for company discovery.",
        capabilities.providers,
    )
    assert ambiguous["schema_version"] == (
        "leadpoet.routerverse_source_suggestions.v3"
    )
    assert ambiguous["requests"] == []
    assert ambiguous["clarifications"][0]["reason_code"] == (
        "approved_source_mention_ambiguous"
    )

    exact = approved_source_router_suggestions(
        "Use first_source for company discovery.",
        capabilities.providers,
    )
    assert [item["provider_id"] for item in exact["requests"]] == [
        "first_source"
    ]


def test_source_add_registration_diff_must_match_approved_attestation():
    provider = _provider_doc(
        "community_accounts",
        origin="source_add",
    )
    context = approved_source_router_suggestions(
        "Use community accounts for company discovery.",
        _capabilities(provider, private_loaded=False).providers,
    )
    good = _router_runtime_diff(
        _EMPTY_ROUTER_RUNTIME,
        _router_registration_source(provider_id="community_accounts"),
    )
    assert validate_source_add_registration_diff(
        good,
        context,
        existing_runtime_source=_EMPTY_ROUTER_RUNTIME,
    ) == []

    wrong_manifest = good.replace("a" * 64, "b" * 64)
    assert validate_source_add_registration_diff(
        wrong_manifest,
        context,
        existing_runtime_source=_EMPTY_ROUTER_RUNTIME,
    ) == ["source_add_registration_patched_source_invalid"]

    unapproved_provider = good.replace(
        "community_accounts",
        "other_accounts",
    )
    assert "source_add_registration_unapproved_provider" in (
        validate_source_add_registration_diff(
            unapproved_provider,
            context,
            existing_runtime_source=_EMPTY_ROUTER_RUNTIME,
        )
    )

    existing = """
class SourceAddRoutingRegistration:
    pass

SOURCE_ADD_ROUTING_REGISTRATIONS = (
    SourceAddRoutingRegistration(
        provider_id="community_accounts",
        stage="candidate_acquisition",
        revision="source-add-aaaaaaaaaaaa",
        manifest_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        priority=80,
        capabilities=("candidate.provider_discovery",),
        idempotency="idempotent",
        cost_class="metered",
        unit_cost=0.005,
        max_calls=1,
        max_results=100,
        timeout_seconds=60.0,
        evidence_types=("provider_database",),
        best_for=("icp.structured_eligible",),
        avoid_when=(),
        best_for_description="Approved SOURCE_ADD company-discovery provider for structured ICP acquisition.",
        avoid_when_description="Avoid when the consumer binding is unavailable, unhealthy, outside its approved categories, or over budget.",
    ),
)
"""
    child_diff = """diff --git a/sourcing_model/discovery.py b/sourcing_model/discovery.py
--- a/sourcing_model/discovery.py
+++ b/sourcing_model/discovery.py
@@ -1,2 +1,2 @@
-VALUE = 1
+VALUE = 2
"""
    assert validate_source_add_registration_diff(
        child_diff,
        context,
        existing_runtime_source=existing,
    ) == []


def test_source_add_registration_diff_requires_every_stage_and_exact_bounds():
    provider = _provider_doc(
        "community_source",
        origin="source_add",
    )
    context = approved_source_router_suggestions(
        (
            "Use community source for company discovery and intent "
            "discovery."
        ),
        _capabilities(provider, private_loaded=False).providers,
    )
    assert len(context["requests"]) == 2
    candidate_source = _router_registration_source(
        provider_id="community_source",
    )
    candidate_only = _router_runtime_diff(
        _EMPTY_ROUTER_RUNTIME,
        candidate_source,
    )
    assert validate_source_add_registration_diff(
        candidate_only,
        context,
        existing_runtime_source=_EMPTY_ROUTER_RUNTIME,
    ) == ["source_add_registration_missing_approved_request"]

    altered_timeout = candidate_only.replace(
        "timeout_seconds=60.0",
        "timeout_seconds=600.0",
    )
    candidate_context = {
        **context,
        "requests": [
            item
            for item in context["requests"]
            if item["stage"] == "candidate_acquisition"
        ],
    }
    assert validate_source_add_registration_diff(
        altered_timeout,
        candidate_context,
        existing_runtime_source=_EMPTY_ROUTER_RUNTIME,
    ) == [
        "source_add_registration_patched_source_invalid",
    ]

    direct_extra_tool_source = candidate_source.replace(
        ")\n",
        '    ToolDefinition(tool_id="intent.source_add.evil_source", origin=ORIGIN_SOURCE_ADD),\n)\n',
        1,
    )
    direct_extra_tool = _router_runtime_diff(
        _EMPTY_ROUTER_RUNTIME,
        direct_extra_tool_source,
    )
    errors = validate_source_add_registration_diff(
        direct_extra_tool,
        candidate_context,
        existing_runtime_source=_EMPTY_ROUTER_RUNTIME,
    )
    assert "source_add_registration_unapproved_tool" in errors
    assert "source_add_registration_direct_definition_forbidden" in errors

    extra_registration_source = candidate_source.replace(
        "\n)\n",
        """
    SourceAddRoutingRegistration(
        provider_id="community_source",
        stage="intent_evidence",
        revision="source-add-aaaaaaaaaaaa",
        manifest_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        priority=35,
        capabilities=("intent.provider_evidence",),
        idempotency="idempotent",
        cost_class="metered",
        unit_cost=0.005,
        max_calls=1,
        max_results=1,
        timeout_seconds=30.0,
        intent_categories=(),
        evidence_types=("external",),
        best_for=("intent.general",),
        avoid_when=(),
        best_for_description="Approved SOURCE_ADD provider for company-scoped intent-evidence discovery.",
        avoid_when_description="Avoid when the consumer binding is unavailable, unhealthy, outside its approved categories, or over budget.",
    ),
)
""",
    )
    extra_registration = _router_runtime_diff(
        _EMPTY_ROUTER_RUNTIME,
        extra_registration_source,
    )
    assert "source_add_registration_unapproved_registration" in (
        validate_source_add_registration_diff(
            extra_registration,
            candidate_context,
            existing_runtime_source=_EMPTY_ROUTER_RUNTIME,
        )
    )

    computed_registration_source = candidate_source.replace(
        "\n)\n",
        "\n    SourceAddRoutingRegistration(**untrusted_registration),\n)\n",
    )
    computed_registration = _router_runtime_diff(
        _EMPTY_ROUTER_RUNTIME,
        computed_registration_source,
    )
    assert "source_add_registration_patched_source_invalid" in (
        validate_source_add_registration_diff(
            computed_registration,
            candidate_context,
            existing_runtime_source=_EMPTY_ROUTER_RUNTIME,
        )
    )


def test_source_add_registration_diff_without_approved_request_fails_closed():
    unauthorized_source = _EMPTY_ROUTER_RUNTIME.replace(
        "SOURCE_ADD_ROUTING_REGISTRATIONS = (\n)",
        """SOURCE_ADD_ROUTING_REGISTRATIONS = (
    SourceAddRoutingRegistration(
        provider_id="unapproved_source",
        stage="candidate_acquisition",
    ),
)""",
    )
    unauthorized = _router_runtime_diff(
        _EMPTY_ROUTER_RUNTIME,
        unauthorized_source,
    )
    assert validate_source_add_registration_diff(
        unauthorized,
        None,
        existing_runtime_source=_EMPTY_ROUTER_RUNTIME,
    ) == ["source_add_registration_patched_source_invalid"]

    direct_existing = (
        "class SourceAddRoutingRegistration:\n"
        "    pass\n\n"
        "TOOLS = (\n"
        ")\n"
        "SOURCE_ADD_ROUTING_REGISTRATIONS = ()\n"
    )
    direct_changed = direct_existing.replace(
        "TOOLS = (\n)",
        """TOOLS = (
    ToolDefinition(tool_id="intent.source_add.unapproved", origin=ORIGIN_SOURCE_ADD),
)""",
    )
    direct_definition = _router_runtime_diff(
        direct_existing,
        direct_changed,
    )
    assert validate_source_add_registration_diff(
        direct_definition,
        {"requests": []},
        existing_runtime_source=direct_existing,
    ) == [
        "source_add_registration_direct_definition_forbidden",
        "source_add_registration_patched_source_invalid",
        "source_add_registration_unapproved_tool",
    ]


@pytest.mark.parametrize(
    "mutation",
    (
        "SOURCE_ADD_ROUTING_REGISTRATIONS += load_unreviewed_registrations()\n",
        "del SOURCE_ADD_ROUTING_REGISTRATIONS\n",
        "if True:\n    SOURCE_ADD_ROUTING_REGISTRATIONS = load_unreviewed_registrations()\n",
        'globals()["SOURCE_ADD_ROUTING_REGISTRATIONS"] = load_unreviewed_registrations()\n',
        'globals()["SOURCE_" + "ADD_ROUTING_REGISTRATIONS"] = ()\n',
        'setattr(sys.modules[__name__], "SOURCE_" + "ADD_ROUTING_REGISTRATIONS", ())\n',
        '__builtins__["glo" + "bals"]()["SOURCE_" + "ADD_ROUTING_REGISTRATIONS"] = ()\n',
        'globals()["SourceAddRoutingRegistration"] = unreviewed.SourceAddRoutingRegistration\n',
        "SourceAddRoutingRegistration.__new__ = unreviewed_new\n",
        'setattr(SourceAddRoutingRegistration, "__new__", unreviewed_new)\n',
        "runtime_self = sys.modules[__name__]\nruntime_self.SourceAddRoutingRegistration = unreviewed_new\n",
        "import unreviewed as SOURCE_ADD_ROUTING_REGISTRATIONS\n",
        "try:\n    pass\nexcept RuntimeError as SOURCE_ADD_ROUTING_REGISTRATIONS:\n    pass\n",
        "match value:\n    case SOURCE_ADD_ROUTING_REGISTRATIONS:\n        pass\n",
    ),
)
def test_source_add_registration_rejects_unreviewed_runtime_mutation(mutation):
    changed = _EMPTY_ROUTER_RUNTIME + mutation
    assert validate_source_add_registration_diff(
        _router_runtime_diff(_EMPTY_ROUTER_RUNTIME, changed),
        None,
        existing_runtime_source=_EMPTY_ROUTER_RUNTIME,
    ) == ["source_add_registration_patched_source_invalid"]

    existing_registration = _router_registration_source(
        provider_id="approved_source",
    )
    removal = _router_runtime_diff(
        existing_registration,
        _EMPTY_ROUTER_RUNTIME,
    )
    assert validate_source_add_registration_diff(
        removal,
        None,
        existing_runtime_source=existing_registration,
    ) == [
        "source_add_registration_removal_forbidden",
        "source_add_registration_without_approved_request",
    ]


def test_source_add_registration_diff_validates_complete_patched_registry():
    provider = _provider_doc("community_accounts", origin="source_add")
    context = approved_source_router_suggestions(
        "Use community accounts for company discovery.",
        _capabilities(provider, private_loaded=False).providers,
    )
    expected = _router_registration_source()

    changed_cost = expected.replace("unit_cost=0.005", "unit_cost=0.01")
    assert validate_source_add_registration_diff(
        _router_runtime_diff(expected, changed_cost),
        context,
        existing_runtime_source=expected,
    ) == [
        "source_add_registration_missing_approved_request",
        "source_add_registration_unapproved_registration",
    ]

    unknown_keyword = expected.replace(
        "    ),\n",
        "        unreviewed_behavior=True,\n    ),\n",
    )
    assert validate_source_add_registration_diff(
        _router_runtime_diff(expected, unknown_keyword),
        context,
        existing_runtime_source=expected,
    ) == ["source_add_registration_patched_source_invalid"]

    dead_code_constructor = expected + """
if False:
    SourceAddRoutingRegistration(
        provider_id="community_accounts",
        stage="candidate_acquisition",
    )
"""
    assert validate_source_add_registration_diff(
        _router_runtime_diff(expected, dead_code_constructor),
        context,
        existing_runtime_source=expected,
    ) == ["source_add_registration_patched_source_invalid"]


def test_source_add_registration_preserves_reviewed_multi_call_budget():
    provider = _provider_doc("community_accounts", origin="source_add")
    current_context = approved_source_router_suggestions(
        "Use community accounts for company discovery.",
        _capabilities(provider, private_loaded=False).providers,
    )
    current = _v8_router_registration_source(
        current_context["requests"][0]
    )
    provider["planner_summary"]["max_calls"] = 2
    context = approved_source_router_suggestions(
        "Use community accounts for company discovery.",
        _capabilities(provider, private_loaded=False).providers,
    )
    assert context["requests"][0]["max_calls"] == 2
    reviewed = _v8_router_registration_source(context["requests"][0])
    assert validate_source_add_registration_diff(
        _router_runtime_diff(current, reviewed),
        context,
        existing_runtime_source=current,
    ) == []

    unsafe_context = {
        **context,
        "requests": [
            {
                **context["requests"][0],
                "max_calls": 10_001,
            }
        ],
    }
    assert validate_source_add_registration_diff(
        _router_runtime_diff(current, reviewed),
        unsafe_context,
        existing_runtime_source=current,
    ) == ["source_add_approved_request_invalid"]


def test_source_add_registration_v7_v8_v7_transition_is_hash_dispatched():
    provider = _provider_doc("community_accounts", origin="source_add")
    context = approved_source_router_suggestions(
        "Use community accounts for company discovery.",
        _capabilities(provider, private_loaded=False).providers,
    )
    request = context["requests"][0]
    v7_source = _router_registration_source()
    v8_source = _v8_router_registration_source(request)

    assert validate_source_add_registration_diff(
        "",
        context,
        existing_runtime_source=v7_source,
    ) == []
    assert validate_source_add_registration_diff(
        "",
        context,
        existing_runtime_source=v8_source,
    ) == []
    assert validate_source_add_registration_diff(
        "",
        context,
        existing_runtime_source=v7_source,
    ) == []

    legacy_request = {
        key: value
        for key, value in request.items()
        if key
        not in {
            "binding_manifest",
            "binding_requirements",
            "category_contracts",
            "execution_mode",
            "legacy_v7_manifest_sha256",
            "provisioning_provenance_sha256",
        }
    }
    legacy_request.update(
        {
            "schema_version": "leadpoet.routerverse_source_incorporation.v2",
            "manifest_sha256": "a" * 64,
            "revision": "source-add-aaaaaaaaaaaa",
        }
    )
    legacy_context = {**context, "requests": [legacy_request]}
    assert validate_source_add_registration_diff(
        "",
        legacy_context,
        existing_runtime_source=v7_source,
    ) == []
    assert validate_source_add_registration_diff(
        "",
        legacy_context,
        existing_runtime_source=v8_source,
    ) == ["source_add_approved_request_invalid"]

    downgraded_v8_source = v8_source.replace(
        "SOURCE_ADD_BINDING_MANIFEST_SCHEMA_VERSION",
        "RENAMED_SOURCE_ADD_BINDING_MANIFEST_SCHEMA_VERSION",
    )
    assert validate_source_add_registration_diff(
        _router_runtime_diff(v8_source, downgraded_v8_source),
        context,
        existing_runtime_source=v8_source,
    ) == ["source_add_registration_patched_source_invalid"]

    rebound_v8_source = v8_source.replace(
        "class SourceAddRoutingRegistration:\n    pass\n",
        "class SourceAddRoutingRegistration:\n    pass\n\n"
        "SourceAddRoutingRegistration = "
        "unreviewed.SourceAddRoutingRegistration\n",
    )
    assert validate_source_add_registration_diff(
        _router_runtime_diff(v8_source, rebound_v8_source),
        context,
        existing_runtime_source=v8_source,
    ) == ["source_add_registration_patched_source_invalid"]

    constant_bound_v7_source = v7_source.replace(
        "stage='candidate_acquisition'",
        "stage=STAGE_CANDIDATE_ACQUISITION",
    )
    assert validate_source_add_registration_diff(
        _router_runtime_diff(v7_source, constant_bound_v7_source),
        context,
        existing_runtime_source=v7_source,
    ) == ["source_add_registration_patched_source_invalid"]


def test_source_add_registration_v8_rejects_bound_and_manifest_tampering():
    provider = _provider_doc("community_accounts", origin="source_add")
    context = approved_source_router_suggestions(
        "Use community accounts for company discovery.",
        _capabilities(provider, private_loaded=False).providers,
    )
    source = _v8_router_registration_source(context["requests"][0])
    for field_name, unsafe_value in (
        ("priority", 10_001),
        ("max_calls", 10_001),
        ("max_results", 100_001),
        ("timeout_seconds", 3_600.001),
    ):
        unsafe_context = {
            **context,
            "requests": [
                {
                    **context["requests"][0],
                    field_name: unsafe_value,
                }
            ],
        }
        assert validate_source_add_registration_diff(
            "",
            unsafe_context,
            existing_runtime_source=source,
        ) == ["source_add_approved_request_invalid"]

    tampered_manifest = {
        **context,
        "requests": [
            {
                **context["requests"][0],
                "binding_manifest": {
                    **context["requests"][0]["binding_manifest"],
                    "max_calls": 2,
                },
            }
        ],
    }
    assert validate_source_add_registration_diff(
        "",
        tampered_manifest,
        existing_runtime_source=source,
    ) == ["source_add_approved_request_invalid"]
    assert validate_source_add_registration_diff(
        "",
        tampered_manifest,
        existing_runtime_source=_router_registration_source(),
    ) == ["source_add_approved_request_invalid"]

    numeric_type_tamper = {
        **context,
        "requests": [
            {
                **context["requests"][0],
                "binding_manifest": {
                    **context["requests"][0]["binding_manifest"],
                    "max_calls": 1.0,
                },
            }
        ],
    }
    for existing_source in (source, _router_registration_source()):
        assert validate_source_add_registration_diff(
            "",
            numeric_type_tamper,
            existing_runtime_source=existing_source,
        ) == ["source_add_approved_request_invalid"]


def test_source_add_v8_rejects_costs_that_normalize_across_zero():
    for cost_class in ("free", "metered"):
        provider = _provider_doc(
            f"tiny_{cost_class}_source",
            origin="source_add",
        )
        provider["planner_summary"].update(
            {
                "cost_class": cost_class,
                "unit_cost": 0.0000004,
            }
        )
        context = approved_source_router_suggestions(
            f"Use tiny {cost_class} source for company discovery.",
            _capabilities(provider, private_loaded=False).providers,
        )
        assert context["requests"] == []
        assert context["clarifications"][0]["reason_code"] == (
            "approved_source_routing_contract_invalid"
        )


def test_source_add_registration_diff_allows_only_exact_approved_replacement():
    provider = _provider_doc("community_accounts", origin="source_add")
    context = approved_source_router_suggestions(
        "Use community accounts for company discovery.",
        _capabilities(provider, private_loaded=False).providers,
    )
    stale = _router_registration_source(manifest="b" * 64)
    expected = _router_registration_source()
    assert validate_source_add_registration_diff(
        _router_runtime_diff(stale, expected),
        context,
        existing_runtime_source=stale,
    ) == []

    malformed_context = {
        **context,
        "requests": [
            {
                **context["requests"][0],
                "registration_symbol": "sourcing_model/routing/other.py::TOOLS",
            }
        ],
    }
    assert validate_source_add_registration_diff(
        _router_runtime_diff(stale, expected),
        malformed_context,
        existing_runtime_source=stale,
    ) == ["source_add_approved_request_invalid"]


def test_source_add_registration_diff_migrates_unchanged_legacy_guidance_only():
    provider = _provider_doc("community_accounts", origin="source_add")
    context = approved_source_router_suggestions(
        "Use community accounts for company discovery.",
        _capabilities(provider, private_loaded=False).providers,
    )
    current = _router_registration_source()
    legacy = "\n".join(
        line
        for line in current.splitlines()
        if not any(
            field in line
            for field in (
                "best_for=",
                "avoid_when=",
                "best_for_description=",
                "avoid_when_description=",
            )
        )
    ) + "\n"

    assert validate_source_add_registration_diff(
        _router_runtime_diff(
            legacy,
            legacy + "ROUTER_BATCH_SIZE = 20\n",
        ),
        None,
        existing_runtime_source=legacy,
    ) == []
    assert validate_source_add_registration_diff(
        _router_runtime_diff(
            legacy,
            legacy.replace("unit_cost=0.005", "unit_cost=0.006"),
        ),
        context,
        existing_runtime_source=legacy,
    ) == ["source_add_registration_patched_source_invalid"]
    assert validate_source_add_registration_diff(
        _router_runtime_diff(legacy, current),
        context,
        existing_runtime_source=legacy,
    ) == []


def test_source_add_registration_diff_rejects_wrong_target_and_unapplicable_patch():
    provider = _provider_doc("community_accounts", origin="source_add")
    context = approved_source_router_suggestions(
        "Use community accounts for company discovery.",
        _capabilities(provider, private_loaded=False).providers,
    )
    existing = _router_registration_source()
    wrong_target = """diff --git a/sourcing_model/helper.py b/sourcing_model/helper.py
--- a/sourcing_model/helper.py
+++ b/sourcing_model/helper.py
@@ -1 +1,2 @@
 VALUE = 1
+SOURCE_ADD_ROUTING_REGISTRATIONS = ()
"""
    assert validate_source_add_registration_diff(
        wrong_target,
        context,
        existing_runtime_source=existing,
    ) == ["source_add_registration_wrong_model_target"]

    unapplicable = _router_runtime_diff(
        _EMPTY_ROUTER_RUNTIME,
        _router_registration_source(),
    )
    assert validate_source_add_registration_diff(
        unapplicable,
        context,
        existing_runtime_source=existing,
    ) == ["source_add_registration_runtime_diff_unapplicable"]


def test_source_add_registration_diff_allows_unrelated_runtime_change():
    existing = _EMPTY_ROUTER_RUNTIME + "ROUTER_BATCH_SIZE = 10\n"
    changed = _EMPTY_ROUTER_RUNTIME + "ROUTER_BATCH_SIZE = 20\n"
    assert validate_source_add_registration_diff(
        _router_runtime_diff(existing, changed),
        None,
        existing_runtime_source=existing,
    ) == []


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
