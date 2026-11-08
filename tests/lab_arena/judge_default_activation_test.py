"""Reviewed defaults affect new rounds; historical policy bytes stay frozen."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from lab_arena import contracts, reasoning_capture, scoring
from lab_arena.service import ServiceError
from tests.lab_arena.judge_policy_integration_test import NOW, service

DEEPSEEK = "deepseek/deepseek-v4.1-flash-1017"
GPT_OSS = "openai/gpt-oss-120b"
SOL = "openai/gpt-6.1-sol"
GLM = "z-ai/glm-5.3-flash"
MODELS = {
    "company_fit_reverification": DEEPSEEK,
    "intent_signal_judge": GPT_OSS,
    "intent_details_review": DEEPSEEK,
    "intent_three_stage_stage3": DEEPSEEK,
    "intent_fetched_evidence": GPT_OSS,
    "role_batch_check": GPT_OSS,
    "industry_evidence_review": DEEPSEEK,
    "industry_research_controller": GLM,
}
LEGACY_MODELS = {
    "company_fit_reverification": "perplexity/sonar",
    "intent_signal_judge": "openai/gpt-6-luna",
    "intent_details_review": "anthropic/claude-sonnet-5.5",
    "intent_three_stage_stage3": "perplexity/sonar-pro",
    "intent_fetched_evidence": "openai/gpt-6-luna",
    "role_batch_check": "google/gemini-3.5-flash-lite",
    "industry_evidence_review": SOL,
}
ROUTES = {DEEPSEEK: "nebius/fp8", GPT_OSS: "near-ai/fp4", GLM: "near-ai/fp8"}
OPTIONS = {
    DEEPSEEK: {"reasoning": {"effort": "low"}, "max_output_tokens": 8192,
               "industry_max_output_tokens": 16384},
    GPT_OSS: {"reasoning": {"effort": "low"}, "max_output_tokens": 4096},
    GLM: {"reasoning": {"effort": "low"}, "max_output_tokens": 8192},
}
EXCEPTIONS = {SOL: {"roles": ["industry_evidence_review"], "reason": "quality_gate_not_met"}}
PREVIOUS_MODELS = {**MODELS, "industry_evidence_review": SOL}
PREVIOUS_OPTIONS = deepcopy(OPTIONS)
del PREVIOUS_OPTIONS[DEEPSEEK]["industry_max_output_tokens"]


def test_reviewed_default_models_freeze_exact_routes_options_without_exception():
    policy = scoring.build_scorer_policy()
    assert policy["judge_models"] == MODELS
    bindings = policy["env_bindings"]
    assert bindings[contracts.JUDGE_REASONING_CAPTURE_BINDING] == reasoning_capture.POLICY
    assert json.loads(bindings[contracts.JUDGE_PROVIDER_ROUTES_BINDING]) == ROUTES
    assert json.loads(bindings[contracts.JUDGE_REQUEST_OPTIONS_BINDING]) == OPTIONS
    assert contracts.JUDGE_REASONING_EXCEPTIONS_BINDING not in bindings
    assert contracts.validate_scorer_policy(json.loads(json.dumps(policy))) == policy
    registry = reasoning_capture.load_registry()
    assert reasoning_capture.validate_judge_registry(registry, policy) == registry
    assert scoring.build_scorer_policy(judge_models=dict(MODELS)) == policy


def test_explicit_empty_routes_preserve_capture_off_without_automatic_options_or_exception():
    policy = scoring.build_scorer_policy(judge_provider_routes={})
    assert policy["judge_models"] == MODELS
    assert policy["env_bindings"] == {
        "ARENA_COMPANY_INPUT_CONTRACT": "linkedin_company_ampersand_v1",
    }
    environment = {}
    scoring.apply_policy_to_environment(policy, environ=environment,
        credentials={name: "test-not-secret" for name in scoring.CREDENTIAL_ENV_NAMES})
    for binding in (contracts.JUDGE_REASONING_CAPTURE_BINDING,
                    contracts.JUDGE_PROVIDER_ROUTES_BINDING,
                    contracts.JUDGE_REQUEST_OPTIONS_BINDING,
                    contracts.JUDGE_REASONING_EXCEPTIONS_BINDING):
        assert environment[binding] == ""


def test_custom_historical_models_keep_original_policy_bytes_and_explicit_route_semantics():
    legacy = scoring.build_scorer_policy(judge_models=LEGACY_MODELS)
    # Reconstruct the saved policy from before prospective input contracts.
    assert legacy["env_bindings"].pop("ARENA_COMPANY_INPUT_CONTRACT") == "linkedin_company_ampersand_v1"
    # Captured from the original builder on a8fcd5b1e before this default change.
    assert contracts.document_hash(legacy) == "sha256:4805efdc8bfca0ae0c33634c1731a54d3be08f85a482776ebf3a8d129f61e260"
    routes = {model: "historical/provider" for model in set(LEGACY_MODELS.values())}
    explicit = scoring.build_scorer_policy(judge_models=LEGACY_MODELS, judge_provider_routes=routes)
    assert json.loads(explicit["env_bindings"][contracts.JUDGE_PROVIDER_ROUTES_BINDING]) == routes
    assert contracts.JUDGE_REQUEST_OPTIONS_BINDING not in explicit["env_bindings"]
    assert contracts.JUDGE_REASONING_EXCEPTIONS_BINDING not in explicit["env_bindings"]


def test_saved_sol_research_policy_is_not_upgraded():
    previous = scoring.build_scorer_policy(
        judge_models={**PREVIOUS_MODELS, "industry_research_controller": SOL},
        judge_provider_routes={k: v for k, v in ROUTES.items() if k != GLM},
        judge_request_options={k: v for k, v in PREVIOUS_OPTIONS.items() if k != GLM},
        judge_reasoning_exceptions={SOL: {
            "roles": ["industry_evidence_review", "industry_research_controller"],
            "reason": "quality_gate_not_met",
        }},
    )
    assert previous["env_bindings"].pop("ARENA_COMPANY_INPUT_CONTRACT") == "linkedin_company_ampersand_v1"
    assert contracts.document_hash(previous) == "sha256:0148cfb34001c5162e20f81a0ecfc72170a1eee2560dc3592ecd16f109692595"
    environment = {}
    scoring.apply_policy_to_environment(previous, environ=environment,
        credentials={name: "test-not-secret" for name in scoring.CREDENTIAL_ENV_NAMES})
    assert environment["ARENA_INDUSTRY_RESEARCH_CONTROLLER_MODEL"] == SOL
    assert GLM not in environment[contracts.JUDGE_PROVIDER_ROUTES_BINDING]


def test_saved_sol_final_review_policy_keeps_original_bytes_and_explicit_exception():
    previous = scoring.build_scorer_policy(judge_models=PREVIOUS_MODELS,
        judge_provider_routes=ROUTES, judge_request_options=PREVIOUS_OPTIONS,
        judge_reasoning_exceptions=EXCEPTIONS)
    assert previous["env_bindings"].pop("ARENA_COMPANY_INPUT_CONTRACT") == "linkedin_company_ampersand_v1"
    assert contracts.document_hash(previous) == "sha256:a9682fa60b314b07c9cd9ee3f1ab5383c130750593e8ea038fc3ad87c4504977"
    assert previous["judge_models"]["industry_evidence_review"] == SOL
    assert json.loads(previous["env_bindings"][contracts.JUDGE_REQUEST_OPTIONS_BINDING]) == PREVIOUS_OPTIONS
    assert contracts.judge_reasoning_exceptions(previous) == EXCEPTIONS
    assert contracts.validate_scorer_policy(json.loads(json.dumps(previous))) == previous
    with pytest.raises(contracts.ArenaContractError, match="reasoning exceptions"):
        scoring.build_scorer_policy(judge_reasoning_exceptions={})


def test_default_industry_cap_does_not_increase_other_role_or_api_caps():
    for role, model in MODELS.items():
        for api in ("openrouter.chat", "openrouter.responses"):
            maximum = contracts.judge_request_max_output_tokens(OPTIONS[model], model=model,
                industry_model=DEEPSEEK, operation_id=api, role=role)
            expected = 16384 if role == "industry_evidence_review" and api == "openrouter.chat" else OPTIONS[model]["max_output_tokens"]
            assert maximum == expected


@pytest.mark.parametrize("research", [False, True])
@pytest.mark.parametrize("industry_override", [False, True])
def test_native_industry_reaches_broker_with_exact_frozen_contract(monkeypatch, research, industry_override):
    from lab_arena import broker
    from qualification.scoring import company_evidence_investigator as investigator
    from qualification.scoring import industry_classification as industry
    from tests.lab_arena import test_lab_arena_broker as bt
    policy = scoring.build_scorer_policy()
    if not industry_override:
        policy["env_bindings"][contracts.JUDGE_REQUEST_OPTIONS_BINDING] = json.dumps(PREVIOUS_OPTIONS)
    environment = {}
    scoring.apply_policy_to_environment(policy, environ=environment,
        credentials={name: "test-not-secret" for name in scoring.CREDENTIAL_ENV_NAMES})
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    calls = []
    async def capture(_session, _url, **kwargs):
        calls.append(deepcopy(kwargs["payload"]))
        return 400, {}
    monkeypatch.setattr(investigator, "_post_json", capture)
    url = "https://acme.example/"
    text = "Acme manufactures and sells its own industrial valves. " * 100
    criteria = {"industry": "Manufacturing", "subindustry": "", "product_service": "Valves"}
    asyncio.run(investigator._review_industry_sources(None, model=DEEPSEEK,
        openrouter_key="test-not-secret", started=time.monotonic(), research=research,
        company_locator={"name": "Acme", "website": url}, criteria=criteria,
        fetched_pages={url: text}, fetched_final_urls={url: url},
        first_party_domains={"acme.example"}, identity_names={"acme"},
        identity_anchor={"submitted_domain": "acme.example", "observed_domain": "acme.example",
                         "verified_domain": "acme.example"}, evaluation_date_value="2026-11-05"))
    assert len(calls) == 1
    expected = industry.request_payload(industry.prepare(criteria,
        {"name": "Acme", "website": url, "linkedin": ""},
        [{"source_id": "s0", "source_url": url, "body": text.strip()}],
        evaluation_date="2026-11-05"), model=DEEPSEEK, research=research)
    maximum = 16384 if industry_override else 8192
    expected.update(max_tokens=maximum, reasoning={"effort": "low", "exclude": False},
                    provider={"only": ["nebius/fp8"]})
    expected["messages"][0]["content"] += (industry.REASONING_RESPONSE_CONTRACT
        + ("\nProvide a nonempty reason explaining how the cited source spans establish the decision."
           if industry_override else ""))
    if industry_override:
        expected["response_format"]["json_schema"]["schema"]["properties"]["reason"]["minLength"] = 1
    assert calls[0] == expected
    assert len(expected["messages"]) > 3
    table = bt.price_table()
    for model in set(policy["judge_models"].values()):
        table["models"][model] = deepcopy(next(iter(table["models"].values())))
    store = bt.FakeLedgerStore()
    transport = bt.FakeTransport([(200, {"choices": [], "usage": {"cost": "0.0001"}})])
    instance = broker.Broker(store=store, transport=transport, price_table=table,
        key_for=lambda provider: bt.HOST_KEYS[provider], judge_models=policy["judge_models"])
    context = replace(bt.CONTEXT, kind="score", judge_reasoning_capture_policy=reasoning_capture.POLICY,
        judge_reasoning_route_registry=reasoning_capture.load_registry(),
        judgment_cache_scope={"scorer_policy": policy})
    response = instance.execute(context, operation_id="openrouter.chat", parameters=calls[0],
        action_sequence=0, timeout_ms=30000)
    assert response.status == 200
    wire = json.loads(transport.sent[0]["body"])
    assert {k: v for k, v in wire.items() if k != "provider"} == {
        **{k: v for k, v in expected.items() if k != "provider"}, "stream": False}
    assert wire["provider"]["only"] == ["nebius/fp8"]
    assert wire["provider"]["zdr"] is True and wire["provider"]["data_collection"] == "deny"
    assert wire["provider"]["allow_fallbacks"] is False
    assert next(iter(store.calls.values()))["call_doc"]["max_output_tokens"] == maximum


def test_service_freezes_new_defaults_and_keeps_both_saved_rounds_byte_identical():
    app = service()
    app._scorer_policy = scoring.build_scorer_policy(judge_models=LEGACY_MODELS)
    old = app.create_round(NOW, round_id="arena-2026-11-05-legacy")
    old_bytes = contracts.canonical_json(old)
    app._scorer_policy = scoring.build_scorer_policy()
    app._config.reasoning_registry_source = reasoning_capture.load_registry
    assert contracts.canonical_json(app.create_round(NOW, round_id=old["round_id"])) == old_bytes
    future = app.create_round(NOW + timedelta(days=1), round_id="arena-2026-11-06-open")
    assert future["scorer_policy"]["judge_models"] == MODELS
    assert future["judge_reasoning_route_registry"] == reasoning_capture.load_registry()
    assert contracts.validate_round_configuration(json.loads(json.dumps(future))) == future
    future_bytes = contracts.canonical_json(future)
    app._scorer_policy = scoring.build_scorer_policy(judge_models=LEGACY_MODELS)
    assert contracts.canonical_json(app.create_round(NOW, round_id=future["round_id"])) == future_bytes
    assert contracts.canonical_json(app.create_round(NOW, round_id=old["round_id"])) == old_bytes


@pytest.mark.parametrize("model,provider,api", [
    (DEEPSEEK, "nebius/fp8", "openrouter.chat"),
    (DEEPSEEK, "nebius/fp8", "openrouter.responses"),
    (GPT_OSS, "near-ai/fp4", "openrouter.chat"),
    (GLM, "near-ai/fp8", "openrouter.chat"),
])
def test_new_default_round_fails_closed_before_persistence_if_exact_route_is_missing(model, provider, api):
    registry = reasoning_capture.load_registry()
    registry["approved_routes"] = [r for r in registry["approved_routes"]
        if (r["model_id"], r["provider"], r["api"]) != (model, provider, api)]
    app = service()
    app._scorer_policy = scoring.build_scorer_policy()
    app._config.reasoning_registry_source = lambda: registry
    with pytest.raises(ServiceError, match="judge_reasoning_capture_activation_not_ready"):
        app.create_round(NOW, round_id="arena-2026-11-05-unready")
    assert app._store.rows == {}


def test_claim_helper_keeps_native_500_token_and_15_second_bound_under_default_capture(monkeypatch):
    from qualification.scoring import claim_equivalence, verification_helpers
    environment = {}
    scoring.apply_policy_to_environment(scoring.build_scorer_policy(), environ=environment,
        credentials={name: "test-not-secret" for name in scoring.CREDENTIAL_ENV_NAMES})
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(verification_helpers, "OPENROUTER_API_KEY", "test-not-secret")
    requests = []
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return False
        async def post(self, url, **kwargs):
            requests.append((url, deepcopy(kwargs)))
            return SimpleNamespace(raise_for_status=lambda: None,
                json=lambda: {"choices": [{"finish_reason": "stop", "message": {
                    "content": '{"decision":"equivalent","material_differences":[]}'}}]})
    monkeypatch.setattr(verification_helpers.httpx, "AsyncClient", Client)
    proof = asyncio.run(claim_equivalence.verify_claim_equivalence(
        {"description": "Acme acquired Beta."}, {"description": "Acme purchased Beta."}, model=GPT_OSS))
    assert proof is not None and len(requests) == 1
    url, request = requests[0]
    assert url.endswith("/chat/completions") and request["timeout"] == 15
    body = request["json"]
    assert body["model"] == GPT_OSS and body["max_tokens"] == 500
    assert body["provider"] == {"only": ["near-ai/fp4"]}
    assert body["reasoning"] == {"effort": "low", "exclude": False}
