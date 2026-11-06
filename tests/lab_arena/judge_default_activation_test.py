"""Reviewed defaults affect new rounds; historical policy bytes stay frozen."""
from copy import deepcopy
from datetime import timedelta
import asyncio
import json
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
    "industry_evidence_review": SOL,
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
    DEEPSEEK: {"reasoning": {"effort": "low"}, "max_output_tokens": 8192},
    GPT_OSS: {"reasoning": {"effort": "low"}, "max_output_tokens": 4096},
    GLM: {"reasoning": {"effort": "low"}, "max_output_tokens": 8192},
}
EXCEPTIONS = {SOL: {"roles": ["industry_evidence_review"], "reason": "quality_gate_not_met"}}


def test_reviewed_default_models_freeze_exact_routes_options_and_exception():
    policy = scoring.build_scorer_policy()
    assert policy["judge_models"] == MODELS
    bindings = policy["env_bindings"]
    assert bindings[contracts.JUDGE_REASONING_CAPTURE_BINDING] == reasoning_capture.POLICY
    assert json.loads(bindings[contracts.JUDGE_PROVIDER_ROUTES_BINDING]) == ROUTES
    assert json.loads(bindings[contracts.JUDGE_REQUEST_OPTIONS_BINDING]) == OPTIONS
    assert json.loads(bindings[contracts.JUDGE_REASONING_EXCEPTIONS_BINDING]) == EXCEPTIONS
    assert contracts.validate_scorer_policy(json.loads(json.dumps(policy))) == policy
    registry = reasoning_capture.load_registry()
    assert reasoning_capture.validate_judge_registry(registry, policy) == registry
    assert scoring.build_scorer_policy(judge_models=dict(MODELS)) == policy


def test_explicit_empty_routes_preserve_capture_off_without_automatic_options_or_exception():
    policy = scoring.build_scorer_policy(judge_provider_routes={})
    assert policy["judge_models"] == MODELS
    assert policy["env_bindings"] == {}
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
    # Captured from the original builder on a8fcd5b1e before this default change.
    assert contracts.document_hash(legacy) == "sha256:4805efdc8bfca0ae0c33634c1731a54d3be08f85a482776ebf3a8d129f61e260"
    routes = {model: "historical/provider" for model in set(LEGACY_MODELS.values())}
    explicit = scoring.build_scorer_policy(judge_models=LEGACY_MODELS, judge_provider_routes=routes)
    assert json.loads(explicit["env_bindings"][contracts.JUDGE_PROVIDER_ROUTES_BINDING]) == routes
    assert contracts.JUDGE_REQUEST_OPTIONS_BINDING not in explicit["env_bindings"]
    assert contracts.JUDGE_REASONING_EXCEPTIONS_BINDING not in explicit["env_bindings"]


def test_saved_sol_research_policy_is_not_upgraded():
    previous = scoring.build_scorer_policy(
        judge_models={**MODELS, "industry_research_controller": SOL},
        judge_provider_routes={k: v for k, v in ROUTES.items() if k != GLM},
        judge_request_options={k: v for k, v in OPTIONS.items() if k != GLM},
        judge_reasoning_exceptions={SOL: {
            "roles": ["industry_evidence_review", "industry_research_controller"],
            "reason": "quality_gate_not_met",
        }},
    )
    assert contracts.document_hash(previous) == "sha256:0148cfb34001c5162e20f81a0ecfc72170a1eee2560dc3592ecd16f109692595"
    environment = {}
    scoring.apply_policy_to_environment(previous, environ=environment,
        credentials={name: "test-not-secret" for name in scoring.CREDENTIAL_ENV_NAMES})
    assert environment["ARENA_INDUSTRY_RESEARCH_CONTROLLER_MODEL"] == SOL
    assert GLM not in environment[contracts.JUDGE_PROVIDER_ROUTES_BINDING]


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
