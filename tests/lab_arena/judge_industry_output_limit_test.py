"""An optional industry ceiling never expands another frozen judge request."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from lab_arena import broker, contracts, reasoning_capture, scoring
from qualification.scoring import industry_classification, openrouter_options
from tests.lab_arena.reasoning_capture_test import _reviewed_registry
from tests.lab_arena import test_lab_arena_broker as bt

MODEL = "test/shared-judge"
PROVIDER = "example/verified"


def policy(*, override=16384):
    option = {"reasoning": {"effort": "low"}, "max_output_tokens": 8192}
    if override is not None:
        option["industry_max_output_tokens"] = override
    return scoring.build_scorer_policy(
        judge_models={role: MODEL for role in scoring.DEFAULT_JUDGE_MODELS},
        judge_provider_routes={MODEL: PROVIDER}, judge_request_options={MODEL: option})


def bind(monkeypatch, frozen):
    environment = {}
    scoring.apply_policy_to_environment(frozen, environ=environment,
        credentials={name: "test-not-secret" for name in scoring.CREDENTIAL_ENV_NAMES})
    for name, value in environment.items():
        monkeypatch.setenv(name, value)


def industry(research=False):
    packet = industry_classification.prepare(
        {"industry": "", "subindustry": "", "product_service": "Manufactures valves."},
        {"name": "Fixture Works", "website": "https://fixture.example.org", "linkedin": ""},
        [{"source_id": "s0", "source_url": "https://fixture.example.org",
          "body": "Fixture Works manufactures and sells its own industrial valves."}],
        evaluation_date="2026-11-06")
    request = industry_classification.request_payload(packet, model=MODEL, research=research)
    request.pop("reasoning")  # The unchanged native investigator does this under its opt-in.
    request["max_tokens"] = 16384
    request["messages"][0]["content"] += industry_classification.REASONING_RESPONSE_CONTRACT
    return request


def registry():
    result = _reviewed_registry(model=MODEL)
    result["approved_routes"] += _reviewed_registry(model=MODEL, api="openrouter.responses")["approved_routes"]
    for route in result["approved_routes"]:
        route["max_output_tokens"] = 32768
    return result


def dispatch(parameters, frozen, *, api="openrouter.chat"):
    table = bt.price_table()
    table["models"][MODEL] = deepcopy(next(iter(table["models"].values())))
    store, transport = bt.FakeLedgerStore(), bt.FakeTransport([(200, {
        "choices": [], "usage": {"cost": "0.0001"}})])
    instance = broker.Broker(store=store, transport=transport,
        key_for=lambda provider: bt.HOST_KEYS[provider], price_table=table,
        judge_models=frozen["judge_models"])
    context = replace(bt.CONTEXT, kind="score", judge_reasoning_capture_policy=reasoning_capture.POLICY,
        judge_reasoning_route_registry=registry(), judgment_cache_scope={"scorer_policy": frozen})
    result = instance.execute(context, operation_id=api, parameters=parameters,
                              action_sequence=0, timeout_ms=30000)
    return result, store, transport


@pytest.mark.parametrize("research", [False, True])
@pytest.mark.parametrize("reason_minimum", [False, True])
def test_native_industry_override_reaches_exact_broker_wire(monkeypatch, research, reason_minimum):
    frozen = policy()
    assert contracts.validate_scorer_policy(json.loads(json.dumps(frozen))) == frozen
    assert reasoning_capture.validate_judge_registry(registry(), frozen) == registry()
    bind(monkeypatch, frozen)
    original = industry(research)
    if reason_minimum:
        original["response_format"]["json_schema"]["schema"]["properties"]["reason"]["minLength"] = 1
    path, prepared = openrouter_options.prepare_judge_request(original)
    assert path == "/chat/completions" and prepared["max_tokens"] == 16384
    assert original["max_tokens"] == 16384 and "provider" not in original
    result, store, transport = dispatch(prepared, frozen)
    assert result.status == 200
    assert next(iter(store.calls.values()))["call_doc"]["max_output_tokens"] == 16384
    wire = json.loads(transport.sent[0]["body"])
    assert wire["messages"] == prepared["messages"]
    assert wire["response_format"] == prepared["response_format"]
    assert wire["max_tokens"] == 16384
    assert wire["reasoning"] == {"effort": "low", "exclude": False}
    assert wire["provider"]["only"] == [PROVIDER]
    assert wire["provider"]["zdr"] is True
    assert wire["provider"]["data_collection"] == "deny"
    assert wire["provider"]["allow_fallbacks"] is False


def test_old_options_and_capture_off_keep_existing_behavior(monkeypatch):
    request = industry()
    bind(monkeypatch, policy(override=None))
    assert openrouter_options.prepare_judge_request(request)[1]["max_tokens"] == 8192
    ordinary = {"model": MODEL, "max_tokens": 16384,
                "messages": [{"role": "user", "content": "Assess company fit."}]}
    assert openrouter_options.prepare_judge_request(ordinary)[1]["max_tokens"] == 8192
    result, store, transport = dispatch(request, policy(override=None))
    assert result.status == 400 and not store.calls and not transport.sent
    monkeypatch.delenv("ARENA_JUDGE_REASONING_CAPTURE_POLICY")
    assert openrouter_options.prepare_judge_request(request) == ("/chat/completions", request)


def test_shared_model_fit_request_keeps_base_ceiling_on_wire(monkeypatch):
    frozen = policy()
    bind(monkeypatch, frozen)
    request = {"model": MODEL, "max_tokens": 16384,
               "messages": [{"role": "user", "content": "Review the supplied company-fit facts."}]}
    prepared = openrouter_options.prepare_judge_request(request)[1]
    assert prepared["max_tokens"] == 8192
    result, store, transport = dispatch(prepared, frozen)
    assert result.status == 200
    assert json.loads(transport.sent[0]["body"])["max_tokens"] == 8192
    assert next(iter(store.calls.values()))["call_doc"]["max_output_tokens"] == 8192


@pytest.mark.parametrize("damage", ["ordinary", "name_only", "missing_field", "wrong_enum",
    "bad_packet", "bad_metadata", "bad_source", "bad_items", "unbound_spans", "tools"])
def test_same_model_nonindustry_or_malformed_request_cannot_claim_override(monkeypatch, damage):
    frozen = policy()
    bind(monkeypatch, frozen)
    request = industry()
    schema = request["response_format"]["json_schema"]["schema"]
    if damage == "ordinary":
        request.pop("response_format")
    elif damage == "name_only":
        request["response_format"]["json_schema"]["schema"] = {"type": "object"}
    elif damage == "missing_field":
        del schema["properties"]["observed_activity"]
    elif damage == "wrong_enum":
        schema["properties"]["decision"]["enum"] = ["match"]
    elif damage == "bad_packet":
        request["messages"][1]["content"] = "Not an industry packet"
    elif damage == "bad_metadata":
        metadata = json.loads(request["messages"][1]["content"])
        metadata["criterion"]["product_service"] = 123
        request["messages"][1]["content"] = json.dumps(metadata)
    elif damage == "bad_source":
        sources = json.loads(request["messages"][2]["content"])
        del sources["untrusted_source_pages"][0]["source_url"]
        request["messages"][2]["content"] = json.dumps(sources)
    elif damage == "bad_items":
        schema["properties"]["missing_requirements"]["items"]["type"] = "integer"
    elif damage == "unbound_spans":
        schema["properties"]["evidence_span_ids"]["items"]["enum"] = ["invented-span"]
    elif damage == "tools":
        request["tools"] = [{"type": "function", "function": {"name": "find",
            "parameters": {"type": "object", "properties": {}}}}]
    assert openrouter_options.prepare_judge_request(request)[1]["max_tokens"] == 8192
    result, store, transport = dispatch(request, frozen)
    assert result.status == 400 and not store.calls and not transport.sent


def test_responses_and_tighter_comparisons_do_not_gain_industry_ceiling(monkeypatch):
    frozen = policy()
    bind(monkeypatch, frozen)
    path, prepared = openrouter_options.prepare_judge_request(industry(), hosted_search=True)
    assert path == "/responses" and prepared["max_output_tokens"] == 8192
    prepared["max_output_tokens"] = 16384
    result, store, transport = dispatch(prepared, frozen, api="openrouter.responses")
    assert result.status == 400 and not store.calls and not transport.sent
    small = {"model": MODEL, "max_tokens": 500,
             "messages": [{"role": "user", "content": "Compare claims."}]}
    assert openrouter_options.prepare_judge_request(small, output_token_floor=500)[1]["max_tokens"] == 500


def test_broker_enforces_frozen_override_before_reservation(monkeypatch):
    frozen = policy()
    bind(monkeypatch, frozen)
    prepared = openrouter_options.prepare_judge_request(industry())[1]
    prepared["max_tokens"] = 16385
    result, store, transport = dispatch(prepared, frozen)
    assert result.status == 400 and not store.calls and not transport.sent


@pytest.mark.parametrize("limit", [True, "16384", 0, 8191, 8192, 32769])
def test_invalid_frozen_industry_limit_is_rejected(limit):
    with pytest.raises(contracts.ArenaContractError, match="request options"):
        policy(override=limit)


def test_override_requires_the_frozen_industry_role_and_registry_ceiling(monkeypatch):
    frozen = policy()
    wrong = deepcopy(frozen)
    wrong["judge_models"]["industry_evidence_review"] = "test/other"
    routes = json.loads(wrong["env_bindings"][contracts.JUDGE_PROVIDER_ROUTES_BINDING])
    routes["test/other"] = PROVIDER
    wrong["env_bindings"][contracts.JUDGE_PROVIDER_ROUTES_BINDING] = json.dumps(routes)
    options = json.loads(wrong["env_bindings"][contracts.JUDGE_REQUEST_OPTIONS_BINDING])
    options["test/other"] = {"reasoning": {"effort": "low"}, "max_output_tokens": 8192}
    wrong["env_bindings"][contracts.JUDGE_REQUEST_OPTIONS_BINDING] = json.dumps(options)
    with pytest.raises(contracts.ArenaContractError, match="request options"):
        contracts.validate_scorer_policy(wrong)
    limited = registry()
    limited["approved_routes"][0]["max_output_tokens"] = 8192
    with pytest.raises(ValueError, match="options differ"):
        reasoning_capture.validate_judge_registry(limited, frozen)
    bind(monkeypatch, frozen)
    monkeypatch.delenv("ARENA_INDUSTRY_EVIDENCE_MODEL")
    with pytest.raises(ValueError, match="request_options"):
        openrouter_options.prepare_judge_request(industry())
    monkeypatch.setenv("ARENA_INDUSTRY_EVIDENCE_MODEL", "test/wrong-role")
    with pytest.raises(ValueError, match="request_options"):
        openrouter_options.prepare_judge_request(industry())
