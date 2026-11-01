"""Capture keeps caller reasoning settings and output limits on the wire."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from lab_arena import broker as br, contracts, operations, reasoning_capture as rc
from lab_arena import training_capture, training_thinking
from tests.lab_arena import test_lab_arena_broker as bt
from tests.lab_arena.reasoning_capture_test import _reviewed_registry
from tests.lab_arena.training_export_test import Objects


def request(api, cap=400, **controls):
    return {"model": bt.CHAT["model"], **({"input": "Keep this prompt.", "max_output_tokens": cap}
        if api.endswith("responses") else {"messages": [{"role": "user", "content": "Keep this prompt."}],
                                            "max_tokens": cap}), **controls}


@pytest.mark.parametrize("api", ["openrouter.chat", "openrouter.responses"])
@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high"])
@pytest.mark.parametrize("cap", [400, 12000])
def test_explicit_settings_survive_policy_and_normalized_request(api, effort, cap):
    params = request(api, cap, reasoning={"effort": effort, "summary": "detailed"})
    original = deepcopy(params)
    registry = _reviewed_registry(api=api)
    updated, annotation = br.agent_reasoning_parameters(api, params, policy=rc.POLICY, registry=registry)
    assert params == original == updated
    assert annotation["changes"] == []
    # These fields describe the saved route probe. The outbound body records
    # actual caller settings; a different effort is not a new probe result.
    assert annotation["mode"] == registry["approved_routes"][0]["mode"]
    assert annotation["max_output_tokens"] == 8192
    assert br.normalized_request(api, params, reasoning_capture_policy=rc.POLICY,
                                 reasoning_route_registry=registry) == params


@pytest.mark.parametrize("controls", [
    {"reasoning_effort": "low", "include_reasoning": True},
    {"reasoning": {"enabled": True, "exclude": False}},
    {"reasoning": {"max_tokens": 200, "exclude": False}},
    {"reasoning": {"effort": "medium"}, "reasoning_effort": "medium"},
])
def test_chat_aliases_and_enabled_budget_are_preserved(controls):
    params = request("openrouter.chat", **controls)
    assert rc.apply("openrouter.chat", params, policy=rc.POLICY,
                    registry=_reviewed_registry())[0] == params


@pytest.mark.parametrize("controls", [
    {"reasoning": {"enabled": False}}, {"reasoning": {"exclude": True}},
    {"reasoning": {"effort": "none"}}, {"reasoning": {"max_tokens": 0}},
    {"reasoning": {"max_tokens": True}}, {"reasoning_effort": "none"},
    {"include_reasoning": False},
    {"reasoning": {"effort": "low"}, "reasoning_effort": "high"},
])
def test_disabled_or_ambiguous_controls_refuse_before_reservation(controls):
    broker, store, transport = bt.make_broker()
    ctx = replace(bt.CONTEXT, reasoning_capture_policy=rc.POLICY,
                  reasoning_route_registry=_reviewed_registry())
    result = broker.execute(ctx, operation_id="openrouter.chat", parameters=request("openrouter.chat", **controls),
                            action_sequence=0, timeout_ms=1000)
    assert result.status == 400 and not store.log and not transport.sent


@pytest.mark.parametrize("reasoning", [{"effort": "none"}, {"effort": ""}, {"effort": "low", "exclude": True}])
def test_responses_disabled_or_invalid_controls_refuse_before_reservation(reasoning):
    broker, store, transport = bt.make_broker()
    ctx = replace(bt.CONTEXT, reasoning_capture_policy=rc.POLICY,
                  reasoning_route_registry=_reviewed_registry(api="openrouter.responses"))
    result = broker.execute(ctx, operation_id="openrouter.responses",
                            parameters=request("openrouter.responses", reasoning=reasoning),
                            action_sequence=0, timeout_ms=1000)
    assert result.status == 400 and not store.log and not transport.sent


@pytest.mark.parametrize("api", ["openrouter.chat", "openrouter.responses"])
def test_absent_controls_use_named_defaults_without_inflating_output_limit(api):
    params = request(api)
    registry = _reviewed_registry(api=api)
    updated, annotation = br.agent_reasoning_parameters(api, params, policy=rc.POLICY, registry=registry)
    assert updated == {**params, "reasoning": ({"effort": "high"} if api.endswith("responses")
                                               else {"effort": "high", "exclude": False})}
    assert annotation["changes"] == ["reasoning_default"]
    without_cap = {k: v for k, v in params.items() if k not in {"max_tokens", "max_output_tokens"}}
    normalized = br.normalized_request(api, without_cap, reasoning_capture_policy=rc.POLICY,
                                       reasoning_route_registry=registry)
    assert normalized["max_output_tokens" if api.endswith("responses") else "max_tokens"] == operations.OPENROUTER_MAX_OUTPUT_TOKENS


@pytest.mark.parametrize("explicit", [False, True])
def test_reasoning_budget_cannot_consume_or_exceed_caller_output_limit(explicit):
    registry = _reviewed_registry()
    registry["approved_routes"][0]["mode"] = {"budget_tokens": 1024}
    params = request("openrouter.chat", **({"reasoning": {"max_tokens": 1024}} if explicit else {}))
    with pytest.raises(br.BrokerError, match="invalid_request"):
        br.agent_reasoning_parameters("openrouter.chat", params, policy=rc.POLICY, registry=registry)
    params["max_tokens"] = 2048
    updated, _ = br.agent_reasoning_parameters("openrouter.chat", params, policy=rc.POLICY, registry=registry)
    assert updated["max_tokens"] == 2048 and updated["reasoning"]["max_tokens"] == 1024


@pytest.mark.parametrize("api", ["openrouter.chat", "openrouter.responses"])
def test_unchanged_controls_still_enforce_schema_and_context(api):
    params = request(api, 32768, reasoning={"effort": "low"})
    with pytest.raises(br.BrokerError, match="invalid_request"):
        br.agent_reasoning_parameters(api, params, policy=rc.POLICY, registry=_reviewed_registry(api=api))
    registry = _reviewed_registry(api=api)
    registry["approved_routes"][0]["context_tokens"] = 100000
    with pytest.raises(br.BrokerError, match="invalid_request"):
        br.agent_reasoning_parameters(api, request(api, 32769, reasoning={"effort": "low"}),
                                      policy=rc.POLICY, registry=registry)
    params = request(api, reasoning={"effort": "low"})
    params["not_a_parameter"] = True
    with pytest.raises(br.BrokerError, match="invalid_request"):
        br.agent_reasoning_parameters(api, params, policy=rc.POLICY, registry=_reviewed_registry(api=api))


@pytest.mark.parametrize("api", ["openrouter.chat", "openrouter.responses"])
@pytest.mark.parametrize("cap", [400, 12000])
def test_broker_capture_prices_and_replay_bind_preserved_request(api, cap):
    registry = _reviewed_registry(api=api)
    objects = Objects()
    transport = bt.FakeTransport([(200, {"usage": {"cost": "0.00001"}})])
    broker, store, _ = bt.make_broker(transport=transport, capture_request=training_capture.RequestCapture(objects).prepare)
    ctx = replace(bt.CONTEXT, reasoning_capture_policy=rc.POLICY, reasoning_route_registry=registry)
    params = request(api, cap, reasoning={"effort": "low"})
    args = dict(operation_id=api, parameters=params, action_sequence=0, timeout_ms=1000)
    result = broker.execute(ctx, **args)
    assert result.status == 200
    wire = json.loads(transport.sent[0]["body"])
    assert all(wire[k] == v for k, v in params.items())
    assert wire["provider"]["only"] == ["example/verified"]
    assert wire["provider"]["zdr"] is True and wire["provider"]["data_collection"] == "deny"
    assert wire["provider"]["allow_fallbacks"] is False and wire["provider"]["require_parameters"] is True
    saved = store.calls[result.call["call_identity"]]
    captured = training_capture.read_request(objects, saved["call_doc"]["request_capture"])
    assert captured["outbound"]["body"] == wire
    assert captured["hashed_parameters"] == params
    assert result.call["request_hash"] == contracts.document_hash(params)
    priced = dict(bt.price_table()["models"][params["model"]], **registry["approved_routes"][0]["price_ceiling"])
    assert saved["amount"] == br._max_openrouter_cost_for_pricing(priced, params, max_output_tokens=cap)
    call = {"provider": "openrouter", "operation_id": api, "request": captured,
            "reasoning_capture": captured["reasoning_capture"]}
    assert training_thinking._policy(call, registry)
    assert captured["reasoning_capture"]["mode"] != wire["reasoning"]
    replay = broker.execute(ctx, **args)
    assert replay.body == result.body and len(transport.sent) == 1


def test_policy_off_request_and_annotation_stay_unchanged():
    params = request("openrouter.chat", reasoning={"enabled": False})
    assert rc.apply("openrouter.chat", params, policy=None) == (params, None)


@pytest.mark.parametrize("api", ["openrouter.chat", "openrouter.responses"])
@pytest.mark.parametrize("control", [{}, {"summary": "auto"},
    {"summary": "detailed", "context": "current_turn"}, {"context": "all_turns"}])
def test_explicit_summary_or_context_does_not_authorize_effort_default(api, control):
    params = request(api, reasoning=control)
    before = deepcopy(params)
    with pytest.raises(br.BrokerError, match="invalid_request"):
        br.agent_reasoning_parameters(api, params, policy=rc.POLICY, registry=_reviewed_registry(api=api))
    assert params == before
    assert rc.apply(api, params, policy=None) == (params, None)
    broker, store, transport = bt.make_broker()
    ctx = replace(bt.CONTEXT, reasoning_capture_policy=rc.POLICY,
                  reasoning_route_registry=_reviewed_registry(api=api))
    result = broker.execute(ctx, operation_id=api, parameters=params, action_sequence=0, timeout_ms=1000)
    assert result.status == 400 and not store.log and not transport.sent


def test_summary_with_explicit_enabled_chat_alias_stays_unchanged():
    params = request("openrouter.chat", reasoning={"summary": "auto"}, include_reasoning=True)
    assert br.agent_reasoning_parameters("openrouter.chat", params, policy=rc.POLICY,
                                         registry=_reviewed_registry())[0] == params


def _bind_preserved_controls(call, cap=400, effort="low"):
    body = call["request"]["outbound"]["body"]
    field = "max_tokens" if call["operation_id"].endswith("chat") else "max_output_tokens"
    body[field] = cap
    body["reasoning"] = {"effort": effort}
    hashed = {k: deepcopy(v) for k, v in body.items() if k not in {"provider", "stream", "store"}}
    call["request"]["hashed_parameters"] = hashed
    call["request"]["effective_parameters"] = deepcopy(hashed)
    call["request"]["admitted_parameters"] = deepcopy(hashed)
    call["request_hash"] = call["request"]["request_hash"] = contracts.document_hash(hashed)
    call["request"]["outbound"]["body_sha256"] = contracts.document_hash(body)


@pytest.mark.parametrize("api", ["openrouter.chat", "openrouter.responses"])
def test_actual_preserved_cap_keeps_existing_truncation_treatment(api, monkeypatch):
    from tests.lab_arena.training_cap_eof_test import cap_example
    from tests.lab_arena.training_eligibility_test import refresh_response_links

    example = cap_example(api)
    call = example["provider_calls"][0]
    _bind_preserved_controls(call)
    call["response"]["usage"]["completion_tokens" if api.endswith("chat") else "output_tokens"] = 400
    refresh_response_links(example, 0)
    before = deepcopy(example)
    assert training_thinking.verified_cap_truncated_target(example, call)
    assert example == before
    # The former tested-cap-only guard rejects this same valid cap-hit proof.
    monkeypatch.setattr(training_thinking, "_captured_reasoning_controls_match", lambda *_: False)
    assert not training_thinking.verified_cap_truncated_target(example, call)


def test_preserved_effort_and_cap_keep_recovered_free_retry_proof(monkeypatch):
    from tests.lab_arena.training_hidden_retry_test import recovered_example

    example = recovered_example()
    failed, recovered = example["provider_calls"][:2]
    for call in (failed, recovered):
        _bind_preserved_controls(call)
    before = deepcopy(example)
    assert training_thinking.verified_recovered_provider_retry(example, failed)
    assert example == before
    monkeypatch.setattr(training_thinking, "_captured_reasoning_controls_match", lambda *_: False)
    assert not training_thinking.verified_recovered_provider_retry(example, failed)


@pytest.mark.parametrize("damage", ["outbound_cap", "outbound_effort", "request_hash", "disabled", "excluded", "oversized_cap"])
def test_preserved_controls_do_not_relax_failure_proof_integrity(damage):
    from tests.lab_arena.training_hidden_retry_test import recovered_example

    example = recovered_example()
    failed, recovered = example["provider_calls"][:2]
    for call in (failed, recovered):
        _bind_preserved_controls(call, cap=32769 if damage == "oversized_cap" else 400,
                                 effort="none" if damage == "disabled" else "low")
        body = call["request"]["outbound"]["body"]
        if damage == "outbound_cap": body["max_output_tokens"] += 1
        elif damage == "outbound_effort": body["reasoning"]["effort"] = "medium"
        elif damage == "request_hash": call["request"]["hashed_parameters"]["reasoning"]["effort"] = "medium"
        elif damage == "excluded":
            body["reasoning"]["exclude"] = True
            call["request"]["hashed_parameters"]["reasoning"]["exclude"] = True
            call["request_hash"] = call["request"]["request_hash"] = contracts.document_hash(call["request"]["hashed_parameters"])
        call["request"]["outbound"]["body_sha256"] = contracts.document_hash(body)
    assert not training_thinking.verified_recovered_provider_retry(example, failed)


@pytest.mark.parametrize("kind", ["cap", "retry"])
@pytest.mark.parametrize("alias", [{"include_reasoning": False}, {"reasoning_effort": "none"},
                                   {"reasoning_effort": "low"}])
def test_historical_tested_settings_do_not_override_disabling_or_conflicting_alias(kind, alias):
    from tests.lab_arena.training_cap_eof_test import cap_example
    from tests.lab_arena.training_hidden_retry_test import recovered_example

    example = cap_example() if kind == "cap" else recovered_example()
    calls = example["provider_calls"][:1 if kind == "cap" else 2]
    for call in calls:
        body = call["request"]["outbound"]["body"]
        body.update(alias)
        for field in ("hashed_parameters", "effective_parameters", "admitted_parameters"):
            call["request"][field].update(alias)
        call["request_hash"] = call["request"]["request_hash"] = contracts.document_hash(call["request"]["hashed_parameters"])
        call["request"]["outbound"]["body_sha256"] = contracts.document_hash(body)
    verify = (training_thinking.verified_cap_truncated_target if kind == "cap"
              else training_thinking.verified_recovered_provider_retry)
    assert not verify(example, calls[0])


@pytest.mark.parametrize("invalid", ["exclude", "enabled", "budget", "include_reasoning", "reasoning_effort"])
def test_responses_retry_proof_requires_the_actual_api_control_schema(invalid):
    from tests.lab_arena.training_hidden_retry_test import recovered_example

    example = recovered_example()
    failed, recovered = example["provider_calls"][:2]
    for call in (failed, recovered):
        _bind_preserved_controls(call, cap=call["reasoning_capture"]["max_output_tokens"],
                                 effort=call["reasoning_capture"]["mode"]["effort"])
        for field in ("hashed_parameters", "effective_parameters", "admitted_parameters"):
            call["request"][field].pop("tool_choice", None)
        call["request"]["outbound"]["body"].pop("tool_choice", None)
        call["request_hash"] = call["request"]["request_hash"] = contracts.document_hash(call["request"]["hashed_parameters"])
        call["request"]["outbound"]["body_sha256"] = contracts.document_hash(call["request"]["outbound"]["body"])
        operations.validate_operation_request("openrouter.responses", call["request"]["hashed_parameters"], allow_full_thinking=True)
    assert training_thinking.verified_recovered_provider_retry(example, failed)
    for call in (failed, recovered):
        for value in (call["request"]["outbound"]["body"], call["request"]["hashed_parameters"],
                      call["request"]["effective_parameters"], call["request"]["admitted_parameters"]):
            if invalid == "exclude": value["reasoning"]["exclude"] = False
            elif invalid == "enabled": value["reasoning"] = {"enabled": True}
            elif invalid == "budget": value["reasoning"] = {"max_tokens": 1024}
            elif invalid == "include_reasoning": value["include_reasoning"] = True
            else: value["reasoning_effort"] = value["reasoning"]["effort"]
        call["request_hash"] = call["request"]["request_hash"] = contracts.document_hash(call["request"]["hashed_parameters"])
        call["request"]["outbound"]["body_sha256"] = contracts.document_hash(call["request"]["outbound"]["body"])
        with pytest.raises(operations.OperationError):
            operations.validate_operation_request("openrouter.responses", call["request"]["hashed_parameters"], allow_full_thinking=True)
    assert not training_thinking.verified_recovered_provider_retry(example, failed)
