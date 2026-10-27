"""Exact full-generated-thinking admission and gateway controls."""

import json
import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from lab_arena import broker as br, contracts, operations, reasoning_capture as rc
from scripts import probe_full_thinking_routes as probe_script
from tests.lab_arena import test_lab_arena_broker as broker_tests


EVIDENCE_DIR = Path(__file__).resolve().parents[2] / "docs/evidence/full-thinking"
PROBE_FILES = {
    "openai/gpt-oss-120b": "openai--gpt-oss-120b-chat-probe-price-bound.json",
    "mistralai/mistral-large-4-0": "mistralai--mistral-large-4-0-chat-probe-price-bound.json",
    "deepseek/deepseek-v4.1-flash-1017": "deepseek--deepseek-v4.1-flash-1017-chat-probe-price-bound.json",
}
LIVE_CAPTURE_FILES = {
    "openai/gpt-oss-20b": "openai--gpt-oss-20b-live-capture.json",
    "reflection/beam-1.0-501b-a23b": "reflection--beam-1.0-501b-a23b-live-capture.json",
    "google/gemma-4-31b-it": "google--gemma-4-31b-it-live-capture.json",
}


def chat(model, **extra):
    return {"model": model, "messages": [{"role": "user", "content": "Find evidence."}],
            "max_tokens": 1500, **extra}


def _near_size_cap(margin):
    parameters = {"model": "openai/gpt-4o-mini", "max_tokens": 200, "messages": []}
    def size():
        return len(contracts.canonical_json(parameters).encode("utf-8"))
    while size() + 31_200 <= 1_000_000 - margin:
        parameters["messages"].append({"role": "user", "content": "x" * 31_000})
    parameters["messages"].append({"role": "user", "content": ""})
    parameters["messages"][-1]["content"] = "x" * (1_000_000 - margin - size())
    return parameters


def _reviewed_registry(model="openai/gpt-4o-mini", api="openrouter.chat"):
    # A synthetic unit-test approval, never added to the committed registry.
    route = {
        "model_id": model, "model_version": model,
        "provider": "example/verified", "provider_name": "Example Provider", "api": api,
        "mode": {"budget_tokens": 1024} if api == "openrouter.chat" else {"effort": "high"},
        "parser_version": rc.PARSER_VERSION,
        "semantic_class": "full_generated_thinking",
        "full_text_fields": ["reasoning_details.text"] if api == "openrouter.chat" else ["output.content.reasoning_text"],
        "max_output_tokens": 8192,
        "answer_headroom_tokens": 1024, "context_tokens": 32768,
        "price_ceiling": {"prompt": "0.0000003", "completion": "0.000001",
                          "request": "0", "internal_reasoning": "0"},
        "evidence": {"documentation_url": "https://example.com/exact-route",
                     "probe_id": "probe-unit-001", "verified_at": "2026-10-23T00:00:00Z",
                     "probe_manifest_sha256": "a" * 64},
        "allowed_fallback_routes": [], "status": "approved",
    }
    return {"schema_version": 1, "approved_routes": [route],
            "baseline_route": {"model_id": model, "api": api}}


def test_committed_registry_approves_only_reviewed_routes():
    registry = rc.load_registry()
    assert {route["model_id"] for route in registry["approved_routes"]} == set(PROBE_FILES) | set(LIVE_CAPTURE_FILES)
    assert registry["baseline_route"] == {
        "model_id": "deepseek/deepseek-v4.1-flash-1017", "api": "openrouter.responses"
    }
    with pytest.raises(ValueError, match="reasoning_route_unsupported"):
        rc.apply("openrouter.chat", chat("openai/gpt-4o-mini"), policy=rc.POLICY,
                 registry=registry)


@pytest.mark.parametrize("model_id", LIVE_CAPTURE_FILES)
def test_new_route_evidence_preserves_real_tool_history_and_final_schema(model_id):
    """Replay committed live evidence; neither metadata nor a model family grants admission."""
    from scripts.verify_arena_live_capture import _valid_final, assistant_history
    from tests.lab_arena.full_thinking_saved_probe_test import historical_registry
    registry = historical_registry()
    route = next(r for r in registry["approved_routes"] if r["model_id"] == model_id)
    raw = (EVIDENCE_DIR / LIVE_CAPTURE_FILES[model_id]).read_bytes()
    proof = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == route["evidence"]["probe_manifest_sha256"]
    assert proof["model"] == model_id
    assert proof["provider"] == route["provider"]
    assert proof["semantic_source"] == route["evidence"]["documentation_url"]
    assert all(proof["checks"].values())
    assert {r["scenario"] for r in proof["runs"]} == {"live", "complex"}
    for run in proof["runs"]:
        assert len(run["calls"]) == 3
        assert any(run["thinking_chars"])
        expected = {"company": "Acme Test", "qualified": run["scenario"] == "live",
                    "evidence_ids": ["profile", "headcount"]}
        assert _valid_final(run["validated_final_json"], expected)
        table = broker_tests.price_table()
        table["models"][model_id] = {component: route["price_ceiling"].get(component, "0")
                                     for component in br.PRICED_COMPONENTS}
        transport = broker_tests.FakeTransport([(200, c["response"]) for c in run["calls"]])
        store = broker_tests.FakeLedgerStore()
        broker = br.Broker(store=store, key_for=lambda p: broker_tests.HOST_KEYS[p],
                           price_table=br.validate_price_table(table), transport=transport,
                           clock=lambda: datetime(2026, 9, 2, 1, 0, tzinfo=timezone.utc))
        context = replace(broker_tests.CONTEXT, reasoning_capture_policy=rc.POLICY,
                          reasoning_route_registry=registry)
        previous_message = None
        for index, call in enumerate(run["calls"]):
            request, response = call["request"], call["response"]
            assert contracts.document_hash(response) == call["response_sha256"]
            parameters = {k: v for k, v in request.items() if k not in {"provider", "stream"}}
            result = broker.execute(context, operation_id="openrouter.chat", parameters=parameters,
                                    action_sequence=index, timeout_ms=30000)
            assert result.status == 200 and json.loads(result.body) == response
            sent = json.loads(transport.sent[-1]["body"])
            assert sent == request
            assert sent["provider"]["zdr"] is True
            assert sent["provider"]["only"] == [route["provider"]]
            assert sent["provider"]["allow_fallbacks"] is False
            if previous_message is not None:
                assert previous_message in sent["messages"]
                tool_id = previous_message["tool_calls"][0]["id"]
                assert sent["messages"][-1]["tool_call_id"] == tool_id
            observation = br.broker_observations(store.calls[result.call["call_identity"]]["terminal"])["reasoning_route_observation"]
            assert observation["status"] == "matched"
            previous_message = assistant_history(response["choices"][0]["message"])


@pytest.mark.parametrize("model_id", PROBE_FILES)
def test_committed_price_bound_probe_replays_through_broker(model_id):
    from tests.lab_arena.full_thinking_saved_probe_test import historical_registry
    registry = historical_registry()
    route = next(route for route in registry["approved_routes"] if route["model_id"] == model_id)
    evidence_path = EVIDENCE_DIR / PROBE_FILES[model_id]
    raw = evidence_path.read_bytes()
    probe = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == route["evidence"]["probe_manifest_sha256"]
    assert probe["probe_id"] == route["evidence"]["probe_id"]
    assert probe["model"] == model_id
    assert probe["provider"] == route["provider"]
    assert probe["api"] == route["api"] == "openrouter.chat"
    assert probe["zdr"] is True
    assert probe["approval_granted"] is False  # The executable cannot approve its own result.
    assert all(probe["checks"].values())
    assert len(probe["attempts"]) == 3

    # A deliberately cheaper catalog entry proves admission and reservation
    # use the reviewed endpoint ceiling, not the model-wide catalog floor.
    table = broker_tests.price_table()
    table["models"][model_id] = {
        component: str(Decimal(route["price_ceiling"].get(component, "0")) / 2)
        for component in br.PRICED_COMPONENTS
    }
    table = br.validate_price_table(table)
    transport = broker_tests.FakeTransport(
        [(attempt["status"], attempt["response"]) for attempt in probe["attempts"]]
    )
    store = broker_tests.FakeLedgerStore()
    broker = br.Broker(
        store=store, key_for=lambda provider: broker_tests.HOST_KEYS[provider],
        price_table=table, transport=transport,
        clock=lambda: datetime(2026, 9, 2, 1, 0, tzinfo=timezone.utc),
    )
    context = replace(broker_tests.CONTEXT, reasoning_capture_policy=rc.POLICY,
                      reasoning_route_registry=registry)
    for index, attempt in enumerate(probe["attempts"]):
        live_request = attempt["request"]
        live_response = attempt["response"]
        assert hashlib.sha256(probe_script.canonical(live_request)).hexdigest() == attempt["request_sha256"]
        assert hashlib.sha256(probe_script.canonical(live_response)).hexdigest() == attempt["response_sha256"]
        if model_id.startswith("deepseek/") and index == 2:
            # The probe escapes Unicode; Arena's canonical ledger JSON does
            # not. Each hash must use its own documented wire convention.
            assert any(ord(char) > 127 for char in contracts.canonical_json(live_response))
            assert attempt["response_sha256"] != contracts.document_hash(live_response).split(":", 1)[1]
        # The agent supplies semantic request fields. The broker supplies the
        # provider pin and transport stream flag on the outbound wire.
        parameters = {key: value for key, value in live_request.items()
                      if key not in {"provider", "stream"}}
        result = broker.execute(context, operation_id="openrouter.chat",
                                parameters=parameters, action_sequence=index,
                                timeout_ms=30000)
        assert result.status == 200
        assert result.call["outcome"] == "settled"
        assert len(transport.sent) == index + 1
        sent = json.loads(transport.sent[-1]["body"])
        assert sent["model"] == model_id
        assert sent["messages"] == live_request["messages"]
        assert sent["reasoning"] == live_request["reasoning"]
        assert sent["max_tokens"] == live_request["max_tokens"]
        assert sent["stream"] is False
        for field in ("tools", "tool_choice", "response_format"):
            assert sent.get(field) == live_request.get(field)
        assert sent["provider"] == live_request["provider"]
        assert sent["provider"]["only"] == [route["provider"]]
        assert sent["provider"]["zdr"] is True
        assert sent["provider"]["allow_fallbacks"] is False
        assert json.loads(result.body) == live_response
        assert contracts.document_hash(json.loads(result.body)) == contracts.document_hash(live_response)

        normalized = br.normalized_request(
            "openrouter.chat", parameters, reasoning_capture_policy=rc.POLICY,
            reasoning_route_registry=registry)
        assert result.call["request_hash"] == contracts.document_hash(normalized)
        ledger = store.calls[result.call["call_identity"]]
        pricing = {**table["models"][model_id], **route["price_ceiling"]}
        assert ledger["amount"] == br._max_openrouter_cost_for_pricing(
            pricing, normalized, max_output_tokens=normalized["max_tokens"])
        annotation = ledger["call_doc"]["reasoning_capture"]
        assert annotation["probe_manifest_sha256"] == route["evidence"]["probe_manifest_sha256"]
        observation = br.broker_observations(ledger["terminal"])["reasoning_route_observation"]
        assert observation["status"] == "matched"
        assert observation["response_model"] == model_id
        assert observation["response_provider_name"] == route["provider_name"]
        assert observation["selected_endpoint_model"] == route["model_version"]


def test_exact_route_registry_is_immutable_and_rejects_fallback_or_summary():
    source = _reviewed_registry()
    frozen = rc.validate_registry(source)
    source["approved_routes"][0]["mode"]["budget_tokens"] = 2048
    assert frozen["approved_routes"][0]["mode"] == {"budget_tokens": 1024}
    for change in ({"allowed_fallback_routes": ["other"]},
                   {"semantic_class": "summary"}, {"status": "candidate"},
                   {"price_ceiling": {"prompt": "NaN", "completion": "0.000001", "request": "0", "internal_reasoning": "0"}},
                   {"evidence": {**frozen["approved_routes"][0]["evidence"], "probe_id": ""}}):
        bad = json.loads(json.dumps(frozen))
        bad["approved_routes"][0].update(change)
        with pytest.raises(ValueError):
            rc.validate_registry(bad)
    duplicate = json.loads(json.dumps(frozen))
    duplicate["approved_routes"].append(duplicate["approved_routes"][0])
    with pytest.raises(ValueError, match="duplicate"):
        rc.validate_registry(duplicate)
    wrong_version = json.loads(json.dumps(frozen))
    wrong_version["approved_routes"][0]["evidence"]["verified_at"] = "2026-99-23T00:00:00Z"
    with pytest.raises(ValueError, match="timestamp"):
        rc.validate_registry(wrong_version)


def test_full_policy_forces_exact_mode_and_bounded_answer_room_before_hash():
    request = chat("openai/gpt-4o-mini", max_tokens=400, reasoning={"enabled": False, "summary": "auto"})
    updated, annotation = rc.apply("openrouter.chat", request, policy=rc.POLICY,
                                   registry=_reviewed_registry())
    assert request["reasoning"]["enabled"] is False
    assert updated["reasoning"] == {"max_tokens": 1024, "exclude": False}
    assert updated["max_tokens"] >= 2048
    assert annotation["reviewed_route"] is True
    assert annotation["semantic_class"] == "full_generated_thinking"
    assert annotation["full_text_fields"] == ["reasoning_details.text"]
    assert annotation["api"] == "openrouter.chat"
    assert annotation["provider_route"] == "example/verified"
    assert len(annotation["registry_sha256"]) == 64
    assert br.agent_reasoning_parameters("openrouter.chat", request, policy=rc.POLICY,
                                          registry=_reviewed_registry())[0] == updated


def test_full_policy_refuses_unapproved_model_api_and_invalid_schema():
    registry = _reviewed_registry()
    with pytest.raises(br.BrokerError, match="invalid_request"):
        br.agent_reasoning_parameters("openrouter.chat", chat("vendor/other"),
                                      policy=rc.POLICY, registry=registry)
    with pytest.raises(br.BrokerError, match="invalid_request"):
        br.agent_reasoning_parameters("openrouter.responses", {"model": "openai/gpt-4o-mini", "input": "hello"},
                                      policy=rc.POLICY, registry=registry)
    nearly_full = _near_size_cap(10)
    with pytest.raises(br.BrokerError, match="invalid_request"):
        br.agent_reasoning_parameters("openrouter.chat", nearly_full,
                                      policy=rc.POLICY, registry=registry)


def test_full_policy_can_add_output_room_above_historical_chat_limit():
    request = chat("openai/gpt-4o-mini", max_tokens=8192)
    updated, _ = br.agent_reasoning_parameters(
        "openrouter.chat", request, policy=rc.POLICY, registry=_reviewed_registry())
    assert updated["max_tokens"] == 8192
    with pytest.raises(operations.OperationRequestError):
        operations.validate_operation_request("openrouter.chat", updated)
    assert br.normalized_request("openrouter.chat", request,
        reasoning_capture_policy=rc.POLICY, reasoning_route_registry=_reviewed_registry())["max_tokens"] == 8192


def test_context_limit_refuses_request_before_dispatch():
    registry = _reviewed_registry()
    registry["approved_routes"][0]["context_tokens"] = 8192
    with pytest.raises(br.BrokerError, match="invalid_request"):
        br.agent_reasoning_parameters("openrouter.chat", chat("openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "x" * 31000}]),
            policy=rc.POLICY, registry=registry)


def test_hosted_search_without_call_bound_is_not_admitted_by_full_responses_route():
    registry = _reviewed_registry(api="openrouter.responses")
    request = {"model": "openai/gpt-4o-mini", "input": "Find evidence.",
               "tools": [{"type": "openrouter:web_search", "parameters": {
                   "engine": "native", "max_uses": 1, "max_total_results": 1}}]}
    with pytest.raises(br.BrokerError, match="invalid_request"):
        br.agent_reasoning_parameters("openrouter.responses", request,
                                      policy=rc.POLICY, registry=registry)


def test_actual_route_observation_distinguishes_match_mismatch_and_absence():
    _, annotation = rc.apply("openrouter.chat", chat("openai/gpt-4o-mini"),
                             policy=rc.POLICY, registry=_reviewed_registry())
    response = {"model": "openai/gpt-4o-mini", "provider": "Example Provider",
                "openrouter_metadata": {"endpoints": {"available": [
                    {"selected": True, "provider": "Example Provider", "model": "openai/gpt-4o-mini"}]}}}
    assert rc.route_observation(response, annotation)["status"] == "matched"
    response["openrouter_metadata"]["endpoints"]["available"][0]["model"] = "vendor/other"
    assert rc.route_observation(response, annotation)["status"] == "mismatch"
    assert rc.route_observation({}, annotation)["status"] == "unverified"


OK = (200, {"id": "gen", "model": "openai/gpt-4o-mini", "choices": [],
            "usage": {"prompt_tokens": 20, "completion_tokens": 50, "cost": "0.000033"}})


def _execute(context, *, response=OK):
    broker, store, transport = broker_tests.make_broker(
        transport=broker_tests.FakeTransport([response]),
        judge_models=(broker_tests.CHAT["model"],))
    result = broker.execute(context, operation_id="openrouter.chat",
                            parameters=broker_tests.CHAT, action_sequence=0, timeout_ms=30000)
    return result, store, transport


def test_full_policy_binds_frozen_route_hash_cost_and_zdr_without_fallback():
    registry = _reviewed_registry()
    context = replace(broker_tests.CONTEXT, reasoning_capture_policy=rc.POLICY,
                      reasoning_route_registry=registry)
    result, store, transport = _execute(context)
    assert result.status == 200
    sent = json.loads(transport.sent[0]["body"])
    assert sent["model"] == broker_tests.CHAT["model"]
    assert sent["max_tokens"] == 8192
    assert sent["reasoning"] == {"max_tokens": 1024, "exclude": False}
    assert sent["provider"]["only"] == ["example/verified"]
    assert sent["provider"]["allow_fallbacks"] is False
    assert sent["provider"]["require_parameters"] is True
    assert sent["provider"]["zdr"] is True
    assert sent["provider"]["max_price"] == {"prompt": 0.3, "completion": 1, "request": 0}
    expected = br.normalized_request("openrouter.chat", broker_tests.CHAT,
        reasoning_capture_policy=rc.POLICY, reasoning_route_registry=registry)
    assert result.call["request_hash"] == contracts.document_hash(expected)
    assert store.calls[result.call["call_identity"]]["call_doc"]["reasoning_capture"]["probe_id"] == "probe-unit-001"
    priced = dict(broker_tests.price_table()["models"][broker_tests.CHAT["model"]],
                  **registry["approved_routes"][0]["price_ceiling"])
    assert store.calls[result.call["call_identity"]]["amount"] == br._max_openrouter_cost_for_pricing(
        priced, expected, max_output_tokens=expected["max_tokens"])
    assert br.broker_observations(store.calls[result.call["call_identity"]]["terminal"])["reasoning_route_observation"]["status"] == "unverified"


def test_exact_full_route_requires_provider_parameter_enforcement():
    route = {
        "data_collection": "deny", "zdr": True,
        "only": ["example/verified"], "allow_fallbacks": False,
        "require_parameters": True,
        "max_price": {"prompt": 0.3, "completion": 1, "request": 0},
    }
    request = chat("openai/gpt-4o-mini")
    outbound = operations.build_outbound_request(
        "openrouter.chat", request, openrouter_provider_policy=route,
        allow_full_thinking=True)
    assert json.loads(outbound.body)["provider"] == route
    for unsafe in ({key: value for key, value in route.items() if key != "require_parameters"},
                   {**route, "require_parameters": False}):
        with pytest.raises(operations.OperationRequestError):
            operations.build_outbound_request(
                "openrouter.chat", request, openrouter_provider_policy=unsafe,
                allow_full_thinking=True)
    with pytest.raises(operations.OperationRequestError):
        operations.build_outbound_request(
            "openrouter.chat", request, openrouter_provider_policy=route)
    with pytest.raises(operations.OperationRequestError):
        operations.build_outbound_request(
            "openrouter.responses",
            {"model": "openai/gpt-4o-mini", "input": "Find evidence."},
            openrouter_provider_policy={key: value for key, value in route.items()
                                        if key not in {"only", "require_parameters"}},
            allow_full_thinking=True)


def test_unapproved_full_policy_call_does_not_dispatch_or_reserve():
    context = replace(broker_tests.CONTEXT, reasoning_capture_policy=rc.POLICY,
                      reasoning_route_registry=rc.load_registry())
    result, store, transport = _execute(context)
    assert result.status != 200
    assert not transport.sent and not store.calls


def test_broker_settlement_records_actual_selected_route_evidence():
    registry = _reviewed_registry()
    registry["approved_routes"][0]["model_version"] = "openai/gpt-4o-mini-20261023"
    body = dict(OK[1], provider="Example Provider",
                openrouter_metadata={"endpoints": {"available": [
                    {"selected": True, "provider": "Example Provider",
                     "model": "openai/gpt-4o-mini-20261023"}]}})
    context = replace(broker_tests.CONTEXT, reasoning_capture_policy=rc.POLICY,
                      reasoning_route_registry=registry)
    result, store, _ = _execute(context, response=(200, body))
    assert result.status == 200
    terminal = store.calls[result.call["call_identity"]]["terminal"]
    assert br.broker_observations(terminal)["reasoning_route_observation"]["status"] == "matched"
    assert br.broker_observations(terminal)["reasoning_route_observation"]["selected_endpoint_model"] == "openai/gpt-4o-mini-20261023"
