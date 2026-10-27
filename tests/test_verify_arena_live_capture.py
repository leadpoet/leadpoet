"""The live harness's no-network controls and durable readback boundary."""
import copy
from decimal import Decimal
import json
import os
from pathlib import Path

import pytest

from lab_arena import broker as br, shim
from scripts import verify_arena_live_capture as harness
from scripts.verify_arena_live_capture import (SharedBudget, WireTransport, run_probe,
                                                _parameters, conservative_bound, request_bound, route_registry,
                                                preflight_probe_controls)
from tests.lab_arena.test_lab_arena_broker import FakeTransport

EVIDENCE = Path(__file__).parents[1] / "docs/evidence/full-thinking"
PROBES = sorted(EVIDENCE.glob("*chat-probe-price-bound.json"))
KEY = "sk-or-v1-" + "q" * 40


@pytest.mark.parametrize("api", ["openrouter.chat", "openrouter.responses"])
@pytest.mark.parametrize("scenario", ["acme", "conflicting_evidence"])
@pytest.mark.parametrize("parallel", [None, False])
def test_text_probe_sends_schema_once_and_keeps_first_input_exact_through_continuations(tmp_path, api, scenario, parallel):
    _, route, _ = route_registry(None, "openai/gpt-oss-120b")
    route = copy.deepcopy(route)
    route["api"] = api
    if api == "openrouter.responses":
        route["full_text_fields"] = ["output.content.reasoning_text"]
    path = tmp_path / "route.json"
    path.write_text(json.dumps(route))
    expected = harness.scenario_spec(scenario)["expected"]
    replies = []
    for turn in range(3):
        call = {"id": "call-%d" % turn, "type": "function", "function": {
            "name": "lookup_evidence", "arguments": json.dumps({"evidence_id": ("profile", "headcount")[turn]})}} if turn < 2 else None
        if api == "openrouter.chat":
            output = {"choices": [{"finish_reason": "tool_calls" if call else "stop", "message": {
                "role": "assistant", "content": None if call else json.dumps(expected),
                **({"tool_calls": [call]} if call else {})}}]}
        else:
            output = {"status": "completed", "output": [{"type": "function_call", "id": call["id"],
                "call_id": call["id"], "name": call["function"]["name"],
                "arguments": call["function"]["arguments"], "status": "completed"}] if call else [{
                    "type": "message", "role": "assistant", "status": "completed", "content": [{
                        "type": "output_text", "text": json.dumps(expected)}]}]}
        replies.append((200, {"model": route["model_id"], "provider": route["provider_name"],
            "openrouter_metadata": {"endpoints": {"available": [{"selected": True,
                "model": route["model_version"], "provider": route["provider_name"]}]}},
            "usage": {"cost": "0.00001", "completion_tokens_details": {"reasoning_tokens": 0}}, **output}))
    transport = FakeTransport(replies)
    result = run_probe(model=route["model_id"], route_file=path, output=tmp_path / "probe", key=KEY,
                       transport=transport, ledger_mode="local_diagnostic", final_format="text",
                       tool_choice="auto", budget_bound_mode="request", scenario=scenario,
                       parallel_tool_calls=parallel)
    assert result["tool_continuations_and_final_json"]
    bodies = [json.loads(item["body"]) for item in transport.sent]
    history_field = "input" if api == "openrouter.responses" else "messages"
    first = bodies[0][history_field][0]
    marker = "Return only JSON conforming to this schema: "
    assert marker in first["content"]
    assert first["content"].count(marker) == 1
    assert json.loads(first["content"].split(marker)[1]) == harness.SCHEMA
    assert '"qualified": true' not in first["content"]
    assert '"qualified": false' not in first["content"]
    assert all(body[history_field][0] == first for body in bodies)
    assert all(body.get("parallel_tool_calls") is parallel for body in bodies[:2])
    assert "parallel_tool_calls" not in bodies[2]
    assert result["native_final_constraint"] == "unconstrained_text"
    assert marker not in harness.scenario_spec(scenario)["initial"][0]["content"]


@pytest.mark.parametrize("choice", ["auto", "required", "named"])
@pytest.mark.parametrize("final_format", ["json_object", "json_schema", "text"])
def test_full_probe_controls_use_normal_broker_path(tmp_path, choice, final_format):
    probe = json.loads(PROBES[-1].read_text())
    transport = FakeTransport([(a["status"], a["response"]) for a in probe["attempts"]])
    result = run_probe(model=probe["model"], output=tmp_path / "probe", key=KEY,
                       transport=transport, ledger_mode="local_diagnostic",
                       tool_choice=choice, final_format=final_format,
                       budget_bound_mode="request")
    assert result["probe_failure"] is None
    assert result["tool_continuations_and_final_json"]
    assert result["mode"] == "full"
    assert not result["diagnostic_tool_choice_auto"]
    sent = [json.loads(item["body"]) for item in transport.sent]
    assert sent[0]["tool_choice"] == ({"type": "function", "function": {
        "name": "lookup_evidence"}} if choice == "named" else choice)
    if final_format == "text":
        assert "response_format" not in sent[2]
        assert result["native_final_constraint"] == "unconstrained_text"
    else:
        assert sent[2]["response_format"]["type"] == final_format
    assert all(row["capture"]["route_reviewed"] for row in result["readback"])
    _, route, _ = route_registry(None, probe["model"])
    assert list(result["request_budget_bounds_usd"].values()) == [
        str(request_bound(route, item["body"])) for item in transport.sent]


@pytest.mark.parametrize("api", ["openrouter.chat", "openrouter.responses"])
@pytest.mark.parametrize("choice", ["auto", "required", "named"])
@pytest.mark.parametrize("final_format", ["json_object", "json_schema", "text"])
def test_probe_preflight_uses_actual_production_api_and_format_contract(api, choice, final_format):
    _, route, _ = route_registry(None, "openai/gpt-oss-120b")
    route = copy.deepcopy(route)
    route["api"] = api
    if api == "openrouter.responses":
        route["full_text_fields"] = ["output.content.reasoning_text"]
    arguments = dict(model=route["model_id"], route=route, tool_choice=choice, final_format=final_format)
    if api == "openrouter.responses" and final_format == "json_object":
        with pytest.raises(br.BrokerError, match="invalid_request"):
            preflight_probe_controls(**arguments)
    else:
        preflight_probe_controls(**arguments)
        parameters = _parameters(route["model_id"], harness.INITIAL, 2, api, "full", route,
                                 tool_choice=choice, final_format=final_format)
        if api == "openrouter.responses":
            assert parameters["text"]["format"]["type"] == final_format


def test_unsupported_responses_json_object_is_rejected_before_first_paid_turn(tmp_path):
    _, route, _ = route_registry(None, "openai/gpt-oss-120b")
    candidate = copy.deepcopy(route)
    candidate.update(api="openrouter.responses", full_text_fields=["output.content.reasoning_text"])
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate))
    transport = FakeTransport()
    with pytest.raises(br.BrokerError, match="invalid_request"):
        run_probe(model=candidate["model_id"], route_file=path, key=KEY,
                  output=tmp_path / "evidence", transport=transport,
                  ledger_mode="local_diagnostic", final_format="json_object")
    assert not transport.sent
    assert not (tmp_path / "evidence").exists()
    assert not (tmp_path / "arena-live-capture-budget.json").exists()


@pytest.mark.parametrize("api", ["openrouter.chat", "openrouter.responses"])
def test_unconstrained_text_still_requires_exact_final_json_values(api):
    bad = json.dumps({"company": "Acme Test", "qualified": False,
                      "evidence_ids": ["profile", "headcount"]})
    response = ({"choices": [{"message": {"role": "assistant", "content": bad}}]}
                if api == "openrouter.chat" else {"output": [{"type": "message", "content": [
                    {"type": "output_text", "text": bad}]}]})
    with pytest.raises(ValueError, match="strict_final_json_mismatch"):
        harness._next_message(json.dumps(response).encode(), 2, api, harness.scenario_spec("acme"))


def test_request_budget_uses_actual_utf8_input_and_full_output_cap():
    _, route, _ = route_registry(None, "openai/gpt-oss-120b")
    route = dict(route, context_tokens=1_048_576)
    document = {"model": route["model_id"], "max_tokens": route["max_output_tokens"],
                "messages": [{"role": "user", "content": "会社"}]}
    body = json.dumps(document, ensure_ascii=False).encode()
    price = route["price_ceiling"]
    expected = (Decimal(br.bounded_input_tokens(document)) * Decimal(price["prompt"])
                + Decimal(route["max_output_tokens"]) *
                (Decimal(price["completion"]) + Decimal(price["internal_reasoning"]))
                + Decimal(price["request"]))
    from decimal import ROUND_UP
    assert request_bound(route, body) == expected.quantize(Decimal("0.000001"), rounding=ROUND_UP)
    assert request_bound(route, body) < conservative_bound(route)
    longer = dict(document, messages=[{"role": "user", "content": "x" * 10_000}])
    assert request_bound(route, json.dumps(longer).encode()) > request_bound(route, body)


def test_exact_request_budget_refuses_before_http_and_retains_uncertain_hold(tmp_path):
    _, route, _ = route_registry(None, "openai/gpt-oss-120b")
    body = json.dumps({"model": route["model_id"], "max_tokens": route["max_output_tokens"]}).encode()
    amount = request_bound(route, body)
    inner = FakeTransport([(503, {"error": "unpriced"})])
    budget = SharedBudget(tmp_path / "budget.json", amount)
    wire = WireTransport(tmp_path, inner, request_budget=budget, budget_route=route)
    arguments = {"url": "https://openrouter.ai/api/v1/chat/completions", "body": body,
                 "method": "POST", "headers": {}, "timeout_seconds": 1}
    wire.arm(0)
    assert wire.send(**arguments).status == 503
    assert Decimal(json.loads(budget.path.read_text())["reserved_usd"]) == amount
    wire.arm(1)
    with pytest.raises(ValueError, match="cumulative_budget_exceeded"):
        wire.send(**arguments)
    assert len(inner.sent) == 1
    assert not wire.sent_in_turn


@pytest.mark.parametrize("path", PROBES, ids=lambda path: path.name)
def test_saved_provider_turns_use_real_broker_capture_and_reopened_export(path, tmp_path):
    probe = json.loads(path.read_text())
    transport = FakeTransport([(attempt["status"], attempt["response"])
                               for attempt in probe["attempts"]])
    result = run_probe(model=probe["model"], output=tmp_path / "evidence",
                       key=KEY, transport=transport, ledger_mode="local_diagnostic")
    assert result["probe_failure"] is None
    assert result["readback_failure"] is None
    assert result["tool_continuations_and_final_json"] is True
    assert result["classification"] == "full_thinking_observed"
    assert result["route_authority"] == "working_registry"
    assert result["registry_listed"] is True
    assert result["production_activation"] is False
    assert len(result["readback"]) == 3
    assert all(item["wire_receipt_exact"] and item["outbound_request_exact"]
               and item["stored_request_readable"] and item["stored_response_readable"]
               and item["capture"]["storage"] == "saved" for item in result["readback"])
    assert all(json.loads(sent["body"])["provider"]["allow_fallbacks"] is False
               for sent in transport.sent)
    assert KEY.encode() not in (tmp_path / "evidence/ledger.jsonl").read_bytes()
    assert KEY.encode() not in (tmp_path / "evidence/report.json").read_bytes()
    export = json.loads((tmp_path / "evidence/diagnostic-export.json").read_text())
    readable = json.loads((tmp_path / "evidence/readable-trace.json").read_text())
    assert export["completeness"]["complete"] is False
    assert len(export["provider_calls"]) == len(readable["calls"]) == 3
    assert readable["projection_truncated"] is False
    markdown = (tmp_path / "evidence/readable-trace.md").read_text()
    assert "diagnostic-export.json" in markdown
    assert "Acme Test" in markdown


def test_over_budget_candidate_refused_before_network_or_artifacts(tmp_path):
    registry, route, _ = route_registry(None, "mistralai/mistral-large-4-0")
    assert conservative_bound(route) * 3 > 1
    candidate = copy.deepcopy(route)
    candidate["model_id"] = "example/expensive-probe"
    candidate["model_version"] = "example/expensive-probe"
    candidate["price_ceiling"]["completion"] = "1"
    candidate.pop("evidence")
    route_file = tmp_path / "candidate.json"
    route_file.write_text(json.dumps(candidate))
    transport = FakeTransport()
    with pytest.raises(ValueError, match="worst_case_three_call_budget_exceeded"):
        run_probe(model=candidate["model_id"], route_file=route_file,
                  output=tmp_path / "evidence", key=KEY, transport=transport,
                  ledger_mode="local_diagnostic")
    assert not transport.sent
    assert not (tmp_path / "evidence").exists()


def test_conservative_bound_adds_completion_and_internal_reasoning_prices():
    _, route, _ = route_registry(None, "openai/gpt-oss-120b")
    candidate = copy.deepcopy(route)
    candidate["context_tokens"] = 8192
    candidate["max_output_tokens"] = 4096
    candidate["price_ceiling"] = {"prompt": "0", "completion": "0.000001",
                                  "internal_reasoning": "0.000002", "request": "0"}
    assert conservative_bound(candidate) == Decimal("0.012288")


def test_requested_output_cannot_exceed_reserved_route_ceiling():
    _, route, _ = route_registry(None, "openai/gpt-oss-120b")
    candidate = dict(route, max_output_tokens=64)
    chat = _parameters(candidate["model_id"], [], 0, "openrouter.chat", "full", candidate)
    responses = _parameters(candidate["model_id"], [], 0, "openrouter.responses", "full", candidate)
    assert chat["max_tokens"] == responses["max_output_tokens"] == 64


@pytest.mark.parametrize("api", ["openrouter.chat", "openrouter.responses"])
def test_full_probe_explicitly_requests_declared_effort_and_output_cap(api):
    _, route, _ = route_registry(None, "openai/gpt-oss-120b")
    route = dict(route, api=api, max_output_tokens=8192, mode={"effort": "high"})
    params = _parameters(route["model_id"], harness.INITIAL, 0, api, "full", route)
    assert params["max_output_tokens" if api.endswith("responses") else "max_tokens"] == 8192
    assert params["reasoning"] == ({"effort": "high"} if api.endswith("responses")
                                   else {"effort": "high", "exclude": False})


def test_full_chat_probe_translates_only_registry_budget_field():
    _, route, _ = route_registry(None, "openai/gpt-oss-120b")
    route = dict(route, mode={"budget_tokens": 1024})
    params = _parameters(route["model_id"], harness.INITIAL, 0, "openrouter.chat", "full", route)
    assert params["reasoning"] == {"max_tokens": 1024, "exclude": False}


def test_conflicting_evidence_scenario_requires_newer_negative_finding(tmp_path):
    probe = json.loads(PROBES[-1].read_text())
    replies = [(attempt["status"], copy.deepcopy(attempt["response"]))
               for attempt in probe["attempts"]]
    # Exact field/value ordering seen in the saved live Reflection final.
    replies[2][1]["choices"][0]["message"]["content"] = (
        '{"company":"Acme Test","evidence_ids":["headcount","profile"],"qualified":false}')
    transport = FakeTransport(replies)
    result = run_probe(model=probe["model"], output=tmp_path / "evidence", key=KEY,
                       transport=transport, ledger_mode="local_diagnostic",
                       scenario="conflicting_evidence")
    assert result["tool_continuations_and_final_json"] is True
    assert result["scenario"] == "conflicting_evidence"
    final_request = json.loads(transport.sent[2]["body"])
    assert any(message.get("role") == "tool" and '"employees":12' in message.get("content", "")
               for message in final_request["messages"])
    assert all("parallel_tool_calls" not in json.loads(sent["body"])
               for sent in transport.sent[:2])


def test_definite_cost_is_accounted_before_invalid_tool_format(tmp_path):
    probe = json.loads(PROBES[-1].read_text())
    response = copy.deepcopy(probe["attempts"][0]["response"])
    response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
        '{"evidence_id":"wrong"}')
    transport = FakeTransport([(200, response)])
    result = run_probe(model=probe["model"], output=tmp_path / "evidence",
                       key=KEY, transport=transport, ledger_mode="local_diagnostic")
    assert result["probe_failure"] == "ValueError"
    assert result["tool_continuations_and_final_json"] is False
    assert Decimal(result["provider_reported_spend_usd"]) == Decimal(str(response["usage"]["cost"]))
    assert Decimal(result["still_reserved_or_unpriced_usd"]) == 0
    budget = json.loads((tmp_path / "arena-live-capture-budget.json").read_text())
    assert Decimal(budget["spent_usd"]) == Decimal(str(response["usage"]["cost"]))
    assert Decimal(budget["reserved_usd"]) == 0


def test_failed_first_provider_call_stops_without_retry_and_keeps_local_evidence(tmp_path):
    probe = json.loads(PROBES[0].read_text())
    transport = FakeTransport([(429, {"error": {"message": "rate limited"}})])
    result = run_probe(model=probe["model"], output=tmp_path / "evidence",
                       key=KEY, transport=transport, ledger_mode="local_diagnostic")
    assert len(transport.sent) == 1
    assert len(result["turns"]) == 1
    assert result["tool_continuations_and_final_json"] is False
    assert result["classification"] == "incomplete_or_unproved"
    assert result["readback_failure"] is None
    assert result["readback"][0]["provider_response_status"] == 429
    assert result["readback"][0]["stored_provider_response_readable"] is True
    assert result["readback"][0]["selected_terminal_differs_from_provider"] is True
    assert (tmp_path / "evidence/report.json").exists()


def test_uncertain_cost_keeps_upstream_body_without_claiming_terminal(tmp_path, monkeypatch):
    probe = json.loads(PROBES[-1].read_text())
    response = copy.deepcopy(probe["attempts"][0]["response"])
    response["usage"].pop("cost", None)
    monkeypatch.setattr(br, "_OPENROUTER_BILLING_MAX_ATTEMPTS", 1)
    transport = FakeTransport([(200, response)])
    result = run_probe(model=probe["model"], output=tmp_path / "evidence",
                       key=KEY, transport=transport, ledger_mode="local_diagnostic")
    assert len(transport.sent) == 1
    assert result["turns"][0]["outcome"] == "uncertain"
    assert result["readback_failure"] is None
    readback = result["readback"][0]
    assert readback["accounting_state"] == "uncertain"
    assert readback["wire_receipt_exact"] is True
    assert readback["stored_provider_response_readable"] is True
    assert readback["stored_response_readable"] is False
    assert "provider_cost_unresolved" in readback["evidence_gaps"]


def test_summary_only_candidate_stays_diagnostic(tmp_path):
    probe = json.loads(PROBES[-1].read_text())
    _, route, _ = route_registry(None, probe["model"])
    candidate = copy.deepcopy(route)
    candidate["model_id"] = "example/summary-only"
    candidate["model_version"] = "example/summary-only"
    candidate.pop("evidence")
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate))
    replies = []
    for attempt in probe["attempts"]:
        response = copy.deepcopy(attempt["response"])
        response["model"] = candidate["model_id"]
        response["openrouter_metadata"]["endpoints"]["available"] = [
            dict(item, model=candidate["model_version"])
            for item in response["openrouter_metadata"]["endpoints"]["available"]]
        message = response["choices"][0]["message"]
        message.pop("reasoning", None)
        message.pop("reasoning_content", None)
        message["reasoning_details"] = [{"type": "reasoning.summary", "summary": "brief"}]
        replies.append((200, response))
    transport = FakeTransport(replies)
    result = run_probe(model=candidate["model_id"], route_file=path,
                       output=tmp_path / "evidence", key=KEY,
                       transport=transport, ledger_mode="local_diagnostic",
                       mode="summary_diagnostic")
    assert result["tool_continuations_and_final_json"] is True
    assert result["registry_listed"] is False
    assert result["classification"] == "summary_only_diagnostic"
    assert all(row["capture"]["returned"] == "summary" for row in result["readback"])
    assert all(json.loads(item["body"])["provider"]["only"] == [candidate["provider"]]
               and json.loads(item["body"])["provider"]["allow_fallbacks"] is False
               for item in transport.sent)
    assert len(transport.sent) == 3
    for sent in transport.sent[1:]:
        history = json.loads(sent["body"])["messages"]
        assert any(item["role"] == "tool" for item in history)
        assert any(item["role"] == "assistant" and item.get("tool_calls") for item in history)
        assert all(not ({"reasoning", "reasoning_content", "reasoning_details", "refusal"}
                        & item.keys()) for item in history if item["role"] == "assistant")
    export = json.loads((tmp_path / "evidence/diagnostic-export.json").read_text())
    assert all("reasoning_capture" not in call for call in export["provider_calls"])


def test_summary_diagnostic_history_projection_preserves_tool_identity():
    continuation = [{"role": "assistant", "content": None,
                     "reasoning_details": [{"type": "reasoning.summary", "summary": "brief"}],
                     "tool_calls": [{"id": "call-1", "index": 0, "type": "function",
                                     "function": {"name": "lookup_evidence", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "call-1", "content": "evidence"}]
    projected = harness._legacy_chat_diagnostic_continuation(continuation)
    assert projected == [{"role": "assistant", "content": "",
                          "tool_calls": [{"id": "call-1", "type": "function",
                                          "function": {"name": "lookup_evidence", "arguments": "{}"}}]},
                         continuation[1]]
    assert continuation[0]["reasoning_details"]
    assert continuation[0]["content"] is None


def test_responses_summary_only_candidate_stays_diagnostic(tmp_path):
    probe = json.loads(PROBES[-1].read_text())
    _, route, _ = route_registry(None, probe["model"])
    candidate = copy.deepcopy(route)
    candidate.update(model_id="example/responses-summary", model_version="example/responses-summary",
                     api="openrouter.responses", full_text_fields=["output.content.reasoning_text"])
    candidate.pop("evidence")
    path = tmp_path / "candidate-responses.json"
    path.write_text(json.dumps(candidate))
    replies = []
    for turn, attempt in enumerate(probe["attempts"]):
        source = attempt["response"]
        metadata = copy.deepcopy(source["openrouter_metadata"])
        metadata["endpoints"]["available"] = [
            dict(item, model=candidate["model_version"])
            for item in metadata["endpoints"]["available"]]
        output = [{"type": "reasoning", "format": "azure-openai-responses-v1",
                   "encrypted_content": "opaque-dummy-token",
                   "summary": [{"type": "summary_text", "text": "brief"}]}]
        if turn < 2:
            output.append({"type": "function_call", "call_id": f"call-{turn}",
                           "name": "lookup_evidence",
                           "arguments": json.dumps({"evidence_id": ("profile", "headcount")[turn]}),
                           "status": "completed"})
        else:
            output.append({"type": "message", "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": json.dumps({
                               "company": "Acme Test", "qualified": True,
                               "evidence_ids": ["profile", "headcount"]})}]})
        replies.append((200, {"id": f"response-{turn}", "model": candidate["model_id"],
                              "provider": candidate["provider_name"], "status": "completed",
                              "output": output, "usage": source["usage"],
                              "openrouter_metadata": metadata}))
    transport = FakeTransport(replies)
    result = run_probe(model=candidate["model_id"], route_file=path,
                       output=tmp_path / "evidence", key=KEY,
                       transport=transport, ledger_mode="local_diagnostic",
                       mode="summary_diagnostic")
    assert result["probe_failure"] is None
    assert result["readback_failure"] is None
    assert result["operation_id"] == "openrouter.responses"
    assert result["tool_continuations_and_final_json"] is True
    assert result["classification"] == "summary_only_diagnostic"
    assert result["route_probe_passed"] is False
    assert all(json.loads(item["body"])["provider"]["only"] == [candidate["provider"]]
               and json.loads(item["body"])["provider"]["allow_fallbacks"] is False
               for item in transport.sent)
    export = json.loads((tmp_path / "evidence/diagnostic-export.json").read_text())
    assert all("reasoning_capture" not in call for call in export["provider_calls"])


@pytest.mark.parametrize("api,option", [
    ("openrouter.chat", "diagnostic_tool_choice_auto"),
    ("openrouter.responses", "diagnostic_responses_compat"),
])
def test_summary_compat_controls_keep_exact_pin_without_full_admission(tmp_path, api, option):
    _, route, _ = route_registry(None, "openai/gpt-oss-120b")
    candidate = copy.deepcopy(route)
    candidate.update(model_id="example/compat", model_version="example/compat", api=api)
    if api == "openrouter.responses":
        candidate["full_text_fields"] = ["output.content.reasoning_text"]
    candidate.pop("evidence")
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate))
    transport = FakeTransport([(429, {"error": {"message": "diagnostic stop"}})])
    result = run_probe(model=candidate["model_id"], route_file=candidate_path,
                       output=tmp_path / "evidence", key=KEY,
                       transport=transport, ledger_mode="local_diagnostic",
                       mode="summary_diagnostic", **{option: True})
    assert result["registry_listed"] is False
    assert result["route_probe_passed"] is False
    assert len(transport.sent) == 1
    outbound = json.loads(transport.sent[0]["body"])
    assert outbound["provider"]["only"] == [candidate["provider"]]
    assert outbound["provider"]["zdr"] is True
    assert outbound["provider"]["allow_fallbacks"] is False
    assert "max_price" in outbound["provider"]
    if api == "openrouter.chat":
        assert outbound["tool_choice"] == "auto"
        assert "parallel_tool_calls" not in outbound
        assert outbound["provider"]["require_parameters"] is True
    else:
        assert outbound["reasoning"]["summary"] == "auto"
        assert outbound["provider"]["require_parameters"] is False


def test_diagnostic_compat_controls_cannot_modify_full_policy(tmp_path):
    for option in ("diagnostic_tool_choice_auto", "diagnostic_responses_compat"):
        with pytest.raises(ValueError, match="diagnostic_controls_require_summary_mode"):
            run_probe(model="openai/gpt-oss-120b", output=tmp_path / option,
                      key=KEY, ledger_mode="local_diagnostic", **{option: True})


@pytest.mark.skipif(os.environ.get("ARENA_CAPTURE_POSTGRES_TEST") != "1",
                    reason="opt-in disposable PostgreSQL migration fixture")
@pytest.mark.parametrize("api,option", [
    ("openrouter.chat", "diagnostic_tool_choice_auto"),
    ("openrouter.responses", "diagnostic_responses_compat"),
])
def test_diagnostic_controls_run_through_real_socket(tmp_path, api, option):
    _, route, _ = route_registry(None, "openai/gpt-oss-120b")
    candidate = copy.deepcopy(route)
    candidate.update(model_id="example/compat-socket", model_version="example/compat-socket", api=api)
    if api == "openrouter.responses":
        candidate["full_text_fields"] = ["output.content.reasoning_text"]
    candidate.pop("evidence")
    route_file = tmp_path / "candidate.json"
    route_file.write_text(json.dumps(candidate))
    transport = FakeTransport([(429, {"error": {"message": "diagnostic stop"}})])
    result = run_probe(model=candidate["model_id"], route_file=route_file,
                       output=tmp_path / "evidence", key=KEY, transport=transport,
                       worker_socket=True, mode="summary_diagnostic", **{option: True})
    assert result["probe_failure"] is None
    assert result["worker_snapshot_failure"] is None
    assert result["readback_failure"] is None
    assert result["worker_receipts_proved"] is True
    assert result["route_probe_passed"] is False
    assert len(transport.sent) == 1


def test_single_route_registry_candidate_file_is_probe_only(tmp_path):
    _, route, _ = route_registry(None, "openai/gpt-oss-120b")
    candidate = copy.deepcopy(route)
    candidate["model_id"] = "example/one-route-registry"
    candidate["model_version"] = candidate["model_id"]
    candidate.pop("evidence")
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"schema_version": 1,
                                "approved_routes": [candidate], "baseline_route": None}))
    _, loaded, authority = route_registry(path, candidate["model_id"])
    assert loaded["model_id"] == candidate["model_id"]
    assert authority == "isolated_candidate_probe_only"


def test_shared_budget_retains_uncertain_hold_and_transport_blocks_second_send(tmp_path):
    first = SharedBudget(tmp_path / "budget.json", Decimal("8"))
    first.reserve(Decimal("5"))
    second = SharedBudget(tmp_path / "budget.json", Decimal("8"))
    with pytest.raises(ValueError, match="cumulative_budget_exceeded"):
        second.reserve(Decimal("4"))
    first.settle(Decimal("5"), Decimal("0.25"))
    second.reserve(Decimal("4"))
    assert json.loads((tmp_path / "budget.json").read_text())["reserved_usd"] == "4"

    transport = FakeTransport([(200, {"ok": True}), (200, {"ok": True})])
    wire = WireTransport(tmp_path, transport)
    wire.arm(0)
    kwargs = {"method": "POST", "url": "https://openrouter.ai/api/v1/chat/completions",
              "headers": {}, "body": b"{}", "timeout_seconds": 1}
    wire.send(**kwargs)
    with pytest.raises(Exception, match="probe_additional_http_blocked"):
        wire.send(**kwargs)
    assert len(transport.sent) == 1


@pytest.mark.skipif(os.environ.get("ARENA_CAPTURE_POSTGRES_TEST") != "1",
                    reason="opt-in disposable PostgreSQL migration fixture")
def test_disposable_postgres_real_rpc_and_sql_readback(tmp_path):
    probe = json.loads(PROBES[-1].read_text())
    transport = FakeTransport([(attempt["status"], attempt["response"])
                               for attempt in probe["attempts"]])
    result = run_probe(model=probe["model"], output=tmp_path / "evidence",
                       key=KEY, transport=transport)
    assert result["ledger_mode"] == "postgres"
    assert result["probe_failure"] is None
    assert result["readback_failure"] is None
    assert result["classification"] == "full_thinking_observed"
    assert len(result["readback"]) == 3
    assert all(row["stored_request_readable"] and row["stored_response_readable"]
               for row in result["readback"])
    assert (tmp_path / "evidence/readable-trace.json").exists()
    ledger = json.loads((tmp_path / "evidence/ledger-readback.json").read_text())
    assert ledger["run_id"]
    assert len([row for row in ledger["rows"] if row["entry_kind"] == "settlement"]) == 3


@pytest.mark.skipif(os.environ.get("ARENA_CAPTURE_POSTGRES_TEST") != "1",
                    reason="opt-in disposable PostgreSQL migration fixture")
def test_disposable_sql_ledger_survives_export_readback_failure(tmp_path, monkeypatch):
    probe = json.loads(PROBES[-1].read_text())
    transport = FakeTransport([(200, probe["attempts"][0]["response"])])

    def broken_export(*args, **kwargs):
        raise ValueError("simulated_export_failure")

    monkeypatch.setattr(harness, "verify_storage", broken_export)
    result = run_probe(model=probe["model"], output=tmp_path / "evidence",
                       key=KEY, transport=transport)
    assert result["readback_failure"] == "ValueError"
    ledger_path = tmp_path / "evidence/ledger-readback.json"
    assert ledger_path.stat().st_mode & 0o077 == 0
    ledger = json.loads(ledger_path.read_text())
    assert ledger["run_id"]
    assert len([row for row in ledger["rows"] if row["entry_kind"] == "settlement"]) == 1
    assert KEY not in ledger_path.read_text()


@pytest.mark.skipif(os.environ.get("ARENA_CAPTURE_POSTGRES_TEST") != "1",
                    reason="opt-in disposable PostgreSQL migration fixture")
@pytest.mark.parametrize("controls", [{}, {"tool_choice": "auto", "final_format": "json_object",
                                         "budget_bound_mode": "request"}])
def test_real_socket_delivery_ack_and_durable_worker_snapshot(tmp_path, controls):
    probe = json.loads(PROBES[-1].read_text())
    transport = FakeTransport([(attempt["status"], attempt["response"])
                               for attempt in probe["attempts"]])
    result = run_probe(model=probe["model"], output=tmp_path / "evidence",
                       key=KEY, transport=transport, worker_socket=True, **controls)
    assert result["probe_failure"] is None
    assert result["worker_snapshot_failure"] is None
    assert result["readback_failure"] is None
    assert result["tool_continuations_and_final_json"] is True
    assert result["worker_receipts_proved"] is True
    assert result["worker_snapshot_payload_sha256"]
    export = json.loads((tmp_path / "evidence/diagnostic-export.json").read_text())
    calls = export["provider_calls"]
    assert len(calls) == 3
    assert all(call["response_provenance"]["delivery_to_worker"] == "socket_written"
               and call["response_provenance"]["agent_receipt"] == "reported_exact_body_received"
               and call["response_provenance"]["selected_response_acknowledged"] is True
               for call in calls)
    worker = export["worker_observation_artifacts"][export["run_id"]]
    assert worker["payload"]["response_receipt_count"] == 3
    assert worker["payload"]["agent_observation_count"] == 3


@pytest.mark.skipif(os.environ.get("ARENA_CAPTURE_POSTGRES_TEST") != "1",
                    reason="opt-in disposable PostgreSQL migration fixture")
def test_real_socket_error_body_ack_is_not_called_provider_answer(tmp_path):
    probe = json.loads(PROBES[-1].read_text())
    transport = FakeTransport([(429, {"error": {"message": "rate limited"}})])
    result = run_probe(model=probe["model"], output=tmp_path / "evidence",
                       key=KEY, transport=transport, worker_socket=True)
    assert result["worker_snapshot_failure"] is None
    assert result["readback_failure"] is None
    assert result["tool_continuations_and_final_json"] is False
    assert result["worker_receipts_proved"] is True
    export = json.loads((tmp_path / "evidence/diagnostic-export.json").read_text())
    call = export["provider_calls"][0]
    provenance = call["response_provenance"]
    assert provenance["agent_receipt"] == "reported_exact_body_received"
    assert provenance["selected_response_acknowledged"] is True
    assert call["response_status"] == 502
    assert call["upstream_response_receipts"][0]["status"] == 429


@pytest.mark.skipif(os.environ.get("ARENA_CAPTURE_POSTGRES_TEST") != "1",
                    reason="opt-in disposable PostgreSQL migration fixture")
def test_socket_predispatch_error_reports_full_budget_hold(tmp_path, monkeypatch):
    probe = json.loads(PROBES[-1].read_text())
    transport = FakeTransport()

    def invalid_local_frame(*args, **kwargs):
        raise shim.ShimProviderError("invalid_frame")

    monkeypatch.setattr(shim, "dispatch", invalid_local_frame)
    result = run_probe(model=probe["model"], output=tmp_path / "evidence",
                       key=KEY, transport=transport, worker_socket=True)
    assert result["probe_failure"] == "ShimProviderError"
    assert result["probe_failure_turn"] == 0
    assert result["probe_failure_phase"] == "before_provider_http_send"
    assert result["turns"] == []
    assert result["worker_receipts_proved"] is False
    assert result["provider_reported_spend_usd"] == "0"
    assert Decimal(result["still_reserved_or_unpriced_usd"]) == Decimal(
        result["worst_case_per_call_usd"])
    assert not transport.sent
    assert json.loads((tmp_path / "evidence/ledger-readback.json").read_text())["rows"] == []
