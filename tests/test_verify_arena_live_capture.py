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
                                                _parameters, conservative_bound, route_registry)
from tests.lab_arena.test_lab_arena_broker import FakeTransport

EVIDENCE = Path(__file__).parents[1] / "docs/evidence/full-thinking"
PROBES = sorted(EVIDENCE.glob("*chat-probe-price-bound.json"))
KEY = "sk-or-v1-" + "q" * 40


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
def test_real_socket_delivery_ack_and_durable_worker_snapshot(tmp_path):
    probe = json.loads(PROBES[-1].read_text())
    transport = FakeTransport([(attempt["status"], attempt["response"])
                               for attempt in probe["attempts"]])
    result = run_probe(model=probe["model"], output=tmp_path / "evidence",
                       key=KEY, transport=transport, worker_socket=True)
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
