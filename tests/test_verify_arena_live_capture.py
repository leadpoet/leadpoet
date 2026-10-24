"""The live harness's no-network controls and durable readback boundary."""
import copy
from decimal import Decimal
import json
import os
from pathlib import Path

import pytest

from scripts.verify_arena_live_capture import (SharedBudget, WireTransport, run_probe,
                                                conservative_bound, route_registry)
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


def test_failed_first_provider_call_stops_without_retry_and_keeps_local_evidence(tmp_path):
    probe = json.loads(PROBES[0].read_text())
    transport = FakeTransport([(429, {"error": {"message": "rate limited"}})])
    result = run_probe(model=probe["model"], output=tmp_path / "evidence",
                       key=KEY, transport=transport, ledger_mode="local_diagnostic")
    assert len(transport.sent) == 1
    assert len(result["turns"]) == 1
    assert result["tool_continuations_and_final_json"] is False
    assert result["classification"] == "incomplete_or_unproved"
    assert (tmp_path / "evidence/report.json").exists()


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
    result = run_probe(model=candidate["model_id"], route_file=path,
                       output=tmp_path / "evidence", key=KEY,
                       transport=FakeTransport(replies), ledger_mode="local_diagnostic")
    assert result["tool_continuations_and_final_json"] is True
    assert result["production_whitelist_approved"] is False
    assert result["classification"] == "summary_only_diagnostic"
    assert all(row["capture"]["returned"] == "summary" for row in result["readback"])


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
        output = [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "brief"}]}]
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
    result = run_probe(model=candidate["model_id"], route_file=path,
                       output=tmp_path / "evidence", key=KEY,
                       transport=FakeTransport(replies), ledger_mode="local_diagnostic",
                       mode="summary_diagnostic")
    assert result["probe_failure"] is None
    assert result["readback_failure"] is None
    assert result["operation_id"] == "openrouter.responses"
    assert result["tool_continuations_and_final_json"] is True
    assert result["classification"] == "summary_only_diagnostic"
    assert result["full_thinking_eligible"] is False


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
