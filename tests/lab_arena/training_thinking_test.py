import copy
import json
import os

import pytest

from lab_arena import contracts, training_capture, training_export, training_steps, training_thinking
from scripts.prepare_arena_full_thinking_dataset import main
from tests.lab_arena.training_steps_test import bind_selected_output, fixture, response_fixture, seal


def full_example():
    example = fixture()
    call = example["provider_calls"][0]
    call["response"]["choices"][0]["message"]["reasoning_content"] = "Inspect the retained page, then answer."
    call["response_sha256"] = contracts.document_hash(call["response"])
    call["response_receipts"][0]["response_sha256"] = call["response_sha256"]
    route = {"model_id": "test/model", "model_version": "test/model", "provider": "route-a",
             "provider_name": "Route A", "api": "openrouter.chat", "mode": {"effort": "high"},
             "parser_version": "arena_openrouter_reasoning_v1", "semantic_class": "full_generated_thinking",
             "full_text_fields": ["reasoning_content"], "max_output_tokens": 4096,
             "answer_headroom_tokens": 1024, "context_tokens": 8192,
             "price_ceiling": {"prompt": "1", "completion": "1", "request": "0", "internal_reasoning": "1"},
             "evidence": {"documentation_url": "https://example.com/docs", "probe_id": "probe-001",
                          "verified_at": "2026-10-23T00:00:00Z", "probe_manifest_sha256": "a" * 64},
             "allowed_fallback_routes": [], "status": "approved"}
    registry = {"schema_version": 1, "approved_routes": [route],
                "baseline_route": {"model_id": "test/model", "api": "openrouter.chat"}}
    call["reasoning_capture"] = {
        "policy": training_thinking.POLICY, "requested": True, "reviewed_route": True,
        "semantic_class": "full_generated_thinking", "model_id": "test/model",
        "model_version": "test/model", "provider_route": "route-a", "provider_name": "Route A",
        "api": "openrouter.chat", "context_tokens": 8192, "max_output_tokens": 4096,
        "price_ceiling": dict(route["price_ceiling"]),
        "mode": {"effort": "high"}, "parser_version": "arena_openrouter_reasoning_v1",
        "full_text_fields": ["reasoning_content"],
        "probe_id": "probe-001", "probe_manifest_sha256": "a" * 64,
        "registry_sha256": contracts.document_hash(registry).split(":", 1)[1],
    }
    call["response_receipts"][0]["metadata"]["reasoning_route_observation"] = {
        "status": "matched", "requested_model": "test/model", "requested_provider_route": "route-a",
        "expected_provider_name": "Route A", "expected_model_version": "test/model",
        "response_model": "test/model", "response_provider_name": "Route A",
        "selected_endpoint_model": "test/model", "selected_endpoint_provider_name": "Route A"}
    example["role"] = "miner"
    example["configuration"] = {"reasoning_capture_policy": training_thinking.POLICY,
                                "reasoning_route_registry": registry}
    example["completeness"].update(complete=True, incomplete_reasons=[])
    return bind_selected_output(example)


def rows(example):
    splits, manifest = training_thinking.build([seal(example)])
    return splits["train"] + splits["eval"], manifest


def test_actual_delivered_text_is_adjacent_to_input_and_action():
    example = full_example()
    observation = {"id": "agent:0", "kind": "web_fetch", "observed_at": "2026-10-23T00:00:00Z",
                   "after_action_sequence": 0, "depends_on": [], "privacy": {"redacted": False},
                   "data": {"url": "https://example.com", "body": "A retained observation", "capture": {"complete": True}}}
    # A prior observation for action 1 must not be confused with a later one.
    call = example["provider_calls"][0]
    call["action_sequence"] = 1
    example["worker_observation_artifacts"]["run"]["payload"]["agent_observations"].append(observation)
    artifact = example["worker_observation_artifacts"]["run"]
    artifact["payload_sha256"] = contracts.document_hash(artifact["payload"])
    for ref in call["response_provenance"]["agent_receipt_refs"] + [call["response_provenance"]["worker_receipt_ref"]]:
        ref["payload_sha256"] = artifact["payload_sha256"]
    # The fixture's receipt action is updated with the model call.
    artifact["payload"]["response_receipts"][0]["action_sequence"] = 1
    artifact["payload_sha256"] = contracts.document_hash(artifact["payload"])
    for ref in call["response_provenance"]["agent_receipt_refs"] + [call["response_provenance"]["worker_receipt_ref"]]:
        ref["payload_sha256"] = artifact["payload_sha256"]
    got, manifest = rows(example)
    assert len(got) == 1
    assert got[0]["input"]["messages"][0]["content"] == "Find a company"
    assert got[0]["generated_thinking"][0]["text"] == "Inspect the retained page, then answer."
    assert got[0]["target"]["content"] == "Observed answer"
    assert got[0]["prior_observations"][0]["data"]["body"] == "A retained observation"
    assert got[0]["source_refs"][0]["response_sha256"] == call["response_sha256"]
    assert manifest["coverage"]["generated_thinking_texts"] == 1


def test_summary_unproved_and_redacted_text_are_excluded():
    for mutation in ("summary", "unproved", "unapproved_field", "redacted"):
        example = full_example()
        call = example["provider_calls"][0]
        if mutation == "summary":
            call["response"]["choices"][0]["message"]["reasoning_details"] = [
                {"type": "reasoning.summary", "summary": [{"type": "summary_text", "text": "Summary"}]}]
        elif mutation == "unproved":
            call["reasoning_capture"]["reviewed_route"] = False
        elif mutation == "unapproved_field":
            call["response"]["choices"][0]["message"]["reasoning"] = "Provider summary alias"
        else:
            call["response_receipts"][0]["privacy_redactions"] = ["credential"]
        call["response_sha256"] = contracts.document_hash(call["response"])
        call["response_receipts"][0]["response_sha256"] = call["response_sha256"]
        got, manifest = rows(example)
        assert not got, mutation
        assert manifest["counts"]["records_by_split"]["train"] == 0


def test_literal_redaction_marker_in_exact_thinking_is_retained_but_capture_redaction_is_not():
    example = full_example()
    call = example["provider_calls"][0]
    text = "The public source literally prints [REDACTED] in its example."
    call["response"]["choices"][0]["message"]["reasoning_content"] = text
    call["response_sha256"] = contracts.document_hash(call["response"])
    call["response_receipts"][0]["response_sha256"] = call["response_sha256"]
    got, _ = rows(example)
    assert len(got) == 1
    assert got[0]["generated_thinking"][0]["text"] == text
    trace = training_thinking.readable_trace(seal(example))
    assert any(event["kind"] == "full_generated_thinking" and event["value"] == text
               for row in trace["calls"] for event in row["events"])

    call["response_receipts"][0]["privacy_redactions"] = ["credential"]
    assert rows(example)[0] == []

    redacted = full_example()
    saved = redacted["provider_calls"][0]
    saved["response"]["choices"][0]["message"]["reasoning_content"] = "Evidence token: fixture-private-token"
    saved["response"], redactions = training_capture.sanitize_training_value(
        saved["response"], secrets=("fixture-private-token",))
    assert redactions == ["credential"]
    assert saved["response"]["choices"][0]["message"]["reasoning_content"] == "Evidence token: [REDACTED]"
    saved["response_privacy_redactions"] = redactions
    saved["response_receipts"][0]["privacy_redactions"] = redactions
    saved["response_sha256"] = contracts.document_hash(saved["response"])
    saved["response_receipts"][0]["response_sha256"] = saved["response_sha256"]
    assert rows(redacted)[0] == []


def test_literal_redaction_marker_in_exact_summary_is_rendered_as_summary():
    text = "The source includes the literal marker [REDACTED]."
    for operation, response in (
        ("openrouter.chat", {"choices": [{"message": {"reasoning_details": [
            {"type": "reasoning.summary", "summary": text}]}, "finish_reason": "stop"}]}),
        ("openrouter.responses", {"status": "completed", "output": [{"type": "reasoning",
            "summary": [{"type": "summary_text", "text": text}]}]}),
    ):
        segments, summary, opaque = training_thinking._segments(response, operation)
        assert summary is True and opaque is False
        assert [segment["text"] for segment in segments] == [text]
        example = full_example()
        call = example["provider_calls"][0]
        call["operation_id"] = operation
        call["response"] = response
        assert training_thinking.attempt_coverage(call, "miner")["returned"] == "summary"
        trace = training_thinking.readable_trace(seal(example))
        assert any(event["kind"] == "summary" and event["value"] == text
                   for row in trace["calls"] for event in row["events"])
        assert text in training_thinking.readable_trace_markdown(trace)


def test_summary_only_text_is_counted_with_source_paths_but_not_admitted():
    for operation, response in (
        ("openrouter.chat", {"choices": [{"message": {"reasoning_details": [
            {"type": "reasoning.summary", "summary": "A short visible summary."}]},
            "finish_reason": "stop"}]}),
        ("openrouter.responses", {"status": "completed", "output": [{"type": "reasoning",
            "summary": [{"type": "summary_text", "text": "A short visible summary."}]}]}),
    ):
        segments, summary, opaque = training_thinking._segments(response, operation)
        assert summary is True and opaque is False
        assert segments == [{"field": ("reasoning_details.summary" if operation == "openrouter.chat"
                                      else "output.summary.summary_text"),
                             "text": "A short visible summary.",
                             "source_paths": [(["choices", 0, "message", "reasoning_details", 0, "summary"]
                                               if operation == "openrouter.chat" else
                                               ["output", 0, "summary", 0, "text"])]}]
        example = full_example()
        call = example["provider_calls"][0]
        call["operation_id"] = operation
        call["response"] = response
        attempt = training_thinking.attempt_coverage(call, "miner")
        assert attempt["returned"] == "summary"
        assert attempt["characters"] == len("A short visible summary.")


def test_summary_route_coverage_uses_actual_request_and_nested_responses_usage():
    call = copy.deepcopy(full_example()["provider_calls"][0])
    call.pop("reasoning_capture")
    call["operation_id"] = "openrouter.responses"
    call["request"]["outbound"]["body"]["reasoning"] = {"effort": "high", "summary": "auto"}
    call["response"] = {"status": "completed", "output": [{"type": "reasoning",
                         "summary": [{"type": "summary_text", "text": "Visible summary"}]}],
                        "usage": {"output_tokens_details": {"reasoning_tokens": 81}}}
    call["response_privacy_redactions"] = ["hidden_reasoning"]
    result = training_thinking.attempt_coverage(call, "miner")
    assert result["request_state"] == "requested" and result["requested"] is True
    assert result["returned"] == "summary" and result["contains_opaque"] is True
    assert result["characters"] == len("Visible summary") and result["tokens_reported"] == 81
    assert result["route_reviewed"] is False


def test_mixed_unverified_readable_text_and_opaque_is_not_labeled_opaque_only():
    call = copy.deepcopy(full_example()["provider_calls"][0])
    call.pop("reasoning_capture")
    call["request"]["outbound"]["body"]["reasoning"] = {"effort": "high", "exclude": False}
    call["response_privacy_redactions"] = ["hidden_reasoning"]
    result = training_thinking.attempt_coverage(call, "miner")
    assert result["returned"] == "unproved_text"
    assert result["contains_opaque"] is True and result["characters"] > 0
    assert result["requested"] is True and result["route_reviewed"] is False


def test_empty_reasoning_class_uses_saved_request_state():
    call = copy.deepcopy(full_example()["provider_calls"][0])
    call.pop("reasoning_capture")
    call["response"]["choices"][0]["message"].pop("reasoning_content")
    call["request"]["outbound"]["body"]["reasoning"] = {"effort": "high", "exclude": False}
    assert training_thinking.attempt_coverage(call, "miner")["returned"] == "requested_empty"
    del call["request"]["outbound"]["body"]["reasoning"]
    assert training_thinking.attempt_coverage(call, "miner")["returned"] == "not_requested"
    del call["request"]
    assert training_thinking.attempt_coverage(call, "miner")["returned"] == "unknown"


def test_request_coverage_reads_budget_and_legacy_include_controls():
    for body, expected in (
        ({"reasoning": {"max_tokens": 1024}}, "requested"),
        ({"reasoning": {"budget_tokens": 1024}}, "requested"),
        ({"include_reasoning": True}, "requested"),
        ({"reasoning": {"effort": "none", "summary": "none"}}, "not_requested"),
        ({"reasoning": {"max_tokens": 0}}, "not_requested"),
        ({"reasoning": {"exclude": True, "effort": "high"}}, "not_requested"),
        ({"reasoning": {"enabled": False}, "include_reasoning": True}, "not_requested"),
    ):
        call = {"request": {"outbound": {"body": body}}}
        assert training_thinking._reasoning_request_state(call, {}) == expected


@pytest.mark.parametrize("controls", [
    {"reasoning": {"effort": "low", "exclude": False}},
    {"reasoning": {"effort": "medium"}},
    {"reasoning": {"max_tokens": 2048}},
    {"reasoning": {"budget_tokens": 1024}},
    {"reasoning_effort": "low", "include_reasoning": True},
    {"reasoning": {"enabled": True}},
    {"reasoning": {"enabled": False}},
    {},
])
def test_attempt_reports_saved_controls_separately_from_high_probe(controls):
    call = copy.deepcopy(full_example()["provider_calls"][0])
    call["reasoning_capture"]["mode"] = {"effort": "high"}
    body = call["request"]["outbound"]["body"]
    for field in ("reasoning", "reasoning_effort", "include_reasoning"):
        body.pop(field, None)
    body.update(copy.deepcopy(controls))
    before = copy.deepcopy(call)

    row = training_thinking.attempt_coverage(call, "miner")

    assert row["outbound_reasoning_controls"] == controls
    assert row["outbound_reasoning_controls_status"] == ("present" if controls else "absent")
    assert row["probe_reasoning_mode"] == {"effort": "high"}
    assert call == before
    if controls:
        row["outbound_reasoning_controls"].clear()
        assert call == before


def test_attempt_does_not_infer_actual_controls_from_probe_without_saved_body():
    call = copy.deepcopy(full_example()["provider_calls"][0])
    call["reasoning_capture"]["mode"] = {"effort": "high"}
    del call["request"]["outbound"]["body"]

    row = training_thinking.attempt_coverage(call, "miner")

    assert row["outbound_reasoning_controls"] is None
    assert row["outbound_reasoning_controls_status"] == "unavailable"
    assert row["probe_reasoning_mode"] == {"effort": "high"}


def test_saved_controls_reach_json_coverage_and_readable_trace():
    example = full_example()
    call = example["provider_calls"][0]
    call["request"]["outbound"]["body"]["reasoning"] = {"effort": "low"}
    example = seal(example)

    coverage = json.loads(json.dumps(training_thinking.coverage([example])))
    trace = training_thinking.readable_trace(example)
    markdown = training_thinking.readable_trace_markdown(trace)
    for row in (coverage["attempts"][0], trace["calls"][0]["reasoning_coverage"]):
        assert row["outbound_reasoning_controls"] == {"reasoning": {"effort": "low"}}
        assert row["probe_reasoning_mode"] == {"effort": "high"}
    assert '"effort": "low"' in markdown
    assert '"probe_reasoning_mode"' in markdown


def test_reasoning_usage_null_alias_does_not_hide_nested_count():
    call = {"operation_id": "openrouter.responses", "response": {
        "status": "completed", "output": [], "usage": {
            "reasoning_tokens": None, "completion_tokens_details": {"reasoning_tokens": None},
            "output_tokens_details": {"reasoning_tokens": 81}}}}
    assert training_thinking.attempt_coverage(call, "miner")["tokens_reported"] == 81


def test_readable_trace_indexes_exact_values_and_keeps_summary_unverified():
    example = full_example()
    call = example["provider_calls"][0]
    call["response"]["usage"] = {"prompt_tokens": 17, "completion_tokens": 31,
                                   "completion_tokens_details": {"reasoning_tokens": 11}}
    call["response"]["choices"][0]["message"]["reasoning_details"] = [
        {"type": "reasoning.summary", "summary": "Summary is visible."}]
    call["response_sha256"] = contracts.document_hash(call["response"])
    call["response_receipts"][0]["response_sha256"] = call["response_sha256"]
    call["actual_microusd"] = 1234
    call["lifecycle"] = [{"created_at": "2026-10-23T00:00:00Z", "entry_kind": "reservation"}]
    report = training_thinking.readable_trace(seal(example))
    assert report["projection_truncated"] is False
    assert report["capture_limits"]["receipt_max_bytes"] == training_export.MAX_RECEIPT_BYTES
    record = report["calls"][0]
    assert record["actual_microusd"] == 1234
    assert record["lifecycle"][0]["created_at"] == "2026-10-23T00:00:00Z"
    assert record["response_sha256"] == call["response_sha256"]
    assert record["reasoning_coverage"]["returned"] == "summary"
    kinds = {event["kind"] for event in record["events"]}
    assert {"summary", "usage", "assistant_output", "input_message"} <= kinds
    assert report["observations"]
    for event in record["events"] + report["observations"]:
        source = example
        for part in event["source_path"]:
            source = source[part]
        assert source == event["value"]
        assert contracts.document_hash(source) == event["value_sha256"]
    markdown = training_thinking.readable_trace_markdown(report, raw_export_name="raw-export.json")
    assert "[Open the raw private export](<raw-export.json>)" in markdown
    assert "Summary is visible." in markdown and "Inspect the retained page, then answer." in markdown
    assert "reasoning_details.summary" in markdown
    assert "1234" in markdown
    assert "[TRUNCATED" not in markdown


def test_readable_trace_preserves_document_context_and_all_evidence_gaps():
    example = full_example()
    example["configuration"]["task_label"] = "Exact saved configuration"
    example["input_snapshots"] = {"run": {"prompt": "Exact saved input"}}
    example["output"] = {"answer": "Exact saved final output"}
    example["runtime"] = {"wall_seconds": 7.5, "code_ref": "test-revision"}
    example["costs"] = {"confirmed_actual_microusd": 1234}
    example["completeness"] = {"complete": False,
                               "incomplete_reasons": ["worker_call_accounting_unverified"]}
    example["provider_calls"][0]["evidence_gaps"] = ["agent_receipt_missing"]
    report = training_thinking.readable_trace(seal(example))
    context = {entry["field"]: entry for entry in report["document_context"]}
    assert context["input_snapshots"]["value"] == example["input_snapshots"]
    assert context["input_snapshots"]["source_path"] == ["input_snapshots"]
    assert context["input_snapshots"]["value_sha256"] == contracts.document_hash(example["input_snapshots"])
    assert context["output"]["value"] == example["output"]
    assert context["judgment"]["value"] == example["judgment"]
    assert context["competition_cost_eligibility"]["status"] == "absent"
    assert report["capture_gaps"] == ["agent_receipt_missing", "worker_call_accounting_unverified"]
    markdown = training_thinking.readable_trace_markdown(report)
    for value in ("Exact saved configuration", "Exact saved input", "Exact saved final output",
                  "test-revision", "worker_call_accounting_unverified", "agent_receipt_missing"):
        assert value in markdown


def test_readable_derived_markers_do_not_claim_exact_raw_value():
    example = full_example()
    call = example["provider_calls"][0]
    call["response"]["choices"][0]["message"]["reasoning_details"] = [
        {"type": "reasoning.summary", "summary": []}]
    call["response_privacy_redactions"] = ["hidden_reasoning"]
    report = training_thinking.readable_trace(seal(example))
    markers = [event for event in report["calls"][0]["events"] if event.get("derived")]
    assert {event["kind"] for event in markers} == {"opaque_reasoning_removed", "summary_present_without_readable_text"}
    assert all("source_path" not in event and "value_sha256" not in event for event in markers)
    assert "Derived status" in training_thinking.readable_trace_markdown(report)


def test_readable_trace_markdown_fence_cannot_be_closed_by_provider_text():
    report = {"schema_version": "leadpoet.lab_arena.readable_trace.v1", "calls": [{
        "events": [{"kind": "assistant_output", "source_path": ["provider_calls", 0, "response"],
                    "value_sha256": "sha256:" + "a" * 64,
                    "value": "before\n```\nafter"}]}]}
    markdown = training_thinking.readable_trace_markdown(report)
    assert "````\nbefore\n```\nafter\n````" in markdown


def test_readable_trace_shows_repeated_reasoning_alias_once_with_both_paths():
    example = full_example()
    response = example["provider_calls"][0]["response"]
    message = response["choices"][0]["message"]
    message["reasoning"] = message["reasoning_content"]
    report = training_thinking.readable_trace(seal(example))
    reasoning = [event for event in report["calls"][0]["events"]
                 if event["kind"] in {"full_generated_thinking", "unverified_reasoning_text"}]
    assert len(reasoning) == 1
    assert reasoning[0]["alias_source_paths"] == [
        ["provider_calls", 0, "response", "choices", 0, "message", "reasoning_content"]]
    markdown = training_thinking.readable_trace_markdown(report)
    assert markdown.count("Inspect the retained page, then answer.") == 1


def test_legitimate_tool_only_empty_turn_does_not_disqualify_later_text():
    example = full_example()
    extra = copy.deepcopy(example["provider_calls"][0])
    extra["call_identity"] = "tool-only"
    extra["action_sequence"] = 2
    extra["response"]["choices"][0]["message"].pop("reasoning_content")
    extra["response"]["choices"][0]["message"]["content"] = None
    extra["response"]["choices"][0]["message"]["tool_calls"] = [{
        "id": "lookup-1", "type": "function",
        "function": {"name": "lookup", "arguments": "{}"}}]
    extra["response"]["choices"][0]["finish_reason"] = "tool_calls"
    extra["response_sha256"] = contracts.document_hash(extra["response"])
    extra["response_receipts"][0]["response_sha256"] = extra["response_sha256"]
    example["provider_calls"].append(extra)
    report = training_thinking.coverage([seal(example)])
    assert report["returned_classes"] == {"requested_empty": 1, "full_thinking": 1}
    assert report["trajectories"]["eligible_without_disqualifying_capture"] == 1


def test_null_refusal_in_assistant_tool_continuation_keeps_strict_trajectory():
    example = full_example()
    first = example["provider_calls"][0]
    final = copy.deepcopy(first)
    first_message = first["response"]["choices"][0]["message"]
    first_message.update(content=None, refusal=None, tool_calls=[{
        "id": "lookup-1", "type": "function",
        "function": {"name": "lookup", "arguments": "{}"}}])
    first["response"]["choices"][0]["finish_reason"] = "tool_calls"
    first["response_sha256"] = contracts.document_hash(first["response"])
    first["response_receipts"][0]["response_sha256"] = first["response_sha256"]
    first_body = first["request"]["outbound"]["body"]
    first_body["tools"] = [{"type": "function", "function": {
        "name": "lookup", "parameters": {"type": "object"}}}]
    first["request"]["outbound"]["body_sha256"] = contracts.document_hash(first_body)

    final.update(call_identity="call-2", action_sequence=1, terminal_entry_id=6)
    final_body = final["request"]["outbound"]["body"]
    final_body["messages"] = [copy.deepcopy(first_body["messages"][0]),
                              copy.deepcopy(first_message),
                              {"role": "tool", "tool_call_id": "lookup-1", "content": "Observed page"}]
    final_body["tools"] = copy.deepcopy(first_body["tools"])
    final["request"]["outbound"]["body_sha256"] = contracts.document_hash(final_body)
    final["response_receipts"][0].update(entry_id=6, response_ref={"field": "response", "entry_id": 6})
    final["response_provenance"]["worker_received_receipt_entry_ids"] = [6]
    final["response_provenance"]["worker_receipt_ref"]["ordinal"] = 2
    final["response_provenance"]["agent_receipt_refs"][0]["id"] = "ack-2"
    example["provider_calls"].append(final)

    artifact = example["worker_observation_artifacts"]["run"]
    first_data = artifact["payload"]["agent_observations"][0]["data"]
    first_data["body_sha256"] = contracts.document_hash(first["response"])
    first["response_receipts"][0]["wire_body_sha256"] = first_data["body_sha256"]
    artifact["payload"]["response_receipts"][0]["body_sha256"] = first_data["body_sha256"]
    second_data = dict(first_data, call_identity="call-2",
                       body_sha256=contracts.document_hash(final["response"]))
    final["response_receipts"][0]["wire_body_sha256"] = second_data["body_sha256"]
    artifact["payload"]["agent_observations"].append({
        "kind": "response_received", "id": "ack-2",
        "privacy": {"redacted": False}, "data": second_data})
    artifact["payload"]["response_receipts"].append(dict(
        second_data, ordinal=2, write_state="socket_written",
        operation_id="openrouter.chat", action_sequence=1))
    artifact["payload_sha256"] = contracts.document_hash(artifact["payload"])
    for call in (first, final):
        for ref in call["response_provenance"]["agent_receipt_refs"] + [call["response_provenance"]["worker_receipt_ref"]]:
            ref["payload_sha256"] = artifact["payload_sha256"]

    got, manifest = rows(example)
    assert len(got) == 2
    assert manifest["counts"]["exclusion_reasons"] == {}
    assert any(row["input"]["messages"][1].get("refusal") is None
               and "refusal" in row["input"]["messages"][1] for row in got)
    for invalid in ("No", False):
        bad = copy.deepcopy(final_body["messages"])
        bad[1]["refusal"] = invalid
        with pytest.raises(ValueError, match="input_shape_unsupported"):
            training_steps._chat_messages(bad, allow_reasoning_history=True)
    bad = copy.deepcopy(final_body["messages"])
    bad[1]["role"] = "user"
    with pytest.raises(ValueError):
        training_steps._chat_messages(bad, allow_reasoning_history=True)


def test_role_counts_do_not_merge_judge_and_baseline():
    baseline = full_example()
    baseline["role"] = "baseline"
    judge = copy.deepcopy(baseline["provider_calls"][0])
    judge.update(run_id="judge-run", call_identity="judge-call")
    baseline["provider_calls"].append(judge)
    baseline["attempts"] = [{"run_id": "judge-run", "kind": "score"}]
    groups = training_thinking.coverage([seal(baseline)])["by_role_model_route"]
    assert {group["role"] for group in groups} == {"baseline", "judge"}


def test_coverage_deduplicates_identical_verified_exports():
    example = full_example()
    report = training_thinking.coverage([example, copy.deepcopy(example)])
    assert report["distinct_verified_exports"] == 1
    assert report["returned_classes"] == {"full_thinking": 1}


def test_judge_and_non_model_tool_do_not_disqualify_execution_trajectory():
    example = full_example()
    judge = copy.deepcopy(example["provider_calls"][0])
    judge.update(run_id="judge-run", call_identity="judge-call")
    judge.pop("reasoning_capture")
    tool = {"run_id": "run", "call_identity": "tool-call", "provider": "scrapingdog",
            "operation_id": "scrapingdog.search", "response": {"status": "ok"}}
    example["provider_calls"].extend([judge, tool])
    example["attempts"].append({"run_id": "judge-run", "kind": "score"})
    got, manifest = rows(example)
    assert len(got) == 1
    assert manifest["coverage"]["trajectories"]["complete_full_thinking"] == 1
    assert {row["role"] for row in manifest["coverage"]["attempts"]} == {"miner", "judge"}


@pytest.mark.parametrize("role", ["miner", "baseline"])
def test_failed_execution_before_accepted_retry_does_not_disqualify_selected_trajectory(role):
    example = full_example()
    example["role"] = role
    failed = copy.deepcopy(example["provider_calls"][0])
    failed.update(run_id="failed-run", call_identity="failed-call")
    failed["response"]["choices"][0]["message"]["content"] = "Invalid company output"
    example["provider_calls"].insert(0, failed)
    example["attempts"].append(
        {"run_id": "failed-run", "kind": "execute", "status": "failed",
         "terminal_cause": "invalid_output"})

    got, manifest = rows(example)
    assert len(got) == 1
    assert got[0]["source_refs"][0]["run_id"] == example["run_id"]
    assert manifest["counts"]["exclusion_reasons"] == {}
    assert len(manifest["coverage"]["attempts"]) == 2
    assert manifest["coverage"]["trajectories"]["complete_full_thinking"] == 1
    assert len(training_thinking.readable_trace(seal(example))["calls"]) == 2
    assert example["provider_calls"][0]["run_id"] == "failed-run"

    selected = example["provider_calls"][1]
    selected["response"]["choices"][0]["message"].pop("reasoning_content")
    selected["response_sha256"] = contracts.document_hash(selected["response"])
    selected["response_receipts"][0]["response_sha256"] = selected["response_sha256"]
    got, manifest = rows(example)
    assert not got
    assert manifest["counts"]["exclusion_reasons"] == {"requested_empty": 1}


def test_registry_and_actual_route_must_match_frozen_request():
    for mutation in ("registry", "observation", "missing_observation", "malformed_fields", "malformed_mode"):
        example = full_example()
        call = example["provider_calls"][0]
        if mutation == "registry":
            example["configuration"]["reasoning_route_registry"]["approved_routes"][0]["provider_name"] = "Other"
        elif mutation == "observation":
            call["response_receipts"][0]["metadata"]["reasoning_route_observation"]["selected_endpoint_provider_name"] = "Other"
        elif mutation == "missing_observation":
            call["response_receipts"][0]["metadata"].pop("reasoning_route_observation")
        elif mutation == "malformed_mode":
            call["reasoning_capture"]["mode"] = {"effort": ["high"]}
        else:
            call["reasoning_capture"]["full_text_fields"] = [["reasoning_content"]]
        got, manifest = rows(example)
        assert not got, mutation
        assert manifest["coverage"]["trajectories"]["ineligible"] == 1


def test_history_alias_retained_for_chat_assistant_replay_independent_of_full_route():
    annotation = copy.deepcopy(full_example()["provider_calls"][0]["reasoning_capture"])
    document = {"operation_id": "openrouter.chat", "reasoning_capture": annotation,
                "outbound": {"body": {"messages": [
                    {"role": "assistant", "content": None, "reasoning_content": "Generated prior thought"},
                    {"role": "user", "content": "Request", "reasoning_content": "User-authored claim"}]}}}
    cleaned, redactions = training_capture.sanitize_training_value(document)
    assert cleaned["outbound"]["body"]["messages"][0]["reasoning_content"] == "Generated prior thought"
    assert cleaned["outbound"]["body"]["messages"][1]["reasoning_content"] == "[REDACTED]"
    assert redactions == ["hidden_reasoning"]
    projected, removed = training_export.without_hidden_reasoning(document)
    assert projected["outbound"]["body"]["messages"][0]["reasoning_content"] == "Generated prior thought"
    assert "reasoning_content" not in projected["outbound"]["body"]["messages"][1]
    assert removed
    annotation["full_text_fields"] = ["reasoning"]
    cleaned, _ = training_capture.sanitize_training_value(document)
    assert cleaned["outbound"]["body"]["messages"][0]["reasoning_content"] == "Generated prior thought"


def test_missing_saved_response_is_storage_failed_without_fabricating_text():
    example = full_example()
    call = example["provider_calls"][0]
    call.pop("response")
    call["evidence_gaps"] = ["full_response_unavailable"]
    call["upstream_response_capture"] = {"status": "unavailable", "reason": "response_capture_failed"}
    row = training_thinking.coverage([seal(example)])["attempts"][0]
    assert row["storage"] == row["returned"] == "storage_failed"
    assert row["characters"] == 0


def test_truncated_attempt_is_excluded_and_opaque_state_stays_separate_from_full_text():
    truncated = full_example()
    truncated["provider_calls"][0]["response"]["choices"][0]["finish_reason"] = "length"
    row = training_thinking.coverage([seal(truncated)])["attempts"][0]
    assert row["returned"] == "truncated"
    assert row["characters"] > 0
    assert not rows(truncated)[0]

    opaque = full_example()
    call = opaque["provider_calls"][0]
    call["response_receipts"][0]["privacy_redactions"] = ["hidden_reasoning"]
    row = training_thinking.coverage([seal(opaque)])["attempts"][0]
    assert row["returned"] == "unproved_text"
    assert row["contains_opaque"] is True
    assert row["storage"] == "saved"
    exported, _ = rows(opaque)
    assert len(exported) == 1
    assert [part["text"] for part in exported[0]["generated_thinking"]] == [
        "Inspect the retained page, then answer."]
    call["response"]["choices"][0]["message"].pop("reasoning_content")
    call["response_sha256"] = contracts.document_hash(call["response"])
    call["response_receipts"][0]["response_sha256"] = call["response_sha256"]
    assert not rows(opaque)[0]


def test_responses_full_content_with_ancillary_summary_qualifies_but_summary_only_does_not():
    example = response_fixture()
    chat = full_example()["provider_calls"][0]
    call = example["provider_calls"][0]
    # Reuse the reviewed fixture route with the Responses API and text field.
    route = copy.deepcopy(full_example()["configuration"]["reasoning_route_registry"]["approved_routes"][0])
    route.update(api="openrouter.responses", full_text_fields=["output.content.reasoning_text"])
    registry = {"schema_version": 1, "approved_routes": [route],
                "baseline_route": {"model_id": "test/model", "api": "openrouter.responses"}}
    call["reasoning_capture"] = dict(chat["reasoning_capture"], api="openrouter.responses",
                                      full_text_fields=["output.content.reasoning_text"],
                                      registry_sha256=contracts.document_hash(registry).split(":", 1)[1])
    call["response_receipts"][0]["metadata"]["reasoning_route_observation"] = copy.deepcopy(
        chat["response_receipts"][0]["metadata"]["reasoning_route_observation"])
    call["response"]["output"][0] = {
        "type": "reasoning", "id": "r", "status": "completed",
        "content": [{"type": "reasoning_text", "text": "Check evidence first."}],
    }
    call["response_sha256"] = contracts.document_hash(call["response"])
    call["response_receipts"][0]["response_sha256"] = call["response_sha256"]
    example["configuration"] = {"reasoning_capture_policy": training_thinking.POLICY,
                                "reasoning_route_registry": registry}
    example["completeness"].update(complete=True, incomplete_reasons=[])
    got, _ = rows(example)
    assert len(got) == 1
    assert got[0]["generated_thinking"][0]["text"] == "Check evidence first."
    call["response"]["output"][0]["summary"] = [{"type": "summary_text", "text": "Brief summary"}]
    call["response_sha256"] = contracts.document_hash(call["response"])
    call["response_receipts"][0]["response_sha256"] = call["response_sha256"]
    got, _ = rows(example)
    assert len(got) == 1
    assert [part["text"] for part in got[0]["generated_thinking"]] == ["Check evidence first."]
    call["response"]["output"][0].pop("content")
    call["response_sha256"] = contracts.document_hash(call["response"])
    call["response_receipts"][0]["response_sha256"] = call["response_sha256"]
    assert not rows(example)[0]


def test_private_cli_writes_text_and_versioned_manifest(tmp_path):
    source = tmp_path / "export.json"
    source.write_text(json.dumps(full_example()))
    output = tmp_path / "thinking"
    assert main(["--examples", str(source), "--output", str(output)]) == 0
    assert os.stat(output).st_mode & 0o777 == 0o700
    assert all(os.stat(path).st_mode & 0o777 == 0o600 for path in output.iterdir())
    text = (output / "train.jsonl").read_text() + (output / "eval.jsonl").read_text()
    assert "Inspect the retained page, then answer." in text
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["dataset_version"].startswith("sha256:")
    assert manifest["training_permission"]["status"] == "unconfirmed"


@pytest.mark.parametrize('delivery', ['selected_terminal', 'terminal_receipt'])
def test_chat_citation_annotations_are_readable_with_exact_order_and_source_path(delivery):
    # Schema observed in production Chat replies; all values are synthetic.
    example = full_example()
    call = example['provider_calls'][0]
    response = call['response']
    root = ['provider_calls', 0, 'response']
    if delivery == 'terminal_receipt':
        response = copy.deepcopy(response)
        call['response_receipts'][0]['response'] = response
        root = ['provider_calls', 0, 'response_receipts', 0, 'response']
    message = response['choices'][0]['message']
    annotations = [
        {'type': 'url_citation', 'url_citation': {
            'url': 'https://example.com/source-b', 'title': 'Second source',
            'start_index': 8, 'end_index': 11}},
        {'type': 'url_citation', 'url_citation': {
            'url': 'https://example.com/source-a', 'title': 'First source',
            'start_index': 12, 'end_index': 15}},
    ]
    message['content'] = 'Sources [1] [2].'
    message['annotations'] = annotations
    seal(example)
    original = copy.deepcopy(example)
    report = training_thinking.readable_trace(example)
    path = root + ['choices', 0, 'message', 'annotations']
    events = [event for event in report['calls'][0]['events']
              if event.get('source_path') == path]
    assert len(events) == 1
    event = events[0]
    assert event['kind'] == 'citation_annotations'
    assert event['delivery'] == delivery
    assert event['value'] == annotations
    assert event['value_sha256'] == contracts.document_hash(annotations)
    value = example
    for part in path:
        value = value[part]
    assert value == event['value']
    markdown = training_thinking.readable_trace_markdown(report)
    assert 'Sources [1] [2].' in markdown
    assert markdown.index('https://example.com/source-b') < markdown.index('https://example.com/source-a')
    assert 'Second source' in markdown and 'First source' in markdown
    assert contracts.canonical_json(path) in markdown
    assert example == original


@pytest.mark.parametrize('delivery', ['selected_terminal', 'terminal_receipt'])
def test_readable_chat_preserves_unindexed_metadata_without_repeating_reasoning(delivery):
    example = full_example()
    call = example['provider_calls'][0]
    response = call['response']
    root = ['provider_calls', 0, 'response']
    if delivery == 'terminal_receipt':
        response = copy.deepcopy(response)
        call['response_receipts'][0]['response'] = response
        root = ['provider_calls', 0, 'response_receipts', 0, 'response']
    choice = response['choices'][0]
    choice['index'] = 0
    choice['native_finish_reason'] = 'end_turn'
    message = choice['message']
    text = message['reasoning_content']
    message['reasoning'] = text
    message['reasoning_details'] = [
        {'type': 'reasoning.text', 'text': text, 'format': 'unknown', 'index': 2},
        {'type': 'summary_text', 'text': 'Ancillary summary.'}]
    message['refusal'] = None
    message['tool_calls'] = [{'id': 'tool-1', 'type': 'function',
                              'function': {'name': 'lookup', 'arguments': '{}'}}]
    response['usage'] = {'completion_tokens': 7}
    seal(example)
    original = copy.deepcopy(example)
    report = training_thinking.readable_trace(example)
    events = [event for event in report['calls'][0]['events']
              if event.get('delivery') == delivery and 'source_path' in event]
    values = {tuple(event['source_path']): event for event in events}
    base = root + ['choices', 0]
    expected = [
        (base + ['index'], 0),
        (base + ['finish_reason'], choice['finish_reason']),
        (base + ['message', 'role'], message['role']),
        (base + ['message', 'reasoning_details', 0, 'type'], 'reasoning.text'),
        (base + ['message', 'reasoning_details', 0, 'format'], 'unknown'),
        (base + ['message', 'reasoning_details', 0, 'index'], 2),
        (base + ['message', 'reasoning_details', 1, 'type'], 'summary_text'),
        (base + ['message', 'refusal'], None),
        (base + ['native_finish_reason'], 'end_turn'),
    ]
    for path, value in expected:
        assert tuple(path) in values
        assert values[tuple(path)]['value'] == value
    metadata_paths = [event['source_path'] for event in events
                      if event['kind'] == 'response_metadata' and event['source_path'][:len(base)] == base]
    assert metadata_paths == [path for path, _ in expected]
    reasoning = [event for event in events if event.get('value') == text]
    assert len(reasoning) == 1
    assert reasoning[0]['source_path'] == base + ['message', 'reasoning']
    assert reasoning[0]['alias_source_paths'] == [
        base + ['message', 'reasoning_content'], base + ['message', 'reasoning_details', 0, 'text']]
    preserved = [event['kind'] for event in events if event['kind'] != 'response_metadata']
    assert preserved == ['unverified_reasoning_text', 'unverified_reasoning_text',
                         'usage', 'assistant_output', 'tool_call']
    for event in events:
        source = example
        for key in event['source_path']:
            source = source[key]
        assert event['value'] == source
        assert event['value_sha256'] == contracts.document_hash(source)
    markdown = training_thinking.readable_trace_markdown(report)
    assert 'native_finish_reason' in markdown and 'end_turn' in markdown
    assert example == original
    assert report['projection_truncated'] is False
    assert report['capture_limits'] == {
        'request_max_bytes': training_capture.MAX_OBJECT_BYTES,
        'receipt_max_bytes': training_export.MAX_RECEIPT_BYTES}


def test_readable_generic_provider_output_does_not_drop_reserved_chat_fields():
    example = full_example()
    call = example['provider_calls'][0]
    call['operation_id'] = 'synthetic.tool'
    call['response'] = {'choices': [{'label': 'first'}],
                        'output': [{'result': 'Generic tool output.'}], 'usage': None}
    report = training_thinking.readable_trace(seal(example))
    events = [event for event in report['calls'][0]['events']
              if event.get('delivery') == 'selected_terminal']
    assert [(event['source_path'][3:], event['value']) for event in events] == [
        (['choices', 0, 'label'], 'first'),
        (['output', 0, 'result'], 'Generic tool output.'), (['usage'], None)]


def test_readable_cli_keeps_raw_bytes_and_renders_native_metadata(tmp_path):
    from scripts.render_arena_training_trace import main as render_trace

    example = full_example()
    example['provider_calls'][0]['response']['choices'][0]['native_finish_reason'] = 'end_turn'
    raw = (json.dumps(seal(example), indent=2) + '\n').encode()
    source = tmp_path / 'source.json'
    source.write_bytes(raw)
    output = tmp_path / 'rendered'
    assert render_trace(['--example', str(source), '--output', str(output)]) == 0
    assert (output / 'raw-export.json').read_bytes() == raw
    trace = json.loads((output / 'readable-trace.json').read_text())
    path = ['provider_calls', 0, 'response', 'choices', 0, 'native_finish_reason']
    event = next(event for event in trace['calls'][0]['events']
                 if event.get('source_path') == path)
    assert event['value'] == 'end_turn'
    assert event['value_sha256'] == contracts.document_hash('end_turn')
    markdown = (output / 'readable-trace.md').read_text()
    assert 'native_finish_reason' in markdown and 'end_turn' in markdown
    assert os.stat(output).st_mode & 0o777 == 0o700
    assert all(os.stat(path).st_mode & 0o777 == 0o600 for path in output.iterdir())


def test_build_does_not_retain_raw_episode_batch():
    import weakref

    class TrackedDict(dict):
        pass

    refs = []

    def sources():
        for index in range(4):
            # The consumer may still hold its current source while requesting
            # the next one, but earlier raw sources must already be released.
            assert all(ref() is None for ref in refs[:-1])
            example = TrackedDict(full_example())
            refs.append(weakref.ref(example))
            yield example
            del example

    expected = training_thinking.build([full_example()])
    assert training_thinking.build(sources()) == expected
    assert all(ref() is None for ref in refs)


def test_invalid_neighbor_with_claimed_hash_cannot_poison_valid_source():
    valid = full_example()
    invalid = copy.deepcopy(valid)
    invalid["provider_calls"][0]["response"]["choices"][0]["message"]["reasoning_content"] = "unbound tampered text"
    # Deliberately keep the valid source's claimed hash. No invalid evidence
    # can replace its bound trajectory, regardless of source order.
    expected, _ = training_thinking.build([valid])
    forward = training_thinking.build(iter([valid, invalid]))
    backward = training_thinking.build(iter([invalid, valid]))
    assert forward == backward
    for split in ("train", "eval"):
        assert forward[0][split] == expected[split]
    assert forward[0]["analysis"] == [{"source_example_hash": valid["example_hash"],
                                        "reason": "example_hash_mismatch"}]
    assert forward[1]["coverage"]["distinct_verified_exports"] == 1
