import copy
import json
import os

from lab_arena import contracts, training_capture, training_export, training_thinking
from scripts.prepare_arena_full_thinking_dataset import main
from tests.lab_arena.training_steps_test import fixture, response_fixture, seal


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
    example["completeness"] = {"complete": True, "incomplete_reasons": []}
    return seal(example)


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


def test_legitimate_tool_only_empty_turn_does_not_disqualify_later_text():
    example = full_example()
    extra = copy.deepcopy(example["provider_calls"][0])
    extra["call_identity"] = "tool-only"
    extra["action_sequence"] = 2
    extra["response"]["choices"][0]["message"].pop("reasoning_content")
    extra["response"]["choices"][0]["finish_reason"] = "tool_calls"
    extra["response_sha256"] = contracts.document_hash(extra["response"])
    extra["response_receipts"][0]["response_sha256"] = extra["response_sha256"]
    example["provider_calls"].append(extra)
    report = training_thinking.coverage([seal(example)])
    assert report["returned_classes"] == {"empty": 1, "full_thinking": 1}
    assert report["trajectories"]["eligible_without_disqualifying_capture"] == 1


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
    example["attempts"] = [{"run_id": "judge-run", "kind": "score"}]
    got, manifest = rows(example)
    assert len(got) == 1
    assert manifest["coverage"]["trajectories"]["complete_full_thinking"] == 1
    assert {row["role"] for row in manifest["coverage"]["attempts"]} == {"miner", "judge"}


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


def test_history_alias_retained_only_for_approved_assistant_replay():
    annotation = copy.deepcopy(full_example()["provider_calls"][0]["reasoning_capture"])
    document = {"reasoning_capture": annotation,
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
    assert cleaned["outbound"]["body"]["messages"][0]["reasoning_content"] == "[REDACTED]"


def test_missing_saved_response_is_storage_failed_without_fabricating_text():
    example = full_example()
    call = example["provider_calls"][0]
    call.pop("response")
    call["evidence_gaps"] = ["full_response_unavailable"]
    call["upstream_response_capture"] = {"status": "unavailable", "reason": "response_capture_failed"}
    row = training_thinking.coverage([seal(example)])["attempts"][0]
    assert row["storage"] == row["returned"] == "storage_failed"
    assert row["characters"] == 0


def test_truncated_and_opaque_attempts_are_separate_from_full_text():
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
    assert row["returned"] == "opaque"
    assert row["storage"] == "saved"
    assert not rows(opaque)[0]


def test_responses_full_content_qualifies_but_summary_does_not():
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
    example["completeness"] = {"complete": True, "incomplete_reasons": []}
    got, _ = rows(example)
    assert len(got) == 1
    assert got[0]["generated_thinking"][0]["text"] == "Check evidence first."
    call["response"]["output"][0]["summary"] = [{"type": "summary_text", "text": "Brief summary"}]
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
