"""Future champion evidence checks are separate from scoring and publication."""

import copy
import json

import pytest

from lab_arena import code_review, contracts, training_eligibility, training_thinking
from tests.lab_arena.full_thinking_saved_probe_test import PROBES, _example
from tests.lab_arena.training_steps_test import seal


SOURCE_HASH = contracts.hash_bytes(b"checked source archive")


def submission():
    return {"submission_id": "sub", "source_archive_sha256": SOURCE_HASH,
            "code_review_status": "passed", "code_review_doc": {
                "passed": True, "research_control": {
                    "policy_id": code_review.RESEARCH_POLICY_ID,
                    "policy_hash": code_review.RESEARCH_POLICY_HASH,
                    "reviewer_version": code_review.RESEARCH_REVIEWER_VERSION,
                    "source_archive_sha256": SOURCE_HASH,
                    "classification": "model_directed",
                    "evidence_claims": ["observation_to_action", "finish_continue"],
                }}}


def captured_example():
    probe = json.loads(PROBES[0].read_text())
    example = _example(probe)
    example["role"] = "miner"
    example["source"] = {"sha256": SOURCE_HASH, "hash_origin": "recorded_archive_hash"}
    example["input_snapshots"] = {"run": {"input": {"icp": "saved input"}}}
    example["output"] = {"companies": []}
    example["judgment"] = {"per_icp_score": 0}
    example["round_status"] = "open"
    return seal(example)


def capture(example, row=None):
    row = row or {"submission_id": example["submission_id"],
                  "source_archive_sha256": SOURCE_HASH}
    return training_eligibility.assess_capture(row, {"run": example}, ["run"], enabled=True)


def refresh_response_links(example, index):
    call = example["provider_calls"][index]
    response = call["response"]
    response_hash = contracts.document_hash(response)
    wire = contracts.canonical_json(response).encode()
    wire_hash = contracts.hash_bytes(wire)
    call["response_sha256"] = response_hash
    receipt = call["response_receipts"][0]
    receipt.update(response_sha256=response_hash, wire_body_sha256=wire_hash,
                   wire_body_bytes=len(wire))
    payload = example["worker_observation_artifacts"]["run"]["payload"]
    for row in payload["response_receipts"] + payload["agent_observations"]:
        if row.get("call_identity") == call["call_identity"] or (row.get("data") or {}).get("call_identity") == call["call_identity"]:
            data = row.get("data", row)
            data.update(body_sha256=wire_hash, body_bytes=len(wire))
    payload_hash = contracts.document_hash(payload)
    example["worker_observation_artifacts"]["run"]["payload_sha256"] = payload_hash
    for item in example["provider_calls"]:
        provenance = item["response_provenance"]
        provenance["worker_receipt_ref"]["payload_sha256"] = payload_hash
        for ref in provenance["agent_receipt_refs"]:
            ref["payload_sha256"] = payload_hash
    seal(example)


def test_source_requires_bound_model_directed_review_but_legacy_is_disabled():
    row = submission()
    assert training_eligibility.assess_source(row, enabled=True)["eligible"] is True
    for change in ("source_archive_sha256", "classification", "policy_hash"):
        bad = copy.deepcopy(row)
        bad["code_review_doc"]["research_control"][change] = "wrong"
        assert training_eligibility.assess_source(bad, enabled=True)["eligible"] is False
    assert training_eligibility.assess_source({"code_review_status": "passed"}, enabled=True)["eligible"] is False
    assert training_eligibility.assess_source({}, enabled=False)["eligible"] is True


def test_selected_run_full_trace_allows_empty_tool_turn_without_dataset_filters():
    result = capture(captured_example())
    assert result == {"eligible": True, "reasons": [], "counts": {
        "enabled": 1, "selected_runs": 1, "checked_runs": 1,
        "model_calls": 3, "full_thinking_calls": 2, "tool_only_empty_calls": 1}}


@pytest.mark.parametrize("extra", ["summary", "opaque"])
def test_full_text_remains_eligible_with_additional_summary_or_opaque_continuation(extra):
    example = captured_example()
    example["round_status"] = "published"
    call = example["provider_calls"][2]
    if extra == "summary":
        call["response"]["choices"][0]["message"]["reasoning_details"].append(
            {"type": "reasoning.summary", "summary": "A separate short summary."})
        refresh_response_links(example, 2)
    else:
        call["response_privacy_redactions"] = ["hidden_reasoning"]
        call["response_receipts"][0]["privacy_redactions"] = ["hidden_reasoning"]
        seal(example)
    result = capture(example)
    assert result["eligible"] is True, result
    assert result["counts"]["full_thinking_calls"] == 2
    splits, manifest = training_thinking.build([example])
    records = splits["train"] + splits["eval"]
    assert len(records) == 2, manifest["counts"]
    assert all(len(record["generated_thinking"]) == 1 for record in records)
    assert all(all("summary" not in segment["field"] for segment in record["generated_thinking"])
               for record in records)
    if extra == "summary":
        assert "A separate short summary." not in json.dumps(records)
    else:
        assert all(record["source_refs"][0]["response_projection"]["opaque_reasoning_omitted"]
                   for record in records if record["source_refs"][0]["call_identity"] == call["call_identity"])
    assert manifest["coverage"]["generated_thinking_texts"] == 2
    assert manifest["coverage"]["trajectories"]["complete_full_thinking"] == 1
    trace = training_thinking.readable_trace(example)
    assert any(event["kind"] == "full_generated_thinking"
               for event in trace["calls"][2]["events"])


@pytest.mark.parametrize("damage", ["summary_only", "opaque_only", "unapproved_text",
                                    "summary_alias", "truncated", "receipt_mismatch",
                                    "route_unverified", "empty_route_unverified"])
def test_capture_and_strict_export_both_reject_unverified_full_text(damage):
    example = captured_example()
    example["round_status"] = "published"
    call = example["provider_calls"][2]
    message = call["response"]["choices"][0]["message"]
    if damage in {"summary_only", "opaque_only"}:
        message.pop("reasoning", None)
        message.pop("reasoning_content", None)
        message.pop("reasoning_details", None)
        if damage == "summary_only":
            message["reasoning_details"] = [{"type": "reasoning.summary", "summary": "Only a summary."}]
        else:
            call["response_privacy_redactions"] = ["hidden_reasoning"]
            call["response_receipts"][0]["privacy_redactions"] = ["hidden_reasoning"]
        refresh_response_links(example, 2)
    elif damage == "unapproved_text":
        message["reasoning_content"] = "Unapproved second text."
        refresh_response_links(example, 2)
    elif damage == "summary_alias":
        message["reasoning_details"].append(
            {"type": "reasoning.summary", "summary": message["reasoning"]})
        refresh_response_links(example, 2)
    elif damage == "truncated":
        call["response"]["choices"][0]["finish_reason"] = "length"
        refresh_response_links(example, 2)
    elif damage == "route_unverified":
        call["reasoning_capture"]["provider_route"] = "wrong-route"
    elif damage == "empty_route_unverified":
        example["provider_calls"][0]["reasoning_capture"]["provider_route"] = "wrong-route"
    else:
        call["response_receipts"][0]["response_sha256"] = contracts.hash_bytes(b"wrong receipt")
    seal(example)
    assert capture(example)["eligible"] is False, damage
    splits, _ = training_thinking.build([example])
    assert not splits["train"] and not splits["eval"], damage


@pytest.mark.parametrize("damage,reason", [
    (lambda e: e["completeness"].update(complete=False, incomplete_reasons=["worker_call_accounting_unverified"]),
     "trajectory_evidence_incomplete"),
    (lambda e: e["provider_calls"][2]["response_provenance"].update(agent_receipt="not_reported"),
     "selected_run_evidence_invalid"),
    (lambda e: e["provider_calls"][2]["reasoning_capture"].update(provider_route="wrong-route"),
     "full_thinking_route_unverified"),
    (lambda e: e["provider_calls"][0]["reasoning_capture"].update(provider_route="wrong-route"),
     "full_thinking_route_unverified"),
    (lambda e: e.pop("output"), "input_or_output_unavailable"),
    (lambda e: e["source"].update(sha256=contracts.hash_bytes(b"other")),
     "selected_run_identity_unverified"),
])
def test_missing_or_mismatched_capture_fails_eligibility_without_raising(damage, reason):
    example = captured_example()
    damage(example)
    result = capture(seal(example))
    assert result["eligible"] is False
    assert reason in result["reasons"]


def test_missing_selected_run_cannot_be_replaced_by_another_export():
    example = captured_example()
    result = training_eligibility.assess_capture(
        {"submission_id": example["submission_id"], "source_archive_sha256": SOURCE_HASH},
        {"run": example}, ["run", "other"], enabled=True)
    assert result["eligible"] is False
    assert "selected_run_export_missing" in result["reasons"]
    assert result["counts"]["checked_runs"] == 1


def test_readable_projection_must_retain_paths_to_raw_provider_text(monkeypatch):
    original = training_thinking.readable_trace

    def missing_text(example):
        report = original(example)
        for call in report["calls"]:
            call["events"] = [event for event in call["events"] if not event.get("reasoning_field")]
        return report

    monkeypatch.setattr(training_thinking, "readable_trace", missing_text)
    result = capture(captured_example())
    assert result["eligible"] is False
    assert "readable_full_thinking_unverified" in result["reasons"]


def test_baseline_uses_same_capture_check_with_verified_frozen_object_hash():
    example = captured_example()
    example["role"] = "baseline"
    example["source"]["hash_origin"] = "computed_from_frozen_object"
    row = {"submission_id": example["submission_id"], "is_king": True}
    assert capture(seal(example), row)["eligible"] is True
    example["source"]["hash_origin"] = "unverified"
    assert capture(seal(example), row)["eligible"] is False


def test_legacy_capture_gate_is_disabled():
    assert training_eligibility.assess_capture({}, {}, [], enabled=False) == {
        "eligible": True, "reasons": [], "counts": {"enabled": 0}}
