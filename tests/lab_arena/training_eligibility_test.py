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


def ten_selected_examples():
    """Distinct sealed run identities with the same valid fixture evidence."""
    def rename(value, run_id):
        if isinstance(value, dict):
            return {run_id if key == "run" else key: rename(item, run_id)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [rename(item, run_id) for item in value]
        return run_id if value == "run" else value

    original = captured_example()
    return {f"run-{position}": seal(rename(copy.deepcopy(original), f"run-{position}"))
            for position in range(10)}


def assess_ten(examples, *, percent=80):
    return training_eligibility.assess_capture(
        submission(), examples, list(examples), enabled=True,
        minimum_coverage_percent=percent)


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
    payload = example["worker_observation_artifacts"][example["run_id"]]["payload"]
    for row in payload["response_receipts"] + payload["agent_observations"]:
        if row.get("call_identity") == call["call_identity"] or (row.get("data") or {}).get("call_identity") == call["call_identity"]:
            data = row.get("data", row)
            data.update(body_sha256=wire_hash, body_bytes=len(wire))
    payload_hash = contracts.document_hash(payload)
    example["worker_observation_artifacts"][example["run_id"]]["payload_sha256"] = payload_hash
    for item in example["provider_calls"]:
        if "response_provenance" not in item:
            continue
        provenance = item["response_provenance"]
        provenance["worker_receipt_ref"]["payload_sha256"] = payload_hash
        for ref in provenance["agent_receipt_refs"]:
            ref["payload_sha256"] = payload_hash
    seal(example)


def judge_only_gap_example(*, cost_gap=False):
    example = captured_example()
    example["round_status"] = "published"
    example["score_run_id"] = "score"
    example["output_identity"] = {"document_hash": contracts.document_hash(example["output"])}
    example["attempts"] = [{"run_id": "score", "kind": "score", "status": "accepted"}]
    for run_id, document in (("run", example["input_snapshots"]["run"]["input"]),
                             ("score", {"scored_run_id": "run", "companies": []})):
        example["input_snapshots"][run_id] = {
            "input": document, "input_sha256": contracts.document_hash(document),
            "privacy": {"redacted": False, "redactions": []}}
    example["competition_cost_eligibility"] = {"status": "available", "eligible": True}
    example["privacy"] = {"exact_replay": False, "privacy_filtered": True}
    example["worker_observation_artifacts"]["run"]["payload"]["complete"] = True
    example["worker_observation_artifacts"]["run"]["payload_sha256"] = contracts.document_hash(
        example["worker_observation_artifacts"]["run"]["payload"])
    for call in example["provider_calls"]:
        provenance = call["response_provenance"]
        provenance["worker_receipt_ref"]["payload_sha256"] = example["worker_observation_artifacts"]["run"]["payload_sha256"]
        for ref in provenance["agent_receipt_refs"]:
            ref["payload_sha256"] = example["worker_observation_artifacts"]["run"]["payload_sha256"]
    reasons = ["response_privacy_nonexact"]
    if cost_gap:
        reasons.append("provider_cost_unresolved")
    example["provider_calls"].append({
        "run_id": "score", "operation_id": "scrapingdog.scrape", "provider": "scrapingdog",
        "evidence_gaps": reasons, "response_privacy_redactions": ["credential"],
    })
    example["completeness"] = {
        "complete": False, "incomplete_reasons": sorted(reasons),
        "dimensions": {"context": True, "source_identity": True, "exact_inputs": True,
                       "input_output_label": True, "costs": not cost_gap, "research_process": False},
        "reasons_by_dimension": {"context": [], "source_identity": [], "exact_inputs": [],
                                 "input_output_label": [], "costs": ["provider_cost_unresolved"] if cost_gap else [],
                                 "research_process": sorted(reasons)},
    }
    return seal(example)


@pytest.mark.parametrize("cost_gap", [False, True])
def test_judge_only_privacy_or_uncertain_cost_does_not_erase_selected_thinking(cost_gap):
    example = judge_only_gap_example(cost_gap=cost_gap)
    assert example["completeness"]["complete"] is False
    assert example["privacy"]["exact_replay"] is False
    assert capture(example)["eligible"] is True
    splits, _ = training_thinking.build([example])
    assert len(splits["train"]) + len(splits["eval"]) == 2
    source = (splits["train"] + splits["eval"])[0]["source_refs"][0]
    assert source["source_quality"]["complete"] is False
    assert source["selected_research_evidence"] == {
        "complete": True,
        "ancillary_judge_gaps": sorted(example["completeness"]["incomplete_reasons"]),
    }


def test_failed_judge_attempt_privacy_is_attributed_without_hiding_other_runs():
    example = judge_only_gap_example()
    example["attempts"].append({"run_id": "failed-score", "kind": "score", "status": "failed"})
    example["provider_calls"].append({"run_id": "failed-score", "operation_id": "scrapingdog.scrape",
                                      "evidence_gaps": ["response_privacy_nonexact"]})
    seal(example)
    assert capture(example)["eligible"] is True
    splits, _ = training_thinking.build([example])
    assert len(splits["train"]) + len(splits["eval"]) == 2


def test_readiness_keeps_aggregate_gap_visible_alongside_selected_proof():
    from lab_arena import training_readiness

    report = training_readiness.build([judge_only_gap_example()], production_rounds_only=False)
    assert report["capture"]["executions_passing_all_export_dimensions"] == 0
    assert report["capture"]["selected_research_complete"] == 1
    source = report["sources"][0]["capture"]
    assert source["status"] == "incomplete"
    assert source["selected_research_complete"] is True
    assert source["ancillary_judge_gaps"] == ["response_privacy_nonexact"]
    assert report["full_thinking"]["candidate_records"] == 2


@pytest.mark.parametrize("damage", ["selected_gap", "prior_gap", "judge_input", "score_binding",
                                     "score_companies", "execution_cost", "selected_unsettled_cost",
                                     "missing_ack", "output_identity", "route", "missing_thinking",
                                     "selected_tool_response_redacted", "selected_tool_request_redacted",
                                     "unmapped_aggregate_gap"])
def test_ancillary_gap_never_masks_missing_selected_or_judgment_evidence(damage):
    example = judge_only_gap_example()
    if damage == "selected_gap":
        example["provider_calls"][0]["evidence_gaps"] = ["response_privacy_nonexact"]
    elif damage == "prior_gap":
        example["provider_calls"].append({"run_id": "prior", "operation_id": "scrapingdog.scrape",
                                          "evidence_gaps": ["response_privacy_nonexact"]})
    elif damage == "judge_input":
        example["input_snapshots"]["score"]["privacy"]["redacted"] = True
    elif damage == "score_binding":
        example["input_snapshots"]["score"]["input"]["scored_run_id"] = "other"
    elif damage == "score_companies":
        document = example["input_snapshots"]["score"]["input"]
        document["companies"] = [{"company": "Different"}]
        example["input_snapshots"]["score"]["input_sha256"] = contracts.document_hash(document)
    elif damage == "execution_cost":
        example["competition_cost_eligibility"]["status"] = "unavailable"
    elif damage == "selected_unsettled_cost":
        example["provider_calls"][0]["accounting_state"] = "reserved"
    elif damage == "output_identity":
        example["output_identity"]["document_hash"] = contracts.hash_bytes(b"wrong")
    elif damage == "route":
        example["provider_calls"][2]["reasoning_capture"]["provider_route"] = "wrong-route"
    elif damage == "missing_thinking":
        message = example["provider_calls"][2]["response"]["choices"][0]["message"]
        message.pop("reasoning", None)
        message.pop("reasoning_content", None)
        message.pop("reasoning_details", None)
        refresh_response_links(example, 2)
    elif damage in {"selected_tool_response_redacted", "selected_tool_request_redacted"}:
        example["provider_calls"].append({
            "run_id": "run", "operation_id": "exa.search", "provider": "exa",
            "evidence_gaps": [], "request_capture": {
                "redacted": damage == "selected_tool_request_redacted"},
            "response_privacy_redactions": ["credential"] if damage == "selected_tool_response_redacted" else [],
            "response_receipts": [], "accounting_state": "settlement", "actual_microusd": 1,
        })
    elif damage == "unmapped_aggregate_gap":
        proof = example["completeness"]
        proof["incomplete_reasons"].append("provider_cost_unresolved")
        proof["reasons_by_dimension"]["research_process"].append("provider_cost_unresolved")
        proof["reasons_by_dimension"]["costs"].append("provider_cost_unresolved")
        proof["dimensions"]["costs"] = False
    else:
        example["provider_calls"][2]["response_provenance"]["agent_receipt"] = "not_reported"
    seal(example)
    assert capture(example)["eligible"] is False
    splits, _ = training_thinking.build([example])
    assert not splits["train"] and not splits["eval"]


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


def test_nine_of_ten_exact_runs_allow_one_private_or_missing_export():
    examples = ten_selected_examples()
    damaged = examples["run-9"]
    damaged["completeness"].update(complete=False, incomplete_reasons=["response_privacy_nonexact"])
    damaged["provider_calls"][0]["response_privacy_redactions"] = ["credential"]
    seal(damaged)
    result = assess_ten(examples)
    assert result["eligible"] is True, result
    assert result["counts"]["complete_runs"] == 9
    assert result["counts"]["required_complete_runs"] == 8
    examples["run-8"] = None
    result = assess_ten(examples)
    assert result["eligible"] is True, result
    assert result["counts"]["complete_runs"] == 8
    examples["run-7"] = None
    result = assess_ten(examples)
    assert result["eligible"] is False
    assert "training_coverage_incomplete" in result["reasons"]


def test_nine_of_ten_keeps_route_tamper_a_hard_failure():
    examples = ten_selected_examples()
    examples["run-9"]["provider_calls"][2]["reasoning_capture"]["provider_route"] = "wrong-route"
    seal(examples["run-9"])
    result = assess_ten(examples)
    assert result["eligible"] is False
    assert "full_thinking_route_unverified" in result["reasons"]


def test_privacy_gap_cannot_hide_wrong_route_at_eighty_percent():
    examples = ten_selected_examples()
    damaged = examples["run-9"]
    damaged["completeness"].update(complete=False, incomplete_reasons=["response_privacy_nonexact"])
    damaged["provider_calls"][0]["response_privacy_redactions"] = ["credential"]
    damaged["provider_calls"][2]["reasoning_capture"]["provider_route"] = "wrong-route"
    seal(damaged)
    result = assess_ten(examples)
    assert result["eligible"] is False
    assert "trajectory_evidence_incomplete" in result["reasons"]


def test_nine_of_ten_allow_one_empty_thinking_run():
    examples = ten_selected_examples()
    damaged = examples["run-9"]
    for index in (1, 2):
        message = damaged["provider_calls"][index]["response"]["choices"][0]["message"]
        for field in ("reasoning", "reasoning_content", "reasoning_details"):
            message.pop(field, None)
        refresh_response_links(damaged, index)
    result = assess_ten(examples)
    assert result["eligible"] is True, result
    assert result["counts"]["complete_runs"] == 9


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
