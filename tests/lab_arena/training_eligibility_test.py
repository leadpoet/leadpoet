"""Future champion evidence checks are separate from scoring and publication."""

import copy
import json

import pytest

from lab_arena import code_review, contracts, training_eligibility, training_steps, training_thinking
from tests.lab_arena.full_thinking_saved_probe_test import PROBES, _example
from tests.lab_arena.training_steps_test import bind_selected_output, seal


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
    example["output"] = {"companies": []}
    example["judgment"] = {"per_icp_score": 0}
    example["round_status"] = "open"
    return bind_selected_output(example)


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
    return {f"run-{position}": bind_selected_output(rename(copy.deepcopy(original), f"run-{position}"))
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
    example["attempts"] = [{"run_id": "run", "kind": "execute", "status": "accepted"},
                           {"run_id": "score", "kind": "score", "status": "accepted"}]
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


def test_judge_only_request_and_response_privacy_preserves_selected_thinking():
    example = judge_only_gap_example()
    judge_call = example["provider_calls"][-1]
    judge_call["evidence_gaps"] = ["request_privacy_nonexact", "response_privacy_nonexact"]
    judge_call["request_capture"] = {"redacted": True, "redactions": ["credential"]}
    example["completeness"]["incomplete_reasons"] = ["request_privacy_nonexact", "response_privacy_nonexact"]
    example["completeness"]["reasons_by_dimension"]["research_process"] = list(
        example["completeness"]["incomplete_reasons"])
    seal(example)
    assert example["privacy"]["exact_replay"] is False
    assert capture(example)["eligible"] is True
    splits, _ = training_thinking.build([example])
    assert len(splits["train"]) + len(splits["eval"]) == 2


def test_judge_request_privacy_marker_without_redacted_request_is_rejected():
    example = judge_only_gap_example()
    judge_call = example["provider_calls"][-1]
    judge_call["evidence_gaps"] = ["request_privacy_nonexact", "response_privacy_nonexact"]
    example["completeness"]["incomplete_reasons"] = list(judge_call["evidence_gaps"])
    example["completeness"]["reasons_by_dimension"]["research_process"] = list(judge_call["evidence_gaps"])
    seal(example)
    assert capture(example)["eligible"] is True
    splits, _ = training_thinking.build([example])
    assert len(splits["train"]) + len(splits["eval"]) == 2


def test_failed_judge_attempt_privacy_is_attributed_without_hiding_other_runs():
    example = judge_only_gap_example()
    example["attempts"].append({"run_id": "failed-score", "kind": "score", "status": "failed"})
    example["provider_calls"].append({"run_id": "failed-score", "operation_id": "scrapingdog.scrape",
                                      "evidence_gaps": ["response_privacy_nonexact"]})
    seal(example)
    assert capture(example)["eligible"] is True
    splits, _ = training_thinking.build([example])
    assert len(splits["train"]) + len(splits["eval"]) == 2


def failed_score_gap_example():
    example = judge_only_gap_example()
    example["attempts"].append({"run_id": "failed-score", "kind": "score", "status": "failed"})
    for call in example["provider_calls"]:
        if call.get("run_id") == "run":
            call["response_provenance"]["selected_response_acknowledged"] = True
    example["provider_calls"].append({"run_id": "failed-score", "operation_id": "scrapingdog.scrape",
                                      "evidence_gaps": ["full_response_unavailable", "provider_cost_unresolved"]})
    failed = {"direct_web_usage_unverified", "full_response_unavailable",
              "provider_capture_count_unverified", "provider_cost_unresolved",
              "worker_local_refusals_unverified", "worker_observation_link_invalid"}
    reasons = failed | {"response_privacy_nonexact"}
    example["related_run_gap_attribution"] = {
        "run": {"kind": "execute", "status": "accepted", "research_process_gaps": [], "cost_gaps": []},
        "failed-score": {"kind": "score", "status": "failed",
                         "research_process_gaps": sorted(failed),
                         "cost_gaps": ["provider_capture_count_unverified", "provider_cost_unresolved"]},
        "score": {"kind": "score", "status": "accepted",
                  "research_process_gaps": ["response_privacy_nonexact"], "cost_gaps": []},
    }
    example["completeness"]["incomplete_reasons"] = sorted(reasons)
    example["completeness"]["reasons_by_dimension"]["research_process"] = sorted(reasons)
    example["completeness"]["reasons_by_dimension"]["costs"] = [
        "provider_capture_count_unverified", "provider_cost_unresolved"]
    example["completeness"]["dimensions"]["costs"] = False
    return seal(example)


def test_failed_score_hard_gaps_do_not_erase_exact_selected_research():
    example = failed_score_gap_example()
    assert example["completeness"]["complete"] is False
    assert example["privacy"]["exact_replay"] is False
    assert training_thinking.selected_research_evidence(example) == frozenset(
        example["completeness"]["incomplete_reasons"])
    assert capture(example)["eligible"] is True
    splits, _ = training_thinking.build([example])
    assert len(splits["train"]) + len(splits["eval"]) == 2


@pytest.mark.parametrize("damage", [
    lambda e: e["worker_observation_artifacts"]["run"].update(payload_sha256="0" * 64),
    lambda e: e["provider_calls"][0]["response_receipts"][0].update(wire_body_sha256="0" * 64),
    lambda e: e["provider_calls"][0]["request"].update(request_hash="0" * 64),
    lambda e: e["input_snapshots"]["score"].update(input_sha256="0" * 64),
    lambda e: e["output_identity"].update(document_hash="0" * 64),
])
def test_score_only_diagnostics_cannot_hide_broken_selected_proof(damage):
    example = failed_score_gap_example()
    damage(example)
    seal(example)
    assert capture(example)["eligible"] is False


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
    diagnostic_only = damage in {"prior_gap", "execution_cost", "selected_unsettled_cost", "unmapped_aggregate_gap", "selected_tool_response_redacted", "selected_tool_request_redacted"}
    assert capture(example)["eligible"] is diagnostic_only
    splits, _ = training_thinking.build([example])
    expected = 2 if diagnostic_only or damage in {"selected_gap", "selected_tool_response_redacted", "selected_tool_request_redacted"} else (1 if damage in {"route", "missing_thinking", "missing_ack"} else 0)
    assert len(splits["train"]) + len(splits["eval"]) == expected


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
    records = splits["train"] + splits["eval"]
    # Export each exact full target independently of ICP qualification.
    assert len(records) == (2 if damage == "empty_route_unverified" else 1), damage
    assert all(ref["call_identity"] != call["call_identity"] for record in records
               for ref in record["source_refs"]) if damage != "empty_route_unverified" else True


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
    if example["completeness"].get("incomplete_reasons") == ["worker_call_accounting_unverified"]:
        assert result["eligible"] is True  # Nonessential all-attempt accounting diagnostics.
    else:
        assert result["eligible"] is False
        assert ("trajectory_evidence_incomplete" if reason == "input_or_output_unavailable" else reason) in result["reasons"]


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
    damaged["provider_calls"][0]["response_receipts"][0]["privacy_redactions"] = ["credential"]
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
    result = assess_ten(examples, percent=100)
    assert result["eligible"] is False
    assert "full_thinking_route_unverified" in result["reasons"]


def test_privacy_gap_cannot_hide_wrong_route_at_eighty_percent():
    examples = ten_selected_examples()
    damaged = examples["run-9"]
    damaged["completeness"].update(complete=False, incomplete_reasons=["response_privacy_nonexact"])
    damaged["provider_calls"][0]["response_privacy_redactions"] = ["credential"]
    damaged["provider_calls"][0]["response_receipts"][0]["privacy_redactions"] = ["credential"]
    damaged["provider_calls"][2]["reasoning_capture"]["provider_route"] = "wrong-route"
    seal(damaged)
    result = assess_ten(examples, percent=100)
    assert result["eligible"] is False
    assert result["reasons"]


def selected_privacy_cascade_example():
    example = judge_only_gap_example()
    for call in example["provider_calls"]:
        if call.get("run_id") == "run":
            call["response_provenance"]["selected_response_acknowledged"] = True
    first, second = example["provider_calls"][:2]
    first["response_privacy_redactions"] = ["credential"]
    first["response_receipts"][0]["privacy_redactions"] = ["credential"]
    first["evidence_gaps"] = ["response_privacy_nonexact"]
    second["request_capture"].update(redacted=True, redactions=["credential"])
    second["request"]["privacy"] = {"redacted": True, "redactions": ["credential"]}
    second["request"]["hashed_parameters"] = {"message": "[redacted:credential]"}
    second["evidence_gaps"] = ["request_privacy_nonexact"]
    reasons = ["context_privacy_nonexact", "request_privacy_nonexact", "response_privacy_nonexact"]
    example["privacy"]["context_redactions"] = ["credential"]
    example["completeness"]["incomplete_reasons"] = reasons
    example["completeness"]["reasons_by_dimension"]["research_process"] = reasons
    return seal(example)


def test_eighty_percent_counts_documented_selected_privacy_cascade_as_incomplete():
    examples = ten_selected_examples()
    examples.pop("run-9")
    damaged = selected_privacy_cascade_example()
    examples["run"] = damaged
    assert training_thinking.selected_research_evidence(damaged) is not None
    splits, _ = training_thinking.build([damaged])
    assert len(splits["train"]) + len(splits["eval"]) == 1
    result = assess_ten(examples)
    assert result["eligible"] is True, result
    assert result["counts"]["complete_runs"] == 9
    assert result["counts"]["incomplete_runs"] == 1


@pytest.mark.parametrize("damage", [
    lambda e: e["provider_calls"][1]["request_capture"].update(redacted=False),
    lambda e: e["provider_calls"][1]["request_capture"].update(redactions=[]),
    lambda e: e["provider_calls"][0]["response_receipts"][0].update(privacy_redactions=[]),
    lambda e: e["provider_calls"][0]["response_provenance"].update(agent_receipt="not_reported"),
    lambda e: e["provider_calls"][0]["response_provenance"].update(selected_response_acknowledged=False),
    lambda e: e["provider_calls"][1]["request"].update(effective_operation_id="wrong.operation"),
    lambda e: e["provider_calls"][2]["reasoning_capture"].update(provider_route="wrong-route"),
    lambda e: e["worker_observation_artifacts"]["run"].update(payload_sha256="0" * 64),
    lambda e: e["output_identity"].update(document_hash="0" * 64),
    lambda e: e["input_snapshots"]["score"].update(input_sha256="0" * 64),
    lambda e: e["input_snapshots"]["score"]["input"].update(scored_run_id="other"),
    lambda e: e["provider_calls"][0]["response_receipts"][0].update(wire_body_sha256="0" * 64),
    lambda e: e["completeness"]["incomplete_reasons"].append("worker_observation_link_invalid"),
])
def test_selected_privacy_cascade_cannot_hide_broken_proof(damage):
    examples = ten_selected_examples()
    examples.pop("run-9")
    damaged = selected_privacy_cascade_example()
    damage(damaged)
    examples["run"] = seal(damaged)
    result = assess_ten(examples, percent=100)
    assert result["eligible"] is False
    assert result["reasons"]


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


def test_exact_truncated_provider_reply_counts_as_one_incomplete_icp():
    examples = ten_selected_examples()
    damaged = examples["run-9"]
    damaged["provider_calls"][2]["response"]["choices"][0]["finish_reason"] = "length"
    refresh_response_links(damaged, 2)
    seal(damaged)
    result = assess_ten(examples)
    assert result["eligible"] is True, result
    assert result["counts"]["complete_runs"] == 9
    assert result["counts"]["required_complete_runs"] == 8
    strict, _ = training_thinking.build(
        list(examples.values()), training_steps.Rules(production_rounds_only=False))
    strict_runs = {ref["run_id"] for split in ("train", "eval")
                   for record in strict[split] for ref in record["source_refs"]}
    assert strict_runs == set(examples)
    assert assess_ten(examples, percent=100)["eligible"] is False
    examples["run-8"] = None
    examples["run-7"] = None
    assert "training_coverage_incomplete" in assess_ten(examples)["reasons"]


def test_all_exact_truncated_replies_count_as_one_incomplete_icp():
    examples = ten_selected_examples()
    damaged = examples["run-9"]
    for index in (0, 1, 2):
        damaged["provider_calls"][index]["response"]["choices"][0]["finish_reason"] = "length"
        refresh_response_links(damaged, index)
    result = assess_ten(examples)
    assert result["eligible"] is True, result
    assert result["counts"]["complete_runs"] == 9
    strict, _ = training_thinking.build(
        list(examples.values()), training_steps.Rules(production_rounds_only=False))
    strict_runs = {ref["run_id"] for split in ("train", "eval")
                   for record in strict[split] for ref in record["source_refs"]}
    assert strict_runs == set(examples) - {"run-9"}
    assert assess_ten(examples, percent=100)["eligible"] is False


@pytest.mark.parametrize("damage", ["route", "receipt", "source"])
def test_all_truncated_replies_still_require_route_receipt_and_source(damage):
    examples = ten_selected_examples()
    damaged = examples["run-9"]
    for index in (0, 1, 2):
        damaged["provider_calls"][index]["response"]["choices"][0]["finish_reason"] = "length"
        refresh_response_links(damaged, index)
    if damage == "route":
        damaged["provider_calls"][1]["reasoning_capture"]["provider_route"] = "wrong-route"
    elif damage == "receipt":
        damaged["provider_calls"][1]["response_receipts"][0]["response_sha256"] = contracts.hash_bytes(b"wrong")
    else:
        damaged["source"]["sha256"] = contracts.hash_bytes(b"wrong archive")
    seal(damaged)
    assert assess_ten(examples, percent=100)["eligible"] is False


def test_truncated_reply_with_wrong_route_remains_ineligible():
    examples = ten_selected_examples()
    damaged = examples["run-9"]
    damaged["provider_calls"][2]["response"]["choices"][0]["finish_reason"] = "length"
    refresh_response_links(damaged, 2)
    damaged["provider_calls"][2]["reasoning_capture"]["provider_route"] = "wrong-route"
    seal(damaged)
    assert assess_ten(examples, percent=100)["eligible"] is False


def test_truncated_reply_with_broken_receipt_remains_ineligible():
    examples = ten_selected_examples()
    damaged = examples["run-9"]
    damaged["provider_calls"][2]["response"]["choices"][0]["finish_reason"] = "length"
    refresh_response_links(damaged, 2)
    damaged["provider_calls"][2]["response_receipts"][0]["response_sha256"] = (
        contracts.hash_bytes(b"different response"))
    seal(damaged)
    assert assess_ten(examples, percent=100)["eligible"] is False


def test_eighty_percent_does_not_accept_unproved_thinking_as_empty():
    examples = ten_selected_examples()
    damaged = examples["run-9"]
    message = damaged["provider_calls"][2]["response"]["choices"][0]["message"]
    message["reasoning_content"] = "Unapproved second text."
    refresh_response_links(damaged, 2)
    result = assess_ten(examples, percent=100)
    assert result["eligible"] is False
    assert "full_thinking_unavailable" in result["reasons"]


def _unsettled_selected_call(example, *, privacy=False):
    call = example["provider_calls"][0]
    call["accounting_state"] = "uncertain"
    call["actual_microusd"] = None
    call["evidence_gaps"] = ["provider_cost_unresolved"]
    reasons = ["provider_cost_unresolved"]
    if privacy:
        call["response_privacy_redactions"] = ["credential"]
        call["response_receipts"][0]["privacy_redactions"] = ["credential"]
        call["evidence_gaps"].append("response_privacy_nonexact")
        reasons.append("response_privacy_nonexact")
    artifact = example["worker_observation_artifacts"][example["run_id"]]
    artifact["payload"]["complete"] = True
    artifact["payload_sha256"] = contracts.document_hash(artifact["payload"])
    for item in example["provider_calls"]:
        provenance = item.get("response_provenance") or {}
        if provenance.get("worker_receipt_ref"):
            provenance["worker_receipt_ref"]["payload_sha256"] = artifact["payload_sha256"]
        for ref in provenance.get("agent_receipt_refs") or []:
            ref["payload_sha256"] = artifact["payload_sha256"]
    score_id = "score-" + example["run_id"]
    example["score_run_id"] = score_id
    example["output_identity"] = {"document_hash": contracts.document_hash(example["output"])}
    example["attempts"] = [{"run_id": example["run_id"], "kind": "execute", "status": "accepted"},
                           {"run_id": score_id, "kind": "score", "status": "accepted"}]
    for identity, document in (
        (example["run_id"], example["input_snapshots"][example["run_id"]]["input"]),
        (score_id, {"scored_run_id": example["run_id"],
                    "companies": example["output"]["companies"]}),
    ):
        example["input_snapshots"][identity] = {
            "input": document, "input_sha256": contracts.document_hash(document),
            "privacy": {"redacted": False, "redactions": []},
        }
    example["configuration"]["sourcing_cost_eligibility_policy"] = (
        contracts.FLAT_PER_ICP_SUCCESSFUL_CALLS_COST_POLICY)
    example["competition_cost_eligibility"] = {
        "status": "unavailable", "reason": "provider_cost_uncertain",
        "source_rpc": "lab_arena_icp_cost_eligibility",
        "policy": contracts.FLAT_PER_ICP_SUCCESSFUL_CALLS_COST_POLICY,
        "observed_at": "2026-10-24T00:00:00Z",
    }
    reasons.append("competition_cost_eligibility_unavailable")
    example["completeness"] = {
        "complete": False, "incomplete_reasons": sorted(reasons),
        "dimensions": {"context": True, "source_identity": True, "exact_inputs": True,
                       "input_output_label": True, "costs": False, "research_process": False},
        "reasons_by_dimension": {"context": [], "source_identity": [], "exact_inputs": [],
                                 "input_output_label": [], "costs": sorted(
                                     ["provider_cost_unresolved", "competition_cost_eligibility_unavailable"]),
                                 "research_process": sorted(set(reasons) - {"competition_cost_eligibility_unavailable"})},
    }
    return seal(example)


@pytest.mark.parametrize("privacy", [False, True])
def test_eighty_percent_tolerates_two_attributable_unsettled_calls(privacy):
    examples = ten_selected_examples()
    for run_id in ("run-8", "run-9"):
        _unsettled_selected_call(examples[run_id], privacy=privacy)
    result = assess_ten(examples)
    assert result["eligible"] is True, result
    assert result["counts"]["complete_runs"] == (8 if privacy else 10)
    assert result["counts"]["incomplete_runs"] == (2 if privacy else 0)
    assert assess_ten(examples, percent=100)["eligible"] is (not privacy)
    assert training_thinking.selected_research_evidence(examples["run-8"]) is not None
    splits, _ = training_thinking.build([examples["run-8"]], training_steps.Rules(production_rounds_only=False))
    assert len(splits["train"]) + len(splits["eval"]) == 2
    _unsettled_selected_call(examples["run-7"], privacy=privacy)
    assert assess_ten(examples)["eligible"] is (not privacy)


@pytest.mark.parametrize("damage", [
    lambda e: e["competition_cost_eligibility"].update(reason="quality_or_attempt_not_final"),
    lambda e: e["competition_cost_eligibility"].update(source_rpc="untrusted_source"),
    lambda e: e["competition_cost_eligibility"].update(policy="untrusted_policy"),
    lambda e: e["competition_cost_eligibility"].update(observed_at=""),
    lambda e: e["competition_cost_eligibility"].update(status="available"),
    lambda e: e["completeness"]["reasons_by_dimension"]["costs"].remove(
        "competition_cost_eligibility_unavailable"),
    lambda e: e["provider_calls"][0]["evidence_gaps"].append(
        "competition_cost_eligibility_unavailable"),
])
def test_unsettled_cost_snapshot_requires_exact_authority_and_proof(damage):
    examples = ten_selected_examples()
    damaged = _unsettled_selected_call(examples["run-9"])
    damage(damaged)
    seal(damaged)
    assert assess_ten(examples)["eligible"] is True
    selected = training_thinking.selected_research_evidence(damaged)
    assert selected is not None


@pytest.mark.parametrize("damage", [
    lambda e: e["provider_calls"][0].update(accounting_state="dispatch"),
    lambda e: e["provider_calls"][0].update(actual_microusd=1),
    lambda e: e["provider_calls"][0]["response_provenance"].update(agent_receipt="not_reported"),
    lambda e: e["provider_calls"][0]["reasoning_capture"].update(provider_route="wrong-route"),
    lambda e: e["provider_calls"][0]["request"].update(operation_id="exa.search"),
    lambda e: e["provider_calls"][0]["response_receipts"][0].update(wire_body_sha256="sha256:" + "0" * 64),
])
def test_unsettled_cost_cannot_mask_invalid_accounting_or_provenance(damage):
    examples = ten_selected_examples()
    damaged = _unsettled_selected_call(examples["run-9"])
    damage(damaged)
    seal(damaged)
    result = assess_ten(examples, percent=100)
    call = damaged["provider_calls"][0]
    only_billing_changed = (call["accounting_state"] == "dispatch" or call["actual_microusd"] == 1)
    assert result["eligible"] is only_billing_changed, result


def test_unsettled_cost_with_unsubstantiated_privacy_label_is_hard_failure():
    examples = ten_selected_examples()
    damaged = _unsettled_selected_call(examples["run-9"], privacy=True)
    damaged["provider_calls"][0]["response_receipts"][0]["privacy_redactions"] = []
    seal(damaged)
    assert assess_ten(examples, percent=100)["eligible"] is False


def test_unsettled_cost_cannot_hide_behind_complete_flag():
    examples = ten_selected_examples()
    damaged = _unsettled_selected_call(examples["run-9"])
    damaged["completeness"] = {"complete": True, "incomplete_reasons": []}
    seal(damaged)
    assert assess_ten(examples, percent=100)["eligible"] is False
    assert training_thinking.selected_research_evidence(damaged) is None
    splits, _ = training_thinking.build([damaged], training_steps.Rules(production_rounds_only=False))
    assert not splits["train"] and not splits["eval"]


def test_unsettled_cost_and_exact_length_stop_count_one_incomplete_icp():
    examples = ten_selected_examples()
    damaged = examples["run-9"]
    damaged["provider_calls"][0]["response"]["choices"][0]["finish_reason"] = "length"
    refresh_response_links(damaged, 0)
    _unsettled_selected_call(damaged)
    result = assess_ten(examples)
    assert result["eligible"] is True, result
    assert result["counts"]["complete_runs"] == 9
    assert result["counts"]["incomplete_runs"] == 1


def _add_settled_exa_call(example):
    """A synthetic Exa request with its own bound worker write and read."""
    from copy import deepcopy
    call = deepcopy(example["provider_calls"][1])
    original_identity = call["call_identity"]
    call.update(operation_id="exa.search", provider="deepline",
                call_identity="settled-exa", action_sequence=9)
    request = call["request"]
    request["operation_id"] = "exa.search"
    request["hashed_parameters"] = {"query": "synthetic company"}
    call["request_hash"] = request["request_hash"] = contracts.document_hash(request["hashed_parameters"])
    body = {"operation": "exa_search", "query": "synthetic company"}
    request["outbound"] = {"method": "POST", "host": "code.deepline.com",
                           "path": "/api/v2/integrations/exa_search/execute", "body": body,
                           "body_sha256": contracts.document_hash(body)}
    call["response"] = {"results": []}
    digest = contracts.document_hash(call["response"])
    wire = contracts.canonical_json(call["response"]).encode()
    wire_hash = contracts.hash_bytes(wire)
    call["response_sha256"] = digest
    call["terminal_entry_id"] = 999
    receipt = call["response_receipts"][0]
    receipt.update(entry_id=999, response_ref={"field": "response", "entry_id": 999},
                   response_sha256=digest, wire_body_sha256=wire_hash,
                   wire_body_bytes=len(wire))
    provenance = call["response_provenance"]
    provenance["worker_received_receipt_entry_ids"] = [999]
    provenance["worker_receipt_ref"]["ordinal"] = 99
    provenance["agent_receipt_refs"][0]["id"] = "ack-exa"
    artifact = example["worker_observation_artifacts"][example["run_id"]]
    payload = artifact["payload"]
    source_write = next(row for row in payload["response_receipts"]
                        if row["call_identity"] == original_identity)
    write = deepcopy(source_write)
    write.update(ordinal=99, call_identity="settled-exa", operation_id="exa.search",
                 action_sequence=9, body_sha256=wire_hash, body_bytes=len(wire))
    payload["response_receipts"].append(write)
    source_read = next(row for row in payload["agent_observations"]
                       if row.get("kind") == "response_received"
                       and (row.get("data") or {}).get("call_identity") == original_identity)
    read = deepcopy(source_read)
    read["id"] = "ack-exa"
    read["data"].update(call_identity="settled-exa", body_sha256=wire_hash,
                        body_bytes=len(wire))
    payload["agent_observations"].append(read)
    example["provider_calls"].append(call)
    artifact["payload_sha256"] = contracts.document_hash(payload)
    for item in example["provider_calls"]:
        refs = item.get("response_provenance") or {}
        if refs.get("worker_receipt_ref"):
            refs["worker_receipt_ref"]["payload_sha256"] = artifact["payload_sha256"]
        for ref in refs.get("agent_receipt_refs") or []:
            ref["payload_sha256"] = artifact["payload_sha256"]
    return seal(example)


@pytest.mark.parametrize("field", ["request_hash", "response_sha256", "response_receipts"])
def test_unsettled_cost_cannot_hide_broken_settled_exa(field):
    examples = ten_selected_examples()
    damaged = _add_settled_exa_call(_unsettled_selected_call(examples["run-9"]))
    assert assess_ten(examples, percent=100)["eligible"] is True
    tool = damaged["provider_calls"][-1]
    tool[field] = [] if field == "response_receipts" else "sha256:" + "0" * 64
    seal(damaged)
    # Unused web receipts are not the model's required tool input history.
    assert assess_ten(examples, percent=100)["eligible"] is True


@pytest.mark.parametrize("field", ["request_hash", "response_sha256", "response_receipts"])
def test_unsettled_cost_cannot_hide_broken_settled_model(field):
    examples = ten_selected_examples()
    damaged = _unsettled_selected_call(examples["run-9"])
    model = damaged["provider_calls"][1]
    model[field] = [] if field == "response_receipts" else "sha256:" + "0" * 64
    seal(damaged)
    assert assess_ten(examples, percent=100)["eligible"] is False


@pytest.mark.parametrize("damage", [
    lambda e: e["output_identity"].update(document_hash="sha256:" + "0" * 64),
    lambda e: e.update(score_run_id="wrong-score"),
    lambda e: e["input_snapshots"].pop(e["run_id"]),
    lambda e: e["input_snapshots"][e["score_run_id"]]["input"].update(scored_run_id="wrong-run"),
])
def test_unsettled_cost_cannot_hide_unbound_score_or_output(damage):
    examples = ten_selected_examples()
    damaged = _unsettled_selected_call(examples["run-9"])
    damage(damaged)
    seal(damaged)
    assert assess_ten(examples, percent=100)["eligible"] is False


def test_readable_projection_must_retain_paths_to_raw_provider_text(monkeypatch):
    original = training_thinking.readable_trace

    def missing_text(example, **kwargs):
        report = original(example, **kwargs)
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
