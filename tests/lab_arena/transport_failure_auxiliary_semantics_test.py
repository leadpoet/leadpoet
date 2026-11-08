"""More private failed-stream evidence must not change native target semantics."""
from copy import deepcopy
import json

import pytest

from lab_arena import broker, contracts, training_steps, training_thinking
from lab_arena.training_export import _calls
from tests.lab_arena.training_eligibility_test import (
    assess_ten, capture, captured_example, refresh_response_links,
)
from tests.lab_arena.training_provider_output_coverage_test import bank
from tests.lab_arena.training_steps_test import seal
from tests.lab_arena.training_unavailable_upstream_coverage_test import unavailable_upstream
from tests.lab_arena.training_export_test import add_call, fixture


RESPONSES = [
    {"status": "completed", "output": [{"type": "reasoning", "content": [
        {"type": "reasoning_text", "text": "Undelivered full reasoning."}]}],
     "usage": {"output_tokens_details": {"reasoning_tokens": 12}}},
    {"status": "completed", "output": [], "usage": {"output_tokens_details": {"reasoning_tokens": 0}}},
    {"status": "completed", "output": [{"type": "reasoning", "content": [
        {"type": "reasoning_text", "text": " "}]}],
     "usage": {"output_tokens_details": {"reasoning_tokens": 1}}},
    {"status": "completed", "output": [{"type": "reasoning", "summary": [
        {"type": "summary_text", "text": "Undelivered summary."}]}],
     "usage": {"output_tokens_details": {"reasoning_tokens": 4}}},
    {"status": "completed", "output": [{"type": "reasoning", "encrypted_content": "fixture-opaque"}],
     "api_key": "fixture-other-key", "usage": {"output_tokens_details": {"reasoning_tokens": 1}}},
]


def with_auxiliary(response=RESPONSES[0], *, generation=False, retained_document=None):
    example = unavailable_upstream()
    call = example["provider_calls"][0]
    values = fixture()
    add_call(values, status="uncertain", operation=call["operation_id"])
    reserve, dispatch, terminal = values["ledger_rows"]
    for row in (reserve, dispatch, terminal):
        row.update(run_id=call["run_id"], call_identity=call["call_identity"])
    reserve["entry_doc"]["request_hash"] = call["request_hash"]
    terminal.update(entry_id=call["terminal_entry_id"], terminal_response={})
    receipt = broker._capture_upstream_response(
        broker.ProviderResponse(200, {}, json.dumps(response).encode()), operation_id=call["operation_id"],
        request_hash=call["request_hash"], call_identity=call["call_identity"], call_succeeded=False)
    authority = {"reason": "missing_provider_cost" if generation else "transport_failure",
                 "call_succeeded": False, "transport_error_class": "ReadTimeout",
                 "observed_provider_status": 200, "upstream_response_receipt": receipt}
    if generation:
        authority["transport_failure"] = True
    if retained_document is not None:
        authority = deepcopy(retained_document)
        authority["upstream_response_receipt"].update(call_identity=call["call_identity"],
            operation_id=call["operation_id"], request_hash=call["request_hash"])
    terminal["entry_doc"] = {"call": authority}
    exported, _ = _calls(values["ledger_rows"], {call["run_id"]}, values["objects"], {})
    call["upstream_response_receipts"] = exported[0]["upstream_response_receipts"]
    if "upstream_response_privacy_nonexact" in exported[0]["evidence_gaps"]:
        call["evidence_gaps"].append("upstream_response_privacy_nonexact")
        example["completeness"]["incomplete_reasons"].append("upstream_response_privacy_nonexact")
        example["completeness"]["reasons_by_dimension"]["research_process"].append("upstream_response_privacy_nonexact")
        example.setdefault("privacy", {}).update(privacy_filtered=True, hidden_reasoning_removed=True, exact_replay=False)
    call["lifecycle"][-1]["accounting"] = exported[0]["lifecycle"][-1]["accounting"]
    return seal(example)


@pytest.mark.parametrize("response", RESPONSES, ids=["full", "zero", "positive_blank", "summary", "redacted"])
@pytest.mark.parametrize("generation", [False, True])
def test_native_failed_attempt_exemption_and_targets_ignore_bound_undelivered_auxiliary(response, generation):
    original = unavailable_upstream()
    example = with_auxiliary(response, generation=generation)
    call = example["provider_calls"][0]
    assert training_thinking.verified_failed_attempt(example, call)
    assert "response" not in call and "response_status" not in call and "response_sha256" not in call
    assert call["actual_microusd"] is None and call["accounting_state"] == "uncertain"
    before = assess_ten(bank(original))
    after = assess_ten(bank(example))
    assert after == before
    assert assess_ten(bank(example), percent=100)["eligible"] is True
    rules = training_steps.Rules(production_rounds_only=False)
    before_targets, _ = training_thinking.build([original], rules)
    after_targets, _ = training_thinking.build([example], rules)
    before_records = before_targets["train"] + before_targets["eval"]
    after_records = after_targets["train"] + after_targets["eval"]
    assert len(after_records) == len(before_records) == 2
    # The sealed source hash changes when private diagnostic evidence is added.
    for before_row, after_row in zip(before_records, after_records):
        assert {key: value for key, value in after_row.items() if key != "source_refs"} == {
            key: value for key, value in before_row.items() if key != "source_refs"}
    assert all(row["source_refs"][0]["call_identity"] != call["call_identity"] for row in after_records)
    coverage = training_thinking.attempt_coverage(call, "miner", example["configuration"]["reasoning_route_registry"])
    assert coverage["returned"] == "unavailable" and coverage["tokens_reported"] is None


@pytest.mark.parametrize("damage", [
    "authority_missing", "authority_false_reason", "authority_succeeded", "authority_nontransport",
    "authority_bad_status", "authority_status_bool", "receipt_entry", "receipt_status", "receipt_succeeded",
    "receipt_identity", "receipt_operation", "receipt_request", "receipt_provenance", "receipt_delivery",
    "receipt_privacy", "receipt_response_hash", "receipt_wire_hash", "receipt_body_bytes", "receipt_duplicate",
    "primary_response", "primary_status", "known_cost", "settled", "worker_missing",
])
def test_failed_transport_auxiliary_cannot_bypass_existing_proof_guards(damage):
    example = with_auxiliary()
    call = example["provider_calls"][0]
    authority = call["lifecycle"][-1]["accounting"]["call"]
    receipt = call["upstream_response_receipts"][0]
    if damage == "authority_missing": call["lifecycle"][-1]["accounting"] = {}
    elif damage == "authority_false_reason": authority["reason"] = "missing_provider_cost"
    elif damage == "authority_succeeded": authority["call_succeeded"] = True
    elif damage == "authority_nontransport": authority["transport_error_class"] = "made_up"
    elif damage == "authority_bad_status": authority["observed_provider_status"] = 599
    elif damage == "authority_status_bool": authority["observed_provider_status"] = True
    elif damage == "receipt_entry": receipt["entry_id"] -= 1
    elif damage == "receipt_status": receipt["status"] = 503
    elif damage == "receipt_succeeded": receipt["call_succeeded"] = True
    elif damage == "receipt_identity": receipt["call_identity"] = "other"
    elif damage == "receipt_operation": receipt["operation_id"] = "openrouter.responses"
    elif damage == "receipt_request": receipt["request_hash"] = contracts.hash_bytes(b"other")
    elif damage == "receipt_provenance": receipt["provenance"] = "provider"
    elif damage == "receipt_delivery": receipt["delivery_to_worker"] = "socket_written"
    elif damage == "receipt_privacy": receipt["privacy_redactions"] = ["credential"]
    elif damage == "receipt_response_hash": receipt["response_sha256"] = contracts.hash_bytes(b"other")
    elif damage == "receipt_wire_hash": receipt["wire_body_sha256"] = contracts.hash_bytes(b"other")
    elif damage == "receipt_body_bytes": receipt["body_bytes"] = True
    elif damage == "receipt_duplicate": call["upstream_response_receipts"].append(deepcopy(receipt))
    elif damage == "primary_response": call["response"] = {"output": []}
    elif damage == "primary_status": call["response_status"] = 200
    elif damage == "known_cost": call["actual_microusd"] = 0
    elif damage == "settled": call["accounting_state"] = "settlement"
    else: example["worker_observation_artifacts"] = {}
    seal(example)
    assert not training_thinking.verified_failed_attempt(example, call), damage
    assert not assess_ten(bank(example), percent=100)["eligible"], damage


def test_delivered_positive_blank_primary_still_counts_missing_reasoning():
    example = captured_example()
    example["configuration"]["training_maximum_missing_reasoning_percent"] = 0
    call = example["provider_calls"][0]
    call["response"]["choices"][0]["message"]["reasoning"] = ""
    call["response"]["usage"]["completion_tokens_details"] = {"reasoning_tokens": 1}
    refresh_response_links(example, 0)
    assert not training_thinking.verified_failed_attempt(example, call)
    state = training_thinking.attempt_coverage(call, "miner", example["configuration"]["reasoning_route_registry"])
    assert state["returned"] == "missing_thinking" and state["tokens_reported"] == 1
    result = capture(example)
    assert not result["eligible"]
    assert result["counts"]["missing_reasoning_tokens"] == 1
    assert result["counts"]["missing_reasoning_calls"] == 1
