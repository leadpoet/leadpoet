"""Exact byte joins must not turn a host write into proof of agent use."""

import base64
import copy
import json

import pytest

from lab_arena import contracts, training_observations
from lab_arena.training_export import assemble_example
from tests.lab_arena.training_export_test import CLAIM_HASH, add_call, fixture


def values_with_receipts(*, acknowledged=True):
    values = fixture()
    add_call(values)
    terminal = values["ledger_rows"][-1]
    body = base64.b64decode(terminal["terminal_response"]["body_b64"])
    receipt = {
        "ordinal": 0, "action_sequence": 0, "operation_id": "openrouter.chat",
        "call_identity": terminal["call_identity"], "protocol": "http", "status": 200,
        "body_sha256": contracts.hash_bytes(body), "body_bytes": len(body),
        "write_state": "socket_written", "socket_written_at": "2026-10-16T01:00:00Z",
    }
    buffer = training_observations.Buffer()
    buffer.append_response_receipt(receipt)
    rows = []
    if acknowledged:
        rows.append({"kind": "response_received", "data": {
            key: receipt[key] for key in ("call_identity", "status", "body_sha256", "body_bytes")}})
    for name in ("after_discovery", "after_verification", "after_admission", "final"):
        rows.append({"kind": "state_snapshot", "data": {"name": name, "value": {"rows": [{"selected": 0}]}}})
    rows.append({"kind": "state_snapshot", "data": {"name": "training_capture_summary", "value": {
        "contract_version": 1, "scope": "agent_direct_egress_and_worker_socket",
        "sdk_available": True, "state_recorded": 4}}})
    for row in rows:
        buffer.append_agent_observation({
            **row, "id": len(buffer.agent_observations), "after_action_sequence": 0,
            "observed_at": "2026-10-16T01:00:01Z", "depends_on": [],
            "privacy": {"redacted": False, "redactions": []},
        })
    return values, buffer.document()


def save(values, payload):
    snapshot, raw = training_observations.build_snapshot(
        "execute:1", "execute", payload, lease_generation=1, claim_request_hash=CLAIM_HASH)
    values["objects"].put(training_observations.object_ref("execute:1", 1, CLAIM_HASH), raw)
    return snapshot


def test_receipts_join_to_exact_ledger_bytes_and_agent_report_without_body_duplication():
    values, payload = values_with_receipts()
    snapshot = save(values, payload)
    exported = assemble_example(**values)
    assert exported["completeness"]["complete"]
    call = exported["provider_calls"][0]
    assert call["response_provenance"]["delivery_to_worker"] == "socket_written"
    assert call["response_provenance"]["agent_receipt"] == "reported_exact_body_received"
    assert call["response_provenance"]["agent_receipt_refs"] == [{
        "run_id": "execute:1", "id": 0, "payload_sha256": snapshot["payload_sha256"]}]
    assert "body" not in exported["worker_observation_artifacts"]["execute:1"]["payload"]["response_receipts"][0]


def test_socket_write_alone_does_not_prove_agent_received_response():
    values, payload = values_with_receipts(acknowledged=False)
    save(values, payload)
    exported = assemble_example(**values)
    call = exported["provider_calls"][0]
    assert call["response_provenance"]["delivery_to_worker"] == "socket_written"
    assert call["response_provenance"]["agent_receipt"] == "not_reported"
    assert "agent_response_receipts_unverified" in exported["completeness"]["incomplete_reasons"]


@pytest.mark.parametrize("changed_body", [False, True])
def test_later_billing_recovery_cannot_replace_acknowledged_response_silently(changed_body):
    values, payload = values_with_receipts()
    recovered = copy.deepcopy(values["ledger_rows"][-1])
    recovered.update(entry_id=4, entry_kind="recovery")
    if changed_body:
        recovered["terminal_response"]["body_b64"] = base64.b64encode(json.dumps({"result": "later body"}).encode()).decode()
    values["ledger_rows"].append(recovered)
    save(values, payload)
    exported = assemble_example(**values)
    provenance = exported["provider_calls"][0]["response_provenance"]
    assert provenance["selected_response_acknowledged"] is not changed_body
    assert ("agent_selected_response_unacknowledged" in exported["completeness"]["incomplete_reasons"]) is changed_body
    assert provenance["worker_received_receipt_entry_ids"] == ([3] if changed_body else [3, 4])
    if changed_body:
        assert exported["provider_calls"][0]["response_receipts"][0]["response"]["choices"][0]["message"]["content"] == "result"


@pytest.mark.parametrize("change", ["missing_phase", "overflow", "wrong_count", "only_summary"])
def test_arbitrary_snapshot_or_incomplete_phase_report_cannot_clear_state_gap(change):
    values, payload = values_with_receipts()
    if change == "missing_phase": payload["agent_observations"].pop(2)
    elif change == "only_summary": payload["agent_observations"] = payload["agent_observations"][:1] + payload["agent_observations"][-1:]
    else:
        summary = payload["agent_observations"][-1]["data"]["value"]
        summary["state_overflow_rows" if change == "overflow" else "state_recorded"] = 1
    for index, row in enumerate(payload["agent_observations"]): row["id"] = index
    payload["agent_observation_count"] = len(payload["agent_observations"])
    save(values, payload)
    assert "agent_task_state_unverified" in assemble_example(**values)["completeness"]["incomplete_reasons"]


@pytest.mark.parametrize("mutation", ["status", "bytes", "hash", "action", "identity", "unwritten", "ack"])
def test_mismatched_or_unwritten_receipt_cannot_prove_agent_delivery(mutation):
    values, payload = values_with_receipts()
    receipt = payload["response_receipts"][0]
    if mutation == "status": receipt["status"] = 201
    elif mutation == "bytes": receipt["body_bytes"] += 1
    elif mutation == "hash": receipt["body_sha256"] = "sha256:" + "a" * 64
    elif mutation == "action": receipt["action_sequence"] = 9
    elif mutation == "identity": receipt["call_identity"] = "sha256:" + "a" * 64
    elif mutation == "ack": payload["agent_observations"][0]["data"]["body_bytes"] += 1
    elif mutation == "unwritten":
        receipt.update(write_state="prepared", socket_written_at=None)
        payload.update(complete=False, failures=["capture_incomplete"])
    save(values, payload)
    exported = assemble_example(**values)
    assert exported["completeness"]["dimensions"]["input_output_label"]
    assert "agent_response_receipts_unverified" in exported["completeness"]["incomplete_reasons"]
    assert exported["provider_calls"][0]["response_provenance"].get("agent_receipt") != "reported_exact_body_received"
