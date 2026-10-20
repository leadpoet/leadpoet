"""Exact byte joins must not turn a host write into proof of agent use."""

import base64
import copy

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
    rows.append({"kind": "state_snapshot", "data": {"name": "final_selection", "value": {"selected": [0]}}})
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
