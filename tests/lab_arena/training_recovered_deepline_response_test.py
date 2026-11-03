"""A saved Deepline recovery is delivery evidence, never a rewritten bill."""

import base64
import copy

import pytest

from lab_arena import broker, contracts, training_capture, training_export, training_observations, training_steps, training_thinking
from lab_arena.training_export import ExportError, assemble_example
from lab_arena.training_reader import _ledger
from lab_arena.store import ArenaStore
from tests.lab_arena.training_export_test import CLAIM_HASH, add_call, fixture


def recovered_values(*, operation="deepline.execute", kind="execute"):
    values = fixture()
    add_call(values, provider="deepline", operation=operation)
    run_id = "judge:1" if kind == "score" else "execute:1"
    parameters = ({"tool": "exa_contents", "payload": {"urls": ["https://example.com"]}}
                  if operation == "deepline.execute" else {"urls": ["https://example.com"]})
    if kind == "score":
        values["runs"][0]["result_doc"]["resource_summary"]["provider_call_count"] = 0
        values["runs"][1]["result_doc"]["resource_summary"]["provider_call_count"] = 1
    request_hash = contracts.document_hash(parameters)
    pointer = training_capture.RequestCapture(values["objects"])(
        {"hashed_parameters": parameters, "request_hash": request_hash,
         "operation_id": operation, "admitted_parameters": parameters,
         "effective_parameters": parameters, "outbound": {"body": parameters}})
    for row in values["ledger_rows"]:
        row["run_id"] = run_id
        row["entry_doc"] = {**row["entry_doc"], "request_hash": request_hash,
                            "request_capture": pointer, "tool": "exa_contents"}
    terminal = values["ledger_rows"][-1]
    original = broker._terminal_response_document(
        502, {}, b'{"error":{"code":"provider_unavailable"}}', call_succeeded=False,
        provider_cost={"request_id": "native-recovered", "operation": "exa_contents"})
    terminal["terminal_response"] = original
    success = broker._terminal_response_document(
        200, {"content-type": "application/json"}, b'{"result":{"text":"retained research"}}',
        call_succeeded=True,
        provider_cost={"request_id": "native-recovered", "operation": "exa_contents"})
    terminal["training_recovered_response"] = {
        "table": "lab_arena_deepline_call_responses", "call_identity": terminal["call_identity"],
        "reservation_entry_id": 1, "run_id": run_id, "request_id": "native-recovered",
        "created_at": "2026-11-02T01:00:00+00:00", "terminal_entry_id": 3,
        "request_hash": request_hash, "tool": "exa_contents", "terminal_response": success}
    raw = base64.b64decode(success["body_b64"])
    write = {"ordinal": 0, "action_sequence": 0, "operation_id": operation,
             "call_identity": terminal["call_identity"], "protocol": "http", "status": 200,
             "body_sha256": contracts.hash_bytes(raw), "body_bytes": len(raw),
             "write_state": "socket_written", "socket_written_at": "2026-11-02T01:00:01Z"}
    buffer = training_observations.Buffer()
    buffer.append_response_receipt(write)
    buffer.append_agent_observation({
        "id": 0, "kind": "response_received", "after_action_sequence": 0,
        "observed_at": "2026-11-02T01:00:02Z", "depends_on": [],
        "privacy": {"redacted": False, "redactions": []},
        "data": {key: write[key] for key in ("call_identity", "status", "body_sha256", "body_bytes")}})
    _, raw = training_observations.build_snapshot(
        run_id, kind, buffer.document(), lease_generation=1, claim_request_hash=CLAIM_HASH)
    values["objects"].put(training_observations.object_ref(run_id, 1, CLAIM_HASH), raw)
    return values


@pytest.mark.parametrize("operation", ["deepline.execute", "exa.contents"])
@pytest.mark.parametrize("kind", ["execute", "score"])
def test_saved_recovery_links_exact_host_body_without_rewriting_ledger_receipts(operation, kind):
    values = recovered_values(operation=operation, kind=kind)
    original = copy.deepcopy(values["ledger_rows"])
    exported = assemble_example(**values)
    assert "worker_response_receipt_link_invalid" not in exported["completeness"]["incomplete_reasons"]
    assert "agent_response_receipt_link_invalid" not in exported["completeness"]["incomplete_reasons"]
    call = exported["provider_calls"][0]
    assert call["response_status"] == 200
    assert call["response"] == {"result": {"text": "retained research"}}
    assert call["actual_microusd"] == 42 and call["terminal_entry_id"] == 3
    assert call["response_receipts"][0]["metadata"]["status"] == 502
    assert call["response_receipts"][0]["response"] == {"error": {"code": "provider_unavailable"}}
    recovery = call["recovered_response_ref"]
    assert recovery["table"] == "lab_arena_deepline_call_responses"
    assert recovery["call_identity"] == call["call_identity"]
    assert "body_b64" not in recovery["metadata"]
    assert recovery["response_ref"] == {"field": "response"}
    assert call["response_provenance"]["worker_received_receipt_entry_ids"] == []
    assert call["response_provenance"]["worker_body_source"] == "lab_arena_deepline_call_responses"
    assert training_steps._host_result(exported, call)["response"] == call["response"]
    trace = training_thinking.readable_trace(exported)
    assert trace["calls"][0]["recovered_response_ref"] == recovery
    assert values["ledger_rows"] == original


def test_no_recovery_preserves_unmatched_502_and_does_not_claim_delivery():
    values = recovered_values()
    values["ledger_rows"][-1].pop("training_recovered_response")
    exported = assemble_example(**values)
    assert "worker_response_receipt_link_invalid" in exported["completeness"]["incomplete_reasons"]
    call = exported["provider_calls"][0]
    assert call["response_status"] == 502 and "recovered_response_ref" not in call


def test_recovered_response_with_unresolved_bill_keeps_uncertain_accounting():
    values = recovered_values()
    terminal = values["ledger_rows"][-1]
    terminal.update(entry_kind="uncertain", terminal_response=None)
    exported = assemble_example(**values)
    call = exported["provider_calls"][0]
    assert call["accounting_state"] == "uncertain" and call["actual_microusd"] is None
    assert call["response_status"] == 200
    assert "provider_cost_unresolved" in call["evidence_gaps"]
    assert "full_response_unavailable" not in call["evidence_gaps"]
    assert training_steps._host_result(exported, call)["response"] == call["response"]


def test_fixed_exa_tool_binding_does_not_depend_on_optional_cost_metadata():
    exported = assemble_example(**recovered_values(operation="exa.contents"))
    call = exported["provider_calls"][0]
    call["recovered_response_ref"]["metadata"].pop("provider_cost")
    call["recovered_response_ref"]["tool"] = "wrong-tool"
    with pytest.raises(ExportError, match="recovered_deepline_response_identity_invalid"):
        training_steps._host_result(exported, call)


def test_identical_success_already_in_ledger_is_parsed_only_once(monkeypatch):
    values = recovered_values()
    terminal = values["ledger_rows"][-1]
    terminal["terminal_response"] = copy.deepcopy(terminal["training_recovered_response"]["terminal_response"])
    parse = training_export._response_value
    parsed = []

    def counted(*args):
        parsed.append(args)
        return parse(*args)

    monkeypatch.setattr(training_export, "_response_value", counted)
    exported = assemble_example(**values)
    call = exported["provider_calls"][0]
    assert len(parsed) == 1
    assert "recovered_response_ref" not in call
    assert call["response_provenance"]["source"] == "stored_provider_ledger_receipt"
    assert call["response_receipts"][0]["response_ref"] == {"entry_id": 3, "field": "response"}
    assert training_steps._host_result(exported, call)["response"] == call["response"]


@pytest.mark.parametrize("field,value", [
    ("call_identity", "sha256:" + "9" * 64), ("run_id", "foreign-run"),
    ("reservation_entry_id", 2), ("terminal_entry_id", 2),
    ("request_hash", "sha256:" + "9" * 64), ("request_id", "wrong-native"),
    ("wire_body_sha256", "sha256:" + "9" * 64), ("wire_body_bytes", 1),
    ("response_sha256", "sha256:" + "9" * 64), ("table", "lab_arena_ledger"), ("tool", "wrong-tool"),
])
def test_exported_recovery_binding_cannot_forge_host_result(field, value):
    exported = assemble_example(**recovered_values())
    call = exported["provider_calls"][0]
    call["recovered_response_ref"][field] = value
    with pytest.raises(ValueError):
        training_steps._host_result(exported, call)


@pytest.mark.parametrize("field,value", [
    ("call_identity", "sha256:" + "9" * 64), ("run_id", "foreign-run"),
    ("reservation_entry_id", 2), ("terminal_entry_id", 2),
    ("request_hash", "sha256:" + "9" * 64), ("request_id", "wrong-native"),
    ("tool", "exa.search"), ("table", "lab_arena_ledger"),
])
def test_recovery_identity_mismatch_fails_closed(field, value):
    values = recovered_values()
    values["ledger_rows"][-1]["training_recovered_response"][field] = value
    with pytest.raises(ExportError, match="recovered_deepline_response_identity_invalid"):
        assemble_example(**values)


def test_only_training_reads_opt_in_to_the_private_computed_projection():
    class Transport:
        def __init__(self):
            self.calls = []

        def select(self, table, **kwargs):
            self.calls.append((table, kwargs))
            return []

    transport = Transport()
    store = ArenaStore(transport)
    store.list_ledger(run_id="run")
    assert "columns" not in transport.calls[-1][1]
    assert _ledger(store, "run") == []
    assert transport.calls[-1][1]["columns"] == "*,training_recovered_response:lab_arena_training_deepline_response_v1"
