"""A saved Deepline recovery is delivery evidence, never a rewritten bill."""

import base64
import copy

import pytest

from lab_arena import broker, contracts, operations, training_capture, training_export, training_observations, training_steps, training_thinking
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


def recovered_compat_values(*, failed_write=True, operation="scrapingdog.scrape", parameters=None):
    """Match the retained 189,033-byte legacy judge recovery without private text."""
    values = recovered_values(kind="score")
    from lab_arena import scoring_provider_compat
    requested = parameters or {"url": "https://example.com/source", "dynamic": False}
    selected = scoring_provider_compat.route_for(
        kind="score", funding_source="miner_key", round_id="arena-2026-11-12",
        operation_id=operation, parameters=requested)
    effective, route = selected.effective_parameters, selected.summary()
    request_hash = contracts.document_hash(requested)
    outbound = {"provider": operations.DEEPLINE_TOOL_PROVIDERS[effective["tool"]], "operation": effective["tool"],
                "payload": effective["payload"]}
    pointer = training_capture.RequestCapture(values["objects"])(
        {"operation_id": operation, "effective_operation_id": "deepline.execute",
         "route": route, "request_hash": request_hash, "hashed_parameters": requested,
         "admitted_parameters": requested, "effective_parameters": effective,
         "outbound": {"method": "POST", "host": "code.deepline.com",
                      "path": "/api/v2/integrations/" + effective["tool"] + "/execute",
                      "body": outbound, "body_sha256": contracts.document_hash(outbound)}})
    for row in values["ledger_rows"]:
        row["operation_id"] = operation
        row["entry_doc"].update(request_hash=request_hash, request_capture=pointer,
                                tool=effective["tool"], **route)
    terminal = values["ledger_rows"][-1]
    raw = b"<html>" + b"x" * (189_033 - len(b"<html></html>")) + b"</html>"
    success = broker._terminal_response_document(
        200, {"content-type": "text/html"}, raw, call_succeeded=True,
        provider_cost={"request_id": "native-recovered", "operation": effective["tool"]})
    terminal["terminal_response"]["provider_cost"]["operation"] = effective["tool"]
    terminal["training_recovered_response"].update(
        request_hash=request_hash, tool=effective["tool"], terminal_response=success)
    buffer = training_observations.Buffer()
    buffer.append_response_receipt({
        "ordinal": 0, "action_sequence": 0, "operation_id": operation,
        "call_identity": terminal["call_identity"], "protocol": "http", "status": 200,
        "body_sha256": contracts.hash_bytes(raw), "body_bytes": len(raw),
        "write_state": "socket_write_failed" if failed_write else "socket_written",
        "socket_written_at": None if failed_write else "2026-11-02T01:00:01Z"})
    _, raw = training_observations.build_snapshot(
        "judge:1", "score", buffer.document(), lease_generation=1, claim_request_hash=CLAIM_HASH)
    values["objects"].put(training_observations.object_ref("judge:1", 1, CLAIM_HASH), raw)
    return values


@pytest.mark.parametrize("failed_write", [True, False])
def test_recovered_compatibility_keeps_original_operation_and_exact_native_response(failed_write):
    values = recovered_compat_values(failed_write=failed_write)
    original = copy.deepcopy(values["ledger_rows"])
    exported = assemble_example(**values)
    call = exported["provider_calls"][0]
    assert call["operation_id"] == "scrapingdog.scrape" and call["provider"] == "deepline"
    assert call["recovered_response_ref"]["wire_body_bytes"] == 189_033
    assert call["request"]["effective_operation_id"] == "deepline.execute"
    expected = "socket_write_failed" if failed_write else "socket_written"
    assert call["response_provenance"]["delivery_to_worker"] == expected
    evidence = (training_steps._failed_host_result(exported, call) if failed_write else
                training_steps._host_response(exported, call, write_state=expected, require_client_ack=False))
    assert evidence["response"] == call["response"]
    trace = training_thinking.readable_trace(exported)
    assert trace["calls"][0]["response_provenance"]["delivery_to_worker"] == expected
    assert trace["calls"][0]["recovered_response_ref"] == call["recovered_response_ref"]
    assert trace["projection_truncated"] is False
    assert values["ledger_rows"] == original


@pytest.mark.parametrize("damage", [
    lambda c: c.update(provider="scrapingdog"),
    lambda c: c.update(operation_id="openrouter.responses"),
    lambda c: c["request"].update(operation_id="exa.contents"),
    lambda c: c["request"].update(effective_operation_id="openrouter.responses"),
    lambda c: c["request"]["route"].update(effective_operation_id="exa.contents"),
    lambda c: c["request"]["route"].update(adapter="different-adapter"),
    lambda c: c["request"]["route"].update(compatibility_version="unapproved-version"),
    lambda c: c["lifecycle"][0]["accounting"].pop("effective_operation_id"),
    lambda c: c["lifecycle"][0]["accounting"].update(tool="exa_contents"),
    lambda c: c["request"]["effective_parameters"].update(tool="exa_contents"),
    lambda c: c["request"]["outbound"]["body"].update(provider="exa"),
    lambda c: c["request"]["outbound"].update(body_sha256="sha256:" + "0" * 64),
    lambda c: c["request"]["outbound"].update(host="example.com"),
    lambda c: c["request"]["outbound"].update(path="/other"),
    lambda c: c["request"]["outbound"].update(method="GET"),
    lambda c: c["recovered_response_ref"].update(call_identity="sha256:" + "0" * 64),
    lambda c: c["recovered_response_ref"].update(reservation_entry_id=2),
    lambda c: c["recovered_response_ref"].update(run_id="other-attempt"),
    lambda c: c["recovered_response_ref"].update(request_hash="sha256:" + "0" * 64),
    lambda c: c["recovered_response_ref"].update(response_sha256="sha256:" + "0" * 64),
])
def test_compatibility_recovery_rejects_changed_recorded_binding(damage):
    call = assemble_example(**recovered_compat_values())["provider_calls"][0]
    damage(call)
    with pytest.raises(ExportError, match="recovered_deepline_response_identity_invalid"):
        training_export.recovered_response_ref(call)


def test_compatibility_recovery_preserves_privacy_exclusion_and_no_worker_read():
    exported = assemble_example(**recovered_compat_values())
    call = exported["provider_calls"][0]
    call["request_capture"]["redacted"] = True
    with pytest.raises(ValueError, match="request_privacy_nonexact"):
        training_steps._failed_host_result(exported, call)
    call["request_capture"]["redacted"] = False
    call["response_privacy_redactions"] = ["credential"]
    call["recovered_response_ref"]["privacy_redactions"] = ["credential"]
    with pytest.raises(ValueError, match="response_privacy_nonexact"):
        training_steps._failed_host_result(exported, call)
    call["response_privacy_redactions"] = []
    call["recovered_response_ref"]["privacy_redactions"] = []
    with pytest.raises(ValueError, match="worker_write_unverified"):
        training_steps._host_result(exported, call)


def test_ordinary_scrapingdog_receipt_needs_no_deepline_recovery():
    values = recovered_compat_values()
    source = values["ledger_rows"][-1].pop("training_recovered_response")["terminal_response"]
    requested = {"url": "https://example.com/source", "dynamic": False}
    pointer = training_capture.RequestCapture(values["objects"])(
        {"operation_id": "scrapingdog.scrape", "effective_operation_id": "scrapingdog.scrape",
         "route": {}, "request_hash": contracts.document_hash(requested),
         "hashed_parameters": requested, "admitted_parameters": requested,
         "effective_parameters": requested, "outbound": {"body": None}})
    for row in values["ledger_rows"]:
        row["provider"] = "scrapingdog"
        row["entry_doc"]["request_capture"] = pointer
        for key in ("adapter", "compatibility_version", "effective_operation_id", "tool"):
            row["entry_doc"].pop(key, None)
    values["ledger_rows"][-1]["terminal_response"] = source
    exported = assemble_example(**values)
    call = exported["provider_calls"][0]
    assert call["provider"] == "scrapingdog" and "recovered_response_ref" not in call
    assert training_steps._failed_host_result(exported, call)["response"] == call["response"]
    assert training_thinking.readable_trace(exported)["projection_truncated"] is False


@pytest.mark.parametrize("operation,parameters", [
    ("scrapingdog.x_post", {"tweetId": "1234567890123456789"}),
    ("scrapingdog.linkedinjobs", {"job_id": "1234567890"}),
    ("scrapingdog.profile_post", {"id": "1234567890123456789"}),
])
def test_other_recorded_compatibility_operations_keep_their_original_identity(operation, parameters):
    exported = assemble_example(**recovered_compat_values(operation=operation, parameters=parameters))
    call = exported["provider_calls"][0]
    assert call["operation_id"] == operation
    assert training_steps._failed_host_result(exported, call)["response"] == call["response"]
    assert training_thinking.readable_trace(exported)["projection_truncated"] is False
