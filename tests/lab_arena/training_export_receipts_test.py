"""Exact byte joins must not turn a host write into proof of agent use."""

import base64
import copy
import json

import pytest

from lab_arena import contracts, training_input, training_observations
from lab_arena.training_export import assemble_example, _link_worker_receipts
from tests.lab_arena.training_export_test import CLAIM_HASH, add_call, fixture, save_input


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


def test_binary_audit_capture_is_preserved_but_never_claims_privacy_readiness():
    values, payload = values_with_receipts()
    values['runs'][0]['result_doc']['resource_summary']['web_egress']['connection_count'] = 1
    raw = b'%PDF-1.7\x00\xffPRIVATE_BINARY_AUDIT'
    digest = contracts.hash_bytes(raw)
    payload['agent_observations'].append({
        'id': len(payload['agent_observations']), 'kind': 'web_fetch_binary',
        'observed_at': '2026-10-16T01:00:01Z', 'after_action_sequence': 0, 'depends_on': [],
        'privacy': {'redacted': False, 'redactions': []},
        'data': {'url': 'https://example.com/a.pdf', 'final_url': 'https://example.com/a.pdf',
                 'status': 200, 'retrieved_at': '2026-10-16T01:00:01Z', 'content_type': 'application/pdf',
                 'body_b64': base64.b64encode(raw).decode(), 'body_ref': None,
                 'body_sha256': digest, 'body_bytes': len(raw),
                 'stored_body_sha256': None, 'stored_body_bytes': None,
                 'privacy_review': 'unreviewed_binary',
                 'capture': {'source': 'runtime_http_v1', 'complete': True,
                             'received_body_sha256': digest, 'received_body_bytes': len(raw)}}})
    payload['agent_observation_count'] += 1
    save(values, payload)
    exported = assemble_example(**values)
    reasons = exported['completeness']['incomplete_reasons']
    assert 'binary_privacy_review_required' in reasons
    assert 'direct_web_coverage_unverified' in reasons
    assert 'direct_web_content_not_captured' not in reasons
    assert exported['privacy']['exact_replay'] is False
    captured = exported['worker_observation_artifacts']['execute:1']['payload']['agent_observations'][-1]
    assert base64.b64decode(captured['data']['body_b64'], validate=True) == raw


@pytest.mark.parametrize("extra_names,state_valid", [
    (["automatic_capture_summary"], True),
    (["automatic_capture_summary", "automatic_capture_summary"], False),
    (["automatic_capture_summary", "extra_agent_state"], False),
    (["extra_agent_state"], False),
])
def test_runtime_capture_status_does_not_change_baseline_phase_count(extra_names, state_valid):
    values, payload = values_with_receipts()
    for name in extra_names:
        payload["agent_observations"].append({
            "id": len(payload["agent_observations"]), "kind": "state_snapshot",
            "observed_at": "2026-10-16T01:00:02Z", "after_action_sequence": 0,
            "depends_on": [], "privacy": {"redacted": False, "redactions": []},
            "data": {"name": name, "value": {"source": "runtime_http_v1", "complete": True}},
        })
    payload["agent_observation_count"] = len(payload["agent_observations"])
    snapshot = save(values, payload)
    exported = assemble_example(**values)
    assert snapshot["payload"]["complete"]
    reasons = exported["completeness"]["incomplete_reasons"]
    assert ("agent_task_state_unverified" not in reasons) is state_valid
    assert exported["completeness"]["complete"] is state_valid
    assert exported["worker_observation_artifacts"]["execute:1"] == snapshot


def test_socket_write_alone_does_not_prove_agent_received_response():
    values, payload = values_with_receipts(acknowledged=False)
    save(values, payload)
    exported = assemble_example(**values)
    call = exported["provider_calls"][0]
    assert call["response_provenance"]["delivery_to_worker"] == "socket_written"
    assert call["response_provenance"]["agent_receipt"] == "not_reported"
    assert "agent_response_receipts_unverified" in exported["completeness"]["incomplete_reasons"]


def test_unread_http_metadata_is_retained_without_claiming_content_or_read_coverage():
    values, payload = values_with_receipts(acknowledged=False)
    values["runs"][0]["result_doc"]["resource_summary"]["web_egress"]["connection_count"] = 1
    payload["agent_observations"].append({
        "id": len(payload["agent_observations"]), "kind": "http_response_metadata",
        "observed_at": "2026-10-16T01:00:01Z", "after_action_sequence": 0, "depends_on": [],
        "privacy": {"redacted": False, "redactions": []},
        "data": {"url": "https://example.com/old", "status": 302, "content_type": "text/html",
                 "redirect_url": "https://example.com/new", "retrieved_at": "2026-10-16T01:00:01Z",
                 "source": "runtime_http_v1", "body_unconsumed": True}})
    payload["agent_observation_count"] += 1
    payload["complete"] = False
    payload["failures"] = ["capture_incomplete"]
    snapshot = save(values, payload)
    exported = assemble_example(**values)
    assert exported["worker_observation_artifacts"]["execute:1"] == snapshot
    assert exported["completeness"]["dimensions"]["input_output_label"] is True
    reasons = exported["completeness"]["incomplete_reasons"]
    assert "direct_web_content_not_captured" in reasons
    assert "agent_response_receipts_unverified" in reasons
    assert "worker_observations_incomplete" in reasons
    assert exported["completeness"]["complete"] is False


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


def test_failed_execution_phases_do_not_poison_accepted_retry_or_hide_old_receipts():
    values, selected_payload = values_with_receipts()
    selected = values["runs"][0]
    selected["attempt"] = 2
    prior = copy.deepcopy(selected)
    prior.update(run_id="execute:0", attempt=1, status="failed",
                 terminal_cause="invalid_output", output_ref=None)
    values["runs"].append(prior)
    prior_identity = "sha256:" + "2" * 64
    prior_ledger = copy.deepcopy(values["ledger_rows"])
    for index, row in enumerate(prior_ledger, start=4):
        row.update(run_id=prior["run_id"], call_identity=prior_identity, entry_id=index)
        row["entry_doc"]["base_call_identity"] = prior_identity
    values["ledger_rows"].extend(prior_ledger)
    prior_payload = copy.deepcopy(selected_payload)
    prior_payload["response_receipts"][0]["call_identity"] = prior_identity
    prior_payload["agent_observations"][0]["data"]["call_identity"] = prior_identity
    prior_payload["agent_observations"] = [row for row in prior_payload["agent_observations"]
                                           if row["kind"] != "state_snapshot" or row["data"]["name"]
                                           in {"after_discovery", "after_verification"}]
    for index, row in enumerate(prior_payload["agent_observations"]):
        row["id"] = index
    prior_payload["agent_observation_count"] = len(prior_payload["agent_observations"])
    prior_snapshot, raw = training_observations.build_snapshot(
        prior["run_id"], "execute", prior_payload,
        lease_generation=1, claim_request_hash=CLAIM_HASH)
    values["objects"].put(training_observations.object_ref(prior["run_id"], 1, CLAIM_HASH), raw)
    selected_input = training_input.read_input(values["objects"], selected["run_id"], "execute", 1, CLAIM_HASH)
    save_input(values["objects"], prior["run_id"], "execute", selected_input["input"])
    selected_snapshot = save(values, selected_payload)

    exported = assemble_example(**values)
    assert exported["completeness"]["complete"]
    assert {row["run_id"] for row in exported["attempts"] if row["kind"] == "execute"} == {
        prior["run_id"], selected["run_id"]}
    assert len(exported["provider_calls"]) == 2
    assert exported["costs"]["confirmed_actual_microusd"] == 84
    assert exported["worker_observation_artifacts"][prior["run_id"]] == prior_snapshot
    assert exported["worker_observation_artifacts"][selected["run_id"]] == selected_snapshot
    assert all(call["response_provenance"]["agent_receipt"] == "reported_exact_body_received"
               for call in exported["provider_calls"])

    selected_payload["agent_observations"] = [row for row in selected_payload["agent_observations"]
                                              if row["kind"] != "state_snapshot" or row["data"]["name"] != "final"]
    for index, row in enumerate(selected_payload["agent_observations"]):
        row["id"] = index
    selected_payload["agent_observation_count"] = len(selected_payload["agent_observations"])
    save(values, selected_payload)
    assert "agent_task_state_unverified" in assemble_example(**values)["completeness"]["incomplete_reasons"]

    save(values, selected_snapshot["payload"])
    prior_payload["agent_observations"][0]["data"]["body_bytes"] += 1
    prior_snapshot, raw = training_observations.build_snapshot(
        prior["run_id"], "execute", prior_payload,
        lease_generation=1, claim_request_hash=CLAIM_HASH)
    values["objects"].put(training_observations.object_ref(prior["run_id"], 1, CLAIM_HASH), raw)
    exported = assemble_example(**values)
    assert not exported["completeness"]["complete"]
    assert "agent_response_receipt_link_invalid" in exported["completeness"]["incomplete_reasons"]
    assert exported["costs"]["confirmed_actual_microusd"] == 84


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


@pytest.mark.parametrize("ledger_body,expected_mismatch", [(None, False), (b'{"error":"different"}', True)])
def test_error_read_links_worker_body_without_settling_or_replacing_provider_evidence(ledger_body, expected_mismatch):
    identity = "sha256:" + "1" * 64
    body = b'{"error":"uncertain worker reply"}'
    receipt = {"ordinal": 0, "action_sequence": 0, "operation_id": "openrouter.chat",
               "call_identity": identity, "protocol": "http", "status": 503,
               "body_sha256": contracts.hash_bytes(body), "body_bytes": len(body),
               "write_state": "socket_written", "socket_written_at": "2026-10-16T01:00:00Z"}
    buffer = training_observations.Buffer()
    buffer.append({"ordinal": 0, "action_sequence": 0, "operation_id": "openrouter.chat",
                   "kind": "nonstandard_response", "call_identity": identity,
                   "contributes_to_provider_call_count": True,
                   "request": {"admitted_parameters": {}, "timeout_ms": 1000},
                   "outcome": {"protocol": "http", "response": {
                       "status": 503, "headers": {}, "body_b64": base64.b64encode(body).decode()}},
                   "privacy": {"redacted": False, "redactions": []}})
    buffer.append_response_receipt(receipt)
    buffer.append_agent_observation({"id": 0, "kind": "response_received",
        "observed_at": "2026-10-16T01:00:01Z", "after_action_sequence": 0,
        "depends_on": [], "data": {key: receipt[key] for key in
            ("call_identity", "status", "body_sha256", "body_bytes")},
        "privacy": {"redacted": False, "redactions": []}})
    snapshot, _ = training_observations.build_snapshot("execute:1", "execute", buffer.document(),
        lease_generation=1, claim_request_hash=CLAIM_HASH)
    ledger_receipts = [] if ledger_body is None else [{"entry_id": 3, "wire_body_sha256": contracts.hash_bytes(ledger_body),
                                                      "wire_body_bytes": len(ledger_body), "metadata": {"status": 503}}]
    call = {"call_identity": identity, "action_sequence": 0, "operation_id": "openrouter.chat",
            "response_status": 503, "response_receipts": ledger_receipts,
            "accounting_state": "uncertain", "actual_microusd": None,
            "response_provenance": {"source": "stored_provider_ledger_receipt", "delivery_to_worker": "not_proved"}}
    errors = _link_worker_receipts({"run_id": "execute:1", "kind": "execute"}, [call], snapshot)
    provenance = call["response_provenance"]
    assert provenance["agent_receipt"] == "reported_exact_body_received"
    assert provenance["worker_body_source"] == "worker_error_observation"
    assert provenance["worker_error_body_acknowledged"] is True
    assert ("worker_provider_response_mismatch" in errors) is expected_mismatch
    assert call["accounting_state"] == "uncertain" and call["actual_microusd"] is None
    assert call["response_receipts"] == ledger_receipts
