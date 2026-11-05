"""Delivered model bytes survive unresolved and later settled billing."""
import base64
from copy import deepcopy
import json

import pytest

from lab_arena import broker, contracts, training_export, training_thinking
from tests.lab_arena.training_export_test import add_call, fixture
from tests.lab_arena.training_closed_unused_snapshot_test import closed_unused_values


def retained_success(*, settled=False, reason="missing_provider_cost"):
    values = fixture()
    add_call(values, status="uncertain")
    reservation, _, uncertain = values["ledger_rows"]
    body = json.dumps({"model": "openai/gpt-test", "choices": [{"message": {
        "content": "retained model answer"}}], "usage": {
        "prompt_tokens": 7, "completion_tokens": 11, "reasoning_tokens": 3}}).encode()
    terminal = broker._terminal_response_document(200, {
        "content-type": "application/json", "content-length": str(len(body))}, body, call_succeeded=True)
    uncertain.pop("terminal_response", None)
    uncertain["entry_doc"] = {"reason": "worker_reported", "call": {
        "reason": reason, "call_succeeded": True, "provider_status": 200,
        "credential_fingerprint": "sha256:" + "c" * 64,
        "openrouter_generation_id": "gen-retained-success",
        "openrouter_terminal_response": terminal}}
    if settled:
        values["ledger_rows"].append(dict(reservation, entry_id=4, entry_kind="settlement",
            amount_microusd=19, entry_doc={"openrouter_delayed_reconciliation": True,
                "reconciled_uncertainty_entry_id": 3},
            terminal_response=broker._terminal_response_document(502, {},
                broker.operations.GENERIC_UNAVAILABLE_BODY, call_succeeded=False,
                provider_cost={"basis": "openrouter_generation_cost", "units": "0.000019",
                    "unit_name": "usd", "operation": "openrouter.chat",
                    "request_id": "gen-retained-success"})))
    return values, body


@pytest.mark.parametrize("settled", [False, True])
@pytest.mark.parametrize("reason", ["missing_provider_cost", "settle_failure"])
def test_retained_success_uses_response_and_accounting_from_separate_authorities(settled, reason):
    values, body = retained_success(settled=settled, reason=reason)
    before = deepcopy(values["ledger_rows"])
    example = training_export.assemble_example(**values)
    call = example["provider_calls"][0]
    assert call["response"] == json.loads(body) and call["response_status"] == 200
    assert training_export.response_entry_id(call) == 3
    assert call["terminal_entry_id"] == (4 if settled else 3)
    assert call["actual_microusd"] == (19 if settled else None)
    assert ("provider_cost_unresolved" in call["evidence_gaps"]) is not settled
    if not settled:
        assert not example["completeness"]["complete"]
    assert call["response_provenance"]["delivery_to_worker"] == "not_proved"
    assert values["ledger_rows"] == before
    assert "credential_fingerprint" not in json.dumps(example)


@pytest.mark.parametrize("fault", ["fingerprint", "conflicting_fingerprint", "status", "provider_status", "generation",
                                    "reason", "cache_source", "body", "membership", "stale",
                                    "miner_hotkey", "stage", "funding_source"])
def test_retained_success_rejects_wrong_identity_and_malformed_authority(fault):
    values, _ = retained_success(settled=True)
    saved = values["ledger_rows"][2]["entry_doc"]["call"]
    terminal = saved["openrouter_terminal_response"]
    if fault == "fingerprint":
        saved["credential_fingerprint"] = "private-secret"
    elif fault == "conflicting_fingerprint":
        values["ledger_rows"][0]["entry_doc"]["credential_fingerprint"] = "sha256:" + "d" * 64
    elif fault == "status":
        terminal["status"] = "200"
    elif fault == "provider_status":
        saved["provider_status"] = False
    elif fault == "generation":
        saved["openrouter_generation_id"] = "bad generation"
    elif fault == "reason":
        saved["reason"] = "transport_failure"
    elif fault == "cache_source":
        terminal["judgment_cache_source_run_id"] = "other-run"
    elif fault == "body":
        terminal["body_b64"] = "@@malformed@@"
    elif fault == "membership":
        values["ledger_rows"][2]["operation_id"] = "openrouter.responses"
    elif fault in {"miner_hotkey", "stage", "funding_source"}:
        values["ledger_rows"][2][fault] = 9 if fault == "stage" else "wrong-owner"
    else:
        values["ledger_rows"][-1]["entry_doc"]["reconciled_uncertainty_entry_id"] = 2
    with pytest.raises(training_export.ExportError):
        training_export.assemble_example(**values)


def test_private_upstream_response_never_substitutes_for_the_explicit_terminal():
    values, _ = retained_success()
    saved = values["ledger_rows"][2]["entry_doc"]["call"]
    saved["upstream_response_receipt"] = saved.pop("openrouter_terminal_response")
    call = training_export.assemble_example(**values)["provider_calls"][0]
    assert "response" not in call
    assert "full_response_unavailable" in call["evidence_gaps"]
    assert call["actual_microusd"] is None


def test_secret_in_retained_response_is_redacted_and_cannot_make_exact_training():
    values, _ = retained_success(settled=True)
    secret = "sk-or-v1-" + "c" * 64
    terminal = values["ledger_rows"][2]["entry_doc"]["call"]["openrouter_terminal_response"]
    terminal["body_b64"] = base64.b64encode(json.dumps({"api_key": secret}).encode()).decode()
    example = training_export.assemble_example(**values)
    assert secret not in json.dumps(example)
    assert "response_privacy_nonexact" in example["provider_calls"][0]["evidence_gaps"]
    assert not example["completeness"]["complete"]


@pytest.mark.parametrize("operation", ["openrouter.chat", "openrouter.responses"])
@pytest.mark.parametrize("reason", ["provider_admission_cancelled", "provider_admission_deadline"])
@pytest.mark.parametrize("marked", [False, True])
def test_exact_no_transport_recovery_keeps_request_without_inventing_a_response(operation, reason, marked):
    values = closed_unused_values(operation)
    reservation, recovery = values["ledger_rows"][-2:]
    recovery["entry_doc"]["reason"] = reason
    if marked:
        dispatch = dict(reservation, entry_id=10002, entry_kind="dispatch", entry_doc={"dispatched_at": "test"})
        recovery["entry_id"] = 10003
        recovery["entry_doc"].update(transport_not_started=True, undispatched_dispatch_entry_id=10002)
        values["ledger_rows"].insert(-1, dispatch)
    example = training_export.assemble_example(**values)
    call = example["provider_calls"][-1]
    assert training_thinking.verified_undispatched_transport(example, call)
    assert call["actual_microusd"] == 0 and "response" not in call
    if marked:
        for field, value in (("transport_not_started", False), ("undispatched_dispatch_entry_id", 10001),
                             ("undispatched_dispatch_entry_id", True)):
            changed = deepcopy(call)
            changed["lifecycle"][-1]["accounting"][field] = value
            assert not training_thinking.verified_undispatched_transport(example, changed)
