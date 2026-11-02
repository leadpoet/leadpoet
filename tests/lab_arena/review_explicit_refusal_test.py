"""Typed refusals permit one alternate without becoming code verdicts."""
import json

import pytest

from lab_arena import broker, code_review, code_review_policy
from tests.lab_arena.test_lab_arena_code_review_runtime import (
    MINER_KEY, ReviewTransport, _archive, _review_body, _reviewer, _source,
)
from tests.lab_arena.refusal_review_fallback599_postgres_test import (
    recovery_database, recovery_connect, recovery_store, recovery_superuser,
    accepted_recovery_submission,
)
from tests.lab_arena.test_lab_arena_code_review_postgres import (
    _age_review_error,
)
from tests.lab_arena.refusal_review_fallback_test import reviewer_for, router_receipt


REFUSALS = (
    ("content_filter", "refusal", None),
    ("stop", "refusal", None),
    ("refusal", "end_turn", None),
    ("stop", "end_turn", "private provider refusal prose"),
)


def refusal_body(parameters, *, finish="content_filter", native="refusal", refusal=None, cost="0.001234"):
    body = _review_body(parameters, cost=cost)
    choice = body["choices"][0]
    choice.update(finish_reason=finish, native_finish_reason=native)
    choice["message"]["content"] = ""
    if refusal is not None:
        choice["message"]["refusal"] = refusal
    return body


@pytest.mark.parametrize("finish,native,refusal", REFUSALS)
@pytest.mark.parametrize("cost,actual", [("0.001234", 1234), (None, None)])
def test_http_200_explicit_refusal_permits_one_alternate_claim(finish, native, refusal, cost, actual):
    transport = ReviewTransport(lambda parameters: {"body": refusal_body(
        parameters, finish=finish, native=native, refusal=refusal, cost=cost,
    )})
    reviewer, row, store, _ = _reviewer(_archive(_source()), transport)
    assert reviewer.review(row)["status"] == "error"
    document = store.finish_calls[0][4]
    assert document["error_code"] == "code_review_provider_refused"
    assert document["provider_http_status"] == 200
    assert document["retryable"] is True
    assert "categories" not in document and "passed" not in document
    assert "private provider refusal prose" not in json.dumps(document)
    assert store.finish_calls[0][5] == actual
    assert len(transport.sent) == 1
    # This is the actual admission read-time policy, independent of telemetry.
    assert code_review_policy.review_retry_available({
        "code_review_status": "error", "code_review_attempts": 1,
        "code_review_doc": document,
    })
    assert code_review_policy.review_model_for({"code_review_doc": document}) == code_review.FALLBACK_REVIEW_MODEL


@pytest.mark.parametrize("change", [
    lambda body: body.update(choices=[]),
    lambda body: body.update(choices=body["choices"] * 2),
    lambda body: body.update(choices=[None]),
    lambda body: body["choices"][0].update(finish_reason="stop", native_finish_reason=None, message=None),
    lambda body: body["choices"][0].update(finish_reason=["content_filter"], native_finish_reason=None),
    lambda body: body["choices"][0].update(finish_reason="length", native_finish_reason="max_tokens"),
    lambda body: body["choices"][0].update(finish_reason="stop", native_finish_reason="unknown_refusal"),
    lambda body: body["choices"][0].update(finish_reason="stop", native_finish_reason="end_turn", message={"refusal": True}),
    lambda body: body["choices"][0].update(finish_reason="stop", native_finish_reason="end_turn", message={"refusal": "  "}),
])
def test_malformed_or_nonrefusal_metadata_is_not_a_typed_refusal(change):
    body = refusal_body({"model": code_review.DEFAULT_REVIEW_MODEL, "messages": [None, {"content": '{"submission_files":[]}'}]})
    change(body)
    assert code_review_policy.explicit_provider_refusal(body) is False
    assert code_review_policy.explicit_provider_refusal({"error": {"code": 400, "message": "refusal"}}) is False


@pytest.mark.parametrize("finish,native", [
    ("content_filter", None), ("refusal", None),
    ("stop", "refusal"), ("stop", "content_filter"),
])
@pytest.mark.parametrize("message_shape", ["missing", "null", "invalid"])
@pytest.mark.parametrize("cost,actual", [("0.001234", 1234), (None, None)])
def test_typed_finish_refusal_without_message_selects_alternate(finish, native, message_shape, cost, actual):
    def respond(parameters):
        body = refusal_body(parameters, finish=finish, native=native, cost=cost)
        choice = body["choices"][0]
        if message_shape == "missing":
            choice.pop("message")
        else:
            choice["message"] = None if message_shape == "null" else []
        return {"body": body}
    transport = ReviewTransport(respond)
    reviewer, row, store, _ = _reviewer(_archive(_source()), transport)
    assert reviewer.review(row)["status"] == "error"
    document = store.finish_calls[0][4]
    assert document["error_code"] == "code_review_provider_refused"
    assert document["retryable"] is True
    assert code_review_policy.review_model_for({"code_review_doc": document}) == code_review.FALLBACK_REVIEW_MODEL
    assert "passed" not in document and "categories" not in document
    assert store.finish_calls[0][5] == actual
    assert len(transport.sent) == 1


@pytest.mark.parametrize("verdict", ["pass", "reject"])
def test_complete_pass_and_evidence_backed_reject_keep_existing_validation(verdict):
    findings = [] if verdict == "pass" else [{
        "category": "fabricated_evidence", "file": "harness.py",
        "evidence": "from src.agent import run_icp", "explanation": "Source-backed finding.",
    }]
    transport = ReviewTransport(lambda parameters: {
        "body": _review_body(parameters, verdict=verdict, findings=findings),
    })
    reviewer, row, store, _ = _reviewer(_archive(_source()), transport)
    assert reviewer.review(row)["status"] == ("passed" if verdict == "pass" else "rejected")
    assert store.finish_calls[0][5] == 1234
    assert len(transport.sent) == 1
    assert store.finish_calls[0][4]["categories"] == ([] if verdict == "pass" else ["fabricated_evidence"])


@pytest.mark.parametrize("failure", ["truncated", "content_json", "wire_json", "timeout"])
def test_other_unusable_reviews_keep_existing_retry_and_cost_rules(failure):
    class Transport:
        sent = 0
        def send(self, **request):
            self.sent += 1
            if failure == "timeout":
                raise broker.ProviderTransportError("private transport exception")
            if failure == "wire_json":
                return broker.ProviderResponse(200, {}, b'{"choices":')
            body = _review_body(json.loads(request["body"]))
            if failure == "truncated":
                body["choices"][0]["finish_reason"] = "length"
            else:
                body["choices"][0]["message"]["content"] = "{"
            return broker.ProviderResponse(200, {}, json.dumps(body).encode())
    transport = Transport()
    reviewer, row, store, _ = _reviewer(_archive(_source()), transport)
    assert reviewer.review(row)["status"] == "error"
    document = store.finish_calls[0][4]
    assert document["error_code"] == ("code_review_transport_error" if failure == "timeout" else "review_response_invalid")
    assert document["retryable"] is True
    assert store.finish_calls[0][5] == (1234 if failure in {"truncated", "content_json"} else None)
    assert transport.sent == 1


def test_credential_echo_precedes_refusal_and_keeps_cost_uncertain():
    transport = ReviewTransport(lambda parameters: {"body": refusal_body(parameters, refusal=MINER_KEY)})
    reviewer, row, store, _ = _reviewer(_archive(_source()), transport)
    assert reviewer.review(row)["status"] == "error"
    document = store.finish_calls[0][4]
    assert document["error_code"] == "code_review_credential_echo"
    assert document["retryable"] is False and store.finish_calls[0][5] is None
    assert MINER_KEY not in json.dumps(document)
    assert "provider_diagnostic" not in document


def test_private_diagnostic_refusal_label_cannot_change_a_complete_review(monkeypatch):
    monkeypatch.setattr(code_review_policy, "private_response_diagnostic", lambda *a, **k: {
        "finish_reason": "content_filter", "native_finish_reason": "refusal",
    })
    transport = ReviewTransport(lambda parameters: {"body": _review_body(parameters)})
    reviewer, row, store, _ = _reviewer(_archive(_source()), transport)
    assert reviewer.review(row)["status"] == "passed"
    assert store.finish_calls[0][4]["passed"] is True
    assert store.finish_calls[0][5] == 1234


@pytest.mark.parametrize("cost,kind,message_shape", [
    ("0.001234", "settlement", "normal"), (None, "uncertain", "normal"),
    ("0.001234", "settlement", "null"), (None, "uncertain", "missing"),
])
def test_real_ledger_refusal_keeps_charge_and_allows_one_alternate(recovery_store, recovery_superuser, cost, kind, message_shape):
    suffix = ("f71" if cost else "f72") + message_shape
    _, submission, miner = accepted_recovery_submission(recovery_store, suffix)
    payload = _archive(_source())
    def respond(parameters):
        if parameters["model"] == code_review.FALLBACK_REVIEW_MODEL:
            return {"body": router_receipt(_review_body(parameters), parameters)}
        body = refusal_body(parameters, cost=cost)
        if message_shape == "null":
            body["choices"][0]["message"] = None
        elif message_shape == "missing":
            body["choices"][0].pop("message")
        return {"body": body}
    transport = ReviewTransport(respond)
    reviewer, _ = reviewer_for(payload, transport, recovery_store)
    row = {**recovery_store.get_submission(submission), "source_size_bytes": len(payload)}
    assert reviewer.review(row)["status"] == "error"
    stored = recovery_store.get_submission(submission)
    assert stored["code_review_doc"]["error_code"] == "code_review_provider_refused"
    assert stored["code_review_doc"]["retryable"] is True
    assert "provider_diagnostic" not in stored["code_review_doc"]
    ledger = recovery_store.list_ledger(submission_id=submission)
    assert [entry["entry_kind"] for entry in ledger] == ["reservation", "dispatch", kind]
    assert ledger[-1]["amount_microusd"] == (1234 if cost else ledger[0]["amount_microusd"])
    assert ledger[-1]["entry_doc"]["provider_diagnostic"]["finish_reason"] == "content_filter"
    _age_review_error(recovery_superuser, submission)
    assert recovery_store.begin_submission_review(
        submission, miner, "b" * 64, 50_000, code_review.DEFAULT_REVIEW_MODEL, 3, 1200,
    )["status"] == "fallback_required"
    assert recovery_store.list_ledger(submission_id=submission) == ledger
    assert len(transport.sent) == 1
    row = {**recovery_store.get_submission(submission), "source_size_bytes": len(payload)}
    assert reviewer.review(row)["status"] == "passed"
    completed = recovery_store.list_ledger(submission_id=submission)
    assert completed[:3] == ledger
    assert [entry["entry_kind"] for entry in completed[3:]] == ["reservation", "dispatch", "settlement"]
    assert completed[3]["entry_doc"]["review_model"] == code_review.FALLBACK_REVIEW_MODEL
    assert reviewer.review(recovery_store.get_submission(submission))["status"] == "existing"
    assert recovery_store.list_ledger(submission_id=submission) == completed
    assert len(transport.sent) == 2


def test_legacy_generic_request_rejection_remains_terminal():
    transport = ReviewTransport(lambda _parameters: pytest.fail("legacy terminal error must not dispatch"))
    reviewer, row, store, _ = _reviewer(_archive(_source()), transport)
    document = {"model": code_review.DEFAULT_REVIEW_MODEL,
                "error_code": "code_review_provider_request_rejected", "retryable": False}
    row.update(code_review_status="error", code_review_attempts=1, code_review_doc=document)
    assert not code_review_policy.review_retry_available(row)
    assert code_review_policy.review_model_for(row) == code_review.DEFAULT_REVIEW_MODEL
    assert reviewer.review(row)["status"] == "exhausted"
    assert store.begin_calls == []
    assert store.finish_calls == []
    assert transport.sent == []
