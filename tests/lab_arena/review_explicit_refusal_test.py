"""Typed provider refusals stop repeat dispatch without becoming code verdicts."""
import json

import pytest

from lab_arena import broker, code_review, code_review_policy
from tests.lab_arena.test_lab_arena_code_review_runtime import (
    MINER_KEY, ReviewTransport, _archive, _review_body, _reviewer, _source,
)
from tests.lab_arena.review_private_diagnostics594_postgres_test import (
    recovery_database, recovery_connect, recovery_store, recovery_superuser,
)
from tests.lab_arena.test_lab_arena_code_review_postgres import (
    _age_review_error, accepted_recovery_submission,
)


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
def test_http_200_explicit_refusal_is_permanent_infrastructure_error(finish, native, refusal, cost, actual):
    transport = ReviewTransport(lambda parameters: {"body": refusal_body(
        parameters, finish=finish, native=native, refusal=refusal, cost=cost,
    )})
    reviewer, row, store, _ = _reviewer(_archive(_source()), transport)
    assert reviewer.review(row)["status"] == "error"
    document = store.finish_calls[0][4]
    assert document["error_code"] == "code_review_provider_request_rejected"
    assert document["provider_http_status"] == 200
    assert document["retryable"] is False
    assert "categories" not in document and "passed" not in document
    assert "private provider refusal prose" not in json.dumps(document)
    assert store.finish_calls[0][5] == actual
    assert len(transport.sent) == 1
    # This is the actual admission read-time policy, independent of telemetry.
    assert not code_review_policy.review_retry_available({
        "code_review_status": "error", "code_review_attempts": 1,
        "code_review_doc": document,
    })


@pytest.mark.parametrize("change", [
    lambda body: body.update(choices=[]),
    lambda body: body.update(choices=body["choices"] * 2),
    lambda body: body.update(choices=[None]),
    lambda body: body["choices"][0].update(message=None),
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


@pytest.mark.parametrize("cost,kind", [("0.001234", "settlement"), (None, "uncertain")])
def test_real_ledger_refusal_keeps_charge_and_stops_next_paid_claim(recovery_store, recovery_superuser, cost, kind):
    _, submission, miner = accepted_recovery_submission(recovery_store, "f71" if cost else "f72")
    payload = _archive(_source())
    transport = ReviewTransport(lambda parameters: {"body": refusal_body(parameters, cost=cost)})
    reviewer, _, _, _ = _reviewer(payload, transport, store=recovery_store)
    row = {**recovery_store.get_submission(submission), "source_size_bytes": len(payload)}
    assert reviewer.review(row)["status"] == "error"
    stored = recovery_store.get_submission(submission)
    assert stored["code_review_doc"]["error_code"] == "code_review_provider_request_rejected"
    assert stored["code_review_doc"]["retryable"] is False
    assert "provider_diagnostic" not in stored["code_review_doc"]
    ledger = recovery_store.list_ledger(submission_id=submission)
    assert [entry["entry_kind"] for entry in ledger] == ["reservation", "dispatch", kind]
    assert ledger[-1]["amount_microusd"] == (1234 if cost else ledger[0]["amount_microusd"])
    assert ledger[-1]["entry_doc"]["provider_diagnostic"]["finish_reason"] == "content_filter"
    _age_review_error(recovery_superuser, submission)
    assert reviewer.review(recovery_store.get_submission(submission))["status"] == "exhausted"
    assert recovery_store.begin_submission_review(
        submission, miner, "b" * 64, 50_000, code_review.DEFAULT_REVIEW_MODEL, 3, 1200,
    )["status"] == "exhausted"
    assert recovery_store.list_ledger(submission_id=submission) == ledger
    assert len(transport.sent) == 1
