"""Trusted admission traces preserve the real review and committed outcome."""

import json

import pytest

from lab_arena import broker
from tests.lab_arena.test_lab_arena_code_review_runtime import (
    MINER_KEY, ReviewTransport, _archive, _review_body, _reviewer, _source,
)


class Capture:
    def __init__(self, fail=None):
        self.events = []
        self.fail = fail

    def _record(self, stage, data):
        if self.fail == stage:
            raise OSError("capture unavailable")
        self.events.append((stage, data))

    def start(self, **data):
        self._record("request", data)
        return self

    def response(self, **data):
        self._record("response", data)

    def finish(self, **data):
        self._record("outcome", data)


def run(capture, *, response=None, finish=None):
    transport = ReviewTransport(response or (lambda p: {"body": _review_body(p)}))
    reviewer, row, store, _ = _reviewer(_archive(_source()), transport)
    reviewer._capture = capture
    if finish:
        store.finish_submission_review = finish
    result = reviewer.review(row)
    return result, store, transport


def test_request_response_and_committed_result_survive_separately():
    capture = Capture()
    def respond(p):
        value = _review_body(p)
        value["choices"][0]["message"]["reasoning"] = "Inspect every reachable source path."
        return {"body": value}
    result, store, transport = run(capture, response=respond)
    assert result["status"] == "passed"
    assert [stage for stage, _ in capture.events] == ["request", "response", "outcome"]
    request, response, outcome = [data for _, data in capture.events]
    assert request["request"] == json.loads(transport.sent[0]["body"])
    assert request["secrets"] == (MINER_KEY,)
    assert request["purpose"] == "admission_review"
    assert request["context"]["attempt"] == 1
    assert request["context"]["submission_id"] == "miner-submission"
    assert request["context"]["source_archive_sha256"].startswith("sha256:")
    assert json.loads(response["body"])["choices"][0]["message"]["reasoning"]
    assert response["status"] == 200 and response["elapsed_seconds"] >= 0
    assert outcome["outcome"]["committed_review"] == result
    assert outcome["outcome"]["host_decision"] == store.finish_calls[0][4]
    assert outcome["cost_microusd"] == 1234
    assert "Inspect every" not in json.dumps(store.finish_calls)


def test_late_provider_pass_is_not_exported_as_committed_pass():
    capture = Capture()
    result, _, _ = run(capture, finish=lambda *a: {
        "status": "deadline", "code_review_status": "error", "attempt": 1,
    })
    outcome = capture.events[-1][1]["outcome"]
    assert outcome["host_decision"]["passed"] is True
    assert outcome["committed_review"] == result
    assert result["code_review_status"] == "error"


@pytest.mark.parametrize("fail", ["request", "response", "outcome"])
def test_capture_failure_never_changes_admission_or_bill(fail):
    result, store, transport = run(Capture(fail))
    assert result["status"] == "passed"
    assert len(transport.sent) == len(store.begin_calls) == len(store.finish_calls) == 1
    assert store.finish_calls[0][5] == 1234


def test_invalid_response_is_retained_without_fabricating_a_verdict():
    capture = Capture()
    result, _, _ = run(capture, response=lambda p: {"body": {"unexpected": "reply"}})
    assert result["status"] == "error"
    assert json.loads(capture.events[1][1]["body"]) == {"unexpected": "reply"}
    assert capture.events[-1][1]["cost_microusd"] is None


def test_uncommitted_database_failure_is_explicit_and_still_raises():
    capture = Capture()
    def failed_finish(*args):
        raise RuntimeError("database unavailable")
    with pytest.raises(RuntimeError, match="database unavailable"):
        run(capture, finish=failed_finish)
    assert capture.events[-1][1]["outcome"]["committed_review"] == {"status": "unavailable"}


def test_transport_failure_retains_request_and_uncertain_cost():
    capture = Capture()
    def failed_response(p):
        raise broker.ProviderTransportError("request failed")
    result, _, _ = run(capture, response=failed_response)
    assert result["status"] == "error"
    assert [stage for stage, _ in capture.events] == ["request", "outcome"]
    assert capture.events[-1][1]["cost_microusd"] is None
