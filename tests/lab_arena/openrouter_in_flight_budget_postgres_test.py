"""Exercise review classification through the real claim, ledger and retry RPCs."""

from lab_arena import code_review, code_review_policy
from lab_arena.store import new_lease_token
from tests.lab_arena.openrouter_in_flight_budget_test import budget_error
from tests.lab_arena.test_lab_arena_code_review_postgres import (
    _age_review_error, accepted_recovery_submission,
    recovery_connect, recovery_database, recovery_store, recovery_superuser,
)
from tests.lab_arena.test_lab_arena_code_review_runtime import (
    ReviewTransport, _archive, _review_body, _reviewer, _source,
)


def test_typed_budget_review_claim_backoff_retry_and_pass(recovery_store, recovery_superuser):
    _round_id, submission_id, miner = accepted_recovery_submission(recovery_store, "e1")
    payload = _archive(_source())
    attempts = []

    def respond(parameters):
        attempts.append(parameters)
        if len(attempts) == 1:
            return {"status": 402, "body": budget_error()}
        return {"status": 200, "body": _review_body(parameters)}

    transport = ReviewTransport(respond)
    reviewer, _row, _store, _objects = _reviewer(payload, transport, store=recovery_store)
    row = {**recovery_store.get_submission(submission_id), "source_size_bytes": len(payload)}
    assert reviewer.review(row)["status"] == "error"
    failed = recovery_store.get_submission(submission_id)
    assert failed["code_review_attempts"] == 1
    assert failed["code_review_doc"]["provider_http_status"] == 402
    assert failed["code_review_doc"]["error_code"] == "code_review_provider_rate_limited"
    assert failed["code_review_doc"]["retryable"] is True
    assert failed["code_review_doc"]["cost_status"] == "settled"
    assert failed["code_review_doc"]["cost_microusd"] == 0
    assert code_review_policy.review_retry_available(failed)
    assert reviewer.review(failed)["status"] == "backoff"
    reservation = recovery_store.list_ledger(submission_id=submission_id)[0]
    assert recovery_store.begin_submission_review(
        submission_id, miner, new_lease_token(), reservation["amount_microusd"],
        code_review.DEFAULT_REVIEW_MODEL, reservation["entry_doc"]["file_count"],
        reservation["entry_doc"]["source_bytes"],
    )["status"] == "backoff"
    _age_review_error(recovery_superuser, submission_id)
    row = {**recovery_store.get_submission(submission_id), "source_size_bytes": len(payload)}
    assert reviewer.review(row)["status"] == "passed"
    passed = recovery_store.get_submission(submission_id)
    assert passed["code_review_status"] == "passed"
    assert passed["code_review_attempts"] == 2
    assert passed["code_review_doc"]["review_cost_microusd"] == 1234
    assert reviewer.review(passed)["status"] == "existing"
    assert len(transport.sent) == 2
    ledger = recovery_store.list_ledger(submission_id=submission_id)
    assert [entry["entry_kind"] for entry in ledger] == [
        "reservation", "dispatch", "settlement", "reservation", "dispatch", "settlement",
    ]
    assert [entry["amount_microusd"] for entry in ledger if entry["entry_kind"] == "settlement"] == [0, 1234]
