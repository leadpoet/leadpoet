"""Provider failure is an unfinished review, never a completed code verdict."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from lab_arena import code_review_policy
from lab_arena import public_dashboard
from lab_arena.service import ArenaService, ServiceError
from tests.lab_arena.test_lab_arena_code_review_runtime import (
    ReviewTransport, _archive, _review_body, _reviewer, _source,
)


@pytest.mark.parametrize("attempts", [3, 6, 10])
@pytest.mark.parametrize("retryable", [None, False, True])
def test_saved_unfinished_review_stays_recoverable(attempts, retryable):
    diagnostic = {
        "error_code": "review_response_invalid", "error_reason": "not_finished",
    }
    if retryable is not None:
        diagnostic["retryable"] = retryable
    started = datetime.now(timezone.utc)
    row = {
        "code_review_status": "error", "code_review_attempts": attempts,
        "code_review_started_at": started, "code_review_doc": diagnostic,
    }
    assert code_review_policy.review_retry_available(row)
    assert code_review_policy.retry_ready_at(row) == started + timedelta(
        seconds=code_review_policy.TRANSIENT_BACKOFF_SECONDS[
            min(attempts - 1, len(code_review_policy.TRANSIENT_BACKOFF_SECONDS) - 1)
        ]
    )


@pytest.mark.parametrize("finish_reason", ["length", "error", None])
def test_unfinished_provider_reply_remains_error_and_requests_recovery(finish_reason):
    def response(parameters):
        body = _review_body(parameters)
        body["choices"][0]["finish_reason"] = finish_reason
        return {"body": body}

    transport = ReviewTransport(response)
    reviewer, row, store, _ = _reviewer(_archive(_source()), transport)
    assert reviewer.review(row)["status"] == "error"
    assert len(transport.sent) == 1
    diagnostic = store.finish_calls[0][4]
    assert diagnostic["error_code"] == "review_response_invalid"
    assert diagnostic["error_reason"] == ("envelope" if finish_reason == "error" else "not_finished")
    assert diagnostic["retryable"] is True
    assert store.finish_calls[0][5] == 1234


@pytest.mark.parametrize("status", ["passed", "rejected"])
def test_completed_review_has_no_repeat_provider_call(status):
    transport = ReviewTransport(lambda _parameters: pytest.fail("completed review dispatched"))
    reviewer, row, store, objects = _reviewer(_archive(_source()), transport)
    row["code_review_status"] = status
    assert reviewer.review(row) == {"status": "existing", "code_review_status": status}
    assert not transport.sent
    assert not store.begin_calls
    assert not objects.calls


def test_capped_backoff_bounds_dispatch_in_the_ninety_minute_window():
    # Zero provider latency gives the greatest possible number of starts.
    start = datetime(2026, 10, 27, 23, tzinfo=timezone.utc)
    deadline = start + timedelta(minutes=90)
    moment, attempts = start, 0
    while moment < deadline:
        attempts += 1
        moment = code_review_policy.retry_ready_at({
            "code_review_status": "error", "code_review_attempts": attempts,
            "code_review_started_at": moment,
            "code_review_doc": {"error_code": "review_response_invalid"},
        })
    assert attempts == 9
    assert moment == deadline


@pytest.mark.parametrize("code", sorted(code_review_policy.TRANSIENT_ERROR_CODES))
def test_only_known_reviewer_failures_override_stale_flags(code):
    assert code_review_policy.review_retry_available({
        "code_review_status": "error", "code_review_attempts": 20,
        "code_review_doc": {"error_code": code, "retryable": False},
    })


@pytest.mark.parametrize("code", sorted(code_review_policy.SOURCE_ERROR_CODES | {
    "code_review_provider_authentication", "code_review_provider_credit",
    "code_review_provider_request_rejected", "code_review_credential_echo",
}))
def test_source_and_credential_failures_cannot_gain_unlimited_retries(code):
    assert not code_review_policy.review_retry_available({
        "code_review_status": "error", "code_review_attempts": 1,
        "code_review_doc": {"error_code": code, "retryable": False},
    })


def test_accounting_uncertainty_keeps_legacy_limit():
    for attempts, expected in ((1, True), (2, True), (3, False), (8, False)):
        assert code_review_policy.review_retry_available({
            "code_review_status": "error", "code_review_attempts": attempts,
            "code_review_doc": {"error_code": "code_review_cost_unavailable"},
        }) is expected


@pytest.mark.parametrize("expired", [False, True])
def test_admission_waits_for_unfinished_review_until_deadline_then_excludes(expired):
    now = datetime.now(timezone.utc)
    round_id = "arena-2026-10-28"
    baseline = {"submission_id": "baseline-2026-10-28", "is_king": True, "miner_hotkey": "baseline"}
    unfinished = {
        "submission_id": "unfinished", "code_review_status": "error",
        "code_review_attempts": 10,
        "code_review_doc": {"error_code": "review_response_invalid", "error_reason": "not_finished", "retryable": False},
    }
    updates = []
    round_row = {"round_id": round_id, "champion_funding_frozen": True, "configuration_doc": {
        "baseline_hotkey": "baseline",
        "schedule": {"benchmark_deadline": (now + timedelta(seconds=-1 if expired else 60)).isoformat()},
    }}
    service = object.__new__(ArenaService)
    service._store = SimpleNamespace(
        freeze_champion_funding=lambda _id: {"status": "existing"},
        list_submissions=lambda _id, *, status: [baseline] if status == "frozen" else [unfinished],
        update_submission=lambda *args: updates.append(args) or {"status": "ok"},
    )
    service._round = lambda _id: round_row
    service.now = lambda: now
    service._verify_full_thinking_baseline_source = lambda *_args: None
    service._participant = lambda row, **_kwargs: dict(row)
    if expired:
        assert service.freeze_participants(round_id) == [baseline]
        assert updates[0][-1] == {"rejection_rule": "code_review_incomplete"}
    else:
        with pytest.raises(ServiceError, match="code_review_pending"):
            service.freeze_participants(round_id)
        assert not updates
    summary = public_dashboard.code_review_summary(
        {**unfinished, "status": "accepted"},
        round_row={**round_row, "status": "open"}, now=now,
    )
    assert summary["retryable"] is (not expired)
