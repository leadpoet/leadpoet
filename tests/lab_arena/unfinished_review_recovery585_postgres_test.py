"""Real review claims recover without replacing valid verdicts or admission gates."""

import pytest

from lab_arena import code_review
from lab_arena.store import ArenaStoreError, new_lease_token
from tests.lab_arena.lab_arena_pg_harness import CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration
from tests.lab_arena.test_lab_arena_code_review_postgres import (
    _age_review_error, accepted_recovery_submission, recovery_connect,
    recovery_store, recovery_superuser, review_doc,
)
from tests.lab_arena.test_lab_arena_code_review_runtime import (
    ReviewTransport, _archive, _review_body, _reviewer, _source,
)
from tests.postgres_migration_harness import SCRIPTS


MIGRATION = SCRIPTS / "585-lab-arena-review-deadline-recovery.sql"
ROLLBACK = SCRIPTS / "585-lab-arena-review-deadline-recovery-rollback.sql.template"


@pytest.fixture(scope="module")
def recovery_database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS + (
        "492-lab-arena-review-provider-observations.sql",
        "562-lab-arena-review-context-integrity-reason.sql",
        "563-lab-arena-review-source-digest-reason.sql",
        "566-lab-arena-research-review-result.sql", MIGRATION.name,
    ))


def _begin(store, submission, miner, token=None):
    return store.begin_submission_review(
        submission, miner, token or new_lease_token(), 50_000,
        code_review.DEFAULT_REVIEW_MODEL, 2, 1000,
    )


def _failure(*, error="review_response_invalid", retryable=False):
    return {
        "model": code_review.DEFAULT_REVIEW_MODEL, "file_count": 2,
        "source_bytes": 1000, "error_code": error,
        "error_reason": "not_finished", "retryable": retryable,
    }


def _verdict(verdict):
    return {**review_doc(verdict, file_count=2, source_bytes=1000), "model": code_review.DEFAULT_REVIEW_MODEL}


def test_runtime_recovers_after_old_caps_and_reuses_completed_review(recovery_store, recovery_superuser):
    _, submission, _miner = accepted_recovery_submission(recovery_store, "e20")
    payload = _archive(_source())
    responses = 0
    def response(parameters):
        nonlocal responses
        responses += 1
        body = _review_body(parameters)
        if responses <= 7:
            body["choices"][0]["finish_reason"] = "length"
        return {"body": body}
    transport = ReviewTransport(response)
    reviewer, _, _, _ = _reviewer(payload, transport, store=recovery_store)
    for attempt in range(1, 9):
        row = {**recovery_store.get_submission(submission), "source_size_bytes": len(payload)}
        assert reviewer.review(row)["status"] == ("error" if attempt < 8 else "passed")
        assert len(transport.sent) == attempt
        if attempt < 8:
            row = recovery_store.get_submission(submission)
            assert row["status"] == "accepted"
            assert reviewer.review(row)["status"] == "backoff"
            assert len(transport.sent) == attempt
            _age_review_error(recovery_superuser, submission)
    ledger_before = recovery_store.list_ledger(submission_id=submission)
    assert reviewer.review(recovery_store.get_submission(submission))["status"] == "existing"
    assert len(transport.sent) == 8
    assert recovery_store.list_ledger(submission_id=submission) == ledger_before
    assert recovery_store.get_submission(submission)["code_review_doc"]["review_cost_microusd"] == 8 * 1234


@pytest.mark.parametrize("flag", [False, True, None])
def test_old_diagnostic_flags_do_not_make_failure_a_verdict(recovery_store, recovery_superuser, flag):
    _, submission, miner = accepted_recovery_submission(recovery_store, {False: "e21", True: "e22", None: "e23"}[flag])
    for attempt in range(1, 8):
        token = new_lease_token()
        assert _begin(recovery_store, submission, miner, token)["attempt"] == attempt
        doc = _failure(retryable=flag)
        if flag is None:
            doc.pop("retryable")
        assert recovery_store.finish_submission_review(submission, miner, token, "error", doc, 100)["status"] == "error"
        before = recovery_store.list_ledger(submission_id=submission)
        assert _begin(recovery_store, submission, miner)["status"] == "backoff"
        assert recovery_store.list_ledger(submission_id=submission) == before
        _age_review_error(recovery_superuser, submission)


@pytest.mark.parametrize("kind", ["source", "accounting", "rejected"])
def test_source_accounting_and_negative_verdict_keep_their_limits(recovery_store, recovery_superuser, kind):
    _, submission, miner = accepted_recovery_submission(recovery_store, {"source": "e24", "accounting": "e25", "rejected": "e26"}[kind])
    attempts = 3 if kind == "accounting" else 1
    for _ in range(attempts):
        token = new_lease_token()
        assert _begin(recovery_store, submission, miner, token)["status"] == "claimed"
        if kind == "rejected":
            status, doc = "rejected", _verdict("reject")
        else:
            status, doc = "error", _failure(error="source_archive_invalid" if kind == "source" else "code_review_cost_unavailable")
            doc.pop("error_reason")
            if kind == "accounting":
                doc.pop("retryable")
        recovery_store.finish_submission_review(submission, miner, token, status, doc, None if kind == "accounting" else 1)
        _age_review_error(recovery_superuser, submission)
    before = recovery_store.list_ledger(submission_id=submission)
    assert _begin(recovery_store, submission, miner)["status"] == ("existing" if kind == "rejected" else "exhausted")
    assert recovery_store.list_ledger(submission_id=submission) == before


def test_deadline_stops_claims_and_late_admission(recovery_store, recovery_superuser):
    round_id, submission, miner = accepted_recovery_submission(recovery_store, "e27")
    token = new_lease_token()
    assert _begin(recovery_store, submission, miner, token)["status"] == "claimed"
    with recovery_superuser.cursor() as cursor:
        cursor.execute("SET session_replication_role='replica'")
        try:
            cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,benchmark_deadline}',to_jsonb((clock_timestamp()-interval '1 second')::text)) WHERE round_id=%s", (round_id,))
        finally:
            cursor.execute("SET session_replication_role='origin'")
    assert recovery_store.finish_submission_review(submission, miner, token, "passed", _verdict("pass"), 1)["status"] == "deadline"
    before = recovery_store.list_ledger(submission_id=submission)
    assert _begin(recovery_store, submission, miner)["status"] == "deadline"
    assert recovery_store.list_ledger(submission_id=submission) == before


def test_interrupted_eighth_claim_keeps_uncertain_charge_and_new_identity(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, "e28")
    for attempt in range(1, 9):
        token = new_lease_token()
        assert _begin(recovery_store, submission, miner, token)["attempt"] == attempt
        if attempt < 8:
            recovery_store.finish_submission_review(submission, miner, token, "error", _failure(), 100)
            _age_review_error(recovery_superuser, submission)
    with recovery_superuser.cursor() as cursor:
        cursor.execute("UPDATE public.lab_arena_submissions SET code_review_started_at=clock_timestamp()-interval '601 seconds',code_review_expires_at=clock_timestamp()-interval '1 second' WHERE submission_id=%s", (submission,))
    assert _begin(recovery_store, submission, miner)["status"] == "recovered"
    ledger = recovery_store.list_ledger(submission_id=submission)
    assert ledger[-1]["entry_kind"] == "uncertain"
    assert ledger[-1]["amount_microusd"] == 50_000
    assert _begin(recovery_store, submission, miner)["status"] == "backoff"
    assert recovery_store.list_ledger(submission_id=submission) == ledger
    _age_review_error(recovery_superuser, submission)
    assert _begin(recovery_store, submission, miner)["attempt"] == 9
    assert len({entry["call_identity"] for entry in recovery_store.list_ledger(submission_id=submission)}) == 9


def test_rejected_admission_cannot_be_reopened_by_retry_policy(recovery_store):
    round_id, submission, miner = accepted_recovery_submission(recovery_store, "e29")
    token = new_lease_token()
    assert _begin(recovery_store, submission, miner, token)["status"] == "claimed"
    recovery_store.finish_submission_review(submission, miner, token, "error", _failure(), 100)
    assert recovery_store.update_submission(round_id, submission, "accepted", "rejected", {"rejection_rule": "code_review_incomplete"})["status"] == "ok"
    before = recovery_store.list_ledger(submission_id=submission)
    with pytest.raises(ArenaStoreError, match="lab_arena_submission_missing"):
        _begin(recovery_store, submission, miner)
    assert recovery_store.list_ledger(submission_id=submission) == before


def test_migration_replay_preserves_functions_and_rows(recovery_store, recovery_superuser):
    with recovery_superuser.cursor() as cursor:
        cursor.execute("SELECT jsonb_agg(to_jsonb(p) ORDER BY p.oid) FROM pg_proc p WHERE p.proname IN ('lab_arena_begin_submission_review','lab_arena_finish_submission_review','lab_arena_code_review_schema_v1')")
        before = cursor.fetchone()[0]
        cursor.execute("SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM public.lab_arena_submissions s")
        rows = cursor.fetchone()[0]
        cursor.execute(MIGRATION.read_text())
        cursor.execute("SELECT jsonb_agg(to_jsonb(p) ORDER BY p.oid) FROM pg_proc p WHERE p.proname IN ('lab_arena_begin_submission_review','lab_arena_finish_submission_review','lab_arena_code_review_schema_v1')")
        assert cursor.fetchone()[0] == before
        cursor.execute("SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM public.lab_arena_submissions s")
        assert cursor.fetchone()[0] == rows
    assert recovery_store.code_review_schema()["retry_policy"] == "deadline_transient_v2"


def test_migration_refuses_unreviewed_function_without_rewriting_rows(recovery_superuser):
    with recovery_superuser.cursor() as cursor:
        cursor.execute("SELECT pg_get_functiondef('public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)'::regprocedure)")
        definition = cursor.fetchone()[0]
        try:
            cursor.execute(definition.replace("DECLARE", "DECLARE -- unreviewed change", 1))
            with pytest.raises(Exception, match="Arena review585 preimage differs"):
                cursor.execute(MIGRATION.read_text())
            cursor.execute("ROLLBACK")
        finally:
            cursor.execute("ROLLBACK")
            cursor.execute(definition)


def test_truthful_schema_rollback_preserves_work_and_resumes_counter_past_six(recovery_store, recovery_superuser):
    _, submission, miner = accepted_recovery_submission(recovery_store, "e30")
    for attempt in range(1, 8):
        token = new_lease_token()
        assert _begin(recovery_store, submission, miner, token)["attempt"] == attempt
        recovery_store.finish_submission_review(submission, miner, token, "error", _failure(), 100)
        _age_review_error(recovery_superuser, submission)
    _, completed, completed_miner = accepted_recovery_submission(recovery_store, "e31")
    token = new_lease_token()
    assert _begin(recovery_store, completed, completed_miner, token)["status"] == "claimed"
    recovery_store.finish_submission_review(completed, completed_miner, token, "passed", _verdict("pass"), 100)
    with recovery_superuser.cursor() as cursor:
        cursor.execute("SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM public.lab_arena_submissions s")
        rows = cursor.fetchone()[0]
        cursor.execute("SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l")
        ledger = cursor.fetchone()[0]
        try:
            cursor.execute(ROLLBACK.read_text())
            cursor.execute(ROLLBACK.read_text())
            schema = recovery_store._transport.rpc("lab_arena_code_review_schema_v1", {})
            assert schema["retry_policy"] == "bounded_transient_v1"
            assert schema["max_transient_attempts"] == 6
            with pytest.raises(ArenaStoreError, match="code review schema mismatch"):
                recovery_store.code_review_schema()
            assert _begin(recovery_store, submission, miner)["status"] == "exhausted"
            assert _begin(recovery_store, completed, completed_miner)["status"] == "existing"
            cursor.execute("SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM public.lab_arena_submissions s")
            assert cursor.fetchone()[0] == rows
            cursor.execute("SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l")
            assert cursor.fetchone()[0] == ledger
        finally:
            cursor.execute(MIGRATION.read_text())
    assert recovery_store.code_review_schema()["retry_policy"] == "deadline_transient_v2"
    assert _begin(recovery_store, submission, miner)["attempt"] == 8
