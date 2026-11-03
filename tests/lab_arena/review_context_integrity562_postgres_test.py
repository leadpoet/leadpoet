"""Exercise the exact review diagnostic migration through real PostgreSQL."""

from datetime import datetime, timedelta, timezone
import hashlib

import pytest

from lab_arena import code_review, code_review_policy
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport, new_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    encrypted_runtime_credentials, hotkey, round_config, source_submission_doc,
)
from tests.postgres_migration_harness import SCRIPTS


MIGRATION = SCRIPTS / "562-lab-arena-review-context-integrity-reason.sql"
SIGNATURE = "public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS + ("492-lab-arena-review-provider-observations.sql",)
    )


@pytest.fixture()
def context(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    store = ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))
    try:
        yield store, connection
    finally:
        store.close()
        connection.close()


def _claimed(store, suffix):
    now = datetime.now(timezone.utc)
    round_id = "arena-2099-03-01-" + hashlib.sha256(suffix.encode()).hexdigest()[:16]
    submission_id = suffix + "-submission"
    miner = hotkey("review562-" + suffix)
    config = round_config(round_id, [hotkey("runner562-" + suffix)])
    config["schedule"].update({
        "submission_open": (now - timedelta(hours=2)).isoformat(),
        "submission_cutoff": (now + timedelta(minutes=30)).isoformat(),
        "benchmark_deadline": (now + timedelta(hours=1)).isoformat(),
    })
    assert store.create_round(round_id, config)["status"] == "created"
    assert store.register_submission(
        round_id, submission_id, miner, source_submission_doc(round_id, submission_id),
    )["status"] == "registered"
    assert store.accept_submission_with_credentials(
        round_id, submission_id, miner, encrypted_runtime_credentials(submission_id),
    )["status"] == "ok"
    token = new_lease_token()
    assert store.begin_submission_review(
        submission_id, miner, token, 50_000, code_review.DEFAULT_REVIEW_MODEL, 71, 2_118_944,
    )["status"] == "claimed"
    return submission_id, miner, token


def _error(reason):
    return {
        **code_review_policy.diagnostic("review_response_invalid", error_reason=reason),
        "model": code_review.DEFAULT_REVIEW_MODEL, "file_count": 71, "source_bytes": 2_118_944,
    }


def test_context_integrity_fails_before_migration_then_settles_same_claim(context):
    store, connection = context
    submission, miner, token = _claimed(store, "beforeafter")
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
        original = cursor.fetchone()[0]
        # Make this proof independent of test order and repeat runs.
        predecessor = original.replace(
            "'finding_evidence_mismatch', 'finding_explanation',\n             'context_integrity'",
            "'finding_evidence_mismatch', 'finding_explanation'",
        )
        cursor.execute(predecessor)
    with pytest.raises(ArenaStoreError, match="lab_arena_code_review_input_invalid"):
        store.finish_submission_review(submission, miner, token, "error", _error("context_integrity"), 12)
    assert store.get_submission(submission)["code_review_status"] == "reviewing"
    assert [item["entry_kind"] for item in store.list_ledger(submission_id=submission)] == [
        "reservation", "dispatch",
    ]
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
    assert store.finish_submission_review(
        submission, miner, token, "error", _error("context_integrity"), 12,
    )["status"] == "error"
    row = store.get_submission(submission)
    assert row["code_review_doc"]["error_reason"] == "context_integrity"
    assert row["code_review_doc"]["cost_microusd"] == 12
    assert "retryable" not in row["code_review_doc"]
    assert [item["entry_kind"] for item in store.list_ledger(submission_id=submission)] == [
        "reservation", "dispatch", "settlement",
    ]


# This fixture applies main's context migration alone. The research reasons
# enter in the later composed review migration and are checked there.
@pytest.mark.parametrize("reason", sorted(
    reason for reason in code_review_policy.RESPONSE_ERROR_REASONS
    if not reason.startswith("research_")
))
def test_every_canonical_response_reason_is_persisted(context, reason):
    store, connection = context
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
    submission, miner, token = _claimed(store, reason)
    assert store.finish_submission_review(
        submission, miner, token, "error", _error(reason), None,
    )["status"] == "error"
    row = store.get_submission(submission)
    assert row["code_review_doc"]["error_reason"] == reason
    assert row["code_review_doc"]["cost_status"] == "uncertain"
    assert row["code_review_doc"]["cost_microusd"] == 50_000
    assert "retryable" not in row["code_review_doc"]


@pytest.mark.parametrize("verdict", ["pass", "reject"])
def test_existing_verdicts_still_settle(context, verdict):
    store, connection = context
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
    submission, miner, token = _claimed(store, verdict)
    document = {**_error("envelope"), "passed": verdict == "pass", "verdict": verdict,
                "categories": [] if verdict == "pass" else ["duplicate_submission"]}
    document.pop("error_code")
    document.pop("error_reason")
    status = "passed" if verdict == "pass" else "rejected"
    assert store.finish_submission_review(submission, miner, token, status, document, 12)["status"] == status
    assert store.get_submission(submission)["code_review_status"] == status


def test_unexpected_reason_remains_rejected_without_settlement(context):
    store, connection = context
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
    submission, miner, token = _claimed(store, "unexpected")
    document = {**_error("envelope"), "error_reason": "private provider prose"}
    with pytest.raises(ArenaStoreError, match="lab_arena_code_review_input_invalid"):
        store.finish_submission_review(submission, miner, token, "error", document, 12)
    assert store.get_submission(submission)["code_review_status"] == "reviewing"
    assert len(store.list_ledger(submission_id=submission)) == 2


def test_migration_replay_preserves_security_and_refuses_unknown_definition(context):
    _, connection = context
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text())
        cursor.execute("SELECT pg_catalog.to_jsonb(p) FROM pg_catalog.pg_proc p WHERE p.oid=%s::regprocedure", (SIGNATURE,))
        before = cursor.fetchone()[0]
        cursor.execute(MIGRATION.read_text())
        cursor.execute(MIGRATION.read_text())
        cursor.execute("SELECT pg_catalog.to_jsonb(p) FROM pg_catalog.pg_proc p WHERE p.oid=%s::regprocedure", (SIGNATURE,))
        assert cursor.fetchone()[0] == before
        cursor.execute("SELECT has_function_privilege('lab_arena_service',%s,'EXECUTE'),has_function_privilege('service_role',%s,'EXECUTE'),has_function_privilege('anon',%s,'EXECUTE'),has_function_privilege('authenticated',%s,'EXECUTE')", (SIGNATURE,) * 4)
        assert cursor.fetchone() == (True, False, False, False)
        cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
        original = cursor.fetchone()[0]
        changed = original.replace("BEGIN\n", "BEGIN\n  -- unexpected fixture change\n", 1)
        try:
            cursor.execute(changed)
            with pytest.raises(Exception, match="Arena review562 preimage differs"):
                cursor.execute(MIGRATION.read_text())
            cursor.execute("ROLLBACK")
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
            assert cursor.fetchone()[0] == changed
        finally:
            cursor.execute("ROLLBACK")
            cursor.execute(original)
