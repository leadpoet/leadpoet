"""Real PostgreSQL persistence and private boundary for reviewer observations."""

import json

import pytest

from lab_arena import code_review, public_dashboard
from lab_arena.store import ArenaStoreError, new_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_code_review_postgres import (
    accepted_recovery_submission, recovery_connect, recovery_store, recovery_superuser,
)
from tests.postgres_migration_harness import SCRIPTS


SIGNATURE = "public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)"
MIGRATION = SCRIPTS / "594-lab-arena-private-review-diagnostics.sql"
BASE = CURRENT_SERVICE_MIGRATIONS + (
    "492-lab-arena-review-provider-observations.sql",
    "562-lab-arena-review-context-integrity-reason.sql",
    "563-lab-arena-review-source-digest-reason.sql",
    "566-lab-arena-research-review-result.sql",
    "585-lab-arena-review-deadline-recovery.sql",
)
DIAGNOSTIC = {
    "finish_reason": "length", "native_finish_reason": "max_tokens",
    "max_output_tokens": 16384, "prompt_tokens": 579839,
    "completion_tokens": 16384, "reasoning_tokens": 1000,
    "context_integrity": "verified", "response_model": code_review.DEFAULT_REVIEW_MODEL,
    "selected_model": "anthropic/claude-sonnet-5.5-20260928",
    "selected_provider": "Google", "generation_id": "gen-123-abc",
}


@pytest.fixture
def recovery_database():
    yield from database_with_lab_arena_migration(BASE + (MIGRATION.name,))


def _claim(store, suffix):
    _round, submission, miner = accepted_recovery_submission(store, suffix)
    token = new_lease_token()
    assert store.begin_submission_review(
        submission, miner, token, 50_000, code_review.DEFAULT_REVIEW_MODEL, 3, 1200,
    )["status"] == "claimed"
    return submission, miner, token


def _document(status="error"):
    doc = {"model": code_review.DEFAULT_REVIEW_MODEL, "file_count": 3, "source_bytes": 1200}
    if status == "error":
        doc.update(error_code="review_response_invalid", error_reason="not_finished", retryable=True)
    else:
        doc.update(passed=status == "passed", verdict="pass" if status == "passed" else "reject", categories=[])
    return doc


@pytest.mark.parametrize("status,actual", [("passed", 1234), ("rejected", 1234), ("error", 1234), ("error", None)])
def test_diagnostics_persist_only_in_per_attempt_private_ledger(recovery_store, status, actual):
    submission, miner, token = _claim(recovery_store, "f%02x" % ([('passed',1234),('rejected',1234),('error',1234),('error',None)].index((status,actual))))
    doc = _document(status)
    result = recovery_store.finish_submission_review(
        submission, miner, token, status, {**doc, "provider_diagnostic": DIAGNOSTIC}, actual,
    )
    assert result["status"] == status
    row = recovery_store.get_submission(submission)
    assert "provider_diagnostic" not in row["code_review_doc"]
    ledger = recovery_store.list_ledger(submission_id=submission)
    assert len(ledger) == 3
    terminal = ledger[-1]
    assert terminal["entry_doc"]["provider_diagnostic"] == DIAGNOSTIC
    assert terminal["entry_kind"] == ("uncertain" if actual is None else "settlement")
    assert terminal["amount_microusd"] == (50_000 if actual is None else actual)
    assert "provider_diagnostic" not in json.dumps(public_dashboard.code_review_summary(row))
    before = json.dumps(ledger, sort_keys=True, default=str)
    assert recovery_store.finish_submission_review(
        submission, miner, token, status, {**doc, "provider_diagnostic": DIAGNOSTIC}, actual,
    )["status"] == "existing"
    assert json.dumps(recovery_store.list_ledger(submission_id=submission), sort_keys=True, default=str) == before


@pytest.mark.parametrize("index,field,value", [
    (10, "finish_reason", "private provider prose"),
    (11, "native_finish_reason", "unknown"), (12, "context_integrity", "private"),
    (13, "prompt_tokens", True), (14, "completion_tokens", -1),
    (15, "reasoning_tokens", 2.5), (16, "prompt_tokens", 10**100),
    (17, "max_output_tokens", 0), (18, "selected_provider", "private provider"),
    (19, "selected_model", "private/model"), (20, "generation_id", "sk-private-value"),
    (21, "generation_id", "gen-123-" + "a" * 129),
    (22, "raw_response", "private provider prose"),
    (23, "finish_reason", "x" * 3000), (24, "completion_tokens", None),
])
def test_invalid_diagnostic_cannot_cross_database_edge(recovery_store, index, field, value):
    submission, miner, token = _claim(recovery_store, "f%02x" % index)
    diagnostic = {**DIAGNOSTIC, field: value}
    with pytest.raises(ArenaStoreError, match="lab_arena_code_review_input_invalid"):
        recovery_store.finish_submission_review(
            submission, miner, token, "error",
            {**_document(), "provider_diagnostic": diagnostic}, 1234,
        )
    assert recovery_store.get_submission(submission)["code_review_status"] == "reviewing"
    assert len(recovery_store.list_ledger(submission_id=submission)) == 2
    # The same old caller can finish unchanged after a refused diagnostic.
    assert recovery_store.finish_submission_review(
        submission, miner, token, "error", _document(), 1234,
    )["status"] == "error"


def test_upgrade_reapply_and_failed_upgrade_preserve_function_security():
    server = database_with_lab_arena_migration(BASE)
    try:
        psycopg2, dsn = next(server)
        with psycopg2.connect(**dsn) as connection:
            connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.execute("CREATE ROLE review594_migrator NOLOGIN NOSUPERUSER INHERIT")
                cursor.execute("GRANT lab_arena_owner TO review594_migrator")
                cursor.execute("ALTER SCHEMA public OWNER TO review594_migrator")
                cursor.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC,lab_arena_owner")
                cursor.execute("SET ROLE review594_migrator")
                cursor.execute("SELECT rolsuper,has_schema_privilege('lab_arena_owner','public','CREATE') "
                               "FROM pg_roles WHERE rolname=current_user")
                assert cursor.fetchone() == (False, False)
                def state():
                    cursor.execute("SELECT pg_get_functiondef(oid),to_jsonb(p)-'prosrc' "
                                   "FROM pg_proc p WHERE oid=to_regprocedure(%s)", (SIGNATURE,))
                    return cursor.fetchone()
                before_definition, before_metadata = state()
                failed = MIGRATION.read_text().replace(
                    "  EXECUTE v_updated;", "  EXECUTE v_updated;\n  PERFORM 1/0;", 1,
                )
                with pytest.raises(psycopg2.errors.DivisionByZero):
                    cursor.execute(failed)
                cursor.execute("ROLLBACK")
                assert state() == (before_definition, before_metadata)
                cursor.execute(MIGRATION.read_text())
                after_definition, after_metadata = state()
                assert after_definition != before_definition
                assert after_metadata == before_metadata
                cursor.execute(MIGRATION.read_text())
                assert state() == (after_definition, after_metadata)
                cursor.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE')")
                assert cursor.fetchone() == (False,)
                for role in ("anon", "authenticated", "service_role"):
                    cursor.execute("SELECT has_function_privilege(%s,%s,'EXECUTE'),"
                                   "has_table_privilege(%s,'public.lab_arena_ledger','SELECT')",
                                   (role, SIGNATURE, role))
                    assert cursor.fetchone() == (False, False)
    finally:
        server.close()


def test_database_vocabulary_matches_python_projection():
    from lab_arena import code_review_policy

    migration = MIGRATION.read_text()
    for value in (code_review_policy.REVIEW_FINISH_REASONS
                  | code_review_policy.REVIEW_CONTEXT_REASONS
                  | code_review_policy.REVIEW_PROVIDER_NAMES):
        assert "'" + value + "'" in migration


def test_real_runtime_keeps_each_attempt_diagnostic_without_changing_recovery(recovery_store, recovery_superuser):
    from tests.lab_arena.review_private_diagnostics_test import _response
    from tests.lab_arena.test_lab_arena_code_review_runtime import ReviewTransport, _archive, _reviewer, _source
    from tests.lab_arena.test_lab_arena_code_review_postgres import _age_review_error

    _round, submission, _miner = accepted_recovery_submission(recovery_store, "f30")
    payload = _archive(_source())
    calls = 0
    def respond(parameters):
        nonlocal calls
        calls += 1
        body = _response(parameters)
        if calls == 1:
            body["choices"][0]["finish_reason"] = "length"
        return {"body": body}
    reviewer, _row, _store, _objects = _reviewer(payload, ReviewTransport(respond), store=recovery_store)
    for status in ("error", "passed"):
        row = {**recovery_store.get_submission(submission), "source_size_bytes": len(payload)}
        assert reviewer.review(row)["status"] == status
        if status == "error":
            assert reviewer.review(recovery_store.get_submission(submission))["status"] == "backoff"
            assert calls == 1
            _age_review_error(recovery_superuser, submission)
    ledger = recovery_store.list_ledger(submission_id=submission)
    terminal = [r for r in ledger if r["entry_kind"] == "settlement"]
    assert [r["entry_doc"]["provider_diagnostic"]["finish_reason"] for r in terminal] == ["length", "stop"]
    row = recovery_store.get_submission(submission)
    assert row["code_review_doc"]["review_cost_microusd"] == 2468
    assert "provider_diagnostic" not in row["code_review_doc"]


def test_existing_deadline_and_long_retry_controls(recovery_store, recovery_superuser):
    from tests.lab_arena.unfinished_review_recovery585_postgres_test import (
        test_runtime_recovers_after_old_caps_and_reuses_completed_review,
        test_deadline_stops_claims_and_late_admission,
        test_source_accounting_and_negative_verdict_keep_their_limits,
    )
    test_runtime_recovers_after_old_caps_and_reuses_completed_review(recovery_store, recovery_superuser)
    test_deadline_stops_claims_and_late_admission(recovery_store, recovery_superuser)
    for kind in ("source", "accounting", "rejected"):
        test_source_accounting_and_negative_verdict_keep_their_limits(recovery_store, recovery_superuser, kind)

