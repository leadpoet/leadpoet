"""Sol observations remain private and do not grant review or cost authority."""

import json

import pytest

from lab_arena import public_dashboard
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport, new_lease_token
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration
from tests.lab_arena.review_private_diagnostics594_postgres_test import BASE, SIGNATURE
from tests.lab_arena.test_lab_arena_code_review_postgres import (
    accepted_recovery_submission, recovery_connect, recovery_store, recovery_superuser,
)
from tests.postgres_migration_harness import SCRIPTS


MIGRATION = SCRIPTS / "599-lab-arena-sol-review-diagnostics.sql"
PREREQUISITES = BASE + ("594-lab-arena-private-review-diagnostics.sql",)
MODEL = "openai/gpt-6.1-sol"
DIAGNOSTIC = {
    "finish_reason": "stop", "native_finish_reason": "stop",
    "max_output_tokens": 16384, "prompt_tokens": 585000,
    "completion_tokens": 1000, "reasoning_tokens": 200,
    "context_integrity": "verified", "response_model": MODEL,
    "selected_model": "openai/gpt-6.1-sol-20260929",
    "selected_provider": "Azure", "generation_id": "gen-123-sol",
}


@pytest.fixture(scope="module")
def recovery_database():
    yield from database_with_lab_arena_migration(PREREQUISITES + (MIGRATION.name,))


def claim(store, suffix, *, model=MODEL):
    _, submission, miner = accepted_recovery_submission(store, suffix)
    token = new_lease_token()
    assert store.begin_submission_review(
        submission, miner, token, 50_000, model, 3, 1200,
    )["status"] == "claimed"
    return submission, miner, token


def document(status="error", *, model=MODEL):
    doc = {"model": model, "file_count": 3, "source_bytes": 1200}
    if status == "error":
        doc.update(error_code="review_response_invalid", error_reason="not_finished", retryable=True)
    else:
        doc.update(passed=status == "passed", verdict="pass" if status == "passed" else "reject", categories=[])
    return doc


@pytest.mark.parametrize("index,status,actual", [
    (1, "passed", 1234), (2, "rejected", 1234), (3, "error", 1234), (4, "error", None),
])
def test_sol_diagnostic_is_private_accounted_and_idempotent(recovery_store, index, status, actual):
    submission, miner, token = claim(recovery_store, "a59%02d" % index)
    doc = {**document(status), "provider_diagnostic": DIAGNOSTIC}
    assert recovery_store.finish_submission_review(submission, miner, token, status, doc, actual)["status"] == status
    row = recovery_store.get_submission(submission)
    ledger = recovery_store.list_ledger(submission_id=submission)
    assert len(ledger) == 3
    assert ledger[-1]["entry_doc"]["provider_diagnostic"] == DIAGNOSTIC
    assert ledger[-1]["entry_kind"] == ("uncertain" if actual is None else "settlement")
    assert ledger[-1]["amount_microusd"] == (50_000 if actual is None else actual)
    assert "provider_diagnostic" not in row["code_review_doc"]
    assert "provider_diagnostic" not in json.dumps(public_dashboard.code_review_summary(row))
    assert recovery_store.finish_submission_review(submission, miner, token, status, doc, actual)["status"] == "existing"
    assert recovery_store.list_ledger(submission_id=submission) == ledger


@pytest.mark.parametrize("index,patch", [
    (10, {"selected_model": "openai/gpt-6.1-sol-20260930"}),
    (11, {"response_model": "openai/gpt-6.1"}),
    (12, {"selected_provider": "Azure private prose"}),
    (13, {"raw_response": "private response text"}),
    (14, {"reasoning_tokens": True}),
    (15, {"selected_provider": "OpenAI"}),
])
def test_nearby_models_and_unbounded_metadata_stay_discarded(recovery_store, index, patch):
    submission, miner, token = claim(recovery_store, "a59%02d" % index)
    assert recovery_store.finish_submission_review(
        submission, miner, token, "error",
        {**document(), "provider_diagnostic": {**DIAGNOSTIC, **patch}}, 1234,
    )["status"] == "error"
    terminal = recovery_store.list_ledger(submission_id=submission)[-1]
    assert terminal["amount_microusd"] == 1234
    assert "provider_diagnostic" not in terminal["entry_doc"]


def test_sol_metadata_cannot_override_bound_model_or_failure_policy(recovery_store):
    submission, miner, token = claim(recovery_store, "a5920")
    with pytest.raises(ArenaStoreError, match="lab_arena_code_review_input_invalid"):
        recovery_store.finish_submission_review(
            submission, miner, token, "passed",
            {**document("passed", model="unbound/model"), "provider_diagnostic": DIAGNOSTIC}, 1234,
        )
    assert recovery_store.get_submission(submission)["code_review_status"] == "reviewing"
    assert len(recovery_store.list_ledger(submission_id=submission)) == 2
    doc = {**document(), "error_code": "code_review_provider_request_rejected", "retryable": False,
           "provider_diagnostic": {**DIAGNOSTIC, "native_finish_reason": "refusal"}}
    assert recovery_store.finish_submission_review(submission, miner, token, "error", doc, 1234)["status"] == "error"
    assert recovery_store.begin_submission_review(submission, miner, new_lease_token(), 50_000, MODEL, 3, 1200)["status"] == "exhausted"


def test_existing_anthropic_diagnostics_still_persist(recovery_store):
    model = "anthropic/claude-sonnet-5.5"
    submission, miner, token = claim(recovery_store, "a5921", model=model)
    diagnostic = {**DIAGNOSTIC, "response_model": model,
                  "selected_model": model + "-20260928", "selected_provider": "Google Vertex"}
    assert recovery_store.finish_submission_review(
        submission, miner, token, "error", {**document(model=model), "provider_diagnostic": diagnostic}, 1234,
    )["status"] == "error"
    assert recovery_store.list_ledger(submission_id=submission)[-1]["entry_doc"]["provider_diagnostic"] == diagnostic


def test_upgrade_regression_rollback_and_security():
    server = database_with_lab_arena_migration(PREREQUISITES)
    transport = None
    try:
        psycopg2, dsn = next(server)
        transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
        store = ArenaStore(transport)
        submission, miner, token = claim(store, "a5930")
        doc = {**document(), "provider_diagnostic": DIAGNOSTIC}
        assert store.finish_submission_review(submission, miner, token, "error", doc, 1234)["status"] == "error"
        old_ledger = store.list_ledger(submission_id=submission)
        assert "provider_diagnostic" not in old_ledger[-1]["entry_doc"]
        with psycopg2.connect(**dsn) as connection:
            connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.execute("CREATE ROLE review599_migrator NOLOGIN NOSUPERUSER INHERIT")
                cursor.execute("GRANT lab_arena_owner TO review599_migrator")
                cursor.execute("ALTER SCHEMA public OWNER TO review599_migrator")
                cursor.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC,lab_arena_owner")
                cursor.execute("SET ROLE review599_migrator")
                def state():
                    cursor.execute("SELECT pg_get_functiondef(oid),to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=to_regprocedure(%s)", (SIGNATURE,))
                    return cursor.fetchone()
                before = state()
                failed = MIGRATION.read_text().replace("  EXECUTE v_updated;", "  EXECUTE v_updated;\n  PERFORM 1/0;", 1)
                with pytest.raises(psycopg2.errors.DivisionByZero):
                    cursor.execute(failed)
                cursor.execute("ROLLBACK")
                assert state() == before
                cursor.execute(MIGRATION.read_text())
                after = state()
                assert after[0] != before[0] and after[1] == before[1]
                cursor.execute(MIGRATION.read_text())
                assert state() == after
                cursor.execute("SELECT has_schema_privilege('lab_arena_owner','public','CREATE')")
                assert cursor.fetchone() == (False,)
                for role in ("anon", "authenticated", "service_role"):
                    cursor.execute("SELECT has_function_privilege(%s,%s,'EXECUTE'),has_table_privilege(%s,'public.lab_arena_ledger','SELECT')", (role, SIGNATURE, role))
                    assert cursor.fetchone() == (False, False)
        assert store.list_ledger(submission_id=submission) == old_ledger
        submission, miner, token = claim(store, "a5931")
        assert store.finish_submission_review(submission, miner, token, "error", doc, 1234)["status"] == "error"
        assert store.list_ledger(submission_id=submission)[-1]["entry_doc"]["provider_diagnostic"] == DIAGNOSTIC
    finally:
        if transport is not None:
            transport.close()
        server.close()
