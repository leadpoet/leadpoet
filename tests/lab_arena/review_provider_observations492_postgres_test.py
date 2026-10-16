"""Verbatim migration and private review/ledger observation contract checks."""

import json

import pytest

from lab_arena import code_review
from lab_arena.store import ArenaStoreError, new_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)
from tests.lab_arena.openrouter_in_flight_budget_postgres_test import (
    test_typed_budget_review_claim_backoff_retry_and_pass as _typed_budget_control,
)
from tests.lab_arena.test_lab_arena_code_review_postgres import (
    accepted_recovery_submission, recovery_connect, recovery_store, recovery_superuser,
)
from tests.postgres_migration_harness import SCRIPTS


MIGRATION = SCRIPTS / "492-lab-arena-review-provider-observations.sql"
SIGNATURE = "public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)"


@pytest.fixture(scope="module")
def recovery_database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS + (MIGRATION.name,))


def _claimed(store, suffix):
    _round, submission, miner = accepted_recovery_submission(store, suffix)
    token = new_lease_token()
    assert store.begin_submission_review(
        submission, miner, token, 50_000, code_review.DEFAULT_REVIEW_MODEL, 3, 1200,
    )["status"] == "claimed"
    document = {
        "model": code_review.DEFAULT_REVIEW_MODEL, "file_count": 3, "source_bytes": 1200,
        "error_code": "code_review_provider_credit", "provider_http_status": 402,
        "retryable": False,
    }
    return submission, miner, token, document


@pytest.mark.parametrize("source", [
    "openrouter_credits", "openrouter_key_limit", "openrouter_in_flight_budget", "unknown",
])
def test_bounded_observations_persist_in_review_and_uncertain_ledger(recovery_store, source):
    sources = ("openrouter_credits", "openrouter_key_limit", "openrouter_in_flight_budget", "unknown")
    submission, miner, token, document = _claimed(recovery_store, "a%02x" % (sources.index(source) + 1))
    observation = {
        "provider_raw_http_status": 402, "provider_effective_http_status": 402,
        "provider_limit_source": source,
    }
    assert recovery_store.finish_submission_review(
        submission, miner, token, "error",
        {**document, **observation, "message": "private provider prose", "api_key": "private-test-key"},
        None,
    )["status"] == "error"
    row = recovery_store.get_submission(submission)
    assert row["code_review_doc"]["cost_status"] == "uncertain"
    assert row["code_review_doc"]["cost_microusd"] == 50_000
    assert row["code_review_doc"]["retryable"] is False
    terminal = recovery_store.list_ledger(submission_id=submission)[-1]
    assert terminal["entry_kind"] == "uncertain"
    for stored in (row["code_review_doc"], terminal["entry_doc"]):
        for name, value in observation.items():
            assert stored[name] == value
        assert "private provider prose" not in json.dumps(stored)
        assert "private-test-key" not in json.dumps(stored)


def test_old_diagnostic_fields_keep_their_existing_contract(recovery_store):
    submission, miner, token, document = _claimed(recovery_store, "b05")
    document["error_reason"] = "envelope"
    assert recovery_store.finish_submission_review(
        submission, miner, token, "error", document, 0,
    )["status"] == "error"
    row = recovery_store.get_submission(submission)
    assert row["code_review_doc"]["cost_status"] == "settled"
    assert row["code_review_doc"]["cost_microusd"] == 0
    assert row["code_review_doc"]["error_reason"] == "envelope"
    assert not any(name in row["code_review_doc"] for name in (
        "provider_raw_http_status", "provider_effective_http_status", "provider_limit_source",
    ))


@pytest.mark.parametrize("index,pair", list(enumerate([
    ("provider_limit_source", "private provider prose"), ("provider_limit_source", []),
    ("provider_limit_source", None), ("provider_raw_http_status", "402"),
    ("provider_raw_http_status", 99), ("provider_raw_http_status", 600),
    ("provider_raw_http_status", 402.5), ("provider_raw_http_status", True),
    ("provider_effective_http_status", None), ("provider_effective_http_status", {}),
], start=16)))
def test_invalid_observations_cannot_cross_database_edge(recovery_store, index, pair):
    field, value = pair
    suffix = "c%02x" % index
    submission, miner, token, document = _claimed(recovery_store, suffix)
    with pytest.raises(ArenaStoreError, match="lab_arena_code_review_input_invalid"):
        recovery_store.finish_submission_review(
            submission, miner, token, "error", {**document, field: value}, None,
        )
    assert recovery_store.get_submission(submission)["code_review_status"] == "reviewing"
    assert [row["entry_kind"] for row in recovery_store.list_ledger(submission_id=submission)] == [
        "reservation", "dispatch",
    ]


def test_migration_replay_keeps_function_identity_and_security(recovery_superuser):
    with recovery_superuser.cursor() as cursor:
        cursor.execute("SELECT pg_catalog.to_jsonb(p) FROM pg_catalog.pg_proc p WHERE p.oid=%s::regprocedure", (SIGNATURE,))
        before = cursor.fetchone()[0]
        cursor.execute(MIGRATION.read_text())
        cursor.execute(MIGRATION.read_text())
        cursor.execute("SELECT pg_catalog.to_jsonb(p) FROM pg_catalog.pg_proc p WHERE p.oid=%s::regprocedure", (SIGNATURE,))
        assert cursor.fetchone()[0] == before
        cursor.execute("SELECT has_function_privilege('lab_arena_service',%s,'EXECUTE'),has_function_privilege('service_role',%s,'EXECUTE'),has_function_privilege('anon',%s,'EXECUTE'),has_function_privilege('authenticated',%s,'EXECUTE')", (SIGNATURE,) * 4)
        assert cursor.fetchone() == (True, False, False, False)


def test_migration_refuses_an_unreviewed_function_definition(recovery_superuser):
    with recovery_superuser.cursor() as cursor:
        cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
        original = cursor.fetchone()[0]
        changed = original.replace("BEGIN\n", "BEGIN\n  -- unrelated fixture change\n", 1)
        assert changed != original
        try:
            cursor.execute(changed)
            with pytest.raises(Exception, match="Arena review492 preimage differs") as caught:
                cursor.execute(MIGRATION.read_text())
            assert caught.value.pgcode == "55000"
            cursor.execute("ROLLBACK")
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
            assert cursor.fetchone()[0] == changed
        finally:
            cursor.execute("ROLLBACK")
            cursor.execute(original)


def test_typed_budget_retry_and_charge_controls_remain_unchanged(recovery_store, recovery_superuser):
    _typed_budget_control(recovery_store, recovery_superuser)
