"""Verbatim migration 274 against the disposable Lab Arena PostgreSQL."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey


MIGRATION = "274-lab-arena-run-quota-snapshot.sql"
LEASE_TOKEN_HASH = "sha256:" + "a" * 64
EXECUTE_ROUND = "arena-2026-09-16-qexec"
SCORE_ROUND = "arena-2026-09-16-qscore"
LEGACY_EXECUTE_ROUND = "arena-2026-09-16-qlegacy"
EXECUTE_RUN = "quota-execute-run"
SCORE_RUN = "quota-score-run"
LEGACY_EXECUTE_RUN = "quota-legacy-execute-run"
SUBMISSION = "quota-submission"
MINER = hotkey("quota-snapshot-miner")


@pytest.fixture(scope="module")
def database():
    # Repeating the exact migration is part of the idempotency proof.
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS + (MIGRATION,)
    )


@pytest.fixture(scope="module")
def connect(database):
    psycopg2, dsn = database

    def _connect():
        return psycopg2.connect(**dsn)

    return _connect


@pytest.fixture(scope="module", autouse=True)
def seeded(connect):
    connection = connect()
    connection.autocommit = True
    configuration = {
        "call_quotas": dict(contracts.CALL_QUOTAS_PER_ICP),
        "scoring_call_quotas": dict(
            contracts.SCORING_CALL_QUOTAS_PER_WORK_ITEM
        ),
    }
    legacy_configuration = {
        **configuration,
        "call_quotas": dict(contracts.LEGACY_CALL_QUOTAS_PER_ICP),
    }
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "TRUNCATE public.lab_arena_ledger,public.lab_arena_runs,"
                "public.lab_arena_submissions,public.lab_arena_rounds "
                "RESTART IDENTITY CASCADE"
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds("
                "round_id,status,stage_generation,configuration_doc) VALUES "
                "(%s,'stage1',1,%s::jsonb),"
                "(%s,'stage1_scoring',1,%s::jsonb),"
                "(%s,'stage1',1,%s::jsonb)",
                (
                    EXECUTE_ROUND,
                    json.dumps(configuration),
                    SCORE_ROUND,
                    json.dumps(configuration),
                    LEGACY_EXECUTE_ROUND,
                    json.dumps(legacy_configuration),
                ),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions("
                "submission_id,round_id,miner_hotkey,status) "
                "VALUES (%s,%s,%s,'frozen')",
                (SUBMISSION, EXECUTE_ROUND, MINER),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions("
                "submission_id,round_id,miner_hotkey,status) "
                "VALUES (%s,%s,%s,'frozen')",
                (SUBMISSION + "-score", SCORE_ROUND, MINER),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions("
                "submission_id,round_id,miner_hotkey,status) "
                "VALUES (%s,%s,%s,'frozen')",
                (SUBMISSION + "-legacy", LEGACY_EXECUTE_ROUND, MINER),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_runs("
                "run_id,assignment_id,round_id,submission_id,miner_hotkey,"
                "stage,icp_position,attempt,kind,status,runner_hotkey,"
                "lease_token_hash,lease_generation,stage_generation,"
                "lease_expires_at) VALUES "
                "(%s,%s,%s,%s,%s,1,0,1,'execute','leased',%s,%s,1,1,"
                "clock_timestamp() + interval '1 hour'),"
                "(%s,%s,%s,%s,%s,1,0,1,'score','leased',%s,%s,1,1,"
                "clock_timestamp() + interval '1 hour'),"
                "(%s,%s,%s,%s,%s,1,0,1,'execute','leased',%s,%s,1,1,"
                "clock_timestamp() + interval '1 hour')",
                (
                    EXECUTE_RUN,
                    "quota-execute-assignment",
                    EXECUTE_ROUND,
                    SUBMISSION,
                    MINER,
                    MINER,
                    LEASE_TOKEN_HASH,
                    SCORE_RUN,
                    "quota-score-assignment",
                    SCORE_ROUND,
                    SUBMISSION + "-score",
                    MINER,
                    MINER,
                    LEASE_TOKEN_HASH,
                    LEGACY_EXECUTE_RUN,
                    "quota-legacy-execute-assignment",
                    LEGACY_EXECUTE_ROUND,
                    SUBMISSION + "-legacy",
                    MINER,
                    MINER,
                    LEASE_TOKEN_HASH,
                ),
            )

            def add(kind, identity, *, response=False):
                cursor.execute(
                    "INSERT INTO public.lab_arena_ledger("
                    "entry_kind,miner_hotkey,round_id,submission_id,run_id,"
                    "stage,call_identity,provider,operation_id,funding_source,"
                    "amount_microusd,entry_doc,terminal_response) VALUES "
                    "(%s,%s,%s,%s,%s,1,%s,'openrouter','openrouter.responses',"
                    "'host',100,'{}'::jsonb,%s::jsonb)",
                    (
                        kind,
                        MINER,
                        EXECUTE_ROUND,
                        SUBMISSION,
                        EXECUTE_RUN,
                        identity,
                        json.dumps({}) if response else None,
                    ),
                )

            def identity(label):
                return contracts.document_hash(["quota", label])

            add("reservation", identity("reservation"))
            add("reservation", identity("dispatch"))
            add("dispatch", identity("dispatch"))
            add("reservation", identity("settlement"))
            add("dispatch", identity("settlement"))
            add("settlement", identity("settlement"), response=True)
            add("reservation", identity("uncertain"))
            add("dispatch", identity("uncertain"))
            add("uncertain", identity("uncertain"), response=True)
            add("refusal", identity("refusal"))
            add("reservation", identity("recovery"))
            add("recovery", identity("recovery"))
            # One broker request can produce four separately admitted
            # credential-retry identities. Every identity remains visible.
            for retry in range(4):
                retry_identity = identity("credential-retry-%d" % retry)
                add("reservation", retry_identity)
                add("dispatch", retry_identity)
                add("settlement", retry_identity, response=True)
            cursor.execute("SET session_replication_role=origin")
        yield
    finally:
        connection.close()


@pytest.fixture()
def store(connect):
    transport = PsycopgTransport(connect)
    yield ArenaStore(transport)
    transport.close()


def durable_state(connection, run_id):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_catalog.to_jsonb(runs) FROM public.lab_arena_runs AS runs "
            "WHERE run_id=%s",
            (run_id,),
        )
        run = cursor.fetchone()[0]
        cursor.execute(
            "SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(ledger) "
            "ORDER BY entry_id),'[]'::jsonb) FROM public.lab_arena_ledger AS ledger "
            "WHERE run_id=%s",
            (run_id,),
        )
        ledger = cursor.fetchone()[0]
    return run, ledger


def test_snapshot_counts_latest_identity_heads_without_mutation(store, connect):
    connection = connect()
    connection.autocommit = True
    try:
        before = durable_state(connection, EXECUTE_RUN)
        result = store.run_quota_snapshot(EXECUTE_RUN, LEASE_TOKEN_HASH)
        after = durable_state(connection, EXECUTE_RUN)
    finally:
        connection.close()

    assert before == after
    assert result == {
        "schema_version": "leadpoet.lab_arena.quota_snapshot.v1",
        "providers": {
            "scrapingdog": {
                "limit": 30,
                "used": 0,
                "remaining": 30,
                "inflight": 0,
            },
            "deepline": {
                "limit": 30,
                "used": 0,
                "remaining": 30,
                "inflight": 0,
            },
            "openrouter": {
                "limit": 200,
                "used": 8,
                "remaining": 192,
                "inflight": 2,
            },
        },
    }
    encoded = json.dumps(result).lower()
    for forbidden in (
        EXECUTE_RUN,
        LEASE_TOKEN_HASH,
        "call_identity",
        "credential",
        "account",
        "microusd",
    ):
        assert forbidden.lower() not in encoded


def test_score_run_uses_frozen_judge_limits(store):
    result = store.run_quota_snapshot(SCORE_RUN, LEASE_TOKEN_HASH)
    assert result["providers"]["openrouter"] == {
        "limit": 120,
        "used": 0,
        "remaining": 120,
        "inflight": 0,
    }
    assert result["providers"]["scrapingdog"]["limit"] == 150
    assert result["providers"]["deepline"]["limit"] == 40


def test_historical_execute_run_uses_its_frozen_60_call_limit(store):
    result = store.run_quota_snapshot(LEGACY_EXECUTE_RUN, LEASE_TOKEN_HASH)
    assert result["providers"]["openrouter"] == {
        "limit": 60,
        "used": 0,
        "remaining": 60,
        "inflight": 0,
    }


def test_wrong_and_stale_lease_fail_without_additional_mutation(store, connect):
    connection = connect()
    connection.autocommit = True
    try:
        before_wrong = durable_state(connection, EXECUTE_RUN)
        with pytest.raises(ArenaStoreError):
            store.run_quota_snapshot(
                EXECUTE_RUN, "sha256:" + "b" * 64
            )
        assert durable_state(connection, EXECUTE_RUN) == before_wrong

        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_runs SET lease_expires_at=%s "
                "WHERE run_id=%s",
                (datetime(2000, 1, 1, tzinfo=timezone.utc), EXECUTE_RUN),
            )
        before_stale = durable_state(connection, EXECUTE_RUN)
        with pytest.raises(ArenaStoreError):
            store.run_quota_snapshot(EXECUTE_RUN, LEASE_TOKEN_HASH)
        assert durable_state(connection, EXECUTE_RUN) == before_stale
    finally:
        connection.close()


def test_quota_rpc_is_service_only(connect):
    connection = connect()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT "
                "pg_catalog.has_function_privilege('lab_arena_service',"
                "'public.lab_arena_run_quota_snapshot_v1(text,text)','EXECUTE'),"
                "pg_catalog.has_function_privilege('anon',"
                "'public.lab_arena_run_quota_snapshot_v1(text,text)','EXECUTE'),"
                "pg_catalog.has_function_privilege('authenticated',"
                "'public.lab_arena_run_quota_snapshot_v1(text,text)','EXECUTE'),"
                "pg_catalog.has_function_privilege('service_role',"
                "'public.lab_arena_run_quota_snapshot_v1(text,text)','EXECUTE')"
            )
            assert cursor.fetchone() == (True, False, False, False)
    finally:
        connection.close()
