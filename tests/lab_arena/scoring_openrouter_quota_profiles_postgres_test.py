"""Frozen scoring-quota profiles against disposable current PostgreSQL."""

from __future__ import annotations

import json

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey, sha


LEASE_TOKEN_HASH = "sha256:" + "a" * 64


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


@pytest.fixture(scope="module")
def connect(database):
    psycopg2, dsn = database

    def _connect():
        return psycopg2.connect(**dsn)

    return _connect


@pytest.fixture()
def store(connect):
    transport = PsycopgTransport(connect)
    yield ArenaStore(transport, lease_ttl_seconds=3_600)
    transport.close()


def _seed_score_run(connect, suffix, scoring_quotas, *, seeded_calls=0):
    round_id = "arena-2026-09-22-" + suffix
    submission_id = "scoring-quota-" + suffix + "-submission"
    run_id = "scoring-quota-" + suffix + "-run"
    assignment_id = "scoring-quota-" + suffix + "-assignment"
    miner = hotkey("scoring-quota-" + suffix)
    configuration = {
        "call_quotas": dict(contracts.CALL_QUOTAS_PER_ICP),
        "scoring_call_quotas": dict(scoring_quotas),
        "stage_1_icp_count": 10,
        "stage_2_icp_count": 10,
        "max_attempts_per_assignment": 2,
        "execution_cap_microusd": 4_000_000,
        "scoring_cap_microusd": 50_000_000,
        "baseline_hotkey": hotkey("scoring-quota-baseline"),
    }
    connection = connect()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds("
                "round_id,status,stage_generation,configuration_doc) "
                "VALUES (%s,'stage1_scoring',1,%s::jsonb)",
                (round_id, json.dumps(configuration)),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions("
                "submission_id,round_id,miner_hotkey,status,is_king) "
                "VALUES (%s,%s,%s,'frozen',FALSE)",
                (submission_id, round_id, miner),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_runs("
                "run_id,assignment_id,round_id,submission_id,miner_hotkey,"
                "stage,icp_position,attempt,kind,status,runner_hotkey,"
                "lease_token_hash,lease_generation,stage_generation,"
                "lease_expires_at) VALUES ("
                "%s,%s,%s,%s,%s,1,0,1,'score','leased',%s,%s,1,1,"
                "clock_timestamp() + interval '1 hour')",
                (
                    run_id,
                    assignment_id,
                    round_id,
                    submission_id,
                    miner,
                    miner,
                    LEASE_TOKEN_HASH,
                ),
            )
            if seeded_calls:
                cursor.execute(
                    "INSERT INTO public.lab_arena_ledger("
                    "entry_kind,miner_hotkey,round_id,submission_id,run_id,"
                    "stage,call_identity,provider,operation_id,funding_source,"
                    "amount_microusd,entry_doc) "
                    "SELECT 'reservation',%s,%s,%s,%s,1,"
                    "'sha256:' || md5(%s || series::text) || "
                    "md5('second-' || %s || series::text),"
                    "'openrouter','openrouter.chat','miner_key',0,'{}'::jsonb "
                    "FROM generate_series(1,%s) AS series",
                    (
                        miner,
                        round_id,
                        submission_id,
                        run_id,
                        run_id,
                        run_id,
                        seeded_calls,
                    ),
                )
            cursor.execute("SET session_replication_role=origin")
    finally:
        connection.close()
    return {
        "assignment_id": assignment_id,
        "run_id": run_id,
        "submission_id": submission_id,
    }


def _reserve(store, seeded, sequence, amount_microusd=0):
    identity = contracts.provider_call_identity(
        attempt=1,
        assignment_id=seeded["assignment_id"],
        icp_position=0,
        action_sequence=sequence,
        operation_id="openrouter.chat",
        request_hash=sha("%s-%d" % (seeded["run_id"], sequence)),
    )
    return identity, store.reserve_call(
        run_id=seeded["run_id"],
        lease_token_hash=LEASE_TOKEN_HASH,
        call_identity=identity,
        operation_id="openrouter.chat",
        provider="openrouter",
        funding_source="miner_key",
        amount_microusd=amount_microusd,
        call_doc={},
    )


def test_frozen_legacy_and_current_scoring_profiles_enforce_exact_boundaries(
    store, connect
):
    legacy = _seed_score_run(
        connect,
        "legacy120",
        contracts.LEGACY_SCORING_CALL_QUOTAS_PER_WORK_ITEM,
        seeded_calls=120,
    )
    _identity, legacy_refusal = _reserve(store, legacy, 121)
    assert (legacy_refusal["status"], legacy_refusal["reason"]) == (
        "refused",
        "per_icp_quota",
    )
    assert store.run_quota_snapshot(legacy["run_id"], LEASE_TOKEN_HASH)[
        "providers"
    ]["openrouter"] == {
        "limit": 120,
        "used": 120,
        "remaining": 0,
        "inflight": 120,
    }

    current = _seed_score_run(
        connect,
        "current2000",
        contracts.SCORING_CALL_QUOTAS_PER_WORK_ITEM,
        seeded_calls=120,
    )
    identity, admitted = _reserve(store, current, 121)
    assert admitted["status"] == "reserved"
    assert admitted["amount_microusd"] == 0
    replay_identity, replay = _reserve(store, current, 121)
    assert replay_identity == identity
    assert (replay["status"], replay["idempotent"]) == ("reserved", True)
    connection = connect()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_ledger "
                "WHERE call_identity=%s",
                (identity,),
            )
            assert cursor.fetchone() == (1,)
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger("
                "entry_kind,miner_hotkey,round_id,submission_id,run_id,"
                "stage,call_identity,provider,operation_id,funding_source,"
                "amount_microusd,entry_doc) "
                "SELECT 'reservation',runs.miner_hotkey,runs.round_id,"
                "runs.submission_id,runs.run_id,1,"
                "'sha256:' || md5(runs.run_id || series::text) || "
                "md5('tail-' || runs.run_id || series::text),"
                "'openrouter','openrouter.chat','miner_key',0,'{}'::jsonb "
                "FROM public.lab_arena_runs AS runs "
                "CROSS JOIN generate_series(122,2000) AS series "
                "WHERE runs.run_id=%s",
                (current["run_id"],),
            )
            cursor.execute("SET session_replication_role=origin")
    finally:
        connection.close()
    _identity, current_refusal = _reserve(store, current, 2001)
    assert (current_refusal["status"], current_refusal["reason"]) == (
        "refused",
        "per_icp_quota",
    )
    assert store.run_quota_snapshot(current["run_id"], LEASE_TOKEN_HASH)[
        "providers"
    ]["openrouter"] == {
        "limit": 2000,
        "used": 2000,
        "remaining": 0,
        "inflight": 2000,
    }


def test_current_scoring_profile_keeps_fifty_dollar_cap(store, connect):
    current = _seed_score_run(
        connect,
        "costcap",
        contracts.SCORING_CALL_QUOTAS_PER_WORK_ITEM,
    )
    identity, admitted = _reserve(store, current, 1, amount_microusd=50_000_000)
    assert (admitted["status"], admitted["amount_microusd"]) == (
        "reserved",
        50_000_000,
    )
    assert store.mark_dispatched(
        run_id=current["run_id"],
        lease_token_hash=LEASE_TOKEN_HASH,
        call_identity=identity,
    )["status"] == "dispatched"
    assert store.settle_call(
        run_id=current["run_id"],
        lease_token_hash=LEASE_TOKEN_HASH,
        call_identity=identity,
        actual_microusd=50_000_000,
        terminal_response={"status": 200},
    )["status"] == "settled"
    _identity, refused = _reserve(store, current, 2, amount_microusd=1)
    assert (refused["status"], refused["reason"]) == ("refused", "money_cap")
