"""Participation evidence through the real service-role RPCs and PostgreSQL."""

from pathlib import Path

import pytest

from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    claim,
    complete,
    expire_now,
    hotkey,
    open_round,
)

MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "scripts/20260911173147_lab_arena_validator_participation.sql"
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(POSTGREST_MIGRATIONS)


@pytest.fixture
def store(database):
    psycopg2, dsn = database
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    yield ArenaStore(transport)
    transport.close()


@pytest.fixture
def admin(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    yield connection
    connection.close()


def test_acceptance_scope_replay_expiry_and_immutable_evidence(store, admin):
    round_id = "arena-2026-09-11-participation"
    runners, _ = open_round(store, round_id, prefix="part", participants=1)
    runner = runners[0]
    assert not store.has_recent_participation("finney", 71, runner)
    leased, token, _, _ = claim(store, round_id, runner)
    run_id, lease_hash = leased["run_id"], hash_lease_token(token)
    assert not store.has_recent_participation("finney", 71, runner)
    assert (
        complete(
            store, run_id, hash_lease_token("wrong"), "accepted", output_ref="output"
        )["status"]
        == "stale"
    )
    assert not store.has_recent_participation("finney", 71, runner)
    assert (
        complete(store, run_id, lease_hash, "accepted", output_ref="output")["status"]
        == "accepted"
    )
    timestamp = store.get_run(run_id)["participation_accepted_at"]
    assert timestamp is not None
    assert store.has_recent_participation("finney", 71, runner)
    assert not store.has_recent_participation("test", 71, runner)
    assert not store.has_recent_participation("finney", 72, runner)
    assert not store.has_recent_participation("finney", 71, hotkey("other"))
    assert (
        complete(store, run_id, lease_hash, "accepted", output_ref="output")[
            "idempotent"
        ]
        is True
    )
    assert store.get_run(run_id)["participation_accepted_at"] == timestamp

    # Cache insertion also stays uncredited, even with a runner copied in.
    with admin.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO public.lab_arena_runs (
                run_id, assignment_id, round_id, submission_id, miner_hotkey,
                stage, icp_position, attempt, kind, status, terminal_cause,
                result_doc, output_ref, runner_hotkey, lease_generation, stage_generation
            ) SELECT run_id || '-cached', assignment_id || '-cached', round_id,
                submission_id, miner_hotkey, stage, icp_position, attempt, kind,
                'accepted', 'accepted',
                '{"schema_version":"leadpoet.lab_arena.cached_run_result.v1","terminal_status":"accepted"}'::jsonb,
                output_ref, %s, lease_generation, stage_generation
            FROM public.lab_arena_runs WHERE run_id = %s
        """,
            (hotkey("cache-only"), run_id),
        )
    assert store.get_run(run_id + "-cached")["participation_accepted_at"] is None
    assert not store.has_recent_participation("finney", 71, hotkey("cache-only"))

    # A later bookkeeping update and migration reapplication cannot refresh work.
    with admin.cursor() as cursor:
        cursor.execute(
            "UPDATE public.lab_arena_runs SET updated_at = clock_timestamp() WHERE run_id = %s",
            (run_id,),
        )
        cursor.execute(MIGRATION.read_text())
    assert store.get_run(run_id)["participation_accepted_at"] == timestamp
    for update in (
        "participation_accepted_at = clock_timestamp()",
        "participation_accepted_at = NULL",
        "runner_hotkey = 'different-validator'",
    ):
        with admin.cursor() as cursor, pytest.raises(
            Exception, match="participation .*immutable"
        ):
            cursor.execute(
                "UPDATE public.lab_arena_runs SET " + update + " WHERE run_id = %s",
                (run_id,),
            )

    # Time travel belongs only to this disposable test database.
    with admin.cursor() as cursor:
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_participation"
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET participation_accepted_at = statement_timestamp() - interval '24 hours' WHERE run_id = %s",
            (run_id,),
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_runs_participation"
        )
    expired = store.get_run(run_id)["participation_accepted_at"]
    assert not store.has_recent_participation("finney", 71, runner)
    complete(store, run_id, lease_hash, "accepted", output_ref="output")
    assert store.get_run(run_id)["participation_accepted_at"] == expired
    assert not store.has_recent_participation("finney", 71, runner)

    # Existing historical acceptance has no trustworthy timestamp to backfill.
    with admin.cursor() as cursor:
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_participation"
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET participation_accepted_at = NULL WHERE run_id = %s",
            (run_id,),
        )
        cursor.execute(
            "ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_runs_participation"
        )
        cursor.execute(MIGRATION.read_text())
    assert store.get_run(run_id)["participation_accepted_at"] is None
    assert not store.has_recent_participation("finney", 71, runner)


def test_failure_and_expired_lease_do_not_earn_participation(store, admin):
    round_id = "arena-2026-09-11-partfailed"
    runners, _ = open_round(store, round_id, prefix="part-failed", participants=1)
    runner = runners[0]
    leased, token, _, _ = claim(store, round_id, runner)
    assert (
        complete(store, leased["run_id"], hash_lease_token(token), "model_error")[
            "status"
        ]
        == "failed"
    )
    assert store.get_run(leased["run_id"])["participation_accepted_at"] is None
    assert not store.has_recent_participation("finney", 71, runner)
    leased, token, _, _ = claim(store, round_id, runner)
    expire_now(admin, leased["run_id"])
    assert (
        complete(
            store,
            leased["run_id"],
            hash_lease_token(token),
            "accepted",
            output_ref="output",
        )["status"]
        == "stale"
    )
    assert store.get_run(leased["run_id"])["participation_accepted_at"] is None
    assert not store.has_recent_participation("finney", 71, runner)


def test_lookup_and_evidence_are_not_public_or_directly_writable(admin):
    with admin.cursor() as cursor:
        for role in ("anon", "authenticated", "service_role"):
            cursor.execute(
                "SELECT has_function_privilege(%s, 'public.lab_arena_has_recent_participation_v1(text,integer,text)', 'EXECUTE')",
                (role,),
            )
            assert cursor.fetchone() == (False,)
        cursor.execute(
            "SELECT has_function_privilege('lab_arena_service', 'public.lab_arena_has_recent_participation_v1(text,integer,text)', 'EXECUTE')"
        )
        assert cursor.fetchone() == (True,)
        cursor.execute(
            "SELECT has_table_privilege('lab_arena_service', 'public.lab_arena_runs', 'UPDATE'), has_table_privilege('lab_arena_service', 'public.lab_arena_runs', 'INSERT'), relrowsecurity FROM pg_class WHERE oid = 'public.lab_arena_runs'::regclass"
        )
        assert cursor.fetchone() == (False, False, True)
        cursor.execute(
            "SELECT to_regclass('public.lab_arena_runs_recent_participation_idx')"
        )
        assert cursor.fetchone()[0] is not None
