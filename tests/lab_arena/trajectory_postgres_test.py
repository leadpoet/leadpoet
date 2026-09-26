"""Disposable PostgreSQL proof for the private Arena trajectory stream."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_arena import trajectory
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport, hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)


MIGRATION = "365-lab-arena-trajectories.sql"
ROUND = "arena-2099-01-01-trajectory"
SUBMISSION = "trajectory-submission"
RUN = "trajectory-run"
MINER = "5" * 48
RUNNER = "6" * 48
LEASE = "trajectory-lease-token"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS + (MIGRATION,)
    )


@pytest.fixture(scope="module")
def seeded(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds("
            "round_id,status,status_generation,stage_generation,configuration_doc) "
            "VALUES (%s,'stage1',1,1,%s::jsonb)",
            (
                ROUND,
                json.dumps(
                    {
                        "schema_version": "leadpoet.lab_arena.round_configuration.v1",
                        "round_id": ROUND,
                        "mode": "live",
                        "schedule": {"submission_cutoff": "2099-01-01T00:00:00Z"},
                    }
                ),
            ),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions("
            "submission_id,round_id,miner_hotkey,status,is_king) "
            "VALUES (%s,%s,%s,'frozen',TRUE)",
            (SUBMISSION, ROUND, MINER),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_runs("
            "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status,runner_hotkey,lease_token_hash,"
            "lease_generation,stage_generation,lease_expires_at) VALUES "
            "(%s,%s,%s,%s,%s,1,3,1,'execute','leased',%s,%s,1,1,"
            "clock_timestamp() + interval '1 hour')",
            (
                RUN,
                "trajectory-assignment",
                ROUND,
                SUBMISSION,
                MINER,
                RUNNER,
                hash_lease_token(LEASE),
            ),
        )
    yield database


def _connect_store(psycopg2, dsn):
    return ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))


def test_migration_replays_and_append_is_idempotent_with_derived_identity(seeded):
    psycopg2, dsn = seeded
    migration = (
        Path(__file__).resolve().parents[2] / "scripts" / MIGRATION
    ).read_text(encoding="utf-8")
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(migration)
        cursor.execute(migration)

    store = _connect_store(psycopg2, dsn)
    document = trajectory.event(
        "runtime.started",
        {"status": "started", "run_id": "spoofed"},
    )
    first = store.append_trajectory_events(
        RUN, hash_lease_token(LEASE), [document]
    )
    replay = store.append_trajectory_events(
        RUN, hash_lease_token(LEASE), [document]
    )
    store.close()
    assert (first["inserted"], replay["existing"]) == (1, 1)

    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT round_id,submission_id,miner_hotkey,runner_hotkey,"
            "icp_identifier,icp_position,attempt,run_kind,model_role,content "
            "FROM public.lab_arena_trajectory_events WHERE run_id=%s",
            (RUN,),
        )
        row = cursor.fetchone()
    assert row[:9] == (
        ROUND, SUBMISSION, MINER, RUNNER, ROUND + ":icp:3", 3, 1,
        "execute", "baseline",
    )
    assert row[9]["run_id"] == "[DERIVED_BY_GATEWAY]"


def test_stale_lease_and_unprivileged_roles_cannot_append_or_read(seeded):
    psycopg2, dsn = seeded
    store = _connect_store(psycopg2, dsn)
    stale = store.append_trajectory_events(
        RUN,
        hash_lease_token("wrong"),
        [trajectory.event("runtime.finished", {"status": "done"})],
    )
    store.close()
    assert stale == {"status": "stale"}

    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='accepted', "
            "lease_expires_at=clock_timestamp() - interval '1 minute' "
            "WHERE run_id=%s",
            (RUN,),
        )
    store = _connect_store(psycopg2, dsn)
    late_provider = store.append_trajectory_events(
        RUN,
        hash_lease_token(LEASE),
        [trajectory.event("provider.response", {"http_status": 200})],
    )
    late_runtime = store.append_trajectory_events(
        RUN,
        hash_lease_token(LEASE),
        [trajectory.event("runtime.finished", {"status": "done"})],
    )
    late_provider_request = store.append_trajectory_events(
        RUN,
        hash_lease_token(LEASE),
        [trajectory.event("provider.request", {"operation_id": "openrouter.chat"})],
    )
    store.close()
    assert late_provider["status"] == "accepted"
    assert late_runtime == {"status": "stale"}
    assert late_provider_request == {"status": "stale"}

    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            for role in ("anon", "authenticated", "service_role"):
                cursor.execute("SET ROLE %s" % role)
                with pytest.raises(psycopg2.Error):
                    cursor.execute(
                        "SELECT * FROM public.lab_arena_trajectory_events"
                    )
                cursor.execute("RESET ROLE")
            cursor.execute("SET ROLE authenticated")
            with pytest.raises(psycopg2.Error):
                cursor.execute(
                    "SELECT public.lab_arena_append_trajectory_events_v1("
                    "%s,%s,%s::jsonb)",
                    (RUN, hash_lease_token(LEASE), json.dumps([])),
                )
    finally:
        connection.close()
