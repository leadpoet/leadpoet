"""Focused PostgreSQL contract for successful-call sourcing costs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    commit_round,
    frozen_participants,
    hotkey,
    open_round,
    round_config,
    sha,
)


MIGRATION = "229-lab-arena-successful-call-cost-eligibility.sql"
PERMISSIONS_MIGRATION = "230-lab-arena-successful-call-cost-permissions.sql"


@pytest.fixture
def database():
    yield from database_with_lab_arena_migration(POSTGREST_MIGRATIONS)


def _seed(database):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    store = ArenaStore(PsycopgTransport(connect))
    round_id = "arena-2098-06-01-successcost"
    config = round_config(
        round_id,
        [hotkey("successful-cost-runner")],
        cost_per_company_microusd=500,
        execution_cap_microusd=500,
    )
    config["sourcing_cost_eligibility_policy"] = (
        contracts.SUCCESSFUL_CALLS_COST_POLICY
    )
    assert store.create_round(round_id, config)["status"] == "created"
    participants = frozen_participants(
        store, round_id, 4, prefix="successful-cost"
    )
    commit_round(store, round_id, participants)
    assert store.open_stage(
        round_id, 1, participants, list(contracts.stage_positions(1))
    )["status"] == "ok"

    connection = connect()
    connection.autocommit = True
    with connection.cursor() as cursor:
        for participant in participants:
            cursor.execute(
                "SELECT run_id FROM public.lab_arena_runs "
                "WHERE submission_id=%s AND kind='execute' "
                "ORDER BY run_id LIMIT 1",
                (participant["submission_id"],),
            )
            participant["run_id"] = cursor.fetchone()[0]
    return connection, round_id, participants


def _head(
    cursor,
    *,
    round_id,
    participant,
    label,
    kind="execute",
    entry_kind="settlement",
    amount=0,
    succeeded=None,
):
    run_id = participant["run_id"]
    if kind == "score":
        run_id = round_id + ":score:" + label
        cursor.execute(
            "INSERT INTO public.lab_arena_runs "
            "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,status,stage_generation,kind,scored_run_id) "
            "VALUES (%s,%s,%s,%s,%s,1,0,1,'pending',1,'score',%s)",
            (
                run_id,
                round_id + ":assignment:" + label,
                round_id,
                participant["submission_id"],
                participant["miner_hotkey"],
                participant["run_id"],
            ),
        )
    terminal = {} if succeeded is None else {"call_succeeded": succeeded}
    entry_doc = (
        {"reason": "worker_reported", "call": terminal}
        if entry_kind == "uncertain"
        else {}
    )
    cursor.execute(
        "INSERT INTO public.lab_arena_ledger "
        "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
        "call_identity,provider,operation_id,funding_source,amount_microusd,"
        "entry_doc,terminal_response) "
        "VALUES (%s,%s,%s,%s,%s,1,%s,'openrouter','openrouter.chat',"
        "'miner_key',%s,%s::jsonb,%s::jsonb)",
        (
            entry_kind,
            participant["miner_hotkey"],
            round_id,
            participant["submission_id"],
            run_id,
            sha("successful-cost-" + label),
            amount,
            json.dumps(entry_doc),
            json.dumps(terminal) if entry_kind == "settlement" else None,
        ),
    )


def test_successful_call_aggregate_and_eligibility_matrix(database):
    connection, round_id, participants = _seed(database)
    try:
        with connection.cursor() as cursor:
            _head(cursor, round_id=round_id, participant=participants[0], label="success", amount=200, succeeded=True)
            _head(cursor, round_id=round_id, participant=participants[0], label="charged-failure", amount=1000, succeeded=False)
            _head(cursor, round_id=round_id, participant=participants[1], label="success-missing-charge", entry_kind="uncertain", amount=400, succeeded=True)
            _head(cursor, round_id=round_id, participant=participants[2], label="failed-uncertain", entry_kind="uncertain", amount=500, succeeded=False)
            _head(cursor, round_id=round_id, participant=participants[2], label="judge-uncertain", kind="score", entry_kind="uncertain", amount=600)
            _head(cursor, round_id=round_id, participant=participants[3], label="missing-proof", amount=300)

            cursor.execute(
                "SELECT public.lab_arena_submission_costs(%s)",
                (participants[0]["submission_id"],),
            )
            costs = cursor.fetchone()[0]
            execution = next(
                row for row in costs["providers"] if row["kind"] == "execute"
            )
            assert execution["settled_microusd"] == 1200
            assert execution["successful_microusd"] == 200
            assert execution["successful_calls"] == 1
            assert execution["success_unresolved_calls"] == 0

            results = []
            for participant in participants:
                cursor.execute(
                    "SELECT public.lab_arena__successful_call_eligibility(%s,%s,1)",
                    (round_id, participant["submission_id"]),
                )
                results.append(cursor.fetchone()[0])
        assert [(row["eligible"], row["eligibility_reason"]) for row in results] == [
            (True, "eligible"),
            (False, "provider_cost_uncertain"),
            (True, "eligible"),
            (False, "provider_cost_uncertain"),
        ]
        assert results[0]["competition_sourcing_microusd"] == 200
        assert results[0]["execution"]["settled_microusd"] == 1200
        assert results[2]["judge"]["uncertain_calls"] == 1
    finally:
        connection.close()


def test_migration_replay_and_confirmation_helper_binding(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    migration = Path(__file__).resolve().parents[2] / "scripts" / MIGRATION
    try:
        with connection.cursor() as cursor:
            for _ in range(2):
                cursor.execute(migration.read_text(encoding="utf-8"))
                cursor.execute((migration.parent / PERMISSIONS_MIGRATION).read_text())
            cursor.execute(
                "SELECT public.lab_arena_successful_call_cost_schema_v1()"
            )
            assert cursor.fetchone()[0] == {
                "schema_version": (
                    "leadpoet.lab_arena.successful_call_cost_schema.v1"
                ),
                "version": 230,
                "policy": "successful_calls_v1",
            }
            cursor.execute(
                "SELECT pg_get_functiondef('public.lab_arena__integrity_eligibility(text,text,integer[])'::regprocedure), "
                "pg_get_functiondef('public.lab_arena_open_confirmation(text,jsonb)'::regprocedure)"
            )
            eligibility, confirmation = cursor.fetchone()
        assert "lab_arena__successful_call_eligibility" in eligibility
        assert "lab_arena__integrity_eligibility" in confirmation
    finally:
        connection.close()


def test_n_minus_one_old_round_cost_rpc_shape_is_unchanged():
    migrations = tuple(name for name in POSTGREST_MIGRATIONS if name not in (MIGRATION, PERMISSIONS_MIGRATION))
    database = database_with_lab_arena_migration(migrations)
    connection = None
    try:
        psycopg2, dsn = next(database)
        connect = lambda: psycopg2.connect(**dsn)
        store = ArenaStore(PsycopgTransport(connect))
        round_id = "arena-2098-06-02-oldcost"
        runners, participants = open_round(
            store,
            round_id,
            participants=1,
            runners=1,
            prefix="old-successful-cost",
            cost_per_company_microusd=500,
        )
        del runners
        connection = connect()
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT run_id FROM public.lab_arena_runs "
                "WHERE submission_id=%s AND kind='execute' ORDER BY run_id LIMIT 1",
                (participants[0]["submission_id"],),
            )
            participants[0]["run_id"] = cursor.fetchone()[0]
            _head(
                cursor,
                round_id=round_id,
                participant=participants[0],
                label="old-settlement",
                amount=123,
            )
            cursor.execute(
                "SELECT public.lab_arena_submission_costs(%s)",
                (participants[0]["submission_id"],),
            )
            before = cursor.fetchone()[0]
            migration = Path(__file__).resolve().parents[2] / "scripts" / MIGRATION
            cursor.execute(migration.read_text(encoding="utf-8"))
            cursor.execute((migration.parent / PERMISSIONS_MIGRATION).read_text())
            cursor.execute(
                "SELECT public.lab_arena_submission_costs(%s)",
                (participants[0]["submission_id"],),
            )
            after = cursor.fetchone()[0]
        assert after == before
        assert set(after["providers"][0]) == {
            "kind",
            "provider",
            "settled_microusd",
            "reserved_or_uncertain_microusd",
            "inflight_calls",
            "uncertain_calls",
            "refused_calls",
            "call_count",
        }
        contracts.validate_submission_costs(after)
    finally:
        if connection is not None:
            connection.close()
        database.close()


def test_supabase_default_grants_cannot_expose_private_cost_functions():
    migrations = tuple(
        name for name in POSTGREST_MIGRATIONS
        if name not in (MIGRATION, PERMISSIONS_MIGRATION)
    )
    database = database_with_lab_arena_migration(migrations)
    connection = None
    try:
        psycopg2, dsn = next(database)
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        scripts = Path(__file__).resolve().parents[2] / "scripts"
        with connection.cursor() as cursor:
            # Match production's named default grants, absent from a fresh PG.
            cursor.execute(
                "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT EXECUTE "
                "ON FUNCTIONS TO anon, authenticated, service_role"
            )
            cursor.execute((scripts / MIGRATION).read_text())
            cursor.execute(
                "SELECT has_function_privilege('anon', "
                "'public.lab_arena__successful_call_cost_state(text,text,text)', 'execute')"
            )
            assert cursor.fetchone()[0] is True
            for _ in range(2):
                cursor.execute((scripts / PERMISSIONS_MIGRATION).read_text())
            cursor.execute(
                "SELECT p.oid, p.proname, role_name, "
                "has_function_privilege(role_name,p.oid,'execute') "
                "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
                "CROSS JOIN unnest(ARRAY['anon','authenticated','service_role',"
                "'lab_arena_service','lab_arena_owner']) role_name "
                "WHERE n.nspname='public' AND (p.proname LIKE 'lab_arena__successful_call%' "
                "OR p.proname IN ('lab_arena_submission_costs','lab_arena_successful_call_cost_schema_v1'))"
            )
            rows = cursor.fetchall()
            assert len(rows) == 30
            for _, name, role, allowed in rows:
                expected = role == 'lab_arena_owner' or (
                    role == 'lab_arena_service' and name in (
                        'lab_arena_submission_costs',
                        'lab_arena_successful_call_cost_schema_v1',
                    )
                )
                assert allowed is expected, (name, role)
            cursor.execute("SET ROLE anon")
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                cursor.execute(
                    "SELECT public.lab_arena__successful_call_cost_state(NULL,NULL,NULL)"
                )
            cursor.execute("RESET ROLE")
            cursor.execute("SET ROLE lab_arena_service")
            cursor.execute("SELECT public.lab_arena_successful_call_cost_schema_v1()")
            assert cursor.fetchone()[0]['version'] == 230
            with pytest.raises(psycopg2.errors.NoDataFound, match='lab_arena_submission_missing'):
                cursor.execute("SELECT public.lab_arena_submission_costs('no-submission')")
            cursor.execute("RESET ROLE")
    finally:
        if connection is not None:
            connection.close()
        database.close()
