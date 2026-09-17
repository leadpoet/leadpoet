"""Exact cancelled Sep17 fixture proof for recovery278."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim

ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/278-arena-2026-09-17-baseline-recovery.sql"
SNAPSHOT = Path(
    "/private/tmp/leadpoet-tyche-rerun-evidence-20260915/private/"
    "sep17-cancelled-snapshot-20260917T035342Z.json"
)
SNAPSHOT_SHA = "307494c29e22f508358b17fd3a021e20c73fee21e624e89040e49bb9640f1ff2"
ROUND = "arena-2026-09-17"
BASELINE = "baseline-2026-09-17"
ARCHIVE = "arena-2026-09-17-archive"
SOURCE_SIZE = 562595
SOURCE_SHA = "8dc02f198bc4c5e5780236f079efb789aace28f5acbd54af2a512c45f6fb5ed1"
SOURCE_COMMIT = "db54c917925cc4a9ba953eb64c604a38c2c803b1"
BANK_SHA = "7023eca8a6518434d5010d31481c1d156aa96abb3e5565c30e033073b088c871"
NONBASELINE_LEDGER_HASH = (
    "sha256:8bd6df08e53b8fe17e846cc5ccb89535a40e42dcc89d87b57825d4f16e67cea2"
)
SCHEDULE = {
    "submission_open": "2026-09-16T00:00:00Z",
    "submission_cutoff": "2026-09-17T00:00:00Z",
    "benchmark_deadline": "2026-09-17T06:00:00Z",
    "stage_1_start": "2026-09-17T06:00:01Z",
    "stage_1_close": "2026-09-17T12:00:00Z",
    "stage_1_scoring_close": "2026-09-17T14:00:00Z",
    "stage_2_start": "2026-09-17T14:00:01Z",
    "stage_2_close": "2026-09-17T14:20:00Z",
    "final_scoring_close": "2026-09-17T16:20:00Z",
    "publication_deadline": "2026-09-17T16:50:00Z",
}


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _insert_rows(cursor, table: str, documents: list[dict]) -> None:
    cursor.execute(
        "SELECT attname FROM pg_catalog.pg_attribute "
        "WHERE attrelid=%s::regclass AND attnum>0 AND NOT attisdropped "
        "AND attgenerated='' ORDER BY attnum",
        ("public." + table,),
    )
    columns = [row[0] for row in cursor.fetchall()]
    names = ",".join(columns)
    selected = ",".join("row_value." + name for name in columns)
    cursor.execute(
        "INSERT INTO public." + table + " (" + names + ") SELECT "
        + selected + " FROM pg_catalog.jsonb_populate_recordset("
        "NULL::public." + table + ",%s::jsonb) AS row_value",
        (json.dumps(documents),),
    )


def _restore(connection) -> dict:
    assert SNAPSHOT.stat().st_mode & 0o777 == 0o600
    assert hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() == SNAPSHOT_SHA
    captured = json.loads(SNAPSHOT.read_text())
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "TRUNCATE public.lab_arena_ledger,public.lab_arena_runs,"
            "public.lab_arena_submissions,public.lab_arena_rounds "
            "RESTART IDENTITY CASCADE"
        )
        for table, key in (
            ("lab_arena_rounds", "round"),
            ("lab_arena_submissions", "submissions"),
            ("lab_arena_runs", "runs"),
            ("lab_arena_ledger", "ledger"),
        ):
            raw = captured[key]
            documents = [json.loads(value) for value in ([raw] if key == "round" else raw)]
            for start in range(0, len(documents), 500):
                _insert_rows(cursor, table, documents[start : start + 500])
        cursor.execute(
            "SELECT pg_catalog.setval('public.lab_arena_ledger_entry_id_seq',"
            "(SELECT max(entry_id) FROM public.lab_arena_ledger),true)"
        )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()
    return captured


def historical_migration_text():
    # This historical recovery admitted only before 06Z on September 17.
    # Freeze only the disposable SQL clock so its guards remain testable later.
    return MIGRATION.read_text().replace(
        'pg_catalog.clock_timestamp()', "'2026-09-17T04:00:00Z'::TIMESTAMPTZ",
    )


def _prepare(cursor):
    cursor.execute(
        "SELECT public.lab_arena_prepare_sep17_baseline_recovery278_v1("
        "%s,%s,%s,%s,%s::jsonb)",
        (SOURCE_SIZE, SOURCE_SHA, SOURCE_COMMIT, BANK_SHA, json.dumps(SCHEDULE)),
    )
    return cursor.fetchone()[0]


def _full_hash(cursor, predicate: str) -> tuple[str, int]:
    cursor.execute(
        "SELECT 'sha256:'||encode(extensions.digest(coalesce(string_agg("
        "encode(extensions.digest(to_jsonb(row_value)::text,'sha256'),'hex'),'' "
        "ORDER BY entry_id),''),'sha256'),'hex'),count(*) FROM "
        "public.lab_arena_ledger AS row_value WHERE round_id=%s AND " + predicate,
        (ROUND,),
    )
    return cursor.fetchone()


def test_recovery278_exact_fixture_replay_and_progress_idempotence(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        captured = _restore(connection)
        old_round = json.loads(captured["round"])
        old_baseline = next(
            json.loads(row) for row in captured["submissions"]
            if json.loads(row)["submission_id"] == BASELINE
        )
        migration = historical_migration_text()
        with connection.cursor() as cursor:
            assert _full_hash(cursor, "submission_id<>%s" % "'" + BASELINE + "'") == (
                NONBASELINE_LEDGER_HASH,
                15,
            )
            cursor.execute(migration)
            cursor.execute(migration)
            signature = (
                "public.lab_arena_prepare_sep17_baseline_recovery278_v1("
                "bigint,text,text,text,jsonb)"
            )
            for role, expected in (
                ("lab_arena_service", True),
                ("service_role", False),
                ("anon", False),
                ("authenticated", False),
            ):
                cursor.execute(
                    "SELECT has_function_privilege(%s,%s,'EXECUTE')",
                    (role, signature),
                )
                assert cursor.fetchone() == (expected,)
            cursor.execute(
                "SELECT relowner::regrole::text,relrowsecurity FROM pg_class "
                "WHERE oid='public.lab_arena_sep17_baseline_recovery278_authority'"
                "::regclass"
            )
            assert cursor.fetchone() == ("lab_arena_owner", True)
            assert _prepare(cursor)["status"] == "prepared"
            assert _prepare(cursor)["status"] == "existing"
        connection.commit()

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT configuration_doc,participants,status FROM "
                "public.lab_arena_rounds WHERE round_id=%s",
                (ROUND,),
            )
            configuration, participants, status = cursor.fetchone()
            assert status == "stage1"
            assert configuration["schedule"] == SCHEDULE
            assert configuration["call_quotas"]["openrouter"] == 200
            assert configuration["scoring_call_quotas"] == old_round[
                "configuration_doc"
            ]["scoring_call_quotas"]
            cursor.execute(
                "SELECT accepted_at,frozen_at,created_at,status,source_ref,"
                "source_size_bytes FROM public.lab_arena_submissions "
                "WHERE submission_id=%s",
                (BASELINE,),
            )
            accepted, frozen, created, submission_status, source_ref, size = cursor.fetchone()
            assert accepted == datetime.fromisoformat(old_baseline["accepted_at"])
            assert frozen == datetime.fromisoformat(old_baseline["frozen_at"])
            assert created == datetime.fromisoformat(old_baseline["created_at"])
            assert submission_status == old_baseline["status"] == "frozen"
            assert source_ref.endswith("baseline-2026-09-17-recovery278.tar.gz")
            assert size == SOURCE_SIZE
            assert _full_hash(cursor, "submission_id<>%s" % "'" + BASELINE + "'") == (
                NONBASELINE_LEDGER_HASH,
                15,
            )
            cursor.execute(
                "SELECT count(*),count(DISTINCT assignment_id) FROM "
                "public.lab_arena_runs WHERE round_id=%s AND submission_id=%s "
                "AND kind='execute' AND assignment_id LIKE '%%:rerun278'",
                (ROUND, BASELINE),
            )
            assert cursor.fetchone() == (20, 20)
            cursor.execute(
                "SELECT configuration_doc FROM public.lab_arena_rounds "
                "WHERE round_id=%s",
                (ARCHIVE,),
            )
            archived = cursor.fetchone()[0]
            expected_archive = dict(old_round["configuration_doc"])
            expected_archive.update({"round_id": ARCHIVE, "rewards_enabled": False})
            assert archived == expected_archive

        store = ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))
        runner = configuration["runner_hotkeys"][0]
        leased, token, _request, _request_hash = claim(
            store,
            ROUND,
            runner,
            parallelism=1,
            ceiling=20,
            excluded=[participants[0]["miner_hotkey"]],
        )
        assert leased["status"] == "leased"
        identity = contracts.provider_call_identity(
            attempt=leased["attempt"],
            assignment_id=leased["assignment_id"],
            icp_position=leased["icp_position"],
            action_sequence=0,
            operation_id="openrouter.responses",
            request_hash="sha256:" + "a" * 64,
        )
        reserved = store.reserve_call(
            run_id=leased["run_id"],
            lease_token_hash=hash_lease_token(token),
            call_identity=identity,
            operation_id="openrouter.responses",
            provider="openrouter",
            funding_source="host",
            amount_microusd=1000,
            call_doc={"test": "recovery278-progress"},
        )
        assert reserved["status"] == "reserved"
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM public.lab_arena_ledger")
            ledger_before = cursor.fetchone()[0]
            assert _prepare(cursor)["status"] == "existing"
            cursor.execute("SELECT count(*) FROM public.lab_arena_ledger")
            assert cursor.fetchone()[0] == ledger_before
        connection.commit()
    finally:
        connection.close()


def test_recovery278_replay_rejects_active_and_archive_drift(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        _restore(connection)
        with connection.cursor() as cursor:
            cursor.execute(historical_migration_text())
            assert _prepare(cursor)["status"] == "prepared"
        connection.commit()
        cases = (
            (
                "UPDATE public.lab_arena_rounds SET configuration_doc="
                "jsonb_set(configuration_doc,'{call_quotas,openrouter}','199') "
                "WHERE round_id=%s",
                (ROUND,),
            ),
            (
                "UPDATE public.lab_arena_rounds SET configuration_doc="
                "jsonb_set(configuration_doc,'{scoring_call_quotas,openrouter}','119') "
                "WHERE round_id=%s",
                (ARCHIVE,),
            ),
        )
        for statement, arguments in cases:
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL session_replication_role=replica")
                cursor.execute(statement, arguments)
                cursor.execute("SET LOCAL session_replication_role=origin")
                with pytest.raises(Exception, match="replay differs"):
                    _prepare(cursor)
            connection.rollback()

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT entry_id,submission_id FROM public.lab_arena_ledger "
                "WHERE round_id=%s AND submission_id<>%s ORDER BY entry_id LIMIT 1",
                (ROUND, BASELINE),
            )
            entry_id, old_submission = cursor.fetchone()
            cursor.execute(
                "SELECT submission_id FROM public.lab_arena_submissions "
                "WHERE round_id=%s AND submission_id NOT IN (%s,%s) LIMIT 1",
                (ROUND, BASELINE, old_submission),
            )
            replacement = cursor.fetchone()[0]
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_ledger SET submission_id=%s "
                "WHERE entry_id=%s",
                (replacement, entry_id),
            )
            cursor.execute("SET LOCAL session_replication_role=origin")
            with pytest.raises(Exception, match="replay differs"):
                _prepare(cursor)
        connection.rollback()
    finally:
        connection.close()


def test_recovery278_terminal_cost_state_fails_closed(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        _restore(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                "CREATE OR REPLACE FUNCTION public.lab_arena__successful_call_cost_state("
                "p_submission_id TEXT,p_kind TEXT,p_provider TEXT DEFAULT NULL) "
                "RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER "
                "SET search_path=pg_catalog,public AS $$ SELECT '{}'::jsonb $$"
            )
            with pytest.raises(Exception, match="terminal seal differs"):
                cursor.execute(historical_migration_text())
        connection.rollback()
    finally:
        connection.close()


def test_recovery278_uses_generic_scoring_and_preexecuted_stage2(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    store = ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))
    try:
        _restore(connection)
        with connection.cursor() as cursor:
            cursor.execute(historical_migration_text())
            assert _prepare(cursor)["status"] == "prepared"
        connection.commit()
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='accepted',"
                "terminal_cause='accepted',output_ref='arena/test/'||run_id||'.json' "
                "WHERE round_id=%s AND submission_id=%s AND kind='execute'",
                (ROUND, BASELINE),
            )
            assert cursor.rowcount == 20
            cursor.execute("SET LOCAL session_replication_role=origin")
        connection.commit()
        assert store.close_parallel_execution(ROUND)["status"] == "closed"
        execute_runs = store.list_runs(ROUND, kind="execute")
        stage1 = [row for row in execute_runs if row["stage"] == 1]
        work_items = [
            {
                "scored_run_id": row["run_id"],
                "submission_id": row["submission_id"],
                "icp_position": row["icp_position"],
                "output_ref": row["output_ref"],
            }
            for row in stage1
        ]
        plan = {"round_id": ROUND, "stage": 1, "work_items": work_items}
        assert store.transition_round(
            ROUND,
            "stage1_closed",
            "stage1_closed",
            {"stage1_scoring_plan_doc": plan},
        )["status"] == "ok"
        assert store.open_scoring(ROUND, 1, work_items)["assignments"] == 10
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='accepted',"
                "terminal_cause='accepted',output_ref='arena/test/'||run_id||'.json' "
                "WHERE round_id=%s AND kind='score' AND stage=1",
                (ROUND,),
            )
            assert cursor.rowcount == 10
            cursor.execute("SET LOCAL session_replication_role=origin")
        connection.commit()
        assert store.close_scoring(ROUND, 1)["status"] == "closed"
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET status='stage1_scored',"
                "status_generation=status_generation+1 WHERE round_id=%s",
                (ROUND,),
            )
            cursor.execute("SET LOCAL session_replication_role=origin")
        connection.commit()
        assert store.activate_preexecuted_stage2(ROUND)["status"] == "ok"
        assert store.get_round(ROUND)["status"] == "stage2"
        after = store.list_runs(ROUND, kind="execute")
        assert len(after) == 20
        assert len({row["assignment_id"] for row in after}) == 20
        assert all(row["assignment_id"].endswith(":rerun278") for row in after)
    finally:
        connection.close()
