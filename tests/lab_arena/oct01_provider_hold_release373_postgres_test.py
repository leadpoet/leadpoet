"""Disposable-PostgreSQL proof for the exact October 1 hold release."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.lab_arena import oct01_held_deadline_recovery372_postgres_test as held


MIGRATION = Path(__file__).parents[2] / "scripts/373-arena-2026-10-01-provider-hold-release.sql"
database = held.database
NEW_ACTOR = "canonical-active-release:" + "d" * 40
ORIGINAL_HASHES = {
    "e7cd2d1d2fea2a1f2ae41b313889545d36ca3e9b09e5d3d4021fec36a99b0a7b":
        "(SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-01')",
    "872f81d1d045dbd2f2d2b7d52b7f5185b9841add7a4ed14eb21a2aea67b17d19":
        "(SELECT participants FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-01')",
    "8fe8de7ccf091baaa2fedde44b4d1c01e84fe1999f8b7f086d82a0f3332c1e61":
        "(SELECT icps FROM public.qualification_private_icp_sets WHERE set_id=20260930)",
    "1c9caa3e983e80458a9c0910255f14387f37098d288c29b596090765d04112b3":
        "(SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM public.lab_arena_submissions s WHERE round_id='arena-2026-10-01')",
}


def _prepare(cursor):
    recovery_sql, _, _ = held._prepare(cursor)
    release_sql = MIGRATION.read_text()
    for original, expression in ORIGINAL_HASHES.items():
        assert original in release_sql
        release_sql = release_sql.replace(original, held._hash(cursor, expression))
    for original, expression in (
        ("dad688e5ab73c5f5e6671299f10d914d5462a5b8333d24b9ddb53b467bf58a63",
         "(SELECT jsonb_agg(jsonb_build_array(run_id,assignment_id,icp_position,attempt,status,terminal_cause,stage_generation,scored_run_id) ORDER BY run_id) FROM public.lab_arena_runs WHERE round_id='arena-2026-10-01' AND kind='score')"),
        ("2b0443f2ba2cf021628d34bebd575886be427c4749536b2c8879d3b9bc10239a",
         "(SELECT jsonb_agg(jsonb_build_array(run_id,assignment_id,submission_id,icp_position,attempt,status,stage_generation) ORDER BY run_id) FROM public.lab_arena_runs WHERE round_id='arena-2026-10-01' AND kind='execute' AND stage=2)"),
    ):
        assert original in release_sql
        release_sql = release_sql.replace(original, held._hash(cursor, expression))
    cursor.execute(recovery_sql)
    cursor.execute("SET session_replication_role=replica")
    cursor.execute(
        "UPDATE public.lab_arena_restart_claim_control "
        "SET actor_ref=%s,guard_generation=302 WHERE singleton", (NEW_ACTOR,)
    )
    cursor.execute("SET session_replication_role=origin")
    decision_clock = "IF pg_catalog.clock_timestamp() >=\n      (v_new_schedule"
    assert decision_clock in release_sql
    release_sql = release_sql.replace(
        decision_clock,
        "IF '2026-10-01T15:00:00Z'::TIMESTAMPTZ >=\n      (v_new_schedule",
    )
    return release_sql


def _snapshot(cursor):
    return {
        (table, round_id): held._rows(cursor, table, round_id)
        for table in ("lab_arena_rounds", "lab_arena_submissions", "lab_arena_runs",
                      "lab_arena_ledger", "lab_arena_trajectory_events")
        for round_id in (held.ROUND, held.ARCHIVE, "arena-2026-10-02")
    }


def test_releases_only_claim_control_and_replay_preserves_timestamp(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql = _prepare(cursor)
            before = _snapshot(cursor)
            cursor.execute(sql)
            cursor.execute("SELECT operator_paused,pause_reason,actor_ref,updated_at "
                           "FROM public.lab_arena_restart_claim_control WHERE singleton")
            released = cursor.fetchone()
            assert released[:3] == (False, "", "oct01-provider-recovery373")
            assert _snapshot(cursor) == before
            cursor.execute(sql)
            cursor.execute("SELECT operator_paused,pause_reason,actor_ref,updated_at "
                           "FROM public.lab_arena_restart_claim_control WHERE singleton")
            assert cursor.fetchone() == released
            assert _snapshot(cursor) == before


@pytest.mark.parametrize("tamper", ["foreign_reason", "active_guard", "expired_window"])
def test_fail_closed_without_mutation(database, tamper):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql = _prepare(cursor)
            cursor.execute("SET session_replication_role=replica")
            if tamper == "foreign_reason":
                cursor.execute("UPDATE public.lab_arena_restart_claim_control "
                               "SET pause_reason='foreign' WHERE singleton")
            elif tamper == "active_guard":
                cursor.execute("UPDATE public.lab_arena_restart_claim_control SET "
                               "guard_commitment='sha256:'||repeat('a',64),"
                               "owner_commitment='sha256:'||repeat('b',64),"
                               "guard_expires_at=now()+interval '1 hour',"
                               "candidate_commit=repeat('c',40),"
                               "restart_scope='all',restart_phase='draining' "
                               "WHERE singleton")
            cursor.execute("SET session_replication_role=origin")
            if tamper == "expired_window":
                sql = sql.replace("'2026-10-01T15:00:00Z'::TIMESTAMPTZ >=\n      (v_new_schedule",
                                  "'2026-10-01T20:01:00Z'::TIMESTAMPTZ >=\n      (v_new_schedule")
                assert "'2026-10-01T20:01:00Z'::TIMESTAMPTZ" in sql
            before = _snapshot(cursor)
            cursor.execute("SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c "
                           "WHERE singleton")
            control_before = cursor.fetchone()[0]
            with pytest.raises(psycopg.Error, match="Oct01 release"):
                cursor.execute(sql)
            cursor.execute("ROLLBACK")
            assert _snapshot(cursor) == before
            cursor.execute("SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c "
                           "WHERE singleton")
            assert cursor.fetchone()[0] == control_before
