"""Verbatim disposable-PostgreSQL proof for the held October 1 deadline recovery."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena import oct01_provider_hold369_postgres_test as hold
from tests.lab_arena.icp_fixtures import daily_icps


ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/372-arena-2026-10-01-held-deadline-recovery.sql"
ROUND = hold.recovery.ROUND
ARCHIVE = ROUND + "-r372archive"
BASELINE = hold.recovery.BASELINE
CANONICAL_ACTOR = (
    "canonical-active-release:07016ffd02e174b6deaaa136e0e8e216d498a8d2"
)
database = hold.database


def _hash(cursor, expression: str) -> str:
    cursor.execute(
        "SELECT encode(extensions.digest((" + expression + ")::text,'sha256'),'hex')"
    )
    return cursor.fetchone()[0]


def _assert_jsonb_array_hash_matches_python(cursor, expression: str) -> None:
    cursor.execute("SELECT " + expression)
    value = cursor.fetchone()[0]
    py_hash = hashlib.sha256(json.dumps(
        value, ensure_ascii=False, separators=(", ", ": ")
    ).encode()).hexdigest()
    assert py_hash == _hash(cursor, expression)


def _prepare(cursor):
    hold._seed_active(cursor)
    cursor.execute(hold._render_hold(cursor))
    cursor.execute("SET session_replication_role=replica")
    cursor.execute(
        "UPDATE public.lab_arena_restart_claim_control "
        "SET actor_ref=%s,guard_generation=301,"
        "pause_reason='canonical_restart_guard' WHERE singleton",
        (CANONICAL_ACTOR,),
    )
    cursor.execute(
        "UPDATE public.lab_arena_runs SET per_icp_score=0 "
        "WHERE round_id=%s AND kind='execute' AND stage=1", (ROUND,)
    )
    assert cursor.rowcount == 10
    cursor.execute(
        "UPDATE public.lab_arena_runs SET status='failed',"
        "terminal_cause=CASE WHEN icp_position IN (0,3) THEN 'credential_error' "
        "WHEN icp_position IN (2) THEN 'judge_error' ELSE 'stage_closed' END,"
        "result_doc='{\"terminal_status\":\"failed\"}'::jsonb "
        "WHERE round_id=%s AND kind='score' AND icp_position<>1", (ROUND,)
    )
    assert cursor.rowcount == 9
    for position in (2, 3):
        cursor.execute(
            "INSERT INTO public.lab_arena_runs "
            "(run_id,assignment_id,round_id,submission_id,miner_hotkey,"
            "stage,icp_position,attempt,kind,status,stage_generation,"
            "scored_run_id,terminal_cause,result_doc) "
            "SELECT assignment_id||':2',assignment_id,round_id,submission_id,"
            "miner_hotkey,stage,icp_position,2,kind,'failed',stage_generation,"
            "scored_run_id,terminal_cause,result_doc "
            "FROM public.lab_arena_runs WHERE run_id=%s",
            (f"score369:{position}:1",),
        )
    cursor.execute(
        "SELECT run_id,submission_id,icp_position,output_ref "
        "FROM public.lab_arena_runs WHERE round_id=%s AND stage=1 "
        "AND kind='execute' ORDER BY icp_position", (ROUND,)
    )
    work_items = [
        {"scored_run_id": run_id, "submission_id": submission_id,
         "icp_position": position, "output_ref": output_ref}
        for run_id, submission_id, position, output_ref in cursor.fetchall()
    ]
    plan = {
        "schema_version": contracts.SCORING_PLAN_SCHEMA_VERSION,
        "round_id": ROUND, "stage": 1,
        "execution_sequence_policy": contracts.BASELINE_SCORED_FIRST_POLICY,
        "work_items": work_items, "zero_rows": [],
    }
    cursor.execute(
        "SELECT participants FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,)
    )
    finalists = sorted(
        p["submission_id"] for p in cursor.fetchone()[0] if not p["is_king"]
    )
    cursor.execute(
        "UPDATE public.lab_arena_rounds SET status='stage2',"
        "status_generation=9,stage_generation=7,"
        "stage1_scoring_plan_doc=%s::jsonb,finalists=%s::jsonb "
        "WHERE round_id=%s",
        (json.dumps(plan), json.dumps(finalists), ROUND),
    )
    cursor.execute(
        "INSERT INTO public.lab_arena_runs "
        "(run_id,assignment_id,round_id,submission_id,miner_hotkey,"
        "stage,icp_position,attempt,kind,status,stage_generation) "
        "SELECT %s||':'||s.submission_id||':2:'||p.position||':1',"
        "%s||':'||s.submission_id||':2:'||p.position,%s,s.submission_id,"
        "s.miner_hotkey,2,p.position,1,'execute','pending',7 "
        "FROM public.lab_arena_submissions s CROSS JOIN generate_series(0,9) p(position) "
        "WHERE s.round_id=%s AND NOT s.is_king",
        (ROUND, ROUND, ROUND, ROUND),
    )
    assert cursor.rowcount == 130
    cursor.execute("SET session_replication_role=origin")
    sql = MIGRATION.read_text()
    rendered_hold = hold._render_hold(cursor)
    import re
    frozen = re.findall(r"'[0-9a-f]{64}'", hold.HOLD.read_text())
    fixture = re.findall(r"'[0-9a-f]{64}'", rendered_hold)
    assert len(frozen) == len(fixture) == 5
    for old, new in zip(frozen[:4], fixture[:4]):
        assert old in sql
        sql = sql.replace(old, new)
    cursor.execute("SELECT configuration_doc->>'scorer_image_digest' "
                   "FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,))
    sql = sql.replace(
        "sha256:f8ab912f739a1c9e30cc33fb4a5f4ea86b7ef4680af571dc203574f6473f13eb",
        cursor.fetchone()[0],
    )
    _assert_jsonb_array_hash_matches_python(
        cursor, "(SELECT jsonb_agg(jsonb_build_array(run_id,assignment_id,"
                "icp_position,attempt,status,terminal_cause,stage_generation,"
                "scored_run_id) ORDER BY run_id) FROM public.lab_arena_runs "
                "WHERE round_id='arena-2026-10-01' AND kind='score')",
    )
    _assert_jsonb_array_hash_matches_python(
        cursor, "(SELECT jsonb_agg(jsonb_build_array(run_id,assignment_id,"
                "submission_id,icp_position,attempt,status,stage_generation) "
                "ORDER BY run_id) FROM public.lab_arena_runs WHERE "
                "round_id='arena-2026-10-01' AND kind='execute' AND stage=2)",
    )
    sql = sql.replace(
        "dad688e5ab73c5f5e6671299f10d914d5462a5b8333d24b9ddb53b467bf58a63",
        _hash(cursor, "(SELECT jsonb_agg(jsonb_build_array(run_id,assignment_id,"
                      "icp_position,attempt,status,terminal_cause,stage_generation,"
                      "scored_run_id) ORDER BY run_id) FROM public.lab_arena_runs "
                      "WHERE round_id='arena-2026-10-01' AND kind='score')"),
    ).replace(
        "2b0443f2ba2cf021628d34bebd575886be427c4749536b2c8879d3b9bc10239a",
        _hash(cursor, "(SELECT jsonb_agg(jsonb_build_array(run_id,assignment_id,"
                      "submission_id,icp_position,attempt,status,stage_generation) "
                      "ORDER BY run_id) FROM public.lab_arena_runs WHERE "
                      "round_id='arena-2026-10-01' AND kind='execute' AND stage=2)"),
    )
    decision_clock = "IF pg_catalog.clock_timestamp() >=\n      (v_new_schedule"
    assert decision_clock in sql
    sql = sql.replace(
        decision_clock,
        "IF '2026-10-01T15:00:00Z'::TIMESTAMPTZ >=\n      (v_new_schedule",
    )
    return sql, plan, work_items


def _rows(cursor, table, round_id):
    cursor.execute(f"SELECT to_jsonb(t) FROM public.{table} t "
                   "WHERE round_id=%s ORDER BY 1", (round_id,))
    return [row[0] for row in cursor.fetchall()]


def test_recovery_archives_invalid_derivation_and_keeps_hold(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql, plan, _ = _prepare(cursor)
            before_scores = _rows(cursor, "lab_arena_runs", ROUND)
            before_ledger = _rows(cursor, "lab_arena_ledger", ROUND)
            before_events = _rows(cursor, "lab_arena_trajectory_events", ROUND)
            foreign_before = _rows(cursor, "lab_arena_rounds", "arena-2026-10-02")
            cursor.execute(sql)
            cursor.execute("SELECT status,status_generation,stage_generation,"
                           "finalists,configuration_doc->'schedule' "
                           "FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,))
            status, status_gen, stage_gen, finalists, schedule = cursor.fetchone()
            assert (status, status_gen, stage_gen, finalists) == (
                "stage1_scoring", 10, 8, None
            )
            assert schedule["stage_1_scoring_close"] == "2026-10-01T20:00:01Z"
            cursor.execute("SELECT operator_paused,pause_reason,actor_ref,guard_generation FROM "
                           "public.lab_arena_restart_claim_control WHERE singleton")
            assert cursor.fetchone() == (
                True, "oct01_deepline_outage", CANONICAL_ACTOR, 301
            )
            assert _rows(cursor, "lab_arena_ledger", ROUND) == before_ledger
            assert _rows(cursor, "lab_arena_trajectory_events", ROUND) == before_events
            assert _rows(cursor, "lab_arena_rounds", "arena-2026-10-02") == foreign_before
            after = _rows(cursor, "lab_arena_runs", ROUND)
            before_score_attempts = [r for r in before_scores if r["kind"] == "score"]
            assert all(r in after for r in before_score_attempts)
            assert len([r for r in after if r["kind"] == "score" and
                        r["status"] == "pending" and r["stage_generation"] == 8]) == 9
            assert len([r for r in after if r["kind"] == "execute" and
                        r["stage"] == 2]) == 0
            assert len([r for r in after if r["kind"] == "execute" and
                        r["stage"] == 1 and r["per_icp_score"] is None]) == 10
            audit = _rows(cursor, "lab_arena_runs", ARCHIVE)
            assert len(audit) == 140
            assert len([r for r in audit if r["stage"] == 1 and
                        r["per_icp_score"] == 0]) == 10
            assert len([r for r in audit if r["stage"] == 2 and
                        r["status"] == "pending"]) == 130
            cursor.execute("SELECT status,finalists,stage1_scoring_plan_doc "
                           "FROM public.lab_arena_rounds WHERE round_id=%s", (ARCHIVE,))
            archived_status, archived_finalists, archived_plan = cursor.fetchone()
            assert archived_status == "cancelled" and len(archived_finalists) == 13
            assert archived_plan == plan
            assert _rows(cursor, "lab_arena_ledger", ARCHIVE) == []
            once = (_rows(cursor, "lab_arena_runs", ROUND),
                    _rows(cursor, "lab_arena_runs", ARCHIVE))
            cursor.execute(sql)
            assert once == (_rows(cursor, "lab_arena_runs", ROUND),
                            _rows(cursor, "lab_arena_runs", ARCHIVE))


def test_replay_rejects_changed_archive_snapshot(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql, _, _ = _prepare(cursor)
            cursor.execute(sql)
            cursor.execute("SET session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_runs SET result_doc='{}'::jsonb "
                           "WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs "
                           "WHERE round_id=%s AND stage=1)", (ARCHIVE,))
            cursor.execute("SET session_replication_role=origin")
            before = _rows(cursor, "lab_arena_runs", ARCHIVE)
            with pytest.raises(psycopg.Error, match="archive replay differs"):
                cursor.execute(sql)
            cursor.execute("ROLLBACK")
            assert _rows(cursor, "lab_arena_runs", ARCHIVE) == before


@pytest.mark.parametrize("tamper", [
    "score", "pending_job", "foreign_stage", "foreign_owner", "foreign_reason",
    "active_guard",
])
def test_preimage_drift_fails_without_mutation(database, tamper):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql, _, _ = _prepare(cursor)
            cursor.execute("SET session_replication_role=replica")
            if tamper == "score":
                cursor.execute("UPDATE public.lab_arena_runs SET terminal_cause='stage_closed' "
                               "WHERE run_id='score369:0:1'")
            elif tamper == "pending_job":
                cursor.execute("UPDATE public.lab_arena_runs SET status='leased' "
                               "WHERE run_id=(SELECT min(run_id) FROM public.lab_arena_runs "
                               "WHERE round_id=%s AND stage=2)", (ROUND,))
            elif tamper == "foreign_owner":
                cursor.execute("UPDATE public.lab_arena_restart_claim_control "
                               "SET actor_ref='foreign-operator' WHERE singleton")
            elif tamper == "foreign_reason":
                cursor.execute("UPDATE public.lab_arena_restart_claim_control "
                               "SET pause_reason='foreign' WHERE singleton")
            elif tamper == "active_guard":
                cursor.execute("UPDATE public.lab_arena_restart_claim_control "
                               "SET guard_commitment='sha256:'||repeat('a',64),"
                               "owner_commitment='sha256:'||repeat('b',64),"
                               "guard_expires_at=now()+interval '1 hour',"
                               "candidate_commit=repeat('c',40),"
                               "restart_scope='all',restart_phase='draining' "
                               "WHERE singleton")
            else:
                cursor.execute("UPDATE public.lab_arena_rounds SET status='stage1' "
                               "WHERE round_id='arena-2026-10-02'")
            cursor.execute("SET session_replication_role=origin")
            before = _rows(cursor, "lab_arena_runs", ROUND)
            with pytest.raises(psycopg.Error, match="Oct01 deadline recovery"):
                cursor.execute(sql)
            cursor.execute("ROLLBACK")
            assert _rows(cursor, "lab_arena_runs", ROUND) == before
            assert _rows(cursor, "lab_arena_runs", ARCHIVE) == []


def test_normal_stage2_open_recreates_original_miner_jobs(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            sql, plan, work_items = _prepare(cursor)
            cursor.execute("SELECT run_id FROM public.lab_arena_runs "
                           "WHERE round_id=%s AND stage=2 ORDER BY run_id", (ROUND,))
            original_ids = [row[0] for row in cursor.fetchall()]
            cursor.execute(sql)
            cursor.execute("SET session_replication_role=replica")
            cursor.execute("UPDATE public.lab_arena_restart_claim_control "
                           "SET operator_paused=false,pause_reason='' WHERE singleton")
            cursor.execute("UPDATE public.lab_arena_runs SET status='accepted',"
                           "terminal_cause='accepted',"
                           "output_ref='arena/test/'||run_id||'.json' "
                           "WHERE round_id=%s AND kind='score' AND status='pending'", (ROUND,))
            cursor.execute("SET session_replication_role=origin")
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    assert store.close_scoring(ROUND, 1)["status"] == "closed"
    service = object.__new__(ArenaService)
    service._store = store
    service._round = lambda round_id: store.get_round(round_id)
    service._load_scoring_plan = lambda round_row, stage: plan
    service.evaluation_icps = lambda round_id: daily_icps()[:10]
    service._outputs_by_run = lambda round_id, stage: {
        item["scored_run_id"]: [] for item in work_items
    }
    service._verified_breakdowns = lambda run, **kwargs: []
    assert service.score_stage(ROUND, 1)["status"] == "ok"
    accepted = next(r for r in store.list_runs(ROUND, stage=1, kind="score")
                    if r["status"] == "accepted" and r["icp_position"] == 1)
    selected = ArenaService._select_scoring_outputs(
        store.list_runs(ROUND, stage=1, kind="score")
    )
    assert selected[accepted["scored_run_id"]]["run_id"] == accepted["run_id"]
    round_row = store.get_round(ROUND)
    participants = [p for p in round_row["participants"] if not p["is_king"]]
    opened = store.open_stage(ROUND, 2, participants, list(range(10)))
    assert opened["status"] == "ok" and opened["assignments"] == 130
    assert sorted(r["run_id"] for r in store.list_runs(ROUND, stage=2,
                                                     kind="execute")) == original_ids
    assert store.get_round(ROUND)["stage_generation"] > 8
