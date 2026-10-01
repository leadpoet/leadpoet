"""Disposable PostgreSQL proof for the sealed October 1 baseline recovery."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from lab_arena import scoring
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_contracts import base_round_configuration
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey


ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/368-arena-2026-10-01-cancelled-baseline-recovery.sql"
ROUND = "arena-2026-10-01"
ARCHIVE = ROUND + "-r368archive"
BASELINE = "baseline-2026-10-01"
SUFFIX = ":r368archive"
OLD_SCHEDULE = {
    "submission_open": "2026-09-30T00:00:00Z",
    "submission_cutoff": "2026-10-01T00:00:00Z",
    "benchmark_deadline": "2026-10-01T00:30:00Z",
    "stage_1_start": "2026-10-01T00:30:01Z",
    "stage_1_close": "2026-10-01T04:30:01Z",
    "stage_1_scoring_close": "2026-10-01T11:00:01Z",
    "stage_2_start": "2026-10-01T11:00:02Z",
    "stage_2_close": "2026-10-01T14:00:02Z",
    "final_scoring_close": "2026-10-01T20:30:02Z",
    "publication_deadline": "2026-10-01T20:30:03Z",
}
NEW_SCHEDULE = {
    **OLD_SCHEDULE,
    "stage_1_close": "2026-10-01T11:00:01Z",
    "stage_1_scoring_close": "2026-10-01T14:00:01Z",
    "stage_2_start": "2026-10-01T14:00:02Z",
    "stage_2_close": "2026-10-01T17:00:02Z",
    "final_scoring_close": "2026-10-01T23:00:02Z",
    "publication_deadline": "2026-10-01T23:00:03Z",
}
PRODUCTION_HASHES = {
    "round": "dbbaf18ff42f77aa56c8dc6f91780a8887195763e43827f3de93f57f80dd3623",
    "submissions": "1c9caa3e983e80458a9c0910255f14387f37098d288c29b596090765d04112b3",
    "runs": "a28d65a9d8e350b081473855e5d239f5d27bc92c8fe9afc083288a3a7f7ac2d8",
    "events": "de57bf99060cfe8cf5170436a5a25ac9fa3bbf71d66dfb3e8a167116f6cc1d0b",
    "bank": "8fe8de7ccf091baaa2fedde44b4d1c01e84fe1999f8b7f086d82a0f3332c1e61",
    "new_config": "e7cd2d1d2fea2a1f2ae41b313889545d36ca3e9b09e5d3d4021fec36a99b0a7b",
    "participants": "872f81d1d045dbd2f2d2b7d52b7f5185b9841add7a4ed14eb21a2aea67b17d19",
    "archive_round": "05b787490ac66170bb64dc1292a8e371f4bbcd3fe9ed7f028cfee02d3143ea5c",
    "archive_submissions": "d9282fda079e8669cf67643c574cc634091423a30f1516125591868c78bcbe15",
    "archive_runs": "a678e3f3da6cc58b66b494e848e7d0d2c265e820c5209ff7ea273107802a871f",
    "archive_events": "903d4594445baf8290d9b88b04a09ae02dce2f18afc98c5d45f86df3ff6543bc",
}


@pytest.fixture
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS + ("365-lab-arena-trajectories.sql",)
    )


def _configuration() -> dict:
    config = base_round_configuration()
    config.update(
        round_id=ROUND,
        mode="live",
        rewards_enabled=True,
        schedule=OLD_SCHEDULE,
        stage_1_icp_count=5,
        stage_2_icp_count=5,
        execution_sequence_policy="baseline_scored_first_v1",
        icp_wall_clock_seconds=3600,
        lease_ttl_seconds=4500,
        sourcing_cost_eligibility_policy="successful_calls_per_icp_v1",
        execution_icp_cap_microusd=4_000_000,
        cost_per_company_microusd=800_000,
        scorer_policy=scoring.build_scorer_policy(
            scoring_adapter_version="qualification_integrity_v2",
            intent_details=True,
            normalize_intent_scale=True,
        ),
    )
    return config


def _participants() -> list[dict]:
    miners = [
        {
            "submission_id": f"sub-oct01-fixture-{index:02d}",
            "miner_hotkey": hotkey(f"oct01-recovery-{index}"),
            "source_ref": f"arena/{ROUND}/sources/sub-oct01-fixture-{index:02d}.tar.gz",
            "source_size_bytes": 100_000 + index,
            "is_king": False,
        }
        for index in range(13)
    ]
    miners.append({
        "submission_id": BASELINE,
        "miner_hotkey": _configuration()["baseline_hotkey"],
        "source_ref": f"arena/{ROUND}/sources/{BASELINE}.tar.gz",
        "source_size_bytes": 239_289,
        "is_king": True,
    })
    return miners


def _seed(cursor, *, foreign: bool = False) -> None:
    config = _configuration()
    participants = _participants()
    cursor.execute("SET session_replication_role=replica")
    cursor.execute(
        "INSERT INTO public.qualification_private_icp_sets "
        "(set_id,icps,active_from,active_until,is_active) VALUES "
        "(20260930,%s::jsonb,'2026-09-30T00:00:00Z',"
        "'2026-10-01T00:00:00Z',false)",
        (json.dumps([{"icp_id": f"oct01-{i}", "prompt": f"ICP {i}"}
                     for i in range(10)]),),
    )
    cursor.execute(
        "INSERT INTO public.lab_arena_rounds "
        "(round_id,status,status_generation,stage_generation,configuration_doc,"
        "rewards_enabled,participants,benchmark_ref,evaluation_date,"
        "icp_set_date,cancel_reason,promotion_required,champion_funding_frozen,"
        "champion_submission_id,champion_hotkey) VALUES "
        "(%s,'cancelled',3,2,%s::jsonb,true,%s::jsonb,%s,"
        "'2026-10-01','2026-09-30','execution_incomplete:stage1:10',"
        "true,true,%s,%s)",
        (ROUND, json.dumps(config), json.dumps(participants),
         f"arena/{ROUND}/benchmark.json",
         participants[0]["submission_id"], participants[0]["miner_hotkey"]),
    )
    for participant in participants:
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions "
            "(submission_id,round_id,miner_hotkey,status,is_king,"
            "submission_doc,source_ref,source_size_bytes) VALUES "
            "(%s,%s,%s,'frozen',%s,%s::jsonb,%s,%s)",
            (participant["submission_id"], ROUND, participant["miner_hotkey"],
             participant["is_king"],
             json.dumps({"source_ref": participant["source_ref"],
                         "source_size_bytes": participant["source_size_bytes"]}),
             participant["source_ref"], participant["source_size_bytes"]),
        )
    for position in range(10):
        assignment = f"{ROUND}:{BASELINE}:1:{position}"
        for attempt in (1, 2):
            run_id = f"{assignment}:{attempt}"
            cause = "stage_closed" if position == 9 and attempt == 2 else "lease_expired"
            cursor.execute(
                "INSERT INTO public.lab_arena_runs "
                "(run_id,assignment_id,round_id,submission_id,miner_hotkey,"
                "stage,icp_position,attempt,kind,status,terminal_cause,"
                "terminal_doc,stage_generation,lease_generation,runner_hotkey) "
                "VALUES (%s,%s,%s,%s,%s,1,%s,%s,'execute','failed',%s,"
                "%s::jsonb,1,1,%s)",
                (run_id, assignment, ROUND, BASELINE,
                 participants[-1]["miner_hotkey"], position, attempt, cause,
                 json.dumps({"closed_at": "2026-10-01T04:31:05Z"} if cause == "stage_closed"
                            else {"expired_at": "2026-10-01T02:33:22Z"}),
                 participants[-1]["miner_hotkey"]),
            )
            for ordinal, kind in enumerate(("runtime.start", "runtime.error", "runtime.end"), 1):
                event_id = str(uuid.UUID(int=(position * 2 + attempt) * 3 + ordinal))
                cursor.execute(
                    "INSERT INTO public.lab_arena_trajectory_events "
                    "(run_id,event_id,round_id,submission_id,miner_hotkey,"
                    "runner_hotkey,assignment_id,icp_identifier,stage,"
                    "icp_position,attempt,run_kind,model_role,event_kind,"
                    "occurred_at,content) VALUES "
                    "(%s,%s,%s,%s,%s,%s,%s,%s,1,%s,%s,'execute','baseline',"
                    "%s,'2026-10-01T02:30:00Z',%s::jsonb)",
                    (run_id, event_id, ROUND, BASELINE,
                     participants[-1]["miner_hotkey"],
                     participants[-1]["miner_hotkey"], assignment,
                     f"oct01-{position}", position, attempt, kind,
                     json.dumps({"kind": kind, "ordinal": ordinal})),
                )
    if foreign:
        foreign_round = "arena-2026-10-02"
        foreign_config = dict(config, round_id=foreign_round)
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds "
            "(round_id,status,configuration_doc,rewards_enabled) "
            "VALUES (%s,'open',%s::jsonb,true)",
            (foreign_round, json.dumps(foreign_config)),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions "
            "(submission_id,round_id,miner_hotkey,status) VALUES "
            "('sub-foreign368',%s,%s,'frozen')",
            (foreign_round, hotkey("foreign368")),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_runs "
            "(run_id,assignment_id,round_id,submission_id,miner_hotkey,"
            "stage,icp_position,attempt,status) VALUES "
            "('foreign368:1','foreign368',%s,'sub-foreign368',%s,1,0,1,'pending')",
            (foreign_round, hotkey("foreign368")),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_trajectory_events "
            "(run_id,event_id,round_id,submission_id,miner_hotkey,"
            "runner_hotkey,assignment_id,icp_identifier,stage,icp_position,"
            "attempt,run_kind,model_role,event_kind,occurred_at,content) VALUES "
            "('foreign368:1',%s,%s,'sub-foreign368',%s,%s,'foreign368',"
            "'foreign-icp',1,0,1,'execute','miner','runtime.start',"
            "'2026-10-01T02:30:00Z','{}'::jsonb)",
            (str(uuid.UUID(int=999)), foreign_round,
             hotkey("foreign368"), hotkey("foreign368")),
        )
    cursor.execute("SET session_replication_role=origin")


def _one_hash(cursor, table: str, row: dict) -> str:
    cursor.execute(
        f"SELECT encode(extensions.digest(to_jsonb(jsonb_populate_record("
        f"NULL::public.{table},%s::jsonb))::text,'sha256'),'hex')",
        (json.dumps(row),),
    )
    return cursor.fetchone()[0]


def _many_hash(cursor, table: str, rows: list[dict], order: str) -> str:
    cursor.execute(
        f"SELECT encode(extensions.digest(jsonb_agg(to_jsonb(r) ORDER BY {order})"
        f"::text,'sha256'),'hex') FROM jsonb_populate_recordset("
        f"NULL::public.{table},%s::jsonb) r",
        (json.dumps(rows),),
    )
    return cursor.fetchone()[0]


def _rows(cursor, table: str, where: str, order: str) -> list[dict]:
    cursor.execute(f"SELECT to_jsonb(t) FROM public.{table} t WHERE {where} ORDER BY {order}")
    return [row[0] for row in cursor.fetchall()]


def _render_sql(cursor) -> str:
    round_row = _rows(cursor, "lab_arena_rounds", f"round_id='{ROUND}'", "round_id")[0]
    subs = _rows(cursor, "lab_arena_submissions", f"round_id='{ROUND}'", "submission_id")
    runs = _rows(cursor, "lab_arena_runs", f"round_id='{ROUND}'", "run_id")
    events = _rows(cursor, "lab_arena_trajectory_events", f"round_id='{ROUND}'", "trajectory_id")
    cursor.execute("SELECT icps FROM public.qualification_private_icp_sets WHERE set_id=20260930")
    bank = cursor.fetchone()[0]
    hashes = {
        "round": _one_hash(cursor, "lab_arena_rounds", round_row),
        "submissions": _many_hash(cursor, "lab_arena_submissions", subs, "submission_id"),
        "runs": _many_hash(cursor, "lab_arena_runs", runs, "run_id"),
        "events": _many_hash(cursor, "lab_arena_trajectory_events", events, "trajectory_id"),
        "participants": _json_hash(cursor, round_row["participants"]),
        "bank": _json_hash(cursor, bank),
    }
    new_config = dict(round_row["configuration_doc"], schedule=NEW_SCHEDULE)
    hashes["new_config"] = _json_hash(cursor, new_config)
    archive_round = dict(round_row)
    archive_round.update(round_id=ARCHIVE, rewards_enabled=False,
                         cancel_reason="authorized_oct01_recovery368_archive")
    archive_round["configuration_doc"] = dict(
        round_row["configuration_doc"], round_id=ARCHIVE,
        mode="shadow", rewards_enabled=False,
        recovery_source_round_id=ROUND,
        recovery_preimage_sha256="sha256:" + hashes["round"],
    )
    archive_round["participants"] = [
        dict(p, submission_id=p["submission_id"] + SUFFIX)
        for p in round_row["participants"]
    ]
    archive_subs = [dict(s, round_id=ARCHIVE,
                         submission_id=s["submission_id"] + SUFFIX) for s in subs]
    archive_runs = [dict(r, round_id=ARCHIVE,
                         submission_id=BASELINE + SUFFIX) for r in runs]
    archive_events = [dict(e, round_id=ARCHIVE,
                           submission_id=BASELINE + SUFFIX) for e in events]
    hashes.update(
        archive_round=_one_hash(cursor, "lab_arena_rounds", archive_round),
        archive_submissions=_many_hash(cursor, "lab_arena_submissions", archive_subs, "submission_id"),
        archive_runs=_many_hash(cursor, "lab_arena_runs", archive_runs, "run_id"),
        archive_events=_many_hash(cursor, "lab_arena_trajectory_events", archive_events, "trajectory_id"),
    )
    sql = MIGRATION.read_text(encoding="utf-8")
    for key, old in PRODUCTION_HASHES.items():
        assert old in sql, key
        sql = sql.replace(old, hashes[key])
    # The fixture is historical; test the wall-clock guard with a fixed instant.
    old_now = "pg_catalog.clock_timestamp() + INTERVAL '150 minutes'"
    assert sql.count(old_now) == 1
    return sql.replace(old_now,
                       "'2026-10-01T05:00:00Z'::TIMESTAMPTZ + INTERVAL '150 minutes'")


def _json_hash(cursor, value) -> str:
    cursor.execute(
        "SELECT encode(extensions.digest((%s::jsonb)::text,'sha256'),'hex')",
        (json.dumps(value),),
    )
    return cursor.fetchone()[0]


def _state(cursor) -> tuple:
    return tuple(
        _rows(cursor, table, "true", order)
        for table, order in (
            ("lab_arena_rounds", "round_id"),
            ("lab_arena_submissions", "submission_id"),
            ("lab_arena_runs", "run_id"),
            ("lab_arena_trajectory_events", "trajectory_id"),
            ("lab_arena_ledger", "entry_id"),
            ("qualification_private_icp_sets", "set_id"),
        )
    )


def test_recovery_preserves_frozen_sources_failures_events_and_foreign_rows(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            _seed(cursor, foreign=True)
            sql = _render_sql(cursor)
            before = _state(cursor)
            cursor.execute(sql)
            after = _state(cursor)
            assert after[-1] == before[-1]  # Frozen ICP row.
            assert next(r for r in after[0] if r["round_id"] == "arena-2026-10-02") == next(
                r for r in before[0] if r["round_id"] == "arena-2026-10-02")
            assert next(s for s in after[1] if s["submission_id"] == "sub-foreign368") == next(
                s for s in before[1] if s["submission_id"] == "sub-foreign368")
            assert next(r for r in after[2] if r["run_id"] == "foreign368:1") == next(
                r for r in before[2] if r["run_id"] == "foreign368:1")
            assert next(e for e in after[3] if e["run_id"] == "foreign368:1") == next(
                e for e in before[3] if e["run_id"] == "foreign368:1")
            assert after[4] == before[4] == []  # No provider spend.
            active = next(r for r in after[0] if r["round_id"] == ROUND)
            assert (active["status"], active["status_generation"],
                    active["stage_generation"], active["cancel_reason"]) == (
                        "stage1", 4, 3, None)
            assert active["configuration_doc"]["schedule"] == NEW_SCHEDULE
            assert active["participants"] == before[0][0]["participants"]
            assert [s for s in after[1] if s["round_id"] == ROUND] == [
                s for s in before[1] if s["round_id"] == ROUND
            ]
            old_runs = [r for r in after[2] if r["round_id"] == ARCHIVE]
            assert len(old_runs) == 20
            assert [(r["run_id"], r["status"], r["terminal_cause"],
                     r["terminal_doc"]) for r in old_runs] == [
                         (r["run_id"], r["status"], r["terminal_cause"],
                          r["terminal_doc"]) for r in before[2] if r["round_id"] == ROUND
                     ]
            assert len([e for e in after[3] if e["round_id"] == ARCHIVE]) == 60
            fresh = [r for r in after[2] if r["round_id"] == ROUND]
            assert len(fresh) == 10
            assert {r["icp_position"] for r in fresh} == set(range(10))
            assert all(r["status"] == "pending" and r["stage_generation"] == 3
                       for r in fresh)
            state_once = _state(cursor)
            cursor.execute(sql)
            assert _state(cursor) == state_once


def test_replay_rejects_mutated_archive_event(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            _seed(cursor)
            sql = _render_sql(cursor)
            cursor.execute(sql)
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_trajectory_events SET content='{}'::jsonb "
                "WHERE trajectory_id=(SELECT min(trajectory_id) "
                "FROM public.lab_arena_trajectory_events WHERE round_id=%s)",
                (ARCHIVE,),
            )
            cursor.execute("SET session_replication_role=origin")
            conn.commit()
            before = _state(cursor)
            with pytest.raises(psycopg.Error, match="replay differs"):
                cursor.execute(sql)
            cursor.execute("ROLLBACK")
            assert _state(cursor) == before


@pytest.mark.parametrize("tamper", [
    "round", "submission", "run", "event", "bank", "extra_run", "ledger",
])
def test_preimage_tamper_aborts_without_writes(database, tamper):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            _seed(cursor)
            sql = _render_sql(cursor)
            cursor.execute("SET session_replication_role=replica")
            if tamper == "round":
                cursor.execute("UPDATE public.lab_arena_rounds SET champion_hotkey=%s WHERE round_id=%s",
                               (hotkey("different-champion"), ROUND))
            elif tamper == "submission":
                cursor.execute("UPDATE public.lab_arena_submissions SET source_size_bytes=1 "
                               "WHERE submission_id=%s", (BASELINE,))
            elif tamper == "run":
                cursor.execute("UPDATE public.lab_arena_runs SET terminal_doc='{}'::jsonb "
                               "WHERE run_id=%s", (f"{ROUND}:{BASELINE}:1:0:1",))
            elif tamper == "event":
                cursor.execute("UPDATE public.lab_arena_trajectory_events SET content='{}'::jsonb "
                               "WHERE trajectory_id=(SELECT min(trajectory_id) "
                               "FROM public.lab_arena_trajectory_events WHERE round_id=%s)", (ROUND,))
            elif tamper == "bank":
                cursor.execute("UPDATE public.qualification_private_icp_sets SET icps='[]'::jsonb "
                               "WHERE set_id=20260930")
            elif tamper == "extra_run":
                cursor.execute("INSERT INTO public.lab_arena_runs "
                               "(run_id,assignment_id,round_id,submission_id,miner_hotkey,"
                               "stage,icp_position,attempt,status) VALUES "
                               "('extra368:1','extra368',%s,%s,%s,1,0,1,'pending')",
                               (ROUND, BASELINE, _configuration()["baseline_hotkey"]))
            else:
                cursor.execute("INSERT INTO public.lab_arena_ledger "
                               "(entry_kind,miner_hotkey,round_id,submission_id,run_id,"
                               "amount_microusd) VALUES ('dispatch',%s,%s,%s,%s,1)",
                               (_configuration()["baseline_hotkey"], ROUND,
                                BASELINE, f"{ROUND}:{BASELINE}:1:0:1"))
            cursor.execute("SET session_replication_role=origin")
            conn.commit()
            before = _state(cursor)
            with pytest.raises(psycopg.Error, match="preimage|bank"):
                cursor.execute(sql)
            cursor.execute("ROLLBACK")
            assert _state(cursor) == before


def test_recovered_attempts_close_stage_without_historical_failures(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            _seed(cursor)
            cursor.execute(_render_sql(cursor))
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='accepted',"
                "terminal_cause='accepted',output_ref='arena/test/accepted.json',"
                "result_doc='{\"terminal_status\":\"accepted\"}'::jsonb "
                "WHERE round_id=%s AND stage=1 AND kind='execute'",
                (ROUND,),
            )
            assert cursor.rowcount == 10
            cursor.execute("SET session_replication_role=origin")
            cursor.execute("SELECT public.lab_arena_close_stage(%s,1::smallint)", (ROUND,))
            result = cursor.fetchone()[0]
            assert result["status"] == "closed"
            assert result["incomplete_assignments"] == 0
            cursor.execute("SELECT status FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,))
            assert cursor.fetchone()[0] == "stage1_closed"
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs "
                           "WHERE round_id=%s AND status='failed'", (ROUND,))
            assert cursor.fetchone()[0] == 0
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs "
                           "WHERE round_id=%s AND status='failed'", (ARCHIVE,))
            assert cursor.fetchone()[0] == 20
