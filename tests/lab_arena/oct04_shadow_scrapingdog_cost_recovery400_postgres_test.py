"""Exact, append-only repair of four observed Oct 4 shadow charges."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey


MIGRATION = Path(__file__).parents[2] / "scripts/400-arena-2026-10-04-shadow-scrapingdog-cost-recovery.sql"
TARGETS = (
    ("unproven3f7051d", "c50ae556daa81714d581c98bd499b1a3e9439d6f7fb9b207daf4412990f37e65", 1618548),
    ("unproven3f7051d", "b13d8eb23e96d3f35f7f8441281f347ad295fd04b714784967f7b0659e03ea3c", 1618554),
    ("costprobe3f7051d", "a60ec5332268b7332d494235df83925f770cd037e2129a34f0d6195703c6afc8", 1618824),
    ("costprobe3f7051d", "b3182d81d5cc65a3d17096802af8c2b865f2ed8f3daa72819f8bc68bb13fe592", 1618830),
)
PROVIDER_COST = {
    "basis": "scrapingdog_legacy_endpoint_map",
    "units": "5",
    "unit_name": "credits",
    "operation": "scrapingdog.scrape",
}


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _seed(cursor, *, published: bool = True, rewards_enabled: bool = False):
    miner = hotkey("oct04-shadow-cost-miner")
    for suffix, count in (("unproven3f7051d", 3), ("costprobe3f7051d", 2)):
        round_id = f"arena-2026-10-04-{suffix}"
        submission_id = f"baseline-2026-10-04-{suffix}"
        configuration = {"round_id": round_id, "mode": "shadow", "rewards_enabled": False}
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds "
            "(round_id,status,configuration_doc,rewards_enabled,publication_doc,published_at) "
            "VALUES (%s,%s,%s::jsonb,%s,%s::jsonb,now())",
            (round_id, "published" if published else "scored", json.dumps(configuration),
             rewards_enabled, json.dumps({"round_id": round_id})),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions "
            "(submission_id,round_id,miner_hotkey,status) VALUES (%s,%s,%s,'accepted')",
            (submission_id, round_id, miner),
        )
        for position in range(count):
            for kind in ("execute", "score"):
                assignment_id = f"{round_id}:{submission_id}:1:{position}:{kind}"
                run_id = f"{assignment_id}:1"
                cursor.execute(
                    "INSERT INTO public.lab_arena_runs "
                    "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                    "icp_position,attempt,kind,status,terminal_cause) "
                    "VALUES (%s,%s,%s,%s,%s,1,%s,1,%s,'accepted','accepted')",
                    (run_id, assignment_id, round_id, submission_id, miner, position, kind),
                )

    for suffix, call_hash, reservation_id in TARGETS:
        round_id = f"arena-2026-10-04-{suffix}"
        submission_id = f"baseline-2026-10-04-{suffix}"
        run_id = f"{round_id}:{submission_id}:1:0:score:1"
        call_identity = f"sha256:{call_hash}"
        for offset, kind in enumerate(("reservation", "dispatch", "uncertain")):
            doc = (
                {"reason": "worker_reported", "call": {
                    "reason": "settle_failure", "call_succeeded": True,
                    "failure_stage": "settlement", "error_class": "ArenaStoreError",
                    "known_actual_microusd": 250, "provider_status": 200,
                    "provider_cost": PROVIDER_COST,
                }} if kind == "uncertain" else {}
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger "
                "(entry_id,entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) "
                "VALUES (%s,%s,%s,%s,%s,%s,1,%s,'scrapingdog','scrapingdog.scrape',"
                "'host',0,%s::jsonb)",
                (reservation_id + offset, kind, miner, round_id, submission_id,
                 run_id, call_identity, json.dumps(doc)),
            )
    cursor.execute("SELECT setval(pg_get_serial_sequence('public.lab_arena_ledger',"
                   "'entry_id'), (SELECT max(entry_id) FROM public.lab_arena_ledger))")


def _apply(cursor):
    cursor.execute(MIGRATION.read_text(encoding="utf-8"))


def _costs(cursor):
    costs = {}
    for suffix in ("unproven3f7051d", "costprobe3f7051d"):
        submission_id = f"baseline-2026-10-04-{suffix}"
        cursor.execute(
            "SELECT public.lab_arena__successful_call_cost_state(%s,'score',NULL), "
            "public.lab_arena__successful_call_cost_state(%s,'execute',NULL)",
            (submission_id, submission_id),
        )
        costs[suffix] = cursor.fetchone()
    return costs


def test_exact_repair_and_replay_preserve_published_results(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            _seed(cursor)
            # A fifth uncertain Scrapingdog call in the same round stays open.
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger "
                "(entry_id,entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc) "
                "SELECT 1618840, 'uncertain',miner_hotkey,round_id,submission_id,run_id,stage,"
                "'sha256:' || repeat('f',64),provider,operation_id,funding_source,0,entry_doc "
                "FROM public.lab_arena_ledger WHERE entry_id=1618832"
            )
            cursor.execute("SELECT setval(pg_get_serial_sequence('public.lab_arena_ledger',"
                           "'entry_id'), 1618840)")
            cursor.execute(
                "SELECT round_id,status,configuration_doc,publication_doc,published_at "
                "FROM public.lab_arena_rounds ORDER BY round_id"
            )
            rounds_before = cursor.fetchall()
            cursor.execute(
                "SELECT run_id,status,terminal_cause,result_doc,per_icp_score "
                "FROM public.lab_arena_runs ORDER BY run_id"
            )
            runs_before = cursor.fetchall()
            costs_before = _costs(cursor)
            _apply(cursor)
            _apply(cursor)
            costs_after = _costs(cursor)
            assert sum(pair[0]["settled_microusd"] for pair in costs_after.values()) - sum(
                pair[0]["settled_microusd"] for pair in costs_before.values()
            ) == 1000
            assert all(costs_after[suffix][1] == costs_before[suffix][1]
                       for suffix in costs_before)
            assert all(costs_after[suffix][0]["successful_microusd"] ==
                       costs_before[suffix][0]["successful_microusd"]
                       for suffix in costs_before)
            cursor.execute(
                "SELECT call_identity,entry_kind,amount_microusd,entry_doc,terminal_response "
                "FROM public.lab_arena_ledger WHERE entry_kind='settlement' ORDER BY call_identity"
            )
            rows = cursor.fetchall()
            assert len(rows) == 4
            assert all(row[2] == 250 for row in rows)
            assert all(row[3]["shadow_scrapingdog_cost_recovery"] is True for row in rows)
            assert all(row[4]["call_succeeded"] is False for row in rows)
            assert all(row[4]["status"] == 502 for row in rows)
            assert all(row[4]["provider_cost"] == PROVIDER_COST for row in rows)
            cursor.execute(
                "SELECT round_id,status,configuration_doc,publication_doc,published_at "
                "FROM public.lab_arena_rounds ORDER BY round_id"
            )
            assert cursor.fetchall() == rounds_before
            cursor.execute(
                "SELECT run_id,status,terminal_cause,result_doc,per_icp_score "
                "FROM public.lab_arena_runs ORDER BY run_id"
            )
            assert cursor.fetchall() == runs_before
            cursor.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE entry_kind='uncertain'")
            assert cursor.fetchone()[0] == 5
            cursor.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE call_identity="
                           "'sha256:' || repeat('f',64)")
            assert cursor.fetchone()[0] == 1
            cursor.execute(
                "SELECT count(*) FROM (SELECT DISTINCT ON (call_identity) entry_kind "
                "FROM public.lab_arena_ledger WHERE call_identity = ANY(%s) "
                "ORDER BY call_identity,entry_id DESC) AS heads "
                "WHERE entry_kind='settlement'",
                ([f"sha256:{call_hash}" for _, call_hash, _ in TARGETS],),
            )
            assert cursor.fetchone()[0] == 4


@pytest.mark.parametrize("change", (
    "unpublished", "rewards", "cost", "foreign_head", "inflight",
    "wrong_namespace", "live_mode", "active_attempt",
))
def test_drift_aborts_all_four_without_partial_settlement(database, change):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            _seed(cursor)
            cursor.execute("SET session_replication_role = replica")
            if change == "unpublished":
                cursor.execute("UPDATE public.lab_arena_rounds SET status='scored' "
                               "WHERE round_id='arena-2026-10-04-costprobe3f7051d'")
            elif change == "rewards":
                cursor.execute("UPDATE public.lab_arena_rounds SET rewards_enabled=true "
                               "WHERE round_id='arena-2026-10-04-costprobe3f7051d'")
            elif change == "cost":
                cursor.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set("
                               "entry_doc,'{call,known_actual_microusd}','251'::jsonb) "
                               "WHERE entry_id=1618832")
            elif change == "wrong_namespace":
                cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                               "jsonb_set(configuration_doc,'{round_id}',"
                               "'\"arena-2026-10-04-other\"'::jsonb) "
                               "WHERE round_id='arena-2026-10-04-costprobe3f7051d'")
            elif change == "live_mode":
                cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                               "jsonb_set(configuration_doc,'{mode}','\"live\"'::jsonb) "
                               "WHERE round_id='arena-2026-10-04-costprobe3f7051d'")
            elif change == "active_attempt":
                cursor.execute("UPDATE public.lab_arena_runs SET status='leased' "
                               "WHERE run_id LIKE 'arena-2026-10-04-costprobe3f7051d:%' "
                               "AND kind='execute' AND icp_position=1")
            elif change == "foreign_head":
                cursor.execute("INSERT INTO public.lab_arena_ledger "
                               "(entry_id,entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                               "call_identity,provider,operation_id,funding_source,amount_microusd) "
                               "SELECT 1618833,'settlement',miner_hotkey,round_id,submission_id,run_id,stage,"
                               "call_identity,provider,operation_id,funding_source,0 "
                               "FROM public.lab_arena_ledger WHERE entry_id=1618832")
            else:
                cursor.execute("INSERT INTO public.lab_arena_ledger "
                               "(entry_id,entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                               "call_identity,provider,operation_id,funding_source,amount_microusd) "
                               "SELECT 1618841,'reservation',miner_hotkey,round_id,submission_id,run_id,stage,"
                               "'sha256:' || repeat('e',64),provider,operation_id,funding_source,0 "
                               "FROM public.lab_arena_ledger WHERE entry_id=1618832")
            cursor.execute("SET session_replication_role = origin")
            connection.commit()
            cursor.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE entry_kind='settlement'")
            existing_settlements = cursor.fetchone()[0]
            with pytest.raises(psycopg.Error, match="shadow_scrapingdog_"):
                _apply(cursor)
            cursor.execute("ROLLBACK")
            cursor.execute("SELECT count(*) FROM public.lab_arena_ledger WHERE entry_kind='settlement'")
            assert cursor.fetchone()[0] == existing_settlements
