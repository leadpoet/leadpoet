"""The one-time margin correction cannot change committed Arena history."""

from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_contracts import base_round_configuration
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey


SQL = Path(__file__).resolve().parents[2] / "scripts/682-lab-arena-open-promotion-margin-five.sql"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _configuration(round_id, *, margin=8, mode="live", network="finney", cutoff="2099-10-24T12:00:00Z"):
    config = base_round_configuration()
    config.update(
        round_id=round_id,
        mode=mode,
        rewards_enabled=mode == "live",
        network_name=network,
        netuid=71,
        promotion_margin=margin,
    )
    config["schedule"] = dict(config["schedule"], submission_cutoff=cutoff)
    return config


def _rounds(cursor):
    cursor.execute("SELECT round_id,to_jsonb(r) FROM public.lab_arena_rounds r ORDER BY round_id")
    return dict(cursor.fetchall())


def _trigger_state(cursor):
    cursor.execute(
        "SELECT tgname,tgenabled FROM pg_trigger "
        "WHERE tgrelid='public.lab_arena_rounds'::regclass AND NOT tgisinternal "
        "ORDER BY tgname"
    )
    return cursor.fetchall()


def test_open_live_round_margin_changes_without_touching_history_or_submission(database):
    psycopg2, dsn = database
    cases = {
        "eligible": ("open", {}),
        "eligibleempty": ("open", {"participants": []}),
        "committed": ("committed", {}),
        "published": ("published", {"publication_doc": {"final_ranking": []}}),
        "cutoff": ("open", {"cutoff": "2000-01-01T00:00:00Z"}),
        "shadow": ("open", {"mode": "shadow"}),
        "testnet": ("open", {"network": "testnet"}),
        "participants": ("open", {"participants": [{"submission_id": "already-selected"}]}),
        "execution": ("open", {}),
        "frozen": ("open", {}),
        "bankready": ("open", {"benchmark_ref": "benchmarks/already-prepared.json"}),
        "reward": ("open", {"reward_basis_hash": "sha256:" + "a" * 64}),
        "rewardactive": ("open", {"reward_activated_at": "2026-11-10T00:00:00Z"}),
        "promoted": ("open", {"baseline_promoted_at": "2026-11-10T00:00:00Z"}),
        "promotion": ("open", {"promotion_doc": {"started": True}}),
        "cancelled": ("cancelled", {}),
        "scored": ("scored", {}),
        "wrongnetuid": ("open", {"netuid": 72}),
        "alreadyfive": ("open", {"margin": 5}),
        "legacyhalf": ("open", {"margin": 0.5}),
        "statusgen": ("open", {"status_generation": 1}),
        "stagegen": ("open", {"stage_generation": 1}),
        "stageoneplan": ("open", {"stage1_scoring_plan_doc": {"frozen": True}}),
        "stagetwoplan": ("open", {"stage2_scoring_plan_doc": {"frozen": True}}),
        "finalists": ("open", {"finalists": []}),
        "pastfrozen": ("open", {}),
        "baseline": ("open", {}),
        "scorejob": ("open", {}),
        "custom": ("open", {"margin": 1.25}),
    }
    ids = {name: "arena-2026-11-10-" + name for name in cases}
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role=replica")
        for name, (status, options) in cases.items():
            config = _configuration(
                ids[name],
                margin=options.get("margin", 8),
                mode=options.get("mode", "live"),
                network=options.get("network", "finney"),
                cutoff=options.get("cutoff", "2099-10-24T12:00:00Z"),
            )
            config["netuid"] = options.get("netuid", 71)
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds "
                "(round_id,status,configuration_doc,rewards_enabled,participants,publication_doc,reward_basis_hash) "
                "VALUES (%s,%s,%s::jsonb,%s,%s::jsonb,%s::jsonb,%s)",
                (
                    ids[name], status, json.dumps(config), config["rewards_enabled"],
                    (json.dumps(options["participants"])
                     if "participants" in options else None),
                    (json.dumps(options["publication_doc"])
                     if "publication_doc" in options else None),
                    options.get("reward_basis_hash"),
                ),
            )
            if "reward_activated_at" in options:
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET reward_activated_at=%s WHERE round_id=%s",
                    (options["reward_activated_at"], ids[name]),
                )
            if "baseline_promoted_at" in options:
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET baseline_promoted_at=%s WHERE round_id=%s",
                    (options["baseline_promoted_at"], ids[name]),
                )
            if "promotion_doc" in options:
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET promotion_doc=%s::jsonb WHERE round_id=%s",
                    (json.dumps(options["promotion_doc"]), ids[name]),
                )
            if "benchmark_ref" in options:
                cursor.execute(
                    "UPDATE public.lab_arena_rounds SET benchmark_ref=%s WHERE round_id=%s",
                    (options["benchmark_ref"], ids[name]),
                )
            for column in ("status_generation", "stage_generation", "stage1_scoring_plan_doc", "stage2_scoring_plan_doc", "finalists"):
                if column in options:
                    from psycopg2 import sql
                    value=options[column]
                    if not isinstance(value,int):value=json.dumps(value)
                    cursor.execute(sql.SQL("UPDATE public.lab_arena_rounds SET {}=%s WHERE round_id=%s").format(sql.Identifier(column)),(value,ids[name]))
        for name in ("eligible", "execution", "scorejob", "pastfrozen", "baseline"):
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions "
                "(submission_id,round_id,miner_hotkey,status,submission_doc) "
                "VALUES (%s,%s,%s,'accepted','{\"preserve\":true}'::jsonb)",
                ("margin-" + name, ids[name], hotkey("margin-" + name)),
            )
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions "
            "(submission_id,round_id,miner_hotkey,status) VALUES (%s,%s,%s,'frozen')",
            ("margin-frozen", ids["frozen"], hotkey("margin-frozen")),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_runs "
            "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,status) "
            "VALUES (%s,%s,%s,%s,%s,1,0,1,'pending')",
            ("margin-run", "margin-assignment", ids["execution"], "margin-execution", hotkey("margin-execution")),
        )
        cursor.execute("INSERT INTO public.lab_arena_runs "
            "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,status,kind) "
            "VALUES ('margin-score','margin-scorejob',%s,'margin-scorejob',%s,1,0,1,'pending','score')",
            (ids["scorejob"],hotkey("margin-scorejob")))
        cursor.execute("UPDATE public.lab_arena_submissions SET frozen_at=clock_timestamp() WHERE round_id=%s",(ids["pastfrozen"],))
        cursor.execute("UPDATE public.lab_arena_submissions SET is_king=true WHERE round_id=%s",(ids["baseline"],))
        cursor.execute("UPDATE public.lab_arena_submissions SET source_ref='arena/arena-2026-11-10-eligible/sources/accepted682.tar.gz',source_size_bytes=123 WHERE round_id=%s",(ids["eligible"],))
        cursor.execute("SET LOCAL session_replication_role=origin")
        cursor.execute("SELECT to_jsonb(s) FROM public.lab_arena_submissions s ORDER BY submission_id")
        all_submissions=cursor.fetchall()
        cursor.execute("SELECT to_jsonb(a) FROM public.lab_arena_runs a ORDER BY run_id")
        all_runs=cursor.fetchall()
        before = _rounds(cursor)
        trigger_before = _trigger_state(cursor)
        assert before[ids["eligible"]]["participants"] is None
        assert before[ids["eligible"]]["benchmark_ref"] is None
        assert before[ids["eligible"]]["promotion_doc"] is None
        assert before[ids["eligible"]]["reward_basis_doc"] is None
        assert before[ids["eligible"]]["promotion_required"] is True
        cursor.execute("SELECT configuration_doc -> 'promotion_margin' = '8'::jsonb, "
                       "(configuration_doc #>> '{schedule,submission_cutoff}')::timestamptz > clock_timestamp() "
                       "FROM public.lab_arena_rounds WHERE round_id=%s", (ids["eligible"],))
        assert cursor.fetchone() == (True, True)
        cursor.execute(
            "SELECT to_jsonb(s) FROM public.lab_arena_submissions s "
            "WHERE round_id=%s", (ids["eligible"],)
        )
        accepted_before = cursor.fetchone()[0]
        cursor.execute("SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE round_id=%s", (ids["execution"],))
        execution_before = cursor.fetchone()[0]

        for _ in range(2):
            cursor.execute(SQL.read_text(encoding="utf-8"))
            after = _rounds(cursor)
            expected = copy.deepcopy(before)
            for eligible in ("eligible", "eligibleempty"):
                expected[ids[eligible]]["configuration_doc"]["promotion_margin"] = 5
                # The database may refresh updated_at on authorized rows.
                after[ids[eligible]]["updated_at"] = expected[ids[eligible]]["updated_at"]
            assert after == expected
            assert _trigger_state(cursor) == trigger_before
            cursor.execute("SELECT to_jsonb(s) FROM public.lab_arena_submissions s ORDER BY submission_id")
            assert cursor.fetchall()==all_submissions
            cursor.execute("SELECT to_jsonb(a) FROM public.lab_arena_runs a ORDER BY run_id")
            assert cursor.fetchall()==all_runs
            cursor.execute(
                "SELECT to_jsonb(s) FROM public.lab_arena_submissions s "
                "WHERE round_id=%s", (ids["eligible"],)
            )
            assert cursor.fetchone()[0] == accepted_before
            cursor.execute("SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE round_id=%s", (ids["execution"],))
            assert cursor.fetchone()[0] == execution_before


def _insert_open(cursor, round_id, *, cutoff="2099-11-10T12:00:00Z"):
    configuration=_configuration(round_id,cutoff=cutoff)
    cursor.execute("INSERT INTO public.lab_arena_rounds(round_id,status,configuration_doc,rewards_enabled) "
                   "VALUES (%s,'open',%s::jsonb,true)",(round_id,json.dumps(configuration)))


@pytest.mark.parametrize('mutation',['committed','frozen','run','deadline'])
def test_cutover_waits_for_existing_round_owner_and_checks_committed_state(database,mutation):
    pg,dsn=database
    rid='arena-2026-11-10-race'+mutation
    observer=pg.connect(**dsn);observer.autocommit=True
    blocker=pg.connect(**dsn)
    application_name='arena682-'+mutation
    def apply():
        with pg.connect(**dsn,application_name=application_name) as connection:
            with connection.cursor() as cur:cur.execute(SQL.read_text())
    try:
        with observer.cursor() as cur:
            _insert_open(cur,rid)
            cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status) "
                        "VALUES (%s,%s,%s,'accepted')",('sub682-'+mutation,rid,hotkey('race682-'+mutation)))
        with blocker.cursor() as cur:
            cur.execute('LOCK TABLE public.lab_arena_rounds IN ROW EXCLUSIVE MODE')
            cur.execute('SELECT round_id FROM public.lab_arena_rounds WHERE round_id=%s FOR UPDATE',(rid,))
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(apply)
            deadline=time.monotonic()+3
            while True:
                with observer.cursor() as cur:
                    cur.execute("SELECT count(*) FROM pg_stat_activity WHERE application_name=%s AND wait_event_type='Lock'",(application_name,))
                    waiting=cur.fetchone()[0]
                if waiting:break
                if time.monotonic()>deadline:
                    blocker.rollback()
                    pytest.fail('cutover did not wait for the existing round owner')
                time.sleep(.02)
            with blocker.cursor() as cur:
                if mutation=='committed':cur.execute("UPDATE public.lab_arena_rounds SET status='committed' WHERE round_id=%s",(rid,))
                elif mutation=='frozen':
                    # Fixture-only frozen state; admission review is unrelated.
                    cur.execute('SET LOCAL session_replication_role=replica')
                    cur.execute("UPDATE public.lab_arena_submissions SET status='frozen',frozen_at=clock_timestamp() WHERE round_id=%s",(rid,))
                elif mutation=='run':
                    cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,status) "
                                "VALUES ('run682-race','assignment682-race',%s,%s,%s,1,0,1,'pending')",
                                (rid,'sub682-'+mutation,hotkey('race682-'+mutation)))
                else:
                    cur.execute('SET LOCAL session_replication_role=replica')
                    cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,submission_cutoff}', "
                                "to_jsonb((clock_timestamp()-interval '1 second')::text)) WHERE round_id=%s",(rid,))
                cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s',(rid,))
                expected=cur.fetchone()[0]
            blocker.commit()
            future.result(timeout=5)
        with observer.cursor() as cur:
            cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s',(rid,))
            assert cur.fetchone()[0]==expected
    finally:
        blocker.rollback();blocker.close();observer.close()


def test_cutover_fails_closed_without_the_exact_enabled_guard(database):
    pg,dsn=database
    c=pg.connect(**dsn);c.autocommit=True
    try:
        with c.cursor() as cur:
            rid='arena-2026-11-10-disabled'
            _insert_open(cur,rid)
            before=_rounds(cur)
            cur.execute('ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once')
            try:
                with pytest.raises(Exception,match='write-once trigger must be enabled'):cur.execute(SQL.read_text())
                cur.execute('ROLLBACK')
                assert _rounds(cur)==before
            finally:
                cur.execute('ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once')
    finally:c.close()


def test_cutover_fails_promptly_on_row_owner_without_deadlock_then_retries(database):
    pg,dsn=database
    observer=pg.connect(**dsn);observer.autocommit=True
    blocker=pg.connect(**dsn)
    rid='arena-2026-11-10-rowowner'
    try:
        with observer.cursor() as cur:
            _insert_open(cur,rid)
            before=_rounds(cur);triggers=_trigger_state(cur)
        with blocker.cursor() as cur:
            cur.execute('SELECT round_id FROM public.lab_arena_rounds WHERE round_id=%s FOR UPDATE',(rid,))
        with observer.cursor() as cur:
            with pytest.raises(pg.errors.LockNotAvailable,match='could not obtain lock on row'):
                cur.execute(SQL.read_text())
            cur.execute('ROLLBACK')
            assert _rounds(cur)==before
            assert _trigger_state(cur)==triggers
        # The existing owner can still write, and a later canonical retry can
        # apply only the now-unlocked eligible round.
        with blocker.cursor() as cur:
            cur.execute('UPDATE public.lab_arena_rounds SET updated_at=clock_timestamp() WHERE round_id=%s',(rid,))
        blocker.commit()
        with observer.cursor() as cur:
            cur.execute(SQL.read_text())
            cur.execute("SELECT configuration_doc->'promotion_margin' FROM public.lab_arena_rounds WHERE round_id=%s",(rid,))
            assert cur.fetchone()[0]==5
            assert _trigger_state(cur)==triggers
    finally:blocker.rollback();blocker.close();observer.close()
