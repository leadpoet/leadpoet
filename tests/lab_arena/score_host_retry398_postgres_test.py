"""One final score attempt only for a lease-bound, zero-call host failure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.lab_arena import execute_host_cooldown396_postgres_test as host396
from tests.lab_arena import zero_setup_runner_handoff360_postgres_test as prior
from lab_arena.service import ArenaService


ROOT = Path(__file__).parents[2]
SQL390 = ROOT / "scripts/390-lab-arena-expired-retry-runner-handoff.sql"
SQL398 = ROOT / "scripts/398-lab-arena-zero-call-score-host-retry.sql"
LIVE_390_PREIMAGE = host396.LIVE_390_PREIMAGE
LIVE_398_PREIMAGE = "6db3ae0ffd11b21c867930e6df271bb8eef16aa56c3d52feae6a0190a89151b5"
EXPIRY = "public.lab_arena_expire_leases(text)"
database = host396.database


def _hash(cursor, signature):
    cursor.execute(
        "SELECT encode(extensions.digest(pg_get_functiondef(%s::regprocedure),"
        "'sha256'),'hex')", (signature,)
    )
    return cursor.fetchone()[0]


def _security(cursor):
    cursor.execute(
        "SELECT owner.rolname,p.proacl,p.prosecdef,p.provolatile,p.proconfig "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "JOIN pg_roles owner ON owner.oid=p.proowner "
        "WHERE n.nspname='public' AND p.proname='lab_arena_expire_leases' "
        "AND p.pronargs=1"
    )
    return cursor.fetchone()


@pytest.fixture(scope="module")
def migrated(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            claim_hash = _hash(cursor, host396.SIGNATURE)
            sql390 = SQL390.read_text().replace(LIVE_390_PREIMAGE, claim_hash)
            cursor.execute(sql390)
            before = _security(cursor)
            expiry_hash = _hash(cursor, EXPIRY)
            sql398 = SQL398.read_text()
            assert LIVE_398_PREIMAGE in sql398
            sql398 = sql398.replace(LIVE_398_PREIMAGE, expiry_hash)
            cursor.execute("BEGIN")
            with pytest.raises(psycopg.Error, match="preimage differs"):
                cursor.execute(sql398.replace(expiry_hash, "0" * 64, 1))
            cursor.execute("ROLLBACK")
            assert _hash(cursor, EXPIRY) == expiry_hash
            cursor.execute(sql398)
            applied_hash = _hash(cursor, EXPIRY)
            cursor.execute(sql398)
            assert _hash(cursor, EXPIRY) == applied_hash
            assert _security(cursor) == before
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (EXPIRY,))
            definition = cursor.fetchone()[0]
            assert definition.count("lab_arena_zero_call_score_host_retry_v1") == 1
            assert definition.count("lab_arena_judgment_group_expiry_handoff_v1") == 1
    return True


def _seed(conn, *, kind="score", event=True, ledger=False, result=False,
          output=False, deadline=False, generation=False, attempt=2,
          event_content=None, provider_event=False, unexpired=False,
          wrong_round_status=False, wrong_event_runner=False):
    prior._seed(
        conn, {"terminal_status": "judge_error"}, "judge_error", kind=kind,
        prior_has_ledger=True,
    )
    run_id = prior.ASSIGNMENT + ":2"
    with conn.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET configuration_doc="
            "jsonb_set(configuration_doc,'{schedule,stage_1_scoring_close}',"
            "to_jsonb((now()+interval '2 hours')::text)) WHERE round_id=%s",
            (prior.ROUND,),
        )
        if deadline:
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET configuration_doc="
                "jsonb_set(configuration_doc,'{schedule,stage_1_scoring_close}',"
                "to_jsonb((now()-interval '1 second')::text)) WHERE round_id=%s",
                (prior.ROUND,),
            )
        if generation:
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET stage_generation=2 "
                "WHERE round_id=%s", (prior.ROUND,),
            )
        if wrong_round_status:
            cursor.execute("UPDATE public.lab_arena_rounds SET status='stage1' "
                           "WHERE round_id=%s", (prior.ROUND,))
        if attempt != 2:
            cursor.execute("UPDATE public.lab_arena_runs SET attempt=%s,"
                           "run_id=%s WHERE run_id=%s", (attempt,
                           prior.ASSIGNMENT + ":" + str(attempt), run_id))
            run_id = prior.ASSIGNMENT + ":" + str(attempt)
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='leased',"
            "runner_hotkey=%s,lease_token_hash=%s,"
            "lease_expires_at=now()+(%s*interval '1 minute'),"
            "result_doc=%s::jsonb,output_ref=%s WHERE run_id=%s",
            (prior.RUNNER_A, "sha256:" + "a" * 64,
             5 if unexpired else -1,
             json.dumps({"terminal_status": "judge_error"}) if result else None,
             "arena/test/output.json" if output else None, run_id),
        )
        if ledger:
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger("
                "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,amount_microusd)"
                "VALUES ('reservation',%s,%s,%s,%s,1,%s,'deepline',"
                "'deepline.execute','host',1)",
                (prior.MINER, prior.ROUND, prior.SUBMISSION, run_id,
                 "sha256:" + "c" * 64),
            )
        if event:
            host396._event(cursor, run_id, content=event_content)
            if wrong_event_runner:
                cursor.execute(
                    "UPDATE public.lab_arena_trajectory_events "
                    "SET runner_hotkey=%s WHERE run_id=%s",
                    (prior.RUNNER_B, run_id),
                )
        if provider_event:
            cursor.execute(
                "INSERT INTO public.lab_arena_trajectory_events("
                "run_id,event_id,round_id,submission_id,miner_hotkey,"
                "runner_hotkey,assignment_id,icp_identifier,stage,icp_position,"
                "attempt,run_kind,model_role,event_kind,occurred_at,content) "
                "SELECT r.run_id,gen_random_uuid(),r.round_id,r.submission_id,"
                "r.miner_hotkey,r.runner_hotkey,r.assignment_id,"
                "r.round_id||':icp:'||r.icp_position,r.stage,r.icp_position,"
                "r.attempt,r.kind,'baseline','provider.request',now(),'{}'::jsonb "
                "FROM public.lab_arena_runs r WHERE r.run_id=%s", (run_id,),
            )
        cursor.execute("SET session_replication_role=origin")
    conn.commit()
    return run_id


def _expire(conn):
    with conn.cursor() as cursor:
        cursor.execute("SELECT public.lab_arena_expire_leases(%s)", (prior.ROUND,))
        result = cursor.fetchone()[0]
    conn.commit()
    return result


def test_zero_call_host_expiry_preserves_paid_history_and_hands_off(database, migrated):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        old_run = _seed(conn)
        assert _expire(conn)["retried"] == 1
        assert _expire(conn)["retried"] == 0
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT attempt,status,terminal_cause,runner_hotkey,"
                "previous_runner_hotkey,result_doc,output_ref FROM "
                "public.lab_arena_runs WHERE assignment_id=%s ORDER BY attempt",
                (prior.ASSIGNMENT,),
            )
            rows = cursor.fetchall()
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_ledger WHERE run_id=%s",
                (prior.ASSIGNMENT + ":1",),
            )
            paid_history_count = cursor.fetchone()[0]
        assert len(rows) == 3
        assert rows[0][1:3] == ("failed", "judge_error")
        assert paid_history_count == 1
        assert rows[1][1:3] == ("failed", "lease_expired")
        assert rows[2][0:2] == (3, "pending")
        assert rows[2][4] == prior.RUNNER_A
        assert rows[2][5:] == (None, None)
        assert prior._claim(conn, prior.RUNNER_A, "a") == {"status": "no_pending"}
        claim = prior._claim(conn, prior.RUNNER_B, "b")
        assert claim["run_id"] == prior.ASSIGNMENT + ":3"


def test_third_attempt_can_be_accepted_and_late_second_cannot_replace_it(
    database, migrated
):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        old_run = _seed(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_runs SET scored_run_id=%s "
                "WHERE assignment_id=%s",
                (prior.ASSIGNMENT + ":execute:1", prior.ASSIGNMENT),
            )
        conn.commit()
        assert _expire(conn)["retried"] == 1
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT public.lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)",
                (old_run, "sha256:" + "a" * 64,
                 json.dumps({"terminal_status": "accepted"}),
                 "accepted", "arena/late.json"),
            )
            late = cursor.fetchone()[0]
        conn.commit()
        assert late["status"] == "failed"
        claim = prior._claim(conn, prior.RUNNER_B, "b")
        assert claim["run_id"] == prior.ASSIGNMENT + ":3"
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT public.lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)",
                (claim["run_id"], "sha256:" + "b" * 64,
                 json.dumps({"terminal_status": "accepted"}),
                 "accepted", "arena/score-accepted.json"),
            )
            accepted = cursor.fetchone()[0]
            cursor.execute("SELECT * FROM public.lab_arena_runs "
                           "WHERE assignment_id=%s ORDER BY attempt",
                           (prior.ASSIGNMENT,))
            columns = [column.name for column in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.commit()
        assert accepted["status"] == "accepted"
        chosen = ArenaService._select_scoring_outputs(rows)
        assert chosen[prior.ASSIGNMENT + ":execute:1"]["run_id"] == claim["run_id"]
        with conn.cursor() as cursor:
            cursor.execute("SELECT public.lab_arena_close_scoring(%s,1::smallint)",
                           (prior.ROUND,))
            closed = cursor.fetchone()[0]
        conn.commit()
        assert closed["status"] == "closed"
        assert closed["incomplete_assignments"] == 0
        assert [row["status"] for row in rows] == ["failed", "failed", "accepted"]


def test_third_expiry_is_terminal_and_hands_one_group_follower_off(
    database, migrated
):
    psycopg, dsn = database
    cache_key = "sha256:" + "d" * 64
    with psycopg.connect(**dsn) as conn:
        _seed(conn)
        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET judgment_cache_key=%s,"
                "judgment_input_hash=%s,judgment_scope_doc=%s::jsonb,"
                "judgment_group_leader=TRUE,"
                "judgment_group_miner_hotkeys=ARRAY[%s]::text[] "
                "WHERE assignment_id=%s",
                (cache_key, "sha256:" + "e" * 64,
                 json.dumps({"cache_key": cache_key}), prior.MINER,
                 prior.ASSIGNMENT),
            )
            follower_submission = "follower-398"
            follower_miner = "5" + "D" * 47
            follower_source = f"arena/{prior.ROUND}/sources/{follower_submission}.tar.gz"
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions("
                "submission_id,round_id,miner_hotkey,status,is_king,"
                "submission_doc,source_ref,source_size_bytes,consent,frozen_at) "
                "SELECT %s,round_id,%s,status,FALSE,"
                "jsonb_set(submission_doc,'{source_ref}',to_jsonb(%s::text)),"
                "%s,source_size_bytes,consent,frozen_at "
                "FROM public.lab_arena_submissions WHERE submission_id=%s",
                (follower_submission, follower_miner, follower_source,
                 follower_source, prior.SUBMISSION),
            )
            follower_assignment = f"{prior.ROUND}:{follower_submission}:1:0:score"
            cursor.execute(
                "INSERT INTO public.lab_arena_runs("
                "run_id,assignment_id,round_id,submission_id,miner_hotkey,"
                "stage,icp_position,attempt,kind,status,stage_generation,"
                "judgment_cache_key,judgment_input_hash,judgment_scope_doc,"
                "judgment_group_leader,judgment_group_miner_hotkeys) VALUES "
                "(%s,%s,%s,%s,%s,1,0,1,'score','pending',1,%s,%s,%s::jsonb,"
                "FALSE,ARRAY[%s,%s]::text[])",
                (follower_assignment + ":1", follower_assignment,
                 prior.ROUND, follower_submission, follower_miner,
                 cache_key, "sha256:" + "e" * 64,
                 json.dumps({"cache_key": cache_key}),
                 prior.MINER, follower_miner),
            )
            cursor.execute("SET session_replication_role=origin")
        conn.commit()
        assert _expire(conn)["retried"] == 1
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT status,judgment_group_leader FROM public.lab_arena_runs "
                "WHERE run_id=%s", (follower_assignment + ":1",),
            )
            assert cursor.fetchone() == ("pending", False)
        claim = prior._claim(conn, prior.RUNNER_B, "b")
        assert claim["run_id"] == prior.ASSIGNMENT + ":3"
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_runs SET "
                "lease_expires_at=now()-interval '1 second' WHERE run_id=%s",
                (claim["run_id"],),
            )
        conn.commit()
        assert _expire(conn)["retried"] == 0
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT status,judgment_group_leader FROM public.lab_arena_runs "
                "WHERE run_id=%s", (follower_assignment + ":1",),
            )
            assert cursor.fetchone() == ("pending", True)
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs "
                           "WHERE assignment_id=%s", (prior.ASSIGNMENT,))
            assert cursor.fetchone()[0] == 3


@pytest.mark.parametrize("change", [
    "no_event", "wrong_class", "paid_call", "result", "output",
    "provider_event", "deadline", "generation", "execute", "attempt3",
    "unexpired", "wrong_round_status", "wrong_event_runner",
])
def test_only_current_zero_call_second_score_gets_extra_attempt(
    database, migrated, change
):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        kwargs = {}
        if change == "no_event":
            kwargs["event"] = False
        elif change == "wrong_class":
            kwargs["event_content"] = {"status": "abandoned",
                "failure_stage": "runtime", "error_class": "ValueError"}
        elif change == "paid_call":
            kwargs["ledger"] = True
        elif change in {"result", "output", "provider_event", "deadline", "generation",
                        "unexpired", "wrong_round_status", "wrong_event_runner"}:
            kwargs[change] = True
        elif change == "execute":
            kwargs["kind"] = "execute"
        elif change == "attempt3":
            kwargs["attempt"] = 3
        _seed(conn, **kwargs)
        assert _expire(conn)["retried"] == 0
        with conn.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs "
                           "WHERE assignment_id=%s", (prior.ASSIGNMENT,))
            assert cursor.fetchone()[0] == 2
