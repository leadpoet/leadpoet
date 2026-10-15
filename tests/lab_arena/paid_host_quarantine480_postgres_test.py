"""The real claim RPC quarantines repeated proved paid host interruptions."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import uuid

import pytest

from tests.lab_arena import active_host_fault_guard397_postgres_test as faults397
from tests.lab_arena import infrastructure_retry_cost478_postgres_test as prior478
from tests.lab_arena import zero_setup_runner_handoff360_postgres_test as prior
from tests.lab_arena.execute_host_cooldown396_postgres_test import SIGNATURE, _hash, _security


ROOT = Path(__file__).parents[2]
SQL = ROOT / "scripts/480-lab-arena-paid-host-interruption-round-quarantine.sql"
PRE = "23c29c40dd0f91e16035675c13a2e1c64046c81a447c372f50f1e94391983199"
POST = "8ea400f297a8a1da73536098a9e216d1017255d1a963c24cc9129ccb1445736c"


@pytest.fixture(scope="module")
def database():
    yield from prior478.database.__wrapped__()


def _rows(cursor):
    cursor.execute("SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r")
    runs = cursor.fetchone()[0]
    cursor.execute("SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l")
    ledger = cursor.fetchone()[0]
    cursor.execute("SELECT jsonb_agg(to_jsonb(e) ORDER BY event_id) FROM public.lab_arena_trajectory_events e")
    events = cursor.fetchone()[0]
    return runs, ledger, events


@pytest.fixture(scope="module")
def migrated(database):
    pg, dsn = database
    with pg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            fresh, fault_runs = faults397._seed(conn)
            _paid_faults(conn, fault_runs)
            assert _proofs(conn, fault_runs) == [True] * 3
            before_rows = _rows(cursor)
            before_security = _security(cursor)
            assert _hash(cursor) == PRE
            sql = SQL.read_text(encoding="utf-8")
            cursor.execute("BEGIN")
            with pytest.raises(pg.Error, match="preimage differs"):
                cursor.execute(sql.replace(PRE, "0" * 64, 1))
            cursor.execute("ROLLBACK")
            assert _hash(cursor) == PRE and _rows(cursor) == before_rows
            cursor.execute(sql)
            assert _hash(cursor) == POST
            cursor.execute(sql)
            assert _hash(cursor) == POST
            assert _security(cursor) == before_security
            assert _rows(cursor) == before_rows
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (SIGNATURE,))
            definition = cursor.fetchone()[0]
            assert definition.count("public.lab_arena__execute_host_interruption_v1(host_fault)") == 1
            assert definition.count("lab_arena_round_host_fault_quarantine_v1") == 1
    return True


def _paid_faults(conn, runs, *, expired=False, preserve_claim=False):
    with conn.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        if not preserve_claim:
            cursor.execute(
                "UPDATE public.lab_arena_runs SET kind='execute',"
                "lease_expires_at=now()+interval '5 minutes',"
                "lease_token_hash=%s,"
                "claim_request_id=run_id,claim_response=jsonb_build_object('status','leased',"
                "'lease_generation',lease_generation,'lease_expires_at',"
                "to_jsonb(now()+interval '5 minutes')) WHERE run_id=ANY(%s)",
                ("sha256:" + "a" * 64, runs),
            )
        cursor.execute("DELETE FROM public.lab_arena_trajectory_events WHERE run_id=ANY(%s)", (runs,))
        for run in runs:
            for kind, content in (
                ("runtime.started", {"status": "starting", "lease_generation": 1}),
                ("runtime.error", {
                    "status": "abandoned", "failure_stage": "runtime",
                    "error_class": "RuntimeHostError",
                    "runtime_host_reason": "sandbox_launcher_signaled",
                    "launch_exit_code": -2, "launch_timed_out": False,
                }),
            ):
                cursor.execute(
                    "INSERT INTO public.lab_arena_trajectory_events("
                    "run_id,event_id,round_id,submission_id,miner_hotkey,runner_hotkey,"
                    "assignment_id,icp_identifier,stage,icp_position,attempt,run_kind,"
                    "model_role,event_kind,occurred_at,content) "
                    "SELECT r.run_id,%s,r.round_id,r.submission_id,r.miner_hotkey,"
                    "r.runner_hotkey,r.assignment_id,r.round_id||':icp:'||r.icp_position,"
                    "r.stage,r.icp_position,r.attempt,r.kind,'baseline',%s,now(),%s::jsonb "
                    "FROM public.lab_arena_runs r WHERE r.run_id=%s",
                    (str(uuid.uuid4()), kind, json.dumps(content), run),
                )
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,"
                "submission_id,run_id,stage,call_identity,provider,operation_id,"
                "funding_source,amount_microusd) "
                "SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,"
                "%s,'deepline','deepline.execute','host',100 "
                "FROM public.lab_arena_runs WHERE run_id=%s",
                ("sha256:" + hashlib.sha256(run.encode()).hexdigest(), run),
            )
        if expired:
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='failed',"
                "terminal_cause='lease_expired',lease_expires_at=now()-interval '1 second',"
                "claim_response=jsonb_set(claim_response,'{lease_expires_at}',"
                "to_jsonb(now()-interval '1 second')) WHERE run_id=ANY(%s)", (runs,),
            )
            cursor.execute(
                "UPDATE public.lab_arena_trajectory_events SET "
                "created_at=now()-interval '2 seconds',"
                "occurred_at=now()-interval '2 seconds' WHERE run_id=ANY(%s)", (runs,),
            )
        cursor.execute("SET session_replication_role=origin")
    conn.commit()


def _proofs(conn, runs):
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT public.lab_arena__execute_host_interruption_v1(r) "
            "FROM public.lab_arena_runs r WHERE run_id=ANY(%s) ORDER BY run_id", (runs,),
        )
        return [row[0] for row in cursor.fetchall()]


def _run_state(conn, runs):
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) "
            "FROM public.lab_arena_runs r WHERE run_id=ANY(%s)", (runs,),
        )
        return cursor.fetchone()[0]


def _allow_healthy_runner(conn):
    with conn.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET configuration_doc="
            "jsonb_set(configuration_doc,'{runner_hotkeys}',%s::jsonb) "
            "WHERE round_id=%s",
            (json.dumps([prior.RUNNER_A, prior.RUNNER_B]), prior.ROUND),
        )
        cursor.execute("SET session_replication_role=origin")
    conn.commit()


@pytest.mark.parametrize("expired", [False, True])
def test_three_paid_interruptions_block_runner_but_keep_healthy_claim(database, migrated, expired):
    with database[0].connect(**database[1]) as conn:
        fresh, runs = faults397._seed(conn)
        _allow_healthy_runner(conn)
        _paid_faults(conn, runs, expired=expired)
        assert _proofs(conn, runs) == [True] * 3
        before = _run_state(conn, runs)
        assert prior._claim(conn, prior.RUNNER_A, "a") == {"status": "no_pending"}
        assert _run_state(conn, runs) == before
        if expired:
            assert prior._claim(conn, prior.RUNNER_B, "b")["run_id"] == fresh
        # With active leases, the existing model-bank guard separately bars
        # another runner from this submission. The two-fault control below
        # shows that this runner could otherwise claim the pending execute.
        with conn.cursor() as cursor:
            cursor.execute("SELECT count(*),sum(amount_microusd) FROM public.lab_arena_ledger")
            assert cursor.fetchone() == (3, 300)
            cursor.execute("SELECT count(*) FROM public.lab_arena_runs WHERE run_id=ANY(%s)", (runs,))
            assert cursor.fetchone()[0] == 3


def test_prior_successful_claim_replays_after_third_paid_fault(database, migrated):
    with database[0].connect(**database[1]) as conn:
        fresh, runs = faults397._seed(conn, positions=(0, 1))
        _paid_faults(conn, runs)
        response = prior._claim(conn, prior.RUNNER_A, "d")
        assert response["run_id"] == fresh
        _paid_faults(conn, [fresh], preserve_claim=True)
        assert _proofs(conn, runs + [fresh]) == [True] * 3
        before = _run_state(conn, runs + [fresh])
        assert prior._claim(conn, prior.RUNNER_A, "d") == response
        assert _run_state(conn, runs + [fresh]) == before


@pytest.mark.parametrize("positions", [(0, 1), (0, 0, 1)])
def test_fewer_than_three_distinct_assignments_do_not_quarantine(database, migrated, positions):
    with database[0].connect(**database[1]) as conn:
        if len(set(positions)) < len(positions):
            fresh = faults397.score390._cooldown_seed(conn, failure_positions=positions,
                fresh_kind="execute", round_status="stage1")
            with conn.cursor() as cursor:
                cursor.execute("SELECT run_id FROM public.lab_arena_runs "
                               "WHERE status='failed' ORDER BY run_id")
                runs = [row[0] for row in cursor.fetchall()]
        else:
            fresh, runs = faults397._seed(conn, positions=positions)
        _paid_faults(conn, runs, expired=len(set(positions)) < len(positions))
        assert _proofs(conn, runs) == [True] * len(runs)
        assert prior._claim(conn, prior.RUNNER_A, "c")["run_id"] == fresh


@pytest.mark.parametrize("change", [
    "model", "provider", "cleanup", "spoofed_exit", "missing_start",
    "other_runner", "other_round", "result", "output",
])
def test_paid_nonproof_or_other_scope_does_not_quarantine(database, migrated, change):
    with database[0].connect(**database[1]) as conn:
        fresh, runs = faults397._seed(conn)
        _paid_faults(conn, runs)
        run = runs[0]
        with conn.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            if change in ("model", "provider", "spoofed_exit"):
                key, value = {
                    "model": ("error_class", "ModelError"),
                    "provider": ("failure_stage", "provider"),
                    "spoofed_exit": ("launch_exit_code", 1),
                }[change]
                cursor.execute(
                    "UPDATE public.lab_arena_trajectory_events SET "
                    "content=jsonb_set(content,%s::text[],%s::jsonb) "
                    "WHERE run_id=%s AND event_kind='runtime.error'",
                    ([key], json.dumps(value), run),
                )
            elif change == "cleanup":
                cursor.execute(
                    "UPDATE public.lab_arena_trajectory_events SET "
                    "event_kind='runtime.cleanup_error' "
                    "WHERE run_id=%s AND event_kind='runtime.error'", (run,),
                )
            elif change == "missing_start":
                cursor.execute(
                    "DELETE FROM public.lab_arena_trajectory_events "
                    "WHERE run_id=%s AND event_kind='runtime.started'", (run,),
                )
            elif change == "other_runner":
                cursor.execute("UPDATE public.lab_arena_runs SET runner_hotkey=%s WHERE run_id=%s",
                               (prior.RUNNER_B, run))
                cursor.execute("UPDATE public.lab_arena_trajectory_events SET runner_hotkey=%s "
                               "WHERE run_id=%s", (prior.RUNNER_B, run))
            elif change == "other_round":
                earlier = "arena-2026-09-24-earlier"
                cursor.execute(
                    "INSERT INTO public.lab_arena_rounds("
                    "round_id,status,stage_generation,configuration_doc) "
                    "SELECT %s,status,stage_generation,configuration_doc "
                    "FROM public.lab_arena_rounds r WHERE round_id=%s",
                    (earlier, prior.ROUND),
                )
                cursor.execute("UPDATE public.lab_arena_runs SET round_id=%s "
                               "WHERE run_id=%s", (earlier, run))
                cursor.execute("UPDATE public.lab_arena_trajectory_events SET round_id=%s "
                               "WHERE run_id=%s", (earlier, run))
                cursor.execute("UPDATE public.lab_arena_ledger SET round_id=%s "
                               "WHERE run_id=%s", (earlier, run))
            elif change == "result":
                cursor.execute("UPDATE public.lab_arena_runs SET result_doc='{}'::jsonb "
                               "WHERE run_id=%s", (run,))
            elif change == "output":
                cursor.execute("UPDATE public.lab_arena_runs SET output_ref='present' "
                               "WHERE run_id=%s", (run,))
            cursor.execute("SET session_replication_role=origin")
        conn.commit()
        if change not in ("other_runner", "other_round"):
            assert _proofs(conn, [run]) == [False]
        else:
            assert _proofs(conn, [run]) == [True]
        assert prior._claim(conn, prior.RUNNER_A, "d")["run_id"] == fresh


def test_existing_zero_call_quarantine_unchanged(database, migrated):
    with database[0].connect(**database[1]) as conn:
        _, runs = faults397._seed(conn)
        assert _proofs(conn, runs) == [False] * 3
        assert prior._claim(conn, prior.RUNNER_A, "e") == {"status": "no_pending"}
