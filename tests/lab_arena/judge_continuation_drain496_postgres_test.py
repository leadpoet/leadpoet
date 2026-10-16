"""Canonical restart retains the ordinary unfinished judge continuation proof."""
import json
import hashlib
from pathlib import Path

import pytest

from tests.lab_arena import baseline_drain_lineage487_postgres_test as lineage
from tests.lab_arena import general_judge_continuation472_postgres_test as judge
from tests.lab_arena import progression_lock_order490_postgres_test as current

ROOT = Path(__file__).parents[2]
SQL = ROOT / "scripts/496-lab-arena-judge-continuation-restart-phase.sql"
SIGNATURES = (
    "public.lab_arena_expire_leases(text)",
    "public.lab_arena__restart_drain_state_v1()",
    "public.lab_arena_restart_quiescence_v1(text,text,bigint)",
)


@pytest.fixture(scope="module")
def database():
    generator = lineage.database.__wrapped__()
    db = next(generator)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute(current.HISTORICAL.read_text())
            cur.execute((ROOT / "scripts/490-lab-arena-progression-lock-order.sql").read_text())
            cur.execute((ROOT / "scripts/495-lab-arena-reviewer-runtime-selection.sql").read_text())
            cur.execute("CREATE ROLE judge_drain496_migrator NOLOGIN NOSUPERUSER INHERIT")
            cur.execute("GRANT lab_arena_owner TO judge_drain496_migrator")
            cur.execute("GRANT USAGE,CREATE ON SCHEMA public TO judge_drain496_migrator")
        yield db
    finally:
        generator.close()


def functions(cur):
    cur.execute("SELECT oid::regprocedure::TEXT,pg_get_functiondef(oid),to_jsonb(p)-'prosrc' "
                "FROM pg_proc p WHERE oid=ANY(%s::regprocedure[]) ORDER BY oid", (list(SIGNATURES),))
    return cur.fetchall()


@pytest.fixture(scope="module")
def originals(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        return functions(cur)


@pytest.fixture(autouse=True)
def restore(database, originals):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        for _, definition, _ in originals:
            cur.execute(definition)
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE,"
                    "pause_reason='',actor_ref='',guard_commitment='',owner_commitment='',"
                    "guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',"
                    "captured_leases='[]'")


def seed(conn, phase="stage2_judged", *, baseline=False):
    assignment, source, _ = judge.prepare(conn)
    with conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("UPDATE public.lab_arena_runs SET run_id=replace(run_id,':1:0',':2:6'),"
                    "assignment_id=replace(assignment_id,':1:0',':2:6'),"
                    "scored_run_id=replace(scored_run_id,':1:0',':2:6'),stage=2,icp_position=6")
        cur.execute("UPDATE public.lab_arena_rounds SET status=%s,stage2_scoring_plan_doc=%s::jsonb,"
                    "participants=jsonb_set(participants,'{0,is_king}',%s::jsonb)", (
                        phase, json.dumps({"work_items": [{
                            "scored_run_id": source.replace(":1:0", ":2:6"),
                            "submission_id": judge.previous.SUB, "icp_position": 6,
                            "output_ref": "source-output",
                        }]}), json.dumps(baseline)))
        cur.execute("UPDATE public.lab_arena_submissions SET is_king=%s", (baseline,))
        cur.execute("SET LOCAL session_replication_role=origin")
    conn.commit()
    assert judge.previous.rpc(conn,"lab_arena_expire_leases(%s)",(judge.previous.ROUND,))["retried"] == 1
    lineage.prior.cool(conn)
    lease = judge.previous.claim(conn)
    assert (lease["stage"], lease["kind"], lease["attempt"], lease["stage_generation"]) == (2,"score",3,8)
    with conn.cursor() as cur:
        cur.execute("SELECT public.lab_arena__judge_continuation_v1(a,r),"
                    "public.lab_arena__baseline_drain_phase_v1(a,r) "
                    "FROM public.lab_arena_runs r JOIN public.lab_arena_rounds a USING(round_id) "
                    "WHERE r.run_id=%s", (lease["run_id"],))
        assert cur.fetchone() == (True,False)
    return lease


def frozen(cur):
    cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE status='accepted' ORDER BY run_id")
    accepted = cur.fetchall()
    cur.execute("SELECT to_jsonb(c) FROM public.lab_arena_judgment_cache c ORDER BY cache_key")
    cache = cur.fetchall()
    cur.execute("SELECT configuration_doc,participants,stage1_scoring_plan_doc,stage2_scoring_plan_doc,"
                "publication_doc FROM public.lab_arena_rounds ORDER BY round_id")
    return accepted,cache,cur.fetchall()


def apply(conn):
    with conn.cursor() as cur:
        before = functions(cur)
        cur.execute("SET ROLE judge_drain496_migrator")
        cur.execute(SQL.read_text())
        cur.execute(SQL.read_text())
        cur.execute("RESET ROLE")
        after = functions(cur)
        assert [row[2] for row in after] == [row[2] for row in before]
    conn.commit()


def provider_call(conn,lease,*,settle=True):
    with conn.cursor() as cur:
        cur.execute("SELECT is_king FROM public.lab_arena_submissions WHERE submission_id=%s",(lease["submission_id"],))
        funding = "host" if cur.fetchone()[0] else "miner_key"
    identity = "sha256:"+hashlib.sha256((lease["run_id"]+"496").encode()).hexdigest()
    rpc = judge.previous.rpc
    assert rpc(conn,"lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter',%s,100,'{}'::jsonb,4500)",
               (lease["run_id"],judge.previous.TOKEN,identity,funding))["status"] == "reserved"
    assert rpc(conn,"lab_arena_mark_dispatched(%s,%s,%s)",
               (lease["run_id"],judge.previous.TOKEN,identity))["status"] == "dispatched"
    if settle:
        assert rpc(conn,"lab_arena_settle_call(%s,%s,%s,100,%s::jsonb,4500)",
                   (lease["run_id"],judge.previous.TOKEN,identity,
                    json.dumps({"status":200,"call_succeeded":True})))["status"] == "settled"


@pytest.mark.parametrize("phase", ["stage2_judged", "scored"])
@pytest.mark.parametrize("baseline", [False,True])
def test_existing_judge_claim_drains_then_retries_and_completes_without_changing_frozen_results(
    database,phase,baseline,
):
    psycopg,dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = seed(conn,phase,baseline=baseline)
        generation = lineage.capture(conn)
        before = lineage.audit(conn)
        with pytest.raises(psycopg.Error, match="captured write set differs") as error:
            lineage.quiescence(conn,generation)
        assert error.value.pgcode == "55000"
        conn.rollback()
        assert lineage.audit(conn) == before
        with conn.cursor() as cur:
            immutable = frozen(cur)
        apply(conn)
        assert lineage.audit(conn) == before
        waiting = lineage.quiescence(conn,generation)
        assert (waiting["still_leased_count"],waiting["lost_or_mutated_count"],waiting["preserved"]) == (1,0,False)
        assert lineage.audit(conn) == before
        provider_call(conn,lease)
        lineage.host_error(conn,lease)
        drained = lineage.quiescence(conn,generation)
        assert (drained["expired_receipt_count"],drained["pending_retry_count"],drained["preserved"]) == (1,1,True)
        assert lineage.quiescence(conn,generation)["outcome_commitment"] == drained["outcome_commitment"]
        with conn.cursor() as cur:
            assert frozen(cur) == immutable
        lineage.release(conn,generation)
        lineage.prior.cool(conn)
        retry = judge.previous.claim(conn,suffix="b")
        assert retry["attempt"] == 4 and retry["scored_run_id"] == lease["scored_run_id"]
        assert lineage.prior.complete(conn,retry)["status"] == "accepted"
        with conn.cursor() as cur:
            accepted = frozen(cur)
        assert judge.previous.rpc(conn,"lab_arena_expire_leases(%s)",(judge.previous.ROUND,))["retried"] == 0
        with conn.cursor() as cur:
            assert frozen(cur) == accepted


@pytest.mark.parametrize("fault", ["wrong_lineage","future_generation","foreign_capture","unsettled_provider"])
def test_existing_capture_lineage_generation_and_provider_fences_remain(database,fault):
    psycopg,dsn = database
    with psycopg.connect(**dsn) as conn:
        lease = seed(conn)
        generation = lineage.capture(conn)
        apply(conn)
        if fault == "unsettled_provider":
            provider_call(conn,lease,settle=False)
            lineage.host_error(conn,lease)
        else:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL session_replication_role=replica")
                if fault == "wrong_lineage":
                    cur.execute("UPDATE public.lab_arena_runs SET judgment_scope_doc="
                                "jsonb_set(judgment_scope_doc,'{scorer_image_digest}',to_jsonb(%s::TEXT)) "
                                "WHERE run_id=%s", ("sha256:"+"f"*64,lease["run_id"]))
                elif fault == "future_generation":
                    cur.execute("UPDATE public.lab_arena_runs SET stage_generation=9 WHERE run_id=%s",(lease["run_id"],))
                else:
                    cur.execute("UPDATE public.lab_arena_restart_claim_control SET captured_leases='[]'")
                cur.execute("SET LOCAL session_replication_role=origin")
            conn.commit()
        before = lineage.audit(conn)
        if fault in {"wrong_lineage","future_generation"}:
            with pytest.raises(psycopg.Error,match="captured write set differs"):
                lineage.quiescence(conn,generation)
            conn.rollback()
        else:
            state = lineage.quiescence(conn,generation)
            assert state["preserved"] is False and state["current_leased_count"] == 1
        assert lineage.audit(conn) == before


def test_existing_late_baseline_execute_lane_retains_its_authority(database):
    with database[0].connect(**database[1]) as conn:
        apply(conn)
        lease = lineage.seed(conn)
        assert lease["kind"] == "execute" and lease["stage"] == 1
        generation = lineage.capture(conn)
        before = lineage.audit(conn)
        assert lineage.quiescence(conn,generation)["still_leased_count"] == 1
        assert lineage.audit(conn) == before
        lineage.call(conn,lease)
        lineage.host_error(conn,lease)
        state = lineage.quiescence(conn,generation)
        assert state["preserved"] is True and state["expired_receipt_count"] == 1


@pytest.mark.parametrize("fault", ["preimage","postimage","helper","acl"])
def test_migration_rejects_changed_authority_atomically(database,fault):
    psycopg,dsn = database
    with psycopg.connect(**dsn) as conn,conn.cursor() as cur:
        seed(conn)
        before = functions(cur)
        rows = lineage.audit(conn)
        text = SQL.read_text()
        if fault == "preimage":
            text = text.replace("35cbc97d9d8390f2d208e36617e7684cfcad4e1d1a8296ca4fbb461cc625f594","0"*64)
        elif fault == "postimage":
            text = text.replace("7835213f7d4991be60194c60589da3d18f62f05f8640136e7df2dfce1d8e256f","0"*64)
        elif fault == "helper":
            text = text.replace("89cfed833d6bdeb3c15b0677b155679c33cf524d0055cd8ccab56986138e3c0d","0"*64)
        else:
            cur.execute("GRANT EXECUTE ON FUNCTION public.lab_arena_expire_leases(TEXT) TO anon")
        with pytest.raises(psycopg.Error,match="differs"):
            cur.execute(text)
        conn.rollback()
        assert functions(cur) == before
        assert lineage.audit(conn) == rows
