"""Bounded baseline-first cutoff drain, using actual SQL and signed runners."""
from __future__ import annotations

import copy
import json
import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena import contracts, service as svc
from lab_arena.store import hash_lease_token
from tests.lab_arena.oct07_cutoff_recovery419_postgres_test import (
    base_database, database as current_database, _snapshot,
    fixture_configuration, QualityHarness, _install_company_only_sandbox,
    _verified_test_pool, daily_icps,
)
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

ROOT = Path(__file__).resolve().parents[2]
SQL = ROOT / "scripts/420-lab-arena-bounded-stage-cutoff-drain.sql"
ROUND = "arena-2026-10-07-drain420"
SUB = "drain420-miner"
RUNNER = hotkey("drain420-runner")
PROFILES = ((45, 2700, 3600), (60, 3600, 4500), (90, 5400, 6300))


@pytest.fixture(scope="module")
def database(current_database):
    psycopg, dsn = current_database
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT oid::regprocedure::text,proowner,proacl,prosecdef,proconfig FROM pg_proc WHERE proname IN ('lab_arena_claim_assignment','lab_arena_close_stage','lab_arena_complete_attempt','lab_arena_expire_leases') ORDER BY oid")
        before = cur.fetchall()
        cur.execute(SQL.read_text())
        cur.execute(SQL.read_text())
        cur.execute("SELECT oid::regprocedure::text,proowner,proacl,prosecdef,proconfig FROM pg_proc WHERE proname IN ('lab_arena_claim_assignment','lab_arena_close_stage','lab_arena_complete_attempt','lab_arena_expire_leases') ORDER BY oid")
        assert cur.fetchall() == before
    return current_database


def _stamp(at):
    return at.isoformat().replace("+00:00", "Z")


def _seed(conn, *, profile=PROFILES[1], kind="execute", policy=True, positions=2):
    config = fixture_configuration(ROUND)
    minutes, wall, ttl = profile
    config.update(checkpoint_deadline_policy=f"atomic_checkpoint_{minutes}m_v1",
                  icp_wall_clock_seconds=wall, lease_ttl_seconds=ttl,
                  runner_hotkeys=[RUNNER], parallel_twenty_icp_execution=False,
                  sourcing_cost_eligibility_policy=contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY,
                  execution_icp_cap_microusd=4_000_000)
    if policy:
        config["execution_sequence_policy"] = contracts.BASELINE_SCORED_FIRST_POLICY
    else:
        config.pop("execution_sequence_policy", None)
    config["schedule"]["stage_2_close"] = _stamp(datetime.now(timezone.utc) + timedelta(hours=1))
    with conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("TRUNCATE public.lab_arena_rounds RESTART IDENTITY CASCADE")
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,guard_commitment='',owner_commitment='',guard_expires_at=NULL,captured_leases='[]'")
        source=f"arena/{ROUND}/sources/{SUB}.tar.gz"
        cur.execute("INSERT INTO public.lab_arena_rounds(round_id,status,status_generation,stage_generation,configuration_doc,participants,benchmark_ref,rewards_enabled) VALUES (%s,%s,9,7,%s::jsonb,%s::jsonb,%s,false)",
                    (ROUND,"stage2" if kind=="execute" else "stage2_scoring",json.dumps(config),json.dumps([dict(submission_id=SUB,miner_hotkey=hotkey(SUB),is_king=False,source_ref=source)]),f"arena/{ROUND}/benchmark.json"))
        cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status,is_king,source_ref,source_size_bytes,submission_doc,code_review_status,code_review_doc,code_review_claim,code_review_started_at,code_review_attempts) VALUES (%s,%s,%s,'frozen',false,%s,123,'{}','passed','{\"decision\":\"pass\"}','sha256:'||repeat('1',64),now(),1)",(SUB,ROUND,hotkey(SUB),source))
        for index in range(positions):
            assignment=f"{ROUND}:{SUB}:2:{10+index}" + (":score" if kind=="score" else "")
            cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation) VALUES (%s,%s,%s,%s,%s,2,%s,1,%s,'pending',7)",(assignment+":1",assignment,ROUND,SUB,hotkey(SUB),10+index,kind))
        cur.execute("SET LOCAL session_replication_role=origin")
    conn.commit()
    return config


def _boundary(conn, seconds=-1, *, ttl=None, paused=False):
    with conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,stage_2_close}',to_jsonb(%s::text)) WHERE round_id=%s",(_stamp(datetime.now(timezone.utc)+timedelta(seconds=seconds)),ROUND))
        if ttl is not None:
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{lease_ttl_seconds}',to_jsonb(%s::integer)) WHERE round_id=%s",(ttl,ROUND))
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=%s",(paused,))
        cur.execute("SET LOCAL session_replication_role=origin")
    conn.commit()


def _rpc(conn, sql, args=()):
    with conn.cursor() as cur:
        cur.execute(sql,args)
        value=cur.fetchone()[0]
    conn.commit()
    return value


def _claim(conn, suffix="a", *, ttl=4500):
    return _rpc(conn,"SELECT public.lab_arena_claim_assignment(%s,%s,20,20,%s,%s,%s,%s,%s)",(ROUND,RUNNER,[],suffix*32,"sha256:"+suffix*64,"sha256:"+suffix*64,ttl))


def _close(conn):
    return _rpc(conn,"SELECT public.lab_arena_close_stage(%s,2::smallint)",(ROUND,))


def _state(conn):
    with conn.cursor() as cur:
        result=_snapshot(cur)
    return result


def _complete(conn, lease, cause="accepted", token="a"):
    return _rpc(conn,"SELECT public.lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)",(lease["run_id"],"sha256:"+token*64,json.dumps({"terminal_status":cause}),cause,f"arena/{ROUND}/outputs/preserved.json" if cause=="accepted" else None))


@pytest.mark.parametrize("profile",PROFILES)
def test_live_cutoff_drain_is_byte_preserving_with_frozen_profile(database,profile):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        config=_seed(conn,profile=profile)
        lease=_claim(conn,ttl=profile[2]); assert lease["status"]=="leased"
        assert config["icp_wall_clock_seconds"]==profile[1]
        _boundary(conn)
        before=_state(conn)
        assert _close(conn)=={"status":"draining","round_status":"stage2","stage_generation":7}
        assert _state(conn)==before
        assert _claim(conn)==lease  # exact stored claim replay, no new lease
        assert _state(conn)==before
        assert _claim(conn,"b",ttl=profile[2])=={"status":"no_pending"}
        assert _state(conn)==before
        pending=[r for r in before["lab_arena_runs"] if r["status"]=="pending"]
        assert pending and all(r["result_doc"] is None and r["per_icp_score"] is None for r in pending)


@pytest.mark.parametrize("profile",PROFILES)
def test_renewed_billing_lease_cannot_extend_absolute_bound(database,profile):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        _seed(conn,profile=profile)
        lease=_claim(conn,ttl=profile[2]); assert lease["status"]=="leased"
        with conn.cursor() as cur:
            cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()+interval '1 day' WHERE run_id=%s",(lease["run_id"],))
        conn.commit()
        _boundary(conn,-profile[2]-1)
        result=_close(conn); assert result["status"]=="cancelled"
        after=_state(conn)
        roundrow=after["lab_arena_rounds"][0]
        assert (roundrow["status_generation"],roundrow["stage_generation"])==(10,8)
        assert all(r["terminal_cause"]=="stage_closed" and r["per_icp_score"] is None for r in after["lab_arena_runs"])
        assert after["lab_arena_ledger"]==[]
        assert _close(conn)["status"]=="stale"
        assert _state(conn)==after


@pytest.mark.parametrize("policy,ttl",[(False,None),(True,4499)])
def test_legacy_or_invalid_frozen_profile_gets_no_new_grace(database,policy,ttl):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        _seed(conn,policy=policy); assert _claim(conn)["status"]=="leased"
        _boundary(conn,ttl=ttl)
        assert _close(conn)["status"]=="cancelled"


@pytest.mark.parametrize("after",[False,True])
@pytest.mark.parametrize("method",["failure","expiry"])
def test_execute_retry_admission_uses_same_cutoff(database,after,method):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        _seed(conn,positions=1); lease=_claim(conn)
        assert lease["status"]=="leased"
        if after: _boundary(conn)
        if method=="failure":
            assert _complete(conn,lease,"provider_error")["status"]=="failed"
        else:
            with conn.cursor() as cur:
                cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s",(lease["run_id"],))
            conn.commit()
            result=_rpc(conn,"SELECT public.lab_arena_expire_leases(%s)",(ROUND,))
            assert result["expired"]==1 and result["retried"]==int(not after)
        runs=_state(conn)["lab_arena_runs"]
        assert len(runs)==(1 if after else 2)
        original=next(r for r in runs if r["attempt"]==1)
        assert original["terminal_cause"]==("provider_error" if method=="failure" else "lease_expired")
        if after:
            assert _close(conn)["status"]=="cancelled"
            assert all(r["per_icp_score"] is None for r in _state(conn)["lab_arena_runs"])


def test_late_valid_completion_and_early_complete_close(database):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        _seed(conn,positions=1); lease=_claim(conn); _boundary(conn)
        assert _complete(conn,lease)["status"]=="accepted"
        assert _close(conn)["status"]=="closed"

        assert _state(conn)["lab_arena_rounds"][0]["status"]=="stage2_closed"
        _seed(conn,positions=1); lease=_claim(conn)
        assert _complete(conn,lease)["status"]=="accepted"
        assert _close(conn)["status"]=="closed"


def test_current_recovery_queue_shape_drains_without_zeroing_unstarted_work(database):
    """Synthetic equivalent of the observed21 accepted/10 active/49 pending.

    No production rows or credentials are copied. Current9/7 generations and
    the current60-minute frozen profile are preserved through the drain.
    """
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        _seed(conn,positions=10)
        with conn.cursor() as cur:
            cur.execute("SET LOCAL session_replication_role=replica")
            for group in range(1,8):
                sub=f"{SUB}-{group}"
                cur.execute("INSERT INTO public.lab_arena_submissions SELECT (jsonb_populate_record(NULL::public.lab_arena_submissions,to_jsonb(s)||jsonb_build_object('submission_id',%s,'miner_hotkey',%s))).* FROM public.lab_arena_submissions s WHERE submission_id=%s",(sub,hotkey(sub),SUB))
                for position in range(10,20):
                    assignment=f"{ROUND}:{sub}:2:{position}"
                    cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation) VALUES (%s,%s,%s,%s,%s,2,%s,2,'execute','pending',7)",(assignment+":2",assignment,ROUND,sub,hotkey(sub),position))
            cur.execute("WITH ids AS (SELECT run_id,row_number() OVER (ORDER BY run_id) n FROM public.lab_arena_runs) UPDATE public.lab_arena_runs r SET status=CASE WHEN n<=21 THEN 'accepted' WHEN n<=31 THEN 'leased' ELSE 'pending' END,runner_hotkey=CASE WHEN n BETWEEN 22 AND 31 THEN %s END,lease_expires_at=CASE WHEN n BETWEEN 22 AND 31 THEN clock_timestamp()+interval '75 minutes' END,lease_token_hash=CASE WHEN n BETWEEN 22 AND 31 THEN 'sha256:'||repeat('a',64) END,lease_generation=CASE WHEN n BETWEEN 22 AND 31 THEN 1 ELSE 0 END,output_ref=CASE WHEN n<=21 THEN %s END,result_doc=CASE WHEN n<=21 THEN '{\"terminal_status\":\"accepted\"}'::jsonb END FROM ids WHERE r.run_id=ids.run_id",(RUNNER,f"arena/{ROUND}/outputs/preserved.json"))
            cur.execute("SET LOCAL session_replication_role=origin")
        conn.commit(); _boundary(conn,0)
        before=_state(conn)
        assert {status:sum(r["status"]==status for r in before["lab_arena_runs"]) for status in ("accepted","leased","pending")}=={"accepted":21,"leased":10,"pending":49}
        assert _close(conn)["status"]=="draining"
        assert _state(conn)==before
        assert _claim(conn,"b")=={"status":"no_pending"}
        assert _state(conn)==before


def test_score_claim_and_confirmation_retry_remain_available(database):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        _seed(conn,kind="score",positions=1); _boundary(conn)
        lease=_claim(conn); assert lease["status"]=="leased"
        assert _complete(conn,lease,"judge_error")["status"]=="failed"
        runs=_state(conn)["lab_arena_runs"]
        assert len(runs)==2 and runs[-1]["status"]=="pending"


def test_hold_stale_token_and_close_claim_lock_preserve_work(database):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn,psycopg.connect(**dsn) as other:
        _seed(conn); lease=_claim(conn); _boundary(conn,paused=True)
        before=_state(conn)
        assert _close(conn)["status"]=="draining"
        assert _state(conn)==before
        _boundary(conn,-4501,paused=True)
        before=_state(conn)
        with pytest.raises(Exception, match="progression_paused"):
            _close(conn)
        conn.rollback()
        assert _state(conn)==before
        _boundary(conn,paused=False)
        assert _complete(conn,lease,token="f")["status"]!="accepted"
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM public.lab_arena_rounds WHERE round_id=%s FOR UPDATE",(ROUND,))
        with other.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout='100ms'")
            with pytest.raises(Exception,match="lock timeout"):
                cur.execute("SELECT public.lab_arena_claim_assignment(%s,%s,20,20,%s,%s,%s,%s,4500)",(ROUND,RUNNER,[],"c"*32,"sha256:"+"c"*64,"sha256:"+"c"*64))
        other.rollback(); conn.rollback()
        assert _claim(other,"c")=={"status":"no_pending"}
        assert _close(conn)["status"]=="draining"


@pytest.mark.parametrize("case",["before","after","scoring","legacy","held"])
def test_service_billing_return_only_yields_at_execution_cutoff(case):
    now=datetime.now(timezone.utc)
    service=object.__new__(svc.ArenaService)
    calls=[]
    status="stage2_scoring" if case=="scoring" else "stage2"
    config={"schedule":{"stage_2_close":_stamp(now+timedelta(seconds=1 if case=="before" else -1))}}
    if case!="legacy": config["execution_sequence_policy"]=contracts.BASELINE_SCORED_FIRST_POLICY
    service._clock=lambda:now
    service._invalidate_hot_round=lambda:None
    service._round=lambda _:dict(status=status,configuration_doc=config)
    service._reconcile_deepline_cost=lambda _:{"status":"none"}
    service._reconcile_openrouter_cost=lambda _:{"status":"pending","run_status":"leased","lease_expires_at":_stamp(now+timedelta(days=1))}
    service._store=SimpleNamespace(operator_hold_active=lambda:case=="held",expire_leases=lambda _:calls.append("expire"))
    service._advance_round_locked=lambda _:calls.append("advance") or {"status":"draining"}
    result=service.advance_round(ROUND)
    if case=="after": assert result=={"status":"draining"} and calls==["advance"]
    elif case=="held": assert result["status"]=="paused" and calls==[]
    else: assert result["reason"]=="provider_billing_pending" and calls==["expire"]


@pytest.mark.parametrize("mutation,expected",[("security","security shape differs"),("partial","partial state differs"),("body","preimage differs")])
def test_migration_replay_rejects_changed_security_or_partial_patch(database,mutation,expected):
    import re
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn,conn.cursor() as cur:
        if mutation=="security": cur.execute("ALTER FUNCTION public.lab_arena_close_stage(text,smallint) SET search_path=public")
        else:
            cur.execute("SELECT pg_get_functiondef('public.lab_arena_close_stage(text,smallint)'::regprocedure)")
            definition=cur.fetchone()[0]
            if mutation=="partial":
                old=re.search(r'\$old1\$(.*?)\$old1\$',SQL.read_text(),re.S).group(1)
                new=re.search(r'\$new1\$(.*?)\$new1\$',SQL.read_text(),re.S).group(1)
                definition=definition.replace(new,old)
            else: definition=definition.replace("BEGIN","BEGIN\n  -- test drift",1)
            cur.execute(definition)
        with pytest.raises(Exception,match=expected): cur.execute(SQL.read_text())
        conn.rollback()
        cur.execute(SQL.read_text())


def test_last_function_postimage_failure_rolls_back_all_four_patches(database):
    import re
    psycopg,dsn=database
    names=("lab_arena_claim_assignment","lab_arena_close_stage","lab_arena_complete_attempt","lab_arena_expire_leases")
    with psycopg.connect(**dsn) as conn,conn.cursor() as cur:
        cur.execute("SELECT proname,pg_get_functiondef(oid) FROM pg_proc WHERE proname=ANY(%s)",(list(names),))
        definitions=dict(cur.fetchall())
        try:
            for index,name in enumerate(names):
                old=re.search(r'\$old%d\$(.*?)\$old%d\$'%(index,index),SQL.read_text(),re.S).group(1)
                new=re.search(r'\$new%d\$(.*?)\$new%d\$'%(index,index),SQL.read_text(),re.S).group(1)
                assert definitions[name].count(new)==1
                cur.execute(definitions[name].replace(new,old))
            conn.commit()
            cur.execute("SELECT proname,pg_get_functiondef(oid) FROM pg_proc WHERE proname=ANY(%s)",(list(names),))
            before=cur.fetchall()
            state=_state(conn)
            corrupted=SQL.read_text().replace("ef61f7bf17884267c7c126529148099e48591eb973e88ecb841eea4498439688","0"*64)
            with pytest.raises(Exception,match="postimage differs"):
                cur.execute(corrupted)
            conn.rollback()
            cur.execute("SELECT proname,pg_get_functiondef(oid) FROM pg_proc WHERE proname=ANY(%s)",(list(names),))
            assert cur.fetchall()==before
            assert _state(conn)==state
        finally:
            conn.rollback()
            for definition in definitions.values(): cur.execute(definition)
            conn.commit()


@pytest.mark.parametrize("cause",["model_timeout","model_error","invalid_output","budget_exhausted","credential_error"])
def test_genuine_model_failures_keep_terminal_classification_without_retry(database,cause):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        _seed(conn,positions=1); lease=_claim(conn); _boundary(conn)
        assert _complete(conn,lease,cause)["status"]=="failed"
        runs=_state(conn)["lab_arena_runs"]
        assert len(runs)==1 and runs[0]["terminal_cause"]==cause
        assert runs[0]["result_doc"]=={"terminal_status":cause}
        assert _close(conn)["status"]=="closed"
        assert _state(conn)["lab_arena_runs"]==runs


def test_stale_generation_does_not_create_retry_or_drain(database):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        _seed(conn,positions=1); lease=_claim(conn); _boundary(conn)
        with conn.cursor() as cur:
            cur.execute("SET LOCAL session_replication_role=replica")
            cur.execute("UPDATE public.lab_arena_runs SET stage_generation=6 WHERE run_id=%s",(lease["run_id"],))
        conn.commit()
        assert _complete(conn,lease)["status"]!="accepted"
        assert _close(conn)["status"]=="cancelled"
        assert len(_state(conn)["lab_arena_runs"])==1


def test_authenticated_abandoned_host_after_cutoff_keeps_failure_without_retry(database):
    import uuid
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        _seed(conn,positions=1); lease=_claim(conn); _boundary(conn)
        events=[{"event_id":str(uuid.uuid4()),"kind":"runtime.error","occurred_at":_stamp(datetime.now(timezone.utc)),"content":{"status":"abandoned","failure_stage":"runtime","error_class":"RuntimeHostError"}}]
        with conn.cursor() as cur:
            cur.execute("SET ROLE lab_arena_service")
            cur.execute("SELECT public.lab_arena_append_trajectory_events_v1(%s,%s,%s::jsonb)",(lease["run_id"],"sha256:"+"a"*64,json.dumps(events)))
            assert cur.fetchone()[0]["status"]=="accepted"
            cur.execute("RESET ROLE")
        conn.commit()
        assert _rpc(conn,"SELECT public.lab_arena_expire_leases(%s)",(ROUND,))=={"status":"ok","expired":1,"retried":0}
        runs=_state(conn)["lab_arena_runs"]
        assert len(runs)==1 and runs[0]["terminal_cause"]=="lease_expired"
        assert runs[0]["terminal_doc"]["recovery_reason"]=="authenticated_zero_call_runtime_host_error"
        assert _close(conn)["status"]=="cancelled"


def test_real_service_billing_renewal_reaches_sql_bound_and_keeps_cost_history(database):
    from lab_arena.store import ArenaStore,PsycopgTransport
    psycopg,dsn=database
    connect=lambda:psycopg.connect(**dsn)
    store=ArenaStore(PsycopgTransport(connect),lease_ttl_seconds=4500)
    try:
        with connect() as conn:
            _seed(conn); lease=_claim(conn)
            _boundary(conn)
        arguments=dict(run_id=lease["run_id"],lease_token_hash="sha256:"+"a"*64,call_identity="sha256:"+"e"*64)
        reserved=store.reserve_call(**arguments,operation_id="openrouter.chat",provider="openrouter",funding_source="miner_key",amount_microusd=1000,call_doc={})
        assert (reserved["status"],reserved["amount_microusd"])==("reserved",0)
        assert store.mark_dispatched(**arguments)["status"]=="dispatched"
        assert store.mark_uncertain(**arguments,call_doc={"reason":"missing_provider_cost","call_succeeded":True,"provider_status":200})["status"]=="uncertain"
        with connect() as conn,conn.cursor() as cur:
            cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()+interval '1 day' WHERE run_id=%s",(lease["run_id"],))
        service=object.__new__(svc.ArenaService)
        service._store=store
        service._lock=threading.RLock()
        service._clock=lambda:datetime.now(timezone.utc)
        service._invalidate_hot_round=lambda:None
        service._round=lambda _:store.get_round(ROUND)
        service._reconcile_deepline_cost=lambda _:{"status":"none"}
        service._reconcile_openrouter_cost=lambda _:{"status":"pending","run_status":"leased","lease_expires_at":_stamp(datetime.now(timezone.utc)+timedelta(days=1))}
        with connect() as conn: before=_state(conn)
        assert service.advance_round(ROUND)["status"]=="draining"
        with connect() as conn:
            assert _state(conn)==before
            _boundary(conn,-4501)
            before_close=_state(conn)
        assert service.advance_round(ROUND)["status"]=="cancelled"
        with connect() as conn: after=_state(conn)
        # Original uncertain charge stays counted and retains its source key.
        assert after["lab_arena_ledger"][:len(before_close["lab_arena_ledger"])]==before_close["lab_arena_ledger"]
        assert any(e["entry_kind"]=="uncertain" and e["call_identity"]==arguments["call_identity"] for e in after["lab_arena_ledger"])
        assert all(r["per_icp_score"] is None for r in after["lab_arena_runs"])
    finally:
        store.close()


def test_real_signed_late_execution_preserves_costs_and_publishes(database,tmp_path):
    """Actual service, signed runner, cost RPCs and scoring/publication.

    Only local fixture schedule boundaries are projected before claiming.
    Provider transport and judge are deterministic substitutes, not live calls.
    """
    import time
    from tests.lab_arena.confirmed_cost_admission_postgres_test import _reserve,_dispatch,_settle
    psycopg,dsn=database
    connect=lambda:psycopg.connect(**dsn)
    with connect() as conn,conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("TRUNCATE public.lab_arena_rounds CASCADE")
    h=QualityHarness(connect,tmp_path,challengers=["PublicBaselineAlternate"],runners=["alpha","beta"])
    h.service.config.defaults=replace(h.service.config.defaults,benchmark_icp_count=10,checkpoint_deadline_enabled=True,runner_slot_ceiling=10,company_quality_from=None,intent_details_from="2026-01-01T00:00:00Z",contacts_from=None,execution_sequence_from="2026-01-01T00:00:00Z",per_icp_cost_policy=True,rewards_enabled=True)
    h.service.config.daily_icp_source=lambda **kwargs:{"status":"ready","set_id":int(kwargs["set_id"]),"icps":copy.deepcopy(daily_icps()[:10])}
    observed=_install_company_only_sandbox(h)
    runtime_budgets=[]
    sandbox_run=h.sandbox.run_icp
    def bounded_sandbox(spec,**kwargs):
        from lab_arena import runtime,scoring
        document=json.loads((spec.input_dir/runtime.INPUT_FILE_NAME).read_text())
        if document.get("schema_version")!=scoring.SCORING_INPUT_SCHEMA_VERSION:
            runtime_budgets.append(spec.wall_clock_seconds)
        return sandbox_run(spec,**kwargs)
    h.sandbox.run_icp=bounded_sandbox
    h.round_id=ROUND
    h.service.create_round(datetime.now(timezone.utc)+timedelta(minutes=30),round_id=ROUND)
    h.submit("PublicBaselineAlternate",ROUND)
    h.clock.advance_to(h.schedule()["submission_cutoff"])
    assert h.service.advance_round(ROUND)["status"]=="ok"
    participants=h.service.store.get_round(ROUND)["participants"]
    for participant in participants: h.flavors.setdefault(participant["submission_id"],"PublicBaseline")
    h.clock.advance_to(h.schedule()["stage_1_start"])
    assert h.service.advance_round(ROUND)["status"]=="ok"
    original_runner=h.runner
    def verified_runner(index,parallel=10):
        runner=original_runner(index,parallel)
        runner._config.proxy_worker_pool=_verified_test_pool(parallel)
        return runner
    h.runner=verified_runner
    h.advance_until("stage2",runners=1,max_steps=40)
    store=h.service.store
    projected=copy.deepcopy(store.get_round(ROUND)["configuration_doc"])
    now=datetime.now(timezone.utc)
    fields=("submission_open","submission_cutoff","benchmark_deadline","stage_1_start","stage_1_close","stage_1_scoring_close","stage_2_start","stage_2_close","final_scoring_close","publication_deadline")
    # Preserve monotonic dates while placing the local stage2 cutoff seconds
    # away. All subsequent signed leases bind this same frozen document.
    for index,field in enumerate(fields):
        if field=="submission_open": continue  # frozen previous-day bank date
        if field=="stage_2_close": at=now+timedelta(seconds=5)
        elif field in ("final_scoring_close","publication_deadline"):
            at=now+timedelta(hours=6,seconds=index)
        else: at=now-timedelta(hours=3,minutes=20-index)
        projected["schedule"][field]=_stamp(at)
    with connect() as conn,conn.cursor() as cur:
        cur.execute("SET LOCAL session_replication_role=replica")
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s",(json.dumps(projected),ROUND))
    h.clock.now=datetime.now(timezone.utc)
    runner=h.runner(0)
    try:
        leases=[runner.claim_one() for _ in range(10)]
        assert all(l["status"]=="leased" for l in leases)
        assert all((l["icp_wall_clock_seconds"],l["lease_ttl_seconds"])==(3600,4500) for l in leases)
        assert all(l["attempt"]==1 for l in leases)
        identity,reserved=_reserve(store,leases[0],leases[0]["lease_token"],"cutoff-drain-paid",1234)
        assert reserved["status"]=="reserved"
        assert _dispatch(store,leases[0],leases[0]["lease_token"],identity)["status"]=="dispatched"
        deadline=datetime.fromisoformat(projected["schedule"]["stage_2_close"].replace("Z","+00:00"))
        time.sleep(max(0,(deadline-datetime.now(timezone.utc)).total_seconds())+.03)
        h.clock.now=datetime.now(timezone.utc)
        h.service._reconcile_openrouter_cost=lambda _:{"status":"pending","run_status":"leased","lease_expires_at":leases[0]["lease_expires_at"]}
        with connect() as conn: before=_state(conn)
        assert h.service.advance_round(ROUND)["status"]=="draining"
        with connect() as conn: assert _state(conn)==before
        assert runner.claim_one()["status"]=="no_free_slot"
        assert _settle(store,leases[0],leases[0]["lease_token"],identity,1234)["status"]=="settled"
        h.service._reconcile_openrouter_cost=lambda _:{"status":"none"}
        for lease in leases:
            assert runner._slots.acquire(blocking=False)
            runner._run_lease(lease)
            assert runner.completed[-1]["result"]["status"]=="accepted"
        h.advance_until("published",runners=1,max_steps=60)
    finally:
        runner.close()
    final=store.get_round(ROUND)
    assert final["configuration_doc"]==projected
    assert final["publication_doc"]["final_ranking"]
    assert observed=={"execute":20,"score":20}
    assert runtime_budgets==[3600]*20
    assert h.service.activate_reward(ROUND)["status"]=="activated"
    h.clock.now=datetime.now(timezone.utc)+timedelta(days=2)
    for participant in participants:
        public=h.service.public_results(ROUND,participant["submission_id"])
        assert public["score_status"]=="published"
        assert len(public["scores"]["stage_1"]+public["scores"]["stage_2"])==10
    with connect() as conn:
        after=_state(conn)
        old_accepted=[r for r in before["lab_arena_runs"] if r["status"]=="accepted"]
        for old in old_accepted:
            current=next(r for r in after["lab_arena_runs"] if r["run_id"]==old["run_id"])
            assert current==old
        assert all(r["status"]=="accepted" for r in after["lab_arena_runs"])
        assert _close(conn)["status"]=="existing"
        assert _state(conn)==after
    proof={"configuration_hash":contracts.document_hash(projected),"scorer_policy_hash":contracts.document_hash(projected["scorer_policy"]),"scoring_adapter":projected["scorer_policy"]["scoring_adapter_version"],"signed_wall_seconds":3600,"frozen_host_ttl_seconds":4500,"precutoff_leases":10,"late_accepted":10,"old_accepted_unchanged":len(old_accepted),"drain_byte_preserving":True,"unresolved_dispatch_preserved":True,"settled_microusd":1234,"executed":observed,"publication":final["status"],"reward_activated":True,"limitation":"Projected local dates; fake provider transport and deterministic judge. No production image/provider run."}
    (tmp_path/"stage-cutoff-drain-transition-proof.json").write_text(json.dumps(proof,indent=2)+"\n")


def test_confirmed_billed_failure_still_reaches_cap_across_attempts(database,tmp_path):
    from tests.lab_arena.confirmed_cost_admission_postgres_test import test_confirmed_billed_failure_reaches_cap_across_attempts
    test_confirmed_billed_failure_reaches_cap_across_attempts(database,tmp_path)
