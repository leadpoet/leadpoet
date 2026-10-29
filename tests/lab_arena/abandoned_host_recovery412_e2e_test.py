"""A real HTTP runner host fault recovers before expiry and the day publishes."""
from dataclasses import replace
from datetime import datetime, timezone
import json
import threading
import time

import pytest

from lab_arena import runtime, scoring
from lab_arena.runtime_host import RuntimeHostError
from tests.lab_arena import proxy_model_capacity401_e2e_test as fullflow
from tests.lab_arena import abandoned_host_recovery412_postgres_test as recovery
from tests.lab_arena.proxy_model_capacity401_e2e_test import complete_application_database


# Exact migration 412 upgrades retain their historical SQL fixture. This file
# drives the current HTTP service, which also needs all later read/recovery RPCs.
database=fullflow.database
migrated=fullflow.migrated


@pytest.mark.parametrize('fault_kind', ['execute', 'score_continuation'])
def test_host_fault_recovers_via_signed_http_run_then_publishes_and_promotes(complete_application_database,tmp_path,monkeypatch,fault_kind):
    _exercise_host_fault(complete_application_database,True,tmp_path,monkeypatch,fault_kind=fault_kind)


def _exercise_host_fault(database,migrated,tmp_path,monkeypatch,*,fault_kind='execute'):
    install=fullflow._install_company_only_sandbox
    drain=fullflow._drain_both
    observed={}
    replacements={}
    fault_lock=threading.Lock()
    def failing_install(harness):
        harness.service.config.defaults=replace(harness.service.config.defaults,checkpoint_deadline_enabled=True)
        counts=install(harness)
        original=harness.sandbox.run_icp
        def run_icp(spec,**kwargs):
            document=json.loads((spec.input_dir/runtime.INPUT_FILE_NAME).read_text())
            with fault_lock:
                is_score=document.get('schema_version')==scoring.SCORING_INPUT_SCHEMA_VERSION
                fail=is_score==(fault_kind=='score_continuation') and not observed.get('injected')
                if fail: observed['injected']=True
            if fail: raise RuntimeHostError(reason='sandbox_launch_failed')
            return original(spec,**kwargs)
        harness.sandbox.run_icp=run_icp
        observed['harness']=harness
        return counts
    def recovering_drain(harness,runners):
        answer=drain(harness,tuple(dict.fromkeys(replacements.get(id(r),r) for r in runners)))
        if observed.get('injected') and not observed.get('recovered'):
            psycopg,dsn=database
            with psycopg.connect(**dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT r.run_id,r.runner_hotkey,r.claim_response,r.lease_expires_at FROM public.lab_arena_runs r JOIN public.lab_arena_trajectory_events e ON e.run_id=r.run_id WHERE r.round_id=%s AND e.event_kind='runtime.error' AND e.content->>'error_class'='RuntimeHostError'",(harness.round_id,))
                    rows=cur.fetchall();assert len(rows)==1
                    run,failed_hotkey,claim,expiry=rows[0]
                    assert expiry>datetime.now(timezone.utc)
                    assert (expiry-datetime.now(timezone.utc)).total_seconds()>4400
                    cur.execute('SELECT count(*) FROM public.lab_arena_ledger WHERE run_id=%s',(run,));assert cur.fetchone()[0]==0
            # Gateway's normal scheduling path calls the exact expiry RPC.
            transition=harness.service.advance_round(harness.round_id)
            assert transition['status'] not in ('cancelled','retry','stale')
            if fault_kind=='score_continuation':
                # Modern baseline judges may continue while stage 2 executes.
                # An old-generation lease cannot take the same-stage fast
                # expiry path. Prove that fence, then advance only its fixture
                # lease clock and exercise the real natural-expiry RPC.
                with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
                    cur.execute('SELECT r.status,r.stage_generation<>a.stage_generation,public.lab_arena__judge_continuation_v1(a,r) '
                                'FROM public.lab_arena_runs r JOIN public.lab_arena_rounds a USING(round_id) WHERE run_id=%s',(run,))
                    assert cur.fetchone()==('leased',True,True)
                    cur.execute('SET LOCAL session_replication_role=replica')
                    cur.execute('UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp() WHERE run_id=%s',(run,))
                    cur.execute('SET LOCAL session_replication_role=origin')
                assert harness.service.store.expire_leases(harness.round_id)['expired']==1
                with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
                    # Advance the existing score-retry cooldown without
                    # changing failed history, claim receipts or admission.
                    cur.execute('SET LOCAL session_replication_role=replica')
                    cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' "
                                "WHERE run_id=%s AND status='pending'",(run.rsplit(':',1)[0]+':2',))
                    cur.execute('SET LOCAL session_replication_role=origin')
            with psycopg.connect(**dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute('SELECT status,terminal_cause,terminal_doc,claim_response FROM public.lab_arena_runs WHERE run_id=%s',(run,))
                    state,cause,doc,frozen=cur.fetchone()
                    assert (state,cause)==('failed','lease_expired') and frozen==claim
                    if fault_kind=='execute':
                        assert doc['recovery_reason']=='authenticated_zero_call_runtime_host_error'
            failed=next(r for r in runners if r._config.identity.hotkey==failed_hotkey)
            healthy=next(r for r in runners if r is not failed)
            failed.close()
            # The surviving validator is reconstructed, as after its restart.
            healthy_config=healthy._config
            healthy.close()
            replacement=type(healthy)(healthy_config)
            # The current score retry has a real 30-second backoff. Preserve
            # its queue timestamp and wait only for that existing admission
            # boundary, rather than requiring an immediate retry lease.
            retry=run.rsplit(':',1)[0]+':2'
            with psycopg.connect(**dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT extract(epoch FROM created_at+interval '30 seconds'-clock_timestamp()) FROM public.lab_arena_runs WHERE run_id=%s",(retry,))
                    delay=float(cur.fetchone()[0])
            assert delay<=30
            time.sleep(max(0,delay)+0.01)
            drain(harness,(replacement,))
            with psycopg.connect(**dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute('SELECT status,runner_hotkey FROM public.lab_arena_runs WHERE run_id=%s',(retry,))
                    state,runner=cur.fetchone()
                    assert state=='accepted' and runner!=failed_hotkey
                    assert expiry>datetime.now(timezone.utc)
            observed['recovered']=True
            replacements[id(failed)]=replacement
            replacements[id(healthy)]=replacement
        return answer
    monkeypatch.setattr(fullflow,'_install_company_only_sandbox',failing_install)
    monkeypatch.setattr(fullflow,'_drain_both',recovering_drain)
    try:
        fullflow._exercise_company_only_round(database,migrated,tmp_path)
    finally:
        for runner in set(replacements.values()):
            runner.close()
    assert observed['injected'] and observed['recovered']
