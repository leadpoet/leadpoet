"""Recover only authenticated, stopped, zero-call host failures; retain history."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import uuid

import pytest

from tests.lab_arena import round_host_fault_quarantine406_postgres_test as prior406
from tests.lab_arena import score_host_retry398_postgres_test as prior398
from tests.lab_arena import zero_setup_runner_handoff360_postgres_test as prior

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/412-lab-arena-abandoned-host-recovery.sql'
PREHASH = '21634a503b8526bd51dbe102bb89b0bbcbdef582057070ae1d66324b0ebc0532'
database = prior406.database


@pytest.fixture(scope='module')
def migrated(database):
    prior406.migrated.__wrapped__(database)
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for number in (407,408,409,410,411):
                paths = list((ROOT/'scripts').glob(f'{number}-*.sql'))
                assert len(paths) == 1
                cur.execute(paths[0].read_text())
            before = prior398._security(cur)
            assert prior398._hash(cur, prior398.EXPIRY) == PREHASH
            cur.execute('BEGIN')
            with pytest.raises(psycopg.Error, match='preimage differs'):
                cur.execute(SQL.read_text().replace(PREHASH, '0'*64, 1))
            cur.execute('ROLLBACK')
            cur.execute(SQL.read_text())
            post = prior398._hash(cur, prior398.EXPIRY)
            cur.execute(SQL.read_text())
            assert prior398._hash(cur, prior398.EXPIRY) == post
            assert prior398._security(cur) == before
    return True


def _claim(conn, runner, suffix):
    with conn.cursor() as cur:
        cur.execute('SELECT public.lab_arena_claim_assignment(%s,%s,20,20,%s,%s,%s,%s,4500)',
                    (prior.ROUND,runner,[],suffix*32,'sha256:'+suffix*64,'sha256:'+suffix*64))
        value = cur.fetchone()[0]
    conn.commit()
    return value


def _seed(conn, kind='score', event=True):
    prior._seed(conn, {'terminal_status':'judge_error'}, 'judge_error', kind=kind)
    with conn.cursor() as cur:
        cur.execute('SET session_replication_role=replica')
        cur.execute('DELETE FROM public.lab_arena_runs WHERE attempt=2')
        cur.execute("UPDATE public.lab_arena_runs SET status='pending',terminal_cause=NULL,result_doc=NULL,runner_hotkey=NULL,lease_generation=0")
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{lease_ttl_seconds}','4500'::jsonb)")
        cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,lease_generation,stage_generation) SELECT assignment_id||'-sibling:1',assignment_id||'-sibling',round_id,submission_id,miner_hotkey,stage,1,1,kind,'pending',0,stage_generation FROM public.lab_arena_runs")
        cur.execute('SET session_replication_role=origin')
    conn.commit()
    lease = _claim(conn, prior.RUNNER_A, 'a')
    assert lease['status'] == 'leased'
    if event:
        batch = [{'event_id':str(uuid.uuid4()),'kind':'runtime.error',
                  'occurred_at':datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                  'content':{'status':'abandoned','failure_stage':'runtime','error_class':'RuntimeHostError'}}]
        with conn.cursor() as cur:
            cur.execute('SET ROLE lab_arena_service')
            cur.execute('SELECT public.lab_arena_append_trajectory_events_v1(%s,%s,%s::jsonb)',
                        (lease['run_id'],'sha256:'+'a'*64,json.dumps(batch)))
            assert cur.fetchone()[0]['status'] == 'accepted'
            cur.execute('RESET ROLE')
        conn.commit()
    return lease


def _expire(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT public.lab_arena_expire_leases(%s)',(prior.ROUND,))
        value=cur.fetchone()[0]
    conn.commit()
    return value


@pytest.mark.parametrize('kind',['score','execute'])
def test_recovery_hands_off_without_ttl_wait_and_preserves_audit(database,migrated,kind):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        lease=_seed(conn,kind)
        if kind=='score':
            assert _claim(conn,prior.RUNNER_B,'b') == {'status':'no_pending'}
        assert _expire(conn) == {'status':'ok','expired':1,'retried':1}
        assert _expire(conn) == {'status':'ok','expired':0,'retried':0}
        with conn.cursor() as cur:
            cur.execute('SELECT status,terminal_cause,terminal_doc,claim_response,lease_expires_at FROM public.lab_arena_runs WHERE run_id=%s',(lease['run_id'],))
            status,cause,doc,claim,expiry=cur.fetchone()
            assert (status,cause)==('failed','lease_expired')
            assert doc['recovery_reason']=='authenticated_zero_call_runtime_host_error'
            assert datetime.fromisoformat(doc['original_lease_expires_at']) > datetime.now(timezone.utc)
            assert claim == lease
            assert expiry <= datetime.now(timezone.utc)
            cur.execute('SET session_replication_role=replica')
            cur.execute('DELETE FROM public.lab_arena_runs WHERE assignment_id<>%s',(lease['assignment_id'],))
            cur.execute('SET session_replication_role=origin')
        conn.commit()
        if kind=='score':
            assert _claim(conn,prior.RUNNER_A,'c') == {'status':'no_pending'}
        retry=_claim(conn,prior.RUNNER_B,'d')
        assert retry['attempt']==2 and retry['status']=='leased'
        assert retry['run_id']==lease['assignment_id']+':2'


@pytest.mark.parametrize('change',[
 'live','ledger','paid','provider','result','output','wrong_runner','wrong_round',
 'wrong_assignment','wrong_attempt','wrong_stage','wrong_kind','wrong_class',
 'wrong_submission','wrong_miner','wrong_position','wrong_status',
 'setup','cleanup','secondary_cleanup','finished','before_claim','after_expiry','old_generation','other_stage',
])
def test_only_exact_zero_call_current_lease_recovers(database,migrated,change):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        lease=_seed(conn,event=change!='live')
        with conn.cursor() as cur:
            cur.execute('SET session_replication_role=replica')
            if change in ('ledger','paid'):
                cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd) VALUES (%s,%s,%s,%s,%s,1,%s,'deepline','exa.contents','host',%s)",
                            ('settlement' if change=='paid' else 'reservation',prior.MINER,prior.ROUND,prior.SUBMISSION,lease['run_id'],'sha256:'+'f'*64,100 if change=='paid' else 0))
            elif change in ('result','output'):
                cur.execute('UPDATE public.lab_arena_runs SET '+('result_doc=\'{}\'::jsonb' if change=='result' else "output_ref='present'")+' WHERE run_id=%s',(lease['run_id'],))
            elif change=='secondary_cleanup':
                cur.execute("INSERT INTO public.lab_arena_trajectory_events(run_id,event_id,round_id,submission_id,miner_hotkey,runner_hotkey,assignment_id,icp_identifier,stage,icp_position,attempt,run_kind,model_role,event_kind,occurred_at,content) SELECT run_id,gen_random_uuid(),round_id,submission_id,miner_hotkey,runner_hotkey,assignment_id,icp_identifier,stage,icp_position,attempt,run_kind,model_role,'runtime.cleanup_error',now(),'{}'::jsonb FROM public.lab_arena_trajectory_events")
            elif change in ('provider','finished'):
                cur.execute('UPDATE public.lab_arena_trajectory_events SET event_kind=%s',('provider.request' if change=='provider' else 'runtime.finished',))
            elif change in ('wrong_class','wrong_status','setup','cleanup'):
                key,value=({'wrong_class':('error_class','OSError'), 'wrong_status':('status','accepted')}.get(change,('failure_stage',change)))
                cur.execute('UPDATE public.lab_arena_trajectory_events SET content=jsonb_set(content,%s::text[],to_jsonb(%s::text))',([key],value))
            elif change.startswith('wrong_'):
                column,value={'wrong_runner':('runner_hotkey',prior.RUNNER_B),'wrong_round':('round_id','other'),
                  'wrong_assignment':('assignment_id','other'),'wrong_attempt':('attempt',2),
                  'wrong_stage':('stage',2),'wrong_kind':('run_kind','execute'),
                  'wrong_submission':('submission_id','other'), 'wrong_miner':('miner_hotkey',prior.RUNNER_B),
                  'wrong_position':('icp_position',9)}[change]
                cur.execute('UPDATE public.lab_arena_trajectory_events SET '+column+'=%s',(value,))
            elif change=='before_claim':
                cur.execute("UPDATE public.lab_arena_trajectory_events SET occurred_at=now()-interval '1 day'")
            elif change=='after_expiry':
                cur.execute("UPDATE public.lab_arena_trajectory_events SET created_at=now()+interval '1 day'")
            elif change in ('old_generation','other_stage'):
                cur.execute('UPDATE public.lab_arena_runs SET '+('stage_generation=0' if change=='old_generation' else 'stage=2')+' WHERE run_id=%s',(lease['run_id'],))
            cur.execute('SET session_replication_role=origin')
        conn.commit()
        assert _expire(conn) == {'status':'ok','expired':0,'retried':0}
        with conn.cursor() as cur:
            cur.execute('SELECT status,lease_expires_at FROM public.lab_arena_runs WHERE run_id=%s',(lease['run_id'],))
            state,expiry=cur.fetchone()
            assert state=='leased' and expiry>datetime.now(timezone.utc)


def _reserve(conn,lease):
    with conn.cursor() as cur:
        cur.execute("SELECT public.lab_arena_reserve_call(%s,%s,%s,'exa.contents','deepline','host',0,%s::jsonb,4500)",
                    (lease['run_id'],'sha256:'+'a'*64,'sha256:'+'f'*64,json.dumps({'reserve_remaining_budget':True})))
        return cur.fetchone()[0]


@pytest.mark.parametrize('winner',['reservation','recovery'])
def test_reservation_recovery_race_never_discards_provider_work(database,migrated,winner):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        lease=_seed(conn)
    owner=psycopg.connect(**dsn)
    try:
        if winner=='reservation':
            assert _reserve(owner,lease)['status']=='reserved'
        else:
            with owner.cursor() as cur:
                cur.execute('SELECT public.lab_arena_expire_leases(%s)',(prior.ROUND,))
                assert cur.fetchone()[0]['expired']==1
        started=threading.Event()
        def losing_call():
            with psycopg.connect(**dsn) as loser:
                started.set()
                return _expire(loser) if winner=='reservation' else _reserve(loser,lease)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(losing_call)
            assert started.wait(2)
            # Wait until the real second DB connection is blocked by the first.
            with psycopg.connect(**dsn) as observer:
                for _ in range(100):
                    with observer.cursor() as cur:
                        cur.execute("SELECT count(*) FROM pg_stat_activity WHERE wait_event_type='Lock' AND query LIKE 'SELECT public.lab_arena_%'")
                        if cur.fetchone()[0]: break
                    threading.Event().wait(.01)
                else: pytest.fail('race never reached DB lock')
            owner.commit()
            answer=future.result(timeout=5)
        assert answer['expired']==0 if winner=='reservation' else answer['status']=='stale'
        with owner.cursor() as cur:
            cur.execute('SELECT count(*) FROM public.lab_arena_ledger WHERE run_id=%s',(lease['run_id'],))
            assert cur.fetchone()[0]==(1 if winner=='reservation' else 0)
    finally:
        owner.close()


def test_rejected_lease_cannot_authorize_recovery(database,migrated):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        lease=_seed(conn,event=False)
        event={'event_id':str(uuid.uuid4()),'kind':'runtime.error',
               'occurred_at':datetime.now(timezone.utc).isoformat(),
               'content':{'status':'abandoned','failure_stage':'runtime','error_class':'RuntimeHostError'}}
        with conn.cursor() as cur:
            cur.execute('SET ROLE lab_arena_service')
            cur.execute('SELECT public.lab_arena_append_trajectory_events_v1(%s,%s,%s::jsonb)',
                        (lease['run_id'],'sha256:'+'f'*64,json.dumps([event])))
            assert cur.fetchone()[0]['status']=='stale'
            cur.execute('RESET ROLE')
        conn.commit()
        assert _expire(conn)['expired']==0


def test_migration_rejects_acl_drift_even_after_application(database,migrated):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        with conn.cursor() as cur:
            cur.execute('GRANT EXECUTE ON FUNCTION public.lab_arena_expire_leases(text) TO anon')
            with pytest.raises(psycopg.Error,match='security shape differs'):
                cur.execute(SQL.read_text())
        conn.rollback()
