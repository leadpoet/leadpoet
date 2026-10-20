"""A round-quarantined live sibling cannot strand an otherwise lone runner."""
import hashlib
import json
from pathlib import Path
import re
import time

import pytest
from tests.lab_arena import leased_image_setup_recovery515_postgres_test as image

started = image.started
fast = started.fast
database = image.database
original = image.original
restore = image.restore
ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/516-lab-arena-quarantined-runner-alternatives.sql'
PRE = image.CLAIM_POST
POST = 'a1ee8afde0f2f04f6cdd5e9c0cda3d0710c5d9d2b1c72e05e6f604722e7dd345'


def fragments():
    source = SQL.read_text()
    return {name: re.search(r'\$'+name+r'\$(.*?)\$'+name+r'\$', source, re.S)[1]
            for name in ('old_select','new_select','old_gate','new_gate',
                         'old_alternative','new_alternative')}


def apply(conn):
    with conn.cursor() as cur:
        cur.execute(SQL.read_text())
    conn.commit()


def scenario(conn, *, mode='quarantined_active', aged=True, fault=None):
    """Use real allocation, authenticated events, expiry and frozen successors."""
    primary = started.cohort(conn, count=1, extra=7, started=1)[0]
    image.apply(conn)
    b = fast.hotkey('516-alternative-b')
    c = fast.hotkey('516-alternative-c')
    allocations = []
    suffixes = iter('bcdef01')
    for runner, count in ((b, 3 if mode=='quarantined_idle' else 4), (c, 3)):
        for n in range(count):
            suffix = next(suffixes)
            lease = fast.rpc(conn, runner=runner, suffix=suffix)
            assert lease['status']=='leased'
            assert started.event(conn, lease, suffix)['status']=='accepted'
            allocations.append((runner,n,suffix,lease))
    for runner,n,suffix,lease in allocations:
        if mode!='healthy_active' and n<3:
            assert started.event(conn, lease, suffix, kind='runtime.error', content={
                'status':'abandoned','failure_stage':'runtime','error_class':'RuntimeHostError'
            })['status']=='accepted'
    assert started.event(conn, primary, 'a', kind='runtime.error', content={
        'status':'abandoned','failure_stage':'setup','error_class':'LeasedImageError'
    })['status']=='accepted'
    assert started.expire(conn)['expired']==(1 if mode=='healthy_active' else 7)
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE status='pending'")
        if aged:
            cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=lease_expires_at-interval '4505 seconds',claim_response=jsonb_set(claim_response,'{lease_expires_at}',to_jsonb((claim_response->>'lease_expires_at')::timestamptz-interval '4505 seconds')),terminal_doc=jsonb_set(jsonb_set(terminal_doc,'{expired_at}',to_jsonb((terminal_doc->>'expired_at')::timestamptz-interval '4505 seconds')),'{original_lease_expires_at}',to_jsonb((terminal_doc->>'original_lease_expires_at')::timestamptz-interval '4505 seconds')) WHERE run_id=%s", (primary['run_id'],))
            cur.execute("UPDATE public.lab_arena_trajectory_events SET created_at=created_at-interval '4505 seconds',occurred_at=occurred_at-interval '4505 seconds' WHERE run_id=%s", (primary['run_id'],))
        cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='judge_error' WHERE status='pending' AND assignment_id<>%s", (primary['assignment_id'],))
        if fault:
            # One of B's three otherwise qualifying proofs is invalid. The
            # existing claimant and alternative predicates must agree on it.
            target = next(lease['run_id'] for runner,n,_,lease in allocations if runner==b and n==0)
            if fault in ('error_class','failure_stage','status'):
                key,value = {'error_class':('error_class','RunnerError'),
                    'failure_stage':('failure_stage','setup'),'status':('status','other')}[fault]
                cur.execute("UPDATE public.lab_arena_trajectory_events SET content=jsonb_set(content,%s::text[],to_jsonb(%s::text)) WHERE run_id=%s AND event_kind='runtime.error'", ([key],value,target))
            elif fault in ('runner_hotkey','run_kind','event_kind'):
                value = {'runner_hotkey':fast.hotkey('wrong-proof'), 'run_kind':'execute', 'event_kind':'runtime.cleanup_error'}[fault]
                cur.execute('UPDATE public.lab_arena_trajectory_events SET '+fault+"=%s WHERE run_id=%s AND event_kind='runtime.error'", (value,target))
            elif fault=='assignment':
                other = next(lease['assignment_id'] for runner,n,_,lease in allocations if runner==b and n==1)
                cur.execute('UPDATE public.lab_arena_runs SET assignment_id=%s,attempt=99 WHERE run_id=%s', (other,target))
            elif fault=='ledger':
                cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd) SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,%s,'deepline','exa.contents','host',1 FROM public.lab_arena_runs WHERE run_id=%s", ('sha256:'+'f'*64,target))
            elif fault in ('result','output'):
                cur.execute('UPDATE public.lab_arena_runs SET '+("result_doc='{}'::jsonb" if fault=='result' else "output_ref='fixture-present'")+' WHERE run_id=%s', (target,))
            else:
                raise AssertionError(fault)
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    return primary,b,c


def test_exact_source_security_non_superuser_idempotence_and_no_data_change(database):
    pg,dsn = database
    with pg.connect(**dsn) as conn,conn.cursor() as cur:
        started.cohort(conn,count=1,extra=1,started=1)
        image.apply(conn)
        before = fast.definition(cur)
        metadata = fast.metadata(cur)
        rows = image.snapshots(cur)
        expiry = image.definition(cur)
        assert hashlib.sha256(before.encode()).hexdigest()==PRE
        for digest,error in ((PRE,'preimage differs'),(POST,'postimage differs')):
            with pytest.raises(pg.Error,match=error):
                cur.execute(SQL.read_text().replace(digest,'0'*64))
            conn.rollback()
            assert fast.definition(cur)==before and fast.metadata(cur)==metadata
            assert image.snapshots(cur)==rows
        cur.execute('GRANT EXECUTE ON FUNCTION '+fast.SIG+' TO anon')
        with pytest.raises(pg.Error,match='security shape differs'):
            cur.execute(SQL.read_text())
        conn.rollback()
        cur.execute('CREATE ROLE claim516_migrator NOLOGIN NOSUPERUSER INHERIT')
        cur.execute('GRANT lab_arena_owner TO claim516_migrator')
        cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO claim516_migrator')
        cur.execute('ALTER DEFAULT PRIVILEGES FOR ROLE claim516_migrator IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
        conn.commit()
        cur.execute('SET ROLE claim516_migrator')
        apply(conn)
        after = fast.definition(cur)
        apply(conn)
        assert fast.definition(cur)==after and fast.metadata(cur)==metadata
        cur.execute('RESET ROLE')
        expected = before
        parts = fragments()
        for label in ('select','gate','alternative'):
            assert expected.count(parts['old_'+label])==1
            expected = expected.replace(parts['old_'+label],parts['new_'+label])
        assert after==expected and hashlib.sha256(after.encode()).hexdigest()==POST
        assert image.definition(cur)==expiry and image.snapshots(cur)==rows
        for role in ('anon','authenticated','service_role'):
            cur.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')",(role,fast.SIG))
            assert cur.fetchone()==(False,)


@pytest.mark.parametrize('mode,aged,expected',[
    ('quarantined_idle',False,'no_pending'),('quarantined_idle',True,'leased'),
    ('healthy_active',True,'no_pending'),('quarantined_active',False,'no_pending'),
    ('quarantined_active',True,'leased')])
def test_real_expiry_handoff_and_live_alternative_eligibility(database,mode,aged,expected):
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        primary,b,c = scenario(conn,mode=mode,aged=aged)
        # Reproduce the bug against the exact current source before migration.
        before = fast.rpc(conn,suffix='4')
        if mode=='quarantined_active': assert before=={'status':'no_pending'}
        if before['status']=='leased':
            # Preserve a positive preexisting allocation: exact request replay
            # after migration cannot create another run or token.
            apply(conn)
            assert fast.rpc(conn,suffix='4')==before
            return
        cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE status='accepted' ORDER BY run_id")
        accepted = cur.fetchall()
        cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE status='leased' ORDER BY run_id")
        live = cur.fetchall()
        cur.execute('SELECT configuration_doc,participants,stage2_scoring_plan_doc FROM public.lab_arena_rounds')
        frozen = cur.fetchall()
        cur.execute('SELECT scored_run_id,judgment_input_hash,judgment_cache_key,judgment_scope_doc,stage_generation FROM public.lab_arena_runs WHERE run_id=%s',(primary['run_id'],))
        bindings = cur.fetchone()
        apply(conn)
        result = fast.rpc(conn,suffix='5')
        assert result['status']==expected
        if expected=='leased':
            assert result['assignment_id']==primary['assignment_id'] and result['attempt']==2
            assert result['lease_generation']==primary['lease_generation']+1
            assert fast.rpc(conn,suffix='5')==result
            cur.execute('SELECT scored_run_id,judgment_input_hash,judgment_cache_key,judgment_scope_doc,stage_generation FROM public.lab_arena_runs WHERE run_id=%s',(result['run_id'],))
            assert cur.fetchone()==bindings
        cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE status='accepted' ORDER BY run_id")
        assert cur.fetchall()==accepted
        cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE run_id=ANY(%s) ORDER BY run_id",([row[0]['run_id'] for row in live],))
        assert cur.fetchall()==live
        cur.execute('SELECT configuration_doc,participants,stage2_scoring_plan_doc FROM public.lab_arena_rounds')
        assert cur.fetchall()==frozen
        if mode=='healthy_active': assert fast.rpc(conn,runner=b,suffix='6')['status']=='leased'
        else:
            assert fast.rpc(conn,runner=b,suffix='6')=={'status':'no_pending'}
            assert fast.rpc(conn,runner=c,suffix='7')=={'status':'no_pending'}


@pytest.mark.parametrize('fault',['error_class','failure_stage','status','runner_hotkey',
    'run_kind','event_kind','assignment','ledger','result','output'])
def test_existing_quarantine_proof_is_required_to_ignore_an_alternative(database,fault):
    with database[0].connect(**database[1]) as conn:
        _,b,_ = scenario(conn,fault=fault)
        apply(conn)
        # B now has only two qualifying distinct assignments: its live sibling
        # must retain preference; no broad host exclusion is introduced.
        assert fast.rpc(conn,suffix='5')=={'status':'no_pending'}
        # Its independent recent-score-expiry cooldown may still reject B.
        # Verify the shared quarantine predicate itself accepts B, rather than
        # claiming that this separate migration changes its cooldown.
        with conn.cursor() as cur:
            cte=fragments()['new_select'].split('  SELECT * INTO v_run',1)[0]
            cte=cte.replace('v_round.status',"'stage2_scoring'").replace('p_round_id',cur.mogrify('%s',(fast.seed.ROUND,)).decode())
            cur.execute(cte+' SELECT NOT EXISTS(SELECT 1 FROM round_quarantined_runners WHERE runner_hotkey=%s)',(b,))
            assert cur.fetchone()==(True,)


def test_exact_quarantine_predicate_parity_including_unknown_phase(database):
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        _,b,c = scenario(conn)
        parts = fragments()
        old = parts['old_gate'].split('    AND ',1)[1]
        new = parts['new_select'].split('  SELECT * INTO v_run',1)[0]
        for phase in ('stage1','stage2','stage1_scoring','stage1_judged','stage1_scored',
                      'stage2_closed','stage2_scoring','stage2_judged','scored','completed',None):
            for runner in (fast.seed.RUNNER,b,c,fast.hotkey('never-faulted')):
                def bind(sql):
                    return sql.replace('v_round.status',cur.mogrify('%s::text',(phase,)).decode()).replace('p_round_id',cur.mogrify('%s',(fast.seed.ROUND,)).decode()).replace('p_runner_hotkey',cur.mogrify('%s',(runner,)).decode())
                cur.execute('SELECT ('+bind(old)+') IS TRUE')
                expected = cur.fetchone()[0]
                cur.execute(bind(new)+" SELECT NOT EXISTS(SELECT 1 FROM round_quarantined_runners WHERE runner_hotkey=%s)",(runner,))
                assert cur.fetchone()[0]==expected


def nodes(plan):
    yield plan
    for child in plan.get('Plans',[]):
        yield from nodes(child)


def actual_claim_plan(conn, suffix):
    """auto_explain observes the real PL/pgSQL Params and exact locking query.

    This is local disposable PostgreSQL only. The entire allocation is rolled
    back to a savepoint, including trigger/reservation writes.
    """
    from collections import deque
    with conn.cursor() as cur:
        cur.execute("LOAD 'auto_explain'")
        for key,value in [('log_min_duration','0'),('log_analyze','true'),
            ('log_nested_statements','true'),('log_format','json'),
            ('log_level','notice'),('log_timing','false')]:
            cur.execute('SET auto_explain.'+key+'=%s',(value,))
        cur.execute('SET client_min_messages=notice')
        cur.execute('SAVEPOINT claim_plan')
        conn.notices = deque()
        began = time.monotonic()
        cur.execute('SET LOCAL ROLE lab_arena_service')
        cur.execute('SELECT public.lab_arena_claim_assignment(%s,%s,7,20,%s,%s,%s,%s,4500)',
            (fast.seed.ROUND,fast.seed.RUNNER,[],suffix*32,'sha256:'+suffix*64,'sha256:'+suffix*64))
        result = cur.fetchone()[0]
        elapsed = time.monotonic()-began
        plans = []
        for notice in conn.notices:
            if 'plan:\n' in notice:
                plan = json.loads(notice.split('plan:\n',1)[1])
                if 'FROM public.lab_arena_runs AS runs' in plan.get('Query Text',''):
                    plan['duration_ms'] = float(re.search(r'duration: ([0-9.]+) ms',notice)[1])
                    plans.append(plan)
        cur.execute('ROLLBACK TO SAVEPOINT claim_plan')
        cur.execute('RESET auto_explain.log_min_duration')
        assert len(plans)==1
        return result,plans[0],elapsed


@pytest.mark.parametrize('case',['idle','healthy','recovery'])
def test_actual_nested_plan_no_unconditional_idle_history_scan(database,case):
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        primary,b,c = scenario(conn,mode='healthy_active' if case=='healthy' else 'quarantined_active')
        if case=='healthy':
            # Retain four live healthy B leases, and real historical C faults.
            # This tests a healthy alternative with a nonempty quarantine set.
            cur.execute("SELECT claim_response FROM public.lab_arena_runs WHERE runner_hotkey=%s AND status='leased'",(c,))
            for (lease,) in cur.fetchall():
                assert started.event(conn,lease,lease['request_id'][0],kind='runtime.error',content={
                    'status':'abandoned','failure_stage':'runtime','error_class':'RuntimeHostError'
                })['status']=='accepted'
            assert started.expire(conn)['expired']==3
        # 320 real round rows approximate accepted and historical attempts;
        # auto_explain observes the real function's Params, not a round JOIN.
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,to_jsonb(r)||jsonb_build_object('run_id','history516-'||n,'assignment_id','history516-'||n))).* FROM (SELECT * FROM public.lab_arena_runs WHERE kind='execute' AND status='accepted' LIMIT 1) r CROSS JOIN generate_series(1,320) n")
        cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='judge_error' WHERE status='pending' AND (%s OR assignment_id<>%s)",(case=='idle',primary['assignment_id']))
        cur.execute('SET LOCAL session_replication_role=origin')
        cur.execute('ANALYZE public.lab_arena_runs')
        conn.commit()
        # Compare warm nested executor plans; whole-RPC timings include local
        # instrumentation and are not a production allocation latency claim.
        actual_claim_plan(conn,'8')
        old_result,old_plan,old_elapsed = actual_claim_plan(conn,'8')
        apply(conn)
        guarded_source = fast.definition(cur)
        demand = """      -- An idle claim does not need a fault-history scan. This cheap indexed
      -- demand gate shares the candidate statement's snapshot.
      AND EXISTS (SELECT 1 FROM public.lab_arena_runs AS pending_fault_check
        WHERE pending_fault_check.round_id=p_round_id
          AND pending_fault_check.status='pending')
"""
        assert guarded_source.count(demand)==1
        # Preserve a reproducible control for the real performance regression
        # found while designing the CTE. This source is never a release target.
        cur.execute(guarded_source.replace(demand,''))
        conn.commit()
        actual_claim_plan(conn,'9')
        _,ungated_plan,_ = actual_claim_plan(conn,'9')
        cur.execute(guarded_source)
        conn.commit()
        actual_claim_plan(conn,'2')
        result,plan,elapsed = actual_claim_plan(conn,'2')
        assert old_result['status']=='no_pending'
        assert result['status']==('leased' if case=='recovery' else 'no_pending')
        cte = [n for n in nodes(plan['Plan']) if n.get('Subplan Name')=='CTE round_quarantined_runners']
        assert len(cte)==1 and cte[0]['Actual Loops']<=1
        history = [n for n in nodes(cte[0]) if n.get('Alias')=='host_fault']
        assert history
        if case=='idle':
            assert all(n['Actual Loops']==0 for n in history)
            assert any(n.get('Alias')=='host_fault' and n['Actual Loops']>0 for n in nodes(ungated_plan['Plan']))
        # The materialized producer runs once even when each consumer is used.
        # Its inner event-driven index probe may run once per qualifying fault.
        scans = [n for n in nodes(plan['Plan']) if n.get('CTE Name')=='round_quarantined_runners']
        assert len(scans)==2
        import os
        evidence = os.environ.get('ARENA_516_PLAN_EVIDENCE')
        if evidence:
            path = Path(evidence)/('claim516-plan-'+case+'-fault-history.json')
            cur.execute('SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s',(fast.seed.ROUND,))
            round_rows=cur.fetchone()[0]
            path.write_text(json.dumps({'case':case,'round_rows':round_rows,
                'original':old_plan,'unguarded_design_control':ungated_plan,'candidate':plan,
                'original_rpc_seconds':old_elapsed,'candidate_rpc_seconds':elapsed},indent=2)+'\n')
            path.chmod(0o600)
