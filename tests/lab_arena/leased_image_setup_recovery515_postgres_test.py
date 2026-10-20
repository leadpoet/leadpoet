"""Image access setup failures reuse the authenticated zero-call expiry path."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from tests.lab_arena import score_expiry_start_proof511_postgres_test as started
from tests.lab_arena import zero_call_setup_recovery467_postgres_test as setup

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/515-lab-arena-zero-call-leased-image-recovery.sql'
SIG = 'public.lab_arena_expire_leases(text)'
PRE = '7835213f7d4991be60194c60589da3d18f62f05f8640136e7df2dfce1d8e256f'
POST = '7c70db8f77d1552c315a1ef6ab4d0ad325f0a93850bcb3d302424858680ad002'
OLD = "host_error.content ->> 'error_class' = 'RunnerError'"
NEW = "host_error.content ->> 'error_class' IN ('RunnerError','LeasedImageError')"
CLAIM = started.fast.SIG
CLAIM_POST = '584a2b835f0d74ede019f4076b7000c9a3e8e681342e5e6d356c7fb90c8c7ad1'
HANDOFF_OLD = '          AND prior_score.lease_expires_at<=pg_catalog.clock_timestamp()'
HANDOFF_NEW = """          AND prior_score.lease_expires_at<=pg_catalog.clock_timestamp()
          AND (prior_score.lease_expires_at>
            pg_catalog.clock_timestamp()-pg_catalog.make_interval(
              secs => (v_round.configuration_doc->>'lease_ttl_seconds')::INTEGER)
            OR EXISTS (
              SELECT 1 FROM public.lab_arena_trajectory_events AS host_error
              WHERE host_error.run_id=prior_score.run_id
                AND (host_error.runner_hotkey,host_error.round_id,host_error.submission_id,
                  host_error.miner_hotkey,host_error.assignment_id,host_error.stage,
                  host_error.icp_position,host_error.attempt,host_error.run_kind)
                  IS NOT DISTINCT FROM
                  (prior_score.runner_hotkey,prior_score.round_id,prior_score.submission_id,
                   prior_score.miner_hotkey,prior_score.assignment_id,prior_score.stage,
                   prior_score.icp_position,prior_score.attempt,prior_score.kind)
                AND host_error.event_kind='runtime.error'
                AND host_error.content->>'status'='abandoned'
                AND host_error.content->>'failure_stage'='runtime'
                AND host_error.content->>'error_class'='RuntimeHostError'
                AND host_error.created_at BETWEEN
                  (prior_score.claim_response->>'lease_expires_at')::TIMESTAMPTZ
                    -pg_catalog.make_interval(secs=>(v_round.configuration_doc->>'lease_ttl_seconds')::INTEGER)
                  AND COALESCE((prior_score.terminal_doc->>'original_lease_expires_at')::TIMESTAMPTZ,
                    prior_score.lease_expires_at)
                AND host_error.occurred_at BETWEEN pg_catalog.date_trunc('second',
                  (prior_score.claim_response->>'lease_expires_at')::TIMESTAMPTZ
                    -pg_catalog.make_interval(secs=>(v_round.configuration_doc->>'lease_ttl_seconds')::INTEGER))
                  AND COALESCE((prior_score.terminal_doc->>'original_lease_expires_at')::TIMESTAMPTZ,
                    prior_score.lease_expires_at)
            ))"""


def definition(cur):
    return started.fast.definition(cur, SIG)


def metadata(cur):
    cur.execute("SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE oid=%s::regprocedure", (SIG,))
    return cur.fetchone()[0]


@pytest.fixture(scope='module')
def database():
    gen = started.database.__wrapped__()
    db = next(gen)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            for number in (511, 513, 514):
                paths = list((ROOT / 'scripts').glob(str(number)+'-*.sql'))
                assert len(paths) == 1
                cur.execute(paths[0].read_text())
            assert hashlib.sha256(definition(cur).encode()).hexdigest() == PRE
        yield db
    finally:
        gen.close()


@pytest.fixture(scope='module')
def original(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        return {SIG: definition(cur), CLAIM: started.fast.definition(cur)}


@pytest.fixture(autouse=True)
def restore(database, original):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        for source in original.values():
            cur.execute(source)
        # Trajectory rows intentionally have no cascading round deletion.
        # Each negative must start from its own otherwise eligible proof.
        cur.execute('TRUNCATE public.lab_arena_ledger,public.lab_arena_trajectory_events RESTART IDENTITY CASCADE')
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE,pause_reason='',actor_ref='',guard_commitment='',owner_commitment='',guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]' WHERE singleton")
    yield
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        for source in original.values():
            cur.execute(source)


def apply(conn):
    with conn.cursor() as cur:
        cur.execute(SQL.read_text())
    conn.commit()


def seed(conn, *, error_class='LeasedImageError', emit_start=True, emit_error=True):
    lease = started.cohort(conn, count=1, extra=0, started=1 if emit_start else 0)[0]
    if emit_error:
        assert started.event(conn, lease, 'a', kind='runtime.error', content={
            'status':'abandoned', 'failure_stage':'setup', 'error_class':error_class
        })['status'] == 'accepted'
    return lease


def snapshots(cur):
    result = {}
    for table in ('lab_arena_runs', 'lab_arena_rounds', 'lab_arena_trajectory_events',
                  'lab_arena_ledger', 'lab_arena_restart_claim_control'):
        cur.execute('SELECT to_jsonb(t) FROM public.'+table+' t ORDER BY to_jsonb(t)::text')
        result[table] = cur.fetchall()
    return result


def test_exact_migration_only_two_predicates_security_idempotence_and_rows(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        seed(conn)
        before = definition(cur)
        claim_before = started.fast.definition(cur)
        claim_identity = started.fast.metadata(cur)
        identity = metadata(cur)
        rows = snapshots(cur)
        assert before.count(OLD) == 2
        assert started.expire(conn) == {'status':'ok','expired':0,'retried':0}
        apply(conn)
        assert definition(cur) == before.replace(OLD, NEW)
        assert started.fast.definition(cur) == claim_before.replace(HANDOFF_OLD,HANDOFF_NEW)
        assert hashlib.sha256(started.fast.definition(cur).encode()).hexdigest()==CLAIM_POST
        assert started.fast.metadata(cur)==claim_identity
        assert hashlib.sha256(definition(cur).encode()).hexdigest() == POST
        assert metadata(cur) == identity and snapshots(cur) == rows
        apply(conn)
        assert metadata(cur) == identity and snapshots(cur) == rows


@pytest.mark.parametrize('error_class', ['LeasedImageError', 'RunnerError'])
def test_authenticated_image_or_existing_setup_error_gets_frozen_successor(database, error_class):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        lease = seed(conn, error_class=error_class)
        cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE kind='execute' AND status='accepted'")
        source = cur.fetchall()
        cur.execute('SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s', (started.fast.seed.ROUND,))
        frozen = cur.fetchone()[0]
        apply(conn)
        assert started.expire(conn) == {'status':'ok','expired':1,'retried':1}
        assert started.expire(conn) == {'status':'ok','expired':0,'retried':0}
        cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE run_id=%s', (lease['run_id'],))
        old = cur.fetchone()[0]
        assert old['status'] == 'failed' and old['terminal_cause'] == 'lease_expired'
        assert old['claim_response'] == lease
        assert old['result_doc'] is None and old['output_ref'] is None
        assert old['terminal_doc']['recovery_reason'] == 'authenticated_zero_call_setup_host_error'
        assert datetime.fromisoformat(old['terminal_doc']['original_lease_expires_at']) > datetime.now(timezone.utc)
        # Simulate the existing retry backoff; production timestamps are never edited.
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE status='pending'")
        cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        # The existing started-expiry handoff still excludes this runner.
        # This migration does not weaken that separate host inference policy.
        assert started.fast.rpc(conn, suffix='b') == {'status':'no_pending'}
        retry = started.fast.rpc(conn, suffix='c', runner=started.fast.hotkey('image515-alternate'))
        assert retry['status'] == 'leased' and retry['attempt'] == 2
        cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE run_id=%s', (retry['run_id'],))
        successor = cur.fetchone()[0]
        for field in ('round_id','submission_id','miner_hotkey','assignment_id','stage',
                      'stage_generation','icp_position','kind','scored_run_id',
                      'judgment_input_hash','judgment_cache_key','judgment_scope_doc'):
            assert successor[field] == old[field]
        assert successor['lease_generation'] == old['lease_generation'] + 1
        cur.execute("SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE kind='execute' AND status='accepted'")
        assert cur.fetchall() == source
        cur.execute('SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s', (started.fast.seed.ROUND,))
        assert cur.fetchone()[0] == frozen
        cur.execute('SELECT count(*) FROM public.lab_arena_ledger')
        assert cur.fetchone()[0] == 0


@pytest.mark.parametrize('fault', [
    'live','nostart','wrong_start_generation','wrong_claim_generation','wrong_start_runner',
    'wrong_runner','wrong_round','wrong_submission','wrong_miner','wrong_assignment',
    'wrong_stage','wrong_position','wrong_attempt','wrong_kind','wrong_class','runtime_stage',
    'cleanup_stage','wrong_status','ledger','paid','provider','output','result','cleanup_event',
    'finished','before_claim','after_expiry','old_generation','other_stage','renewed',
])
def test_image_setup_keeps_every_existing_identity_provider_and_lease_fence(database, fault):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        lease = seed(conn, emit_start=fault!='nostart', emit_error=fault!='live')
        apply(conn)
        cur.execute('SET LOCAL session_replication_role=replica')
        if fault in ('ledger','paid'):
            cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd) SELECT %s,miner_hotkey,round_id,submission_id,run_id,stage,%s,'deepline','exa.contents','host',%s FROM public.lab_arena_runs WHERE run_id=%s", ('settlement' if fault=='paid' else 'reservation','sha256:'+'f'*64,100 if fault=='paid' else 0,lease['run_id']))
        elif fault in ('output','result','old_generation','other_stage','wrong_claim_generation','renewed'):
            mutation = {'output':"output_ref='present'",'result':"result_doc='{}'::jsonb",'old_generation':'stage_generation=0','other_stage':'stage=1','wrong_claim_generation':"claim_response=jsonb_set(claim_response,'{lease_generation}','99'::jsonb)",'renewed':"lease_expires_at=lease_expires_at+interval '1 minute'"}[fault]
            cur.execute('UPDATE public.lab_arena_runs SET '+mutation+' WHERE run_id=%s',(lease['run_id'],))
        elif fault in ('cleanup_event','provider','finished'):
            kind={'cleanup_event':'runtime.cleanup_error','provider':'provider.request','finished':'runtime.finished'}[fault]
            cur.execute("UPDATE public.lab_arena_trajectory_events SET event_kind=%s WHERE event_kind='runtime.error'",(kind,))
        elif fault in ('wrong_class','runtime_stage','cleanup_stage','wrong_status'):
            key,value={'wrong_class':('error_class','OtherError'),'runtime_stage':('failure_stage','runtime'),'cleanup_stage':('failure_stage','cleanup'),'wrong_status':('status','accepted')}[fault]
            cur.execute("UPDATE public.lab_arena_trajectory_events SET content=jsonb_set(content,%s::text[],to_jsonb(%s::text)) WHERE event_kind='runtime.error'",([key],value))
        elif fault=='wrong_start_generation':
            cur.execute("UPDATE public.lab_arena_trajectory_events SET content=jsonb_set(content,'{lease_generation}','99'::jsonb) WHERE event_kind='runtime.started'")
        elif fault=='wrong_start_runner':
            cur.execute("UPDATE public.lab_arena_trajectory_events SET runner_hotkey=%s WHERE event_kind='runtime.started'",(started.fast.hotkey('wrong-start'),))
        elif fault.startswith('wrong_'):
            column,value={'wrong_runner':('runner_hotkey',started.fast.hotkey('wrong-runner')),'wrong_round':('round_id','other'),'wrong_submission':('submission_id','other'),'wrong_miner':('miner_hotkey',started.fast.hotkey('wrong-miner')),'wrong_assignment':('assignment_id','other'),'wrong_stage':('stage',1),'wrong_position':('icp_position',9),'wrong_attempt':('attempt',9),'wrong_kind':('run_kind','execute')}[fault]
            cur.execute('UPDATE public.lab_arena_trajectory_events SET '+column+"=%s WHERE event_kind='runtime.error'",(value,))
        elif fault=='before_claim':
            cur.execute("UPDATE public.lab_arena_trajectory_events SET occurred_at=now()-interval '1 day' WHERE event_kind='runtime.error'")
        elif fault=='after_expiry':
            cur.execute("UPDATE public.lab_arena_trajectory_events SET created_at=now()+interval '1 day' WHERE event_kind='runtime.error'")
        cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert started.expire(conn)=={'status':'ok','expired':0,'retried':0}
        cur.execute('SELECT status FROM public.lab_arena_runs WHERE run_id=%s',(lease['run_id'],))
        assert cur.fetchone()==('leased',)


@pytest.mark.parametrize('hold', ['operator','guard'])
def test_image_setup_expiry_cannot_bypass_operator_or_canonical_guard(database, hold):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        lease = seed(conn)
        apply(conn)
        if hold=='guard':
            generation = setup.deployed.prior.prior._hold_capture(conn)
            before = setup.deployed.guard._quiescence(conn, generation=generation)
            assert before['still_leased_count']==1 and before['lost_or_mutated_count']==0
        else:
            cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=TRUE WHERE singleton')
            conn.commit()
        before = snapshots(cur)
        assert started.expire(conn)=={'status':'ok','expired':0,'retried':0}
        assert snapshots(cur)==before
        cur.execute('SELECT status,claim_response FROM public.lab_arena_runs WHERE run_id=%s',(lease['run_id'],))
        assert cur.fetchone()==('leased',lease)


def test_explicit_runtime_host_error_keeps_original_recovery_classification(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        lease = seed(conn, emit_error=False)
        assert started.event(conn,lease,'a',kind='runtime.error',content={
            'status':'abandoned','failure_stage':'runtime','error_class':'RuntimeHostError'
        })['status']=='accepted'
        cur.execute('SAVEPOINT original_host_expiry')
        cur.execute('SELECT public.lab_arena_expire_leases(%s)',(started.fast.seed.ROUND,))
        before = cur.fetchone()[0]
        cur.execute('ROLLBACK TO SAVEPOINT original_host_expiry')
        apply(conn)
        assert started.expire(conn)==before=={'status':'ok','expired':1,'retried':1}
        cur.execute('SELECT terminal_doc FROM public.lab_arena_runs WHERE run_id=%s',(lease['run_id'],))
        assert cur.fetchone()[0]['recovery_reason']=='authenticated_zero_call_runtime_host_error'


def test_wrong_lease_token_cannot_create_image_setup_recovery_authority(database):
    with database[0].connect(**database[1]) as conn:
        lease = seed(conn, emit_error=False)
        assert started.event(conn,lease,'f',kind='runtime.error',content={
            'status':'abandoned','failure_stage':'setup','error_class':'LeasedImageError'
        })['status']=='stale'
        apply(conn)
        assert started.expire(conn)=={'status':'ok','expired':0,'retried':0}


@pytest.mark.parametrize('installed',[False,True])
@pytest.mark.parametrize('fault',['body','acl','owner','path','security','volatility','missing'])
def test_changed_source_or_security_is_rejected_before_apply_or_replay(database, installed, fault):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        seed(conn)
        if installed:
            apply(conn)
        before = definition(cur)
        if fault=='body':
            cur.execute(before.replace('-- lab_arena_abandoned_host_recovery_v1:', '-- concurrent migration source:'))
        else:
            command={'acl':'GRANT EXECUTE ON FUNCTION '+SIG+' TO anon',
                'owner':'ALTER FUNCTION '+SIG+' OWNER TO postgres',
                'path':'ALTER FUNCTION '+SIG+' SET search_path=public',
                'security':'ALTER FUNCTION '+SIG+' SECURITY INVOKER',
                'volatility':'ALTER FUNCTION '+SIG+' STABLE',
                'missing':'DROP FUNCTION '+SIG}[fault]
            cur.execute(command)
        with pytest.raises(database[0].Error,match='preimage differs' if fault=='body' else 'security shape differs'):
            cur.execute(SQL.read_text())
        conn.rollback()
        assert definition(cur)==before


def test_non_superuser_owner_member_can_apply_and_replay_without_acl_drift(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        seed(conn)
        before = metadata(cur)
        cur.execute('CREATE ROLE image515_migrator NOLOGIN NOSUPERUSER INHERIT')
        cur.execute('GRANT lab_arena_owner TO image515_migrator')
        cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO image515_migrator')
        cur.execute('ALTER DEFAULT PRIVILEGES FOR ROLE image515_migrator IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
        conn.commit()
        for _ in range(2):
            cur.execute('SET ROLE image515_migrator')
            cur.execute(SQL.read_text())
            cur.execute('RESET ROLE')
            assert metadata(cur)==before
            assert hashlib.sha256(definition(cur).encode()).hexdigest()==POST
        conn.commit()


@pytest.mark.parametrize('age_seconds,expected',[(4490,'no_pending'),(4500,'leased'),(4510,'leased')])
def test_inferred_handoff_ages_out_at_frozen_ttl_without_changing_source(database, age_seconds, expected):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        lease = seed(conn)
        apply(conn)
        assert started.expire(conn)['retried']==1
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE status='pending'")
        cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-make_interval(secs=>%s) WHERE run_id=%s",(age_seconds,lease['run_id']))
        cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        retry = started.fast.rpc(conn,suffix='b')
        assert retry['status']==expected
        if expected=='leased':
            assert retry['attempt']==2 and retry['lease_generation']==2


def test_three_distinct_inferred_expiries_keep_existing_within_window_cooldown(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        started.cohort(conn,count=3,extra=1,started=3)
        apply(conn)
        started.expire_clock(conn)
        assert started.expire(conn)['expired']==3
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE status='pending'")
        cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert started.fast.rpc(conn,suffix='e')=={'status':'no_pending'}


def test_three_explicit_runtime_host_faults_still_quarantine_after_ttl(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        started.cohort(conn,count=3,extra=1,started=3,explicit_errors=True)
        apply(conn)
        started.expire_clock(conn)
        assert started.expire(conn)['expired']==3
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '76 minutes' WHERE status='failed'")
        cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE status='pending'")
        cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert started.fast.rpc(conn,suffix='e')=={'status':'no_pending'}


def test_single_explicit_runtime_host_fault_keeps_handoff_ban_after_ttl(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        lease = seed(conn,emit_error=False)
        assert started.event(conn,lease,'a',kind='runtime.error',content={
            'status':'abandoned','failure_stage':'runtime','error_class':'RuntimeHostError'
        })['status']=='accepted'
        apply(conn)
        assert started.expire(conn)['retried']==1
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE status='pending'")
        cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '76 minutes' WHERE run_id=%s",(lease['run_id'],))
        cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert started.fast.rpc(conn,suffix='b')=={'status':'no_pending'}


def test_aged_inferred_failure_keeps_other_active_runner_preference(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        lease = started.cohort(conn,count=1,extra=1,started=1)[0]
        other = started.fast.hotkey('image515-other-active')
        active = started.fast.rpc(conn,suffix='b',runner=other)
        assert active['status']=='leased' and active['run_id']!=lease['run_id']
        assert started.event(conn,lease,'a',kind='runtime.error',content={
            'status':'abandoned','failure_stage':'setup','error_class':'LeasedImageError'
        })['status']=='accepted'
        apply(conn)
        assert started.expire(conn)['retried']==1
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '76 minutes' WHERE run_id=%s",(lease['run_id'],))
        cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE status='pending'")
        cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert started.fast.rpc(conn,suffix='c')=={'status':'no_pending'}
        retry = started.fast.rpc(conn,suffix='d',runner=other)
        assert retry['status']=='leased' and retry['attempt']==2


@pytest.mark.parametrize('fence',['retry_backoff','scoring_close','operator','guard'])
def test_aged_handoff_never_bypasses_existing_claim_fences(database,fence):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        lease = seed(conn)
        apply(conn)
        assert started.expire(conn)['retried']==1
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '76 minutes' WHERE run_id=%s",(lease['run_id'],))
        if fence!='retry_backoff':
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE status='pending'")
        if fence=='scoring_close':
            cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}',to_jsonb((clock_timestamp()-interval '1 second')::text))")
            # Valid accepted-source infrastructure continuations remain legal
            # after close. This negative has no such canonical source binding.
            cur.execute("UPDATE public.lab_arena_runs SET scored_run_id=NULL WHERE status='pending'")
        if fence=='operator':
            cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=TRUE WHERE singleton')
        cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        if fence=='guard':
            setup.deployed.prior.prior._hold_capture(conn)
        assert started.fast.rpc(conn,suffix='b')['status']!='leased'


def test_changed_claim_preimage_rolls_back_both_atomic_function_updates(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        seed(conn)
        expiry = definition(cur)
        claim = started.fast.definition(cur)
        cur.execute(claim.replace('-- lab_arena_zero_call_score_expiry_handoff_v1','-- concurrent claim migration drift'))
        conn.commit()
        with pytest.raises(database[0].Error,match='Bounded score handoff preimage differs'):
            cur.execute(SQL.read_text())
        conn.rollback()
        assert definition(cur)==expiry
        assert started.fast.definition(cur)==claim.replace('-- lab_arena_zero_call_score_expiry_handoff_v1','-- concurrent claim migration drift')


@pytest.mark.parametrize('fault',[
    'run_id','runner_hotkey','round_id','submission_id','miner_hotkey','assignment_id',
    'stage','icp_position','attempt','run_kind','event_kind','status','failure_stage',
    'error_class','created_early','created_late','occurred_early','occurred_late',
    'wrong_start_generation',
])
def test_unbound_or_stale_explicit_fault_cannot_extend_aged_inferred_ban(database,fault):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        lease = seed(conn,emit_error=False)
        assert started.event(conn,lease,'a',kind='runtime.error',content={
            'status':'abandoned','failure_stage':'runtime','error_class':'RuntimeHostError'
        })['status']=='accepted'
        apply(conn)
        assert started.expire(conn)['retried']==1
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE status='pending'")
        cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '76 minutes' WHERE run_id=%s",(lease['run_id'],))
        text_fields={'run_id':'unrelated-run','runner_hotkey':started.fast.hotkey('wrong-host-error'),
            'round_id':'wrong-round','submission_id':'wrong-submission','miner_hotkey':started.fast.hotkey('wrong-miner'),
            'assignment_id':'wrong-assignment','run_kind':'execute','event_kind':'runtime.other'}
        if fault in text_fields:
            cur.execute('UPDATE public.lab_arena_trajectory_events SET '+fault+"=%s WHERE event_kind='runtime.error'",(text_fields[fault],))
        elif fault in ('stage','icp_position','attempt'):
            cur.execute('UPDATE public.lab_arena_trajectory_events SET '+fault+'='+fault+"+1 WHERE event_kind='runtime.error'")
        elif fault in ('status','failure_stage','error_class'):
            value={'status':'accepted','failure_stage':'setup','error_class':'LeasedImageError'}[fault]
            cur.execute("UPDATE public.lab_arena_trajectory_events SET content=jsonb_set(content,%s::text[],to_jsonb(%s::text)) WHERE event_kind='runtime.error'",([fault],value))
        elif fault=='wrong_start_generation':
            cur.execute("UPDATE public.lab_arena_trajectory_events SET content=jsonb_set(content,'{lease_generation}','999'::jsonb) WHERE event_kind='runtime.started'")
        else:
            column='created_at' if fault.startswith('created') else 'occurred_at'
            offset=-4501 if fault.endswith('early') else 1
            cur.execute('UPDATE public.lab_arena_trajectory_events e SET '+column+"=(r.claim_response->>'lease_expires_at')::timestamptz+make_interval(secs=>%s) FROM public.lab_arena_runs r WHERE e.run_id=r.run_id AND e.event_kind='runtime.error'",(offset,))
        cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert started.fast.rpc(conn,suffix='b')['status']=='leased'
