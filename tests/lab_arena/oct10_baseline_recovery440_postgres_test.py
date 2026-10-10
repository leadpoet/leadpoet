"""Disposable PostgreSQL checks for the exact Oct10 baseline recovery."""
import json
from pathlib import Path

import pytest

from lab_arena import contracts, contact_policy, provider_observations, scoring
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena import oct07_cutoff_recovery419_postgres_test as prior
from tests.lab_arena.sep24_position7_startup_retry358_postgres_test import _configuration
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

ROOT=Path(__file__).parents[2]
MIGRATION=ROOT/'scripts/440-arena-2026-10-10-baseline-recovery.sql'
ROUND='arena-2026-10-10'
BASE='baseline-2026-10-10'
ARCHIVE=ROUND+'-r440archive'
NEW_DIGEST='sha256:3be7e227ea62eddcdd6871af28bbacd8623db929f5fe393769bc98dad34a35ba'
HASHES={
 'a0e29abee1790aa5d525a2145c59b4272695fbd7bc80bf098b4d90993ecc31a3':"(SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-10')",
 'e272d0d25813b1d4c60f62cee8753c96d7f58b93660ea3086a46d03fd252acb8':"(SELECT participants FROM public.lab_arena_rounds WHERE round_id='arena-2026-10-10')",
 'f4608f08670b523fccc7de0708c7c24208dfb3afbfb3ad5f0fda6afe137a5e4a':"(SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) FROM public.lab_arena_submissions s WHERE round_id='arena-2026-10-10')",
 '6fd4541aa2858cb1413351ffc5227e72d1b244d21bbac83e44ecc2f64a4dcb45':"(SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r WHERE round_id='arena-2026-10-10')",
 '30b673c4c32d2ef3716701aa65d543b70cc87c6414fe281a45d6fb7e8d84dfb8':"(SELECT icps FROM public.qualification_private_icp_sets WHERE set_id=20261009)",
}

@pytest.fixture(scope='module')
def database():
    for base in prior.base_database.__wrapped__():
        yield prior.database.__wrapped__(base)

@pytest.fixture
def seeded(database):
    psycopg,dsn=database
    with psycopg.connect(**dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_get_functiondef('public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure)")
            original_function=cur.fetchone()[0]
            _seed(cur)
    yield database
    with psycopg.connect(**dsn) as conn,conn.cursor() as cur:
        cur.execute('SET session_replication_role=replica')
        for table in ('lab_arena_trajectory_events','lab_arena_ledger','lab_arena_runs','lab_arena_submissions','lab_arena_rounds'):
            cur.execute(f'DELETE FROM public.{table} WHERE round_id IN (%s,%s)',(ROUND,ARCHIVE))
        cur.execute('DELETE FROM public.qualification_private_icp_sets WHERE set_id=20261009')
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='' WHERE singleton")
        cur.execute('SET session_replication_role=origin')
        cur.execute(original_function)


def _seed(cur):
    cur.execute('SET session_replication_role=replica')
    config=json.loads(json.dumps(_configuration()).replace('2026-09-24','2026-10-10').replace('2026-09-23','2026-10-09'))
    config['round_id']=ROUND
    config['baseline_hotkey']=hotkey(BASE)
    config['checkpoint_deadline_policy']='atomic_checkpoint_60m_v1'
    config['lease_ttl_seconds']=4500
    participants=[]
    for index in range(70):
        baseline=index==0
        sid=BASE if baseline else f'miner440-{index}'
        participant={'submission_id':sid,'miner_hotkey':hotkey(sid),'is_king':baseline,
                     'source_ref':f'arena/{ROUND}/sources/{sid}.tar.gz','source_size_bytes':1000}
        participants.append(participant)
    cur.execute("INSERT INTO public.qualification_private_icp_sets (set_id,icps,active_from,active_until,is_active) VALUES (20261009,%s,'2026-10-09','2026-10-10',false)",(json.dumps(daily_icps()[:10]),))
    cur.execute("INSERT INTO public.lab_arena_rounds (round_id,status,status_generation,stage_generation,configuration_doc,rewards_enabled,participants,benchmark_ref,evaluation_date,icp_set_date,champion_funding_frozen) VALUES (%s,'stage1_judged',5,4,%s,true,%s,%s,'2026-10-10','2026-10-09',true)",(ROUND,json.dumps(config),json.dumps(participants),f'arena/{ROUND}/benchmark.json'))
    for p in participants:
        cur.execute("INSERT INTO public.lab_arena_submissions (submission_id,round_id,miner_hotkey,status,is_king,source_ref,source_size_bytes,submission_doc,owner_coldkey,owner_block_number,owner_block_hash) VALUES (%s,%s,%s,'frozen',%s,%s,1000,%s,%s,1,%s)",(p['submission_id'],ROUND,p['miner_hotkey'],p['is_king'],p['source_ref'],json.dumps(p),hotkey('owner440'),'0x'+'a'*64))
    for pos in range(10):
        assignment=f'{ROUND}:{BASE}:1:{pos}'
        for attempt in (1,2):
            accepted=attempt==2 and pos not in (2,5,7,8)
            cause='accepted' if accepted else ('model_error' if pos==8 and attempt==1 else 'lease_expired')
            scored=attempt==2 and (accepted or pos==8)
            cur.execute("INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,terminal_cause,terminal_doc,result_doc,output_ref,stage_generation,runner_hotkey,per_icp_score,qualification_doc) VALUES (%s,%s,%s,%s,%s,1,%s,%s,'execute',%s,%s,%s,%s,%s,1,%s,%s,%s)",(assignment+f':{attempt}',assignment,ROUND,BASE,hotkey(BASE),pos,attempt,'accepted' if accepted else 'failed',cause,json.dumps({'infrastructure_incomplete':True}) if attempt==2 and pos in (2,5,7) else None,json.dumps({'terminal_status':cause}) if accepted or cause=='model_error' else None,f'arena/test/output440-{pos}.json' if accepted else None,hotkey('runner440'),0 if scored else None,json.dumps({'companies':[]}) if scored else None))
        if pos not in (2,5,7,8):
            score=assignment+':score:1'
            cur.execute("INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,terminal_cause,output_ref,stage_generation,runner_hotkey,scored_run_id,participation_accepted_at) VALUES (%s,%s,%s,%s,%s,1,%s,1,'score','accepted','accepted',%s,3,%s,%s,now())",(score,assignment+':score',ROUND,BASE,hotkey(BASE),pos,f'arena/test/score440-{pos}.json',hotkey('runner440'),assignment+':2'))
            cur.execute("INSERT INTO public.lab_arena_ledger (entry_kind,round_id,submission_id,miner_hotkey,run_id,stage,provider,operation_id,funding_source,call_identity,amount_microusd) VALUES ('settlement',%s,%s,%s,%s,1,'openrouter','test','host',%s,100)",(ROUND,BASE,hotkey(BASE),score,'sha256:'+f'{pos+1:064x}'))
            cur.execute("INSERT INTO public.lab_arena_trajectory_events (run_id,event_id,round_id,submission_id,miner_hotkey,runner_hotkey,assignment_id,icp_identifier,stage,icp_position,attempt,run_kind,model_role,event_kind,occurred_at,content) VALUES (%s,gen_random_uuid(),%s,%s,%s,%s,%s,%s,1,%s,1,'score','baseline','runtime.result',now(),'{}')",(score,ROUND,BASE,hotkey(BASE),hotkey('runner440'),assignment+':score',str(pos),pos))
    cur.execute("INSERT INTO public.lab_arena_ledger (entry_kind,round_id,submission_id,miner_hotkey,run_id,stage,provider,operation_id,funding_source,call_identity,amount_microusd) VALUES ('uncertain',%s,'miner440-1',%s,NULL,1,'openrouter','code_review','miner_key',%s,100)",(ROUND,hotkey('miner440-1'),'sha256:'+'d'*64))
    cur.execute("INSERT INTO public.lab_arena_ledger (entry_kind,round_id,submission_id,miner_hotkey,run_id,stage,provider,operation_id,funding_source,call_identity,amount_microusd) VALUES ('settlement',%s,%s,%s,%s,1,'openrouter','test','host',%s,100)",(ROUND,BASE,hotkey(BASE),f'{ROUND}:{BASE}:1:0:2','sha256:'+'e'*64))
    cur.execute('SET session_replication_role=origin')


def _sql(cur):
    sql=MIGRATION.read_text().replace('PENDING_SCORER_DIGEST',NEW_DIGEST)
    for expected,expression in HASHES.items():
        cur.execute(f"SELECT encode(extensions.digest(({expression})::text,'sha256'),'hex')")
        sql=sql.replace(expected,cur.fetchone()[0])
    cur.execute("SELECT encode(extensions.digest(pg_get_functiondef('public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure),'sha256'),'hex')")
    sql=sql.replace('847e3ad30de957162f59ec3cd58d52e99199e6a68c098484406d5830012e1b5c',cur.fetchone()[0])
    cur.execute("SELECT encode(extensions.digest(jsonb_agg(jsonb_build_array(c.relname,t.tgname,t.tgenabled,pg_get_triggerdef(t.oid),pg_get_functiondef(t.tgfoid),owner.rolname,p.proacl::text,p.prosecdef,p.proconfig) ORDER BY c.relname,t.tgname)::text,'sha256'),'hex') FROM pg_trigger t JOIN pg_class c ON c.oid=t.tgrelid JOIN pg_proc p ON p.oid=t.tgfoid JOIN pg_roles owner ON owner.oid=p.proowner WHERE t.tgrelid IN ('public.lab_arena_rounds'::regclass,'public.lab_arena_runs'::regclass,'public.lab_arena_ledger'::regclass) AND t.tgname IN ('lab_arena_rounds_write_once','lab_arena_runs_terminal','lab_arena_integrity_run_guard','lab_arena_runs_participation','lab_arena_ledger_append_only') AND NOT t.tgisinternal")
    sql=sql.replace('5b74d744d9aaae0bbc1b18c12e6a943b07c83ee5b8cad92aacc094117cc52067',cur.fetchone()[0])
    return sql.replace("pg_catalog.clock_timestamp()+INTERVAL '4500 seconds'", "'2026-10-10T03:00:00Z'::timestamptz+INTERVAL '4500 seconds'")


def _snapshot(cur):
    result={}
    for table,order in (('lab_arena_rounds','round_id'),('lab_arena_submissions','submission_id'),('lab_arena_runs','run_id'),('lab_arena_ledger','entry_id'),('lab_arena_trajectory_events','trajectory_id')):
        cur.execute(f"SELECT coalesce(jsonb_agg(to_jsonb(x) ORDER BY {order}),'[]') FROM public.{table} x")
        result[table]=cur.fetchone()[0]
    cur.execute("SELECT jsonb_agg(to_jsonb(t) ORDER BY tgrelid,tgname) FROM pg_trigger t WHERE tgrelid IN ('public.lab_arena_rounds'::regclass,'public.lab_arena_runs'::regclass,'public.lab_arena_ledger'::regclass) AND NOT tgisinternal")
    result['triggers']=cur.fetchone()[0]
    cur.execute("SELECT pg_get_functiondef('public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure)")
    result['scoring_function']=cur.fetchone()[0]
    return result


def test_recovery_preserves_history_rejudges_and_replays(seeded,monkeypatch):
    psycopg,dsn=seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            before=_snapshot(cur)
            cur.execute("SELECT public.lab_arena_has_recent_participation_v1('finney',71,%s)",(hotkey('runner440'),))
            participation_before=cur.fetchone()[0]
            sql=_sql(cur);cur.execute(sql);after=_snapshot(cur)
            cur.execute("SELECT public.lab_arena_has_recent_participation_v1('finney',71,%s)",(hotkey('runner440'),))
            assert cur.fetchone()[0]==participation_before=={'eligible':True}
            assert after['triggers']==before['triggers']
            live=next(r for r in after['lab_arena_rounds'] if r['round_id']==ROUND)
            old=next(r for r in before['lab_arena_rounds'] if r['round_id']==ROUND)
            assert (live['status'],live['stage_generation'])==('stage1',5)
            assert live['configuration_doc']['schedule']=={**old['configuration_doc']['schedule'],'stage_1_close':'2026-10-10T05:30:01Z'}
            archive=next(r for r in after['lab_arena_rounds'] if r['round_id']==ARCHIVE)
            assert archive['configuration_doc']['schedule']==old['configuration_doc']['schedule']
            assert live['configuration_doc']['max_attempts_per_assignment']==2
            assert [s for s in after['lab_arena_submissions'] if s['round_id']==ROUND]==before['lab_arena_submissions']
            new=[r for r in after['lab_arena_runs'] if r['round_id']==ROUND and r['attempt']==3]
            assert {r['icp_position'] for r in new}=={2,5,7,8}
            assert all(r['status']=='pending' for r in new)
            cur.execute(sql);assert _snapshot(cur)==after
            for index in range(4):
                cur.execute('SELECT public.lab_arena_claim_assignment(%s,%s,20,20,%s,%s,%s,%s,4500)',
                    (ROUND,hotkey('healthy440'),[],f'{index+100:032x}',
                     'sha256:'+f'{index+100:064x}','sha256:'+f'{index+200:064x}'))
                lease=cur.fetchone()[0]
                assert lease['status']=='leased' and lease['attempt']==3
            cur.execute(sql)
            # Test-only completion simulates execution; production stays on the ordinary runner path.
            cur.execute('SET session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',output_ref='arena/test/recovered440.json' WHERE round_id=%s AND attempt=3",(ROUND,))
            cur.execute("UPDATE public.lab_arena_rounds SET status='stage1_closed',stage_generation=6 WHERE round_id=%s",(ROUND,))
            cur.execute('SET session_replication_role=origin')
    store=ArenaStore(PsycopgTransport(lambda:psycopg.connect(**dsn)))
    configuration=store.get_round(ROUND)['configuration_doc']
    plan=scoring.build_scoring_plan(round_id=ROUND,stage=1,runs=store.list_runs(ROUND,kind='execute'),execution_sequence_policy=contracts.BASELINE_SCORED_FIRST_POLICY,configuration=configuration)
    assert len(plan['work_items'])==10 and not plan.get('incomplete_rows') and not plan['zero_rows']
    service=object.__new__(ArenaService);service._store=store;service._round=lambda rid:store.get_round(rid)
    service._load_scoring_plan=lambda row,stage:plan;service._require_code_review=lambda *a:None
    service.evaluation_icps=lambda rid:daily_icps()[:10]
    class Objects:
        @staticmethod
        def get_bounded(ref,max_bytes):
            return json.dumps({'schema_version':contact_policy.output_schema(configuration),'companies':[]}).encode()
    service._objects=Objects()
    monkeypatch.setattr(provider_observations,'resolve_observations',lambda *a,**kw:[])
    assert service.open_scoring(ROUND,1)['assignments']==10
    scores=store.list_runs(ROUND,kind='score')
    assert all(r['assignment_id'].endswith(':score:recovery440') for r in scores)
    assert all(r['judgment_scope_doc']['scorer_image_digest']==NEW_DIGEST for r in scores)


@pytest.mark.parametrize('mutation',[
 "UPDATE public.lab_arena_rounds SET status='stage2' WHERE round_id='arena-2026-10-10'",
 "UPDATE public.lab_arena_rounds SET stage_generation=99 WHERE round_id='arena-2026-10-10'",
 "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,stage_2_close}','\"2026-10-11T14:00:02Z\"') WHERE round_id='arena-2026-10-10'",
 "UPDATE public.lab_arena_runs SET status='accepted',output_ref='unexpected' WHERE round_id='arena-2026-10-10' AND icp_position=2 AND attempt=2",
 "UPDATE public.lab_arena_submissions SET source_size_bytes=1001 WHERE submission_id='miner440-1'",
 "UPDATE public.qualification_private_icp_sets SET icps='[]' WHERE set_id=20261009",
 "UPDATE public.lab_arena_restart_claim_control SET operator_paused=true WHERE singleton",
 "ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal",
])
def test_drift_fails_closed(seeded,mutation):
    psycopg,dsn=seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            sql=_sql(cur)
            cur.execute('SET session_replication_role=replica');cur.execute(mutation);cur.execute('SET session_replication_role=origin')
            conn.commit()
            before=_snapshot(cur)
            with pytest.raises(psycopg.Error,match='Oct10 recovery'):
                cur.execute(sql)
            cur.execute('ROLLBACK');assert _snapshot(cur)==before
            cur.execute('ALTER TABLE public.lab_arena_runs ENABLE TRIGGER lab_arena_runs_terminal')


def test_no_shortened_execution_window(seeded):
    psycopg,dsn=seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            sql=_sql(cur).replace("'2026-10-10T03:00:00Z'", "'2026-10-10T04:16:00Z'")
            before=_snapshot(cur)
            with pytest.raises(psycopg.Error,match='full execution window'):
                cur.execute(sql)
            cur.execute('ROLLBACK');assert _snapshot(cur)==before


@pytest.mark.parametrize('at,allowed',[
    ('2026-10-10T04:15:00Z',True),
    ('2026-10-10T04:15:01Z',False),
    ('2026-10-10T04:16:00Z',False),
])
def test_full_frozen_lease_boundary(seeded,at,allowed):
    psycopg,dsn=seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            sql=_sql(cur).replace('2026-10-10T03:00:00Z',at)
            before=_snapshot(cur)
            if allowed:
                cur.execute(sql)
            else:
                with pytest.raises(psycopg.Error,match='full execution window'):
                    cur.execute(sql)
                cur.execute('ROLLBACK');assert _snapshot(cur)==before


@pytest.mark.parametrize('mutation',[
    "UPDATE public.lab_arena_ledger SET amount_microusd=101 WHERE round_id='arena-2026-10-10-r440archive'",
    "UPDATE public.lab_arena_trajectory_events SET content='{\"changed\":true}' WHERE round_id='arena-2026-10-10-r440archive'",
    "UPDATE public.lab_arena_runs SET output_ref='arena/test/changed440.json' WHERE round_id='arena-2026-10-10-r440archive'",
    "UPDATE public.lab_arena_runs SET output_ref='arena/test/changed440.json' WHERE round_id='arena-2026-10-10' AND kind='execute' AND icp_position=0 AND attempt=2",
    "DELETE FROM public.lab_arena_runs WHERE round_id='arena-2026-10-10' AND attempt=3 AND icp_position=2",
])
def test_replay_rejects_corrupt_archive_or_history(seeded,mutation):
    psycopg,dsn=seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            sql=_sql(cur);cur.execute(sql)
            cur.execute('SET session_replication_role=replica');cur.execute(mutation);cur.execute('SET session_replication_role=origin');conn.commit()
            before=_snapshot(cur)
            with pytest.raises(psycopg.Error,match='replay differs'):
                cur.execute(sql)
            cur.execute('ROLLBACK');assert _snapshot(cur)==before


def test_baseline_inflight_accounting_blocks_but_review_uncertainty_does_not(seeded):
    psycopg,dsn=seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            sql=_sql(cur)
            cur.execute("INSERT INTO public.lab_arena_ledger (entry_kind,round_id,submission_id,miner_hotkey,run_id,stage,provider,operation_id,funding_source,call_identity,amount_microusd) VALUES ('reservation',%s,%s,%s,%s,1,'openrouter','test','host',%s,100)",(ROUND,BASE,hotkey(BASE),f'{ROUND}:{BASE}:1:0:2','sha256:'+'f'*64))
            conn.commit();before=_snapshot(cur)
            with pytest.raises(psycopg.Error,match='provider accounting remains open'):
                cur.execute(sql)
            cur.execute('ROLLBACK');assert _snapshot(cur)==before


def test_late_error_rolls_back_archive_retries_trigger_and_function_changes(seeded):
    psycopg,dsn=seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            sql=_sql(cur).replace('  EXECUTE replace(v_definition,v_anchor,v_namespace||v_anchor);',
                "  EXECUTE replace(v_definition,v_anchor,v_namespace||v_anchor);\n  RAISE EXCEPTION 'injected late failure';")
            before=_snapshot(cur)
            with pytest.raises(psycopg.Error,match='injected late failure'):
                cur.execute(sql)
            cur.execute('ROLLBACK');assert _snapshot(cur)==before


def test_wrong_commitment_fails_closed(seeded):
    import re
    psycopg,dsn=seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            sql=re.sub(r"v_bank_hash CONSTANT TEXT := '[0-9a-f]+'", "v_bank_hash CONSTANT TEXT := '"+'0'*64+"'",_sql(cur))
            before=_snapshot(cur)
            with pytest.raises(psycopg.Error,match='exact state'):
                cur.execute(sql)
            cur.execute('ROLLBACK');assert _snapshot(cur)==before


def test_preexisting_miner_execution_blocks_rewind(seeded):
    psycopg,dsn=seeded
    with psycopg.connect(**dsn) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            sql=_sql(cur)
            cur.execute("INSERT INTO public.lab_arena_runs (run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation) VALUES ('miner440-execution:1','miner440-execution',%s,'miner440-1',%s,2,0,1,'execute','pending',5)",(ROUND,hotkey('miner440-1')))
            conn.commit();before=_snapshot(cur)
            with pytest.raises(psycopg.Error,match='exact state'):
                cur.execute(sql)
            cur.execute('ROLLBACK');assert _snapshot(cur)==before
