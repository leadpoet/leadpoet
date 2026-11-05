"""One receipt replaces only the stale baseline and follows native late recovery."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import re
from types import SimpleNamespace
import uuid

import pytest

from lab_arena import contracts, scoring, company_judgments, verify
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena import nov03_corrected_judge607_postgres_test as prior
from tests.lab_arena.test_lab_arena_contracts import base_round_configuration
from tests.lab_arena.full_thinking_round_test import approved_registry
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

SCRIPTS=prior.prior.SCRIPTS
SQL=SCRIPTS/'621-arena-2026-11-04-baseline-source-correction.sql'
ROUND='arena-2026-11-04'
BASELINE='baseline-2026-11-04'
REQUEST='nov04-baseline-source621-v1'
scalar=prior.scalar


@pytest.fixture(scope='module')
def database():
    g=prior.database.__wrapped__();pg,dsn=next(g)
    try:
        # Install the exact current617/618 trigger bodies through committed
        # upgrade paths. The historical fixture's disposable rows are removed.
        with pg.connect(**dsn) as conn:
            store,service,objects,corrected,_,_=prior.setup(conn,dsn,pg,empty=True)
            store.close()
            with conn.cursor() as cur:
                cur.execute(corrected)
                # These are the schema-only preambles of committed618. Its
                # November3 live data correction is deliberately not executed.
                source=(SCRIPTS/'618-arena-2026-11-03-baseline-rejudge.sql').read_text()
                cur.execute(source.split('DO $rejudge618$')[0]+'COMMIT;')
                source605=(SCRIPTS/'605-arena-2026-11-03-admission-review-recovery.sql').read_text()
                cur.execute('DO $patch605_2$'+source605.split('DO $patch605_2$')[1].split('$patch605_2$;')[0]+'$patch605_2$;')
                # Supabase's historical default grants are present in the live
                # trigger function; preserve and verify that exact ACL.
                cur.execute('GRANT EXECUTE ON FUNCTION public.lab_arena_source_digests_immutable_v1() TO anon,authenticated,service_role')
                cur.execute('GRANT EXECUTE ON FUNCTION public.lab_arena__prepare_company_judgment_claim_v1(text,text,bigint,timestamptz) TO anon,authenticated,service_role')
                # Install only the committed reserve-call source patches. The
                # unrelated November3 data repairs are not run in this fixture.
                for filename,tag in [('609-arena-2026-11-03-three-unfinished-executions.sql','609'),('611-arena-2026-11-03-g3-natural-expiry-executions.sql','611')]:
                    source=(SCRIPTS/filename).read_text()
                    pattern=r"\('public.lab_arena_reserve_call[^']*','([a-f0-9]+)','([a-f0-9]+)'.*?\$old"+tag+r"\$(.*?)\$old"+tag+r"\$,\$new"+tag+r"\$(.*?)\$new"+tag+r"\$"
                    patch=re.search(pattern,source,re.S)
                    current=scalar(cur,"SELECT pg_get_functiondef('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::regprocedure)")
                    assert scalar(cur,"SELECT encode(extensions.digest(%s,'sha256'),'hex')",(current,))==patch[1]
                    cur.execute(current.replace(patch[3],patch[4]))
                    assert scalar(cur,"SELECT encode(extensions.digest(pg_get_functiondef('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::regprocedure),'sha256'),'hex')")==patch[2]

                for tag,filename in [('609','609-arena-2026-11-03-three-unfinished-executions.sql'),('610','610-arena-2026-11-03-baseline8-checkpoint-continuation.sql'),('611','611-arena-2026-11-03-g3-natural-expiry-executions.sql'),('612','612-arena-2026-11-03-mychn-checkpoint-continuation.sql'),('613','613-arena-2026-11-03-filson-checkpoint-continuation.sql'),('614','614-arena-2026-11-03-captify-checkpoint-continuation.sql')]:
                    source=(SCRIPTS/filename).read_text()
                    pattern=r"\('public.lab_arena_scoring_policy_history_v1[^']*','([a-f0-9]+)','([a-f0-9]+)'.*?\$old"+tag+r"\$(.*?)\$old"+tag+r"\$,\$new"+tag+r"\$(.*?)\$new"+tag+r"\$"
                    patch=re.search(pattern,source,re.S)
                    current=scalar(cur,"SELECT pg_get_functiondef('public.lab_arena_scoring_policy_history_v1(text)'::regprocedure)")
                    assert scalar(cur,"SELECT encode(extensions.digest(%s,'sha256'),'hex')",(current,))==patch[1]
                    cur.execute(current.replace(patch[3],patch[4]))
                    assert scalar(cur,"SELECT encode(extensions.digest(pg_get_functiondef('public.lab_arena_scoring_policy_history_v1(text)'::regprocedure),'sha256'),'hex')")==patch[2]
        yield pg,dsn
    finally:g.close()


def hash_doc(cur,value):
    return scalar(cur,"SELECT encode(extensions.digest(%s::jsonb::TEXT,'sha256'),'hex')",(json.dumps(value),))


@pytest.fixture
def prepared(database):
    pg,dsn=database
    with pg.connect(**dsn) as conn:
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('TRUNCATE public.lab_arena_rounds CASCADE')
            cur.execute('TRUNCATE public.lab_arena_ledger CASCADE')
            cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,guard_generation=764,guard_commitment='',owner_commitment='',guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]' WHERE singleton")
            config=base_round_configuration()
            # Existing production-compatible company-judgment configuration.
            _,context,_,_,_=prior.prior.context()
            config.update(context['configuration_doc'])
            config.update(round_id=ROUND,mode='live',network_name='finney',netuid=71,
                baseline_hotkey=hotkey('source621-baseline'),stage_1_icp_count=5,stage_2_icp_count=5,
                execution_sequence_policy='baseline_scored_first_v1',infrastructure_retry_cost_policy='trusted_infrastructure_v1',training_minimum_icp_coverage_percent=80,
                max_attempts_per_assignment=2,execution_cap_microusd=60000000,
                cost_per_company_microusd=6000000,execution_icp_cap_microusd=6000000,
                sourcing_cost_eligibility_policy='successful_calls_flat_per_icp_v1',
                reasoning_capture_policy='full_generated_thinking_v1',reasoning_route_registry=approved_registry(),
                baseline_reasoning_configuration={'reasoning_capture_policy':'full_generated_thinking_v1','reasoning_route_registry':approved_registry()},
                baseline_source_url='https://github.com/leadpoet/leadpoet-sales-agent/archive/'+'3'*40+'.tar.gz',
                reasoning_baseline_source_sha256='sha256:'+'3'*64)
            config['scorer_policy']['signal_judgment_policy']='signal_judgment_v1'
            config['schedule']={k:v.replace('2026-09-01','2026-11-03').replace('2026-09-02','2026-11-04') for k,v in base_round_configuration()['schedule'].items()}
            config['schedule']['final_scoring_close']='2099-11-04T23:55:00Z'
            config['schedule']['publication_deadline']='2099-11-04T23:59:00Z'
            config=contracts.validate_round_configuration(config)
            participants=[dict(submission_id=BASELINE,miner_hotkey=config['baseline_hotkey'],is_king=True,source_ref='arena/arena-2026-11-04/sources/old-source.tar.gz',source_size_bytes=100)]
            participants += [dict(submission_id=f'source621-miner-{i}',miner_hotkey=hotkey(f'source621-miner-{i}'),is_king=False) for i in range(8)]
            publication={'king_decision':{'outcome':'crowned','winner_submission_id':'source621-miner-0','king_hotkey':participants[1]['miner_hotkey']}}
            cur.execute("INSERT INTO public.lab_arena_rounds(round_id,status,status_generation,stage_generation,configuration_doc,participants,evaluation_date,icp_set_date,benchmark_ref,publication_doc,published_at,king_outcome,king_hotkey,champion_funding_frozen,champion_hotkey,promotion_doc,baseline_promoted_at,rewards_enabled) VALUES (%s,'published',10,7,%s::jsonb,%s::jsonb,'2026-11-04','2026-11-03','benchmark.json',%s::jsonb,now(),'crowned',%s,true,%s,'{\"commit\":\"bc0d\"}',now(),true)",(ROUND,json.dumps(config),json.dumps(participants),json.dumps(publication),participants[1]['miner_hotkey'],hotkey('stale-champion')))
            prior_config=deepcopy(config);prior_config['round_id']='arena-2026-11-03'
            cur.execute("INSERT INTO public.lab_arena_rounds(round_id,status,configuration_doc,evaluation_date,publication_doc,published_at,reward_activated_at,baseline_promoted_at,promotion_doc) VALUES ('arena-2026-11-03','published',%s::jsonb,'2026-11-03',%s::jsonb,now(),now(),now(),'{\"commit\":\"24d9201c3c5fdc86cb1676628774223205c87166\"}')",(json.dumps(prior_config),json.dumps({'king_decision':{'outcome':'crowned','winner_submission_id':'sub-28594d902080fb5213f4344c09ffb531','king_hotkey':'5HDmNaBQNVQoyApQbFQdSx6pD4ajcJdaYDJePufpWvoxCsrq'}})))
            cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,is_king,status) VALUES ('sub-28594d902080fb5213f4344c09ffb531','arena-2026-11-03','5HDmNaBQNVQoyApQbFQdSx6pD4ajcJdaYDJePufpWvoxCsrq',false,'frozen')")
            future=deepcopy(config);future['round_id']='arena-2026-11-05'
            cur.execute("INSERT INTO public.lab_arena_rounds(round_id,status,configuration_doc,evaluation_date) VALUES ('arena-2026-11-05','open',%s::jsonb,'2026-11-05')",(json.dumps(future),))
            objects={};plans={1:[],2:[]}
            for p in participants:
                cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,is_king,status,source_ref,source_size_bytes) VALUES (%s,%s,%s,%s,'frozen',%s,100)",(p['submission_id'],ROUND,p['miner_hotkey'],p['is_king'],p.get('source_ref','arena/arena-2026-11-04/sources/miner-source.tar.gz')))
                stage=1 if p['is_king'] else 2
                for position in range(10):
                    assignment=f"{ROUND}:{p['submission_id']}:{stage}:{position}"
                    attempt=2 if p['is_king'] else 1
                    rid=f'{assignment}:{attempt}'
                    if p['is_king']:
                        cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,terminal_cause,result_doc,scoring_generation_floor) VALUES (%s,%s,%s,%s,%s,1,%s,1,'execute','failed',7,'provider_error','{\"terminal_status\":\"provider_error\"}',0)",(assignment+':1',assignment,ROUND,p['submission_id'],p['miner_hotkey'],position))
                    output=rid+'.json';objects[output]=json.dumps({'schema_version':contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,'companies':[]}).encode()
                    cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,terminal_cause,result_doc,output_ref,per_icp_score,qualification_doc,scoring_generation_floor) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'execute','accepted',7,'accepted','{\"terminal_status\":\"accepted\"}',%s,0,'{\"companies\":[]}',0)",(rid,assignment,ROUND,p['submission_id'],p['miner_hotkey'],stage,position,attempt,output))
                    judge_attempt=2 if p['is_king'] and position==7 else 1
                    if judge_attempt==2:
                        cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,terminal_cause,result_doc,scored_run_id,company_judgment_refs,scoring_generation_floor) VALUES (%s,%s,%s,%s,%s,1,7,1,'score','failed',7,'judge_error','{\"terminal_status\":\"judge_error\"}',%s,'[]',0)",(assignment+':score:1',assignment+':score',ROUND,p['submission_id'],p['miner_hotkey'],rid))
                    jid=assignment+':score:'+str(judge_attempt);joutput=jid+'.json';objects[joutput]=json.dumps({'schema_version':scoring.SCORING_OUTPUT_SCHEMA_VERSION,'scored_run_id':rid,'breakdowns':[],'company_judgments':[]}).encode()
                    cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,terminal_cause,result_doc,output_ref,scored_run_id,company_judgment_refs,scoring_generation_floor) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'score','accepted',7,'accepted','{\"terminal_status\":\"accepted\",\"company_judgment_consumed_indexes\":[]}',%s,%s,'[]',0)",(jid,assignment+':score',ROUND,p['submission_id'],p['miner_hotkey'],stage,position,judge_attempt,joutput,rid))
                    plans[stage].append(dict(scored_run_id=rid,submission_id=p['submission_id'],icp_position=position,output_ref=output))
                    cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,round_id,run_id,miner_hotkey,amount_microusd,entry_doc) VALUES ('settlement',%s,%s,%s,123,'{}')",(ROUND,rid,p['miner_hotkey']))
            for stage in (1,2):
                plan=dict(schema_version=contracts.SCORING_PLAN_SCHEMA_VERSION,round_id=ROUND,stage=stage,work_items=plans[stage],zero_rows=[],execution_sequence_policy='baseline_scored_first_v1')
                cur.execute(f'UPDATE public.lab_arena_rounds SET stage{stage}_scoring_plan_doc=%s::jsonb WHERE round_id=%s',(json.dumps(plan),ROUND))
            cur.execute("UPDATE public.lab_arena_runs SET claim_response=%s::jsonb WHERE round_id=%s AND kind='score' AND status='accepted'",(json.dumps({'company_judgment_cache':{'schema_version':company_judgments.LEASE_SCHEMA_VERSION,'hits':[],'misses':[]}}),ROUND))
            cur.execute('UPDATE public.lab_arena_rounds SET finalists=%s::jsonb WHERE round_id=%s',(json.dumps([p['submission_id'] for p in participants if not p['is_king']]),ROUND))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        store=ArenaStore(PsycopgTransport(lambda:pg.connect(**dsn)))
        service=ArenaService.__new__(ArenaService);service._store=store
        service._objects=type('Objects',(),{'get':lambda _self,key:objects[key],'get_bounded':lambda _self,key,_limit:objects[key]})()
        service._round=lambda rid:store.get_round(rid)
        service._clock=lambda:datetime(2026,11,4,18,tzinfo=timezone.utc)
        service._config=SimpleNamespace(mode='live',publication_lock_dir=None)
        service.evaluation_icps=lambda _rid:prior.daily_icps()[:10]
        row=store.get_round(ROUND)
        with conn.cursor() as cur:
            before=scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a WHERE round_id=%s',(ROUND,))
            hashes=scalar(cur,"SELECT jsonb_object_agg(run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')) FROM public.lab_arena_runs a WHERE round_id=%s",(ROUND,))
            sub=scalar(cur,'SELECT to_jsonb(s) FROM public.lab_arena_submissions s WHERE submission_id=%s',(BASELINE,))
            from tests.lab_arena.training_export_source_correction_test import corrected
            source_values,source_proof,_=corrected()
            payload=source_values['objects'].get_bounded(source_proof['source_ref'],10_000_000)
            source_ref=json.loads(re.search(r'\$manifest621\$(.*?)\$manifest621\$',SQL.read_text(),re.S)[1])['source_ref']
            objects[source_ref]=payload
            manifest=dict(round_sha256=hash_doc(cur,row),prior_round_sha256=scalar(cur,"SELECT encode(extensions.digest(to_jsonb(r)::TEXT,'sha256'),'hex') FROM public.lab_arena_rounds r WHERE round_id='arena-2026-11-03'"),submission_sha256=hash_doc(cur,sub),minimum_guard_generation=764,run_hashes=hashes,accepted_run_hashes={k:hashes[k] for k,v in before.items() if v['status']=='accepted'},source_ref=source_ref,source_size_bytes=len(payload),source_archive_sha256=source_proof['source_archive_sha256'],source_normalized_sha256=source_proof['source_normalized_sha256'],source_url='https://github.com/leadpoet/leadpoet-sales-agent/archive/24d9201c3c5fdc86cb1676628774223205c87166.tar.gz')
            ledger=scalar(cur,'SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l')
        sql=render(SQL.read_text(),manifest).replace('2026-11-04T21:00:00Z','2099-11-04T21:00:00Z').replace('2026-11-04T23:55:00Z','2099-11-04T23:55:00Z').replace('2026-11-04T23:59:00Z','2099-11-04T23:59:00Z')
        try:yield SimpleNamespace(pg=pg,conn=conn,dsn=dsn,store=store,service=service,objects=objects,row=row,before=before,sub=sub,manifest=manifest,sql=sql,ledger=ledger)
        finally:store.close()


def render(sql,manifest):
    return re.sub(r'\$manifest621\$.*?\$manifest621\$',lambda _: '$manifest621$'+json.dumps(manifest,separators=(',',':'),sort_keys=True)+'$manifest621$',sql,flags=re.S)


def claim(p):
    with p.conn.cursor() as cur:
        cur.execute('SET ROLE lab_arena_service')
        cur.execute('SELECT public.lab_arena_claim_assignment(%s,%s,20,20,ARRAY[]::text[],%s,%s,%s,1500)',(ROUND,hotkey('source621-runner'),uuid.uuid4().hex,'sha256:'+'b'*64,hash_lease_token('source621')))
        result=cur.fetchone()[0];cur.execute('RESET ROLE')
    p.conn.commit();return result


def apply(p):
    with p.conn.cursor() as cur:cur.execute(p.sql)
    p.conn.commit()


def age_pending(p):
    with p.conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=now()-interval '30 minutes' WHERE round_id=%s AND status='pending'",(ROUND,))
        cur.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit()


def assert_current_table_checks(cur):
    # Evaluate every existing CHECK against the actual mutated tuples. Do not
    # replace the committed source key with a shorter fixture-only identity.
    tables=['public.lab_arena_rounds','public.lab_arena_submissions',
        'public.lab_arena_runs','public.lab_arena_scoring_supersessions']
    cur.execute("SELECT conrelid::regclass::text,conname,pg_get_expr(conbin,conrelid) "
        "FROM pg_constraint WHERE contype='c' AND conrelid=ANY(%s::regclass[])",(tables,))
    for table,name,expression in cur.fetchall():
        assert scalar(cur,f'SELECT NOT EXISTS(SELECT 1 FROM {table} WHERE NOT ({expression}))'),name


def test_exact_source_filename_constraint_and_rollback(prepared):
    p=prepared
    expected='arena/arena-2026-11-04/sources/baseline-2026-11-04-24d9201c3c5fdc86cb1676628774223205c87166.tar.gz'
    assert p.manifest['source_ref']==expected
    assert len(expected.rsplit('/',1)[1].removesuffix('.tar.gz'))==60
    invalid=expected.replace('baseline-2026-11-04-','baseline-2026-11-04-source-')
    with p.conn.cursor() as cur:
        with pytest.raises(p.pg.errors.CheckViolation) as error:
            cur.execute(p.sql.replace(expected,invalid))
    assert error.value.diag.constraint_name=='lab_arena_submissions_source_ref_check'
    p.conn.rollback()
    with p.conn.cursor() as cur:
        assert scalar(cur,'SELECT count(*) FROM public.lab_arena_scoring_supersessions WHERE round_id=%s AND request_id=%s',(ROUND,REQUEST))==0
        assert scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a WHERE round_id=%s',(ROUND,))==p.before
        assert scalar(cur,'SELECT source_ref FROM public.lab_arena_submissions WHERE submission_id=%s',(BASELINE,))==p.sub['source_ref']
    p.conn.commit()
    apply(p)
    with p.conn.cursor() as cur:
        assert scalar(cur,'SELECT source_ref FROM public.lab_arena_submissions WHERE submission_id=%s',(BASELINE,))==expected
        assert_current_table_checks(cur)


def test_exact_supersession_preserves_miners_and_history_and_is_idempotent(prepared):
    p=prepared
    original_id=next(k for k,v in p.before.items() if v['kind']=='execute' and v['status']=='accepted' and v['stage']==1)
    with p.conn.cursor() as cur:
        cur.execute('SAVEPOINT immutable')
        with pytest.raises(p.pg.Error,match='terminal attempt is immutable'):
            cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed' WHERE run_id=%s",(original_id,))
        cur.execute('ROLLBACK TO SAVEPOINT immutable')
        prospective={**p.before[original_id],'run_id':p.before[original_id]['assignment_id']+':3','attempt':3}
        assert scalar(cur,'SELECT public.lab_arena__baseline_recovery_run_v1(jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb))',(json.dumps(prospective),)) is False
    p.conn.commit();apply(p)
    row=p.store.get_round(ROUND)
    assert row['status']=='stage2_scoring' and row['scoring_generation_floor']==0
    assert row['champion_hotkey']=='5HDmNaBQNVQoyApQbFQdSx6pD4ajcJdaYDJePufpWvoxCsrq'
    for key in ('stage1_scoring_plan_doc','stage2_scoring_plan_doc','stage_generation','finalists','benchmark_ref','confirmation_bank_hash'):
        assert row[key]==p.row[key]
    assert row['configuration_doc']['baseline_reasoning_configuration']==p.row['configuration_doc']['baseline_reasoning_configuration'] if 'baseline_reasoning_configuration' in p.row['configuration_doc'] else True
    assert row['configuration_doc']['scorer_policy']==p.row['configuration_doc']['scorer_policy']
    with p.conn.cursor() as cur:
        assert_current_table_checks(cur)
        rows=scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a WHERE round_id=%s',(ROUND,))
        receipt=scalar(cur,'SELECT recovery_doc FROM public.lab_arena_scoring_supersessions WHERE round_id=%s AND request_id=%s',(ROUND,REQUEST))
        assert len(receipt['superseded_runs'])==20 and len(rows)==len(p.before)+10
        for key,old in p.before.items():
            omitted={'status','terminal_cause','terminal_doc','updated_at'} if key in receipt['superseded_runs'] else set()
            assert {k:v for k,v in rows[key].items() if k not in omitted}=={k:v for k,v in old.items() if k not in omitted}
        assert scalar(cur,'SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l')==p.ledger
        assert scalar(cur,'SELECT public.lab_arena__baseline_final_ready_v1(%s)',(ROUND,)) is False
        cur.execute(p.sql)
        assert scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a WHERE round_id=%s',(ROUND,))==rows
        cur.execute("SET ROLE lab_arena_service")
        assert scalar(cur,"SELECT public.lab_arena_freeze_champion_funding('arena-2026-11-05')")=={'status':'promotion_pending'}
        cur.execute('RESET ROLE')
    for run in rows.values():
        if run['status']=='pending':assert ArenaService._is_baseline_recovery(run,list(rows.values()),row)
    # Existing native driver and RPC both retain the source recovery phase.
    assert not p.service.scoring_is_complete(ROUND,2)
    assert p.service.close_scoring(ROUND,2)['status']=='draining'
    age_pending(p);lease=claim(p)
    assert lease['status']=='leased' and lease['kind']=='execute' and lease['attempt']==3


def finish_source_executions(p, *, nonempty_position=None, count=10):
    for _ in range(count):
        lease=claim(p);assert lease['status']=='leased' and lease['kind']=='execute',lease
        output=json.dumps({'schema_version':contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,'companies':prior._input()['companies'] if lease['icp_position']==nonempty_position else []}).encode()
        ref=lease['run_id']+'.json';p.objects[ref]=output
        result=p.store.complete_attempt(run_id=lease['run_id'],lease_token_hash=hash_lease_token('source621'),result={'terminal_status':'accepted'},terminal_cause='accepted',output_ref=ref,output_hash=contracts.hash_bytes(output),completion_request_hash=contracts.document_hash({'run':lease['run_id']}))
        assert result['status']=='accepted',result


def finish_source_judgments(p):
    for _ in range(10):
        lease=claim(p);assert lease['status']=='leased' and lease['kind']=='score',lease
        assert lease['scored_run_id'].endswith(':3') and lease['attempt']==(3 if lease['icp_position']==7 else 2)
        output=json.dumps({'schema_version':scoring.SCORING_OUTPUT_SCHEMA_VERSION,'scored_run_id':lease['scored_run_id'],'breakdowns':[],'company_judgments':[]}).encode()
        ref=lease['run_id']+'.json';p.objects[ref]=output
        result=p.store.complete_attempt(run_id=lease['run_id'],lease_token_hash=hash_lease_token('source621'),result={'terminal_status':'accepted','company_judgment_consumed_indexes':[]},terminal_cause='accepted',output_ref=ref,output_hash=contracts.hash_bytes(output),company_judgment_evidence=[],completion_request_hash=contracts.document_hash({'run':lease['run_id']}))
        assert result['status']=='accepted',result


def test_ten_native_execution_and_judgment_completions_use_new_source_plan(prepared):
    p=prepared;apply(p);age_pending(p)
    finish_source_executions(p)
    row=p.store.get_round(ROUND);p.service._recover_baseline(row)
    row=p.store.get_round(ROUND)
    sources=p.store.list_runs(ROUND,stage=1,kind='execute')
    expected=scoring.build_scoring_plan(round_id=ROUND,stage=1,runs=[r for r in sources if r['status']=='accepted'],execution_sequence_policy='baseline_scored_first_v1',configuration=row['configuration_doc'])
    assert row['stage1_scoring_plan_doc']==expected
    assert row['stage2_scoring_plan_doc']==p.row['stage2_scoring_plan_doc']
    age_pending(p)
    finish_source_judgments(p)
    assert claim(p)['status']=='no_pending'
    # Follow the native phase transition; no behavior under test uses replica.
    assert p.service.close_scoring(ROUND,2)['status']=='closed'
    assert p.service.score_stage(ROUND,2)['status']=='ok'
    p.service._record_recovered_judgment_scores(p.store.get_round(ROUND))
    with p.conn.cursor() as cur:
        assert scalar(cur,"SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND stage=1 AND kind='execute' AND status='accepted' AND per_icp_score=0",(ROUND,))==10
        assert scalar(cur,"SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND stage=2 AND status='accepted'",(ROUND,))==160
        assert scalar(cur,'SELECT public.lab_arena__baseline_final_ready_v1(%s)',(ROUND,)) is True
    publication=p.service.publish(ROUND)
    assert publication['status']=='ok',publication
    published=p.store.get_round(ROUND)
    assert published['status']=='published' and published['king_outcome']=='no_king'
    assert len(published['publication_doc']['final_ranking'])==9
    assert published['reward_activated_at'] is None
    from lab_arena.training_export import _source_evidence
    history=p.store.scoring_policy_history(ROUND)
    source=p.store.get_submission(BASELINE)
    participant=next(p for p in published['participants'] if p['is_king'])
    accepted=next(r for r in p.store.list_runs(ROUND,stage=1,kind='execute') if r['status']=='accepted')
    proof=_source_evidence(p.service._objects,accepted,published,source,participant,history)
    assert proof['source_correction']['request_id']==REQUEST
    assert proof['sha256']==source['source_archive_sha256']


@pytest.mark.parametrize('fault',['round','history','source','prior','paused','active_guard','clock'])
def test_wrong_authority_rolls_back_all_changes(prepared,fault):
    p=prepared;manifest=deepcopy(p.manifest);sql=p.sql
    if fault=='round':manifest['round_sha256']='0'*64
    if fault=='history':manifest['run_hashes'][next(iter(manifest['run_hashes']))]='0'*64
    if fault=='source':manifest['submission_sha256']='0'*64
    if fault=='prior':manifest['prior_round_sha256']='0'*64
    if fault=='clock':sql=sql.replace('2099-11-04T21:00:00Z','2000-11-04T21:00:00Z')
    with p.conn.cursor() as cur:
        if fault=='paused':cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=true WHERE singleton')
        if fault=='active_guard':cur.execute("SELECT public.lab_arena_acquire_restart_guard_v1(%s,%s,764,600,%s,'gateway','source621-test')",('lab_arena_restart_guard:'+'a'*64,'lab_arena_restart_owner:'+'b'*64,'c'*40))
    p.conn.commit()
    with p.conn.cursor() as cur:
        with pytest.raises(p.pg.Error):cur.execute(render(sql,manifest))
    p.conn.rollback()
    assert p.store.get_round(ROUND)==p.row
    with p.conn.cursor() as cur:
        assert scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a WHERE round_id=%s',(ROUND,))==p.before
        assert scalar(cur,'SELECT count(*) FROM public.lab_arena_scoring_supersessions WHERE round_id=%s AND request_id=%s',(ROUND,REQUEST))==0


@pytest.mark.parametrize("call_succeeded",[True,False])
def test_cost_replacement_generation(prepared,call_succeeded):
    from tests.lab_arena.nov04_baseline_source621_postgres_test import (
        ROUND, BASELINE, apply, age_pending, claim, scalar,
    )
    p=prepared
    old=[v for v in p.before.values() if v['kind']=='execute' and v['submission_id']==BASELINE and v['icp_position']==0]
    assert len(old)==2 and {v['attempt'] for v in old}=={1,2}
    # Both physical attempts belong to the retired source; excluding only the
    # previously accepted attempt2 is insufficient.
    with p.conn.cursor() as cur:
        for row in old:
            identity='sha256:'+('a' if row['attempt']==1 else 'b')*64
            cur.execute("""INSERT INTO public.lab_arena_ledger
              (entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,
               call_identity,provider,operation_id,funding_source,amount_microusd,
               entry_doc,terminal_response)
              VALUES ('settlement',%s,%s,%s,%s,1,%s,'openrouter',
               'openrouter.chat','host',3000000,'{}','{"status":200,"call_succeeded":true}')""",
              (row['miner_hotkey'],ROUND,BASELINE,row['run_id'],identity))
        old_ledger=scalar(cur,'SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l')
        old_max=scalar(cur,'SELECT max(entry_id) FROM public.lab_arena_ledger')
        cost=scalar(cur,'SELECT public.lab_arena__successful_icp_cost_state(%s,%s,0)',(ROUND,BASELINE))
        assert cost['settled_microusd']==6000000
        miner=next(r['submission_id'] for r in p.before.values() if r['stage']==2)
        ordinary_before=scalar(cur,'SELECT public.lab_arena_icp_cost_eligibility(%s,%s,0,0)',(ROUND,miner))
    p.conn.commit()
    apply(p);age_pending(p)
    lease=claim(p)
    assert lease['kind']=='execute' and lease['attempt']==3
    # Claim ordering is part of this fixture. If it changes, choose the actual
    # lease ICP in the historical seed; do not weaken the assertion.
    assert lease['icp_position']==0
    token=hash_lease_token('source621')
    def rpc(sql,args=()):
        with p.conn.cursor() as cur:
            cur.execute('SET ROLE lab_arena_service')
            cur.execute('SELECT public.'+sql,args);result=cur.fetchone()[0]
            cur.execute('RESET ROLE')
        p.conn.commit();return result
    funding=p.store.get_run(lease['run_id'])['champion_funding_sources']['openrouter']
    identity='sha256:'+'c'*64
    assert rpc("lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter',%s,100,'{}'::jsonb,1500)",(lease['run_id'],token,identity,funding))['status']=='reserved'
    assert rpc('lab_arena_mark_dispatched(%s,%s,%s)',(lease['run_id'],token,identity))['status']=='dispatched'
    assert rpc('lab_arena_settle_call(%s,%s,%s,6000000,%s::jsonb,1500)',(lease['run_id'],token,identity,json.dumps({'status':200,'call_succeeded':call_succeeded})))['status']=='settled'
    with p.conn.cursor() as cur:
        state=scalar(cur,'SELECT public.lab_arena__successful_icp_cost_state(%s,%s,0)',(ROUND,BASELINE))
        assert state['settled_microusd']==12000000
        assert state['successful_microusd']==(12000000 if call_succeeded else 6000000)
        assert state['competition_settled_microusd']==6000000
        assert state['competition_successful_microusd']==(6000000 if call_succeeded else 0)
        eligibility=scalar(cur,'SELECT public.lab_arena_icp_cost_eligibility(%s,%s,0,5)',(ROUND,BASELINE))
        assert eligibility['eligible'] is True and eligibility['competition_sourcing_microusd']==6000000
        assert scalar(cur,'SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l WHERE entry_id<=%s',(old_max,))==old_ledger
    p.conn.commit()
    refused=rpc("lab_arena_reserve_call(%s,%s,%s,'openrouter.chat','openrouter',%s,100,'{}'::jsonb,1500)",(lease['run_id'],token,'sha256:'+'d'*64,funding))
    assert refused['status']=='refused' and refused['reason']=='money_cap',refused
    ref=lease['run_id']+'.json';output=json.dumps({'schema_version':contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,'companies':[]}).encode();p.objects[ref]=output
    assert p.store.complete_attempt(run_id=lease['run_id'],lease_token_hash=token,result={'terminal_status':'accepted'},terminal_cause='accepted',output_ref=ref,output_hash=contracts.hash_bytes(output),completion_request_hash=contracts.document_hash({'run':lease['run_id']}))['status']=='accepted'
    finish_source_executions(p,count=9)
    p.service._recover_baseline(p.store.get_round(ROUND));age_pending(p);finish_source_judgments(p)
    assert p.service.close_scoring(ROUND,2)['status']=='closed'
    assert p.service.score_stage(ROUND,2)['status']=='ok'
    p.service._record_recovered_judgment_scores(p.store.get_round(ROUND))
    from lab_arena.training_export import _competition_cost_snapshot,_source_evidence
    run=p.store.get_run(lease['run_id']);row=p.store.get_round(ROUND);source=p.store.get_submission(BASELINE)
    participant=next(p for p in row['participants'] if p['is_king'])
    binding=_source_evidence(p.service._objects,run,row,source,participant,p.store.scoring_policy_history(ROUND))['source_correction']
    scored=verify.scored_row(BASELINE,0,run['run_id'],p.service.evaluation_icps(ROUND)[0],[],[],row['configuration_doc']['scorer_policy'])
    related=[r for r in p.store.list_runs(ROUND) if r['assignment_id']==run['assignment_id'] or r.get('scored_run_id')==run['run_id']]
    observed={'observed_at':p.service.now().isoformat(),'eligibility':p.store.icp_cost_eligibility(round_id=ROUND,submission_id=BASELINE,icp_position=0,qualified_company_count=0)}
    execution=observed['eligibility']['execution']
    assert execution['excluded_superseded_source_settled_microusd']==6000000
    assert execution['excluded_superseded_source_successful_microusd']==6000000
    assert execution['excluded_infrastructure_settled_microusd']==0
    args=dict(run=run,configuration=row['configuration_doc'],scored=scored,related=related)
    result=_competition_cost_snapshot(**args,reader=lambda **_:observed,source_correction=binding)
    assert result['status']=='available' and result['competition_sourcing_microusd']==6000000,result
    assert _competition_cost_snapshot(**args,reader=lambda **_:observed)['status']=='unavailable'
    wrong=deepcopy(observed);wrong['eligibility']['execution']['excluded_superseded_source_settled_microusd']+=1
    assert _competition_cost_snapshot(**args,reader=lambda **_:wrong,source_correction=binding)['status']=='unavailable'
    miner=next(r['submission_id'] for r in p.before.values() if r['stage']==2)
    ordinary=p.store.icp_cost_eligibility(round_id=ROUND,submission_id=miner,icp_position=0,qualified_company_count=0)
    assert ordinary==ordinary_before
    assert not any(key.startswith('excluded_superseded_source') for key in ordinary['execution'])
    stale={**binding,'commit':'0'*40}
    assert _competition_cost_snapshot(**args,reader=lambda **_:observed,source_correction=stale)['status']=='unavailable'


def test_retired_cache_authorities_miss_and_unrelated_authorities_still_hit(prepared):
    from tests.lab_arena.nov04_source_correction_cache_fixtures import company_controls,signal_controls
    p=prepared;apply(p);age_pending(p);finish_source_executions(p,nonempty_position=1)
    p.service._recover_baseline(p.store.get_round(ROUND));age_pending(p)
    old_execute=next(r for r in p.before.values() if r['stage']==1 and r['kind']=='execute' and r['status']=='accepted' and r['icp_position']==0)
    old_judge=next(r for r in p.before.values() if r['scored_run_id']==old_execute['run_id'] and r['kind']=='score')
    unrelated_execute=next(r for r in p.before.values() if r['stage']==2 and r['kind']=='execute' and r['status']=='accepted')
    unrelated_judge=next(r for r in p.before.values() if r['scored_run_id']==unrelated_execute['run_id'] and r['kind']=='score')
    pending=next(r for r in p.store.list_runs(ROUND,stage=1,kind='score') if r['status']=='pending' and r['icp_position']==0)
    with p.conn.cursor() as cur:
        company_controls(cur,pending_score=pending,retired_execution=old_execute['run_id'],retired_score=old_judge['run_id'],
            unrelated_execution=unrelated_execute['run_id'],unrelated_score=unrelated_judge['run_id'],runner_hotkey=hotkey('source621-runner'))
    p.conn.commit()
    lease=claim(p);assert lease['status']=='leased' and lease['kind']=='score' and lease['icp_position']==1,lease
    leased=p.store.get_run(lease['run_id'])
    with p.conn.cursor() as cur:
        signal_controls(cur,leased_score=leased,lease_token_hash=hash_lease_token('source621'),round_row=p.store.get_round(ROUND),
            retired_execution=old_execute['run_id'],retired_score=old_judge['run_id'],
            unrelated_execution=unrelated_execute['run_id'],unrelated_score=unrelated_judge['run_id'])


@pytest.mark.parametrize('fault',['cost_acl','helper_strict','helper_missing'])
def test_replay_rejects_changed_security_and_missing_authority(prepared,fault):
    p=prepared;apply(p)
    with p.conn.cursor() as cur:
        if fault=='cost_acl':cur.execute('GRANT EXECUTE ON FUNCTION public.lab_arena__execute_assignment_calls_v1(text,text) TO anon')
        elif fault=='helper_strict':cur.execute('ALTER FUNCTION public.lab_arena__source_correction621_v1(public.lab_arena_runs) STRICT')
        else:cur.execute('DROP FUNCTION public.lab_arena__source_correction621_v1(public.lab_arena_runs)')
        # Keep the fault inside this transaction so rollback restores the exact
        # original catalog for the next fixture and proves no silent repair.
        cur.execute('SAVEPOINT security_fault')
        with pytest.raises(p.pg.Error,match='security differs|installed helper differs|helper is missing'):
            cur.execute(p.sql)
    p.conn.rollback()


@pytest.mark.parametrize("entrypoint",["funding","full_thinking"])
def test_freeze_waits_for_correction_and_reads_pending_receipt_after_lock(prepared,entrypoint):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    p=prepared
    # Schema-only installation has no receipt. This is the race window in
    # which the old guard could read absence before waiting for promotion.
    controls=p.sql.split('-- Bind the correction to an exact terminal snapshot.')[0]+'COMMIT;'
    with p.conn.cursor() as cur:cur.execute(controls)
    p.conn.commit()
    with p.conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0))")
    entered=threading.Event()
    def freeze():
        with p.pg.connect(**p.dsn) as conn,conn.cursor() as cur:
            cur.execute("SET statement_timeout='10s'")
            entered.set()
            if entrypoint=="funding":
                cur.execute("SELECT public.lab_arena_freeze_champion_funding('arena-2026-11-05')")
            else:
                source_url=p.row['configuration_doc']['baseline_source_url']
                source_hash=p.row['configuration_doc']['reasoning_baseline_source_sha256']
                cur.execute("SELECT public.lab_arena_freeze_full_thinking_baseline_v1('arena-2026-11-05',%s,%s,%s,%s,%s,%s,%s)",
                    (ROUND,'source621-miner-0','b'*40,source_url,source_hash,p.manifest['source_url'],p.manifest['source_archive_sha256']))
            return cur.fetchone()[0]
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending=pool.submit(freeze)
        assert entered.wait(3)
        with p.conn.cursor() as cur:cur.execute(p.sql)
        p.conn.commit()
        assert pending.result(timeout=10)=={'status':'promotion_pending'}
    future=p.store.get_round('arena-2026-11-05')
    assert future['status']=='open' and future['champion_funding_frozen'] is False
    assert future['participants'] is None and future['benchmark_ref'] is None


@pytest.mark.parametrize("outcome",["accepted","provider_error"])
def test_replacement_retry_uses_only_frozen_attempt_budget(prepared,outcome):
    p=prepared;apply(p);age_pending(p)
    lease=claim(p);assert lease['attempt']==3 and lease['kind']=='execute'
    assert p.store.complete_attempt(run_id=lease['run_id'],lease_token_hash=hash_lease_token('source621'),
        result={'terminal_status':'provider_error'},terminal_cause='provider_error',output_ref=None)['status']=='failed'
    assert p.store.recover_baseline(ROUND)['queued']==1
    age_pending(p)
    replacement=next(r for r in p.store.list_runs(ROUND,stage=1,kind='execute') if r['assignment_id']==lease['assignment_id'] and r['attempt']==4)
    assert ArenaService._is_baseline_recovery(replacement,p.store.list_runs(ROUND,stage=1),p.store.get_round(ROUND))
    # Other root positions remain claimable; use the ordinary runner to reach4.
    for _ in range(11):
        resumed=claim(p)
        if resumed['run_id']==replacement['run_id']:break
        assert resumed['status']=='leased' and resumed['kind']=='execute'
        ref=resumed['run_id']+'.json';output=json.dumps({'schema_version':contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,'companies':[]}).encode();p.objects[ref]=output
        assert p.store.complete_attempt(run_id=resumed['run_id'],lease_token_hash=hash_lease_token('source621'),result={'terminal_status':'accepted'},terminal_cause='accepted',output_ref=ref,output_hash=contracts.hash_bytes(output),completion_request_hash=contracts.document_hash({'run':resumed['run_id']}))['status']=='accepted'
    else:pytest.fail('native replacement4 was never claimable')
    assert resumed['attempt']==4
    if outcome=='provider_error':
        assert p.store.complete_attempt(run_id=resumed['run_id'],lease_token_hash=hash_lease_token('source621'),result={'terminal_status':'provider_error'},terminal_cause='provider_error',output_ref=None)['status']=='failed'
        assert p.store.recover_baseline(ROUND)['queued']==0
        assert not any(r['assignment_id']==resumed['assignment_id'] and r['attempt']>4 for r in p.store.list_runs(ROUND,stage=1,kind='execute'))
        return
    prospective={**p.store.get_run(resumed['run_id']),'attempt':5,'run_id':resumed['assignment_id']+':5'}
    with p.conn.cursor() as cur:
        assert scalar(cur,'SELECT public.lab_arena__source_correction621_v1(jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb))',(json.dumps(prospective),)) is False
    ref=resumed['run_id']+'.json';output=json.dumps({'schema_version':contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,'companies':[]}).encode();p.objects[ref]=output
    assert p.store.complete_attempt(run_id=resumed['run_id'],lease_token_hash=hash_lease_token('source621'),result={'terminal_status':'accepted'},terminal_cause='accepted',output_ref=ref,output_hash=contracts.hash_bytes(output),completion_request_hash=contracts.document_hash({'run':resumed['run_id']}))['status']=='accepted'
    history=p.store.scoring_policy_history(ROUND)
    assert resumed['run_id'] in history['source_corrections'][0]['run_ids']
    assert p.store.recover_baseline(ROUND)['queued']==0
    p.service._recover_baseline(p.store.get_round(ROUND));age_pending(p)
    judgment=claim(p)
    assert judgment['status']=='leased' and judgment['kind']=='score' and judgment['scored_run_id']==resumed['run_id'],judgment
    ref=judgment['run_id']+'.json'
    output=json.dumps({'schema_version':scoring.SCORING_OUTPUT_SCHEMA_VERSION,'scored_run_id':resumed['run_id'],'breakdowns':[],'company_judgments':[]}).encode();p.objects[ref]=output
    assert p.store.complete_attempt(run_id=judgment['run_id'],lease_token_hash=hash_lease_token('source621'),result={'terminal_status':'accepted','company_judgment_consumed_indexes':[]},terminal_cause='accepted',output_ref=ref,output_hash=contracts.hash_bytes(output),company_judgment_evidence=[],completion_request_hash=contracts.document_hash({'run':judgment['run_id']}))['status']=='accepted'



def test_old_funding_rpc_waiter_cannot_freeze_after_correction_commits(prepared):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    p=prepared
    with p.conn.cursor() as cur:
        current=scalar(cur,"SELECT pg_get_functiondef('public.lab_arena_freeze_champion_funding(text)'::regprocedure)")
        old=current.replace("  PERFORM pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));\n",'')
        old=re.sub(r'  -- The promotion lock and target row lock cover this fresh receipt read[.]\n  IF EXISTS\(.*?  END IF;\n','',old,flags=re.S)
        assert scalar(cur,"SELECT encode(extensions.digest(%s,'sha256'),'hex')",(old,))=='558a380c6831c6747e8c5a47e59777549e8c1d66d0a0c3a88c058c8ccfb9a9f8'
        cur.execute(old)
    p.conn.commit()
    with p.conn.cursor() as cur:
        cur.execute("SELECT 1 FROM public.lab_arena_rounds WHERE round_id='arena-2026-11-05' FOR UPDATE")
    entered=threading.Event();state={}
    def older_freeze():
        with p.pg.connect(**p.dsn) as conn,conn.cursor() as cur:
            cur.execute("SET statement_timeout='10s'");state['pid']=scalar(cur,'SELECT pg_backend_pid()');entered.set()
            try:
                cur.execute("SELECT public.lab_arena_freeze_champion_funding('arena-2026-11-05')")
                return cur.fetchone()[0]
            except p.pg.Error as exc:
                conn.rollback();return str(exc).splitlines()[0]
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending=pool.submit(older_freeze);assert entered.wait(3)
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            with p.conn.cursor() as cur:
                waiting=scalar(cur,"SELECT wait_event_type='Lock' FROM pg_stat_activity WHERE pid=%s",(state['pid'],))
                cur.execute('SELECT pg_stat_clear_snapshot()')
            if waiting:break
            entered.wait(.02)
        else:pytest.fail('older native funding RPC did not reach the row-lock boundary')
        with p.conn.cursor() as cur:cur.execute(p.sql)
        p.conn.commit()
        assert pending.result(timeout=10)=='baseline_source_correction_pending'
    assert p.store.get_round('arena-2026-11-05')['champion_funding_frozen'] is False


@pytest.mark.parametrize('terminal_status',['scored','published','cancelled'])
def test_prior_verifier_correction_only_blocks_while_unfinished(prepared,terminal_status):
    p=prepared
    controls=p.sql.split('-- Bind the correction to an exact terminal snapshot.')[0]+'COMMIT;'
    with p.conn.cursor() as cur:cur.execute(controls)
    p.conn.commit()
    with p.conn.cursor() as cur:
        old=scalar(cur,"SELECT to_jsonb(r)-'updated_at' FROM public.lab_arena_rounds r WHERE round_id='arena-2026-11-03'")
        new={**old,'status':terminal_status}
        if terminal_status!='published':new.update(publication_doc=None,published_at=None)
        cur.execute("""INSERT INTO public.lab_arena_scoring_supersessions
          (round_id,generation_floor,request_id,request_hash,actor_ref,transaction_id,backend_pid,
           old_round_doc,new_round_doc,old_score_projections,recovery_doc)
          VALUES ('arena-2026-11-03',1,'verifier-correction-control','sha256:'||repeat('a',64),
           'fixture',txid_current(),pg_backend_pid(),%s::jsonb,%s::jsonb,'{}',
           '{"operation":"nov03_baseline_rejudge618_v1"}')""",(json.dumps(old),json.dumps(new)))
        if terminal_status!='published':
            cur.execute("UPDATE public.lab_arena_rounds SET status=%s,publication_doc=NULL,published_at=NULL WHERE round_id='arena-2026-11-03'",(terminal_status,))
    p.conn.commit()
    with p.conn.cursor() as cur:
        verdict=scalar(cur,"SELECT public.lab_arena_freeze_champion_funding('arena-2026-11-05')")
    assert verdict['status']==('promotion_pending' if terminal_status=='scored' else 'frozen')
