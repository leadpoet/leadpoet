"""Current SQL: preserve nine null projections and judge only new work."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import re
import uuid

import pytest

from lab_arena import company_judgments, contracts, scoring
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import current_service_database, SCRIPTS
from tests.lab_arena.nov03_forward_verifier_test import context
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey
from tests.lab_arena.company_judgments_test import _input
from tests.lab_arena.icp_fixtures import daily_icps

SQL=SCRIPTS/'604-arena-2026-11-03-forward-only-verifier.sql'
ROUND='arena-2026-11-03'


@pytest.fixture(scope='module')
def database():
    g=current_service_database();pg,dsn=next(g)
    try:
        with pg.connect(**dsn) as conn,conn.cursor() as cur:
            cur.execute((SCRIPTS/'596-lab-arena-bounded-judge-attempts.sql').read_text())
            cur.execute((SCRIPTS/'597-lab-arena-zero-call-judge-attempt-credit.sql').read_text())
        yield pg,dsn
    finally:g.close()


def scalar(cur,sql,params=()):
    cur.execute(sql,params);return cur.fetchone()[0]


def pg_hash(cur,doc):
    return scalar(cur,"SELECT encode(extensions.digest(%s::jsonb::text,'sha256'),'hex')",(json.dumps(doc),))


def seed(conn):
    _,row,_,_,proof=context()
    old=deepcopy(row['configuration_doc']);old.update({key:proof[key] for key in ('scorer_policy','scorer_image_digest','scorer_image_reference')})
    old['scorer_policy']['judge_models'].pop('industry_evidence_review')
    old['scorer_policy']['signal_judgment_policy']='signal_judgment_v1'
    future=(datetime.now(timezone.utc)+timedelta(days=2)).isoformat()
    old.update(mode='live',network_name='finney',netuid=71,execution_sequence_policy='baseline_scored_first_v1',
        sourcing_cost_eligibility_policy='successful_calls_per_icp_v1',stage_1_icp_count=5,stage_2_icp_count=5,
        max_attempts_per_assignment=2,lease_ttl_seconds=1500,runner_slot_ceiling=20,
        execution_cap_microusd=10000000,cost_per_company_microusd=1000000,
        schedule={key:future for key in ('stage_1_scoring_close','final_scoring_close','publication_deadline','stage_2_close','stage_1_close')})
    new=deepcopy(old);new.update({key:row['configuration_doc'][key] for key in ('scorer_image_digest','scorer_image_reference')})
    new['scorer_policy']['judge_models']['industry_evidence_review']='openai/gpt-6.1-sol'
    baseline=hotkey('nov03-baseline');miner=hotkey('nov03-miner')
    participants=[dict(submission_id='nov03-baseline',miner_hotkey=baseline,is_king=True),dict(submission_id='nov03-miner',miner_hotkey=miner,is_king=False)]
    objects={};sources=[];judges=[]
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('TRUNCATE public.lab_arena_rounds CASCADE')
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=true,pause_reason='nov03_improved_verifier_required603',guard_generation=754 WHERE singleton")
        cur.execute("INSERT INTO public.lab_arena_rounds(round_id,status,status_generation,stage_generation,configuration_doc,participants,evaluation_date,icp_set_date,benchmark_ref,confirmation_bank_ref,confirmation_bank_hash) VALUES (%s,'stage2',5,4,%s::jsonb,%s::jsonb,'2026-11-03','2026-11-02','arena/arena-2026-11-03/benchmark.json','arena/confirmation.json','sha256:'||repeat('a',64))",(ROUND,json.dumps(old),json.dumps(participants)))
        for p in participants:
            cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status,is_king) VALUES (%s,%s,%s,'frozen',%s)",(p['submission_id'],ROUND,p['miner_hotkey'],p['is_king']))
        for pos in range(10):
            assignment=f'{ROUND}:nov03-baseline:1:{pos}';source=assignment+':1'
            cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,terminal_cause,output_ref,scoring_generation_floor) VALUES (%s,%s,%s,'nov03-baseline',%s,1,%s,1,'execute','accepted',3,'accepted',%s,0)",(source,assignment,ROUND,baseline,pos,f'output{pos}.json'))
            objects[f'output{pos}.json']=json.dumps({'schema_version':contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,'companies':[]}).encode()
            sources.append(source)
            if pos==8:continue
            judge=assignment+':score:1';judges.append(judge)
            result={'terminal_status':'accepted','company_judgment_consumed_indexes':[]}
            claim={'company_judgment_cache':{'schema_version':company_judgments.LEASE_SCHEMA_VERSION,'hits':[],'misses':[]}}
            cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,runner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,terminal_cause,output_ref,company_judgment_refs,result_doc,claim_response,scored_run_id,scoring_generation_floor) VALUES (%s,%s,%s,'nov03-baseline',%s,%s,1,%s,1,'score','accepted',3,'accepted',%s,'[]',%s::jsonb,%s::jsonb,%s,0)",(judge,assignment+':score',ROUND,baseline,hotkey('nov03-oldrunner'),pos,f'score{pos}.json',json.dumps(result),json.dumps(claim),source))
            objects[f'score{pos}.json']=json.dumps({'schema_version':scoring.SCORING_OUTPUT_SCHEMA_VERSION,'scored_run_id':source,'breakdowns':[],'company_judgments':[]}).encode()
        cur.execute("UPDATE public.lab_arena_submissions SET code_review_status='passed',code_review_attempts=1,code_review_doc='{}',code_review_claim='sha256:'||repeat('a',64),code_review_started_at=now() WHERE submission_id='nov03-miner'")
        miner_source=f'{ROUND}:nov03-miner:2:0:1'
        cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,terminal_cause,output_ref,scoring_generation_floor) VALUES (%s,%s,%s,'nov03-miner',%s,2,0,1,'execute','accepted',4,'accepted','miner-output.json',0)",(miner_source,miner_source[:-2],ROUND,miner))
        objects['miner-output.json']=json.dumps({'schema_version':contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,'companies':_input()['companies']}).encode()
        plan={'work_items':[{'scored_run_id':s,'submission_id':'nov03-baseline','icp_position':i,'output_ref':f'output{i}.json'} for i,s in enumerate(sources)]}
        cur.execute('UPDATE public.lab_arena_rounds SET stage1_scoring_plan_doc=%s::jsonb WHERE round_id=%s',(json.dumps(plan),ROUND))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit();return old,new,sources,judges,objects


def render(cur,old,new):
    sql=SQL.read_text()
    original=json.loads(re.search(r'\$manifest604\$(.*?)\$manifest604\$',sql,re.S)[1])
    rows=scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(r)) FROM public.lab_arena_runs r WHERE round_id=%s',(ROUND,))
    hashes={rid:pg_hash(cur,row) for rid,row in rows.items()}
    preserved={rid:dict(judge_sha256=hashes[rid],source_run_id=row['scored_run_id'],
        source_original_sha256=hashes[row['scored_run_id']],source_immutable_sha256=pg_hash(cur,{k:v for k,v in rows[row['scored_run_id']].items() if k not in ('per_icp_score','qualification_doc','updated_at')}))
        for rid,row in rows.items() if row['kind']=='score' and row['status']=='accepted'}
    manifest=dict(run_hashes=hashes,preserved_judgments=preserved)
    old_policy=json.loads(re.search(r'\$policy604\$(.*?)\$policy604\$',sql,re.S)[1])
    substitutions={
        pg_hash(cur,original):pg_hash(cur,manifest),
        'e91499fe3037f17d0363f94ffdcfe383f49ceace8a365dc40c4a72277e21337c':pg_hash(cur,old),
        # Derive these constants from the committed exact live policy/delta.
        pg_hash(cur,old_policy):pg_hash(cur,new['scorer_policy']),
        contracts.document_hash(old_policy):contracts.document_hash(new['scorer_policy']),
    }
    config_hash=re.search(r"n.configuration_doc::TEXT,'sha256'\),'hex'\) IS DISTINCT FROM '([a-f0-9]{64})'",sql)[1]
    participant_hash=re.search(r"r.participants::TEXT,'sha256'\),'hex'\) IS DISTINCT FROM '([a-f0-9]{64})'",sql)[1]
    substitutions[config_hash]=pg_hash(cur,new)
    substitutions[participant_hash]=scalar(cur,"SELECT encode(extensions.digest(participants::text,'sha256'),'hex') FROM public.lab_arena_rounds WHERE round_id=%s",(ROUND,))
    sql=sql.replace(json.dumps(original,separators=(',',':'),sort_keys=True),json.dumps(manifest,separators=(',',':'),sort_keys=True))
    sql=sql.replace(json.dumps(old_policy,separators=(',',':'),sort_keys=True),json.dumps(new['scorer_policy'],separators=(',',':'),sort_keys=True))
    sql=re.sub(r"new_digest CONSTANT TEXT:='[^']+'; new_reference CONSTANT TEXT:='[^']+';",f"new_digest CONSTANT TEXT:='{new['scorer_image_digest']}'; new_reference CONSTANT TEXT:='{new['scorer_image_reference']}';",sql)
    for before,after in substitutions.items():sql=sql.replace(before,after)
    return sql,rows


def test_exact_migration_nine_null_projections_and_normal_new_claim(database):
    pg,dsn=database
    with pg.connect(**dsn) as conn:
        old,new,sources,judges,objects=seed(conn)
        with conn.cursor() as cur:
            rendered,before=render(cur,old,new)
            cur.execute(rendered);conn.commit()
            assert scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(r)) FROM public.lab_arena_runs r WHERE round_id=%s',(ROUND,))==before
            assert scalar(cur,'SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s',(ROUND,))==new
            cur.execute(rendered);conn.commit()
        store=ArenaStore(PsycopgTransport(lambda:pg.connect(**dsn)))
        try:
            for judge in judges:
                source=before[judge]['scored_run_id']
                assert store.preserved_judgment_policy(ROUND,judge,source,[])['scorer_policy']==old['scorer_policy']
            assert store.preserved_judgment_policy('unrelated',judges[0],sources[0],[]) is None
            assert store.preserved_judgment_policy(ROUND,judges[0],sources[1],[]) is None
            assert store.preserved_judgment_policy(ROUND,judges[0],sources[0],[{}]) is None
            service=ArenaService.__new__(ArenaService);service._store=store
            service._objects=type('Objects',(),{'get':lambda _self,key:objects[key],'get_bounded':lambda _self,key,_limit:objects[key]})()
            service._round=lambda rid:store.get_round(rid)
            service.evaluation_icps=lambda _rid:daily_icps()[:10]
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_rounds SET status='scored' WHERE round_id=%s",(ROUND,))
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='' WHERE singleton")
                cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit()
            service._record_recovered_judgment_scores(store.get_round(ROUND))
            for judge in judges:
                source=before[judge]['scored_run_id'];actual=store.get_run(source)
                assert actual['per_icp_score']==0 and actual['qualification_doc']=={'companies':[]}
                assert store.preserved_judgment_policy(ROUND,judge,source,[]) is not None
                assert {k:v for k,v in actual.items() if k not in ('per_icp_score','qualification_doc','updated_at')}=={k:v for k,v in before[source].items() if k not in ('per_icp_score','qualification_doc','updated_at')}
                assert store.get_run(judge)==before[judge]
            # A repeat uses stored projections and cannot create an attempt.
            service._record_recovered_judgment_scores(store.get_round(ROUND))
            with conn.cursor() as cur:
                assert scalar(cur,'SELECT count(*) FROM public.lab_arena_runs')==len(before)
                assert scalar(cur,'SELECT count(*) FROM public.lab_arena_ledger')==0
                cur.execute(rendered);conn.commit()
            # The normal scoring initializer builds and claims a new miner
            # assignment from the current image and policy. It does not touch
            # any of the nine accepted baseline judgments.
            miner_source=f'{ROUND}:nov03-miner:2:0:1'
            plan={'work_items':[dict(scored_run_id=miner_source,submission_id='nov03-miner',icp_position=0,output_ref='miner-output.json')]}
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_closed',stage2_scoring_plan_doc=%s::jsonb WHERE round_id=%s",(json.dumps(plan),ROUND))
                cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit()
            work=service._scoring_work_items(store.get_round(ROUND),2,plan)
            assert all(ref['scope_doc']['scorer_image_digest']==new['scorer_image_digest'] for ref in work[0]['company_judgment_refs'])
            assert store.open_scoring(ROUND,2,work,company_quality_cache=True)['status']=='ok'
            with conn.cursor() as cur:
                cur.execute('SET ROLE lab_arena_service')
                cur.execute('SELECT public.lab_arena_claim_assignment(%s,%s,20,20,ARRAY[]::text[],%s,%s,%s,1500)',(ROUND,hotkey('new-verifier-runner'),uuid.uuid4().hex,'sha256:'+'c'*64,hash_lease_token('fixture604')))
                lease=cur.fetchone()[0]
                assert lease['status']=='leased',lease
                assert lease['scored_run_id']==miner_source
                assert {x['cache_key'] for x in lease['company_judgment_cache']['misses']}=={x['cache_key'] for x in work[0]['company_judgment_refs']}
                cur.execute('RESET ROLE')
            for judge in judges:assert store.get_run(judge)==before[judge]
        finally:store._transport.close()


def test_tampered_preimage_rolls_back_and_tampered_authority_is_rejected(database):
    pg,dsn=database
    with pg.connect(**dsn) as conn:
        old,new,sources,judges,_=seed(conn)
        with conn.cursor() as cur:
            rendered,before=render(cur,old,new)
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET output_ref='changed.json' WHERE run_id=%s",(judges[0],))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        with conn.cursor() as cur:
            with pytest.raises(pg.Error,match='original run preimage differs'):
                cur.execute(rendered)
        conn.rollback()
        with conn.cursor() as cur:
            assert scalar(cur,'SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s',(ROUND,))==old
            assert scalar(cur,'SELECT count(*) FROM public.lab_arena_scoring_supersessions')==0
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('UPDATE public.lab_arena_runs SET output_ref=%s WHERE run_id=%s',(before[judges[0]]['output_ref'],judges[0]))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        # updated_at is part of the preserved judge hash. Restore only this
        # test-created row in the disposable fixture before rendering afresh.
        with conn.cursor() as cur:
            rendered,_=render(cur,old,new);cur.execute(rendered)
        conn.commit()
        store=ArenaStore(PsycopgTransport(lambda:pg.connect(**dsn)))
        try:
            assert store.preserved_judgment_policy(ROUND,judges[0],sources[0],[]) is not None
            with conn.cursor() as cur:
                for role in ('anon','authenticated','service_role'):
                    assert scalar(cur,"SELECT has_function_privilege(%s,'public.lab_arena_preserved_judgment_policy_v1(text,text,text,jsonb)','EXECUTE')",(role,)) is False
                assert scalar(cur,"SELECT has_function_privilege('lab_arena_service','public.lab_arena_preserved_judgment_policy_v1(text,text,text,jsonb)','EXECUTE')") is True
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_runs SET output_ref='tampered-source.json' WHERE run_id=%s",(sources[0],))
                cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit()
            assert store.preserved_judgment_policy(ROUND,judges[0],sources[0],[]) is None
        finally:store._transport.close()
