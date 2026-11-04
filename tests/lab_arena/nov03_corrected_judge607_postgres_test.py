"""One exact counted attempt six; ordinary limits can never create seven."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import re
import uuid

import pytest

from lab_arena import company_judgments, contracts, scoring
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena import nov03_forward_verifier604_postgres_test as prior
from tests.lab_arena.company_judgments_test import _input
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

SQL=prior.SCRIPTS/'607-arena-2026-11-03-baseline8-corrected-judge.sql'
ROUND=prior.ROUND
ASSIGNMENT=ROUND+':baseline-2026-11-03:1:8:score'
SOURCE=ROUND+':baseline-2026-11-03:1:8:1'
RUN=ASSIGNMENT+':6'
database=prior.database
scalar=prior.scalar


def setup(conn,dsn,pg,*,empty=False):
    old,new,_,_,objects=prior.seed(conn)
    old['baseline_hotkey']=new['baseline_hotkey']=hotkey('nov03-baseline')
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_submissions SET submission_id='baseline-2026-11-03' WHERE submission_id='nov03-baseline'")
        cur.execute("UPDATE public.lab_arena_runs SET run_id=replace(run_id,'nov03-baseline','baseline-2026-11-03'),assignment_id=replace(assignment_id,'nov03-baseline','baseline-2026-11-03'),submission_id=replace(submission_id,'nov03-baseline','baseline-2026-11-03'),scored_run_id=replace(scored_run_id,'nov03-baseline','baseline-2026-11-03')")
        cur.execute("UPDATE public.lab_arena_rounds SET participants=replace(participants::text,'nov03-baseline','baseline-2026-11-03')::jsonb,stage1_scoring_plan_doc=replace(stage1_scoring_plan_doc::text,'nov03-baseline','baseline-2026-11-03')::jsonb,configuration_doc=%s::jsonb WHERE round_id=%s",(json.dumps(old),ROUND))
        cur.execute("UPDATE public.lab_arena_runs SET result_doc='{\"terminal_status\":\"accepted\"}' WHERE kind='execute'")
        old_input=_input();old_input['evaluation_date']='2026-11-03';old_input['scorer_policy']=old['scorer_policy']
        if empty:old_input['companies']=[]
        old_refs=company_judgments.build_company_scopes(scoring_input=old_input,round_id=ROUND,network_name='finney',netuid=71,scorer_image_digest=old['scorer_image_digest'],scorer_image_reference=old['scorer_image_reference'],integrity_policy='arena_integrity_v1',company_judgment_policy='company_judgment_v1')
        for n in range(1,6):
            cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,runner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,terminal_cause,company_judgment_refs,result_doc,scored_run_id,scoring_generation_floor) VALUES (%s,%s,%s,'baseline-2026-11-03',%s,%s,1,8,%s,'score','failed',3,'judge_error',%s::jsonb,'{\"terminal_status\":\"judge_error\"}',%s,0)",(ASSIGNMENT+':'+str(n),ASSIGNMENT,ROUND,hotkey('nov03-baseline'),hotkey('old-judge607'),n,json.dumps(old_refs),SOURCE))
            cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,run_id,amount_microusd,entry_doc) VALUES ('settlement',%s,%s,%s,100,'{}')",(hotkey('nov03-baseline'),ROUND,ASSIGNMENT+':'+str(n)))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    objects={k:v.replace(b'nov03-baseline',b'baseline-2026-11-03') for k,v in objects.items()}
    objects['output8.json']=json.dumps({'schema_version':contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,'companies':old_input['companies']}).encode()
    with conn.cursor() as cur:
        sql,_=prior.render(cur,old,new);cur.execute(sql)
    conn.commit()
    store=ArenaStore(PsycopgTransport(lambda:pg.connect(**dsn)))
    service=ArenaService.__new__(ArenaService);service._store=store
    service._objects=type('Objects',(),{'get':lambda _self,key:objects[key],'get_bounded':lambda _self,key,_limit:objects[key]})()
    service._round=lambda rid:store.get_round(rid)
    service.evaluation_icps=lambda _rid:daily_icps()[:10]
    item=dict(scored_run_id=SOURCE,submission_id='baseline-2026-11-03',icp_position=8,output_ref='output8.json')
    work=service._scoring_work_items(store.get_round(ROUND),1,{'work_items':[item]})[0]
    with conn.cursor() as cur:
        old_rows=scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a')
        manifest=dict(source_sha256=prior.pg_hash(cur,old_rows[SOURCE]),parent_sha256=prior.pg_hash(cur,old_rows[ASSIGNMENT+':5']),prior_run_hashes={ASSIGNMENT+':'+str(n):prior.pg_hash(cur,old_rows[ASSIGNMENT+':'+str(n)]) for n in range(1,6)},company_judgment_refs=work['company_judgment_refs'])
        sql=SQL.read_text();literal=re.search(r'\$manifest607\$(.*?)\$manifest607\$',sql,re.S)[1]
        sql=sql.replace(literal,json.dumps(manifest,separators=(',',':'),sort_keys=True))
    return store,service,objects,sql,old_rows,work


def claim(conn):
    with conn.cursor() as cur:
        cur.execute('SET ROLE lab_arena_service')
        cur.execute('SELECT public.lab_arena_claim_assignment(%s,%s,20,20,ARRAY[]::text[],%s,%s,%s,1500)',(ROUND,hotkey('corrected-runner607'),uuid.uuid4().hex,'sha256:'+'b'*64,hash_lease_token('corrected607')))
        result=cur.fetchone()[0];cur.execute('RESET ROLE')
    conn.commit();return result


@pytest.mark.parametrize('outcome',['accepted','judge_error','expired'])
def test_one_corrected_attempt_claim_completion_and_no_seventh(database,outcome):
    pg,dsn=database
    with pg.connect(**dsn) as conn:
        store,service,objects,sql,before,work=setup(conn,dsn,pg,empty=outcome=='accepted')
        try:
            with conn.cursor() as cur:
                ledger_before=scalar(cur,'SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l')
                # Reproduce the old count fence before the exact exception.
                assert scalar(cur,'SELECT public.lab_arena__score_attempt_count_v1(a) FROM public.lab_arena_runs a WHERE run_id=%s',(ASSIGNMENT+':5',))==5
                cur.execute(sql);conn.commit()
                cur.execute(sql);conn.commit()
                assert scalar(cur,'SELECT public.lab_arena__score_attempt_count_v1(a) FROM public.lab_arena_runs a WHERE run_id=%s',(RUN,))==6
                assert scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a WHERE run_id<>%s',(RUN,))==before
                assert scalar(cur,'SELECT public.lab_arena__all_judgments_ready_v1(%s)',(ROUND,)) is False
            assert service.scoring_is_complete(ROUND,2) is False
            assert service._unfinished_judgments(store.get_round(ROUND)) is True
            assert claim(conn)['status']=='paused'
            # Materialize the preserved nine through the real service path so
            # pending six is the sole unfinished baseline judgment.
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_rounds SET status='scored' WHERE round_id=%s",(ROUND,))
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='' WHERE singleton")
                cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit()
            service._record_recovered_judgment_scores(store.get_round(ROUND))
            selected=service._scoring_outputs(ROUND,1,round_row=store.get_round(ROUND))
            assert selected[SOURCE]['run_id']==RUN and selected[SOURCE]['status']=='pending'
            with conn.cursor() as cur:
                assert scalar(cur,'SELECT public.lab_arena__baseline_final_ready_v1(%s)',(ROUND,)) is False
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_rounds SET status='stage2' WHERE round_id=%s",(ROUND,))
                cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit()
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='' WHERE singleton")
                cur.execute("UPDATE public.lab_arena_runs SET created_at=now()-interval '30 minutes' WHERE run_id=%s",(RUN,))
                cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit()
            lease=claim(conn);assert lease['status']=='leased',lease
            assert lease['run_id']==RUN and lease['attempt']==6
            assert {x['cache_key'] for x in lease['company_judgment_cache']['misses']}=={x['cache_key'] for x in work['company_judgment_refs']}
            assert service.scoring_is_complete(ROUND,2) is False
            assert service._unfinished_judgments(store.get_round(ROUND)) is True
            if outcome=='expired':
                with conn.cursor() as cur:
                    cur.execute('SET LOCAL session_replication_role=replica')
                    cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=now()-interval '1 second' WHERE run_id=%s",(RUN,))
                    cur.execute('SET LOCAL session_replication_role=origin')
                conn.commit();store.expire_leases(ROUND)
                expired=store.get_run(RUN)
                assert expired['status']=='failed' and expired['terminal_cause']=='lease_expired'
            elif outcome=='accepted':
                objects['corrected-score.json']=json.dumps({'schema_version':scoring.SCORING_OUTPUT_SCHEMA_VERSION,'scored_run_id':SOURCE,'breakdowns':[],'company_judgments':[]}).encode()
                result=store.complete_attempt(run_id=RUN,lease_token_hash=hash_lease_token('corrected607'),result={'terminal_status':'accepted','company_judgment_consumed_indexes':[]},terminal_cause='accepted',output_ref='corrected-score.json',output_hash=contracts.hash_bytes(objects['corrected-score.json']),company_judgment_evidence=[],completion_request_hash='sha256:'+'d'*64)
                assert result['status']=='accepted'
                with conn.cursor() as cur:
                    cur.execute('SET LOCAL session_replication_role=replica')
                    cur.execute("UPDATE public.lab_arena_rounds SET status='scored' WHERE round_id=%s",(ROUND,))
                    cur.execute('SET LOCAL session_replication_role=origin')
                conn.commit()
                service._record_recovered_judgment_scores(store.get_round(ROUND))
                with conn.cursor() as cur:
                    assert scalar(cur,'SELECT public.lab_arena__baseline_final_ready_v1(%s)',(ROUND,)) is True
            else:
                assert store.complete_attempt(run_id=RUN,lease_token_hash=hash_lease_token('corrected607'),result={'terminal_status':'judge_error'},terminal_cause='judge_error',output_ref=None)['status']=='failed'
            store.expire_leases(ROUND)
            with conn.cursor() as cur:
                assert scalar(cur,'SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt>6',(ASSIGNMENT,))==0
                assert scalar(cur,'SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l')==ledger_before
                for rid,old in before.items():
                    if old['kind']=='score':assert scalar(cur,'SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE run_id=%s',(rid,))==old
        finally:store._transport.close()
