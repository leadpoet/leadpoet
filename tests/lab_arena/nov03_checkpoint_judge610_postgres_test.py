"""Exact operator seventh reuses two cached negatives; no automatic eighth."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch
import json
import re
import uuid

import pytest

from lab_arena import company_judgments, contracts, scoring
from lab_arena.store import hash_lease_token
from lab_arena.service import ArenaService, ServiceConfig
from tests.lab_arena.partial_company_judgments_postgres_test import _sign
from tests.lab_arena.legacy_partial_postgres_test import _completion
from tests.lab_arena.test_lab_arena_service_round import keypair, wallet_verify, FakeChain
from tests.lab_arena import nov03_corrected_judge607_postgres_test as prior
from tests.lab_arena.company_judgments_test import _input, _raw
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

SQL=prior.prior.SCRIPTS/'610-arena-2026-11-03-baseline8-checkpoint-continuation.sql'
ROUND=prior.ROUND; ASSIGNMENT=prior.ASSIGNMENT; SOURCE=prior.SOURCE; PARENT=prior.RUN; RUN=ASSIGNMENT+':7'
IMAGE='sha256:3576adedad0e858b8a68b6760e61d0a060512ade572bc17f323f27a6e37f541b'
database=prior.database; scalar=prior.scalar


def setup(conn,dsn,pg):
    with conn.cursor() as cur:
        for sig in ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)','public.lab_arena_scoring_policy_history_v1(text)'):
            d=scalar(cur,'SELECT pg_get_functiondef(%s::regprocedure)',(sig,))
            d=d.replace("\n      OR public.lab_arena__nov03_checkpoint_judge610_v1(v_round,runs)","").replace(",'nov03_baseline8_checkpoint_continuation610_v1'","").replace(",'nov03_three_unfinished_executions609_v1'","")
            cur.execute(d)
    conn.commit()
    context=prior.prior.context
    def bound():
        values=context();values[1]['configuration_doc'].update(scorer_image_digest=IMAGE,scorer_image_reference='493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@'+IMAGE,scoring_cap_microusd=50000000)
        return values
    with patch.object(prior.prior,'context',bound):
        store,service,objects,sql,_,work=prior.setup(conn,dsn,pg)
    with conn.cursor() as cur:
        cur.execute(sql);conn.commit()
        support=(prior.prior.SCRIPTS/'609-arena-2026-11-03-three-unfinished-executions.sql').read_text().split('DO $append609$')[0]
        cur.execute(support);conn.commit()
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='' WHERE singleton")
        cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='judge_error',runner_hotkey=%s,lease_generation=1,output_ref='partial6.json',result_doc=%s::jsonb WHERE run_id=%s",(hotkey('corrected-runner607'),json.dumps({'terminal_status':'judge_error','failure_diagnostic':{'error_class':'judge_timeout','stage':'sandbox'}}),PARENT))
        # Model the actual immutable partial output and its complete provenance.
        negatives=[]
        for ref in work['company_judgment_refs'][:2]:
            raw={**_raw(0),'company_judgment_policy':'company_judgment_v1','failure_reason':'intent_details_mismatch' if ref['company_index']==0 else 'required_intent_unverified'}
            negatives.append({**{k:ref[k] for k in ('company_index','cache_key','company_input_hash')},'authority_slot':0,'raw_judgment':raw})
        partial=scoring.build_scoring_failure(SOURCE,'judge_error',company_judgments=negatives)
        objects['partial6.json']=contracts.canonical_json(partial).encode()
        exclusions=sorted([hotkey('corrected-runner607'),hotkey('nov03-baseline')])
        prior_cache={'schema_version':company_judgments.LEASE_SCHEMA_VERSION,'hits':[],'misses':[{**{k:ref[k] for k in ('company_index','cache_key','company_input_hash')},'authority_slot':0} for ref in work['company_judgment_refs']]}
        cur.execute("UPDATE public.lab_arena_runs SET claim_request_id=%s,claim_request_hash=%s,claim_response=%s::jsonb,result_doc=result_doc||jsonb_build_object('company_judgment_completion_request_hash',%s::text) WHERE run_id=%s",('c'*32,'sha256:'+'d'*64,json.dumps({'company_judgment_cache':prior_cache,'runner_authority_exclusions':exclusions}),'sha256:'+'f'*64,PARENT))
        for ref,new_judgment in zip(work['company_judgment_refs'],negatives):
            evidence=company_judgments.build_evidence_snapshot(new_judgment=new_judgment,company_ref=ref,source_score_run_id=PARENT,source_scored_run_id=SOURCE,source_output_ref='partial6.json',source_output_hash=contracts.document_hash(partial),source_runner_hotkey=hotkey('corrected-runner607'),source_claim_request_id='c'*32,source_claim_request_hash='sha256:'+'d'*64,source_lease_generation=1,source_completion_request_hash='sha256:'+'f'*64,runner_authority_exclusions=exclusions)
            cur.execute('INSERT INTO public.lab_arena_company_judgments(cache_key,authority_slot,scope_doc,company_input_hash,evidence_hash,evidence_doc,source_score_run_id,source_scored_run_id,source_runner_hotkey) VALUES (%s,0,%s::jsonb,%s,%s,%s::jsonb,%s,%s,%s)',(ref['cache_key'],json.dumps(ref['scope_doc']),ref['company_input_hash'],contracts.document_hash(evidence),json.dumps(evidence),PARENT,SOURCE,hotkey('corrected-runner607')))
        # The recovered baseline claim uses the publication deadline, after the
        # original stage-one scoring close. All prior policies remain frozen.
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,stage_1_scoring_close}',to_jsonb((now()-interval '1 hour')::text)) WHERE round_id=%s",(ROUND,))
        cur.execute("UPDATE public.lab_arena_scoring_supersessions SET new_round_doc=jsonb_set(new_round_doc,'{configuration_doc}',(SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s)) WHERE round_id=%s",(ROUND,ROUND))
        # A different participant already has an active lease; 610 must leave it.
        cur.execute("UPDATE public.lab_arena_runs SET status='leased',terminal_cause=NULL,runner_hotkey=%s,lease_expires_at=now()+interval '1 hour' WHERE submission_id='nov03-miner'",(hotkey('other-live610'),))
        key='sha256:'+'9'*64;ih='sha256:'+'8'*64
        cur.execute("INSERT INTO public.lab_arena_signal_judgments(cache_key,authority_slot,signal_input_hash,scope_doc,judgment_hash,judgment_doc,source_score_run_id,source_scored_run_id,source_runner_hotkey,source_claim_request_hash,source_lease_generation,runner_authority_exclusions) VALUES (%s,0,%s,%s::jsonb,%s,'{}',%s,%s,%s,%s,1,%s::jsonb)",(key,ih,json.dumps({'cache_key':key,'signal_input_hash':ih}),'sha256:'+'7'*64,PARENT,SOURCE,hotkey('corrected-runner607'),'sha256:'+'d'*64,json.dumps([hotkey('corrected-runner607')])))
        cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,run_id,amount_microusd,entry_doc) VALUES ('settlement',%s,%s,%s,3731545,'{}')",(hotkey('nov03-baseline'),ROUND,PARENT))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    with conn.cursor() as cur:
        before=scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a')
        manifest=dict(source_sha256=prior.prior.pg_hash(cur,before[SOURCE]),prior_run_hashes={rid:prior.prior.pg_hash(cur,row) for rid,row in before.items() if row['assignment_id']==ASSIGNMENT},company_judgment_refs=work['company_judgment_refs'])
        for table in ('lab_arena_company_judgments','lab_arena_signal_judgments'):
            manifest[table+'_row_hashes']=scalar(cur,f"SELECT jsonb_object_agg(cache_key,encode(extensions.digest(to_jsonb(a)::text,'sha256'),'hex')) FROM public.{table} a")
        sql=SQL.read_text();literal=re.search(r'\$manifest610\$(.*?)\$manifest610\$',sql,re.S)[1];sql=sql.replace(literal,json.dumps(manifest,separators=(',',':'),sort_keys=True))
        preserved=scalar(cur,"SELECT jsonb_build_object('runs',(SELECT jsonb_agg(to_jsonb(a) ORDER BY run_id) FROM public.lab_arena_runs a),'ledger',(SELECT jsonb_agg(to_jsonb(a) ORDER BY entry_id) FROM public.lab_arena_ledger a),'company',(SELECT jsonb_agg(to_jsonb(a) ORDER BY cache_key) FROM public.lab_arena_company_judgments a),'signal',(SELECT jsonb_agg(to_jsonb(a) ORDER BY cache_key) FROM public.lab_arena_signal_judgments a),'round',(SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s),'control',(SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton))",(ROUND,))
    return store,service,objects,sql,before,work,preserved


def claim(conn):
    runner=hotkey('corrected-runner607')
    with conn.cursor() as cur:
        cur.execute('SET ROLE lab_arena_service')
        cur.execute('SELECT public.lab_arena_claim_assignment(%s,%s,20,20,%s::text[],%s,%s,%s,1500)',(ROUND,runner,[runner],uuid.uuid4().hex,'sha256:'+'b'*64,hash_lease_token('corrected607')))
        result=cur.fetchone()[0];cur.execute('RESET ROLE')
    conn.commit();return result


def mature(conn):
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='' WHERE singleton")
        cur.execute("UPDATE public.lab_arena_runs SET created_at=now()-interval '30 minutes' WHERE run_id=%s",(RUN,))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()


def test_claim_reuses_both_negatives_and_only_dispatches_three_misses(database,monkeypatch):
    pg,dsn=database
    with pg.connect(**dsn) as conn:
        store,service,objects,sql,before,work,preserved=setup(conn,dsn,pg)
        try:
            with conn.cursor() as cur:
                # One append preserves all old rows, costs and live leases.
                cur.execute(sql);conn.commit();cur.execute(sql);conn.commit()
                assert scalar(cur,'SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s',(ASSIGNMENT,))==7
                assert scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a WHERE run_id<>%s',(RUN,))==before
                assert scalar(cur,'SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s',(ROUND,))==preserved['round']
                assert scalar(cur,'SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton')==preserved['control']
                assert scalar(cur,'SELECT jsonb_agg(to_jsonb(a) ORDER BY entry_id) FROM public.lab_arena_ledger a')==preserved['ledger']
                assert scalar(cur,"SELECT has_function_privilege('lab_arena_service','public.lab_arena__nov03_checkpoint_judge610_v1(public.lab_arena_rounds,public.lab_arena_runs)','EXECUTE')") is False
                assert scalar(cur,'SELECT public.lab_arena__judge_continuation_v1(r,a) FROM public.lab_arena_rounds r JOIN public.lab_arena_runs a ON a.round_id=r.round_id WHERE a.run_id=%s',(RUN,)) is True
            mature(conn);lease=claim(conn);assert lease['status']=='leased',lease
            assert lease['run_id']==RUN and lease['attempt']==7
            cache=lease['company_judgment_cache']
            assert sorted(x['company_index'] for x in cache['hits'])==[0,1]
            assert sorted(x['company_index'] for x in cache['misses'])==[2,3,4]
            document=_input();calls=[]
            def run_score(item,**kwargs):
                calls.append(kwargs['_original_company_indexes'][0])
                return [{**_raw(0),'company_judgment_policy':'company_judgment_v1','failure_reason':'required_intent_unverified'}]
            monkeypatch.setattr(scoring,'score_work_item',run_score)
            scorer=lambda *a:None;scorer.integrity_policy=True;scorer.company_quality=False
            breakdowns,new=scoring.score_quality_work_item({'scored_run_id':SOURCE},icp=document['icp'],companies=document['companies'],scorer=scorer,cache_context=cache)
            assert calls==[2,3,4];assert [r['company_index'] for r in new]==[2,3,4]
            assert len(breakdowns)==5
            assert [h['evidence_doc']['raw_judgment']['failure_reason'] for h in sorted(cache['hits'],key=lambda h:h['company_index'])]==['intent_details_mismatch','required_intent_unverified']
        finally:store._transport.close()


@pytest.mark.parametrize('outcome',['judge_error','expired','accepted'])
def test_terminal_seventh_never_creates_eighth(database,outcome):
    pg,dsn=database
    with pg.connect(**dsn) as conn:
        store,service,objects,sql,*_=setup(conn,dsn,pg)
        try:
            with conn.cursor() as cur:cur.execute(sql)
            conn.commit();mature(conn);assert claim(conn)['run_id']==RUN
            if outcome=='expired':
                with conn.cursor() as cur:
                    cur.execute('SET LOCAL session_replication_role=replica');cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=now()-interval '1 second' WHERE run_id=%s",(RUN,));cur.execute('SET LOCAL session_replication_role=origin')
                conn.commit();store.expire_leases(ROUND)
            elif outcome=='accepted':
                key=keypair('corrected-runner607')
                objects_api=SimpleNamespace(get=lambda key:objects[key],get_bounded=lambda key,limit:objects[key],get_optional_bounded=lambda key,limit:objects.get(key),put=lambda key,value:objects.__setitem__(key,value))
                gateway=ArenaService(ServiceConfig(mode='live',store=store,object_store=objects_api,signer=None,chain=FakeChain([key.ss58_address]),verify_signature=wallet_verify,daily_icp_source=lambda **_: {},banned_hotkeys_source=lambda: (),broker_factory=lambda *_: None))
                gateway.evaluation_icps=service.evaluation_icps
                current=store.get_run(RUN);lease=current['claim_response'];cache=lease['company_judgment_cache']
                new=[{**miss,'raw_judgment':{**_raw(0),'company_judgment_policy':'company_judgment_v1','failure_reason':'required_intent_unverified'}} for miss in cache['misses']]
                raw={x['company_index']:x['evidence_doc']['raw_judgment'] for x in cache['hits']};raw.update({x['company_index']:x['raw_judgment'] for x in new})
                companies=json.loads(objects['output8.json'])['companies']
                breakdowns=scoring.project_company_judgments(companies,[raw[i] for i in range(5)],company_quality=False,scored_indexes=scoring.verify.bucket_skip(gateway.evaluation_icps(ROUND)[8],companies,max_scored_companies=0)[0])
                output={'schema_version':scoring.SCORING_OUTPUT_SCHEMA_VERSION,'scored_run_id':SOURCE,'breakdowns':breakdowns,'company_judgments':new}
                request=_completion(ROUND,key,lease,'corrected607',output,terminal='accepted')
                try:
                    accepted=gateway.handle_complete(request)
                except Exception as exc:
                    raise AssertionError(str(exc.__cause__)) from None
                assert accepted['status']=='accepted'
                assert gateway.handle_complete(request)['idempotent'] is True
                with conn.cursor() as cur:
                    assert scalar(cur,'SELECT count(*) FROM public.lab_arena_company_judgments WHERE source_score_run_id=%s',(PARENT,))==2
                    assert scalar(cur,'SELECT count(*) FROM public.lab_arena_company_judgments WHERE source_score_run_id=%s',(RUN,))==3
            else:
                assert store.complete_attempt(run_id=RUN,lease_token_hash=hash_lease_token('corrected607'),result={'terminal_status':'judge_error'},terminal_cause='judge_error',output_ref=None)['status']=='failed'
            store.expire_leases(ROUND)
            with conn.cursor() as cur:
                assert scalar(cur,'SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt>7',(ASSIGNMENT,))==0
                cur.execute(sql);conn.commit()
                assert scalar(cur,'SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s',(ASSIGNMENT,))==7
        finally:store._transport.close()


@pytest.mark.parametrize('mutation',["UPDATE public.lab_arena_runs SET terminal_cause='lease_expired' WHERE run_id='"+PARENT+"'", "UPDATE public.lab_arena_runs SET output_ref=NULL WHERE run_id='"+PARENT+"'", "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,publication_deadline}',to_jsonb((now()-interval '1 second')::text))", "UPDATE public.lab_arena_company_judgments SET evidence_hash='sha256:'||repeat('0',64)", "UPDATE public.lab_arena_signal_judgments SET judgment_hash='sha256:'||repeat('0',64)", "UPDATE public.lab_arena_runs SET status='leased',lease_expires_at=now()+interval '1 hour' WHERE run_id='"+PARENT+"'", "UPDATE public.lab_arena_restart_claim_control SET operator_paused=true WHERE singleton", "UPDATE public.lab_arena_runs SET result_doc='{}' WHERE run_id='"+PARENT+"'", "UPDATE public.lab_arena_runs SET output_ref='tampered' WHERE run_id='"+SOURCE+"'"])
def test_wrong_timeout_cache_or_deadline_refuses_append(database,mutation):
    pg,dsn=database
    with pg.connect(**dsn) as conn:
        store,_,_,sql,*_=setup(conn,dsn,pg)
        try:
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica');cur.execute(mutation);cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit()
            with pytest.raises(Exception):
                with conn.cursor() as cur:cur.execute(sql)
            conn.rollback()
            with conn.cursor() as cur:assert scalar(cur,'SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s',(ASSIGNMENT,))==6
        finally:store._transport.close()


def test_count_fence_proof_deadline_and_history(database):
    pg,dsn=database
    with pg.connect(**dsn) as conn:
        store,_,_,sql,*_=setup(conn,dsn,pg)
        try:
            with conn.cursor() as cur:cur.execute(sql)
            conn.commit();mature(conn)
            with conn.cursor() as cur:
                post=scalar(cur,"SELECT pg_get_functiondef('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::regprocedure)")
                pre=post.replace('\n      OR public.lab_arena__nov03_checkpoint_judge610_v1(v_round,runs)','')
                cur.execute(pre)
            conn.commit()
            assert claim(conn)['status']!='leased'  # Witness before exact fence.
            with conn.cursor() as cur:cur.execute(post)
            conn.commit()
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,publication_deadline}',to_jsonb((now()-interval '1 second')::text)) WHERE round_id=%s",(ROUND,))
                cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit()
            assert claim(conn)['status']!='leased'
            with conn.cursor() as cur:
                history=scalar(cur,'SELECT public.lab_arena_scoring_policy_history_v1(%s)',(ROUND,))
                item=next(x for x in history['revisions'] if x['recovery']['operation']=='nov03_baseline8_checkpoint_continuation610_v1')
                assert item['generation_floor']==0 and item['receipt_generation']>0
        finally:store._transport.close()
