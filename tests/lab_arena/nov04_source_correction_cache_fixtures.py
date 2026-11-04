"""Disposable-PG selector controls; caller supplies actual candidate621 fixture.
Use after apply: native-complete ten executions, call _recover_baseline, then
obtain a pending corrected score for company preparation and a genuine leased
corrected score for signal claims. Seed rows are synthetic fixtures only.
These controls prove selector behavior, not complete company evidence validity;
retain501/502/503 canonical evidence and signature tests for that separate gate.
"""
import json
from lab_arena import contracts


def scalar(cur, sql, args=()):
    cur.execute(sql,args);return cur.fetchone()[0]


def company_controls(cur, *, pending_score, retired_execution, retired_score,
                     unrelated_execution, unrelated_score, runner_hotkey):
    """Call as DB fixture owner; pending_score is a full real pending row.
    Its ref assignment below is fixture-only and never part of production SQL.
    """
    refs=[]
    for marker in ('retired-only','retired-plus-valid'):
        key=contracts.document_hash({'cache621':marker})
        ih=contracts.document_hash({'input621':marker})
        scope={'cache_key':key,'company_input_hash':ih}
        refs.append({'company_index':len(refs),'cache_key':key,'company_input_hash':ih,'scope_doc':scope})
        authorities=[(0,retired_execution,retired_score)]
        if marker=='retired-plus-valid':authorities.append((1,unrelated_execution,unrelated_score))
        for slot,source,jid in authorities:
            evidence={**scope,'authority_slot':slot,'source_score_run_id':jid,
                'source_scored_run_id':source,'source_runner_hotkey':'cache621-authority',
                'runner_authority_exclusions':['cache621-authority'],'raw_judgment':{}}
            cur.execute("""INSERT INTO public.lab_arena_company_judgments
              (cache_key,authority_slot,scope_doc,company_input_hash,evidence_hash,
               evidence_doc,source_score_run_id,source_scored_run_id,source_runner_hotkey)
              VALUES (%s,%s,%s::jsonb,%s,%s,%s::jsonb,%s,%s,'cache621-authority')""",
              (key,slot,json.dumps(scope),ih,contracts.document_hash(evidence),json.dumps(evidence),jid,source))
    before=scalar(cur,'SELECT jsonb_agg(to_jsonb(j) ORDER BY cache_key,authority_slot) FROM public.lab_arena_company_judgments j')
    cur.execute('SET LOCAL session_replication_role=replica')
    cur.execute('UPDATE public.lab_arena_runs SET company_judgment_refs=%s::jsonb WHERE run_id=%s',(json.dumps(refs),pending_score['run_id']))
    cur.execute('SET LOCAL session_replication_role=origin')
    sql="SELECT public.lab_arena__prepare_company_judgment_claim_v1(%s,%s,%s,clock_timestamp()+interval '20 minutes')"
    args=(pending_score['run_id'],runner_hotkey,pending_score['lease_generation']+1)
    result=scalar(cur,sql,args)
    assert len(result['hits'])==len(result['misses'])==1,result
    assert result['hits'][0]['company_index']==1 and result['hits'][0]['authority_slot']==1,result
    assert result['misses'][0]['company_index']==0 and result['misses'][0]['authority_slot']==1,result
    replay=scalar(cur,sql,args)
    assert replay==result
    assert scalar(cur,'SELECT jsonb_agg(to_jsonb(j) ORDER BY cache_key,authority_slot) FROM public.lab_arena_company_judgments j')==before


def signal_controls(cur, *, leased_score, lease_token_hash, round_row,
                    retired_execution, retired_score, unrelated_execution, unrelated_score):
    """Requires real signal-enabled valid score lease; never fake lease validity."""
    for index,marker in enumerate(('retired-only','retired-plus-valid')):
        scope={'schema_version':'leadpoet.lab_arena.signal_judgment_scope.v1',
            'round_id':round_row['round_id'],'evaluation_date':round_row['evaluation_date'],
            'signal_judgment_policy':'signal_judgment_v1',
            'scorer_image_digest':round_row['configuration_doc']['scorer_image_digest'],
            'scorer_image_reference':round_row['configuration_doc']['scorer_image_reference'],
            'signal_input_hash':contracts.document_hash({'signal621':marker})}
        scope['cache_key']=contracts.document_hash(scope)
        authorities=[(0,retired_execution,retired_score)]
        if index==1:authorities.append((1,unrelated_execution,unrelated_score))
        for slot,source,jid in authorities:
            judgment={'test_marker':marker,'authority_slot':slot}
            cur.execute("""INSERT INTO public.lab_arena_signal_judgments
              (cache_key,authority_slot,signal_input_hash,scope_doc,judgment_hash,
               judgment_doc,source_score_run_id,source_scored_run_id,source_runner_hotkey,
               source_claim_request_hash,source_lease_generation,runner_authority_exclusions)
              VALUES (%s,%s,%s,%s::jsonb,%s,%s::jsonb,%s,%s,'cache621-authority',%s,1,'["cache621-authority"]')""",
              (scope['cache_key'],slot,scope['signal_input_hash'],json.dumps(scope),contracts.document_hash(judgment),json.dumps(judgment),jid,source,'sha256:'+'a'*64))
        before=scalar(cur,'SELECT jsonb_agg(to_jsonb(j) ORDER BY cache_key,authority_slot) FROM public.lab_arena_signal_judgments j')
        sql='SELECT public.lab_arena_claim_signal_judgment_v1(%s,%s,0,%s,%s::jsonb)'
        args=(leased_score['run_id'],lease_token_hash,index,json.dumps(scope))
        result=scalar(cur,sql,args)
        assert result['status']==('miss' if index==0 else 'hit'),result
        assert result['reference']['authority_slot']==1,result
        replay=scalar(cur,sql,args)
        assert replay==result
        assert scalar(cur,'SELECT jsonb_agg(to_jsonb(j) ORDER BY cache_key,authority_slot) FROM public.lab_arena_signal_judgments j')==before
