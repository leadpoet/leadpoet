"""Exact correction reuses saved executions and native floor-zero judging."""
from copy import deepcopy
import json
import re
from types import SimpleNamespace
from datetime import datetime, timezone
from tests.lab_arena.test_lab_arena_contracts import base_round_configuration

import pytest

from lab_arena.service import ArenaService
from lab_arena import contracts, scoring, company_judgments
from lab_arena.store import hash_lease_token
from tests.lab_arena import nov03_corrected_judge607_postgres_test as prior

SQL=prior.prior.SCRIPTS/'618-arena-2026-11-03-baseline-rejudge.sql'
ROUND=prior.ROUND
REQUEST='nov03-baseline-rejudge618-v1'
scalar=prior.scalar


@pytest.fixture
def prepared(request):
    generator=prior.database.__wrapped__(); pg,dsn=next(generator)
    store=None
    try:
        with pg.connect(**dsn) as conn:
            store,service,objects,corrected,_,_=prior.setup(conn,dsn,pg,empty=True)
            with conn.cursor() as cur:
                cur.execute(corrected)
                conn.commit()
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',output_ref='score8.json',result_doc='{\"terminal_status\":\"accepted\",\"company_judgment_consumed_indexes\":[]}' WHERE run_id=%s",(prior.RUN,))
                cur.execute("DELETE FROM public.lab_arena_runs WHERE submission_id='nov03-miner'")
                cur.execute("DELETE FROM public.lab_arena_submissions WHERE submission_id='nov03-miner'")
                participants=scalar(cur,'SELECT participants FROM public.lab_arena_rounds WHERE round_id=%s',(ROUND,))[:1]
                for group in range(10):
                    submission=f'rejudge-miner-{group}'; hotkey=prior.hotkey(submission)
                    participants.append(dict(submission_id=submission,miner_hotkey=hotkey,is_king=False))
                    cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status) VALUES (%s,%s,%s,'frozen')",(submission,ROUND,hotkey))
                    for position in range(10):
                        assignment=f'{ROUND}:{submission}:2:{position}'
                        cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,terminal_cause,output_ref,scoring_generation_floor) VALUES (%s,%s,%s,%s,%s,2,%s,1,'execute','accepted',7,'accepted',%s,0)",(assignment+':1',assignment,ROUND,submission,hotkey,position,assignment+'.json'))
                        if group<9:
                            cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,terminal_cause,output_ref,company_judgment_refs,result_doc,scored_run_id,scoring_generation_floor) VALUES (%s,%s,%s,%s,%s,2,%s,1,'score','accepted',7,'accepted',%s,'[]','{\"terminal_status\":\"accepted\"}',%s,0)",(assignment+':score:1',assignment+':score',ROUND,submission,hotkey,position,assignment+'-score.json',assignment+':1'))
                cur.execute("UPDATE public.lab_arena_runs SET per_icp_score=42 WHERE round_id=%s AND kind='execute'",(ROUND,))
                cur.execute("UPDATE public.lab_arena_rounds SET status='published',status_generation=10,stage_generation=7,participants=%s::jsonb,publication_doc='{\"original\":true}',king_outcome='no_king',published_at=now(),reward_basis_hash='sha256:'||repeat('a',64),reward_basis_doc='{\"original_reward\":true}',signing_key_doc='{\"original_key\":true}',effective_reward_epoch=26177,reward_activated_at=now() WHERE round_id=%s",(json.dumps(participants),ROUND))
                cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,guard_generation=759,guard_commitment='',owner_commitment='',guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]' WHERE singleton")
                configuration=base_round_configuration()
                configuration.update(scalar(cur,'SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s',(ROUND,)))
                configuration.update(round_id=ROUND,cost_per_company_microusd=800000,execution_icp_cap_microusd=4000000)
                configuration['schedule']={key:value.replace('2026-09-01','2026-11-02').replace('2026-09-02','2026-11-03') for key,value in base_round_configuration()['schedule'].items()}
                configuration=contracts.validate_round_configuration(configuration)
                cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s',(json.dumps(configuration),ROUND))
                cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit()
            if getattr(request,'param',False):
                objects['output0.json']=json.dumps({'schema_version':contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,'companies':prior._input()['companies']}).encode()
            row=store.get_round(ROUND)
            sources=store.list_runs(ROUND,stage=1,kind='execute')
            items=[dict(scored_run_id=s['run_id'],submission_id=s['submission_id'],icp_position=s['icp_position'],output_ref=s['output_ref']) for s in sources if s['icp_position']!=8]
            work=service._scoring_work_items(row,1,{'work_items':items})
            with conn.cursor() as cur:
                before=scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a WHERE round_id=%s',(ROUND,))
                hashes=scalar(cur,"SELECT jsonb_object_agg(run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')) FROM public.lab_arena_runs a WHERE round_id=%s",(ROUND,))
                accepted={key:hashes[key] for key,value in before.items() if value['status']=='accepted'}
                assert len(accepted)==210
                old_by_source={value['scored_run_id']:key for key,value in before.items() if value['kind']=='score' and value['status']=='accepted'}
                manifest=dict(round_sha256=scalar(cur,"SELECT encode(extensions.digest(to_jsonb(r)::TEXT,'sha256'),'hex') FROM public.lab_arena_rounds r WHERE round_id=%s",(ROUND,)),run_hashes=hashes,accepted_run_hashes=accepted,work_items=[dict(item,old_run_id=old_by_source[item['scored_run_id']]) for item in work])
            rendered=re.sub(r'\$manifest618\$.*?\$manifest618\$',lambda _: '$manifest618$'+json.dumps(manifest,separators=(',',':'),sort_keys=True)+'$manifest618$',SQL.read_text(),flags=re.S)
            # Real SQL and trigger paths; only the time-bound live fixture is
            # rendered into the future so this regression remains executable.
            rendered=rendered.replace('2026-11-03T23:45:00Z','2099-11-03T23:45:00Z').replace('2026-11-03T23:59:00Z','2099-11-03T23:59:00Z')
            yield SimpleNamespace(pg=pg,dsn=dsn,conn=conn,store=store,service=service,objects=objects,row=row,before=before,manifest=manifest,sql=rendered)
    finally:
        if store is not None: store.close()
        generator.close()


def all_rows(cur):
    return scalar(cur,'SELECT jsonb_object_agg(run_id,to_jsonb(a)) FROM public.lab_arena_runs a WHERE round_id=%s',(ROUND,))


def test_supersession_requires_receipt_then_preserves_all_payloads_and_replays(prepared):
    p=prepared
    with p.conn.cursor() as cur:
        old_id=p.manifest['work_items'][0]['old_run_id']
        cur.execute('SAVEPOINT immutable')
        with pytest.raises(p.pg.Error,match='terminal attempt is immutable'):
            cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed' WHERE run_id=%s",(old_id,))
        cur.execute('ROLLBACK TO SAVEPOINT immutable')
        ledger=scalar(cur,'SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l')
    p.conn.commit()
    with p.conn.cursor() as cur: cur.execute(p.sql)
    after=p.store.get_round(ROUND)
    assert after['status']=='scored' and after['scoring_generation_floor']==0
    assert after['publication_doc'] is None and after['published_at'] is None
    assert after['king_outcome'] is None and after['king_hotkey'] is None
    assert contracts.validate_round_configuration(after['configuration_doc'])==after['configuration_doc']
    assert after['configuration_doc']['schedule']['stage_1_scoring_close']==p.row['configuration_doc']['schedule']['stage_1_scoring_close']
    for key in ('reward_basis_doc','reward_basis_hash','signing_key_doc','effective_reward_epoch','reward_activated_at'):
        assert after[key]==p.row[key]
    with p.conn.cursor() as cur:
        rows=all_rows(cur)
        receipt=scalar(cur,'SELECT recovery_doc FROM public.lab_arena_scoring_supersessions WHERE round_id=%s AND request_id=%s',(ROUND,REQUEST))
        assert len(receipt['superseded_judgments'])==9
        assert len(rows)==len(p.before)+9
        old_ids={w['old_run_id'] for w in p.manifest['work_items']}
        source_ids={w['scored_run_id'] for w in p.manifest['work_items']}
        for key,old in p.before.items():
            now=rows[key]
            if key in old_ids:
                assert receipt['superseded_judgments'][key]==old
                assert now['status']=='failed' and now['terminal_cause']=='stage_closed'
                omitted={'status','terminal_cause','terminal_doc','updated_at'}
            elif key in source_ids:
                assert now['per_icp_score'] is None and now['qualification_doc'] is None
                omitted={'per_icp_score','qualification_doc','updated_at'}
            else:
                omitted=set()
            assert {k:v for k,v in now.items() if k not in omitted}=={k:v for k,v in old.items() if k not in omitted}
        assert scalar(cur,'SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l')==ledger
        assert scalar(cur,'SELECT public.lab_arena__all_judgments_ready_v1(%s)',(ROUND,)) is False
        cur.execute(p.sql)
        assert all_rows(cur)==rows
    selected=ArenaService._select_scoring_outputs(list(rows.values()),round_row=after)
    assert all(selected[w['scored_run_id']]['status']=='pending' for w in p.manifest['work_items'])
    assert not p.service.scoring_is_complete(ROUND,2)
    # Keep the native retry cooldown; age only disposable fixture rows.
    p.conn.commit()
    with p.conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=now()-interval '30 minutes' WHERE round_id=%s AND status='pending'",(ROUND,))
        cur.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit()
    assert prior.claim(p.conn)['status']=='leased'


def test_nine_native_judgments_complete_and_materialize_without_replaying_execution(prepared):
    p=prepared
    with p.conn.cursor() as cur: cur.execute(p.sql)
    p.conn.commit()
    with p.conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=now()-interval '30 minutes' WHERE round_id=%s AND status='pending'",(ROUND,))
        cur.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit()
    for _ in range(9):
        lease=prior.claim(p.conn)
        assert lease['status']=='leased'
        output=json.dumps({'schema_version':scoring.SCORING_OUTPUT_SCHEMA_VERSION,
            'scored_run_id':lease['scored_run_id'],'breakdowns':[],'company_judgments':[]}).encode()
        reference=lease['run_id']+'.json'
        # The same in-memory object store used by the native historical fixture.
        result=p.store.complete_attempt(run_id=lease['run_id'],lease_token_hash=hash_lease_token('corrected607'),
            result={'terminal_status':'accepted','company_judgment_consumed_indexes':[]},
            terminal_cause='accepted',output_ref=reference,output_hash=contracts.hash_bytes(output),
            company_judgment_evidence=[],completion_request_hash=contracts.document_hash({'run':lease['run_id']}))
        assert result['status']=='accepted'
        p.objects[reference]=output
    p.service._record_recovered_judgment_scores(p.store.get_round(ROUND))
    with p.conn.cursor() as cur:
        rows=all_rows(cur)
        for item in p.manifest['work_items']:
            assert rows[item['scored_run_id']]['per_icp_score']==0
        assert scalar(cur,"SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' AND status='accepted'",(ROUND,))==100
    assert prior.claim(p.conn)['status']=='no_pending'


@pytest.mark.parametrize('outcome',['crowned','no_king'])
def test_corrected_publication_can_set_new_decision_without_changing_signed_basis(prepared,outcome):
    """Exercise the actual write-once guard at its changed decision boundary.

    Other publication gates remain on the real table and are covered by their
    existing native publication tests; this temporary row isolates the latch.
    """
    p=prepared
    with p.conn.cursor() as cur: cur.execute(p.sql)
    p.conn.commit()
    with p.conn.cursor() as cur:
        cur.execute('CREATE TEMP TABLE correction_guard AS SELECT * FROM public.lab_arena_rounds WITH NO DATA')
        cur.execute('CREATE TRIGGER correction_guard_write_once BEFORE UPDATE ON correction_guard FOR EACH ROW EXECUTE FUNCTION public.lab_arena_rounds_write_once_v1()')
        stale={**p.row,'status':'scored','publication_doc':None,'published_at':None,'king_outcome':'no_king'}
        cur.execute('INSERT INTO correction_guard SELECT (jsonb_populate_record(NULL::public.lab_arena_rounds,%s::jsonb)).*',(json.dumps(stale,default=str),))
        cur.execute('SAVEPOINT stale_decision')
        with pytest.raises(p.pg.Error,match='write-once'):
            cur.execute("UPDATE correction_guard SET status='published',king_outcome='crowned'")
        cur.execute('ROLLBACK TO SAVEPOINT stale_decision')
        cur.execute('TRUNCATE correction_guard')
        cur.execute('INSERT INTO correction_guard SELECT * FROM public.lab_arena_rounds WHERE round_id=%s',(ROUND,))
        cur.execute("UPDATE correction_guard SET status='published',status_generation=status_generation+1,publication_doc=jsonb_build_object('king_decision',jsonb_build_object('outcome',%s)),published_at=now(),king_outcome=%s",(outcome,outcome))
        assert scalar(cur,'SELECT reward_basis_doc FROM correction_guard')==p.row['reward_basis_doc']
        assert scalar(cur,'SELECT reward_basis_hash FROM correction_guard')==p.row['reward_basis_hash']


@pytest.mark.parametrize('fault',['round_hash','run_hash','wrong_source','icp8','duplicate','old_refs','clock','guard'])
def test_wrong_authority_rolls_back_every_change(prepared,fault):
    p=prepared; manifest=deepcopy(p.manifest); sql=p.sql
    if fault=='round_hash': manifest['round_sha256']='0'*64
    if fault=='run_hash': manifest['run_hashes'][next(iter(manifest['run_hashes']))]='0'*64
    if fault=='wrong_source': manifest['work_items'][0]['scored_run_id']='wrong-round:source'
    if fault=='icp8': manifest['work_items'][0]['old_run_id']=prior.RUN
    if fault=='duplicate': manifest['work_items'][1]=deepcopy(manifest['work_items'][0])
    if fault=='old_refs': manifest['work_items'][0]['company_judgment_refs']=[{'invalid':True}]
    if fault=='clock': sql=sql.replace('2099-11-03T23:45:00Z','2000-11-03T23:45:00Z')
    if fault=='guard':
        with p.conn.cursor() as cur: cur.execute('UPDATE public.lab_arena_restart_claim_control SET guard_generation=760 WHERE singleton')
        p.conn.commit()
    sql=re.sub(r'\$manifest618\$.*?\$manifest618\$',lambda _: '$manifest618$'+json.dumps(manifest,separators=(',',':'),sort_keys=True)+'$manifest618$',sql,flags=re.S)
    with p.conn.cursor() as cur:
        with pytest.raises(p.pg.Error): cur.execute(sql)
    p.conn.rollback()
    with p.conn.cursor() as cur:
        assert all_rows(cur)==p.before
        assert scalar(cur,'SELECT count(*) FROM public.lab_arena_scoring_supersessions WHERE round_id=%s AND request_id=%s',(ROUND,REQUEST))==0
    assert p.store.get_round(ROUND)==p.row


@pytest.mark.parametrize('prepared',[True],indirect=True)
def test_nonempty_native_lease_uses_valid_configuration_after_expired_stage_one(prepared):
    p=prepared
    assert datetime.fromisoformat(p.row['configuration_doc']['schedule']['stage_1_scoring_close'].replace('Z','+00:00'))<datetime.now(timezone.utc)
    with p.conn.cursor() as cur: cur.execute(p.sql)
    p.conn.commit()
    current=p.store.get_round(ROUND)
    assert contracts.validate_round_configuration(current['configuration_doc'])==current['configuration_doc']
    invalid=deepcopy(current['configuration_doc'])
    invalid['schedule']['stage_1_scoring_close']=invalid['schedule']['final_scoring_close']
    with pytest.raises(contracts.ArenaContractError,match='strictly increasing'):
        contracts.validate_round_configuration(invalid)
    with p.conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=now()-interval '30 minutes' WHERE round_id=%s AND status='pending'",(ROUND,))
        cur.execute('SET LOCAL session_replication_role=origin')
    p.conn.commit()
    lease=prior.claim(p.conn)
    assert lease['status']=='leased' and lease['icp_position']==0
    context=company_judgments.validate_lease_context(lease['company_judgment_cache'])
    assert context['misses']
