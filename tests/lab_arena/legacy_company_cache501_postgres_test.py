"""Future intrinsic company judgments use the existing PostgreSQL authority."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena import company_judgments_postgres_test as original
from tests.lab_arena import flat_per_icp_cost500_postgres_test as latest
from tests.lab_arena import bounded_baseline_recovery470_postgres_test as baseline

SQL = Path(__file__).parents[2] / 'scripts/501-lab-arena-legacy-company-judgment-cache.sql'
POLICY = 'company_judgment_v1'


@pytest.fixture(scope='module')
def database():
    gen = latest.database_with_flat_cost_migration()
    db = next(gen)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            before = latest.catalog(cur)
            rows = latest.rows(cur)
            cur.execute('CREATE ROLE company501_migrator NOLOGIN NOSUPERUSER INHERIT')
            cur.execute('GRANT lab_arena_owner TO company501_migrator')
            cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO company501_migrator WITH GRANT OPTION')
            cur.execute('ALTER DEFAULT PRIVILEGES FOR ROLE company501_migrator IN SCHEMA public '
                        'GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
            conn.commit()
            broken=SQL.read_text().replace("  IF v_round.configuration_doc ->> 'integrity_policy'",
                "  IF v_round.configuration_doc ->> 'wrong_policy'",1)
            with pytest.raises(db[0].Error,match='patch shape differs'):
                cur.execute(broken)
            conn.rollback()
            assert latest.catalog(cur)==before and latest.rows(cur)==rows
            cur.execute('GRANT EXECUTE ON FUNCTION public.lab_arena_open_scoring_v3(text,smallint,jsonb) TO anon')
            with pytest.raises(db[0].Error,match='security shape differs'):
                cur.execute(SQL.read_text())
            conn.rollback()
            assert latest.catalog(cur)==before and latest.rows(cur)==rows
            cur.execute('SET ROLE company501_migrator')
            cur.execute(SQL.read_text())
            after = latest.catalog(cur)
            cur.execute(SQL.read_text())
            assert latest.catalog(cur) == after
            cur.execute('RESET ROLE')
            assert latest.rows(cur) == rows
            by_name = {r[0]: r for r in after}
            for old in before:
                assert old[2:] == by_name[old[0]][2:]
            for name in ('lab_arena__company_judgment_policy_v1(public.lab_arena_rounds)',
                         'lab_arena__company_judgment_refs_v1(public.lab_arena_rounds,jsonb)'):
                for role in ('anon', 'authenticated', 'service_role', 'lab_arena_service'):
                    cur.execute('SELECT has_function_privilege(%s,%s,\'EXECUTE\')', (role, 'public.' + name))
                    assert cur.fetchone() == (False,)
        yield db
    finally:
        gen.close()


@pytest.fixture()
def store(database):
    store = ArenaStore(PsycopgTransport(lambda: database[0].connect(**database[1])))
    store._test_database = database
    try:
        yield store
    finally:
        store.close()


def open_new(store, *, round_id, markers_by_submission, policy=POLICY, quality=False):
    # Seed only immutable test inputs, then exercise public owner RPCs with
    # ordinary service-role access. Production tables are never involved.
    db = store._test_database
    cfg = baseline.cutoff.fixture_configuration(round_id)
    from datetime import datetime,timedelta,timezone
    cfg['schedule']['stage_1_scoring_close']=(datetime.now(timezone.utc)+timedelta(hours=3)).isoformat().replace('+00:00','Z')
    cfg['schedule']['final_scoring_close']=(datetime.now(timezone.utc)+timedelta(hours=5)).isoformat().replace('+00:00','Z')
    cfg['parallel_twenty_icp_execution']=False
    cfg['max_attempts_per_assignment']=2
    cfg.pop('contact_policy',None)
    cfg['integrity_policy'] = 'arena_integrity_v1'
    from lab_arena import scoring
    cfg['scorer_policy'] = scoring.build_scorer_policy(scoring_adapter_version='qualification_integrity_v2',company_quality=quality)
    if quality:
        cfg['company_quality_policy'] = 'company_quality_v1'
    else:
        cfg['company_judgment_policy'] = policy
        cfg['scorer_policy']['company_judgment_policy'] = policy
    participants=[]; items=[]
    for index, markers in enumerate(markers_by_submission):
        sub = round_id[-2:]+'-sub-'+str(index)
        miner = original.hotkey(sub)
        source = f'{round_id}:{sub}:1:0:1'
        participants.append(dict(submission_id=sub,miner_hotkey=miner,is_king=False))
        refs = [(OLD_SCOPE if quality else scope_original_safe)(round_id,m,i) for i,m in enumerate(markers)]
        for ref in refs:
            doc=ref['scope_doc']
            doc.update(evaluation_date='2099-10-20',scorer_image_digest=cfg['scorer_image_digest'],scorer_image_reference=cfg['scorer_image_reference'])
            doc.pop('cache_key');ref['cache_key']=doc['cache_key']=contracts.document_hash(doc)
        items.append(dict(scored_run_id=source,submission_id=sub,icp_position=0,
                          output_ref='fixture/'+source,company_judgment_refs=refs))
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('TRUNCATE public.lab_arena_rounds,public.lab_arena_ledger,public.lab_arena_trajectory_events RESTART IDENTITY CASCADE')
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,guard_commitment='',owner_commitment='',guard_expires_at=NULL,captured_leases='[]'")
        cur.execute("INSERT INTO public.lab_arena_rounds(round_id,status,stage_generation,configuration_doc,participants,benchmark_ref,evaluation_date,rewards_enabled,stage1_scoring_plan_doc) VALUES (%s,'stage1_closed',1,%s::jsonb,%s::jsonb,'benchmark','2099-10-20',false,%s::jsonb)",
                    (round_id,json.dumps(cfg),json.dumps(participants),json.dumps({'work_items':items})))
        for p,item in zip(participants,items):
            cur.execute("INSERT INTO public.lab_arena_submissions(submission_id,round_id,miner_hotkey,status,is_king,source_ref,source_size_bytes,code_review_status,code_review_doc,code_review_claim,code_review_started_at,code_review_attempts) VALUES (%s,%s,%s,'frozen',false,'arena/'||%s||'/sources/'||%s||'.tar.gz',100,'passed','{}','sha256:'||repeat('a',64),now(),1)", (p['submission_id'],round_id,p['miner_hotkey'],round_id,p['submission_id']))
            cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,kind,status,stage_generation,terminal_cause,output_ref) VALUES (%s,%s,%s,%s,%s,1,0,1,'execute','accepted',1,'accepted',%s)",
                        (item['scored_run_id'],item['scored_run_id'][:-2],round_id,p['submission_id'],p['miner_hotkey'],item['output_ref']))
        cur.execute('SET LOCAL session_replication_role=origin')
    assert store.open_scoring(round_id,1,items,company_quality_cache=True)['status']=='ok'
    return participants,items


OLD_SCOPE = original._scope

def scope_original_safe(round_id, marker, index):
    ref = OLD_SCOPE(round_id, marker, index)
    doc = ref['scope_doc']
    doc.pop('company_quality_policy')
    doc['company_judgment_policy'] = POLICY
    doc.pop('cache_key')
    key = contracts.document_hash(doc)
    ref['cache_key'] = doc['cache_key'] = key
    return ref


def complete_subset(store, leased, token, refs, *, limit=None, terminal='accepted', tamper=None, token_hash=None):
    original_complete = store.complete_attempt
    def complete(**kwargs):
        rows = kwargs['company_judgment_evidence']
        order={r['cache_key']:r['company_index'] for r in refs}
        rows.sort(key=lambda r:order[r['cache_key']])
        if limit is not None:
            rows = rows[:limit]
        for row in rows:
            row['evidence_doc']['raw_judgment']['company_judgment_policy'] = POLICY
            row['evidence_hash'] = contracts.document_hash(row['evidence_doc'])
        if tamper:
            tamper(rows)
        kwargs['company_judgment_evidence'] = rows
        if token_hash is not None:kwargs['lease_token_hash']=token_hash
        kwargs['terminal_cause'] = terminal
        kwargs['result'] = {'terminal_status': terminal}
        return original_complete(**kwargs)
    from tests.test_arena_score_integrity import _positive_breakdown
    from qualification.scoring.competition import raw_company_judgment
    def raw(marker):
        return {**raw_company_judgment(_positive_breakdown(marker,marker+'.example')),
                'company_identity_key':'domain:'+marker+'.example|name:'+marker,
                'company_identity_alias_keys':['domain:'+marker+'.example|name:'+marker],
                'company_judgment_policy':POLICY}
    with patch.object(store, 'complete_attempt', complete), patch.object(original,'_raw',raw):
        return original._complete(store, leased, token, refs=refs)


def test_new_policy_partial_reuse_subset_and_original_work(store, database):
    rid = 'arena-2026-09-11-n1'
    _, items = open_new(store, round_id=rid, markers_by_submission=[['same','skip'], ['same','different']])
    runner = original.hotkey('n1-validator')
    lease, token, _, _ = original.claim(store, rid, runner, excluded=[runner])
    assert lease['status']=='leased',lease
    refs = next(i['company_judgment_refs'] for i in items if i['submission_id'] == lease['submission_id'])
    assert complete_subset(store, lease, token, refs, limit=1)['company_judgments_stored'] == 1
    assert store.get_run(lease['run_id'])['participation_accepted_at'] is not None
    other, token, _, _ = original.claim(store, rid, runner, excluded=[runner])
    assert len(other['company_judgment_cache']['hits']) == 1
    refs = next(i['company_judgment_refs'] for i in items if i['submission_id'] == other['submission_id'])
    assert complete_subset(store, other, token, refs, limit=0)['company_judgments_stored'] == 0
    assert store.get_run(other['run_id'])['participation_accepted_at'] is None
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SELECT count(*) FROM public.lab_arena_company_judgment_reservations WHERE run_id=ANY(%s)', ([lease['run_id'], other['run_id']],))
        assert cur.fetchone() == (0,)


@pytest.mark.parametrize('fault', ['key','runner','raw_context','slot'])
def test_forged_evidence_cannot_persist(store, database, fault):
    rid = 'arena-2026-09-11-n' + str(['key','runner','raw_context','slot'].index(fault)+2)
    _, items = open_new(store, round_id=rid, markers_by_submission=[['a','b']])
    runner = original.hotkey(rid)
    lease, token, _, _ = original.claim(store, rid, runner, excluded=[runner])
    def tamper(rows):
        if fault == 'key': rows[0]['cache_key'] = 'sha256:'+'f'*64
        elif fault == 'runner': rows[0]['evidence_doc']['source_runner_hotkey'] = 'forged'
        elif fault == 'slot': rows[0]['authority_slot'] += 1
        else: rows[0]['evidence_doc']['raw_judgment']['company_qualified'] = True
    with pytest.raises(ArenaStoreError, match='evidence_invalid'):
        complete_subset(store, lease, token, items[0]['company_judgment_refs'], limit=1, tamper=tamper)
    assert store.get_run(lease['run_id'])['status'] == 'leased'
    assert store.get_company_judgment(items[0]['company_judgment_refs'][0]['cache_key'], 0) is None


def test_unknown_policy_cannot_open(store, database):
    with pytest.raises(ArenaStoreError, match='company_judgment_policy_invalid'):
        open_new(store, round_id='arena-2026-09-11-n6', markers_by_submission=[['a']], policy='unknown')


def test_existing_quality_still_requires_all_misses(store,database):
    _,items=open_new(store,round_id='arena-2026-09-11-n7',markers_by_submission=[['a','b']],quality=True)
    runner=original.hotkey('quality')
    lease,token,_,_=original.claim(store,'arena-2026-09-11-n7',runner,excluded=[runner])
    original_complete=store.complete_attempt
    def omit(**kwargs):
        kwargs['company_judgment_evidence']=kwargs['company_judgment_evidence'][:1]
        return original_complete(**kwargs)
    with patch.object(store,'complete_attempt',omit),pytest.raises(ArenaStoreError,match='evidence_incomplete'):
        original._complete(store,lease,token,refs=items[0]['company_judgment_refs'])
    assert original._complete(store,lease,token,refs=items[0]['company_judgment_refs'])['company_judgments_stored']==2


def seed_commit(conn, *, marker=False):
    _, cfg = baseline.seed(conn)
    cfg.pop('contact_policy',None)
    cfg['scorer_policy']['scoring_adapter_version']='qualification_integrity_v2'
    cfg['schedule']['submission_open']='2030-01-01T00:00:00Z'
    cfg['schedule']['submission_cutoff']='2030-01-02T00:00:00Z'
    if marker:
        cfg['company_judgment_policy']=POLICY
        cfg['scorer_policy']['company_judgment_policy']=POLICY
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('TRUNCATE public.lab_arena_runs CASCADE')
        cur.execute("UPDATE public.lab_arena_rounds SET status='open',configuration_doc=%s::jsonb,benchmark_ref=NULL,evaluation_date=NULL,icp_set_date=NULL,stage1_scoring_plan_doc=NULL",(json.dumps(cfg),))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    return cfg


def commit_v4(conn, *, catalog=None, policy=POLICY):
    participants=[dict(submission_id=baseline.SUB,miner_hotkey=baseline.MINER,is_king=True)]
    return baseline.rpc(conn,'lab_arena_commit_round_v4(%s,%s::jsonb,%s,%s,%s::date,%s,%s,%s::jsonb,%s)',
        (baseline.ROUND,json.dumps(participants),'benchmark','2030-01-02','2030-01-01',
         'sha256:'+'b'*64,'example/scorer@sha256:'+'b'*64,json.dumps(catalog) if catalog else None,policy))


@pytest.mark.parametrize('catalog_enabled',[False,True])
@pytest.mark.parametrize('marker',[False,True])
def test_commit_atomically_freezes_marker_and_pin_without_other_policy_change(database,catalog_enabled,marker):
    with database[0].connect(**database[1]) as conn:
        cfg=seed_commit(conn,marker=marker)
        catalog=dict(schema_version='leadpoet.lab_arena.deepline_catalog.v1',tools=[{'name':'fixture'}],
                     catalog_hash='a'*64,policy_version='leadpoet.lab_arena.deepline_company_research.v1',allow_people=False) if catalog_enabled else None
        assert commit_v4(conn,catalog=catalog)['status']=='ok'
        expected=copy.deepcopy(cfg)
        expected.update(company_judgment_policy=POLICY,scorer_image_digest='sha256:'+'b'*64,
                        scorer_image_reference='example/scorer@sha256:'+'b'*64,
                        infrastructure_retry_cost_policy='trusted_infrastructure_v1')
        expected['scorer_policy']['company_judgment_policy']=POLICY
        if catalog:
            expected['deepline_catalog']=catalog
            expected['call_quotas']['deepline']=0
            expected['scoring_call_quotas']['deepline']=0
        with conn.cursor() as cur:
            cur.execute('SELECT configuration_doc,status FROM public.lab_arena_rounds')
            assert cur.fetchone()==(expected,'committed')
            before=latest.rows(cur)
        assert commit_v4(conn,catalog=catalog)['status']=='stale'
        with conn.cursor() as cur:
            assert latest.rows(cur)==before


@pytest.mark.parametrize('fault',['unknown','mismatch','adapter','contacts','quality','started','unrelated','open_open'])
def test_commit_rejects_unbound_or_started_policy_without_changes(database,fault):
    with database[0].connect(**database[1]) as conn:
        cfg=seed_commit(conn)
        if fault=='mismatch':cfg['company_judgment_policy']=POLICY
        elif fault=='adapter':cfg['scorer_policy']['scoring_adapter_version']='qualification_contacts_v3'
        elif fault=='contacts':cfg['contact_policy']='contacts_v1'
        elif fault=='quality':cfg['company_quality_policy']='company_quality_v1'
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb',(json.dumps(cfg),))
            if fault=='started':cur.execute("UPDATE public.lab_arena_rounds SET status='stage1'")
            cur.execute('SET LOCAL session_replication_role=origin')
            before=latest.rows(cur)
        conn.commit()
        if fault in ('unrelated','open_open'):
            changed=copy.deepcopy(cfg)
            changed['company_judgment_policy']=POLICY
            changed['scorer_policy']['company_judgment_policy']=POLICY
            if fault=='unrelated':changed['scorer_policy']['fp_penalty_points']=999
            with pytest.raises(database[0].Error,match='write-once'):
                with conn.cursor() as cur:
                    cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb,status=%s',
                                (json.dumps(changed),'open' if fault=='open_open' else 'committed'))
            conn.rollback()
        elif fault=='started':
            assert commit_v4(conn)['status']=='stale'
        else:
            with pytest.raises(database[0].Error,match='company_judgment_commit_invalid'):
                commit_v4(conn,policy='unknown' if fault=='unknown' else POLICY)
            conn.rollback()
        with conn.cursor() as cur:
            assert latest.rows(cur)==before


def test_new_mode_concurrent_overlapping_keys_and_authority_slots(store,database):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    rid='arena-2026-09-11-n8'
    participants,items=open_new(store,round_id=rid,markers_by_submission=[['same'],['same'],['independent']])
    barrier=threading.Barrier(2)
    def claim_one(index):
        target=ArenaStore(PsycopgTransport(lambda:database[0].connect(**database[1])))
        try:
            runner=original.hotkey('race'+str(index));barrier.wait(timeout=10)
            return original.claim(target,rid,runner,excluded=[runner])
        finally:target.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(claim_one,(0,1)))
    leases=[r[0] for r in results if r[0]['status']=='leased']
    assert len(leases)==2
    assert len({l['company_judgment_cache']['misses'][0]['cache_key'] for l in leases})==2
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute('SELECT count(*),count(DISTINCT cache_key) FROM public.lab_arena_company_judgment_reservations')
        assert cur.fetchone()==(2,2)
        cur.execute('SELECT count(*) FROM public.lab_arena_ledger')
        assert cur.fetchone()==(0,)
    # Use a separate two-member case to prove incompatible authorities retain
    # separate immutable slots rather than replacing an earlier judgment.
    participants,items=open_new(store,round_id=rid,markers_by_submission=[['same'],['same']])
    runner=original.hotkey('authority-one')
    first,token,_,_=original.claim(store,rid,runner,excluded=[runner,participants[1]['miner_hotkey']])
    refs=next(i['company_judgment_refs'] for i in items if i['submission_id']==first['submission_id'])
    assert complete_subset(store,first,token,refs)['company_judgments_stored']==1
    evidence_before=store.get_company_judgment(refs[0]['cache_key'],0)
    runner2=original.hotkey('authority-two')
    second,token2,_,_=original.claim(store,rid,runner2,excluded=[runner2])
    assert second['company_judgment_cache']['hits']==[]
    assert second['company_judgment_cache']['misses'][0]['authority_slot']==1
    refs2=next(i['company_judgment_refs'] for i in items if i['submission_id']==second['submission_id'])
    assert complete_subset(store,second,token2,refs2)['company_judgments_stored']==1
    assert store.get_company_judgment(refs[0]['cache_key'],0)==evidence_before


def test_partial_failure_persists_completed_only_and_retry_reuses_without_new_credit(store,database):
    rid='arena-2026-09-11-n9'
    _,items=open_new(store,round_id=rid,markers_by_submission=[['a','b','c']])
    runner=original.hotkey('partial-one')
    first,token,_,_=original.claim(store,rid,runner,excluded=[runner])
    refs=items[0]['company_judgment_refs']
    done=complete_subset(store,first,token,refs,limit=2,terminal='judge_error')
    assert done['company_judgments_stored']==2
    old=store.get_run(first['run_id'])
    assert old['participation_accepted_at'] is None
    assert store.get_company_judgment(refs[2]['cache_key'],0) is None
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE status='pending'")
        cur.execute('SET LOCAL session_replication_role=origin')
    runner2=original.hotkey('partial-two')
    retry,token2,_,_=original.claim(store,rid,runner2,excluded=[runner2])
    assert retry['attempt']==2
    assert len(retry['company_judgment_cache']['hits'])==2
    assert len(retry['company_judgment_cache']['misses'])==1
    assert complete_subset(store,retry,token2,refs,limit=0)['company_judgments_stored']==0
    assert store.get_run(retry['run_id'])['participation_accepted_at'] is None
    assert store.get_run(first['run_id'])==old


def test_expiry_releases_reservations_retains_refs_and_rejects_stale_evidence(store,database):
    rid='arena-2026-09-11-e1'
    _,items=open_new(store,round_id=rid,markers_by_submission=[['a']])
    runner=original.hotkey('expiry-one')
    first,token,_,_=original.claim(store,rid,runner,excluded=[runner])
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s",(first['run_id'],))
    with pytest.raises(ArenaStoreError,match='reservation_stale'):
        complete_subset(store,first,token,items[0]['company_judgment_refs'])
    assert store.expire_leases(rid)['retried']==1
    runs=store.list_runs(rid,kind='score')
    retry=next(r for r in runs if r['attempt']==2)
    assert retry['company_judgment_refs']==items[0]['company_judgment_refs']
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute('SELECT count(*) FROM public.lab_arena_company_judgment_reservations')
        assert cur.fetchone()==(0,)


def baseline_claim(conn,suffix='a'):
    return baseline.rpc(conn,'lab_arena_claim_assignment(%s,%s,20,20,ARRAY[%s]::text[],%s,%s,%s,4500)',
        (baseline.ROUND,baseline.RUNNER,baseline.RUNNER,suffix*32,'sha256:'+suffix*64,baseline.TOKEN))


def recovered_company_item(conn,source):
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('SELECT configuration_doc,evaluation_date,arena_network_name,arena_netuid FROM public.lab_arena_rounds')
        cfg,date,network,netuid=cur.fetchone()
        cfg.pop('contact_policy',None)
        cfg['scorer_policy']['scoring_adapter_version']='qualification_integrity_v2'
        cfg['company_judgment_policy']=POLICY
        cfg['scorer_policy']['company_judgment_policy']=POLICY
        cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb',(json.dumps(cfg),))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    ref=scope_original_safe(baseline.ROUND,'recovered',0)
    doc=ref['scope_doc']
    doc.update(evaluation_date=date,network_name=network,netuid=netuid,
               scorer_image_digest=cfg['scorer_image_digest'],scorer_image_reference=cfg['scorer_image_reference'])
    doc.pop('cache_key');ref['cache_key']=doc['cache_key']=contracts.document_hash(doc)
    return dict(scored_run_id=source['run_id'],submission_id=baseline.SUB,icp_position=0,
                output_ref='new-output',company_judgment_refs=[ref])


@pytest.mark.parametrize('fault',[None,'image','refs','credential','model_error'])
def test_recovered_baseline_company_refs_and_member_evidence_keep_recovery_gates(database,store,fault):
    from tests.lab_arena.recovered_execution_judge473_postgres_test import cool
    with database[0].connect(**database[1]) as conn:
        baseline.seed(conn)
        assert baseline.recover(conn)['queued']==1
        cool(conn)
        source=baseline_claim(conn)
        assert source['status']=='leased'
        assert baseline.rpc(conn,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',
            (source['run_id'],baseline.TOKEN,json.dumps({'terminal_status':'accepted'}),'accepted','new-output'))['status']=='accepted'
        item=recovered_company_item(conn,source)
        if fault in ('credential','model_error'):
            with conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                if fault=='credential':
                    cur.execute("UPDATE public.lab_arena_runs SET terminal_cause='credential_error',result_doc='{\"terminal_status\":\"credential_error\"}' WHERE kind='execute' AND attempt=2")
                else:
                    # Accepted source remains immutable; a new live provider
                    # liability must continue to block further infrastructure.
                    cur.execute("UPDATE public.lab_arena_runs SET terminal_cause='model_error',result_doc='{\"terminal_status\":\"model_error\"}' WHERE kind='execute' AND attempt=2")
                cur.execute('SET LOCAL session_replication_role=origin')
            conn.commit()
            with pytest.raises(database[0].Error,match='recovery_scoring_item_invalid'):
                baseline.recover(conn,[item])
            conn.rollback();return
        assert baseline.recover(conn,[item])['queued']==1
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_scoring',stage_generation=7")
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert baseline.recover(conn)['queued']==0
        lease=baseline_claim(conn,suffix='b')
        assert lease['status']=='leased' and lease['company_judgment_cache']['misses']
        assert complete_subset(store,lease,'unused',item['company_judgment_refs'],terminal='judge_error',token_hash=baseline.TOKEN)['confirmation_attempt']==2
        for attempt in (2,3):
            cool(conn)
            lease=baseline_claim(conn,suffix=chr(ord('b')+attempt))
            assert lease['attempt']==attempt and len(lease['company_judgment_cache']['hits'])==1
            done=baseline.rpc(conn,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',
                (lease['run_id'],baseline.TOKEN,json.dumps({'terminal_status':'judge_error'}),'judge_error',''))
            assert done['confirmation_attempt']==attempt+1
        cool(conn)
        lease=baseline_claim(conn,suffix='f')
        assert lease['attempt']==4 and len(lease['company_judgment_cache']['hits'])==1
        assert complete_subset(store,lease,'unused',item['company_judgment_refs'],token_hash=baseline.TOKEN)['status']=='accepted'
        assert store.get_run(lease['run_id'])['participation_accepted_at'] is None
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_judged'")
            if fault=='image':
                cur.execute("UPDATE public.lab_arena_runs SET company_judgment_refs=jsonb_set(company_judgment_refs,'{0,scope_doc,scorer_image_digest}','\"wrong\"'::jsonb) WHERE run_id=%s",(lease['run_id'],))
            elif fault=='refs':
                cur.execute("UPDATE public.lab_arena_runs SET company_judgment_refs=jsonb_set(company_judgment_refs,'{0,scope_doc,company_input_hash}','\"sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff\"'::jsonb) WHERE run_id=%s",(lease['run_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
            before=latest.rows(cur)
        conn.commit()
        args=(baseline.ROUND,json.dumps([{'run_id':source['run_id'],'per_icp_score':0,'qualification_doc':{'companies':[]}}]))
        if fault:
            with pytest.raises(database[0].Error,match='late_baseline_score_invalid'):
                baseline.rpc(conn,'lab_arena_record_run_scores(%s,1::smallint,%s::jsonb)',args)
            conn.rollback()
            with conn.cursor() as cur:assert latest.rows(cur)==before
        else:
            assert baseline.rpc(conn,'lab_arena_record_run_scores(%s,1::smallint,%s::jsonb)',args)['status']=='ok'


@pytest.mark.parametrize('fault',[None,'unproved','completion_stamp','claim_binding','deadline','wrong_owner'])
def test_company_partial_credit_retry_keeps_existing_owner_proof_binding_and_deadline(store,database,fault):
    rid='arena-2026-09-11-c5'
    participants,items=open_new(store,round_id=rid,markers_by_submission=[['a','b']])
    runner=original.hotkey('credit-one')
    lease,token,_,_=original.claim(store,rid,runner,excluded=[runner])
    response=dict(status=402,call_succeeded=False,credit_failure_proof=dict(
        schema_version='leadpoet.lab_arena.credit_failure_proof.v1',provider='openrouter',
        reason='out_of_credit',provider_status=402,actual_microusd=0))
    if fault=='unproved':response.pop('credit_failure_proof')
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,terminal_response) VALUES ('settlement',%s,%s,%s,%s,1,%s,'openrouter','openrouter.chat','miner_key',0,%s::jsonb)",
            (lease['miner_hotkey'],rid,lease['submission_id'],lease['run_id'],contracts.document_hash('credit'),json.dumps(response)))
    assert complete_subset(store,lease,token,items[0]['company_judgment_refs'],limit=1,terminal='credential_error')['status']=='failed'
    with database[0].connect(**database[1]) as conn:
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            # No ordinary continuation may mask whether owner retry admits the
            # bound partial artifact. Remove only fixture-generated pending rows.
            cur.execute("DELETE FROM public.lab_arena_runs WHERE kind='score' AND attempt=2")
            if fault=='completion_stamp':
                cur.execute("UPDATE public.lab_arena_runs SET result_doc=result_doc-'company_judgment_completion_request_hash' WHERE run_id=%s",(lease['run_id'],))
            elif fault=='claim_binding':
                cur.execute("UPDATE public.lab_arena_company_judgments SET evidence_doc=jsonb_set(evidence_doc,'{source_claim_request_hash}',%s::jsonb)",(json.dumps('sha256:'+'f'*64),))
            elif fault=='deadline':
                cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,stage_1_scoring_close}','\"2000-01-01T00:00:00Z\"'::jsonb)")
            cur.execute('SET LOCAL session_replication_role=origin')
            before=latest.rows(cur)
        conn.commit()
        request='sha256:'+'d'*64
        owner=original.hotkey('wrong-owner') if fault=='wrong_owner' else participants[0]['miner_hotkey']
        result=baseline.rpc(conn,'lab_arena_retry_credit_failures_v1(%s,%s,%s,%s)',(rid,lease['submission_id'],owner,request))
        if fault:
            assert result['status']=='no_eligible',result
            with conn.cursor() as cur:assert latest.rows(cur)==before
        else:
            assert result=={'status':'queued','requeued_count':1}
            assert baseline.rpc(conn,'lab_arena_retry_credit_failures_v1(%s,%s,%s,%s)',(rid,lease['submission_id'],owner,request))=={'status':'replayed','requeued_count':1}
            from tests.lab_arena.recovered_execution_judge473_postgres_test import cool
            cool(conn)
            runner2=original.hotkey('credit-two')
            retry,_,_,_=original.claim(store,rid,runner2,excluded=[runner2])
            assert retry['attempt']==2
            assert len(retry['company_judgment_cache']['hits'])==1
            assert len(retry['company_judgment_cache']['misses'])==1


def test_signed_capacity_above_task_round_cap_can_hold_only_one_lease(store,database):
    rid='arena-2026-09-11-b1'
    open_new(store,round_id=rid,markers_by_submission=[['a'],['b']])
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{runner_slot_ceiling}','1'::jsonb)")
        cur.execute('SET LOCAL session_replication_role=origin')
    runner=original.hotkey('bounded-canary-capacity')
    first,_,_,_=original.claim(store,rid,runner,parallelism=20,ceiling=1,excluded=[runner])
    assert first['status']=='leased'
    second,_,_,_=original.claim(store,rid,runner,parallelism=20,ceiling=1,excluded=[runner])
    assert second['status']=='no_free_slot'
    assert len([r for r in store.list_runs(rid,kind='score') if r['status']=='leased'])==1


def test_whole_item_evidence_preserves_exact_database_authority_order(store,database):
    from lab_arena import judgment_cache,scoring
    from lab_arena.store import hash_lease_token
    from tests.lab_arena import judgment_cache_postgres_test as whole
    rid='arena-2026-09-11-w1'
    participants,company_items=open_new(store,round_id=rid,markers_by_submission=[['a']])
    source=company_items[0]
    scope=whole._scope(rid,source['scored_run_id'],0,'a')
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("DELETE FROM public.lab_arena_runs WHERE kind='score'")
        cur.execute('SELECT configuration_doc FROM public.lab_arena_rounds')
        cfg=cur.fetchone()[0]
        cfg.pop('company_judgment_policy')
        cfg['scorer_policy'].pop('company_judgment_policy')
        cur.execute("UPDATE public.lab_arena_rounds SET status='stage1_closed',configuration_doc=%s::jsonb",(json.dumps(cfg),))
        cur.execute('SET LOCAL session_replication_role=origin')
    scope.pop('cache_key')
    scope.update(evaluation_date='2099-10-20',scorer_image_digest=cfg['scorer_image_digest'],scorer_image_reference=cfg['scorer_image_reference'])
    scope['cache_key']=contracts.document_hash(scope)
    item={k:source[k] for k in ('scored_run_id','submission_id','icp_position','output_ref')}
    item.update(judgment_cache_key=scope['cache_key'],judgment_scope_doc=scope,
                judgment_input_hash=scope['scoring_input_hash'],judgment_group_leader=True,
                judgment_group_miner_hotkeys=[participants[0]['miner_hotkey']])
    assert store.open_scoring(rid,1,[item],integrity_cache=True)['status']=='ok'
    runner=original.hotkey('authority-one')
    # These two SS58 identities deliberately have different en_US.utf8 and C
    # ordering. The snapshot must retain either database's exact lease order.
    exclusions=[runner,original.hotkey('n8-sub-1')]
    lease,token,_,_=original.claim(store,rid,runner,excluded=exclusions)
    output=scoring.build_scoring_output(lease['scored_run_id'],[{'company_name':'A','final_score':55.0}])
    evidence=judgment_cache.build_evidence_snapshot(output=output,cache_scope=scope,
        source_score_run_id=lease['run_id'],source_scored_run_id=lease['scored_run_id'],
        source_output_ref='arena/x/scores/'+lease['run_id']+'.json',source_runner_hotkey=runner,
        runner_authority_exclusions=lease['runner_authority_exclusions'])
    assert evidence['runner_authority_exclusions']==lease['runner_authority_exclusions']
    assert store.complete_attempt(run_id=lease['run_id'],lease_token_hash=hash_lease_token(token),
        result={'terminal_status':'accepted'},terminal_cause='accepted',output_ref=evidence['source_output_ref'],
        judgment_evidence=evidence,judgment_evidence_hash=contracts.document_hash(evidence))['status']=='accepted'
