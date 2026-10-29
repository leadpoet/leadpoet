"""Exact day exception through real PostgreSQL and unchanged promotion/signing.

Default fixtures use synthetic persisted scores, a local signing key and local
Git. Set OCT29_LIVE_INPUTS and OCT29_SOURCE_ARCHIVE to test the verbatim release
SQL against protected live input copies; production credentials are never used.
"""
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from lab_arena import contracts, rewards, signing, weight_state, public_dashboard
from lab_arena.service import LocalObjectStore
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from leadpoet_canonical.lab_arena_rewards import slot_allocations
from tests.lab_arena.lab_arena_pg_harness import current_service_database
from tests.lab_arena.test_lab_arena_promotion_git import repository, source, git
from tests.lab_arena.test_lab_arena_promotion_service import _service

ROOT=Path(__file__).resolve().parents[2]
TEMPLATE=ROOT/'scripts/592-arena-2026-10-29-authorized-champion.sql.template'
RELEASE=ROOT/'scripts/592-arena-2026-10-29-authorized-champion.sql'
RID='arena-2026-10-29'
ARCHIVE=RID+'-rewardhistory592'
WINNER='sub-cb3105f830853c1452b889ab5cadee7e'
HOTKEY='5HSutSjNe6Mpfxjvifv7y2p4HbkTenVhZ75rRgHRdmSrXKu9'
OLD_HOTKEY='5DygibkPHDL1V22KSi1F2mzm6dK9oJuqFkJ74urKTyjvFo2K'
BASELINE='5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9'
POLICY=rewards.reward_slot_policy_document(effective_reward_epoch=26080)

@pytest.fixture(scope='module')
def database():
    yield from current_service_database()

@pytest.fixture()
def state(database):
    pg,dsn=database
    control=pg.connect(**dsn);control.autocommit=True
    with control.cursor() as c:
        c.execute("DROP TRIGGER IF EXISTS test_oct29_activation_checkpoint ON public.lab_arena_rounds")
        c.execute("DROP TRIGGER IF EXISTS lab_arena_oct29_reward_guard592 ON public.lab_arena_rounds")
        c.execute("DROP TRIGGER IF EXISTS lab_arena_oct29_accepted_guard592 ON public.lab_arena_accepted_weight_states")
        c.execute("DROP TRIGGER IF EXISTS lab_arena_oct29_chain_guard592 ON public.lab_arena_chain_outcomes")
        c.execute("TRUNCATE public.lab_arena_rounds CASCADE")
        c.execute("TRUNCATE public.lab_arena_accepted_weight_states,public.lab_arena_chain_outcomes")
        c.execute("SELECT pg_get_functiondef('public.lab_arena_reward_slot_snapshot(text,jsonb)'::regprocedure)")
        original=c.fetchone()[0]
    store=ArenaStore(PsycopgTransport(lambda:pg.connect(**dsn)))
    try:yield store,control
    finally:
        with control.cursor() as c:
            c.execute('ROLLBACK')
            c.execute(original)
        store.close();control.close()

def _sha(c, value):
    c.execute("SELECT encode(extensions.digest(%s::jsonb::text,'sha256'),'hex')",(json.dumps(value),))
    return c.fetchone()[0]

def render(c, r, s):
    before=_sha(c,r['publication_doc'])
    pub=deepcopy(r['publication_doc'])
    pub['king_decision']={'outcome':'crowned','king_hotkey':HOTKEY,'king_submission_id':WINNER,
        'winner_submission_id':WINNER,'operator_override':{
            'reason':'authorized_oct29_training_evidence_exception','migration':592,
            'original_publication_sha256':before}}
    after=_sha(c,pub)
    archived=deepcopy(r)
    archived.update(round_id=ARCHIVE,status='cancelled',promotion_required=False,
        cancel_reason='authorized_oct29_champion_reward_archive592')
    archived['configuration_doc'].update(round_id=ARCHIVE,recovery_source_round_id=RID,
        authorized_champion_override=pub['king_decision']['operator_override'])
    values={'ROUND_BEFORE_SHA256':_sha(c,r),'SUBMISSION_SHA256':_sha(c,s),'ARCHIVE_SHA256':_sha(c,archived),
        'PUBLICATION_BEFORE_SHA256':before,'PUBLICATION_AFTER_SHA256':after,
        'SOURCE_ARCHIVE_SHA256':s['source_archive_sha256'],
        'SOURCE_NORMALIZED_SHA256':s['source_normalized_sha256'],
        'SOURCE_SIZE_BYTES':str(s['source_size_bytes']),
        'OLD_REWARD_BASIS_HASH':r['reward_basis_hash'],
        'OLD_EFFECTIVE_EPOCH':str(r['effective_reward_epoch'])}
    body=TEMPLATE.read_text()
    for k,v in values.items():body=body.replace('__'+k+'__',v)
    old=re.search(r'anchor CONSTANT TEXT := \$old\$(.*?)\$old\$;',body,re.S)[1]
    new=re.search(r'replacement CONSTANT TEXT := \$new\$(.*?)\$new\$;',body,re.S)[1]
    c.execute("SELECT pg_get_functiondef('public.lab_arena_reward_slot_snapshot(text,jsonb)'::regprocedure)")
    definition=c.fetchone()[0]
    assert definition.count(old)==1
    body=body.replace('__SNAPSHOT_AFTER_SHA256__',hashlib.sha256(definition.replace(old,new).encode()).hexdigest())
    assert not re.search(r'__[A-Z0-9_]+__',body)
    return body

def _insert_document(c, table, document):
    from psycopg2 import sql
    c.execute('SELECT attname FROM pg_attribute WHERE attrelid=%s::regclass AND attnum>0 AND NOT attisdropped AND attgenerated=\'\' ORDER BY attnum',('public.'+table,))
    fields=[r[0] for r in c.fetchall() if r[0] in document]
    names=sql.SQL(',').join(map(sql.Identifier,fields))
    c.execute(sql.SQL('INSERT INTO public.{}({}) SELECT {} FROM jsonb_populate_record(NULL::public.{},%s::jsonb)').format(sql.Identifier(table),names,names,sql.Identifier(table)),(json.dumps(document),))

def _baseline_runs(c,r):
    baseline=next(p for p in r['participants'] if p['is_king'])
    sid=baseline['submission_id'];hk=baseline['miner_hotkey'];cfg=r['configuration_doc']
    if sid!=WINNER:
        _insert_document(c,'lab_arena_submissions',{'submission_id':sid,'round_id':RID,'miner_hotkey':hk,'status':'frozen','is_king':True})
    for stage in ((1,) if cfg.get('execution_sequence_policy')==contracts.BASELINE_SCORED_FIRST_POLICY else (1,2)):
        for pos in contracts.execution_positions(stage,cfg.get('execution_sequence_policy'),cfg):
            runid=f'{sid}-execute-{pos}'
            for kind in ('execute','score'):
                _insert_document(c,'lab_arena_runs',{'run_id':f'{sid}-{kind}-{pos}',
                    'assignment_id':f'{sid}-execute-{pos}-assignment'+(':score' if kind=='score' else ''),'round_id':RID,
                    'submission_id':sid,'miner_hotkey':hk,'stage':stage,'icp_position':pos,
                    'attempt':1,'kind':kind,'scored_run_id':runid if kind=='score' else None,
                    'status':'accepted','terminal_cause':'accepted','per_icp_score':8.55,
                    'result_doc':{'terminal_status':'accepted','schema_version':'fixture'},
                    'output_ref':f'fixture/{sid}/{kind}/{pos}',
                    'judgment_scope_doc':{'scorer_image_digest':cfg.get('scorer_image_digest'),
                        'scorer_image_reference':cfg.get('scorer_image_reference')} if kind=='score' else None,
                    'scoring_generation_floor':r.get('scoring_generation_floor',0)})

def seed(control, *, live=False):
    signer=signing.LocalSigner.generate()
    if live:
        path=os.environ.get('OCT29_LIVE_INPUTS')
        if not path:pytest.skip('protected live input copy not configured')
        private=json.loads(Path(path).read_text());r=private['round'];s=private['submission']
        payload=Path(os.environ['OCT29_SOURCE_ARCHIVE']).read_bytes()
    else:
        payload=Path(os.environ['OCT29_SOURCE_ARCHIVE']).read_bytes() if os.environ.get('OCT29_SOURCE_ARCHIVE') else source();sha='sha256:'+hashlib.sha256(payload).hexdigest()
        pub={'schema_version':contracts.PUBLICATION_SCHEMA_VERSION,'round_id':RID,
            'published_at':'2026-10-29T21:28:21Z','stage1_ranking':[],'finalists':[],
            'participants':[{'submission_id':WINNER,'miner_hotkey':HOTKEY,'is_baseline':False},
                {'submission_id':'baseline-2026-10-29','miner_hotkey':BASELINE,'is_baseline':True}],
            'final_ranking':[{'rank':1,'submission_id':WINNER,'final_score':20.275,
                'is_baseline':False,'eligible':False,'eligibility_reason':'training_data_incomplete',
                'training_data_eligibility':{'eligible':False,'reasons':['training_evidence_check_failed']}},
                {'rank':2,'submission_id':'baseline-2026-10-29','final_score':8.55,
                 'is_baseline':True,'eligible':True}],
            'king_decision':{'outcome':'no_king','king_hotkey':'','king_submission_id':None,'winner_submission_id':None}}
        old_slot={'round_id':'arena-2026-10-27','submission_id':'sub-273b98c23fcdc0f2edb8eb00a590e815',
            'miner_hotkey':OLD_HOTKEY,'baseline_submission_id':'baseline-2026-10-27',
            'baseline_score':13.7,'winner_score':23.15,'start_epoch':26031}
        basis=signing.sign_document(signer,rewards.reward_basis_document(round_id=RID,
            published_at=pub['published_at'],finalized_epoch=26077,king_hotkey=OLD_HOTKEY,
            king_outcome='defended',previous_king_start_epoch=26031,
            slot_policy=POLICY,reward_slots=[None,None,old_slot]),hash_field='reward_basis_hash')
        r={'round_id':RID,'status':'published','status_generation':1,'stage_generation':1,
           'configuration_doc':{'round_id':RID,'mode':'live','network_name':'finney','netuid':71,
               'rewards_enabled':True,'baseline_hotkey':BASELINE,'scorer_policy':{},'reward_constants':rewards.reward_constants_document(),
               'stage_1_icp_count':1,'stage_2_icp_count':1,
               'scorer_image_digest':'sha256:'+'a'*64,'scorer_image_reference':'fixture/scorer@sha256:'+'a'*64,
               'schedule':{'submission_open':'2026-10-28T00:00:00Z','submission_cutoff':'2026-10-29T00:00:00Z'}},'rewards_enabled':True,
           'participants':[{'submission_id':WINNER,'miner_hotkey':HOTKEY,'is_king':False},
               {'submission_id':'baseline-2026-10-29','miner_hotkey':BASELINE,'is_king':True}],
           'evaluation_date':'2026-10-29','icp_set_date':'2026-10-28','publication_doc':pub,'king_outcome':'no_king','king_hotkey':None,
           'king_start_epoch':26031,'effective_reward_epoch':26078,'reward_basis_hash':basis['reward_basis_hash'],
           'reward_basis_doc':basis,'signing_key_doc':signing.signing_key_document(signer.public_key_der),
           'reward_activated_at':'2026-10-29T21:30:00Z','published_at':pub['published_at'],
           'promotion_required':True,'created_at':'2026-10-28T00:00:00Z','updated_at':'2026-10-29T21:30:00Z'}
        s={'submission_id':WINNER,'round_id':RID,'miner_hotkey':HOTKEY,'status':'frozen','is_king':False,
           'source_ref':f'arena/{RID}/sources/{WINNER}.tar.gz','source_size_bytes':len(payload),
           'source_archive_sha256':sha,'source_normalized_sha256':sha,'consent':{'public_rerun':True},
           'accepted_at':'2026-10-28T01:00:00Z','created_at':'2026-10-28T01:00:00Z','updated_at':'2026-10-28T01:00:00Z'}
    with control.cursor() as c:
        c.execute('BEGIN');c.execute('SET LOCAL session_replication_role=replica')
        _insert_document(c,'lab_arena_rounds',r)
        _insert_document(c,'lab_arena_submissions',s)
        if r.get('champion_submission_id'):
            _insert_document(c,'lab_arena_submissions',{'submission_id':r['champion_submission_id'],
                'round_id':RID,'miner_hotkey':r['champion_hotkey'],'status':'frozen','is_king':False})
        _baseline_runs(c,r)
        c.execute('COMMIT')
        c.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s',(RID,));r=c.fetchone()[0]
        c.execute('SELECT to_jsonb(s) FROM public.lab_arena_submissions s WHERE submission_id=%s',(WINNER,));s=c.fetchone()[0]
        c.execute('SELECT public.lab_arena__baseline_final_ready_v1(%s)',(RID,));assert c.fetchone()==(True,)
    assert 'sha256:'+hashlib.sha256(payload).hexdigest()==s['source_archive_sha256']
    return r,s,payload,signer

def test_verbatim_release_on_protected_live_copy(state):
    store,control=state;r,s,_,_=seed(control,live=True)
    with control.cursor() as c:
        expected=render(c,r,s)
        if os.environ.get('OCT29_RENDER_RELEASE')=='1':RELEASE.write_text(expected)
        assert RELEASE.read_text()==expected
        c.execute(RELEASE.read_text());c.execute(RELEASE.read_text())
    now=store.get_round(RID);arch=store.get_round(ARCHIVE)
    assert arch['reward_basis_doc']==r['reward_basis_doc']
    assert arch['signing_key_doc']==r['signing_key_doc']
    assert arch['publication_doc']==r['publication_doc']
    assert now['publication_doc']['final_ranking']==r['publication_doc']['final_ranking']
    assert now['reward_basis_doc'] is None
    assert now['publication_doc']['king_decision']['winner_submission_id']==WINNER
    assert store.published_reward_bases(mode='live',network_name='finney',netuid=71)[0]['reward_basis_hash']==r['reward_basis_hash']

@pytest.mark.parametrize('mutation',['miner_hotkey','source_archive_sha256','final_score','training_report','basis','signing_key'])
def test_pinned_identity_or_evidence_change_fails_without_partial_writes(state,mutation):
    store,control=state;r,s,_,_=seed(control)
    with control.cursor() as c:
        sql=render(c,r,s)
        c.execute('BEGIN');c.execute('SET LOCAL session_replication_role=replica')
        if mutation in ('miner_hotkey','source_archive_sha256'):
            c.execute(f'UPDATE public.lab_arena_submissions SET {mutation}=%s WHERE submission_id=%s',('5'+'A'*47 if mutation=='miner_hotkey' else 'sha256:'+'a'*64,WINNER))
        elif mutation in ('basis','signing_key'):
            field='reward_basis_doc' if mutation=='basis' else 'signing_key_doc'
            c.execute(f"UPDATE public.lab_arena_rounds SET {field}={field} || '{{\"tampered\":true}}'::jsonb WHERE round_id=%s",(RID,))
        else:
            path={'final_score':'{final_ranking,0,final_score}','training_report':'{final_ranking,0,training_data_eligibility}','basis':'{king_decision,outcome}'}[mutation]
            c.execute('UPDATE public.lab_arena_rounds SET publication_doc=jsonb_set(publication_doc,%s,%s::jsonb) WHERE round_id=%s',(path,json.dumps(20.276 if mutation=='final_score' else {}),RID))
        c.execute('COMMIT')
        before=store.get_round(RID)
        with pytest.raises(Exception,match='Oct29 champion (submission|preimage) differs'):c.execute(sql)
        c.execute('ROLLBACK')
    assert store.get_round(RID)==before
    assert store.get_round(ARCHIVE) is None

def test_native_atomic_promotion_reward_and_old_epoch_continuity(state,tmp_path):
    store,control=state;r,s,payload,signer=seed(control)
    remote,main_before,lab_before=repository(tmp_path)
    objects=LocalObjectStore(tmp_path/'objects');objects.put(s['source_ref'],payload)
    service=_service(store,objects,remote,tmp_path/'promotion-cache',signer)
    service._clock=lambda:datetime(2026,10,29,23,45,tzinfo=timezone.utc)
    service._config.chain=SimpleNamespace(current_settlement_epoch=lambda:26080)
    assert service._baseline_final_ready(r)
    assert service.promote_baseline(RID)=={'status':'not_required'}
    assert service.activate_reward(RID)['status']=='existing'
    with control.cursor() as c:
        sql=render(c,r,s);c.execute(sql);c.execute(sql)
    with control.cursor() as c:
        c.execute('SELECT public.lab_arena__baseline_final_ready_v1(%s)',(RID,))
        assert c.fetchone()==(True,)
    assert service.public_reward_basis(26080)==r['reward_basis_doc']
    assert service.activate_reward(RID)=={'status':'waiting_for_promotion'}
    # Atomic capability failure must change neither branch nor reward basis.
    git(remote,'config','receive.advertiseAtomic','false')
    from lab_arena.promotion import PromotionError
    with pytest.raises(PromotionError):service.promote_baseline(RID)
    assert git(remote,'rev-parse','main')==main_before
    assert git(remote,'rev-parse','lab')==lab_before
    assert store.get_round(RID)['reward_basis_doc'] is None
    git(remote,'config','receive.advertiseAtomic','true')
    assert service.promote_baseline(RID)['status']=='promoted'
    promoted=store.get_round(RID);commit=promoted['promotion_doc']['commit']
    assert git(remote,'rev-parse','main')==git(remote,'rev-parse','lab')==commit
    assert git(remote,'show','-s','--format=%P',commit).split()==[main_before,lab_before]
    from lab_arena.source_bundle import validate_source_archive
    import tarfile,io
    facts=validate_source_archive(payload);prefix=facts['source_root']
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as tar:
        for member in tar:
            if member.isfile():
                name=member.name[len(prefix)+1:] if prefix else member.name
                import subprocess
                actual=subprocess.run(['git','show',commit+':'+name],cwd=remote,check=True,capture_output=True).stdout
                assert hashlib.sha256(actual).digest()==hashlib.sha256(tar.extractfile(member).read()).digest()
    old_state=weight_state.build_accepted_weight_state(signer,network='finney',genesis_hash='1'*64,netuid=71,
        epoch=26080,valid_from_block=1,valid_until_block=360,reward_basis=r['reward_basis_doc'],burn_hotkey=BASELINE,
        issued_at=r['publication_doc']['published_at'])
    assert store.publish_weight_state('finney',71,26080,old_state['state_hash'],old_state)['status']=='accepted'
    service._config.chain=SimpleNamespace(current_settlement_epoch=lambda:26079)
    with pytest.raises(ArenaStoreError,match='Oct29 replacement reward conflicts'):
        service.activate_reward(RID)
    assert store.get_round(RID)['reward_basis_doc'] is None
    service._config.chain=SimpleNamespace(current_settlement_epoch=lambda:26080)
    assert service.activate_reward(RID)['status']=='activated'
    now=store.get_round(RID);basis=now['reward_basis_doc']
    assert basis['effective_reward_epoch']==26081
    assert basis['king_hotkey']==HOTKEY and basis['king_outcome']=='crowned'
    assert basis['reward_slots'][:2]==r['reward_basis_doc']['reward_slots'][:2]
    assert basis['reward_slots'][2]['submission_id']==WINNER
    assert basis['reward_slots'][2]['start_epoch'] is None
    assert slot_allocations(basis,26081,[BASELINE,HOTKEY,OLD_HOTKEY])=={HOTKEY:Fraction(3,20)}
    assert service.public_reward_basis(26080)==r['reward_basis_doc']
    assert service.public_weight_state(26080)['state']==old_state
    assert service.public_reward_basis(26081)==basis
    assert now['publication_doc']['final_ranking']==r['publication_doc']['final_ranking']
    assert public_dashboard._champion_submission_id(now)==WINNER
    latest=store.latest_promoted_champion(network_name='finney',netuid=71)
    assert latest['winner_submission_id']==WINNER and latest['commit']==commit
    assert store.get_submission(WINNER)==s
    with control.cursor() as c:c.execute(sql)
    assert store.get_round(RID)==now
    assert service.promote_baseline(RID)=={'status':'existing'}
    assert service.activate_reward(RID)['status']=='existing'

    # Correct replacement and future normal bases remain admissible. Historical
    # exact replays still return their original body; the archived basis cannot
    # newly acquire an epoch at or after the replacement.
    service._config.chain=SimpleNamespace(current_settlement_epoch=lambda:26080,
        accepted_weight_epoch_scope=lambda:{'epoch':26081,'genesis_hash':'1'*64,
            'valid_from_block':361,'valid_until_block':720})
    service._config.accepted_burn_hotkey=BASELINE
    correct=service.public_weight_state(26081)['state']
    assert correct['reward_basis']==basis
    assert store.publish_weight_state('finney',71,26080,old_state['state_hash'],old_state)['state']==old_state
    wrong=weight_state.build_accepted_weight_state(signer,network='finney',genesis_hash='1'*64,netuid=71,
        epoch=26081,valid_from_block=361,valid_until_block=720,reward_basis=r['reward_basis_doc'],
        burn_hotkey=BASELINE,issued_at=r['publication_doc']['published_at'])
    with pytest.raises(ArenaStoreError,match='archived basis cannot acquire'):
        store.publish_weight_state('finney',71,26081,wrong['state_hash'],wrong)
    later_slots=deepcopy(basis['reward_slots']);later_slots[2]['start_epoch']=26081
    later_basis=signing.sign_document(signer,rewards.reward_basis_document(round_id='arena-2026-10-31',
        published_at='2026-10-31T00:00:00Z',finalized_epoch=26081,king_hotkey=HOTKEY,
        king_outcome='defended',previous_king_start_epoch=26081,slot_policy=POLICY,reward_slots=later_slots),
        hash_field='reward_basis_hash')
    later_state=weight_state.build_accepted_weight_state(signer,network='finney',genesis_hash='1'*64,netuid=71,
        epoch=26082,valid_from_block=721,valid_until_block=1080,reward_basis=later_basis,
        burn_hotkey=BASELINE,issued_at='2026-10-31T00:00:00Z')
    assert store.publish_weight_state('finney',71,26082,later_state['state_hash'],later_state)['status']=='accepted'
    outcome={'network':'finney','netuid':71,'epoch':26081,'state_hash':correct['state_hash']}
    assert store.record_chain_outcome(network='finney',netuid=71,epoch=26081,validator_hotkey=HOTKEY,
        request_id='sha256:'+'a'*64,extrinsic_hash='0x'+'b'*64,outcome_doc=outcome)['status']=='recorded'
    with pytest.raises(ArenaStoreError,match='chain outcome requires its accepted state'):
        store.record_chain_outcome(network='finney',netuid=71,epoch=26081,validator_hotkey=HOTKEY,
            request_id='sha256:'+'c'*64,extrinsic_hash='0x'+'d'*64,
            outcome_doc={**outcome,'state_hash':wrong['state_hash']})


def test_exception_does_not_admit_other_ineligible_results_or_change_normal_crowns(state):
    from tests.lab_arena.reward_slots_postgres_test import _insert,_publication
    store,control=state;r,s,_,_=seed(control)
    with control.cursor() as c:c.execute(render(c,r,s))
    # A different ineligible winner with even larger improvement remains excluded.
    badrid='arena-2026-10-30-71impostor'
    pub=_publication(badrid,delta=25)
    pub['final_ranking'][0].update(eligible=False,eligibility_reason='training_data_incomplete')
    bad=_insert(control,71,'impostor',day='2026-10-30',publication=pub,activated=False)
    with pytest.raises(ArenaStoreError,match='current_achievement_invalid'):
        store.reward_slot_snapshot(bad,POLICY)
    good=_insert(control,71,'ordinary',day='2026-10-31',delta=21,activated=False)
    slots=store.reward_slot_snapshot(good,POLICY)
    assert [x['round_id'] for x in slots]==[good]*3


def test_archive_tamper_rejected_on_replay(state):
    store,control=state;r,s,_,_=seed(control)
    with control.cursor() as c:
        sql=render(c,r,s);c.execute(sql)
        c.execute('BEGIN');c.execute('SET LOCAL session_replication_role=replica')
        c.execute("UPDATE public.lab_arena_rounds SET signing_key_doc=signing_key_doc || '{\"tampered\":true}'::jsonb WHERE round_id=%s",(ARCHIVE,))
        c.execute('COMMIT')
        with pytest.raises(Exception,match='Oct29 champion replay differs'):c.execute(sql)
        c.execute('ROLLBACK')
    assert store.get_round(RID)['reward_basis_doc'] is None


def test_competing_accepted_epoch_insert_is_seen_before_replacement(state,database,tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    store,control=state;r,s,payload,signer=seed(control)
    remote,_,_=repository(tmp_path)
    objects=LocalObjectStore(tmp_path/'objects');objects.put(s['source_ref'],payload)
    service=_service(store,objects,remote,tmp_path/'cache',signer)
    service._clock=lambda:datetime(2026,10,29,23,45,tzinfo=timezone.utc)
    service._config.chain=SimpleNamespace(current_settlement_epoch=lambda:26080)
    with control.cursor() as c:c.execute(render(c,r,s))
    assert service.promote_baseline(RID)['status']=='promoted'
    accepted=weight_state.build_accepted_weight_state(signer,network='finney',genesis_hash='1'*64,netuid=71,
        epoch=26081,valid_from_block=1,valid_until_block=360,reward_basis=r['reward_basis_doc'],
        burn_hotkey=BASELINE,issued_at=r['publication_doc']['published_at'])
    pg,dsn=database
    writer=pg.connect(**dsn)
    try:
        with writer.cursor() as c:
            c.execute('INSERT INTO public.lab_arena_accepted_weight_states(network,netuid,epoch,state_hash,state_doc) VALUES(\'finney\',71,26081,%s,%s::jsonb)',(accepted['state_hash'],json.dumps(accepted)))
        started=threading.Event()
        def activate():
            started.set()
            with pytest.raises(ArenaStoreError,match='Oct29 replacement reward conflicts'):
                service.activate_reward(RID)
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending=executor.submit(activate)
            try:
                assert started.wait(2)
                # Prove an advisory wait, not a timing-only race.
                import time
                deadline=time.monotonic()+3
                blocked=False
                while time.monotonic()<deadline:
                    with control.cursor() as c:
                        c.execute("SELECT EXISTS(SELECT 1 FROM pg_locks WHERE locktype='advisory' AND NOT granted)")
                        blocked=c.fetchone()[0]
                    if blocked:break
                    time.sleep(.01)
                assert blocked
            finally:
                writer.commit()
            pending.result(timeout=5)
    finally:
        writer.rollback();writer.close()
    assert store.get_round(RID)['reward_basis_doc'] is None
    assert service.public_reward_basis(26081)==r['reward_basis_doc']


def test_activation_first_blocks_then_rejects_queued_archived_basis(state,database,tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import time
    store,control=state;r,s,payload,signer=seed(control)
    remote,_,_=repository(tmp_path)
    objects=LocalObjectStore(tmp_path/'objects');objects.put(s['source_ref'],payload)
    service=_service(store,objects,remote,tmp_path/'cache',signer)
    service._clock=lambda:datetime(2026,10,29,23,45,tzinfo=timezone.utc)
    service._config.chain=SimpleNamespace(current_settlement_epoch=lambda:26080)
    with control.cursor() as c:
        c.execute(render(c,r,s))
    assert service.promote_baseline(RID)['status']=='promoted'
    # A test-only checkpoint holds the real activation transaction open after
    # the canonical RPC updates the round while holding the production lock.
    with control.cursor() as c:
        c.execute("CREATE OR REPLACE FUNCTION public.test_oct29_checkpoint() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF NEW.round_id='arena-2026-10-29' AND OLD.reward_activated_at IS NULL AND NEW.reward_activated_at IS NOT NULL THEN PERFORM pg_advisory_xact_lock(592); END IF; RETURN NEW; END; $$")
        c.execute('CREATE TRIGGER test_oct29_activation_checkpoint AFTER UPDATE ON public.lab_arena_rounds FOR EACH ROW EXECUTE FUNCTION public.test_oct29_checkpoint()')
        c.execute('SELECT pg_advisory_lock(592)')
    old=weight_state.build_accepted_weight_state(signer,network='finney',genesis_hash='1'*64,netuid=71,
        epoch=26081,valid_from_block=361,valid_until_block=720,reward_basis=r['reward_basis_doc'],
        burn_hotkey=BASELINE,issued_at=r['publication_doc']['published_at'])
    def blocked_count():
        with control.cursor() as c:
            c.execute("SELECT count(*) FROM pg_locks WHERE locktype='advisory' AND NOT granted")
            return c.fetchone()[0]
    def wait_for(count):
        deadline=time.monotonic()+4
        while time.monotonic()<deadline:
            if blocked_count()>=count:return
            time.sleep(.01)
        raise AssertionError('expected advisory wait not observed')
    def stale_insert():
        with pytest.raises(ArenaStoreError,match='archived basis cannot acquire'):
            store.publish_weight_state('finney',71,26081,old['state_hash'],old)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            activation=executor.submit(service.activate_reward,RID)
            try:
                wait_for(1)
                writer=executor.submit(stale_insert)
                wait_for(2)
            finally:
                with control.cursor() as c:c.execute('SELECT pg_advisory_unlock(592)')
            assert activation.result(timeout=5)['status']=='activated'
            writer.result(timeout=5)
    finally:
        with control.cursor() as c:
            c.execute('SELECT pg_advisory_unlock(592)')
            c.execute('DROP TRIGGER test_oct29_activation_checkpoint ON public.lab_arena_rounds')
    assert store.get_weight_state('finney',71,26081) is None
    assert store.get_round(RID)['reward_basis_doc']['king_hotkey']==HOTKEY


def test_filtered_history_trigger_is_rejected_on_replay(state):
    store,control=state;r,s,_,_=seed(control)
    with control.cursor() as c:
        sql=render(c,r,s);c.execute(sql)
        c.execute('DROP TRIGGER lab_arena_oct29_accepted_guard592 ON public.lab_arena_accepted_weight_states')
        c.execute('CREATE TRIGGER lab_arena_oct29_accepted_guard592 BEFORE INSERT ON public.lab_arena_accepted_weight_states FOR EACH ROW WHEN(FALSE) EXECUTE FUNCTION public.lab_arena_oct29_issued_guard592()')
        before=store.get_round(RID)
        with pytest.raises(Exception,match='Oct29 issued trigger differs'):c.execute(sql)
        c.execute('ROLLBACK')
    assert store.get_round(RID)==before
