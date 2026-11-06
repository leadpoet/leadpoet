"""One explicit organizer OpenRouter attempt; old work and bills remain intact.

Only disposable fixture setup uses replica writes. Recovery and all accounting,
claims, completions and expiry use real PostgreSQL functions and service role.
No provider request is sent.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import re
import uuid

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, PsycopgTransport, hash_lease_token
from tests.lab_arena import nov05_b66_sixth639_postgres_test as historical
from tests.lab_arena.test_lab_arena_contracts import base_round_configuration
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey

SQL = Path(__file__).parents[2] / 'scripts/665-arena-2026-11-06-organizer-openrouter-recovery.sql'
ROUND = 'arena-2026-11-06'
TOKEN = hash_lease_token('organizer665-fixture')
SIGS = [
 'public.lab_arena_provider_funding(text,text)',
 'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)',
 'public.lab_arena_complete_attempt(text,text,jsonb,text,text)',
 'public.lab_arena_expire_leases(text)',
]
COHORT = json.loads(SQL.read_text().split('$cohort665$')[1])


def scalar(q, sql, args=()):
    q.execute(sql, args)
    return q.fetchone()[0]


@pytest.fixture(scope='module')
def database():
    gen = historical.database.__wrapped__()
    pg, dsn, _ = next(gen)
    try:
        with pg.connect(**dsn) as c, c.cursor() as q:
            # Install committed639's helper and claim fragment without its dated
            # live correction. The fixture already has all other predecessors.
            text = historical.SQL.read_text()
            restore = 'DO $restore639$' + text.split('DO $restore639$')[1].split('NOTIFY')[0]
            q.execute(text.split('DO $patch639$')[0] + restore + 'COMMIT;')
            d = scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (SIGS[1],))
            old, new = next((a, b) for a, b in re.findall(r'\$old639\$(.*?)\$old639\$,\$new639\$(.*?)\$new639\$', text, re.S) if 'judge614' in a)
            assert d.count(old) == 1
            q.execute(d.replace(old, new))
            definitions = {sig: scalar(q, 'SELECT pg_get_functiondef(%s::regprocedure)', (sig,)) for sig in SIGS}
        yield pg, dsn, definitions
    finally:
        gen.close()


def insert(q, table, row):
    columns = list(row)
    q.execute('INSERT INTO public.' + table + '(' + ','.join(columns) + ') VALUES(' + ','.join('%s::jsonb' if isinstance(row[k], (dict, list)) else '%s' for k in columns) + ')', [json.dumps(row[k]) if isinstance(row[k], (dict, list)) else row[k] for k in columns])


def rpc(p, signature, args):
    c = p.pg.connect(**p.dsn)
    try:
        with c, c.cursor() as q:
            q.execute('SET ROLE lab_arena_service')
            assert scalar(q, 'SELECT current_user') == 'lab_arena_service'
            return scalar(q, 'SELECT public.' + signature, args)
    finally:
        c.close()


def owned_settings(q):
    for key, value in dict(guard_commitment='sha256:'+'a'*64, owner_commitment='sha256:'+'b'*64, guard_generation='100', candidate_commit='c'*40).items():
        q.execute('SELECT set_config(%s,%s,FALSE)', ('leadpoet.arena665.' + key, value))


@pytest.fixture
def prepared(database):
    pg, dsn, definitions = database
    c = pg.connect(**dsn)
    now = datetime.now(timezone.utc)
    cfg = base_round_configuration()
    cfg.update(round_id=ROUND, baseline_hotkey=hotkey('665-baseline'), mode='live', network_name='finney', netuid=71,
               stage_1_icp_count=5, stage_2_icp_count=5, execution_sequence_policy='baseline_scored_first_v1',
               infrastructure_retry_cost_policy='trusted_infrastructure_v1')
    cfg['schedule'].update(final_scoring_close='2099-11-06T23:30:00Z', publication_deadline='2099-11-06T23:59:00Z')
    owners = [dict(submission_id=s, miner_hotkey=hotkey('665-'+s), is_king=False) for s in COHORT]
    healthy = [dict(submission_id='healthy665-'+str(i), miner_hotkey=hotkey('healthy665-'+str(i)), is_king=False) for i in range(7)]
    hashes = {}
    executions = []
    scores = []
    plan = []
    with c.cursor() as q:
        for d in definitions.values():
            q.execute(d)
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute('TRUNCATE public.lab_arena_rounds,public.lab_arena_ledger,public.lab_arena_trajectory_events RESTART IDENTITY CASCADE')
        q.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=FALSE,pause_reason='',actor_ref='665-fixture',guard_commitment=%s,owner_commitment=%s,guard_generation=100,guard_expires_at=%s,candidate_commit=%s,restart_scope='gateway',restart_phase='draining',captured_leases='[]' WHERE singleton", ('sha256:'+'a'*64, 'sha256:'+'b'*64, now+timedelta(hours=1), 'c'*40))
        insert(q, 'lab_arena_rounds', dict(round_id=ROUND,status='stage2_scoring',status_generation=5,stage_generation=4,configuration_doc=cfg,participants=owners+healthy,evaluation_date='2026-11-06',icp_set_date='2026-11-05',benchmark_ref='665-benchmark',finalists=[o['submission_id'] for o in owners+healthy]))
        for owner in owners+healthy:
            insert(q, 'lab_arena_submissions', dict(**owner,round_id=ROUND,status='frozen',source_ref=f'arena/{ROUND}/sources/{owner["submission_id"]}.tar.gz',source_size_bytes=100,source_archive_sha256='sha256:'+'4'*64,source_normalized_sha256='sha256:'+'5'*64,code_review_status='passed',code_review_attempts=1,code_review_doc={},code_review_claim='sha256:'+'6'*64,code_review_started_at=now))
            cipher = ('synthetic-cipher-'+owner['submission_id']).encode()
            if owner in owners:
                hashes[owner['submission_id']] = hashlib.sha256(cipher).hexdigest()
            q.execute('INSERT INTO public.lab_arena_submission_credentials(submission_id,miner_hotkey,provider,ciphertext) VALUES(%s,%s,\'openrouter\',%s)', (owner['submission_id'], owner['miner_hotkey'], cipher))
            count = (7 if owner == owners[0] else 8) if owner in owners else 10
            for pos in range(count):
                aid = f'{ROUND}:{owner["submission_id"]}:2:{pos}'
                source = dict(run_id=aid+':1',assignment_id=aid,round_id=ROUND,submission_id=owner['submission_id'],miner_hotkey=owner['miner_hotkey'],stage=2,icp_position=pos,attempt=1,kind='execute',status='accepted',terminal_cause='accepted',output_ref=aid+'-output',result_doc={'terminal_status':'accepted'},stage_generation=2,scoring_generation_floor=0)
                insert(q, 'lab_arena_runs', source)
                executions.append(source)
                if owner not in owners:
                    continue
                score_aid = aid+':score'
                cache = 'sha256:'+hashlib.sha256(score_aid.encode()).hexdigest()
                scope = dict(round_id=ROUND,scorer_image_digest=cfg['scorer_image_digest'],scorer_image_reference=cfg['scorer_image_reference'],scoring_input_hash=cache,cache_key=cache)
                plan.append(dict(submission_id=owner['submission_id'],icp_position=pos,scored_run_id=source['run_id'],output_ref=source['output_ref']))
                for attempt in (range(1,6) if pos == 0 else [1]):
                    status = 'failed' if pos == 0 else 'accepted' if pos == 1 else 'pending'
                    score = dict(run_id=score_aid+':'+str(attempt),assignment_id=score_aid,round_id=ROUND,submission_id=owner['submission_id'],miner_hotkey=owner['miner_hotkey'],stage=2,icp_position=pos,attempt=attempt,kind='score',scored_run_id=source['run_id'],status=status,stage_generation=4,scoring_generation_floor=0,judgment_cache_key=cache,judgment_input_hash=cache,judgment_scope_doc=scope)
                    if status != 'pending':
                        score.update(terminal_cause='accepted' if status=='accepted' else 'credential_error', result_doc={'terminal_status':'accepted' if status=='accepted' else 'credential_error'})
                    if status == 'accepted':
                        score['output_ref'] = score['run_id']+'-judge-output'
                    insert(q, 'lab_arena_runs', score)
                    scores.append(score)
                if pos == 0:
                    old = scores[-1]
                    identity = 'sha256:'+hashlib.sha256(old['run_id'].encode()).hexdigest()
                    doc = dict(base_call_identity=identity,action_sequence=0,provider_attempt=1,request_hash='sha256:'+'d'*64)
                    evidence = {**{k:doc[k] for k in ['base_call_identity','action_sequence','provider_attempt']},'error_class':'account_credential_failure','provider_status':401,'submitted_account_authentication':True}
                    for kind, entry, terminal in [('reservation',doc,None),('dispatch',{},None),('settlement',{},dict(status=402,call_succeeded=False,account_failure_evidence=evidence))]:
                        insert(q,'lab_arena_ledger',dict(entry_kind=kind,run_id=old['run_id'],round_id=ROUND,submission_id=old['submission_id'],miner_hotkey=old['miner_hotkey'],stage=2,call_identity=identity,provider='openrouter',operation_id='openrouter.chat',funding_source='miner_key',amount_microusd=0,entry_doc=entry,terminal_response=terminal))
        assert len(executions) == 157
        q.execute('UPDATE public.lab_arena_rounds SET stage2_scoring_plan_doc=%s::jsonb WHERE round_id=%s', (json.dumps({'work_items':plan}),ROUND))
        q.execute('SET LOCAL session_replication_role=origin')
        owned_settings(q)
        config_hash = scalar(q,"SELECT encode(extensions.digest(configuration_doc::text,'sha256'),'hex') FROM public.lab_arena_rounds WHERE round_id=%s",(ROUND,))
    c.commit()
    text = SQL.read_text().replace('5a761eeae62154946cef9af7e18d3a24544713221fb982d4bd6ba8ca91c9be4c',config_hash).replace('2026-11-06T23:30:00Z','2099-11-06T23:30:00Z').replace('2026-11-06T23:59:00Z','2099-11-06T23:59:00Z')
    text = re.sub(r'\$ciphertexts665\$.*?\$ciphertexts665\$', lambda _: '$ciphertexts665$'+json.dumps(hashes,separators=(',',':'))+'$ciphertexts665$',text,flags=re.S)
    p = SimpleNamespace(c=c,pg=pg,dsn=dsn,sql=text,owners=owners,executions=executions,scores=scores)
    try:
        yield p
    finally:
        c.close()


def apply(p):
    with p.c.cursor() as q:
        q.execute(p.sql)
    p.c.commit()


def snapshot(p):
    with p.c.cursor() as q:
        return {table: scalar(q,'SELECT coalesce(jsonb_agg(to_jsonb(a) ORDER BY '+key+"),'[]'::jsonb) FROM public."+table+' a') for table,key in [('lab_arena_rounds','round_id'),('lab_arena_runs','run_id'),('lab_arena_ledger','entry_id'),('lab_arena_submission_credentials','submission_id,provider'),('lab_arena_restart_claim_control','singleton')]}


def release_guard(p):
    with p.c.cursor() as q:
        q.execute("UPDATE public.lab_arena_restart_claim_control SET guard_commitment='',owner_commitment='',guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]' WHERE singleton")
    p.c.commit()


def funding(p,run,provider='openrouter'):
    return rpc(p,'lab_arena_provider_funding(%s,%s)',(run,provider))


def test_exact_recovery_preserves_all_old_work_and_payers_and_is_replay_safe(prepared):
    p=prepared
    before=snapshot(p)
    apply(p)
    after=snapshot(p)
    for table in ['lab_arena_rounds','lab_arena_ledger','lab_arena_submission_credentials','lab_arena_restart_claim_control']:
        assert after[table]==before[table]
    old={r['run_id']:r for r in before['lab_arena_runs']}
    new={r['run_id']:r for r in after['lab_arena_runs'] if r['run_id'] not in old}
    assert len(new)==76
    assert all(r['kind']=='score' and r['status']=='pending' for r in new.values())
    for rid,row in old.items():
        assert funding(p,rid)['funding_source']=='miner_key'
        saved=next(r for r in after['lab_arena_runs'] if r['run_id']==rid)
        if row['status']=='pending':
            assert saved['status']=='failed' and saved['terminal_cause']=='stage_closed'
            assert saved['terminal_doc']['recovery_reason']=='superseded_by_explicit_organizer_judgment665'
        else:
            assert saved==row
    for rid in list(new)[:3]:
        assert funding(p,rid)['funding_source']=='host'
        assert funding(p,rid,'deepline')['funding_source']=='miner_key'
        assert funding(p,rid,'scrapingdog')['funding_source']=='miner_key'
    apply(p)
    assert snapshot(p)==after


def claim(p):
    runner=hotkey('665-runner')
    return rpc(p,'lab_arena_claim_assignment(%s,%s,20,20,%s::text[],%s,%s,%s,1500)',(ROUND,runner,[runner],uuid.uuid4().hex,'sha256:'+'e'*64,TOKEN))


def age_new_pending(p):
    with p.c.cursor() as q:
        q.execute("UPDATE public.lab_arena_runs a SET created_at=clock_timestamp()-interval '1 hour' WHERE public.lab_arena__nov06_organizer_openrouter665_v1(a) AND status='pending'")
    p.c.commit()


def settle_one(p,lease, *, sequence=0,amount=120):
    request='sha256:'+'f'*64
    identity=contracts.provider_call_identity(assignment_id=lease['assignment_id'],attempt=lease['attempt'],icp_position=lease['icp_position'],action_sequence=sequence,operation_id='openrouter.chat',request_hash=request)
    doc=dict(request_hash=request,base_call_identity=identity,provider_attempt=1,action_sequence=sequence,max_output_tokens=8)
    args=(lease['run_id'],TOKEN,identity,'openrouter.chat','openrouter','host',500,json.dumps(doc),1500)
    reserved=rpc(p,'lab_arena_reserve_call(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)',args)
    assert reserved['status']=='reserved'
    assert rpc(p,'lab_arena_mark_dispatched(%s,%s,%s)',(lease['run_id'],TOKEN,identity))['status']=='dispatched'
    terminal=dict(status=200,headers={},body_b64='e30=',call_succeeded=True,provider_cost={'request_id':'synthetic-host665-'+identity})
    assert rpc(p,'lab_arena_settle_call(%s,%s,%s,%s,%s::jsonb,%s)',(lease['run_id'],TOKEN,identity,amount,json.dumps(terminal),1500))['status']=='settled'
    replay=rpc(p,'lab_arena_reserve_call(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)',args)
    assert replay['status']=='settled' and replay['idempotent'] is True
    return identity


@pytest.mark.parametrize('outcome',['accepted','judge_error','expired'])
@pytest.mark.parametrize('attempt',[2,6])
def test_real_claim_billing_completion_and_expiry_are_one_shot(prepared,outcome,attempt):
    p=prepared
    apply(p)
    age_new_pending(p)
    # Isolate either a low attempt or the exhausted sixth through the fixture's
    # claim routing, without changing its payer, source or scoring generation.
    with p.c.cursor() as q:
        target=scalar(q,"SELECT run_id FROM public.lab_arena_runs a WHERE public.lab_arena__nov06_organizer_openrouter665_v1(a) AND attempt=%s ORDER BY run_id LIMIT 1",(attempt,))
        q.execute("SET LOCAL session_replication_role=replica")
        q.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed' WHERE kind='score' AND status='pending' AND run_id<>%s",(target,))
        q.execute("SET LOCAL session_replication_role=origin")
    p.c.commit()
    release_guard(p)
    lease=claim(p)
    assert lease['status']=='leased' and lease['run_id']==target and lease['attempt']==attempt
    assert funding(p,target)['funding_source']=='host'
    identity=settle_one(p,lease)
    before_ledger=snapshot(p)['lab_arena_ledger']
    if outcome=='expired':
        with p.c.cursor() as q:
            q.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s",(target,))
        p.c.commit()
        assert rpc(p,'lab_arena_expire_leases(%s)',(ROUND,))['expired']==1
    else:
        output=target+'-accepted' if outcome=='accepted' else None
        result={'terminal_status':outcome}
        completed=rpc(p,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',(target,TOKEN,json.dumps(result),outcome,output))
        assert completed['status']==('accepted' if outcome=='accepted' else 'failed')
        assert 'confirmation_attempt' not in completed
        assert rpc(p,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',(target,TOKEN,json.dumps(result),outcome,output))['idempotent'] is True
    assert funding(p,target)['funding_source']=='host'
    rpc(p,'lab_arena_expire_leases(%s)',(ROUND,))
    with p.c.cursor() as q:
        assert scalar(q,'SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt>%s',(lease['assignment_id'],attempt))==0
        assert scalar(q,"SELECT count(*) FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='settlement' AND funding_source='host'",(identity,))==1
    assert snapshot(p)['lab_arena_ledger']==before_ledger
    apply(p)  # Terminal replay must not reset the finite attempt or old charges.


@pytest.mark.parametrize('fault',['missing_manifest','wrong_owner','wrong_generation','wrong_guard','wrong_candidate','inactive','expired'])
def test_guard_is_exact_owned_and_fail_closed(prepared,fault):
    p=prepared
    with p.c.cursor() as q:
        if fault=='inactive':
            q.execute("UPDATE public.lab_arena_restart_claim_control SET guard_commitment='',owner_commitment='',guard_expires_at=NULL,candidate_commit='',restart_scope='',restart_phase='',captured_leases='[]' WHERE singleton")
        elif fault=='expired':
            q.execute("UPDATE public.lab_arena_restart_claim_control SET guard_expires_at=clock_timestamp()-interval '1 second' WHERE singleton")
        else:
            key={'missing_manifest':'owner_commitment','wrong_owner':'owner_commitment','wrong_generation':'guard_generation','wrong_guard':'guard_commitment','wrong_candidate':'candidate_commit'}[fault]
            value='' if fault=='missing_manifest' else '101' if fault=='wrong_generation' else 'd'*40 if fault=='wrong_candidate' else 'sha256:'+'d'*64
            q.execute('SELECT set_config(%s,%s,FALSE)',('leadpoet.arena665.'+key,value))
    p.c.commit()
    before=snapshot(p)
    with pytest.raises(p.pg.Error,match='exact owned canonical guard'):
        apply(p)
    p.c.rollback()
    assert snapshot(p)==before


@pytest.mark.parametrize('fault',['leased','submitted','used_pending','upstream_auth','wrong_binding','missing_dispatch','ciphertext_drift'])
def test_active_and_unproved_changes_roll_back_without_relabeling(prepared,fault):
    p=prepared
    owner=p.owners[0]['submission_id']
    with p.c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        if fault in ['leased','submitted','used_pending']:
            run=next(r['run_id'] for r in p.scores if r['submission_id']==owner and r['status']=='pending')
            q.execute('UPDATE public.lab_arena_runs SET status=%s,runner_hotkey=%s WHERE run_id=%s',('pending' if fault=='used_pending' else fault,hotkey('665-existing-runner'),run))
        elif fault=='upstream_auth':
            q.execute("UPDATE public.lab_arena_ledger SET terminal_response=jsonb_set(terminal_response,'{account_failure_evidence,submitted_account_authentication}','false') WHERE submission_id=%s AND entry_kind='settlement'",(owner,))
        elif fault=='wrong_binding':
            q.execute("UPDATE public.lab_arena_ledger SET terminal_response=jsonb_set(terminal_response,'{account_failure_evidence,action_sequence}','99') WHERE submission_id=%s AND entry_kind='settlement'",(owner,))
        elif fault=='missing_dispatch':
            q.execute("DELETE FROM public.lab_arena_ledger WHERE submission_id=%s AND entry_kind='dispatch'",(owner,))
        else:
            q.execute("UPDATE public.lab_arena_submission_credentials SET ciphertext=%s WHERE submission_id=%s AND provider='openrouter'",(b'drifted synthetic ciphertext',owner))
        q.execute('SET LOCAL session_replication_role=origin')
    p.c.commit()
    before=snapshot(p)
    with pytest.raises(p.pg.Error):
        apply(p)
    p.c.rollback()
    assert snapshot(p)==before


def test_write_race_loses_cleanly_and_stale_old_completion_cannot_replace_result(prepared):
    p=prepared
    old=next(r for r in p.scores if r['status']=='pending')
    writer=p.pg.connect(**p.dsn)
    try:
        with writer.cursor() as q:
            q.execute('UPDATE public.lab_arena_runs SET created_at=created_at WHERE run_id=%s',(old['run_id'],))
        with pytest.raises(p.pg.Error):
            apply(p)
        p.c.rollback()
        writer.rollback()
        apply(p)
    finally:
        writer.close()
    stale=rpc(p,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',(old['run_id'],TOKEN,json.dumps({'terminal_status':'accepted'}),'accepted','stale-old-output'))
    assert stale['status']=='failed' and stale['idempotent'] is True
    with p.c.cursor() as q:
        row=scalar(q,'SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE run_id=%s',(old['run_id'],))
        assert row['status']=='failed' and row['terminal_cause']=='stage_closed' and row['output_ref'] is None


def test_prior_miner_cache_replays_on_new_host_payer_without_duplicate_charge(prepared):
    import base64
    p=prepared
    old=next(r for r in p.scores if r['status']=='accepted')
    with p.c.cursor() as q:
        cfg=scalar(q,'SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s',(ROUND,))
        scope=dict(schema_version='leadpoet.lab_arena.verifier_provider_request.v1',round_id=ROUND,evaluation_date='2026-11-06',scorer_image_digest=cfg['scorer_image_digest'],scorer_image_reference=cfg['scorer_image_reference'],scorer_policy=cfg['scorer_policy'],requested_operation_id='openrouter.chat',effective_operation_id='openrouter.chat',request_hash='sha256:'+'7'*64,outbound_body_hash='sha256:'+'8'*64)
        canonical=contracts.canonical_json(scope)
        doc=dict(request_hash=scope['request_hash'],judgment_cache_scope=scope,judgment_cache_canonical=canonical,judgment_cache_key=contracts.hash_bytes(canonical.encode()))
        original='sha256:'+'9'*64
        body=base64.b64encode(json.dumps({'choices':[{'message':{'content':'verified'}}]}).encode()).decode()
        terminal=dict(status=200,headers={'content-type':'application/json'},body_b64=body,call_succeeded=True,judgment_cache_eligible=True,provider_cost={'request_id':'original-miner-cache665'})
        q.execute('SET LOCAL session_replication_role=replica')
        for kind,entry,response,amount in [('reservation',doc,None,500),('dispatch',{},None,500),('settlement',{},terminal,120)]:
            insert(q,'lab_arena_ledger',dict(entry_kind=kind,run_id=old['run_id'],round_id=ROUND,submission_id=old['submission_id'],miner_hotkey=old['miner_hotkey'],stage=2,call_identity=original,provider='openrouter',operation_id='openrouter.chat',funding_source='miner_key',amount_microusd=amount,entry_doc=entry,terminal_response=response))
        q.execute('SET LOCAL session_replication_role=origin')
    p.c.commit()
    before=snapshot(p)['lab_arena_ledger']
    apply(p)
    age_new_pending(p)
    release_guard(p)
    lease=claim(p)
    assert lease['status']=='leased'
    identity=contracts.provider_call_identity(assignment_id=lease['assignment_id'],attempt=lease['attempt'],icp_position=lease['icp_position'],action_sequence=0,operation_id='openrouter.chat',request_hash=scope['request_hash'])
    args=(lease['run_id'],TOKEN,identity,'openrouter.chat','openrouter','host',500,json.dumps(doc),1500)
    hit=rpc(p,'lab_arena_reserve_judgment_call(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)',args)
    assert hit['status']=='cache_hit' and hit['amount_microusd']==0
    assert hit['source_call_identity']==original
    assert rpc(p,'lab_arena_reserve_judgment_call(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)',args)['status']=='settled'
    after=snapshot(p)['lab_arena_ledger']
    assert [r for r in after if r['call_identity']!=identity]==before
    destination=[r for r in after if r['call_identity']==identity]
    assert [r['entry_kind']for r in destination]==['reservation','settlement']
    assert all(r['amount_microusd']==0 and r['funding_source']=='host' for r in destination)


def test_owner_execution_rejection_is_enough_without_new_scoring_spend(prepared):
    p=prepared
    owner=p.owners[0]
    old=next(r for r in p.scores if r['submission_id']==owner['submission_id'] and r['icp_position']==0 and r['attempt']==5)
    aid=f'{ROUND}:{owner["submission_id"]}:2:9'
    with p.c.cursor() as q:
        q.execute('SET LOCAL session_replication_role=replica')
        insert(q,'lab_arena_runs',dict(run_id=aid+':1',assignment_id=aid,round_id=ROUND,submission_id=owner['submission_id'],miner_hotkey=owner['miner_hotkey'],stage=2,icp_position=9,attempt=1,kind='execute',status='failed',terminal_cause='credential_error',result_doc={'terminal_status':'credential_error'},stage_generation=2,scoring_generation_floor=0))
        q.execute('UPDATE public.lab_arena_ledger SET run_id=%s WHERE run_id=%s',(aid+':1',old['run_id']))
        q.execute('SET LOCAL session_replication_role=origin')
    p.c.commit()
    apply(p)
    with p.c.cursor() as q:
        proof=scalar(q,"SELECT recovery_doc#>>ARRAY['authentication_proofs',%s,'run_id'] FROM public.lab_arena_scoring_supersessions WHERE round_id=%s AND request_id='nov06-organizer-openrouter665-v1'",(owner['submission_id'],ROUND))
        assert proof==aid+':1'


def test_host_payer_survives_publication_and_replay(prepared):
    p=prepared
    apply(p)
    with p.c.cursor() as q:
        run=scalar(q,'SELECT run_id FROM public.lab_arena_runs a WHERE public.lab_arena__nov06_organizer_openrouter665_v1(a) ORDER BY run_id LIMIT 1')
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_rounds SET status='published',published_at=clock_timestamp(),publication_doc='{}' WHERE round_id=%s",(ROUND,))
        q.execute("UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',result_doc='{\"terminal_status\":\"accepted\"}',output_ref='published-fixture-result' WHERE run_id=%s",(run,))
        q.execute('SET LOCAL session_replication_role=origin')
    p.c.commit()
    assert funding(p,run)['funding_source']=='host'
    apply(p)
    assert funding(p,run)['funding_source']=='host'


@pytest.fixture
def prepared_company(prepared):
    from lab_arena import company_judgments,scoring
    from tests.lab_arena.company_judgments_test import _company,_input
    p=prepared
    refs_by_source={}
    with p.c.cursor() as q:
        cfg=scalar(q,'SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s',(ROUND,))
        cfg.update(integrity_policy='arena_integrity_v1',company_judgment_policy='company_judgment_v1')
        cfg['scorer_policy']=scoring.build_scorer_policy(scoring_adapter_version='qualification_integrity_v2',company_judgment_cache=True)
        for key in ('company_quality_policy','contact_policy'):
            cfg.pop(key,None)
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute('UPDATE public.lab_arena_rounds SET configuration_doc=%s::jsonb WHERE round_id=%s',(json.dumps(cfg),ROUND))
        for row in p.scores:
            source=row['scored_run_id']
            if source not in refs_by_source:
                document=scoring.build_scoring_input(scored_run_id=source,icp=_input()['icp'],companies=[_company('A')],policy=cfg['scorer_policy'],evaluation_date='2026-11-06')
                refs_by_source[source]=company_judgments.build_company_scopes(scoring_input=document,round_id=ROUND,network_name='finney',netuid=71,scorer_image_digest=cfg['scorer_image_digest'],scorer_image_reference=cfg['scorer_image_reference'],integrity_policy='arena_integrity_v1',company_judgment_policy='company_judgment_v1')
            q.execute('UPDATE public.lab_arena_runs SET company_judgment_refs=%s::jsonb,judgment_cache_key=NULL,judgment_input_hash=NULL,judgment_scope_doc=NULL WHERE run_id=%s',(json.dumps(refs_by_source[source]),row['run_id']))
        q.execute('SET LOCAL session_replication_role=origin')
        config_hash=scalar(q,"SELECT encode(extensions.digest(configuration_doc::text,'sha256'),'hex') FROM public.lab_arena_rounds WHERE round_id=%s",(ROUND,))
    p.c.commit()
    # Only this disposable fixture's config digest changes. No function source,
    # one-shot gate, cohort, identity or ciphertext check is weakened.
    p.sql=re.sub(r"(digest\(r.configuration_doc::TEXT,'sha256'\),'hex'\)\=)'[a-f0-9]{64}'",lambda m:m[1]+"'"+config_hash+"'",p.sql)
    p.refs_by_source=refs_by_source
    yield p


@pytest.mark.parametrize('outcome',['accepted','judge_error','expired'])
def test_real_nov6_company_refs_claim_and_terminal_payer_are_preserved(prepared_company,outcome):
    p=prepared_company
    apply(p)
    age_new_pending(p)
    release_guard(p)
    lease=claim(p)
    assert lease['status']=='leased'
    with p.c.cursor() as q:
        run=scalar(q,'SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE run_id=%s',(lease['run_id'],))
        assert run['company_judgment_refs']==p.refs_by_source[run['scored_run_id']]
        assert run['company_judgment_refs'] and run['judgment_scope_doc'] is None
        assert run['judgment_cache_key'] is None and run['judgment_input_hash'] is None
        assert lease['company_judgment_cache']['misses']
    p.c.commit()
    settle_one(p,lease)
    if outcome=='expired':
        with p.c.cursor() as q:
            q.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s",(lease['run_id'],))
        p.c.commit()
        rpc(p,'lab_arena_expire_leases(%s)',(ROUND,))
    else:
        # The public service's full evidence/projection checks remain unchanged.
        # Exercise the existing native completion path here, including the
        # company-ref participation trigger, with no synthetic provider proof.
        result=rpc(p,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',(lease['run_id'],TOKEN,json.dumps({'terminal_status':outcome}),outcome,lease['run_id']+'-output' if outcome=='accepted' else None))
        assert result['status']==('accepted' if outcome=='accepted' else 'failed')
        assert 'confirmation_attempt' not in result
    assert funding(p,lease['run_id'])['funding_source']=='host'
    rpc(p,'lab_arena_expire_leases(%s)',(ROUND,))
    with p.c.cursor() as q:
        saved=scalar(q,'SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE run_id=%s',(lease['run_id'],))
        assert saved['company_judgment_refs']==run['company_judgment_refs']
        assert scalar(q,'SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt>%s',(lease['assignment_id'],lease['attempt']))==0


@pytest.mark.parametrize('mutation',['update','delete','insert_null_call'])
def test_same_transaction_ledger_tuple_versions_detect_accidental_mutations(prepared,mutation):
    p=prepared
    with p.c.cursor() as q:
        q.execute('LOCK TABLE public.lab_arena_ledger IN SHARE ROW EXCLUSIVE MODE')
        run_ids=sorted({r['run_id'] for r in p.scores})
        query="""SELECT jsonb_object_agg(l.entry_id::TEXT,jsonb_build_array(l.tableoid,l.ctid,l.xmin))
          FROM unnest(%s::TEXT[])original_run(run_id)
          CROSS JOIN LATERAL(SELECT found.entry_id,found.tableoid::TEXT AS tableoid,
            found.ctid::TEXT AS ctid,found.xmin::TEXT AS xmin
           FROM public.lab_arena_ledger found WHERE found.run_id=original_run.run_id
            AND found.round_id=%s OFFSET 0)l"""
        before=scalar(q,query,(run_ids,ROUND))
        q.execute('SAVEPOINT mutation665')
        q.execute('SET LOCAL session_replication_role=replica')
        if mutation=='update':
            # Even a logically equal HOT update changes the local row version.
            q.execute('UPDATE public.lab_arena_ledger SET amount_microusd=amount_microusd WHERE entry_id=(SELECT min(entry_id) FROM public.lab_arena_ledger)')
        elif mutation=='delete':
            q.execute('DELETE FROM public.lab_arena_ledger WHERE entry_id=(SELECT min(entry_id) FROM public.lab_arena_ledger)')
        else:
            original=scalar(q,'SELECT to_jsonb(l) FROM public.lab_arena_ledger l ORDER BY entry_id LIMIT 1')
            insert(q,'lab_arena_ledger',dict(entry_kind='refusal',miner_hotkey=original['miner_hotkey'],round_id=ROUND,submission_id=original['submission_id'],run_id=original['run_id'],call_identity=None,amount_microusd=0,entry_doc={}))
        assert scalar(q,query,(run_ids,ROUND))!=before
        q.execute('ROLLBACK TO SAVEPOINT mutation665')
        assert scalar(q,query,(run_ids,ROUND))==before
    p.c.rollback()
