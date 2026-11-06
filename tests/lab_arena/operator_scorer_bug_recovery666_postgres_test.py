"""One owner grant uses normal claim/accounting/completion without rewriting work."""
from pathlib import Path
import json

import pytest

from lab_arena import company_judgments, contracts, scoring
from qualification.scoring import competition
from tests.lab_arena import nov06_organizer_openrouter665_postgres_test as old
from tests.lab_arena.company_judgments_test import _company, _input
from tests.lab_arena import legacy_company_judgments_test as legacy

SQL = Path(__file__).parents[2] / 'scripts/666-lab-arena-operator-scorer-bug-recovery.sql'
OLD_RUNTIME = 'sha256:' + '1' * 64
RUNTIME = 'sha256:' + '2' * 64
REPOSITORY = 'registry.example/scorer'
GRANT = 'lab_arena_grant_scorer_bug_recovery_v1(%s,%s,%s,%s,%s,%s,%s)'
database = old.database
prepared = old.prepared
prepared_company = old.prepared_company


@pytest.fixture(params=[True, False], ids=['authorized-organizer', 'ordinary-miner'])
def case(prepared_company, request):
    p = prepared_company
    p.organizer = request.param
    original = next(r for r in p.scores if r['submission_id'] == p.owners[0]['submission_id']
                    and r['icp_position'] == 0 and r['attempt'] == 5)
    # Two real company cache scopes are bound before665 captures its payer identity.
    with p.c.cursor() as q:
        cfg = old.scalar(q, 'SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s', (old.ROUND,))
        document = scoring.build_scoring_input(
            scored_run_id=original['scored_run_id'], icp=_input()['icp'],
            companies=[_company('A'), _company('B')], policy=cfg['scorer_policy'],
            evaluation_date='2026-11-06',
        )
        refs = company_judgments.build_company_scopes(
            scoring_input=document, round_id=old.ROUND, network_name='finney', netuid=71,
            scorer_image_digest=cfg['scorer_image_digest'], scorer_image_reference=cfg['scorer_image_reference'],
            integrity_policy='arena_integrity_v1', company_judgment_policy='company_judgment_v1',
        )
        q.execute('ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute('UPDATE public.lab_arena_runs SET company_judgment_refs=%s::jsonb WHERE assignment_id=%s',
                  (json.dumps(refs), original['assignment_id']))
        q.execute('SET LOCAL session_replication_role=origin')
        q.execute('ALTER TABLE public.lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
    p.c.commit()
    if p.organizer:
        old.apply(p)
        parent_id = original['assignment_id'] + ':6'
    else:
        text = old.SQL.read_text()
        restore = 'DO $restore665$' + text.split('DO $restore665$')[1].split('NOTIFY')[0]
        with p.c.cursor() as q:
            q.execute(text.split('-- Exact one-time data recovery.')[0] + restore + 'COMMIT;')
        parent_id = original['run_id']
    with p.c.cursor() as q:
        q.execute(SQL.read_text())
        # This trigger is ALWAYS enabled. Only disposable fixture construction
        # seeds an already failed parent's immutable historical runtime pin.
        q.execute('ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_reviewer_runtime')
        q.execute('ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='judge_error',result_doc=%s::jsonb,reviewer_runtime_doc=%s::jsonb WHERE run_id=%s", (
            json.dumps({'terminal_status':'judge_error', 'failure_diagnostic': {
                'error_class':'judge_error', 'stage':'scorer', 'reason':'unknown'}}),
            json.dumps({'runtime_image_digest': OLD_RUNTIME}), parent_id))
        q.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed' WHERE kind='score' AND status='pending'")
        for ref in refs:
            name = "A" if ref["company_index"] == 0 else "B"
            row = legacy._positive_breakdown(name, name.lower()+".example", name.lower())
            row = competition.apply_company_judgment_context(
                [document["companies"][ref["company_index"]]], [row], company_quality=False,
            )[0]
            raw = competition.raw_company_judgment(row)
            raw["company_judgment_policy"] = company_judgments.POLICY
            evidence = legacy._snapshot(ref, raw)
            evidence.update(source_score_run_id=parent_id, source_scored_run_id=original['scored_run_id'])
            old.insert(q, 'lab_arena_company_judgments', dict(
                cache_key=ref['cache_key'], authority_slot=0, scope_doc=ref['scope_doc'],
                company_input_hash=ref['company_input_hash'], evidence_hash=contracts.document_hash(evidence),
                evidence_doc=evidence, source_score_run_id=parent_id,
                source_scored_run_id=original['scored_run_id'],
                source_runner_hotkey=evidence['source_runner_hotkey'],
            ))
        q.execute('SET LOCAL session_replication_role=origin')
        q.execute('ALTER TABLE public.lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
        q.execute('ALTER TABLE public.lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_reviewer_runtime')
        q.execute('SELECT public.lab_arena_register_reviewer_runtime_v1(%s,%s,%s,%s)', (
            cfg['scorer_image_digest'], cfg['scorer_image_reference'], RUNTIME, REPOSITORY+'@'+RUNTIME))
        for key, value in dict(guard_commitment='sha256:'+'a'*64,owner_commitment='sha256:'+'b'*64,
                               guard_generation='100',candidate_commit='c'*40).items():
            q.execute('SELECT set_config(%s,%s,FALSE)', ('leadpoet.arena666.'+key,value))
        parent = old.scalar(q, 'SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE run_id=%s', (parent_id,))
        parent_hash = 'sha256:' + old.scalar(q, "SELECT encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex') FROM public.lab_arena_runs a WHERE run_id=%s", (parent_id,))
    p.c.commit()
    p.parent, p.parent_hash, p.refs = parent, parent_hash, refs
    yield p


def grant(p, *, dry=False, **changes):
    values = dict(parent=p.parent['run_id'], parent_hash=p.parent_hash, runtime=RUNTIME,
                  request_id='operator-scorer-bug-test', actor='fixture-owner', correction='git:reviewed-scorer-fix', dry=dry)
    values.update(changes)
    with p.c.cursor() as q:
        result = old.scalar(q, 'SELECT public.'+GRANT, tuple(values.values()))
    p.c.commit()
    return result


def snapshot(p):
    result = old.snapshot(p)
    with p.c.cursor() as q:
        for table, key in [('lab_arena_company_judgments','cache_key,authority_slot'),
                           ('lab_arena_signal_judgments','cache_key,authority_slot'),
                           ('lab_arena_scoring_supersessions','round_id,generation_floor')]:
            result[table] = old.scalar(q, 'SELECT coalesce(jsonb_agg(to_jsonb(a) ORDER BY '+key+"),'[]') FROM public."+table+' a')
    return result


def test_dry_run_then_single_grant_is_idempotent_and_preserves_all_state(case):
    p = case
    old.release_guard(p)  # One failed-parent grant needs no hold, drain or restart.
    before = snapshot(p)
    preview = grant(p, dry=True)
    assert preview['status'] == 'dry_run'
    assert len(preview['checkpoints']) == 2
    assert snapshot(p) == before
    result = grant(p)
    assert result['status'] == 'granted' and result['idempotent'] is False
    assert result['child_identity'] == preview['child_identity']
    assert result['organizer_openrouter_authorized'] is p.organizer
    after = snapshot(p)
    for table in set(before)-{'lab_arena_runs','lab_arena_scoring_supersessions'}:
        assert after[table] == before[table]
    originals = {r['run_id']:r for r in before['lab_arena_runs']}
    assert {r['run_id']:r for r in after['lab_arena_runs'] if r['run_id'] in originals} == originals
    child_id = result['child_identity']['run_id']
    assert len(after['lab_arena_runs']) == len(originals)+1
    child = next(r for r in after['lab_arena_runs'] if r['run_id'] == child_id)
    assert child['company_judgment_refs'] == p.parent['company_judgment_refs']
    assert child['scoring_generation_floor'] == p.parent['scoring_generation_floor']
    assert old.funding(p,child_id)['funding_source'] == ('host' if p.organizer else 'miner_key')
    assert old.funding(p,child_id,'deepline')['funding_source'] == 'miner_key'
    assert old.funding(p,p.parent['run_id'])['funding_source'] == ('host' if p.organizer else 'miner_key')
    assert grant(p)['idempotent'] is True
    assert snapshot(p) == after


@pytest.mark.parametrize('fault', ['stale_hash','accepted','later','calls_pending','runtime','same_runtime',
                                   'expired','published','generation','provider_failure'])
def test_invalid_grant_rolls_back_without_any_work_change(case, fault):
    p = case
    changes = {}
    with p.c.cursor() as q:
        q.execute('ALTER TABLE public.lab_arena_runs DISABLE TRIGGER lab_arena_runs_terminal')
        q.execute('SET LOCAL session_replication_role=replica')
        if fault == 'stale_hash':
            changes['parent_hash'] = 'sha256:'+'0'*64
        elif fault == 'accepted':
            q.execute("UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted',output_ref='accepted' WHERE run_id=%s", (p.parent['assignment_id']+':4',))
        elif fault == 'later':
            child = dict(p.parent)
            child.update(run_id=child['assignment_id']+':99', attempt=99, status='failed')
            for key in ['created_at','updated_at','reviewer_runtime_doc']:
                child.pop(key, None)
            old.insert(q, 'lab_arena_runs', child)
        elif fault == 'calls_pending':
            old.insert(q, 'lab_arena_ledger', dict(entry_kind='reservation',run_id=p.parent['run_id'],
                round_id=old.ROUND,submission_id=p.parent['submission_id'],miner_hotkey=p.parent['miner_hotkey'],
                stage=2,call_identity='sha256:'+'7'*64,provider='openrouter',operation_id='openrouter.chat',
                funding_source='host' if p.organizer else 'miner_key',amount_microusd=500,entry_doc={}))
        elif fault == 'runtime':
            changes['runtime'] = 'sha256:'+'3'*64
        elif fault == 'same_runtime':
            q.execute('UPDATE public.lab_arena_reviewer_runtimes SET runtime_image_digest=%s,runtime_image_reference=%s',
                      (OLD_RUNTIME,REPOSITORY+'@'+OLD_RUNTIME))
            changes['runtime'] = OLD_RUNTIME
        elif fault == 'expired':
            q.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,final_scoring_close}',to_jsonb('2000-01-01T00:00:00Z'::text)) WHERE round_id=%s", (old.ROUND,))
        elif fault == 'published':
            q.execute("UPDATE public.lab_arena_rounds SET published_at=clock_timestamp(),status='published',publication_doc='{}' WHERE round_id=%s", (old.ROUND,))
        elif fault == 'generation':
            q.execute('UPDATE public.lab_arena_rounds SET scoring_generation_floor=1 WHERE round_id=%s',(old.ROUND,))
        elif fault == 'provider_failure':
            q.execute("UPDATE public.lab_arena_runs SET result_doc=jsonb_set(result_doc,'{failure_diagnostic,stage}','\"provider\"') WHERE run_id=%s",(p.parent['run_id'],))
            changes['parent_hash'] = 'sha256:' + old.scalar(q,"SELECT encode(extensions.digest(to_jsonb(a)::text,'sha256'),'hex') FROM public.lab_arena_runs a WHERE run_id=%s",(p.parent['run_id'],))
        q.execute('SET LOCAL session_replication_role=origin')
        q.execute('ALTER TABLE public.lab_arena_runs ENABLE ALWAYS TRIGGER lab_arena_runs_terminal')
    p.c.commit()
    before = snapshot(p)
    with pytest.raises(p.pg.Error):
        grant(p, **changes)
    p.c.rollback()
    assert snapshot(p) == before


@pytest.mark.parametrize('outcome',['accepted','judge_error','expired'])
def test_real_claim_and_terminal_paths_never_create_another_attempt(case,outcome):
    p = case
    result = grant(p)
    child_id = result['child_identity']['run_id']
    with p.c.cursor() as q:
        q.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '1 hour' WHERE run_id=%s",(child_id,))
    p.c.commit()
    old.release_guard(p)
    lease = old.claim(p)
    assert lease['status'] == 'leased' and lease['run_id'] == child_id
    with p.c.cursor() as q:
        persisted = old.scalar(q,'SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE run_id=%s',(child_id,))
    p.c.commit()
    assert persisted['reviewer_runtime_doc']['runtime_image_digest'] == RUNTIME
    assert persisted['claim_response']['reviewer_runtime_doc'] == persisted['reviewer_runtime_doc']
    assert len(lease['company_judgment_cache']['hits']) == 2
    assert lease['company_judgment_cache']['misses'] == []
    if p.organizer:
        old.settle_one(p,lease)
    if outcome == 'expired':
        with p.c.cursor() as q:
            q.execute("UPDATE public.lab_arena_runs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE run_id=%s",(child_id,))
        p.c.commit()
        old.rpc(p,'lab_arena_expire_leases(%s)',(old.ROUND,))
    else:
        old.rpc(p,'lab_arena_complete_attempt(%s,%s,%s::jsonb,%s,%s)',(
            child_id,old.TOKEN,json.dumps({'terminal_status':outcome}),outcome,
            child_id+'-output' if outcome == 'accepted' else None))
    old.rpc(p,'lab_arena_expire_leases(%s)',(old.ROUND,))
    assert old.funding(p,child_id)['funding_source'] == ('host' if p.organizer else 'miner_key')
    with p.c.cursor() as q:
        assert old.scalar(q,'SELECT count(*) FROM public.lab_arena_runs WHERE assignment_id=%s AND attempt>%s',
                          (lease['assignment_id'],lease['attempt'])) == 0
        saved = old.scalar(q,'SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE run_id=%s',(child_id,))
        assert saved['company_judgment_refs'] == p.parent['company_judgment_refs']
    p.c.commit()
    assert grant(p)['idempotent'] is True
    if outcome == 'judge_error':
        with p.c.cursor() as q:
            sha = 'sha256:' + old.scalar(q,"SELECT encode(extensions.digest(to_jsonb(a)::text,'sha256'),'hex') FROM public.lab_arena_runs a WHERE run_id=%s",(child_id,))
        with pytest.raises(p.pg.Error):
            grant(p,parent=child_id,parent_hash=sha,request_id='no-chained-grant')
        p.c.rollback()


def test_service_cannot_grant_even_with_matching_inputs(case):
    p = case
    with pytest.raises(p.pg.errors.InsufficientPrivilege):
        old.rpc(p, GRANT, (p.parent['run_id'],p.parent_hash,RUNTIME,
                          'unauthorized-service-grant','service','reviewed-correction',False))


def test_changed_runtime_selection_after_grant_cannot_lease_child(case):
    p = case
    child_id = grant(p)['child_identity']['run_id']
    with p.c.cursor() as q:
        cfg = old.scalar(q,'SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s',(old.ROUND,))
        different = 'sha256:'+'3'*64
        q.execute('SELECT public.lab_arena_register_reviewer_runtime_v1(%s,%s,%s,%s)',(
            cfg['scorer_image_digest'],cfg['scorer_image_reference'],different,REPOSITORY+'@'+different))
        q.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '1 hour' WHERE run_id=%s",(child_id,))
    p.c.commit()
    old.release_guard(p)
    assert old.claim(p)['status'] != 'leased'
    with p.c.cursor() as q:
        assert old.scalar(q,'SELECT status FROM public.lab_arena_runs WHERE run_id=%s',(child_id,)) == 'pending'


def test_prior_costs_still_exhaust_the_same_cumulative_budget(case):
    p = case
    # Existing settled scorer spend is not erased, moved to the organizer, or
    # assigned a new generation. Only disposable fixture seeding is synthetic.
    identity = 'sha256:'+'8'*64
    with p.c.cursor() as q:
        cap = old.scalar(q,"SELECT (configuration_doc->>'scoring_cap_microusd')::bigint FROM public.lab_arena_rounds WHERE round_id=%s",(old.ROUND,))
        q.execute('SET LOCAL session_replication_role=replica')
        for kind in ['reservation','dispatch','settlement']:
            old.insert(q,'lab_arena_ledger',dict(entry_kind=kind,run_id=p.parent['run_id'],
                round_id=old.ROUND,submission_id=p.parent['submission_id'],miner_hotkey=p.parent['miner_hotkey'],
                stage=2,call_identity=identity,provider='openrouter',operation_id='openrouter.chat',
                funding_source='host' if p.organizer else 'miner_key',amount_microusd=cap,
                entry_doc={},terminal_response={'status':200,'call_succeeded':True} if kind=='settlement' else None))
        q.execute('SET LOCAL session_replication_role=origin')
    p.c.commit()
    ledger_before = snapshot(p)['lab_arena_ledger']
    child_id = grant(p)['child_identity']['run_id']
    with p.c.cursor() as q:
        q.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '1 hour' WHERE run_id=%s",(child_id,))
    p.c.commit()
    old.release_guard(p)
    lease = old.claim(p)
    assert lease['status'] == 'leased' and lease['run_id'] == child_id
    request = 'sha256:'+'6'*64
    call = contracts.provider_call_identity(assignment_id=lease['assignment_id'],attempt=lease['attempt'],
        icp_position=lease['icp_position'],action_sequence=1,operation_id='openrouter.chat',request_hash=request)
    result = old.rpc(p,'lab_arena_reserve_call(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)',(
        child_id,old.TOKEN,call,'openrouter.chat','openrouter','host' if p.organizer else 'miner_key',
        500,json.dumps(dict(request_hash=request,base_call_identity=call,action_sequence=1,
                            provider_attempt=1,max_output_tokens=8)),1500))
    assert result['status'] == 'refused'
    assert result['reason'] == 'money_cap'
    assert [r for r in snapshot(p)['lab_arena_ledger'] if r['call_identity'] != call] == ledger_before


def test_existing_stage_call_quota_is_not_reset_by_the_grant(case):
    p = case
    with p.c.cursor() as q:
        cfg = old.scalar(q,'SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s',(old.ROUND,))
        quota = cfg['scoring_call_quotas']['openrouter'] * cfg['stage_2_icp_count'] * cfg['max_attempts_per_assignment']
        q.execute('SET LOCAL session_replication_role=replica')
        q.execute("""INSERT INTO public.lab_arena_ledger(entry_kind,run_id,round_id,submission_id,miner_hotkey,
          stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc,terminal_response)
          SELECT kind,%s,%s,%s,%s,2,'sha256:'||encode(extensions.digest('quota666-'||n::text,'sha256'),'hex'),
            'openrouter','openrouter.chat',%s,0,'{}',
            CASE WHEN kind='settlement' THEN '{"status":200,"call_succeeded":true}'::jsonb ELSE NULL END
          FROM generate_series(1,%s)n CROSS JOIN(VALUES('reservation',1),('settlement',2))k(kind,ordering)
          ORDER BY n,ordering""", (p.parent['run_id'],old.ROUND,p.parent['submission_id'],p.parent['miner_hotkey'],
              'host' if p.organizer else 'miner_key',quota))
        q.execute('SET LOCAL session_replication_role=origin')
    p.c.commit()
    child_id = grant(p)['child_identity']['run_id']
    with p.c.cursor() as q:
        q.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '1 hour' WHERE run_id=%s",(child_id,))
    p.c.commit()
    old.release_guard(p)
    lease = old.claim(p)
    assert lease['status'] == 'leased' and lease['run_id'] == child_id
    request = 'sha256:'+'5'*64
    call = contracts.provider_call_identity(assignment_id=lease['assignment_id'],attempt=lease['attempt'],
        icp_position=lease['icp_position'],action_sequence=2,operation_id='openrouter.chat',request_hash=request)
    result = old.rpc(p,'lab_arena_reserve_call(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)',(
        child_id,old.TOKEN,call,'openrouter.chat','openrouter','host' if p.organizer else 'miner_key',
        500,json.dumps(dict(request_hash=request,base_call_identity=call,action_sequence=2,
                            provider_attempt=1,max_output_tokens=8)),1500))
    assert result['status'] == 'refused' and result['reason'] == 'stage_quota'


def test_migration_is_replay_safe_and_rejects_unknown_helper_source(case):
    p = case
    before = snapshot(p)
    with p.c.cursor() as q:
        q.execute(SQL.read_text())
    p.c.commit()
    assert snapshot(p) == before
    with p.c.cursor() as q:
        original = old.scalar(q,"SELECT pg_get_functiondef('public.lab_arena__operator_scorer_bug_child_v1(public.lab_arena_runs)'::regprocedure)")
        q.execute("""CREATE OR REPLACE FUNCTION public.lab_arena__operator_scorer_bug_child_v1(p_run public.lab_arena_runs)
          RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public
          AS 'SELECT FALSE'""")
        with pytest.raises(p.pg.Error,match='installed function differs'):
            q.execute(SQL.read_text())
    p.c.rollback()
    with p.c.cursor() as q:
        assert old.scalar(q,"SELECT pg_get_functiondef('public.lab_arena__operator_scorer_bug_child_v1(public.lab_arena_runs)'::regprocedure)") == original
    assert snapshot(p) == before


@pytest.mark.parametrize("field", ["parent_hash", "runtime"])
def test_null_required_hash_is_rejected_at_input_validation(case, field):
    p = case
    before = snapshot(p)
    with pytest.raises(p.pg.Error, match="request invalid"):
        grant(p, **{field: None})
    p.c.rollback()
    assert snapshot(p) == before


def test_claim_locks_mapping_until_the_existing_runtime_pin_commits(case):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    import time

    p = case
    child_id = grant(p)['child_identity']['run_id']
    with p.c.cursor() as q:
        cfg = old.scalar(q,'SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s',(old.ROUND,))
        q.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '1 hour' WHERE run_id=%s",(child_id,))
        # Test-only barrier after the real candidate predicate, before the real
        # runtime-pin trigger. This does not replace either production function.
        q.execute("""CREATE FUNCTION public.test_pause_runtime666() RETURNS trigger LANGUAGE plpgsql AS $$
          BEGIN IF OLD.status='pending' AND NEW.status='leased' THEN
            PERFORM pg_advisory_xact_lock(666300); END IF; RETURN NEW; END; $$;
          CREATE TRIGGER aaa_pause_runtime666 BEFORE UPDATE ON public.lab_arena_runs
            FOR EACH ROW EXECUTE FUNCTION public.test_pause_runtime666();""")
        q.execute('SELECT pg_advisory_lock(666300)')
    p.c.commit()
    old.release_guard(p)
    watcher = p.pg.connect(**p.dsn)
    watcher.autocommit = True
    claimant = SimpleNamespace(pg=p.pg,dsn={**p.dsn,'application_name':'arena666-race-claim'})
    registrar = SimpleNamespace(pg=p.pg,dsn={**p.dsn,'application_name':'arena666-race-register'})

    def waiting(name, event=None):
        with watcher.cursor() as q:
            return old.scalar(q,"SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE application_name=%s AND wait_event_type='Lock' AND (%s IS NULL OR wait_event=%s))",(name,event,event))

    def wait_until(predicate):
        end = time.monotonic()+5
        while time.monotonic()<end:
            if predicate():
                return
            time.sleep(0.01)
        raise AssertionError('runtime race barrier did not reach the expected lock')

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            claim = pool.submit(old.claim,claimant)
            registered = None
            try:
                wait_until(lambda: waiting('arena666-race-claim','advisory'))
                different = 'sha256:'+'3'*64
                registered = pool.submit(old.rpc,registrar,
                    'lab_arena_register_reviewer_runtime_v1(%s,%s,%s,%s)',(
                        cfg['scorer_image_digest'],cfg['scorer_image_reference'],different,REPOSITORY+'@'+different))
                # Before the fix this registration commits while the claim is
                # paused, and the subsequent pin silently selects the new image.
                wait_until(lambda: registered.done() or waiting('arena666-race-register'))
                blocked = not registered.done()
            finally:
                with p.c.cursor() as q:
                    q.execute('SELECT pg_advisory_unlock(666300)')
                p.c.commit()
            lease = claim.result(timeout=5)
            assert registered is not None
            registered.result(timeout=5)
        assert blocked
        assert lease['status']=='leased' and lease['run_id']==child_id
        with p.c.cursor() as q:
            pinned = old.scalar(q,'SELECT reviewer_runtime_doc FROM public.lab_arena_runs WHERE run_id=%s',(child_id,))
            assert pinned['runtime_image_digest']==RUNTIME
            assert old.scalar(q,'SELECT runtime_image_digest FROM public.lab_arena_reviewer_runtimes WHERE grading_image_digest=%s',(cfg['scorer_image_digest'],))==different
    finally:
        watcher.close()
        p.c.rollback()
        with p.c.cursor() as q:
            q.execute('SELECT pg_advisory_unlock(666300)')
            q.execute('DROP TRIGGER aaa_pause_runtime666 ON public.lab_arena_runs')
            q.execute('DROP FUNCTION public.test_pause_runtime666()')
        p.c.commit()
