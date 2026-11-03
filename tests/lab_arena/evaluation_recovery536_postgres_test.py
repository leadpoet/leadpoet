"""Controlled recovery preserves real immutable history and ordinary authority."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import base64
import json
from pathlib import Path

import pytest
from lab_arena import contracts
from tests.lab_arena import model_policy_supersession517_postgres_test as base
from tests.lab_arena import judge_final_deadline530_postgres_test as current

SQL = Path(__file__).parents[2] / 'scripts/536-lab-arena-evaluation-recovery.sql'
SIG = 'public.lab_arena_recover_evaluation_v1(text,jsonb,bigint,bigint,jsonb,jsonb,jsonb,text,text)'


@pytest.fixture(scope='module')
def database():
    gen = current.database.__wrapped__()
    db = next(gen)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute(SQL.read_text())
            cur.execute(SQL.read_text())
        yield db
    finally:
        gen.close()


def seed(db, *, gaps=1):
    store, row, old, _ = base.seed(db)
    cfg = deepcopy(row['configuration_doc'])
    now = datetime.now(timezone.utc)
    for index, (key, value) in enumerate(sorted(cfg['schedule'].items(), key=lambda pair: pair[1])):
        cfg['schedule'][key] = (now - timedelta(days=1) + timedelta(minutes=index)).isoformat()
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET status='cancelled',cancel_reason='scoring_incomplete',configuration_doc=%s::jsonb WHERE round_id=%s", (json.dumps(cfg), row['round_id']))
        if gaps == 38:
            participants = deepcopy(row['participants'])
            coordinates = []
            for group, positions in enumerate([range(2, 10), range(10), range(10), range(10)]):
                participant = deepcopy(participants[1])
                participant['submission_id'] = 'recovery-gap-' + str(group)
                participant['miner_hotkey'] = base.company.original.hotkey('recovery-gap-' + str(group))
                participants.append(participant)
                cur.execute("INSERT INTO public.lab_arena_submissions SELECT (jsonb_populate_record(NULL::public.lab_arena_submissions,to_jsonb(s)||jsonb_build_object('submission_id',%s,'miner_hotkey',%s))).* FROM public.lab_arena_submissions s WHERE s.submission_id=%s", (participant['submission_id'], participant['miner_hotkey'], old[1]['submission_id']))
                coordinates.extend((participant['submission_id'], participant['miner_hotkey'], position) for position in positions)
            cfg['stage_1_icp_count'] = cfg['stage_2_icp_count'] = 5
            cur.execute('UPDATE public.lab_arena_rounds SET participants=%s::jsonb,configuration_doc=%s::jsonb WHERE round_id=%s', (json.dumps(participants), json.dumps(cfg), row['round_id']))
        else:
            coordinates = [(old[1]['submission_id'], row['participants'][1]['miner_hotkey'], i + 1) for i in range(gaps)]
        for submission, hotkey, position in coordinates:
            parent = row['round_id'] + ':' + submission + ':2'
            assignment = parent + ':' + str(position)
            cur.execute("INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,status,stage_generation,kind,terminal_cause,terminal_doc,scoring_generation_floor) SELECT %s,%s,round_id,%s,%s,2,%s,1,'failed',stage_generation,'execute','stage_closed','{\"previous_status\":\"pending\",\"infrastructure_incomplete\":true}'::jsonb,0 FROM public.lab_arena_runs WHERE run_id=%s", (assignment + ':1', assignment, submission, hotkey, position, old[1]['scored_run_id']))
        cur.execute('SET LOCAL session_replication_role=origin')
    row = store.get_round(row['round_id'])
    new = deepcopy(cfg)
    new['scorer_image_digest'] = 'sha256:' + '9' * 64
    new['scorer_image_reference'] = cfg['scorer_image_reference'].rsplit('@', 1)[0] + '@' + new['scorer_image_digest']
    new['schedule'].update(stage_2_close=(now + timedelta(hours=2)).isoformat(),
                           final_scoring_close=(now + timedelta(hours=4)).isoformat(),
                           publication_deadline=(now + timedelta(hours=4, seconds=1)).isoformat())
    items = deepcopy(old)
    for item in items:
        for ref in item['company_judgment_refs']:
            scope = ref['scope_doc']
            scope.update(scorer_image_digest=new['scorer_image_digest'], scorer_image_reference=new['scorer_image_reference'])
            scope.pop('cache_key')
            ref['cache_key'] = scope['cache_key'] = contracts.document_hash(scope)
    return store, row, new, items


def invoke(cur, row, cfg, items, *, excluded=(), request='recovery535-request-0001'):
    cur.execute('SELECT ' + SIG.split('(')[0] + '(%s,%s::jsonb,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s)',
                (row['round_id'], json.dumps(row['configuration_doc']), row['status_generation'],
                 row['stage_generation'], json.dumps(cfg), json.dumps(items), json.dumps(list(excluded)), request, 'test-operator'))
    return cur.fetchone()[0]


def test_recovery_preserves_source_history_fees_and_queues_ordinary_attempts(database):
    store, row, cfg, items = seed(database, gaps=38)
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute("SELECT run_id,to_jsonb(a)-ARRAY['per_icp_score','qualification_doc','updated_at'] FROM public.lab_arena_runs a WHERE round_id=%s AND kind='execute' ORDER BY run_id", (row['round_id'],))
            before = cur.fetchall()
            result = invoke(cur, row, cfg, items)
            assert result['queued_executions'] == 38 and result['queued_judges'] == 1
            assert invoke(cur, row, cfg, items)['status'] == 'existing'
            cur.execute("SELECT run_id,to_jsonb(a)-ARRAY['per_icp_score','qualification_doc','updated_at'] FROM public.lab_arena_runs a WHERE round_id=%s AND kind='execute' AND attempt=1 ORDER BY run_id", (row['round_id'],))
            assert cur.fetchall() == before
            cur.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute' AND attempt=2 AND status='pending' AND stage_generation=%s", (row['round_id'], result['generation_floor']))
            assert cur.fetchone() == (38,)
            cur.execute('SELECT old_round_doc,new_round_doc,old_score_projections,recovery_doc FROM public.lab_arena_scoring_supersessions WHERE round_id=%s', (row['round_id'],))
            old, new, projections, audit = cur.fetchone()
            assert old['status'] == 'cancelled' and new['status'] == 'stage2'
            assert old['participants'] == new['participants']
            assert new['configuration_doc']['scorer_policy'] == old['configuration_doc']['scorer_policy']
            assert any(x['per_icp_score'] == 42 for x in projections.values())
            assert audit['operation'] == 'cancelled_evaluation_recovery_v1'
    finally:
        store.close()


@pytest.mark.parametrize('fault', ['published', 'rewarded', 'unpaused', 'stale', 'model', 'funds', 'missing_work', 'old_cache', 'active', 'open_provider'])
def test_unsafe_recovery_is_atomic(database, fault):
    store, row, cfg, items = seed(database)
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            if fault in ('published', 'rewarded', 'unpaused', 'active'):
                cur.execute('SET LOCAL session_replication_role=replica')
                if fault == 'published':
                    cur.execute("UPDATE public.lab_arena_rounds SET status='published' WHERE round_id=%s", (row['round_id'],))
                if fault == 'rewarded':
                    cur.execute("UPDATE public.lab_arena_rounds SET reward_basis_hash=%s WHERE round_id=%s", ('sha256:' + 'b' * 64, row['round_id']))
                if fault == 'unpaused':
                    cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=false')
                if fault == 'active':
                    cur.execute("UPDATE public.lab_arena_runs SET status='leased' WHERE run_id=%s", (items[0]['scored_run_id'],))
                cur.execute('SET LOCAL session_replication_role=origin')
            if fault == 'open_provider':
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd) SELECT 'reservation',miner_hotkey,round_id,submission_id,run_id,stage,%s,'openrouter','openrouter.chat','host',0 FROM public.lab_arena_runs WHERE run_id=%s", ('sha256:' + 'e' * 64, items[0]['scored_run_id']))
                cur.execute('SET LOCAL session_replication_role=origin')
            if fault == 'stale':
                row['stage_generation'] += 1
            if fault == 'model':
                cfg['scorer_policy']['judge_models']['role_batch_check'] = 'unapproved/model'
            if fault == 'funds':
                cfg['execution_cap_microusd'] *= 2
            if fault == 'missing_work':
                items = items[:-1]
            if fault == 'old_cache':
                for item in items:
                    for ref in item['company_judgment_refs']:
                        ref['scope_doc']['scorer_image_digest'] = row['configuration_doc']['scorer_image_digest']
            cur.execute('SAVEPOINT rejected')
            with pytest.raises(database[0].Error):
                invoke(cur, row, cfg, items)
            cur.execute('ROLLBACK TO SAVEPOINT rejected')
            cur.execute('SELECT count(*) FROM public.lab_arena_scoring_supersessions')
            assert cur.fetchone() == (0,)
    finally:
        store.close()


@pytest.mark.parametrize('key', ['stage_2_close', 'final_scoring_close', 'publication_deadline'])
@pytest.mark.parametrize('missing', [True, False])
def test_missing_or_null_deadline_is_not_authority(database, key, missing):
    store, row, cfg, items = seed(database)
    try:
        if missing:
            cfg['schedule'].pop(key)
        else:
            cfg['schedule'][key] = None
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            with pytest.raises(database[0].Error):
                invoke(cur, row, cfg, items)
    finally:
        store.close()


def test_untrusted_roles_cannot_recover_or_queue(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        for role in ['anon', 'authenticated', 'service_role', 'lab_arena_service']:
            cur.execute('SELECT has_function_privilege(%s,%s,%s)', (role, SIG, 'EXECUTE'))
            assert cur.fetchone() == (False,)


def route_exclusion(cur, row, item, *, status=404):
    """Historic native ledger fixture; the protected helper checks object bytes."""
    call = 'sha256:' + 'c' * 64
    native = {'status': status, 'headers': {}, 'body_b64': base64.b64encode(json.dumps({
        'error': {'code': status, 'message': 'No endpoints found matching your data policy (Zero data retention).'}
    }).encode()).decode(), 'call_succeeded': False}
    cur.execute('SET LOCAL session_replication_role=replica')
    terminal_id = None
    for kind in ('reservation', 'dispatch', 'settlement'):
        cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc,terminal_response) SELECT %s,miner_hotkey,round_id,submission_id,run_id,stage,%s,'openrouter','openrouter.chat','miner_key',0,%s::jsonb,%s::jsonb FROM public.lab_arena_runs WHERE run_id=%s RETURNING entry_id", (kind, call, json.dumps({'action_sequence': 1, 'request_hash': 'sha256:' + 'd' * 64}), json.dumps(native) if kind == 'settlement' else None, item['scored_run_id']))
        terminal_id = cur.fetchone()[0]
    cur.execute('SET LOCAL session_replication_role=origin')
    return {'submission_id': item['submission_id'], 'run_id': item['scored_run_id'],
            'output_ref': item['output_ref'], 'output_sha256': 'sha256:' + 'a' * 64,
            'accepted_empty': True, 'settlement_entry_id': terminal_id,
            'terminal_response': native, 'model': 'fixture/submitted-model',
            'current_route_unavailable': True}


def test_native_route_exclusion_is_durable_not_a_zero(database):
    store, row, cfg, items = seed(database, gaps=38)
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            proof = route_exclusion(cur, row, items[1])
            cur.execute('SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE run_id=%s', (items[1]['scored_run_id'],))
            old_execution = cur.fetchone()[0]
            result = invoke(cur, row, cfg, items[:1], excluded=[proof])
            assert result['queued_executions'] == 38 and result['queued_judges'] == 1
            cur.execute('SELECT to_jsonb(a) FROM public.lab_arena_runs a WHERE run_id=%s', (items[1]['scored_run_id'],))
            assert cur.fetchone()[0] == old_execution
            cur.execute('SELECT public.lab_arena_scoring_policy_history_v1(%s)', (row['round_id'],))
            public = cur.fetchone()[0]['revisions'][0]['recovery']['excluded_participants']
            assert public == [{'submission_id': proof['submission_id'],
                               'miner_hotkey': row['participants'][1]['miner_hotkey'],
                               'status': 'provider_blocked', 'evaluation': 'not_evaluated',
                               'reason': 'openrouter_zdr_route_unavailable'}]
            assert 'terminal_response' not in json.dumps(public)
            cur.execute('SELECT count(*) FROM public.lab_arena_ledger WHERE round_id=%s', (row['round_id'],))
            assert cur.fetchone() == (3,)
    finally:
        store.close()


@pytest.mark.parametrize('fault', ['non404', 'different_receipt', 'baseline', 'duplicate'])
def test_exclusion_requires_native_proof_not_arbitrary_cohort_drop(database, fault):
    store, row, cfg, items = seed(database, gaps=38)
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            proof = route_exclusion(cur, row, items[1], status=401 if fault == 'non404' else 404)
            if fault == 'different_receipt':
                proof['terminal_response']['body_b64'] = base64.b64encode(b'{}').decode()
            if fault == 'baseline':
                proof['submission_id'] = items[0]['submission_id']
            proofs = [proof, proof] if fault == 'duplicate' else [proof]
            cur.execute('SAVEPOINT rejected')
            with pytest.raises(database[0].Error):
                invoke(cur, row, cfg, items[:1], excluded=proofs)
            cur.execute('ROLLBACK TO SAVEPOINT rejected')
            cur.execute('SELECT count(*) FROM public.lab_arena_scoring_supersessions')
            assert cur.fetchone() == (0,)
    finally:
        store.close()


def test_current_floor_v3_initializer_reuses_roots_and_rejects_stage_swap(database):
    store, row, cfg, items = seed(database, gaps=0)
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            result = invoke(cur, row, cfg, items)
            # Advance the ordinary disposable phase with its original records.
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_closed' WHERE round_id=%s", (row['round_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
            cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=false WHERE singleton')
            cur.execute('SAVEPOINT wrong_stage')
            with pytest.raises(database[0].Error, match='cohort incomplete'):
                cur.execute('SELECT public.lab_arena_open_scoring_v3(%s,2::smallint,%s::jsonb)', (row['round_id'], json.dumps(items[:1])))
            cur.execute('ROLLBACK TO SAVEPOINT wrong_stage')
            cur.execute('SELECT public.lab_arena_open_scoring_v3(%s,2::smallint,%s::jsonb)', (row['round_id'], json.dumps(items[1:])))
            assert cur.fetchone()[0]['assignments'] == 1
            cur.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' AND scoring_generation_floor=%s", (row['round_id'], result['generation_floor']))
            assert cur.fetchone() == (2,)
    finally:
        store.close()


@pytest.mark.parametrize('statement', [
    'GRANT INSERT ON public.lab_arena_scoring_supersessions TO lab_arena_service',
    'ALTER TABLE public.lab_arena_scoring_supersessions DISABLE TRIGGER lab_arena_scoring_supersessions_append_only',
    'GRANT EXECUTE ON FUNCTION ' + SIG + ' TO anon',
    'ALTER FUNCTION ' + SIG + ' SECURITY INVOKER',
    "ALTER FUNCTION public.lab_arena__queue_recovery_judges_v1(public.lab_arena_rounds,jsonb) SET search_path=public",
    'GRANT EXECUTE ON FUNCTION public.lab_arena_scoring_policy_history_v1(text) TO anon',
    'GRANT EXECUTE ON FUNCTION public.lab_arena_open_scoring_v3(text,smallint,jsonb) TO anon',
])
def test_installed_migration_refuses_security_drift(database, statement):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute(statement)
        cur.execute('SAVEPOINT rejected')
        with pytest.raises(database[0].Error, match='differs'):
            cur.execute(SQL.read_text())
        cur.execute('ROLLBACK TO SAVEPOINT rejected')
        # The connection context is deliberately rolled back: drift is a test.
        conn.rollback()


def test_real_claim_baseline_after_old_cutoff_then_all_38_ordinary_executions(database):
    from lab_arena.store import hash_lease_token
    from tests.lab_arena.test_lab_arena_migration_postgres import _complete_accepted
    store, row, cfg, items = seed(database, gaps=38)
    runner = base.company.original.hotkey('recovery536-actual-claim')
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            result = invoke(cur, row, cfg, items)
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_submissions SET source_ref='arena/'||round_id||'/sources/'||submission_id||'.tar.gz',source_size_bytes=32 WHERE round_id=%s", (row['round_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
            cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=false WHERE singleton')
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE round_id=%s AND status='pending'", (row['round_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        first, token, _, _ = base.company.original.claim(store, row['round_id'], runner, excluded=[runner])
        assert first['status'] == 'leased' and first['kind'] == 'score' and first['stage'] == 1, first
        assert datetime.fromisoformat(row['configuration_doc']['schedule']['stage_1_scoring_close']) < datetime.now(timezone.utc)
        # Even with unused physical slots, pending miner executions cannot pass
        # the baseline's fresh authoritative judgment.
        second, token2, _, _ = base.company.original.claim(store, row['round_id'], runner, parallelism=2, excluded=[runner])
        assert second['status'] == 'no_pending', second
        assert base.company.complete_subset(store, first, token, items[0]['company_judgment_refs'])['status'] == 'accepted'
        assert store.get_run(first['run_id'])['result_doc'] is not None
        executed = []
        for _ in range(40):
            lease, token, _, _ = base.company.original.claim(store, row['round_id'], runner, excluded=[runner])
            if lease['status'] != 'leased':
                break
            if lease['kind'] == 'score':
                base.company.complete_subset(store, lease, token, items[1]['company_judgment_refs'])
                continue
            assert lease['stage'] == 2 and lease['attempt'] == 2
            assert _complete_accepted(store, lease['run_id'], hash_lease_token(token), 'recovery')['status'] == 'accepted'
            executed.append(store.get_run(lease['run_id']))
        assert len(executed) == 38, (lease, [(r['kind'],r['stage'],r['status'],r.get('scoring_generation_floor'),r.get('result_doc') is not None) for r in store.list_runs(row['round_id'])])
        work = items[1:] + [{'scored_run_id': r['run_id'], 'submission_id': r['submission_id'],
                            'icp_position': r['icp_position'], 'output_ref': r['output_ref'],
                            'company_judgment_refs': []} for r in executed]
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SELECT public.lab_arena_close_stage(%s,2::smallint)', (row['round_id'],))
            cur.execute('SELECT public.lab_arena_open_scoring_v3(%s,2::smallint,%s::jsonb)', (row['round_id'], json.dumps(work)))
            assert cur.fetchone()[0]['assignments'] == 39
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE round_id=%s AND status='pending'", (row['round_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
            cur.execute('SELECT public.lab_arena__all_judgments_ready_v1(%s)', (row['round_id'],))
            assert cur.fetchone() == (False,)
        for _ in range(39):
            lease, token, _, _ = base.company.original.claim(store, row['round_id'], runner, excluded=[runner])
            assert lease['status'] == 'leased' and lease['kind'] == 'score' and lease['stage'] == 2, lease
            if lease['scored_run_id'] == items[1]['scored_run_id']:
                assert base.company.complete_subset(store, lease, token, items[1]['company_judgment_refs'])['status'] == 'accepted'
            else:
                assert store.complete_attempt(run_id=lease['run_id'], lease_token_hash=hash_lease_token(token),
                    result={'terminal_status': 'accepted'}, terminal_cause='accepted', output_ref='judge/' + lease['run_id'])['status'] == 'accepted'
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SELECT public.lab_arena__all_judgments_ready_v1(%s)', (row['round_id'],))
            assert cur.fetchone() == (False,)  # Projection aggregation remains mandatory.
            cur.execute("SELECT count(*) FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' AND status='accepted' AND scoring_generation_floor=%s", (row['round_id'],result['generation_floor']))
            assert cur.fetchone() == (40,)
    finally:
        store.close()


def test_recovery_rejudges_saved_outputs_through_aggregate_and_publication(database,tmp_path,monkeypatch):
    from dataclasses import replace
    from lab_arena import scoring
    from tests.lab_arena.test_integrity_round import IntegrityHarness
    from tests.lab_arena import test_lab_arena_service_round as flows
    from tests.lab_arena.baseline_scored_first_postgres_test import _baseline_first_judge,_enable_verified_proxy_runtime
    from tests.lab_arena.icp_fixtures import daily_icps
    with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('TRUNCATE public.lab_arena_rounds,public.lab_arena_ledger,public.lab_arena_trajectory_events RESTART IDENTITY CASCADE')
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,pause_reason='',actor_ref='' WHERE singleton")
        cur.execute('SET LOCAL session_replication_role=origin')
    harness=IntegrityHarness(lambda:database[0].connect(**database[1]),tmp_path,challengers=['Control'],runners=['alpha'])
    harness.service.config.defaults=replace(harness.service.config.defaults,
        execution_sequence_from='2000-01-01T00:00:00Z',company_judgment_from='2000-01-01T00:00:00Z',
        per_icp_cost_policy=True,rewards_enabled=False,benchmark_icp_count=2)
    harness.service.config.daily_icp_source=lambda **kwargs:{'status':'ready','set_id':kwargs['set_id'],'icps':daily_icps()[:2]}
    _enable_verified_proxy_runtime(harness)
    monkeypatch.setattr(flows,'deterministic_scorer',_baseline_first_judge)
    monkeypatch.setitem(scoring.DEFAULT_JUDGE_MODELS,'role_batch_check','google/gemini-2.5-flash')
    original_prices=flows.price_table
    monkeypatch.setattr(flows,'price_table',lambda:original_prices((*flows.PRICED_MODELS,'google/gemini-2.5-flash')))
    # Empty saved-output controls exercise the entire real input/claim/output/
    # score/publication path without inventing external evidence or paid calls.
    harness.sandbox.empty_outputs.update((flavor,position) for flavor in ('Control','PublicBaseline') for position in (0,1))
    try:
        assert flows._start_round(harness,day=22,epoch=62517)==2
        harness.clock.advance_to(harness.schedule()['stage_1_start'])
        harness.advance_until('scored',runners=1)
        row=harness.service.store.get_round(harness.round_id)
        old_judges=harness.service.store.list_runs(harness.round_id,kind='score')
        assert len(old_judges)==4 and all(r['status']=='accepted' for r in old_judges)
        saved_executions=harness.service.store.list_runs(harness.round_id,kind='execute')
        new_row=deepcopy(row)
        new_row['configuration_doc']['scorer_image_digest']='sha256:'+'9'*64
        new_row['configuration_doc']['scorer_image_reference']=flows.SCORER_IMAGE_REFERENCE.rsplit('@',1)[0]+'@'+new_row['configuration_doc']['scorer_image_digest']
        now=datetime.now(timezone.utc)
        for key,hours,seconds in [('stage_2_close',2,0),('final_scoring_close',4,0),('publication_deadline',4,1)]:
            # The protected preparer uses ISO8601 UTC offsets, not only Z.
            new_row['configuration_doc']['schedule'][key]=(now+timedelta(hours=hours,seconds=seconds)).isoformat()
        items=[]
        for stage in (1,2):
            items.extend(harness.service._scoring_work_items(new_row,stage,row['stage%d_scoring_plan_doc'%stage]))
        assert len(items)==4 and all(item['company_judgment_refs']==[] for item in items)
        with database[0].connect(**database[1]) as conn,conn.cursor() as cur:
            cur.execute('SELECT guard_generation FROM public.lab_arena_restart_claim_control WHERE singleton');generation=cur.fetchone()[0]
            assert base.operator_hold(cur,row,generation,True)['status']=='held'
            revised=invoke(cur,row,new_row['configuration_doc'],items)
            cur.execute('SELECT to_jsonb(a) FROM public.lab_arena_rounds a WHERE round_id=%s',(row['round_id'],));fresh=cur.fetchone()[0]
            assert base.operator_hold(cur,fresh,generation,False)['status']=='released'
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE round_id=%s AND status='pending'",(row['round_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
        ordinary_open = harness.service.store.open_scoring
        def open_after_backoff(*args, **kwargs):
            result = ordinary_open(*args, **kwargs)
            with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_runs SET created_at=clock_timestamp()-interval '20 minutes' WHERE round_id=%s AND status='pending'", (row['round_id'],))
                cur.execute('SET LOCAL session_replication_role=origin')
            return result
        monkeypatch.setattr(harness.service.store, 'open_scoring', open_after_backoff)
        monkeypatch.setattr(flows,'SCORER_IMAGE_DIGEST',new_row['configuration_doc']['scorer_image_digest'])
        monkeypatch.setattr(flows,'SCORER_IMAGE_REFERENCE',new_row['configuration_doc']['scorer_image_reference'])
        harness.clock.now=now+timedelta(hours=2,seconds=1)
        assert harness.service.advance_round(harness.round_id)['status'] == 'ok'
        assert harness.status() == 'stage2_closed'
        harness.advance_until('published',runners=1)
        final=harness.service.store.get_round(harness.round_id)
        assert final['status']=='published' and final['scoring_generation_floor']==revised['generation_floor']
        judges=harness.service.store.list_runs(harness.round_id,kind='score')
        assert [r for r in judges if r['attempt']==1]==old_judges
        assert len([r for r in judges if r['stage_generation']>=revised['generation_floor'] and r['status']=='accepted'])==4
        executions=harness.service.store.list_runs(harness.round_id,kind='execute')
        assert len(executions)==len(saved_executions)==4
        assert all(r['per_icp_score'] is not None and r['qualification_doc'] is not None for r in executions)
        assert [(r['run_id'],r['output_ref']) for r in executions]==[(r['run_id'],r['output_ref']) for r in saved_executions]
        for participant in final['participants']:
            public=harness.service.public_results(harness.round_id,participant['submission_id'])
            assert public['submission_scores']['final'] is not None
    finally:harness.service.store.close()

