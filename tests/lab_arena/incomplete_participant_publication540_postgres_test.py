"""Publication isolates unmeasured entrants under the real database guards."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

import pytest

from lab_arena import contracts, scoring
from tests.lab_arena import held_recovery_admission539_postgres_test as prior
from tests.lab_arena import test_lab_arena_service_round as flows
from tests.lab_arena.baseline_scored_first_postgres_test import _baseline_first_judge, _enable_verified_proxy_runtime
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.test_integrity_round import IntegrityHarness
from tests.lab_arena.lab_arena_pg_harness import current_service_database

_BASE_PRICE_TABLE = flows.price_table

SQL = Path(__file__).parents[2] / 'scripts/540-lab-arena-incomplete-participant-publication.sql'
REASON = 'public.lab_arena__participant_incomplete_reason_v1(text,text)'


@pytest.fixture(scope='module')
def database():
    gen = prior.database.__wrapped__()
    db = next(gen)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute(SQL.read_text())
            cur.execute(SQL.read_text())
            # Current public dashboard reads require this additive computed
            # projection. Keep the 540 transition/preimage checks unchanged.
            cur.execute((SQL.parent / '541-lab-arena-public-execution-failure-reason.sql').read_text())
        yield db
    finally:
        gen.close()


@pytest.fixture(scope='module')
def completed(database, tmp_path_factory):
    pg, dsn = database
    harness = IntegrityHarness(lambda: pg.connect(**dsn), tmp_path_factory.mktemp('isolation540'),
        challengers=['Complete', 'MissingJudge', 'HistoricalRoute', 'FailedRoute'], runners=['alpha'])
    harness.service.config.defaults = replace(harness.service.config.defaults,
        execution_sequence_from='2000-01-01T00:00:00Z', company_judgment_from='2000-01-01T00:00:00Z',
        per_icp_cost_policy=True, rewards_enabled=False, benchmark_icp_count=2)
    harness.service.config.daily_icp_source = lambda **kwargs: {'status': 'ready', 'set_id': kwargs['set_id'], 'icps': daily_icps()[:2]}
    _enable_verified_proxy_runtime(harness)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(flows, 'deterministic_scorer', _baseline_first_judge)
    monkey.setitem(scoring.DEFAULT_JUDGE_MODELS, 'role_batch_check', 'google/gemini-2.5-flash')
    prices = _BASE_PRICE_TABLE
    monkey.setattr(flows, 'price_table', lambda: prices((*flows.PRICED_MODELS, 'google/gemini-2.5-flash')))
    harness.sandbox.empty_outputs.update((flavor, position)
        for flavor in (*harness.challengers, 'PublicBaseline') for position in (0, 1))
    try:
        assert flows._start_round(harness, day=22, epoch=62540) == 5
        harness.clock.advance_to(harness.schedule()['stage_1_start'])
        harness.advance_until('scored', runners=1)
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s', (harness.round_id,))
            row = cur.fetchone()[0]
            cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_runs r WHERE round_id=%s ORDER BY run_id', (harness.round_id,))
            runs = [item[0] for item in cur.fetchall()]
        yield harness, row, runs
    finally:
        monkey.undo()
        harness.service.store.close()


@pytest.fixture(scope='module')
def current_application_database():
    yield from current_service_database()


@pytest.fixture(scope='module')
def current_application_completed(current_application_database, tmp_path_factory):
    # The dashboard/history checks use the reader introduced after 540.
    # Reuse the same genuine completion setup on current SQL, without adding
    # future functions to the independent historical migration controls.
    yield from completed.__wrapped__(current_application_database, tmp_path_factory)


def reset(database, completed, *, deadline=-1):
    harness, row, runs = completed
    pg, dsn = database
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('DELETE FROM public.lab_arena_ledger WHERE round_id=%s AND provider=%s', (row['round_id'], 'openrouter'))
        for table, key, docs in [('lab_arena_rounds', 'round_id', [row]), ('lab_arena_runs', 'run_id', runs)]:
            cur.execute('SELECT attname FROM pg_attribute WHERE attrelid=%s::regclass AND attgenerated<>%s', (table, ''))
            generated = {r[0] for r in cur.fetchall()}
            for doc in docs:
                names = [name for name in doc if name not in generated]
                cur.execute(f"UPDATE public.{table} SET ({','.join(names)})=(SELECT {','.join(names)} FROM jsonb_populate_record(NULL::public.{table},%s::jsonb)) WHERE {key}=%s", (json.dumps(doc), doc[key]))
        cutoff = datetime.now(timezone.utc) + timedelta(seconds=deadline)
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,publication_deadline}',to_jsonb(%s::text)) WHERE round_id=%s", (cutoff.isoformat(), row['round_id']))
        cur.execute('SET LOCAL session_replication_role=origin')
    harness.clock.now = cutoff if deadline <= 0 else cutoff - timedelta(seconds=1)
    return harness


def selected(completed, flavor, position=0):
    harness, _, runs = completed
    sid = next(sid for sid, current in harness.flavors.items() if current == flavor)
    return next(r for r in runs if r['submission_id'] == sid and r['kind'] == 'execute' and r['icp_position'] == position)


def native(cur, run, *, status=404, message='No endpoints found matching your data policy (Zero data retention).', call='a'):
    from base64 import b64encode
    response = dict(status=status, headers={}, call_succeeded=False,
        body_b64=b64encode(json.dumps({'error': {'code': status, 'message': message}}).encode()).decode())
    cur.execute('SET LOCAL session_replication_role=replica')
    for kind in ('reservation', 'dispatch', 'settlement'):
        cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd,entry_doc,terminal_response) SELECT %s,miner_hotkey,round_id,submission_id,run_id,stage,%s,'openrouter','openrouter.chat','miner_key',0,%s::jsonb,%s::jsonb FROM public.lab_arena_runs WHERE run_id=%s",
            (kind, 'sha256:' + call * 64, json.dumps({'action_sequence': 3, 'request_hash': 'sha256:' + 'd' * 64}), json.dumps(response) if kind == 'settlement' else None, run['run_id']))
    cur.execute('SET LOCAL session_replication_role=origin')


def gap(cur, run, cause='judge_error', *, execution=False):
    cur.execute('SET LOCAL session_replication_role=replica')
    cur.execute('UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL WHERE run_id=%s', (run['run_id'],))
    cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause=%s,output_ref=NULL WHERE scored_run_id=%s AND kind='score'", (cause, run['run_id']))
    if execution:
        cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause=%s,output_ref=NULL WHERE run_id=%s", (cause, run['run_id']))
    cur.execute('SET LOCAL session_replication_role=origin')


def state(cur, rid):
    cur.execute('SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) FROM public.lab_arena_runs r WHERE round_id=%s', (rid,))
    runs = cur.fetchone()[0]
    cur.execute('SELECT jsonb_agg(to_jsonb(l) ORDER BY entry_id) FROM public.lab_arena_ledger l WHERE round_id=%s', (rid,))
    return runs, cur.fetchone()[0]


def reasons(cur, rid):
    cur.execute("SELECT p->>'submission_id',public.lab_arena__participant_incomplete_reason_v1(%s,p->>'submission_id') FROM public.lab_arena_rounds r,jsonb_array_elements(r.participants)p WHERE round_id=%s", (rid, rid))
    return dict(cur.fetchall())


def test_mixed_completed_and_unmeasured_publish_without_rewriting_history(database, completed):
    h = reset(database, completed)
    missing = selected(completed, 'MissingJudge')
    historic = selected(completed, 'HistoricalRoute')
    failed = selected(completed, 'FailedRoute')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        gap(cur, missing)
        native(cur, historic)
        gap(cur, failed, 'provider_error', execution=True)
        native(cur, failed, call='b')
        before = state(cur, h.round_id)
        proof = reasons(cur, h.round_id)
        assert proof[missing['submission_id']] == 'judge_error'
        assert proof[historic['submission_id']] == proof[failed['submission_id']] == 'openrouter_zdr_route_unavailable'
    result = h.service.publish(h.round_id)
    assert result['status'] == 'ok', result
    row = h.service.store.get_round(h.round_id)
    ranking = {r['submission_id']: r for r in row['publication_doc']['final_ranking']}
    assert len(ranking) == 5 and row['participants'] == completed[1]['participants']
    for run in (missing, historic, failed):
        r = ranking[run['submission_id']]
        assert r['final_score'] is None and r['cost_summary'] is None and r['eligible'] is False
        assert r['evaluation'] == 'not_evaluated' and r['evaluation_reason'] == proof[run['submission_id']]
    complete = selected(completed, 'Complete')
    assert ranking[complete['submission_id']]['final_score'] == 0.0
    assert row['publication_doc']['king_decision']['outcome'] == 'no_king'
    assert h.service.publish(h.round_id)['status'] == 'stale'
    assert h.service.activate_reward(h.round_id)['status'] == 'disabled'
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert state(cur, h.round_id) == before


@pytest.mark.parametrize('offset', [-1, 0, 1])
@pytest.mark.parametrize('active', [None, 'leased', 'submitted', 'reservation', 'dispatch'])
def test_publication_deadline_and_atomic_native_drain(database, completed, offset, active):
    h = reset(database, completed, deadline=60 if offset < 0 else -1)
    run = selected(completed, 'MissingJudge')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        gap(cur, run)
        if active in ('leased', 'submitted'):
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('UPDATE public.lab_arena_runs SET status=%s WHERE scored_run_id=%s', (active, run['run_id']))
            cur.execute('SET LOCAL session_replication_role=origin')
        elif active:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd) SELECT %s,miner_hotkey,round_id,submission_id,run_id,stage,%s,'openrouter','openrouter.chat','miner_key',0 FROM public.lab_arena_runs WHERE run_id=%s", (active, 'sha256:' + 'e' * 64, run['run_id']))
            cur.execute('SET LOCAL session_replication_role=origin')
        before = state(cur, h.round_id)
    result = h.service.publish(h.round_id)
    assert result['status'] == ('ok' if offset >= 0 and active is None else 'retry'), result
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert state(cur, h.round_id) == before


@pytest.mark.parametrize('fault', ['accepted_unmaterialized', 'wrong_round', 'wrong_source', 'bad_qualification', 'baseline_gap', 'invented_null'])
def test_completed_or_corrupt_accepted_evidence_cannot_be_omitted(database, completed, fault):
    h = reset(database, completed)
    run = selected(completed, 'MissingJudge')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        if fault == 'accepted_unmaterialized':
            cur.execute('UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL WHERE run_id=%s', (run['run_id'],))
        elif fault in ('wrong_round', 'wrong_source'):
            column = 'round_id' if fault == 'wrong_round' else 'scored_run_id'
            cur.execute(f'UPDATE public.lab_arena_runs SET {column}=%s WHERE scored_run_id=%s', ('invalid-authority', run['run_id']))
        elif fault == 'bad_qualification':
            cur.execute("UPDATE public.lab_arena_runs SET qualification_doc='{}'::jsonb WHERE run_id=%s", (run['run_id'],))
        elif fault == 'baseline_gap':
            baseline = next(r for r in completed[2] if r['kind'] == 'execute' and r['stage'] == 1)
            gap(cur, baseline)
        cur.execute('SET LOCAL session_replication_role=origin')
        assert reasons(cur, h.round_id)[run['submission_id']] is None
    if fault == 'accepted_unmaterialized':
        # The valid late result must materialize; it remains a completed row.
        assert h.service.publish(h.round_id)['status'] == 'ok'
        ranking = h.service.store.get_round(h.round_id)['publication_doc']['final_ranking']
        assert next(r for r in ranking if r['submission_id'] == run['submission_id'])['final_score'] == 0.0
    elif fault == 'baseline_gap':
        assert h.service.publish(h.round_id)['status'] == 'cancelled'
    elif fault == 'invented_null':
        actual = h.service.store.transition_round
        def corrupt(rid, expected, next_status, patch):
            patch = deepcopy(patch)
            entry = next(r for r in patch['publication_doc']['final_ranking'] if r['submission_id'] == run['submission_id'])
            entry.update(final_score=None, cost_summary=None, eligible=False, eligibility_reason='execution_incomplete', evaluation='not_evaluated', evaluation_reason='judge_error')
            return actual(rid, expected, next_status, patch)
        h.service.store.transition_round = corrupt
        try:
            with pytest.raises(Exception): h.service.publish(h.round_id)
        finally: h.service.store.transition_round = actual
    else:
        try: result = h.service.publish(h.round_id)
        except Exception: result = {'status': 'failed_closed'}
        assert result['status'] in ('failed_closed', 'cancelled', 'retry')
        assert h.status() != 'published'


@pytest.mark.parametrize('cause', ['judge_error', 'judge_timeout', 'worker_lost', 'provider_error', 'result_rejected', 'missing_judge'])
def test_stage_scoring_retains_complete_work_and_publishes_infrastructure_gaps(database, completed, cause):
    h = reset(database, completed)
    run = selected(completed, 'MissingJudge')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        gap(cur, run, cause if cause != 'missing_judge' else 'stage_closed')
        cur.execute('SET LOCAL session_replication_role=replica')
        if cause == 'missing_judge':
            cur.execute("UPDATE public.lab_arena_runs SET round_id='missing-judge-fixture' WHERE scored_run_id=%s", (run['run_id'],))
        cur.execute("UPDATE public.lab_arena_rounds SET status='stage2_judged' WHERE round_id=%s", (h.round_id,))
        cur.execute('SET LOCAL session_replication_role=origin')
    assert h.service.score_stage(h.round_id, 2)['status'] == 'ok'
    assert h.service.publish(h.round_id)['status'] == 'ok'
    row = h.service.store.get_round(h.round_id)
    final = next(r for r in row['publication_doc']['final_ranking'] if r['submission_id'] == run['submission_id'])
    assert final['final_score'] is None
    assert final['evaluation_reason'] == ('judge_missing' if cause == 'missing_judge' else cause)


@pytest.mark.parametrize('kind', ['accepted_empty_unjudged', 'missing_execution', 'malformed_failed_native', 'generic_exhausted_provider'])
def test_unmeasured_edge_cases_reach_real_publication(current_application_database, current_application_completed, kind):
    database, completed = current_application_database, current_application_completed
    h = reset(database, completed)
    run = selected(completed, 'FailedRoute')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        if kind == 'accepted_empty_unjudged':
            gap(cur, run)
            native(cur, run)
        elif kind == 'missing_execution':
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_runs SET round_id='missing-execution-fixture' WHERE run_id=%s OR scored_run_id=%s", (run['run_id'], run['run_id']))
            cur.execute('SET LOCAL session_replication_role=origin')
        else:
            gap(cur, run, 'provider_error', execution=True)
            if kind == 'malformed_failed_native':
                native(cur, run)
                cur.execute('SET LOCAL session_replication_role=replica')
                cur.execute("UPDATE public.lab_arena_ledger SET terminal_response=jsonb_set(terminal_response,'{body_b64}','\"bm90LWpzb24=\"') WHERE run_id=%s AND entry_kind='settlement' AND provider='openrouter'", (run['run_id'],))
                cur.execute('SET LOCAL session_replication_role=origin')
        proof = reasons(cur, h.round_id)[run['submission_id']]
        assert proof == ('openrouter_zdr_route_unavailable' if kind == 'accepted_empty_unjudged' else 'execution_missing' if kind == 'missing_execution' else 'provider_error')
    assert h.service.publish(h.round_id)['status'] == 'ok'
    from lab_arena.public_dashboard import submissions_snapshot, history_snapshot
    public = submissions_snapshot(h.service, h.round_id)
    final = next(r for r in public['submissions'] if r['submission_id'] == run['submission_id'])
    assert final['final_score'] is None and final['score_status'] == 'not_evaluated'
    assert final['evaluation']['state'] in ('provider_blocked', 'evaluation_incomplete')
    assert final['evaluation_reason'] == proof
    assert 'terminal_response' not in json.dumps(final)
    history = history_snapshot(h.service)
    item = next(r for r in history['submissions'] if r['submission_id'] == run['submission_id'])
    assert item['score_status'] == 'not_evaluated' and item['evaluation']['state'] == final['status']


@pytest.mark.parametrize('key,value', [('publication_deadline', None), ('publication_deadline', '__missing__'),
    ('sourcing_cost_eligibility_policy', None), ('sourcing_cost_eligibility_policy', '__missing__'),
    ('sourcing_cost_eligibility_policy', 'unsupported')])
def test_missing_deadline_or_cost_policy_never_authorizes_sql_omission(database, completed, key, value):
    h = reset(database, completed)
    run = selected(completed, 'MissingJudge')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        gap(cur, run)
        cur.execute('SET LOCAL session_replication_role=replica')
        path = ['schedule', key] if key == 'publication_deadline' else [key]
        if value == '__missing__':
            cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=configuration_doc#-%s WHERE round_id=%s', (path, h.round_id))
        else:
            cur.execute('UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,%s,%s::jsonb) WHERE round_id=%s', (path, json.dumps(value), h.round_id))
        cur.execute('SET LOCAL session_replication_role=origin')
        assert reasons(cur, h.round_id)[run['submission_id']] is None


def test_migration_rejects_existing_function_acl_drift(database):
    signatures = ['public.lab_arena_transition_round(text,text,text,jsonb)',
        'public.lab_arena__publication_execution_incomplete_v1(text,text)',
        'public.lab_arena__all_judgments_ready_v1(text)',
        'public.lab_arena_record_run_scores(text,smallint,jsonb)']
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        for signature in signatures:
            cur.execute('GRANT EXECUTE ON FUNCTION ' + signature + ' TO anon')
            conn.commit()
            try:
                with pytest.raises(database[0].Error): cur.execute(SQL.read_text())
            finally:
                conn.rollback()
                cur.execute('REVOKE EXECUTE ON FUNCTION ' + signature + ' FROM anon')
                conn.commit()


def test_positive_ten_icp_large_cohort_publishes_promotes_and_derives_rewards(database, tmp_path, monkeypatch):
    """Real execution/judging proves the winner; cloned rows supply bounded load."""
    from lab_arena.promotion import GitPromoter
    from leadpoet_canonical import arena_weights
    from tests.lab_arena.baseline_cost_zero_round_test import _accepted_state
    pg, dsn = database
    h = IntegrityHarness(lambda: pg.connect(**dsn), tmp_path,
        challengers=['PositiveWinner', 'HigherPartial'], runners=['alpha', 'beta'])
    h.service.config.defaults = replace(h.service.config.defaults,
        execution_sequence_from='2000-01-01T00:00:00Z', company_judgment_from='2000-01-01T00:00:00Z',
        per_icp_cost_policy=True, rewards_enabled=True, benchmark_icp_count=10)
    h.service.config.daily_icp_source = lambda **kw: {'status': 'ready', 'set_id': kw['set_id'], 'icps': daily_icps()[:10]}
    _enable_verified_proxy_runtime(h)
    def judge(companies, icp, reference):
        from tests.test_arena_score_integrity import _positive_breakdown
        from qualification.scoring.competition import apply_company_judgment_context
        from urllib.parse import urlsplit
        judgments = []
        for company in companies:
            name = company['company_name']
            value = 20.0 if name.startswith('PublicBaseline') else 95.0 if name.startswith('HigherPartial') else 80.0
            slug = urlsplit(company.get('company_linkedin') or '').path.strip('/').split('/')[-1]
            raw = _positive_breakdown(name, urlsplit(company['company_website']).hostname, slug)
            raw.update(final_score=value, intent_signal_raw=value, intent_signal_final=value)
            raw['intent_signals_detail'][0].update(raw=value, after_decay=value)
            raw['verifier_gate_receipts'].append({'gate': 'intent_details', 'decision': 'match'})
            judgments.append(raw)
        return apply_company_judgment_context(companies, judgments, contacts_required=False, company_quality=False)
    # Use the actual current company-cache scoring protocol with controlled
    # grading decisions; the ordinary legacy fake omits company_judgments.
    from lab_arena import runtime
    judge.integrity_policy = True
    judge.company_quality = False
    judge.contacts_required = False
    original_runtime = h.sandbox.run_icp
    def current_protocol(spec, **kwargs):
        document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        if document.get("schema_version") != scoring.SCORING_INPUT_SCHEMA_VERSION:
            return original_runtime(spec, **kwargs)
        full, new = scoring.score_quality_work_item(
            {"scored_run_id": document["scored_run_id"]},
            icp=document["icp"], companies=document["companies"], scorer=judge,
            cache_context=document["company_judgment_cache"],
            max_scored_companies=int(document["scorer_policy"]["max_scored_companies"]))
        output = scoring.build_scoring_output(document["scored_run_id"], full, company_judgments=new)
        return runtime.fake_result(exit_code=0, output_bytes=json.dumps(output).encode())
    h.sandbox.run_icp = current_protocol
    monkeypatch.setattr(flows, 'deterministic_scorer', judge)
    monkeypatch.setitem(scoring.DEFAULT_JUDGE_MODELS, 'role_batch_check', 'google/gemini-2.5-flash')
    prices = _BASE_PRICE_TABLE
    monkeypatch.setattr(flows, 'price_table', lambda: prices((*flows.PRICED_MODELS, 'google/gemini-2.5-flash')))
    try:
        assert flows._start_round(h, day=23, epoch=62541) == 3
        h.clock.advance_to(h.schedule()['stage_1_start'])
        h.advance_until('scored', runners=2, max_steps=120)
        row = h.service.store.get_round(h.round_id)
        runs = h.service.store.list_runs(h.round_id)
        winner = next(sid for sid, flavor in h.flavors.items() if flavor == 'PositiveWinner')
        partial = next(sid for sid, flavor in h.flavors.items() if flavor == 'HigherPartial')
        gap_run = next(r for r in runs if r['submission_id'] == partial and r['kind'] == 'execute' and r['icp_position'] == 9)
        before_scores = h.service._score_entries_from_runs(row, range(10), 'final_score')
        assert next(r['final_score'] for r in before_scores if r['submission_id'] == partial) > next(r['final_score'] for r in before_scores if r['submission_id'] == winner), [(r['submission_id'], r.get('per_icp_score'), r.get('qualification_doc')) for r in runs if r['kind']=='execute' and r['icp_position']==0]
        late_run = next(r for r in runs if r['submission_id'] == winner and r['kind'] == 'execute' and r['icp_position'] == 0)
        late_judge = next(r for r in runs if r.get('scored_run_id') == late_run['run_id'] and r['status'] == 'accepted')
        assert len(late_judge['company_judgment_refs']) == 5
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=1)
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            gap(cur, gap_run)
            cur.execute('SET LOCAL session_replication_role=replica')
            participants = deepcopy(row['participants'])
            parent = next(p for p in participants if p['submission_id'] == winner)
            # Performance-only aliases retain the same already-validated company
            # scopes, like legitimate cross-miner cache hits. No new paid work.
            for index in range(22):
                sid = 'zzload540-' + str(index)
                hotkey = flows.keypair('load540-' + str(index)).ss58_address
                item = dict(parent, submission_id=sid, miner_hotkey=hotkey)
                participants.append(item)
                cur.execute("INSERT INTO public.lab_arena_submissions SELECT (jsonb_populate_record(NULL::public.lab_arena_submissions,to_jsonb(s)||%s::jsonb)).* FROM public.lab_arena_submissions s WHERE submission_id=%s", (json.dumps({'submission_id': sid, 'miner_hotkey': hotkey}), winner))
                for old in (r for r in runs if r['submission_id'] == winner):
                    patch = {k: str(old[k]).replace(winner, sid) for k in ('run_id', 'assignment_id')}
                    if old.get('scored_run_id'): patch['scored_run_id'] = old['scored_run_id'].replace(winner, sid)
                    patch.update(submission_id=sid, miner_hotkey=hotkey, claim_request_id=None)
                    if old['kind'] == 'score':
                        output = json.loads(h.objects.get_bounded(old['output_ref'], scoring.MAX_SCORING_OUTPUT_BYTES))
                        output['scored_run_id'] = patch['scored_run_id']
                        patch['output_ref'] = old['output_ref'].replace(winner, sid)
                        h.objects.put(patch['output_ref'], contracts.canonical_json(output).encode())
                    cur.execute("INSERT INTO public.lab_arena_runs SELECT (jsonb_populate_record(NULL::public.lab_arena_runs,to_jsonb(r)||%s::jsonb)).* FROM public.lab_arena_runs r WHERE run_id=%s", (json.dumps(patch), old['run_id']))
            finalists = sorted(p['submission_id'] for p in participants if not p.get('is_king'))
            cur.execute("UPDATE public.lab_arena_rounds SET participants=%s::jsonb,finalists=%s::jsonb,configuration_doc=jsonb_set(configuration_doc,'{schedule,publication_deadline}',to_jsonb(%s::text)) WHERE round_id=%s", (json.dumps(participants), json.dumps(finalists), cutoff.isoformat(), h.round_id))
            cur.execute('UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL WHERE run_id=%s', (late_run['run_id'],))
            cur.execute('SET LOCAL session_replication_role=origin')
            # Real nonempty current-company refs must reject altered scopes,
            # a foreign policy generation, and legacy whole-item authority.
            record = dict(run_id=late_run['run_id'], per_icp_score=late_run['per_icp_score'], qualification_doc=late_run['qualification_doc'])
            for fault in ('wrong_refs', 'foreign_floor', 'whole_scope'):
                cur.execute('SAVEPOINT late_guard540')
                cur.execute('SET LOCAL session_replication_role=replica')
                if fault == 'wrong_refs':
                    cur.execute("UPDATE public.lab_arena_runs SET company_judgment_refs=jsonb_set(company_judgment_refs,'{0,scope_doc,scorer_image_digest}','\"wrong\"') WHERE run_id=%s", (late_judge['run_id'],))
                elif fault == 'foreign_floor':
                    cur.execute('UPDATE public.lab_arena_runs SET scoring_generation_floor=scoring_generation_floor+1 WHERE run_id=%s', (late_judge['run_id'],))
                else:
                    cur.execute("UPDATE public.lab_arena_runs SET judgment_scope_doc='{}'::jsonb WHERE run_id=%s", (late_judge['run_id'],))
                cur.execute('SET LOCAL session_replication_role=origin')
                with pytest.raises(pg.Error):
                    cur.execute('SELECT public.lab_arena_record_run_scores(%s,%s::smallint,%s::jsonb)', (h.round_id, 2, json.dumps([record])))
                cur.execute('ROLLBACK TO SAVEPOINT late_guard540')
            before = state(cur, h.round_id)
        h.clock.now = datetime.fromisoformat(row['configuration_doc']['schedule']['publication_deadline'].replace('Z', '+00:00'))
        started = time.monotonic()
        result = h.service.publish(h.round_id)
        elapsed = time.monotonic() - started
        assert result['status'] == 'ok', result
        assert elapsed < 30, elapsed
        print('25 participants x 10 ICP publication seconds:', round(elapsed, 3))
        published = h.service.store.get_round(h.round_id)
        ranking = {r['submission_id']: r for r in published['publication_doc']['final_ranking']}
        assert len(ranking) == 25 and ranking[partial]['final_score'] is None
        assert ranking[winner]['final_score'] > next(r['final_score'] for r in ranking.values() if r['is_baseline'])
        decision = published['publication_doc']['king_decision']
        assert decision['outcome'] == 'crowned'
        actual_winner = decision['winner_submission_id']
        assert actual_winner != partial and ranking[actual_winner]['eligible']
        assert h.service.activate_reward(h.round_id)['status'] == 'waiting_for_promotion'
        repo = tmp_path / 'promotion-repository'; repo.mkdir()
        remote = flows.promotion_repository(repo)
        h.service.config.baseline_promoter_factory = lambda: GitPromoter(str(remote), tmp_path / 'promotion-cache')
        assert h.service.promote_pending_baselines()['status'] == 'ok'
        assert h.service.activate_reward(h.round_id)['status'] == 'activated'
        activated = h.service.store.get_round(h.round_id)
        assert activated['baseline_promoted_at'] is not None
        assert activated['reward_basis_doc']['reward_slots'][0]['submission_id'] == actual_winner
        burn = flows.keypair('isolated-participant-burn540').ss58_address
        accepted = _accepted_state(h, int(activated['effective_reward_epoch']), burn)
        weights = arena_weights.derive_arena_weights(accepted, [burn, decision['king_hotkey']])
        assert weights['champion_share_ppb'] > 0
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            after = state(cur, h.round_id)
            assert after[1] == before[1]
            old_rows = {r['run_id']: r for r in before[0]}
            new_rows = {r['run_id']: r for r in after[0]}
            assert new_rows[late_run['run_id']]['per_icp_score'] == late_run['per_icp_score']
            assert new_rows[late_run['run_id']]['qualification_doc'] == late_run['qualification_doc']
            for doc in (old_rows[late_run['run_id']], new_rows[late_run['run_id']]):
                for field in ('per_icp_score', 'qualification_doc', 'updated_at'): doc.pop(field, None)
            assert new_rows == old_rows
        assert h.service.activate_reward(h.round_id)['status'] == 'existing'
    finally:
        h.service.store.close()


@pytest.mark.parametrize('code,message', [
    (404.0, 'No endpoints found matching your data policy (Zero data retention).'),
    ('__raw_exponent__', 'No endpoints found matching your data policy (Zero data retention).'),
    ('404', 'No endpoints found matching your data policy (Zero data retention).'),
    (True, 'No endpoints found matching your data policy (Zero data retention).'),
    (401, 'Invalid API key'), (402, 'Insufficient credits'),
    (404, 'Requested model does not exist'),
    (404, 'Quoted user input: No endpoints found matching your data policy (Zero data retention).'),
])
def test_sql_native_classifier_never_overrides_real_completed_zero_for_other_errors(database, completed, code, message):
    from base64 import b64encode
    h = reset(database, completed)
    run = selected(completed, 'HistoricalRoute')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        native(cur, run)
        cur.execute('SET LOCAL session_replication_role=replica')
        body_text = json.dumps({'error': {'code': code, 'message': message}})
        if code == '__raw_exponent__': body_text = body_text.replace('"__raw_exponent__"', '4.04e2')
        body = b64encode(body_text.encode()).decode()
        cur.execute("UPDATE public.lab_arena_ledger SET terminal_response=jsonb_set(terminal_response,'{body_b64}',to_jsonb(%s::text)) WHERE run_id=%s AND entry_kind='settlement' AND provider='openrouter'", (body, run['run_id']))
        cur.execute('SET LOCAL session_replication_role=origin')
        cur.execute('SELECT public.lab_arena__execution_privacy_route_unavailable_v1(%s)', (run['run_id'],))
        assert cur.fetchone() == (False,)
        assert reasons(cur, h.round_id)[run['submission_id']] is None
    assert h.service.publish(h.round_id)['status'] == 'ok'
    row = h.service.store.get_round(h.round_id)
    assert next(r['final_score'] for r in row['publication_doc']['final_ranking'] if r['submission_id'] == run['submission_id']) == 0.0


@pytest.mark.parametrize('deadline_delta,expected', [(-1, 'judge_error'), (0, 'judge_error'), (1, None)])
def test_sql_proof_exact_transaction_deadline_boundary(database, completed, deadline_delta, expected):
    h = reset(database, completed)
    run = selected(completed, 'MissingJudge')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        gap(cur, run)
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{schedule,publication_deadline}',to_jsonb((now()+make_interval(secs=>%s))::text)) WHERE round_id=%s", (deadline_delta, h.round_id))
        cur.execute('SET LOCAL session_replication_role=origin')
        assert reasons(cur, h.round_id)[run['submission_id']] == expected


def test_migration_replay_preserves_representative_published_history(database, completed):
    h = reset(database, completed)
    run = selected(completed, 'MissingJudge')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        gap(cur, run)
    assert h.service.publish(h.round_id)['status'] == 'ok'
    before_row = h.service.store.get_round(h.round_id)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        before = state(cur, h.round_id)
        cur.execute(SQL.read_text())
        cur.execute(SQL.read_text())
        assert state(cur, h.round_id) == before
    assert h.service.store.get_round(h.round_id) == before_row


@pytest.mark.parametrize('fault', ['unknown_reason', 'wrong_status', 'duplicate', 'legacy_untyped'])
def test_published_proof_rejects_untrusted_historical_omission_metadata(database, completed, fault):
    h = reset(database, completed)
    run = selected(completed, 'MissingJudge')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        gap(cur, run)
    assert h.service.publish(h.round_id)['status'] == 'ok'
    row = h.service.store.get_round(h.round_id)
    document = deepcopy(row['publication_doc'])
    item = next(r for r in document['final_ranking'] if r['submission_id'] == run['submission_id'])
    if fault == 'unknown_reason': item['evaluation_reason'] = 'untrusted_omission'
    elif fault == 'wrong_status': item['evaluation_status'] = 'provider_blocked'
    elif fault == 'duplicate': document['final_ranking'].append(deepcopy(item))
    else:
        for key in ('evaluation', 'evaluation_status', 'evaluation_reason'): item.pop(key)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert reasons(cur, h.round_id)[run['submission_id']] == 'judge_error'
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('UPDATE public.lab_arena_rounds SET publication_doc=%s::jsonb WHERE round_id=%s', (json.dumps(document), h.round_id))
        cur.execute('SET LOCAL session_replication_role=origin')
        assert reasons(cur, h.round_id)[run['submission_id']] is None
