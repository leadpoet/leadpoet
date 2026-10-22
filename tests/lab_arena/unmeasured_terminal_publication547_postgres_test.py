"""Unmeasured terminal failures cannot become inferred model zeros."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from lab_arena import contracts, public_dashboard
from tests.lab_arena import incomplete_participant_publication540_postgres_test as prior

SQL = Path(__file__).parents[2] / 'scripts/547-lab-arena-unmeasured-terminal-publication.sql'


@pytest.fixture(scope='module')
def database():
    gen = prior.database.__wrapped__()
    db = next(gen)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute(SQL.read_text())
            cur.execute(SQL.read_text())
        yield db
    finally:
        gen.close()


@pytest.fixture(scope='module')
def completed(database, tmp_path_factory):
    gen = prior.completed.__wrapped__(database, tmp_path_factory)
    try:
        yield next(gen)
    finally:
        gen.close()


def failures(database, completed, causes, *, materialized=False, history=False, route=False, deadline=-1):
    h = prior.reset(database, completed, deadline=deadline)
    sid = prior.selected(completed, 'FailedRoute')['submission_id']
    originals = sorted((r for r in completed[2] if r['submission_id'] == sid and r['kind'] == 'execute'), key=lambda r: r['icp_position'])
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        # Only construct this test's terminal dataset. All publication checks run
        # with the normal triggers and native functions enabled below.
        cur.execute("DELETE FROM public.lab_arena_runs WHERE submission_id=%s AND kind='execute'AND attempt>1", (sid,))
        for source, cause in zip(originals, causes):
            cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause='judge_error',output_ref=NULL WHERE scored_run_id=%s AND kind='score'", (source['run_id'],))
            doc = deepcopy(source)
            result = deepcopy(source['result_doc'])
            result['terminal_status'] = 'model_error' if history else cause
            doc.update(status='failed', terminal_cause=result['terminal_status'], output_ref=None,
                per_icp_score=0 if materialized or history else None,
                qualification_doc={'companies': []} if materialized or history else None, result_doc=result)
            cur.execute("UPDATE public.lab_arena_runs SET status='failed',terminal_cause=%s,output_ref=NULL,per_icp_score=%s,qualification_doc=%s::jsonb,result_doc=%s::jsonb WHERE run_id=%s",
                (doc['terminal_cause'], doc['per_icp_score'], json.dumps(doc['qualification_doc']) if doc['qualification_doc'] is not None else None, json.dumps(result), source['run_id']))
            if history:
                doc.update(run_id=source['assignment_id'] + ':3', attempt=3,
                    terminal_cause=cause, per_icp_score=None, qualification_doc=None)
                doc['result_doc']['terminal_status'] = cause
                doc['claim_request_id'] = str(doc.get('claim_request_id') or 'fixture') + '-fresh'
                cur.execute('INSERT INTO public.lab_arena_runs SELECT(jsonb_populate_record(NULL::public.lab_arena_runs,%s::jsonb)).*', (json.dumps(doc),))
        cur.execute('SET LOCAL session_replication_role=origin')
        cur.execute("SELECT to_jsonb(a)FROM public.lab_arena_runs a WHERE submission_id=%s AND kind='execute'AND attempt=%s ORDER BY icp_position", (sid, 3 if history else 1))
        selected = [x[0] for x in cur.fetchall()]
        if route:
            for run in selected:
                if run['terminal_cause'] == 'provider_error':
                    prior.native(cur, run, call=str(run['icp_position'] + 6))
        before = prior.state(cur, h.round_id)
    h.service._invalidate_hot_round()
    return h, sid, selected, before


@pytest.mark.parametrize('causes', [('model_error', 'provider_error'), ('provider_error', 'model_error')])
@pytest.mark.parametrize('route', [False, True])
def test_concrete_provider_reason_wins_in_both_position_orders(database, completed, causes, route):
    h, sid, _, before = failures(database, completed, causes, history=True, route=route)
    expected = 'openrouter_zdr_route_unavailable' if route else 'provider_error'
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert prior.reasons(cur, h.round_id)[sid] == expected
    result = h.service.publish(h.round_id)
    assert result['status'] == 'ok', result
    row = h.service.store.get_round(h.round_id)
    entry = next(x for x in row['publication_doc']['final_ranking'] if x['submission_id'] == sid)
    assert entry['evaluation_reason'] == expected and entry['final_score'] is None
    assert entry['cost_summary'] is None and entry['eligible'] is False
    assert entry['evaluation'] == 'not_evaluated'
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert prior.state(cur, h.round_id) == before
        assert prior.reasons(cur, h.round_id)[sid] == expected
    assert public_dashboard._cost_projection(entry)['score_status'] == 'not_evaluated'


@pytest.mark.parametrize('cause', contracts.MODEL_CAUSED_TERMINAL_CAUSES)
def test_typed_model_result_without_score_stays_null(database, completed, cause):
    h, sid, _, before = failures(database, completed, [cause, cause])
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert prior.reasons(cur, h.round_id)[sid] == 'execution_score_missing'
    assert h.service.publish(h.round_id)['status'] == 'ok'
    entry = next(x for x in h.service.store.get_round(h.round_id)['publication_doc']['final_ranking'] if x['submission_id'] == sid)
    assert entry['evaluation_reason'] == 'execution_score_missing' and entry['final_score'] is None
    assert public_dashboard._cost_projection(entry)['score_status'] == 'not_evaluated'
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert prior.state(cur, h.round_id) == before


def test_materialized_genuine_model_zero_is_not_omitted(database, completed):
    h, sid, _, before = failures(database, completed, ['model_error'], materialized=True)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert prior.reasons(cur, h.round_id)[sid] is None
    assert h.service.publish(h.round_id)['status'] == 'ok'
    entry = next(x for x in h.service.store.get_round(h.round_id)['publication_doc']['final_ranking'] if x['submission_id'] == sid)
    assert entry['final_score'] == 0 and 'evaluation' not in entry
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert prior.state(cur, h.round_id) == before


@pytest.mark.parametrize('fault', ['pending', 'wrong_terminal', 'missing_result', 'wrong_schema', 'negative_resource', 'float_integer', 'missing_resource', 'offset_time', 'extra_key', 'output_present', 'qualification_present', 'excessive_float', 'extra_resource_key'])
def test_malformed_or_nonterminal_result_cannot_authorize_omission(database, completed, fault):
    h, sid, runs, _ = failures(database, completed, ['model_error', 'model_error'])
    run = runs[0]
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        doc = deepcopy(run['result_doc'])
        if fault == 'pending':
            cur.execute("UPDATE public.lab_arena_runs SET status='pending'WHERE run_id=%s", (run['run_id'],))
        elif fault in ('output_present', 'qualification_present'):
            if fault == 'output_present':
                cur.execute("UPDATE public.lab_arena_runs SET output_ref='invalid-output'WHERE run_id=%s", (run['run_id'],))
            else:
                cur.execute("UPDATE public.lab_arena_runs SET qualification_doc=%s::jsonb WHERE run_id=%s", (json.dumps({"companies": []}), run['run_id']))
        elif fault == 'excessive_float':
            cur.execute("UPDATE public.lab_arena_runs SET result_doc=jsonb_set(result_doc,'{resource_summary,wall_seconds}','1e400'::jsonb)WHERE run_id=%s", (run['run_id'],))
        else:
            if fault == 'wrong_terminal': doc['terminal_status'] = 'provider_error'
            elif fault == 'missing_result': doc = None
            elif fault == 'wrong_schema': doc['schema_version'] = 'unknown'
            elif fault == 'negative_resource': doc['resource_summary']['cpu_seconds'] = -1
            elif fault == 'float_integer': doc['resource_summary']['provider_call_count'] = 0.0
            elif fault == 'missing_resource': doc['resource_summary'].pop('stdout_bytes')
            elif fault == 'offset_time': doc['started_at'] = '2026-10-22T00:00:00+00:00'
            elif fault == 'extra_key': doc['unknown'] = True
            elif fault == 'extra_resource_key': doc['resource_summary']['unknown'] = True
            cur.execute('UPDATE public.lab_arena_runs SET result_doc=%s::jsonb WHERE run_id=%s', (json.dumps(doc) if doc is not None else None, run['run_id']))
        cur.execute('SET LOCAL session_replication_role=origin')
        assert prior.reasons(cur, h.round_id)[sid] is None
    try:
        result = h.service.publish(h.round_id)
    except Exception:
        pass
    else:
        assert result['status'] != 'ok', result
    assert h.service.store.get_round(h.round_id)['publication_doc'] is None


@pytest.mark.parametrize('active', ['leased', 'submitted', 'reservation', 'dispatch'])
def test_unmeasured_omission_cannot_bypass_native_drain(database, completed, active):
    h, sid, runs, _ = failures(database, completed, ['model_error', 'provider_error'])
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        if active in ('leased', 'submitted'):
            cur.execute('UPDATE public.lab_arena_runs SET status=%s WHERE run_id=%s', (active, runs[0]['run_id']))
        else:
            cur.execute("INSERT INTO public.lab_arena_ledger(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,provider,operation_id,funding_source,amount_microusd)SELECT %s,miner_hotkey,round_id,submission_id,run_id,stage,%s,'openrouter','openrouter.chat','miner_key',0 FROM public.lab_arena_runs WHERE run_id=%s", (active, 'sha256:' + 'a' * 64, runs[1]['run_id']))
        cur.execute('SET LOCAL session_replication_role=origin')
    assert h.service.publish(h.round_id)['status'] == 'retry'
    assert h.service.store.get_round(h.round_id)['publication_doc'] is None


def test_missing_score_is_not_finalized_before_deadline(database, completed):
    h, _, _, before = failures(database, completed, ['model_error', 'model_error'], deadline=60)
    assert h.service.publish(h.round_id)['status'] == 'retry'
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert prior.state(cur, h.round_id) == before


def test_healthy_winner_promotion_and_reward_gates(database, tmp_path, monkeypatch):
    prior.test_positive_ten_icp_large_cohort_publishes_promotes_and_derives_rewards(database, tmp_path, monkeypatch)


@pytest.mark.parametrize('fault', ['wrong_round', 'wrong_source', 'bad_qualification', 'baseline_gap', 'invented_null'])
def test_existing_accepted_authority_and_baseline_guards(database, completed, fault):
    prior.test_completed_or_corrupt_accepted_evidence_cannot_be_omitted(database, completed, fault)
