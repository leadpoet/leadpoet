"""Public failure diagnostics reuse native proof without changing saved work."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import pytest

from lab_arena import public_dashboard
from tests.lab_arena import incomplete_participant_publication540_postgres_test as prior

SQL = Path(__file__).parents[2] / 'scripts/541-lab-arena-public-execution-failure-reason.sql'
SIGNATURE = 'public.lab_arena_public_execution_failure_reason_v1(public.lab_arena_runs)'
REASON = 'openrouter_zdr_route_unavailable'
completed = prior.completed


@pytest.fixture(scope='module')
def database():
    gen = prior.database.__wrapped__()
    db = next(gen)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute('SELECT nspacl FROM pg_namespace WHERE nspname=\'public\'')
            acl = cur.fetchone()
            cur.execute(SQL.read_text())
            cur.execute(SQL.read_text())
            cur.execute('SELECT nspacl FROM pg_namespace WHERE nspname=\'public\'')
            assert cur.fetchone() == acl
        yield db
    finally:
        gen.close()


def project(h, run):
    return next(item for item in h.service.store.list_runs(
        h.round_id, columns=public_dashboard._EVALUATION_COLUMNS,
        submission_ids=[run['submission_id']],
    ) if item['run_id'] == run['run_id'])


@pytest.mark.parametrize('cause', ['provider_error', 'stage_closed'])
def test_exact_native_failure_projects_live_and_historical_without_mutation(database, completed, cause):
    h = prior.reset(database, completed)
    run = prior.selected(completed, 'FailedRoute')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        prior.gap(cur, run, cause, execution=True)
        prior.native(cur, run)
        before = prior.state(cur, h.round_id)
    compact = project(h, run)
    assert compact['native_failure_reason'] == REASON
    progress = public_dashboard.evaluation_progress([compact], datetime.now(timezone.utc))
    assert progress['failure_reasons'] == [REASON]
    assert 'body_b64' not in json.dumps(compact)
    assert 'No endpoints' not in json.dumps(progress)
    assert 'request_hash' not in json.dumps(compact)
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert prior.state(cur, h.round_id) == before


@pytest.mark.parametrize('status,message', [
    (404, 'Requested model does not exist'), (404, 'Requested page not found'),
    (401, 'Invalid API key'), (402, 'Insufficient credits'),
    (429, 'Rate limited'), (500, 'Internal provider error'),
    (503, 'No endpoints found matching your data policy.'),
    (200, 'No endpoints found matching your data policy.'),
    (404, 'Quoted user input: No endpoints found matching your data policy.'),
])
def test_unrelated_errors_remain_generic(database, completed, status, message):
    h = prior.reset(database, completed)
    run = prior.selected(completed, 'FailedRoute')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        prior.gap(cur, run, 'provider_error', execution=True)
        prior.native(cur, run, status=status, message=message)
    compact = project(h, run)
    assert compact['native_failure_reason'] is None
    assert public_dashboard._evaluation_failure_reason(compact) == 'provider'


@pytest.mark.parametrize('case', ['missing_dispatch', 'missing_settlement', 'round_binding', 'submission_binding', 'other_run', 'later_success'])
def test_incomplete_cross_identity_and_superseded_native_proof_does_not_label(database, completed, case):
    h = prior.reset(database, completed)
    run = prior.selected(completed, 'FailedRoute')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        prior.gap(cur, run, 'provider_error', execution=True)
        prior.native(cur, prior.selected(completed, 'HistoricalRoute') if case == 'other_run' else run)
        cur.execute('SET LOCAL session_replication_role=replica')
        if case in ('missing_dispatch', 'missing_settlement'):
            cur.execute('DELETE FROM public.lab_arena_ledger WHERE run_id=%s AND entry_kind=%s',
                        (run['run_id'], case.removeprefix('missing_')))
        elif case in ('round_binding', 'submission_binding'):
            column = 'round_id' if case == 'round_binding' else 'submission_id'
            cur.execute('UPDATE public.lab_arena_ledger SET ' + column + '=%s WHERE run_id=%s', ('unrelated-private-identity', run['run_id']))
        elif case == 'later_success':
            prior.native(cur, run, status=200, message='later successful fallback', call='b')
        cur.execute('SET LOCAL session_replication_role=origin')
    assert project(h, run)['native_failure_reason'] is None
    if case == 'round_binding':
        # The older fixture resets by round; remove only this deliberately
        # misbound receipt so it cannot affect the next isolated control.
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute('DELETE FROM public.lab_arena_ledger WHERE run_id=%s', (run['run_id'],))


def test_newer_successful_attempt_wins_without_inheriting_failure_diagnostic(database, completed):
    h = prior.reset(database, completed)
    run = prior.selected(completed, 'FailedRoute')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        prior.gap(cur, run, 'provider_error', execution=True)
        prior.native(cur, run)
    failed = project(h, run)
    success = dict(failed, run_id=run['run_id'] + '-later', attempt=failed['attempt'] + 1,
                   status='accepted', terminal_cause='accepted', native_failure_reason=None)
    progress = public_dashboard.evaluation_progress([failed, success], datetime.now(timezone.utc))
    assert progress['counts']['completed'] == 1
    assert progress['counts']['failed'] == 0
    assert 'failure_reasons' not in progress


def test_published_snapshot_and_history_retain_specific_reason(database, completed):
    h = prior.reset(database, completed)
    run = prior.selected(completed, 'FailedRoute')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        prior.gap(cur, run, 'provider_error', execution=True)
        prior.native(cur, run)
    assert h.service.publish(h.round_id)['status'] == 'ok'
    for snapshot in (public_dashboard.submissions_snapshot(h.service, h.round_id),
                     public_dashboard.history_snapshot(h.service)):
        item = next(item for item in snapshot['submissions'] if item['submission_id'] == run['submission_id'])
        assert item['status'] == 'provider_blocked'
        assert item['score_status'] == 'not_evaluated'
        assert item['final_score'] is None
        assert REASON in item['evaluation']['failure_reasons']
        serialized = json.dumps(item)
        assert 'No endpoints' not in serialized and 'body_b64' not in serialized


def test_function_privileges_and_projection_cost_are_bounded(database, completed):
    h = prior.reset(database, completed)
    run = prior.selected(completed, 'FailedRoute')
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        prior.gap(cur, run, 'provider_error', execution=True)
        prior.native(cur, run)
        for role in ('anon', 'authenticated', 'service_role', 'lab_arena_service'):
            cur.execute('SELECT has_function_privilege(%s,%s,\'EXECUTE\')', (role, SIGNATURE))
            assert cur.fetchone() == (role == 'lab_arena_service',)
        cur.execute('SELECT prosrc FROM pg_proc WHERE oid=%s::regprocedure', (SIGNATURE,))
        source = cur.fetchone()[0]
        assert 'UPDATE ' not in source and 'INSERT ' not in source and 'DELETE ' not in source
        cur.execute('SET LOCAL enable_seqscan=off')
        cur.execute("EXPLAIN (FORMAT JSON) SELECT * FROM public.lab_arena_ledger WHERE run_id=%s AND provider='openrouter' AND entry_kind='reservation' ORDER BY (entry_doc->'action_sequence') DESC NULLS FIRST,(entry_doc->'provider_attempt') DESC NULLS LAST,entry_id DESC LIMIT 1", (run['run_id'],))
        assert 'Index' in json.dumps(cur.fetchone()[0])
    started = time.monotonic()
    assert project(h, run)['native_failure_reason'] == REASON
    elapsed = time.monotonic() - started
    assert elapsed < 5
    print('compact native failure projection seconds:', round(elapsed, 4))
