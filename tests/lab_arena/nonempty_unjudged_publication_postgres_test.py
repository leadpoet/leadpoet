"""Nonempty accepted work keeps native incomplete-judge and cost boundaries."""
import pytest

from lab_arena import contracts
from lab_arena.service import _PublicationReads
from lab_arena.store import ArenaStoreError
from tests.lab_arena import incomplete_participant_publication540_postgres_test as prior
from tests.lab_arena.incomplete_participant_publication540_postgres_test import database, completed
from tests.lab_arena.test_lab_arena_scoring import scored_company


def nonempty_gap(harness, cursor, run, label, *, output_fault=None):
    """Seed the saved nonempty execution and exhausted judge seen in production."""
    prior.gap(cursor, run)
    ref = run['output_ref'] + '-nonempty-' + label
    if output_fault != 'missing':
        payload = (b'not-json' if output_fault == 'corrupt' else contracts.canonical_json({
            'schema_version': contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,
            'companies': [scored_company(0)],
        }).encode())
        harness.objects.put(ref, payload)
    cursor.execute('SET LOCAL session_replication_role=replica')
    cursor.execute('UPDATE public.lab_arena_runs SET output_ref=%s WHERE run_id=%s',
                   (ref, run['run_id']))
    cursor.execute('SET LOCAL session_replication_role=origin')


def uncertain(cursor, run):
    prior.native(cursor, run)
    cursor.execute('SET LOCAL session_replication_role=replica')
    cursor.execute("UPDATE public.lab_arena_ledger SET entry_kind='uncertain',"
                   "amount_microusd=1234,terminal_response=NULL "
                   "WHERE run_id=%s AND provider='openrouter' AND entry_kind='settlement'",
                   (run['run_id'],))
    cursor.execute('SET LOCAL session_replication_role=origin')


@pytest.mark.parametrize('deadline', [60, -1])
def test_nonempty_unjudged_uncertain_call_does_not_cancel_or_bypass_sql(database, completed, deadline):
    h = prior.reset(database, completed, deadline=deadline)
    run = prior.selected(completed, 'MissingJudge')
    with database[0].connect(**database[1]) as connection, connection.cursor() as cursor:
        nonempty_gap(h, cursor, run, 'uncertain')
        uncertain(cursor, run)
        before = prior.state(cursor, h.round_id)
    costs_before = h.service.store.submission_costs(run['submission_id'])
    assert any(p['reserved_or_uncertain_microusd'] >= 1234 for p in costs_before['providers'])
    if deadline > 0:
        result = h.service.publish(h.round_id)
        assert result == {'status': 'retry', 'round_status': 'scored', 'reason': 'judge_incomplete'}
    else:
        # Historical540 still requires a settled receipt. Current664 separately
        # proves narrow unknown-bill omission without changing that helper.
        # This before-664 control must never become cancellation.
        with pytest.raises(ArenaStoreError, match='Arena execution receipt binding invalid'):
            h.service.publish(h.round_id)
    row = h.service.store.get_round(h.round_id)
    assert row['status'] == 'scored' and row['publication_doc'] is None
    assert row['reward_activated_at'] is None
    assert h.service.store.submission_costs(run['submission_id']) == costs_before
    with database[0].connect(**database[1]) as connection, connection.cursor() as cursor:
        assert prior.state(cursor, h.round_id) == before


def test_nonempty_unjudged_settled_call_reaches_native_partial_publication(database, completed):
    h = prior.reset(database, completed)
    run = prior.selected(completed, 'MissingJudge')
    with database[0].connect(**database[1]) as connection, connection.cursor() as cursor:
        nonempty_gap(h, cursor, run, 'settled')
        prior.native(cursor, run, status=403, message='Forbidden')
        before = prior.state(cursor, h.round_id)
    assert h.service.publish(h.round_id)['status'] == 'ok'
    row = h.service.store.get_round(h.round_id)
    entry = next(r for r in row['publication_doc']['final_ranking']
                 if r['submission_id'] == run['submission_id'])
    assert row['status'] == 'published' and row['reward_activated_at'] is None
    assert entry['final_score'] is None and entry['eligible'] is False
    assert entry['cost_summary'] is None and entry['evaluation'] == 'not_evaluated'
    assert entry['evaluation_reason'] == 'judge_error'
    assert row['publication_doc']['king_decision']['outcome'] == 'no_king'
    with database[0].connect(**database[1]) as connection, connection.cursor() as cursor:
        assert prior.state(cursor, h.round_id) == before


@pytest.mark.parametrize('status', [404, 403])
def test_empty_native_failure_still_uses_existing_incomplete_path(database, completed, status):
    h = prior.reset(database, completed, deadline=60)
    run = prior.selected(completed, 'MissingJudge')
    with database[0].connect(**database[1]) as connection, connection.cursor() as cursor:
        prior.gap(cursor, run)
        prior.native(cursor, run, status=status,
                     message='No endpoints found matching your data policy (Zero data retention).' if status == 404 else 'Forbidden')
        before = prior.state(cursor, h.round_id)
    row = h.service.store.get_round(h.round_id)
    reads = _PublicationReads(h.service, row, h.service.store.list_runs(h.round_id, kind='execute'))
    assert h.service._execution_infrastructure_failures(reads) == (
        {run['run_id']: 'openrouter_zdr_route_unavailable'} if status == 404 else {})
    assert h.service.publish(h.round_id) == {'status': 'retry', 'round_status': 'scored',
                                           'reason': 'judge_incomplete'}
    assert h.service.store.get_round(h.round_id)['status'] == 'scored'
    with database[0].connect(**database[1]) as connection, connection.cursor() as cursor:
        assert prior.state(cursor, h.round_id) == before


@pytest.mark.parametrize('fault', ['missing', 'corrupt'])
def test_missing_or_corrupt_output_still_blocks_publication(database, completed, fault):
    h = prior.reset(database, completed, deadline=60)
    run = prior.selected(completed, 'MissingJudge')
    with database[0].connect(**database[1]) as connection, connection.cursor() as cursor:
        nonempty_gap(h, cursor, run, fault, output_fault=fault)
        before = prior.state(cursor, h.round_id)
    assert h.service.publish(h.round_id) == {'status': 'retry', 'round_status': 'scored',
                                           'reason': 'execution_evidence_unavailable'}
    assert h.service.store.get_round(h.round_id)['status'] == 'scored'
    with database[0].connect(**database[1]) as connection, connection.cursor() as cursor:
        assert prior.state(cursor, h.round_id) == before


def test_nonempty_pending_baseline_judgment_remains_fail_closed(database, completed):
    h = prior.reset(database, completed, deadline=60)
    run = next(r for r in completed[2] if r['kind'] == 'execute' and r['stage'] == 1)
    with database[0].connect(**database[1]) as connection, connection.cursor() as cursor:
        nonempty_gap(h, cursor, run, 'baseline')
        uncertain(cursor, run)
        cursor.execute('SET LOCAL session_replication_role=replica')
        cursor.execute("UPDATE public.lab_arena_runs SET status='pending',terminal_cause=NULL "
                       "WHERE kind='score' AND scored_run_id=%s", (run['run_id'],))
        cursor.execute('SET LOCAL session_replication_role=origin')
        before = prior.state(cursor, h.round_id)
    assert h.service.publish(h.round_id) == {'status': 'retry', 'round_status': 'scored',
                                           'reason': 'baseline_incomplete'}
    assert h.service.store.get_round(h.round_id)['status'] == 'scored'
    with database[0].connect(**database[1]) as connection, connection.cursor() as cursor:
        assert prior.state(cursor, h.round_id) == before
