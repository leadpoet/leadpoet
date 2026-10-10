"""A rejected compatibility reply can recover only its exact bill after close."""
from pathlib import Path

import pytest

from tests.lab_arena import closed_miner_success_score_billing_postgres_test as prior
from tests.lab_arena.deepline_budget_only_exact_recovery_postgres_test import (
    database as base_database,
)

database = prior.database
SCRIPTS = Path(__file__).parents[2] / 'scripts'
HELPER = 'public.lab_arena__closed_score_adaptation_uncertainty_v1(bigint)'


@pytest.fixture(scope='module')
def migrated(database):
    prior.upgraded.__wrapped__(database)
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            for filename in (
                '261-lab-arena-settlement-failure-deepline-cost-reconciliation.sql',
                '453-lab-arena-deepline-adaptation-cost-reconciliation.sql',
                '456-lab-arena-score-deepline-cost-priority.sql',
                '457-lab-arena-closed-adaptation-score-billing.sql',
            ):
                cursor.execute((SCRIPTS / filename).read_text())
            cursor.execute((SCRIPTS / '457-lab-arena-closed-adaptation-score-billing.sql').read_text())
            for role in ('anon', 'authenticated', 'service_role', 'lab_arena_service'):
                cursor.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')", (role, HELPER))
                assert cursor.fetchone() == (False,)
    return database


def _closed_adaptation(database, tmp_path, label):
    h, lease, connect, broker, transport, candidate = prior._make_call(
        database, tmp_path, label, 'accepted')
    entry_id = candidate['uncertain_entry_id']
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute('SET LOCAL session_replication_role=replica')
        cursor.execute(
            "UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call}',"
            "'{\"reason\":\"settle_failure\",\"call_succeeded\":false,"
            "\"failure_stage\":\"response_adaptation\","
            "\"error_class\":\"CompatibilityResponseError\"}'::jsonb) "
            'WHERE entry_id=%s', (entry_id,))
        # The live rejected compatibility replies have no cached response row.
        cursor.execute('DELETE FROM public.lab_arena_deepline_call_responses '
                       'WHERE call_identity=%s', (candidate['call_identity'],))
    candidate = h.service.store.list_deepline_cost_reconciliations(h.round_id)[0]
    return h, lease, connect, broker, transport, candidate


def _candidate_ids(store, round_id):
    return (
        [x['uncertain_entry_id'] for x in store.list_deepline_cost_reconciliations(round_id)],
        [x['uncertain_entry_id'] for x in store.list_deepline_cost_reconciliations(round_id, score_only=True)],
        [x['uncertain_entry_id'] for x in store.list_deepline_cost_reconciliations(round_id, successful_execute_only=True)],
    )


def test_closed_accepted_adaptation_exact_bill_preserves_published_result(migrated, tmp_path):
    h, lease, connect, broker, transport, candidate = _closed_adaptation(
        migrated, tmp_path, '30',)
    store = h.service.store
    entry_id = candidate['uncertain_entry_id']
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute('SELECT ' + HELPER.split('(')[0] + '(%s)', (entry_id,))
        assert cursor.fetchone() == (True,)
        cursor.execute('SELECT md5(to_jsonb(r)::text) FROM public.lab_arena_rounds r WHERE round_id=%s',
                       (h.round_id,))
        round_hash = cursor.fetchone()[0]
        cursor.execute('SELECT md5(to_jsonb(r)::text) FROM public.lab_arena_runs r WHERE run_id=%s',
                       (lease['run_id'],))
        run_hash = cursor.fetchone()[0]
        cursor.execute('SELECT public.lab_arena_icp_cost_eligibility(%s,%s,%s,%s)',
                       (h.round_id, lease['submission_id'], lease['icp_position'], 1))
        eligibility = cursor.fetchone()[0]
    assert _candidate_ids(store, h.round_id) == ([entry_id], [entry_id], [])
    selected = store.next_closed_provider_reconciliation(
        mode='live', network_name='finney', netuid=71, round_id=h.round_id)
    assert selected['provider'] == 'deepline' and selected['uncertain_entry_id'] == entry_id
    assert store.reconcile_deepline_cost(
        round_id=h.round_id, run_id=lease['run_id'], call_identity=candidate['call_identity'],
        uncertain_entry_id=entry_id, request_id=candidate['request_id'],
        operation=candidate['operation'], credential_fingerprint='sha256:' + 'b' * 64,
        actual_microusd=2000, cost_units='0.02', execution_key=candidate['execution_key'],
        recovered_request_id='public-execute:wrong457') == {'status': 'stale'}
    original = store.list_ledger(call_identity=candidate['call_identity'])
    transport.bill_final = True
    assert broker.reconcile_deepline_cost(candidate)['status'] == 'settled'
    assert broker.reconcile_deepline_cost(candidate)['status'] == 'settled'
    ledger = store.list_ledger(call_identity=candidate['call_identity'])
    assert ledger[:-1] == original and len(ledger) == 4
    assert ledger[-1]['entry_kind'] == 'settlement' and ledger[-1]['amount_microusd'] == 2000
    assert _candidate_ids(store, h.round_id) == ([], [], [])
    assert store.next_closed_provider_reconciliation(
        mode='live', network_name='finney', netuid=71, round_id=h.round_id) == {'status': 'none'}
    assert sum(request['method'] == 'POST' for request in transport.requests) == 1
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute('SELECT md5(to_jsonb(r)::text) FROM public.lab_arena_rounds r WHERE round_id=%s',
                       (h.round_id,))
        assert cursor.fetchone()[0] == round_hash
        cursor.execute('SELECT md5(to_jsonb(r)::text) FROM public.lab_arena_runs r WHERE run_id=%s',
                       (lease['run_id'],))
        assert cursor.fetchone()[0] == run_hash
        cursor.execute('SELECT public.lab_arena_icp_cost_eligibility(%s,%s,%s,%s)',
                       (h.round_id, lease['submission_id'], lease['icp_position'], 1))
        assert cursor.fetchone()[0] == eligibility


def test_closed_adaptation_rejects_unrelated_and_damaged_binding(migrated, tmp_path):
    h, lease, connect, _, _, candidate = _closed_adaptation(migrated, tmp_path, '31')
    store = h.service.store
    entry_id, identity = candidate['uncertain_entry_id'], candidate['call_identity']
    changes = (
        ("UPDATE public.lab_arena_runs SET kind='execute' WHERE run_id=%s", (lease['run_id'],)),
        ("UPDATE public.lab_arena_runs SET status='failed' WHERE run_id=%s", (lease['run_id'],)),
        ("UPDATE public.lab_arena_runs SET terminal_cause='judge_error' WHERE run_id=%s", (lease['run_id'],)),
        ("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call,error_class}',"
         "'\"ValueError\"'::jsonb) WHERE entry_id=%s", (entry_id,)),
        ("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call,failure_stage}',"
         "'\"settlement\"'::jsonb) WHERE entry_id=%s", (entry_id,)),
        ("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call,reason}',"
         "'\"missing_provider_cost\"'::jsonb) WHERE entry_id=%s", (entry_id,)),
        ("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call,call_succeeded}',"
         "'true'::jsonb) WHERE entry_id=%s", (entry_id,)),
        ("UPDATE public.lab_arena_ledger SET entry_doc=entry_doc-'deepline_execution_key' "
         "WHERE call_identity=%s AND entry_kind='reservation'", (identity,)),
        ("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{deepline_request_id}',"
         "'\"ctx-tool-ffffffffffffffffffffffffffffffff\"'::jsonb) "
         "WHERE call_identity=%s AND entry_kind='reservation'", (identity,)),
        ("DELETE FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='dispatch'", (identity,)),
    )
    with connect() as connection, connection.cursor() as cursor:
        for statement, args in changes:
            cursor.execute('SAVEPOINT bad_adaptation')
            cursor.execute('SET LOCAL session_replication_role=replica')
            cursor.execute(statement, args)
            cursor.execute('SELECT ' + HELPER.split('(')[0] + '(%s)', (entry_id,))
            assert cursor.fetchone() == (False,), statement
            cursor.execute('SELECT public.lab_arena_list_deepline_cost_reconciliations_v1(%s,\'\',0,20),'
                           'public.lab_arena_list_deepline_cost_reconciliations_v3(%s,\'\',0,20),'
                           'public.lab_arena_next_closed_deepline_reconciliation_v1('
                           "'live','finney',71,%s,0)", (h.round_id, h.round_id, h.round_id))
            general, score, closed = cursor.fetchone()
            assert general['items'] == score['items'] == [] and closed == {'status': 'none'}, statement
            cursor.execute('ROLLBACK TO SAVEPOINT bad_adaptation')
