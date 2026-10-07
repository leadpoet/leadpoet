"""Filtered candidate listing on disposable PostgreSQL; no accounting changes."""
from pathlib import Path

import pytest

from lab_arena import broker as br, contracts
from lab_arena.store import ArenaStoreError, hash_lease_token
from tests.lab_arena.deepline_budget_only_exact_recovery_postgres_test import (
    database as base_database, run, reserve,
)
from tests.lab_arena.test_lab_arena_migration_postgres import claim, complete, sha
from tests.lab_arena.deepline_delayed_cost_reconciliation_postgres_test import (
    _store, _open_scoring, _uncertain_call,
)

MIGRATION = '425-lab-arena-successful-deepline-cost-priority.sql'
V1 = 'public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)'
V2 = 'public.lab_arena_list_deepline_cost_reconciliations_v2(text,text,bigint,integer,boolean)'
UNCHANGED = [V1,
    'public.lab_arena_reconcile_deepline_cost_v1(text,text,text,bigint,text,text,text,bigint,text)',
    'public.lab_arena_reconcile_deepline_cost_v2(text,text,text,bigint,text,text,text,bigint,text,text,text)',
    'public.lab_arena__successful_call_cost_state(text,text,text)',
    'public.lab_arena__successful_icp_cost_state(text,text,integer)',
]


@pytest.fixture(scope='module')
def database(base_database):
    with base_database[0].connect(**base_database[1]) as c, c.cursor() as cur:
        cur.execute('SELECT pg_get_functiondef(signature::regprocedure) FROM unnest(%s::text[]) signature', (UNCHANGED,))
        before = cur.fetchall()
        for _ in range(2):
            cur.execute((Path(__file__).parents[2] / 'scripts' / MIGRATION).read_text())
        cur.execute('SELECT pg_get_functiondef(signature::regprocedure) FROM unnest(%s::text[]) signature', (UNCHANGED,))
        assert cur.fetchall() == before
    return base_database


def uncertain(h, lease, token, label, *, success=False, native=None, reason=None, dispatched_only=False):
    identity = contracts.provider_call_identity(
        attempt=lease['attempt'], assignment_id=lease['assignment_id'],
        icp_position=lease['icp_position'], action_sequence=0,
        operation_id='deepline.execute', request_hash=sha(label),
    )
    local = 'ctx-tool-' + identity[7:39]
    key = 'arena:' + identity[7:]
    fingerprint = 'sha256:' + 'a' * 64
    ident, result = reserve(h, lease, token, label, doc={
        'deepline_request_id': local, 'deepline_execution_key': key,
        'tool': 'exa_search', 'credential_fingerprint': fingerprint,
        'deepline_billing_provider': 'exa',
        'deepline_operation_aliases': ['exa_search', 'search'],
    })
    assert ident == identity and result['status'] == 'reserved'
    store = h.service.store
    args = dict(run_id=lease['run_id'], lease_token_hash=hash_lease_token(token), call_identity=identity)
    assert store.mark_dispatched(**args)['status'] == 'dispatched'
    call = dict(reason=reason or ('missing_provider_cost' if success else 'transport_failure'),
        call_succeeded=success, deepline_request_id=local,
        deepline_execution_key=key,
        deepline_operation='exa_search', credential_fingerprint=fingerprint)
    if native:
        call['deepline_job_id'] = native
    if not dispatched_only:
        assert store.mark_uncertain(**args, call_doc=call)['status'] == 'uncertain'
    return identity, local, key, fingerprint


def test_priority_is_exact_success_and_recovered_success_with_fair_wrap(database, tmp_path):
    h, lease, token, connect = run(database, tmp_path, '21')
    uncertain(h, lease, token, 'old-failed')
    native = 'public-execute:11111111-2222-4333-8444-555555555555'
    direct, _, _, _ = uncertain(h, lease, token, 'completed', success=True, native=native)
    recovered, local, key, fingerprint = uncertain(h, lease, token, 'saved')
    result = h.service.store.recover_deepline_response(
        run_id=lease['run_id'], lease_token_hash=hash_lease_token(token),
        call_identity=recovered, request_hash=sha('saved'), execution_key=key,
        credential_fingerprint=fingerprint, request_id='public-execute:saved',
        operation='exa_search', actual_microusd=None,
        terminal_response=br._terminal_response_document(
            200, {}, b'{"result":"saved"}', call_succeeded=True), lease_ttl_seconds=420,
    )
    assert result['status'] == 'uncertain'
    store = h.service.store
    assert complete(store, lease['run_id'], hash_lease_token(token), 'accepted',
                    output_ref='arena/' + h.round_id + '/priority-output.json')['status'] == 'accepted'
    ordinary = store.list_deepline_cost_reconciliations(h.round_id, limit=20)
    priority = store.list_deepline_cost_reconciliations(h.round_id, limit=20, successful_execute_only=True)
    assert len(ordinary) == 3
    assert [c['call_identity'] for c in priority] == [direct, recovered]
    assert priority[0]['request_id'] == native
    assert priority[0]['billing_provider'] == 'exa'
    first = store.list_deepline_cost_reconciliations(h.round_id, successful_execute_only=True)[0]
    second = store.list_deepline_cost_reconciliations(h.round_id, after_entry_id=first['uncertain_entry_id'], successful_execute_only=True)[0]
    wrapped = store.list_deepline_cost_reconciliations(h.round_id, after_entry_id=second['uncertain_entry_id'], successful_execute_only=True)[0]
    assert [first['call_identity'], second['call_identity'], wrapped['call_identity']] == [direct, recovered, direct]
    # A false V2 filter is byte-for-byte ordinary V1 behavior; NULL fails closed.
    with connect() as c, c.cursor() as cur:
        cur.execute('SELECT public.lab_arena_list_deepline_cost_reconciliations_v1(%s,\'\',0,20), public.lab_arena_list_deepline_cost_reconciliations_v2(%s,\'\',0,20,FALSE), public.lab_arena_list_deepline_cost_reconciliations_v2(%s,\'\',0,20,NULL)', (h.round_id,) * 3)
        v1, v2, null = cur.fetchone()
        assert v1 == v2 and null['items'] == []
        cur.execute("SELECT public.lab_arena__successful_call_cost_state(%s,'execute','deepline')", (lease['submission_id'],))
        assert cur.fetchone()[0]['success_unresolved_calls'] == 2
    # Match the two affected outcomes: final paid and final zero cost. V2 uses
    # the original settlement guard, with one immutable settlement per call.
    for candidate, amount, units, native_id in (
        (first, 10000, '0.1', native),
        (second, 0, '0', 'public-execute:saved'),
    ):
        args = dict(round_id=h.round_id, run_id=lease['run_id'],
            call_identity=candidate['call_identity'],
            uncertain_entry_id=candidate['uncertain_entry_id'],
            request_id=candidate['request_id'], operation=candidate['operation'],
            credential_fingerprint=candidate['credential_fingerprint'],
            actual_microusd=amount, cost_units=units,
            execution_key=candidate['execution_key'], recovered_request_id=native_id)
        wrong = store.reconcile_deepline_cost(**{**args, 'credential_fingerprint': 'sha256:' + 'c' * 64})
        assert wrong['status'] == 'stale'
        with pytest.raises(ArenaStoreError, match='execution_recovery_input_invalid'):
            store.reconcile_deepline_cost(**{**args, 'execution_key': 'arena:' + '0' * 64})
        assert store.reconcile_deepline_cost(**args)['status'] == 'settled'
        assert store.reconcile_deepline_cost(**args)['idempotent'] is True
        with connect() as c, c.cursor() as cur:
            cur.execute("SELECT count(*), sum(amount_microusd) FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='settlement'", (candidate['call_identity'],))
            assert cur.fetchone() == (1, amount)
    assert store.list_deepline_cost_reconciliations(h.round_id, successful_execute_only=True) == []
    assert len(store.list_deepline_cost_reconciliations(h.round_id, limit=20)) == 1
    with connect() as c, c.cursor() as cur:
        cur.execute("SELECT public.lab_arena__successful_call_cost_state(%s,'execute','deepline')", (lease['submission_id'],))
        costs = cur.fetchone()[0]
        assert costs['successful_microusd'] == 10000
        assert costs['successful_calls'] == 2
        assert costs['success_unresolved_calls'] == 0
    store.close()


def test_raw_failure_and_cancelled_liabilities_stay_in_ordinary_lane(database, tmp_path):
    h, lease, token, connect = run(database, tmp_path, '22')
    raw, _, _, _ = uncertain(h, lease, token, 'raw-failure', reason='missing_provider_cost')
    uncertain(h, lease, token, 'malformed-true', success='true')
    cancelled, _, _, _ = uncertain(h, lease, token, 'cancelled', dispatched_only=True)
    assert h.service.store.cancel_round(h.round_id, 'priority-control')['status'] == 'cancelled'
    rows = h.service.store.list_deepline_cost_reconciliations(h.round_id, limit=20)
    assert {row['call_identity'] for row in rows} == {raw, cancelled}
    assert h.service.store.list_deepline_cost_reconciliations(h.round_id, successful_execute_only=True) == []
    with connect() as c, c.cursor() as cur:
        cur.execute("SELECT entry_doc FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='uncertain'", (cancelled,))
        doc = cur.fetchone()[0]
        assert doc['reason'] == 'round_cancelled' and doc['call']['call_succeeded'] is False
    h.service.store.close()


def test_successful_and_failed_scoring_liabilities_stay_in_ordinary_lane(database):
    store = _store(database)
    rid = 'arena-2026-09-14-priorityscore'
    runners, _ = _open_scoring(store, rid, participants=1, runners=1)
    lease, token = claim(store, rid, runners[0])[:2]
    assert lease['kind'] == 'score'
    funding = store.provider_funding(lease['run_id'], 'deepline')['funding_source']
    good = _uncertain_call(store, lease, token, label='score-success',
        reason='missing_provider_cost', call_succeeded=True, funding_source=funding)[0]
    bad = _uncertain_call(store, lease, token, label='score-error',
        reason='missing_provider_cost', call_succeeded=False, funding_source=funding)[0]
    rows = store.list_deepline_cost_reconciliations(rid, limit=20)
    assert {row['call_identity'] for row in rows} == {good, bad}
    assert store.list_deepline_cost_reconciliations(rid, successful_execute_only=True) == []
    store.close()


def test_new_list_is_service_only(database):
    with database[0].connect(**database[1]) as c, c.cursor() as cur:
        cur.execute('SELECT pg_get_userbyid(proowner), prosecdef, proconfig FROM pg_proc WHERE oid=%s::regprocedure', (V2,))
        owner, definer, config = cur.fetchone()
        assert owner == 'lab_arena_owner' and definer
        assert 'search_path=pg_catalog, public' in config
        for role in ('anon', 'authenticated', 'service_role', 'lab_arena_service'):
            cur.execute('SELECT has_function_privilege(%s,%s,\'EXECUTE\')', (role, V2))
            assert cur.fetchone()[0] is (role == 'lab_arena_service')
