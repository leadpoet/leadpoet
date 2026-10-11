"""Native outage receipts unlock authoritative GET-only Deepline accounting."""
import copy
import json
from pathlib import Path

import pytest

from lab_arena import broker as br
from tests.lab_arena import adaptation_cost_reconciliation453_postgres_test as prior
from tests.lab_arena.deepline_delayed_cost_reconciliation_postgres_test import _store
from tests.lab_arena.deepline_delayed_cost_reconciliation_unit_test import _broker, SECRET, FINGERPRINT

database = prior.database
ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/462-lab-arena-deepline-native-settlement-binding.sql'
PREHASH = prior.POSTHASH
POSTHASH = '8a973077fb121ef9f01866b301aba0ec5dd05da8143f87644ad8e28afa530f49'
CALL = 'sha256:30271d7435906cabea22e8baaeb8787554064e4811eff07e89e09d55c45e1393'
REQUEST = 'ctx-tool-' + CALL[7:39]
EXECUTION = 'arena:' + CALL[7:]
NATIVE = 'public-execute:79720521-e922-446f-b014-d8bf878c878d'
OPERATION = 'contextdev_get_web_scrape_markdown'
RESERVATION = {**prior.RESERVATION, 'tool': OPERATION, 'deepline_request_id': REQUEST,
               'deepline_execution_key': EXECUTION, 'deepline_billing_provider': 'contextdev',
               'deepline_operation_aliases': [OPERATION, 'get_web_scrape_markdown'],
               'reserve_remaining_budget': True}
WITNESS = {'reason': 'settle_failure', 'failure_stage': 'settlement',
           'error_class': 'Exception', 'call_succeeded': True, 'provider_status': 200,
           'top_level_job_status': 'completed', 'billing_present': True,
           'known_actual_microusd': 1000, 'deepline_job_id': NATIVE,
           'deepline_request_id': REQUEST, 'deepline_execution_key': EXECUTION,
           'deepline_operation': OPERATION, 'credential_fingerprint': FINGERPRINT,
           'provider_cost': {'basis': 'deepline_exact_request_credits_x_0.10_usd',
                             'units': '0.01', 'operation': OPERATION,
                             'unit_name': 'credits', 'request_id': NATIVE}}


@pytest.fixture(scope='module')
def migrated(database):
    prior.migrated.__wrapped__(database)
    psycopg, dsn = database
    with psycopg.connect(**dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            before, security = prior._definition(cur), prior._security(cur)
            signatures = (
                'public.lab_arena_list_deepline_cost_reconciliations_v2(text,text,bigint,integer,boolean)',
                'public.lab_arena_reconcile_deepline_cost_v2(text,text,text,bigint,text,text,text,bigint,text,text,text)',
                prior.recovery.prior.identity.EXPIRY)
            definitions = {}
            for signature in signatures:
                cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
                definitions[signature] = cur.fetchone()[0]
            cur.execute('BEGIN')
            with pytest.raises(psycopg.Error, match='preimage differs'):
                cur.execute(SQL.read_text().replace(PREHASH, '0' * 64, 1))
            cur.execute('ROLLBACK')
            assert prior._definition(cur) == before
            for broken, message in (
                (SQL.read_text().replace('v_old_class CONSTANT TEXT := $old_class$', 'v_old_class CONSTANT TEXT := $old_class$extra', 1), 'preimage differs'),
                (SQL.read_text().replace(POSTHASH, '0' * 64), 'postimage differs')):
                cur.execute('BEGIN')
                with pytest.raises(psycopg.Error, match=message):
                    cur.execute(broken)
                cur.execute('ROLLBACK')
                assert prior._definition(cur) == before
                assert prior._security(cur) == security
            cur.execute(SQL.read_text())
            cur.execute(SQL.read_text())
            assert prior._security(cur) == security
            cur.execute("SELECT encode(extensions.digest(pg_get_functiondef(%s::regprocedure),'sha256'),'hex')", (prior.SIGNATURE,))
            assert cur.fetchone()[0] == POSTHASH
            for signature, definition in definitions.items():
                cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
                assert cur.fetchone()[0] == definition
            for role in ('anon', 'authenticated', 'service_role', 'lab_arena_service'):
                cur.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')", (role, prior.SIGNATURE))
                assert cur.fetchone()[0] is False
    return True


def _binding(cur, witness=None, reservation=None):
    cur.execute('SELECT public.lab_arena__deepline_cost_binding_v1(%s::jsonb,%s::jsonb,%s)',
                (json.dumps({'reason': 'worker_reported', 'call': WITNESS if witness is None else witness}),
                 json.dumps(RESERVATION if reservation is None else reservation), CALL))
    return cur.fetchone()[0]


def test_exact_receipt_and_general_credit_ceiling(database, migrated):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert _binding(cur)
        for units, amount in [('0.02', 2000), ('1.234567', 123457), ('0.00000001', 1), ('0', 0)]:
            assert _binding(cur, {**WITNESS, 'known_actual_microusd': amount,
                'provider_cost': {**WITNESS['provider_cost'], 'units': units}})
        assert _binding(cur, {**WITNESS, 'error_class': 'ArenaStoreError'})
        # New unavailable-family labels need no successful receipt to permit
        # existing exact billing lookup, but native IDs still need the key.
        future = {**WITNESS, 'error_class': 'ArenaStoreError', 'call_succeeded': False}
        future.pop('provider_cost'); future.pop('known_actual_microusd')
        assert _binding(cur, future)


NEGATIVES = [
    ('error_class', 'RuntimeError'), ('failure_stage', 'provider'),
    ('call_succeeded', False), ('call_succeeded', 'true'), ('provider_status', 502),
    ('provider_status', '200'), ('top_level_job_status', 'running'),
    ('billing_present', False), ('billing_present', 'true'),
    ('deepline_request_id', 'ctx-tool-' + '9' * 32),
    ('deepline_execution_key', 'arena:' + '9' * 64),
    ('deepline_operation', 'exa_contents'), ('credential_fingerprint', 'sha256:' + '9' * 64),
    ('deepline_job_id', 'public-execute:not-a-uuid'), ('known_actual_microusd', 1001),
    ('known_actual_microusd', '1000'), ('known_actual_microusd', True),
    ('provider_cost.request_id', 'public-execute:00000000-0000-0000-0000-000000000000'),
    ('provider_cost.operation', 'exa_contents'), ('provider_cost.basis', 'estimated'),
    ('provider_cost.unit_name', 'usd'), ('provider_cost.units', 'bad'),
    ('provider_cost.units', '-0.01'), ('provider_cost.units', '0.02'),
    ('provider_cost.units', 0.01), ('provider_cost.units', '9' * 1000),
]


@pytest.mark.parametrize('field,value', NEGATIVES)
def test_incomplete_or_wrong_receipt_never_becomes_candidate(database, migrated, field, value):
    witness = copy.deepcopy(WITNESS)
    parts = field.split('.')
    target = witness if len(parts) == 1 else witness[parts[0]]
    target[parts[-1]] = value
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert not _binding(cur, witness)
        target.pop(parts[-1])
        assert not _binding(cur, witness)


def test_wrong_immutable_binding_and_v1_no_key(database, migrated):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        for field, value in [('deepline_request_id', 'ctx-tool-' + '9' * 32),
            ('deepline_execution_key', 'arena:' + '9' * 64), ('tool', 'exa_contents'),
            ('credential_fingerprint', 'sha256:' + '9' * 64)]:
            assert not _binding(cur, reservation={**RESERVATION, field: value})
        reservation = dict(RESERVATION); reservation.pop('deepline_execution_key')
        assert not _binding(cur, reservation=reservation)
        assert not _binding(cur, {**WITNESS, 'error_class': 'ArenaStoreError'}, reservation)


class ExactGet:
    def __init__(self, credits=0.01, *, wrong_id=False):
        self.sent = []; self.credits = credits; self.wrong_id = wrong_id
    def send(self, **request):
        self.sent.append(request)
        assert request['method'] == 'GET'
        assert request['headers']['authorization'] == 'Bearer ' + SECRET
        if '/executions/by-key/' in request['url']:
            return br.ProviderResponse(200, {'x-deepline-idempotency-supported': 'true'},
                json.dumps({'toolId': OPERATION, 'requestId': NATIVE,
                    'executionRecovery': {'idempotencyKey': EXECUTION, 'state': 'completed'}}).encode())
        assert NATIVE.replace(':', '%3A') in request['url']
        rid = 'public-execute:wrong' if self.wrong_id else NATIVE
        entry = {'id': 'charge462', 'request_id': rid, 'provider': 'contextdev',
                 'operation': OPERATION, 'credits': self.credits, 'delta': -self.credits,
                 'charge_state': 'posted', 'charge_finality': 'final', 'metadata': {}}
        return br.ProviderResponse(200, {}, json.dumps({'recent': {'request_id': rid, 'entries': [entry]}}).encode())


def _seed(conn, kind='execute', witness=None, miner=None):
    miner = (kind == 'execute') if miner is None else miner
    lease, _ = prior.recovery._seed(conn, count=0, kind=kind)
    rpc = prior.recovery.prior._rpc
    token = prior.recovery.prior.TOKEN_HASH
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('UPDATE public.lab_arena_submissions SET is_king=%s WHERE submission_id=%s', (not miner, lease['submission_id']))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    funding = 'miner_key' if miner else 'host'
    assert rpc(conn, "lab_arena_reserve_call(%s,%s,%s,'deepline.execute','deepline',%s,0,%s::jsonb,4500)",
               (lease['run_id'], token, CALL, funding, json.dumps(RESERVATION)))['status'] == 'reserved'
    assert prior.recovery.prior._dispatch(conn, lease, CALL)['status'] == 'dispatched'
    assert rpc(conn, 'lab_arena_mark_uncertain(%s,%s,%s,%s::jsonb,4500)',
               (lease['run_id'], token, CALL, json.dumps(WITNESS if witness is None else witness)))['status'] == 'uncertain'
    # Model the already accepted outage-shaped execution in this disposable DB.
    with conn.cursor() as cur:
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_runs SET status='accepted',terminal_cause='accepted' WHERE run_id=%s", (lease['run_id'],))
        cur.execute('SET LOCAL session_replication_role=origin')
    conn.commit()
    return lease


@pytest.mark.parametrize('credits,expected', [(0.01, 1000), (0.02, 2000)])
def test_successful_miner_outage_exact_get_settles_once_and_releases_cost_gate(database, migrated, credits, expected):
    with database[0].connect(**database[1]) as conn:
        lease = _seed(conn)
        rpc = prior.recovery.prior._rpc
        round_id = prior.recovery.prior.recovery.prior.ROUND
        args = (round_id, lease['submission_id'], lease['icp_position'], 5)
        before = rpc(conn, 'lab_arena_icp_cost_eligibility(%s,%s,%s,%s)', args)
        assert before['eligible'] is False and before['eligibility_reason'] == 'provider_cost_uncertain'
        store = _store(database)
        try:
            candidates = store.list_deepline_cost_reconciliations(round_id, run_id=lease['run_id'], successful_execute_only=True)
            assert len(candidates) == 1
            candidate = candidates[0]
            assert candidate['request_id'] == NATIVE and candidate['funding_source'] == 'miner_key'
            transport = ExactGet(credits)
            broker = _broker(store, transport)
            broker._provider_funding_source_for = lambda *_: 'miner_key'
            broker._credential_for = lambda *_: 'wrong-test-key'
            assert broker.reconcile_deepline_cost(candidate)['status'] == 'credential_mismatch'
            assert transport.sent == []
            broker._credential_for = lambda *_: SECRET
            bad = ExactGet(wrong_id=True); broker._transport = bad
            assert broker.reconcile_deepline_cost(candidate)['status'] == 'pending'
            assert store.list_ledger(call_identity=CALL)[-1]['entry_kind'] == 'uncertain'
            broker._transport = transport
            assert broker.reconcile_deepline_cost(candidate)['status'] == 'settled'
            assert broker.reconcile_deepline_cost(candidate)['status'] == 'settled'
            settled = store.list_ledger(call_identity=CALL)
            assert [r['entry_kind'] for r in settled] == ['reservation', 'dispatch', 'uncertain', 'settlement']
            assert settled[-1]['amount_microusd'] == expected
            assert settled[-1]['terminal_response']['call_succeeded'] is True
            assert transport.sent and all(r['method'] == 'GET' for r in transport.sent)
            after = rpc(conn, 'lab_arena_icp_cost_eligibility(%s,%s,%s,%s)', args)
            assert after['eligible'] is True
            assert after['competition_sourcing_microusd'] == before['competition_sourcing_microusd'] + expected
        finally:
            store.close()


@pytest.mark.parametrize('kind,miner', [('execute', True), ('score', False), ('score', True)])
def test_published_accepted_run_excluded_without_any_mutation(database, migrated, kind, miner):
    with database[0].connect(**database[1]) as conn:
        lease = _seed(conn, kind=kind, miner=miner)
        round_id = prior.recovery.prior.recovery.prior.ROUND
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_rounds SET status='published' WHERE round_id=%s", (round_id,))
            cur.execute('SET LOCAL session_replication_role=origin')
            cur.execute("SELECT entry_id FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='uncertain'", (CALL,))
            entry_id = cur.fetchone()[0]
        conn.commit()
        before = prior.recovery.prior._audit(conn, lease)
        store = _store(database)
        try:
            assert store.list_deepline_cost_reconciliations(round_id, run_id=lease['run_id']) == []
            result = store.reconcile_deepline_cost(round_id=round_id, run_id=lease['run_id'],
                call_identity=CALL, uncertain_entry_id=entry_id, request_id=NATIVE,
                operation=OPERATION, credential_fingerprint=FINGERPRINT,
                actual_microusd=1000, cost_units='0.01', execution_key=EXECUTION,
                recovered_request_id=NATIVE)
            assert result['status'] == 'stale'
            assert prior.recovery.prior._audit(conn, lease) == before
        finally:
            store.close()


@pytest.mark.parametrize('field,value', [
    ('provider_status', 502), ('provider_status', '200'), ('known_actual_microusd', 1001),
    ('credential_fingerprint', 'sha256:' + '9' * 64),
    ('deepline_request_id', 'ctx-tool-' + '9' * 32),
    ('deepline_job_id', 'public-execute:not-a-uuid'),
    ('deepline_execution_key', 'arena:' + '9' * 64),
    ('top_level_job_status', 'running'),
])
def test_rejected_receipt_is_excluded_by_real_candidate_and_settlement_rpcs(database, migrated, field, value):
    with database[0].connect(**database[1]) as conn:
        lease = _seed(conn, witness={**WITNESS, field: value})
        round_id = prior.recovery.prior.recovery.prior.ROUND
        with conn.cursor() as cur:
            cur.execute("SELECT entry_id FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='uncertain'", (CALL,))
            entry_id = cur.fetchone()[0]
        before = prior.recovery.prior._audit(conn, lease)
        store = _store(database)
        try:
            assert store.list_deepline_cost_reconciliations(round_id, run_id=lease['run_id'], successful_execute_only=True) == []
            assert store.reconcile_deepline_cost(round_id=round_id, run_id=lease['run_id'],
                call_identity=CALL, uncertain_entry_id=entry_id, request_id=NATIVE,
                operation=OPERATION, credential_fingerprint=FINGERPRINT,
                actual_microusd=1000, cost_units='0.01', execution_key=EXECUTION,
                recovered_request_id=NATIVE)['status'] == 'stale'
            assert prior.recovery.prior._audit(conn, lease) == before
            result = prior.recovery.prior._rpc(conn, 'lab_arena_icp_cost_eligibility(%s,%s,%s,5)',
                (round_id, lease['submission_id'], lease['icp_position']))
            assert result['eligible'] is False
        finally:
            store.close()
