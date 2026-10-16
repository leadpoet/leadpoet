"""Exact current SQL corrects only keyed edge hints and appends one real bill."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

import pytest

from lab_arena import broker as br
from lab_arena.store import ArenaStoreError, hash_lease_token
from tests.lab_arena import deepline_budget_only_exact_recovery_postgres_test as base
from tests.lab_arena import deepline_candidate_head_scope483_postgres_test as current
from tests.lab_arena.deepline_delayed_cost_reconciliation_unit_test import SECRET, _broker
from tests.lab_arena.deepline_delayed_single_read_test import NATIVE
from tests.lab_arena.deepline_keyed_edge_identity_test import EDGE
from tests.lab_arena.lab_arena_pg_harness import CURRENT_PROVIDER_SERVICE_MIGRATIONS, database_with_lab_arena_migration
from tests.lab_arena.test_lab_arena_migration_postgres import sha

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/494-lab-arena-deepline-keyed-edge-identity.sql'
SIGNATURE = 'public.lab_arena_reconcile_deepline_cost_v2(text,text,text,bigint,text,text,text,bigint,text,text,text)'


def functions(cur):
    cur.execute("SELECT oid,pg_get_functiondef(oid),to_jsonb(p)-'prosrc' FROM pg_proc p WHERE pronamespace='public'::regnamespace AND prokind='f' ORDER BY oid")
    return cur.fetchall()


def rows(db, identity):
    with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
        cur.execute('SELECT to_jsonb(l) FROM public.lab_arena_ledger l WHERE call_identity=%s ORDER BY entry_id', (identity,))
        return [row[0] for row in cur.fetchall()]


def seed(db, tmp_path, label, *, retained=EDGE, keyed=True):
    h, lease, token, connect = base.run(db, tmp_path, '20-' + sha(label)[-8:])
    identity = br.contracts.provider_call_identity(attempt=lease['attempt'],
        assignment_id=lease['assignment_id'], icp_position=lease['icp_position'],
        action_sequence=0, operation_id='deepline.execute', request_hash=sha(label))
    key = 'arena:' + identity[7:] if keyed else None
    local = 'ctx-tool-' + identity[7:39]
    fingerprint = br._credential_fingerprint(SECRET)
    doc = dict(deepline_request_id=local, tool='firecrawl_scrape',
               credential_fingerprint=fingerprint, deepline_billing_provider='firecrawl')
    if key:
        doc['deepline_execution_key'] = key
    ident, result = base.reserve(h, lease, token, label, amount=1, doc=doc)
    assert ident == identity and result['status'] == 'reserved'
    store = h.service.store
    assert store.mark_dispatched(run_id=lease['run_id'], lease_token_hash=hash_lease_token(token), call_identity=identity)['status'] == 'dispatched'
    assert store.mark_uncertain(run_id=lease['run_id'], lease_token_hash=hash_lease_token(token), call_identity=identity,
        call_doc=dict(reason='missing_provider_cost', call_succeeded=False, provider_status=402, deepline_request_id=local,
                      deepline_job_id=retained, deepline_operation='firecrawl_scrape',
                      credential_fingerprint=fingerprint, **({'deepline_execution_key':key} if key else {})))['status'] == 'uncertain'
    candidate = store.list_deepline_cost_reconciliations(h.round_id, run_id=lease['run_id'])[0]
    args = dict(round_id=h.round_id, run_id=lease['run_id'], call_identity=identity,
                uncertain_entry_id=candidate['uncertain_entry_id'], request_id=retained,
                execution_key=key, recovered_request_id=NATIVE, operation='firecrawl_scrape',
                credential_fingerprint=fingerprint, actual_microusd=2000, cost_units='0.02')
    return h, candidate, args


@pytest.fixture(scope='module')
def database(tmp_path_factory):
    generator = database_with_lab_arena_migration(CURRENT_PROVIDER_SERVICE_MIGRATIONS + current.TAIL)
    db = next(generator)
    try:
        h, candidate, args = seed(db, tmp_path_factory.mktemp('pre494'), '18-edge494')
        assert h.service.store.reconcile_deepline_cost(**args) == {'status':'stale'}
        ledger = rows(db,args['call_identity'])
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            before = functions(cur)
            public_rows = current._rows(cur)
            text = SQL.read_text()
            for name in ('v_preimage','v_postimage'):
                digest = text.split(name + " CONSTANT TEXT := '")[1].split("'")[0]
                with pytest.raises(db[0].Error, match='preimage differs' if name == 'v_preimage' else 'postimage differs'):
                    cur.execute(text.replace(digest, '0' * 64, 1))
                conn.rollback()
                assert functions(cur) == before
            cur.execute('GRANT EXECUTE ON FUNCTION ' + SIGNATURE + ' TO anon')
            with pytest.raises(db[0].Error, match='security shape differs'):
                cur.execute(text)
            conn.rollback()
            assert functions(cur) == before
            cur.execute('CREATE ROLE edge494_migrator NOLOGIN NOSUPERUSER INHERIT')
            cur.execute('GRANT lab_arena_owner TO edge494_migrator')
            cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO edge494_migrator')
            cur.execute('SET ROLE edge494_migrator')
            cur.execute(text)
            after = functions(cur)
            cur.execute(text)
            assert functions(cur) == after
            cur.execute('RESET ROLE')
            assert [(a[0],a[2]) for a in before] == [(a[0],a[2]) for a in after]
            assert [a[1].split('(')[0] for a,b in zip(before,after) if a[1]!=b[1]] == ['CREATE OR REPLACE FUNCTION public.lab_arena_reconcile_deepline_cost_v2']
            assert current._rows(cur) == public_rows
            for role in ('anon','authenticated','service_role','lab_arena_service'):
                cur.execute("SELECT has_function_privilege(%s,%s,'EXECUTE')", (role,SIGNATURE))
                assert cur.fetchone() == (role=='lab_arena_service',)
        assert rows(db,args['call_identity']) == ledger
        assert h.service.store.reconcile_deepline_cost(**args)['status'] == 'settled'
        assert rows(db,args['call_identity'])[:-1] == ledger
        h.service.store.close()
        yield db
    finally:
        generator.close()


class Transport:
    def __init__(self, key, fault='final'):
        self.key, self.fault, self.sent = key, fault, []

    def send(self, **request):
        from tests.lab_arena.deepline_execution_recovery_test import exact_bill
        self.sent.append(request)
        assert request['method']=='GET' and request['timeout_seconds']<=5
        if request['url'] == br.DEEPLINE_EXACT_BILLING_URL + br.quote(EDGE,safe=''):
            return br.ProviderResponse(404, {}, b'{}')
        if '/executions/by-key/' in request['url']:
            assert br.quote(self.key,safe='') in request['url']
            return br.ProviderResponse(200, {'x-deepline-idempotency-supported':'true'}, json.dumps(dict(
                requestId=NATIVE, toolId='firecrawl_scrape', executionRecovery=dict(idempotencyKey=self.key))).encode())
        assert request['url'] == br.DEEPLINE_EXACT_BILLING_URL + br.quote(NATIVE,safe='')
        if self.fault=='404':
            return br.ProviderResponse(404, {}, b'{}')
        return br.ProviderResponse(200, {}, json.dumps(exact_bill(request_id=NATIVE, credits='0.02',delta='-0.02')).encode())


def test_broker_404_stays_pending_then_appends_exact_final_bill_once(database,tmp_path):
    h, candidate, args = seed(database,tmp_path,'19-edge494')
    store = h.service.store
    old = rows(database,args['call_identity'])
    transport = Transport(args['execution_key'],'404')
    broker = _broker(store,transport)
    assert broker.reconcile_deepline_cost(candidate) == {'status':'pending'}
    assert rows(database,args['call_identity']) == old
    transport.fault='final'
    result = broker.reconcile_deepline_cost(candidate)
    assert result['status']=='settled' and result['actual_microusd']==2000
    after = rows(database,args['call_identity'])
    assert after[:-1] == old and after[-1]['entry_kind']=='settlement'
    assert after[-1]['entry_doc']['deepline_request_id']==EDGE
    assert after[-1]['terminal_response']['provider_cost']['request_id']==NATIVE
    assert broker.reconcile_deepline_cost(candidate)['idempotent'] is True
    assert rows(database,args['call_identity']) == after
    store.close()


@pytest.mark.parametrize('fault', ['native', 'malformed_edge', 'unkeyed', 'request', 'key', 'tool', 'fingerprint', 'run', 'round', 'entry'])
def test_wrong_binding_never_appends_a_settlement(database,tmp_path,fault):
    h, candidate, args = seed(database,tmp_path,'20-edge494-'+fault,
        retained='opaque_real_native' if fault=='native' else ('iad1::bad' if fault=='malformed_edge' else EDGE),
        keyed=fault!='unkeyed')
    if fault in ('request','key','tool','fingerprint','run','round','entry'):
        k,v = {'request':('request_id','iad1::other-1789361973860-8634524a6f3a'),
               'key':('execution_key','arena:'+'f'*64), 'tool':('operation','other_tool'),
               'fingerprint':('credential_fingerprint','sha256:'+'c'*64), 'run':('run_id','foreign-run'),
               'round':('round_id','arena-foreign-round'), 'entry':('uncertain_entry_id',candidate['uncertain_entry_id']+1)}[fault]
        args[k]=v
    old = rows(database,candidate['call_identity'])
    try:
        assert h.service.store.reconcile_deepline_cost(**args)['status']=='stale'
    except ArenaStoreError as error:
        assert fault in ('key','round'), str(error)
    assert rows(database,candidate['call_identity'])==old
    h.service.store.close()


@pytest.mark.parametrize('second_native', [NATIVE, EDGE])
def test_simultaneous_late_actual_bills_append_one_settlement(database,tmp_path,second_native):
    h,candidate,args=seed(database,tmp_path,'21-edge494-'+second_native)
    old=rows(database,args['call_identity'])
    barrier=threading.Barrier(2)
    def settle(native):
        from lab_arena.store import ArenaStore,PsycopgTransport
        store=ArenaStore(PsycopgTransport(lambda:database[0].connect(**database[1])))
        try:
            barrier.wait(timeout=5)
            return store.reconcile_deepline_cost(**dict(args,recovered_request_id=native))
        finally:
            store.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(settle,[NATIVE,second_native]))
    assert sorted(r['status'] for r in results)==(['settled','settled'] if second_native==NATIVE else ['conflict','settled'])
    assert sum(r.get('idempotent',False) for r in results)==(1 if second_native==NATIVE else 0)
    after=rows(database,args['call_identity'])
    assert after[:-1]==old and after[-1]['amount_microusd']==2000
    assert h.service.store.reconcile_deepline_cost(**dict(args,recovered_request_id='different_native'))['status']=='conflict'
    assert rows(database,args['call_identity'])==after
    h.service.store.close()
