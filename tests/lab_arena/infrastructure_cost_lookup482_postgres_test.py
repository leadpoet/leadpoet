"""The indexed uncertainty proof preserves decisions and recorded expenses."""
import hashlib
import json
from pathlib import Path
import time

import pytest

from tests.lab_arena import infrastructure_retry_cost478_postgres_test as prior

ROOT = Path(__file__).parents[2]
SQL = ROOT / 'scripts/482-lab-arena-indexed-infrastructure-cost-proof.sql'
SIGNATURE = 'public.lab_arena__execute_infrastructure_cost_exempt_v1(public.lab_arena_runs)'
OLD_NAME = 'lab_arena__execute_infrastructure_cost_exempt_482_control'
PRE = '0f862bf2fdccab3c15a8d863120d7c4e290391f03691910c7487d92aadd456a8'
POST = '295bace433f093c67e2bc28b76b0c2bf5ed1f2fe1d39b609446e4a19bcd07bdb'


def definition(cur):
    cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (SIGNATURE,))
    return cur.fetchone()[0]


def security(cur):
    cur.execute('SELECT pg_get_userbyid(proowner),proacl::text,prosecdef,provolatile,proconfig '
                'FROM pg_proc WHERE oid=%s::regprocedure', (SIGNATURE,))
    return cur.fetchone()


def rows(cur):
    result = []
    for table, order in (('lab_arena_rounds', 'round_id'),
                         ('lab_arena_runs', 'run_id'),
                         ('lab_arena_ledger', 'entry_id')):
        cur.execute(f'SELECT md5(COALESCE(jsonb_agg(to_jsonb(t) ORDER BY {order})::text,\'null\')) '
                    f'FROM public.{table} t')
        result.append(cur.fetchone()[0])
    return tuple(result)


@pytest.fixture(scope='module')
def database():
    gen = prior.database.__wrapped__()
    db = next(gen)
    try:
        with db[0].connect(**db[1]) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                old = definition(cur)
                assert hashlib.sha256(old.encode()).hexdigest() == PRE
                prior.seed(conn)
                accepted = prior.prior.claim(conn)
                prior.prior.pay(conn, accepted, amount=750000)
                assert prior.prior.finish(conn, accepted, 'accepted', token='a')['status'] == 'accepted'
                before_rows = rows(cur)
                cur.execute(old.replace('lab_arena__execute_infrastructure_cost_exempt_v1(',
                                        OLD_NAME + '(', 1))
                cur.execute(f'ALTER FUNCTION public.{OLD_NAME}(public.lab_arena_runs) OWNER TO lab_arena_owner')
                cur.execute(f'REVOKE ALL ON FUNCTION public.{OLD_NAME}(public.lab_arena_runs) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service')
                # Exercise the hosted migration role and hostile defaults.
                cur.execute('CREATE ROLE arena_cost_lookup_migrator NOLOGIN NOSUPERUSER INHERIT')
                cur.execute('GRANT lab_arena_owner TO arena_cost_lookup_migrator')
                cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO arena_cost_lookup_migrator WITH GRANT OPTION')
                cur.execute('ALTER DEFAULT PRIVILEGES FOR ROLE arena_cost_lookup_migrator IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO anon,authenticated,service_role')
                cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
                acl = cur.fetchone()[0]
                before = security(cur)
                cur.execute('SET ROLE arena_cost_lookup_migrator')
                cur.execute('COMMIT')
                for text, error in ((SQL.read_text().replace(PRE, '0' * 64, 1), 'preimage differs'),
                                    (SQL.read_text().replace(POST, '0' * 64, 1), 'readback differs')):
                    with pytest.raises(db[0].Error, match=error):
                        cur.execute(text)
                    cur.execute('ROLLBACK')
                    assert definition(cur) == old and security(cur) == before
                    assert rows(cur) == before_rows
                for _ in range(2):
                    cur.execute(SQL.read_text())
                    assert hashlib.sha256(definition(cur).encode()).hexdigest() == POST
                    assert security(cur) == before
                    assert rows(cur) == before_rows
                    cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
                    assert cur.fetchone()[0] == acl
                cur.execute('RESET ROLE')
        yield db
    finally:
        gen.close()


def proof(conn, run_id):
    with conn.cursor() as cur:
        cur.execute(f'SELECT public.{OLD_NAME}(r), '
                    'public.lab_arena__execute_infrastructure_cost_exempt_v1(r) '
                    'FROM public.lab_arena_runs r WHERE run_id=%s', (run_id,))
        return cur.fetchone()


def late_failure(conn, *, fee=25):
    prior.seed(conn)
    lease = prior.prior.claim(conn)
    prior.prior.pay(conn, lease, amount=700000)
    assert prior.prior.reserve(conn, lease, call='f')['status'] == 'reserved'
    prior.prior.rpc(conn, 'lab_arena_mark_dispatched(%s,%s,%s)',
                    (lease['run_id'], 'sha256:' + 'a' * 64, 'sha256:' + 'f' * 64))
    prior.prior.rpc(conn, 'lab_arena_mark_uncertain(%s,%s,%s,%s::jsonb,4500)',
                    (lease['run_id'], 'sha256:' + 'a' * 64, 'sha256:' + 'f' * 64,
                     json.dumps({'reason': 'transport_failure', 'call_succeeded': False,
                                 'provider_status': 502})))
    prior.provider_finish(conn, lease)
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO public.lab_arena_ledger
          (entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,
           provider,operation_id,funding_source,amount_microusd,entry_doc,terminal_response)
          SELECT 'settlement',miner_hotkey,round_id,submission_id,run_id,stage,call_identity,
            provider,operation_id,funding_source,%s,
            jsonb_build_object('deepline_delayed_reconciliation',true,
              'reconciled_uncertainty_entry_id',entry_id),
            '{"status":200,"call_succeeded":false}'::jsonb
          FROM public.lab_arena_ledger WHERE run_id=%s AND entry_kind='uncertain'""",
                    (fee, lease['run_id']))
    conn.commit()
    return lease


@pytest.mark.parametrize('fee', [0, 25, 900000])
def test_same_valid_proof_and_actual_fee(database, fee):
    with database[0].connect(**database[1]) as conn:
        lease = late_failure(conn, fee=fee)
        with conn.cursor() as cur:
            before = rows(cur)
        assert proof(conn, lease['run_id']) == (True, True)
        state = prior.state(conn)
        assert state['settled_microusd'] == 700000 + fee
        assert state['excluded_infrastructure_settled_microusd'] == 700000 + fee
        with conn.cursor() as cur:
            assert rows(cur) == before


@pytest.mark.parametrize('status', [400, 401, 402, 403, 404, 422, 429, 500, 502, 503, 504])
def test_same_direct_failure_decisions(database, status):
    with database[0].connect(**database[1]) as conn:
        prior.seed(conn)
        lease = prior.prior.claim(conn)
        prior.prior.pay(conn, lease, amount=700000)
        prior.failed_call(conn, lease, status=status, amount=25)
        prior.provider_finish(conn, lease)
        expected = status == 429 or status >= 500
        assert proof(conn, lease['run_id']) == (expected, expected)
        assert prior.state(conn)['settled_microusd'] == 700025


@pytest.mark.parametrize('change', [
    {'run_id': None}, {'round_id': None}, {'kind': 'score'},
    {'status': 'accepted'}, {'output_ref': 'accepted-output'},
    {'champion_restart_required': True}, {'result_doc': None},
])
def test_same_null_or_ineligible_head_decisions(database, change):
    with database[0].connect(**database[1]) as conn:
        lease = late_failure(conn)
        with conn.cursor() as cur:
            cur.execute(f'SELECT public.{OLD_NAME}(candidate), '
                        'public.lab_arena__execute_infrastructure_cost_exempt_v1(candidate) '
                        'FROM (SELECT jsonb_populate_record(NULL::public.lab_arena_runs, '
                        'to_jsonb(r)||%s::jsonb) candidate FROM public.lab_arena_runs r '
                        'WHERE run_id=%s) scoped', (json.dumps(change), lease['run_id']))
            old, new = cur.fetchone()
            assert old == new and new is not True


def test_downstream_budget_retry_quota_and_eligibility_keep_policy(database):
    # Real RPCs, costs and leases continue through the installed 482 helper.
    prior.test_host_loss_retains_actual_bill_and_releases_all_competition_cost(database, 778569)
    prior.test_late_failure_fee_keeps_original_transient_proof_and_exact_cost(database)
    prior.test_host_cost_credit_never_discards_unresolved_success(database)
    prior.test_absent_opt_in_preserves_old_published_cost_policy(database)


@pytest.fixture
def current_database():
    from tests.lab_arena.lab_arena_pg_harness import current_service_database
    for db in current_service_database():
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            # Later service prerequisites retain the exact indexed482 helper.
            assert hashlib.sha256(definition(cur).encode()).hexdigest() == POST
        yield db


def test_real_execution_judge_and_publication_use_indexed_helper(current_database, tmp_path):
    from tests.lab_arena.infrastructure_retry_cost478_lifecycle_test import (
        test_both_roles_keep_cost_credit_through_real_judge_and_publication,
    )
    test_both_roles_keep_cost_credit_through_real_judge_and_publication(current_database, tmp_path)


# Every original/head binding stays enforced, including nullable columns.
@pytest.mark.parametrize('field,value', [
    ('miner_hotkey', 'wrong-owner'), ('round_id', 'wrong-round'),
    ('submission_id', 'wrong-submission'), ('run_id', 'wrong-run'), ('stage', 1),
    ('call_identity', 'sha256:' + 'c' * 64), ('provider', 'openrouter'),
    ('operation_id', 'wrong-operation'), ('funding_source', 'host'),
] + [(field, None) for field in ('round_id', 'submission_id', 'run_id', 'stage',
                                'call_identity', 'provider', 'operation_id', 'funding_source')])
def test_same_wrong_and_null_scope_rejection(database, field, value):
    with database[0].connect(**database[1]) as conn:
        lease = late_failure(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute(f"UPDATE public.lab_arena_ledger SET {field}=%s WHERE run_id=%s AND entry_kind='uncertain'",
                        (value, lease['run_id']))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert proof(conn, lease['run_id']) == (False, False)


@pytest.mark.parametrize('field,value', [
    ('reason', 'model_error'), ('call', {'reason': 'transport_failure', 'call_succeeded': True}),
    ('call', {'reason': 'missing_provider_cost', 'provider_status': 402, 'call_succeeded': False}),
    ('call', {'reason': 'transport_failure', 'call_succeeded': False,
              'account_failure_evidence': {'provider_status': 402}}),
])
def test_same_noninfrastructure_receipt_rejection(database, field, value):
    with database[0].connect(**database[1]) as conn:
        lease = late_failure(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,%s,%s::jsonb) "
                        "WHERE run_id=%s AND entry_kind='uncertain'",
                        ([field], json.dumps(value), lease['run_id']))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert proof(conn, lease['run_id']) == (False, False)


@pytest.mark.parametrize('receipt', ['not-a-number', '9' * 100, None])
def test_malformed_receipt_stays_false_without_numeric_cast(database, receipt):
    with database[0].connect(**database[1]) as conn:
        lease = late_failure(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE public.lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,"
                        "'{reconciled_uncertainty_entry_id}',%s::jsonb) "
                        "WHERE run_id=%s AND call_identity=%s AND entry_kind='settlement'",
                        (json.dumps(receipt), lease['run_id'], 'sha256:' + 'f' * 64))
            cur.execute('SET LOCAL session_replication_role=origin')
        conn.commit()
        assert proof(conn, lease['run_id']) == (False, False)


def test_accepted_rows_unchanged_and_security_drift_rejected(database):
    with database[0].connect(**database[1]) as conn:
        prior.seed(conn)
        lease = prior.prior.claim(conn)
        prior.prior.pay(conn, lease, amount=750000)
        assert prior.prior.finish(conn, lease, 'accepted', token='a')['status'] == 'accepted'
        assert proof(conn, lease['run_id']) == (False, False)
        with conn.cursor() as cur:
            before = rows(cur)
            cur.execute('GRANT EXECUTE ON FUNCTION ' + SIGNATURE + ' TO lab_arena_service')
        conn.commit()
        with conn.cursor() as cur:
            with pytest.raises(database[0].Error, match='security shape differs'):
                cur.execute(SQL.read_text())
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute('REVOKE EXECUTE ON FUNCTION ' + SIGNATURE + ' FROM lab_arena_service')
            cur.execute(SQL.read_text())
            assert rows(cur) == before


def _walk(plan):
    yield plan
    for child in plan.get('Plans', []):
        yield from _walk(child)


def test_production_cardinality_lookup_is_indexed_and_bounded(database):
    with database[0].connect(**database[1]) as conn:
        lease = late_failure(conn)
        with conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            # A live run has hundreds of heads; the database has tens of
            # thousands of unrelated uncertain calls. No new index is used.
            cur.execute("""INSERT INTO public.lab_arena_ledger
              (entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,
               provider,operation_id,funding_source,amount_microusd,entry_doc)
              SELECT 'uncertain',miner_hotkey,'unrelated',submission_id,NULL,NULL,
                'sha256:'||lpad(to_hex(g),64,'0'),'deepline','deepline.execute','miner_key',0,
                '{"reason":"worker_reported","call":{"reason":"transport_failure","call_succeeded":false}}'::jsonb
              FROM public.lab_arena_runs r CROSS JOIN generate_series(1,60000) g
              WHERE r.run_id=%s""", (lease['run_id'],))
            cur.execute("""INSERT INTO public.lab_arena_ledger
              (entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,call_identity,
               provider,operation_id,funding_source,amount_microusd,entry_doc)
              SELECT 'uncertain',miner_hotkey,round_id,submission_id,run_id,stage,
                'sha256:'||lpad(to_hex(g+100000),64,'0'),'deepline','deepline.execute','miner_key',0,
                '{"reason":"worker_reported","call":{"reason":"transport_failure","call_succeeded":false}}'::jsonb
              FROM public.lab_arena_runs r CROSS JOIN generate_series(1,333) g
              WHERE r.run_id=%s""", (lease['run_id'],))
            assert cur.rowcount == 333
            cur.execute('SET LOCAL session_replication_role=origin')
            cur.execute('ANALYZE public.lab_arena_ledger')
            cur.execute("""CREATE TEMP TABLE cost_proof_heads_482 AS
              SELECT g AS sequence,r.run_id,r.round_id,r.submission_id,r.miner_hotkey,r.stage,
                'sha256:'||repeat('f',64) AS call_identity,'deepline'::text AS provider,
                'deepline.execute'::text AS operation_id,'miner_key'::text AS funding_source,
                (g+100000000)::text AS receipt_id
              FROM public.lab_arena_runs r CROSS JOIN generate_series(1,333) g
              WHERE r.run_id=%s""", (lease['run_id'],))
            cur.execute('ANALYZE cost_proof_heads_482')
        conn.commit()
        # Force all 333 independent original-receipt probes. The production
        # EXISTS may stop early; negative/wrong-scope heads must also be fast.
        query = """SELECT count(*) FROM cost_proof_heads_482 head WHERE sequence=0 OR EXISTS (
          SELECT 1 FROM public.lab_arena_ledger original
          WHERE original.entry_id::text=head.receipt_id
            AND original.entry_kind='uncertain'
            AND (original.run_id,original.round_id,original.submission_id,original.miner_hotkey,
              original.stage,original.call_identity,original.provider,original.operation_id,original.funding_source)
              IS NOT DISTINCT FROM (head.run_id,head.round_id,head.submission_id,head.miner_hotkey,
              head.stage,head.call_identity,head.provider,head.operation_id,head.funding_source)
        )"""
        indexed = query.replace("WHERE original.entry_id", "WHERE original.call_identity=head.call_identity AND original.run_id=head.run_id AND original.entry_id")
        with conn.cursor() as cur:
            before = rows(cur)
            cur.execute('SET jit=off')
            cur.execute('SHOW statement_timeout')
            previous_timeout = cur.fetchone()[0]
            cur.execute('EXPLAIN (FORMAT JSON) ' + query)
            old_plan = cur.fetchone()[0][0]['Plan']
            cur.execute('EXPLAIN (FORMAT JSON) ' + indexed)
            new_plan = cur.fetchone()[0][0]['Plan']
            original = [p for p in _walk(new_plan) if p.get('Alias') == 'original']
            assert original and all('Index Cond' in p and 'call_identity' in p['Index Cond'] for p in original)
            assert new_plan['Total Cost'] < old_plan['Total Cost'] / 100
            cur.execute("SET statement_timeout='2s'")
            started = time.monotonic()
            try:
                cur.execute(query)
                assert cur.fetchone()[0] == 0
                old_seconds = time.monotonic() - started
            except database[0].errors.QueryCanceled:
                old_seconds = 2.0
                conn.rollback()
            cur.execute("SET statement_timeout='2s'")
            started = time.monotonic()
            cur.execute(indexed)
            assert cur.fetchone()[0] == 0
            new_seconds = time.monotonic() - started
            assert new_seconds < 1 and new_seconds * 5 < old_seconds
            cur.execute("SELECT set_config('statement_timeout',%s,FALSE)", (previous_timeout,))
            assert proof(conn, lease['run_id']) == (True, True)
            assert rows(cur) == before
            print(f'lookup482 old_cost={old_plan["Total Cost"]} new_cost={new_plan["Total Cost"]} '
                  f'old_seconds={old_seconds:.6f} new_seconds={new_seconds:.6f}')
