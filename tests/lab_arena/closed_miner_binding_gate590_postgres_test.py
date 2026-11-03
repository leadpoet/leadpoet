"""Exact failed-key proof gating preserves billing authority and stored history."""
import hashlib
import json
from pathlib import Path

import pytest

from tests.lab_arena import closed_provider_round_membership507_postgres_test as prior
from tests.lab_arena import closed_miner_success_score_billing_postgres_test as miner
from tests.lab_arena import closed_adaptation_score_billing457_postgres_test as adaptation
from tests.lab_arena import closed_scoring_billing_postgres_test as closed

ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / 'scripts/590-lab-arena-closed-miner-binding-gate.sql'
HELPER = 'public.lab_arena__closed_miner_score_success_uncertainty_v1(bigint)'
BINDER = 'public.lab_arena__deepline_cost_binding_v1(jsonb,jsonb,text)'
OLD = '67fcd9dc83d29491aa4608e069b6680ac652e85b93b1feac64cc01a72b1f93f8'
NEW = 'ff3aaddd632b24f70651c42530eae854a522927c291540bf34faa46f8d23597a'
TAIL = (
    '557-lab-arena-closed-provider-scoped-candidates.sql',
    '559-lab-arena-deepline-uncertain-round-entry-index.sql',
    '560-lab-arena-closed-provider-score-run-scope.sql',
    '573-lab-arena-closed-provider-eligible-run-materialization.sql',
)


def apply(cur, sql=None):
    # Preserve the exact DO block while allowing per-test transactional rollback.
    sql = MIGRATION.read_text() if sql is None else sql
    assert sql.count('\nBEGIN;\n') == 1 and sql.endswith('COMMIT;\n')
    cur.execute(sql.replace('\nBEGIN;\n', '\n', 1).removesuffix('COMMIT;\n'))


def definition(cur, signature=HELPER):
    cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
    return cur.fetchone()[0]


def snapshot(cur):
    cur.execute("SELECT p.oid,pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' "
                "FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
                "WHERE n.nspname='public' ORDER BY p.oid")
    functions = cur.fetchall()
    data = []
    for table in ('lab_arena_ledger','lab_arena_runs','lab_arena_rounds','lab_arena_submissions'):
        cur.execute('SELECT to_jsonb(t) FROM '+table+' t ORDER BY to_jsonb(t)::text')
        data.append(cur.fetchall())
    return functions, data


@pytest.fixture(scope='module')
def database():
    generator = prior.database.__wrapped__()
    try:
        db = next(generator)
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            for name in TAIL:
                cur.execute((ROOT/'scripts'/name).read_text())
            assert hashlib.sha256(definition(cur).encode()).hexdigest() == OLD
        yield db
    finally:
        generator.close()


def seed(database, label):
    return prior.router._call(database, '590'+label)


def positive(cur, cid):
    request = 'ctx-tool-'+cid[7:39]
    key = 'arena:'+cid[7:]
    fingerprint = 'sha256:'+'a'*64
    reservation = {'deepline_request_id':request,'deepline_execution_key':key,
                   'tool':'firecrawl_scrape','credential_fingerprint':fingerprint}
    receipt = {'reason':'worker_reported','call':{'reason':'missing_provider_cost',
               'call_succeeded':True,'provider_status':200,'deepline_request_id':request,
               'deepline_execution_key':key,'deepline_operation':'firecrawl_scrape',
               'credential_fingerprint':fingerprint}}
    cur.execute("UPDATE lab_arena_ledger SET provider='deepline',operation_id='scrapingdog.scrape' "
                'WHERE call_identity=%s', (cid,))
    cur.execute("UPDATE lab_arena_ledger SET entry_doc=%s::jsonb WHERE call_identity=%s "
                "AND entry_kind='reservation'", (json.dumps(reservation),cid))
    cur.execute("UPDATE lab_arena_ledger SET entry_doc=%s::jsonb WHERE call_identity=%s "
                "AND entry_kind='uncertain'", (json.dumps(receipt),cid))


def helper(cur, entry):
    cur.execute('SELECT public.lab_arena__closed_miner_score_success_uncertainty_v1(%s)', (entry,))
    return cur.fetchone()[0]


def test_exact_change_replay_security_and_all_rows(database):
    store, _, _, _, _ = seed(database, 'metadata')
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            before, data = snapshot(cur)
            original = definition(cur)
            try:
                # Exercise the committed transaction text, including replay.
                cur.execute(MIGRATION.read_text())
                assert hashlib.sha256(definition(cur).encode()).hexdigest() == NEW
                after, after_data = snapshot(cur)
                assert after_data == data
                assert [(a[0],a[2]) for a in after] == [(a[0],a[2]) for a in before]
                changed = [(a,b) for a,b in zip(before,after) if a != b]
                assert len(changed) == 1 and changed[0][0][1] == original
                cur.execute(MIGRATION.read_text())
                assert snapshot(cur) == (after, data)
                for role in ('anon','authenticated','service_role','lab_arena_service'):
                    cur.execute('SELECT has_function_privilege(%s,%s,\'EXECUTE\')', (role,HELPER))
                    assert cur.fetchone() == (False,)
            finally:
                conn.rollback()
                cur.execute(original)
                conn.commit()
    finally:
        store.close()


def test_full_transaction_failure_rolls_back_function_replacement(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        before = snapshot(cur)
        sql = MIGRATION.read_text()
        anchor = '  EXECUTE updated;'
        assert sql.count(anchor) == 1
        sql = sql.replace(anchor, anchor + "\n  RAISE EXCEPTION 'Closed miner binding gate injected post-write failure';", 1)
        with pytest.raises(database[0].Error,match='injected post-write failure'):
            cur.execute(sql)
        conn.rollback()
        assert snapshot(cur) == before
        assert hashlib.sha256(definition(cur).encode()).hexdigest() == OLD


def test_failed_execution_key_does_not_start_pure_binding_proof(database):
    store, rid, run, cid, _ = seed(database, 'calls')
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("SET LOCAL track_functions='all'")
            cur.execute("UPDATE lab_arena_ledger SET provider='deepline' WHERE call_identity=%s", (cid,))
            cur.execute("SELECT entry_id FROM lab_arena_ledger WHERE call_identity=%s AND entry_kind='uncertain'", (cid,))
            entry = cur.fetchone()[0]
            def calls():
                cur.execute('SELECT COALESCE((SELECT calls FROM pg_stat_xact_user_functions '
                            'WHERE funcid=%s::regprocedure),0)', (BINDER,))
                return cur.fetchone()[0]
            initial = calls()
            assert helper(cur, entry) is False
            old_calls = calls()
            assert old_calls > initial  # The pre-590 helper starts a doomed pure proof.
            apply(cur)
            assert helper(cur, entry) is False
            assert calls() == old_calls  # This is the fail-before performance regression.
            conn.rollback()
    finally:
        store.close()


VARIANTS = [('accepted',None,True), ('failed',None,True)]
VARIANTS += [('reservation',value,False) for value in (None,False,123,[],{},'', 'wrong')]
VARIANTS += [('uncertainty',value,False) for value in (None,False,123,[],{},'', 'wrong')]
VARIANTS += [('success',value,False) for value in (None,False,'true',1,[],{})]
VARIANTS += [(name,None,False) for name in ('dispatch_missing','dispatch_stage','source_run',
                                               'execute','funding','provider','credential','request')]


@pytest.mark.parametrize('kind,value,expected', VARIANTS)
def test_old_new_receipt_and_source_equivalence(database, kind, value, expected):
    label = hashlib.sha256(repr((kind,value)).encode()).hexdigest()[:10]
    store, rid, run, cid, _ = seed(database,label)
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            positive(cur,cid)
            cur.execute("SELECT entry_id FROM lab_arena_ledger WHERE call_identity=%s AND entry_kind='uncertain'", (cid,))
            entry = cur.fetchone()[0]
            if kind in ('reservation','uncertainty','success','credential','request'):
                reservation = kind == 'reservation'
                path = ['deepline_execution_key'] if reservation else ['call','call_succeeded' if kind=='success' else
                        'credential_fingerprint' if kind=='credential' else 'deepline_request_id' if kind=='request' else 'deepline_execution_key']
                value = 'wrong' if kind in ('credential','request') else value
                cur.execute('UPDATE lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,%s::text[],%s::jsonb) '
                            'WHERE call_identity=%s AND entry_kind=%s',
                            (path,json.dumps(value),cid,'reservation' if reservation else 'uncertain'))
            elif kind == 'failed':
                cur.execute("UPDATE lab_arena_runs SET status='failed',terminal_cause='judge_error' WHERE run_id=%s", (run['run_id'],))
            elif kind == 'dispatch_missing':
                cur.execute("DELETE FROM lab_arena_ledger WHERE call_identity=%s AND entry_kind='dispatch'", (cid,))
            elif kind == 'dispatch_stage':
                cur.execute("UPDATE lab_arena_ledger SET stage=2 WHERE call_identity=%s AND entry_kind='dispatch'", (cid,))
            elif kind == 'source_run':
                cur.execute("UPDATE lab_arena_ledger SET run_id='wrong' WHERE call_identity=%s AND entry_kind='reservation'", (cid,))
            elif kind == 'execute':
                cur.execute("UPDATE lab_arena_runs SET kind='execute' WHERE run_id=%s", (run['run_id'],))
            elif kind in ('funding','provider'):
                cur.execute('UPDATE lab_arena_ledger SET '+('funding_source' if kind=='funding' else 'provider')+'=%s WHERE call_identity=%s',
                            ('host' if kind=='funding' else 'openrouter',cid))
            old = helper(cur,entry)
            before = snapshot(cur)[1]
            apply(cur)
            assert helper(cur,entry) == old == expected
            assert snapshot(cur)[1] == before
            conn.rollback()
    finally:
        store.close()


@pytest.mark.parametrize('replay', [False,True])
@pytest.mark.parametrize('drift', ['source','owner','acl','security','binder_source','binder_acl','selector_source','target'])
def test_drift_failure_is_atomic_including_replay(database,replay,drift):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        if replay:
            apply(cur)
        cur.execute('SAVEPOINT before_drift')
        sql = MIGRATION.read_text()
        if drift.endswith('_source') or drift == 'source':
            signature = BINDER if drift=='binder_source' else prior.SIGNATURE if drift=='selector_source' else HELPER
            source = definition(cur,signature)
            cur.execute(source.replace('AS $function$', 'AS $function$\n-- source drift',1))
        elif drift == 'owner':
            cur.execute('ALTER FUNCTION '+HELPER+' OWNER TO postgres')
        elif drift.endswith('acl'):
            cur.execute('GRANT EXECUTE ON FUNCTION '+(BINDER if drift=='binder_acl' else HELPER)+' TO anon')
        elif drift == 'security':
            cur.execute('ALTER FUNCTION '+HELPER+" SET work_mem='1MB'")
        elif drift == 'target':
            if replay:
                cur.execute(definition(cur).replace('          AND reservation.entry_doc', '           AND reservation.entry_doc',1))
            else:
                sql = sql.replace("updated:=replace(definition,old_gate,new_gate);", "updated:=replace(definition,old_gate,new_gate)||E'\\n';")
        drifted = snapshot(cur)
        cur.execute('SAVEPOINT failed_migration')
        with pytest.raises(database[0].Error,match='Closed miner binding gate'):
            apply(cur,sql)
        cur.execute('ROLLBACK TO SAVEPOINT failed_migration')
        assert snapshot(cur) == drifted
        cur.execute('ROLLBACK TO SAVEPOINT before_drift')
        conn.rollback()


def test_current_selector_and_settlement_keep_exact_miner_bills(database,tmp_path,monkeypatch):
    # Exercise the real current RPC callers, including pending→settled,
    # original payer/credential checks, idempotent settlement and frozen rows.
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        apply(cur)
    try:
        miner.test_exact_miner_success_recovery_after_accepted_or_failed_score(database,tmp_path,monkeypatch)
        miner.test_miner_success_rejects_wrong_scope_and_damaged_identity(database,tmp_path)
        adaptation.test_closed_accepted_adaptation_exact_bill_preserves_published_result(database,tmp_path)
        adaptation.test_closed_adaptation_rejects_unrelated_and_damaged_binding(database,tmp_path)
        for policy in ('successful_calls_v1','successful_calls_per_icp_v1'):
            closed.test_published_late_bill_is_exact_idempotent_and_does_not_rewrite_result(database,policy)
            closed.test_fixed_unknown_is_not_admitted_to_published_mutation_path(database,policy)
    finally:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            source = definition(cur)
            old = ("      AND CASE WHEN uncertainty.entry_doc #> '{call,call_succeeded}' = 'true'::JSONB\n"
                   "          AND reservation.entry_doc ->> 'deepline_execution_key' =\n"
                   "              'arena:' || pg_catalog.substr(uncertainty.call_identity, 8)")
            restored = source.replace(old,old.split('\n')[0],1)
            assert hashlib.sha256(restored.encode()).hexdigest() == OLD
            cur.execute(restored)
