"""Scope existing candidate CTEs; preserve exact closed billing and authority."""
from pathlib import Path
import hashlib
import json
import re

import pytest

from tests.lab_arena import closed_provider_round_membership507_postgres_test as old
from tests.lab_arena import closed_provider_candidate_query_postgres_test as prior
from tests.lab_arena.flat_per_icp_cost500_postgres_test import catalog, rows

ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / 'scripts/557-lab-arena-closed-provider-scoped-candidates.sql'
SIG = old.SIGNATURE
PRE = 'e8076fea79b802cbe4d91da25cd53e560059e609cbee419dad337b9240a879f1'
POST = '629daf05a1f6aa4415450d3e39b69186ffa1ad121182189ca295b71939abd289'


def definition(cur, signature=SIG):
    cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
    return cur.fetchone()[0]


def old_source(source):
    sql = MIGRATION.read_text()
    for label in ('runs557', 'uncertainties557'):
        before = re.search(r'\$old_'+label+r'\$(.*?)\$old_'+label+r'\$', sql, re.S)[1]
        after = re.search(r'\$new_'+label+r'\$(.*?)\$new_'+label+r'\$', sql, re.S)[1]
        assert source.count(after) == 1
        source = source.replace(after, before, 1)
    source = source.replace('lab_arena_closed_provider_scoped_candidates_557',
                            'lab_arena_closed_provider_round_array_507', 1)
    assert hashlib.sha256(source.encode()).hexdigest() == PRE
    return source


@pytest.fixture(scope='module')
def database():
    gen = old.database.__wrapped__()
    db = next(gen)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            assert hashlib.sha256(definition(cur).encode()).hexdigest() == PRE
            # The independent oracle is exact deployed507, including its
            # complete current wrapper, rather than a simplified SQL rewrite.
            for name in prior.NAMES:
                body = definition(cur, 'public.'+name+prior.SUFFIX)
                body = body.replace('FUNCTION public.'+name+'(',
                                    'FUNCTION public.test_before_'+name+'(', 1)
                if name == prior.NAMES[1]:
                    body = body.replace('public.'+prior.NAMES[0]+'(',
                                        'public.test_before_'+prior.NAMES[0]+'(')
                cur.execute(body)
                cur.execute('ALTER FUNCTION public.test_before_'+name+prior.SUFFIX+
                            ' OWNER TO lab_arena_owner')
            before_functions, before_rows = catalog(cur), rows(cur)
            before_relations = prior._catalog(cur)[1]
            cur.execute(MIGRATION.read_text())
            after_functions = catalog(cur)
            assert rows(cur) == before_rows
            assert prior._catalog(cur)[1] == before_relations
            changed = []
            for before, after in zip(before_functions, after_functions):
                assert before[0] == after[0] and before[2:] == after[2:]
                if before[1] != after[1]:
                    changed.append(before[0])
            assert len(before_functions) == len(after_functions)
            assert changed == [SIG.removeprefix('public.')]
            assert hashlib.sha256(definition(cur).encode()).hexdigest() == POST
            assert old_source(definition(cur))
            cur.execute(MIGRATION.read_text())
            assert catalog(cur) == after_functions and rows(cur) == before_rows
        yield db
    finally:
        gen.close()


# Reuse full real PostgreSQL fixtures, not replicas of the eligibility rules.
test_scope_policy_cause_and_proof_controls = old.test_scope_policy_cause_and_proof_controls
test_cursor_latest_head_and_settlement_controls = old.test_cursor_latest_head_and_settlement_controls
test_host_success_bindings = old.test_host_success_bindings
test_openrouter_independent = old.test_openrouter_independent
test_mixed_provider_global_cursor = old.test_mixed_provider_global_cursor
test_flat_policy_and_null_scope_equivalence = old.test_flat_policy_and_null_scope_equivalence
test_active_rounds_malformed_metadata_and_failed_unknowns = old.test_active_rounds_malformed_metadata_and_failed_unknowns
test_closed_bill_full_transition = old.test_closed_bill_full_transition


@pytest.mark.parametrize('installed', [False, True])
@pytest.mark.parametrize('drift', ['source', 'owner', 'acl', 'security', 'volatility', 'path',
                                  'helper_dynamic', 'helper_host', 'helper_miner', 'helper_adaptation', 'helper_cost', 'target'])
def test_source_security_and_helper_guards_precede_replay(database, installed, drift):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        post = definition(cur)
        if not installed:
            cur.execute(old_source(post))
        if drift == 'source':
            cur.execute(definition(cur).replace('WITH uncertainties AS', '-- drift\n  WITH uncertainties AS', 1))
        elif drift.startswith('helper_'):
            signature = {
                'helper_dynamic': 'public.lab_arena__closed_score_dynamic_uncertainty_v1(bigint)',
                'helper_host': 'public.lab_arena__closed_host_score_success_uncertainty_v1(bigint)',
                'helper_miner': 'public.lab_arena__closed_miner_score_success_uncertainty_v1(bigint)',
                'helper_adaptation': 'public.lab_arena__closed_score_adaptation_uncertainty_v1(bigint)',
                'helper_cost': 'public.lab_arena__deepline_cost_binding_v1(jsonb,jsonb,text)',
            }[drift]
            cur.execute(definition(cur, signature).replace('AS $function$\n', 'AS $function$\n-- drift\n', 1))
        elif drift != 'target':
            cur.execute({
                'owner': 'ALTER FUNCTION '+SIG+' OWNER TO postgres',
                'acl': 'GRANT EXECUTE ON FUNCTION '+SIG+' TO anon',
                'security': 'ALTER FUNCTION '+SIG+' SECURITY INVOKER',
                'volatility': 'ALTER FUNCTION '+SIG+' VOLATILE',
                'path': 'ALTER FUNCTION '+SIG+' SET search_path=public,pg_catalog',
            }[drift])
        sql = MIGRATION.read_text()
        if drift == 'target':
            sql = sql.replace(POST, '0'*64)
        body = sql.split('DO $closed_provider_scope557$')[1].split('$closed_provider_scope557$;')[0]
        cur.execute('SAVEPOINT hostile')
        with pytest.raises(database[0].Error, match='Closed provider scope'):
            cur.execute('DO $test$'+body+'$test$')
        cur.execute('ROLLBACK TO SAVEPOINT hostile')
        conn.rollback()


def test_non_superuser_owner_member_replay_preserves_catalog_and_rows(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        post = definition(cur)
        before_functions, before_rows = catalog(cur), rows(cur)
        cur.execute(old_source(post))
        cur.execute('CREATE ROLE closed557_migrator NOLOGIN NOSUPERUSER INHERIT')
        cur.execute('GRANT lab_arena_owner TO closed557_migrator')
        cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO closed557_migrator')
        cur.execute('SET ROLE closed557_migrator')
        cur.execute('SELECT rolsuper FROM pg_roles WHERE rolname=current_user')
        assert cur.fetchone() == (False,)
        cur.execute(MIGRATION.read_text())
        cur.execute(MIGRATION.read_text())
        cur.execute('RESET ROLE')
        assert catalog(cur) == before_functions and rows(cur) == before_rows


def _nodes(node):
    yield node
    for child in node.get('Plans', []):
        yield from _nodes(child)


def test_production_shaped_generic_plan_bounds_both_ctes_before_materialization(database, tmp_path):
    """Old507 fails bounded-row/spill invariants on irrelevant historical work."""
    store, rid, candidate = prior._closed_call(database, 'scope557perf')
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("SET LOCAL statement_timeout='20s'")
            cur.execute("SET LOCAL work_mem='64kB'")
            cur.execute('SET LOCAL jit=off')
            cur.execute('SET LOCAL random_page_cost=1.1')
            cur.execute("UPDATE lab_arena_rounds SET configuration_doc=configuration_doc || "
                        "'{\"network_name\":\"perf557\"}'::jsonb WHERE round_id=%s", (rid,))
            cur.execute("SELECT attname FROM pg_attribute WHERE attrelid='lab_arena_rounds'::regclass "
                        "AND attnum>0 AND NOT attisdropped AND attgenerated='' ORDER BY attnum")
            cols = [x[0] for x in cur.fetchall()]
            replacements = {'round_id': "'arena-2026-10-20-scope557'||s", 'reward_basis_hash': 'NULL',
                            'effective_reward_epoch': 'NULL',
                            'status': "CASE WHEN s=1 THEN 'open' WHEN s=5 THEN 'cancelled' ELSE 'published' END",
                            'configuration_doc': "r.configuration_doc || CASE WHEN s=2 THEN "
                              "'{\"mode\":\"shadow\"}'::jsonb WHEN s=3 THEN "
                              "'{\"network_name\":\"other557\"}'::jsonb WHEN s=5 THEN "
                              "'{\"sourcing_cost_eligibility_policy\":\"legacy\"}'::jsonb ELSE '{}'::jsonb END"}
            cur.execute('INSERT INTO lab_arena_rounds ('+','.join(cols)+') SELECT '+
                        ','.join(replacements.get(k,'r.'+k) for k in cols)+
                        ' FROM lab_arena_rounds r CROSS JOIN generate_series(1,5) s WHERE round_id=%s', (rid,))
            cur.execute("SELECT attname FROM pg_attribute WHERE attrelid='lab_arena_runs'::regclass "
                        "AND attnum>0 AND NOT attisdropped AND attgenerated='' ORDER BY attnum")
            cols = [x[0] for x in cur.fetchall()]
            replacements = {'run_id': "'scope557-run-'||s", 'assignment_id': "'scope557-assignment-'||s",
                            'round_id': "'arena-2026-10-20-scope557'||(1+s%%5)",
                            'status': "'failed'", 'terminal_cause': "'judge_error'", 'claim_request_id': 'NULL'}
            cur.execute('INSERT INTO lab_arena_runs ('+','.join(cols)+') SELECT '+
                        ','.join(replacements.get(k,'r.'+k) for k in cols)+
                        ' FROM lab_arena_runs r CROSS JOIN generate_series(1,12000) s WHERE run_id=%s',
                        (candidate['run_id'],))
            cur.execute('SELECT max(entry_id) FROM lab_arena_ledger')
            maximum = cur.fetchone()[0]
            cur.execute("SELECT attname FROM pg_attribute WHERE attrelid='lab_arena_ledger'::regclass "
                        "AND attnum>0 AND NOT attisdropped ORDER BY attnum")
            cols = [x[0] for x in cur.fetchall()]
            n = '(((s-1)/4)+1)'
            replacements = {'entry_id': '%s+s', 'run_id': "'scope557-run-'||"+n,
                            'round_id': "'arena-2026-10-20-scope557'||(1+("+n+')%%5)',
                            'call_identity': "'sha256:'||md5(("+n+")::text)||md5(("+n+")::text)",
                            'entry_doc': "CASE WHEN (s-1)%%4=2 THEN l.entry_doc || jsonb_build_object('retained_native_body',"
                              "(SELECT string_agg(md5(("+n+")::text||':'||g::text),'') FROM generate_series(1,256) g)) "
                              "ELSE l.entry_doc END",
                            'entry_kind': "CASE (s-1)%%4 WHEN 0 THEN 'reservation' WHEN 1 THEN 'dispatch' WHEN 2 THEN 'uncertain' ELSE 'settlement' END"}
            cur.execute('INSERT INTO lab_arena_ledger SELECT '+','.join(replacements.get(k,'l.'+k) for k in cols)+
                        ' FROM lab_arena_ledger l CROSS JOIN generate_series(1,48000) s '
                        "WHERE call_identity=%s AND entry_kind='uncertain'", (maximum, candidate['call_identity']))
            for table in ('lab_arena_ledger', 'lab_arena_runs', 'lab_arena_rounds'):
                cur.execute('ANALYZE '+table)
            cur.execute('SET LOCAL plan_cache_mode=force_generic_plan')
            results, plans, counts = [], {}, {}
            for index, name in enumerate(('test_before_'+prior.NAMES[0], prior.NAMES[0])):
                cur.execute('SELECT public.'+name+"('live','perf557',71,'',0)")
                results.append(cur.fetchone()[0])
                body = definition(cur, 'public.'+name+prior.SUFFIX).split('AS $function$',1)[1].split('$function$',1)[0].strip().removesuffix(';')
                for number, key in enumerate(('p_mode','p_network_name','p_netuid','p_round_id','p_after_entry_id'),1):
                    body = body.replace(key, '$'+str(number))
                prepared = 'scoped557_'+str(index)
                cur.execute('PREPARE '+prepared+'(text,text,integer,text,bigint) AS '+body)
                cur.execute('EXPLAIN (ANALYZE,BUFFERS,FORMAT JSON,TIMING FALSE) EXECUTE '+
                            prepared+"('live','perf557',71,'',0)")
                plan = cur.fetchone()[0][0]
                plans[name] = plan
                counts[index] = {node['Subplan Name']: node['Actual Rows'] for node in _nodes(plan['Plan'])
                                 if node.get('Subplan Name') in ('CTE uncertainties','CTE eligible_runs')}
            (tmp_path/'plans.json').write_text(json.dumps(plans,indent=2))
            assert results[0] == results[1] and results[1]['status'] == 'ok'
            # These exact required invariants fail on the saved preimage.
            assert counts[0]['CTE uncertainties'] >= 12000
            assert counts[0]['CTE eligible_runs'] >= 12000
            assert counts[1] == {'CTE uncertainties': 1, 'CTE eligible_runs': 1}
            before, after = list(plans.values())
            assert before['Plan']['Temp Written Blocks'] > 0
            assert after['Plan']['Temp Written Blocks'] == 0
            assert after['Plan']['Shared Hit Blocks'] < before['Plan']['Shared Hit Blocks']
            # Wide native receipts reside in real TOAST storage; do not count
            # a tiny all-open-row fixture as proof about historical liabilities.
            cur.execute("SELECT pg_relation_size(reltoastrelid) FROM pg_class WHERE oid='lab_arena_ledger'::regclass")
            assert cur.fetchone()[0] > 10*1024*1024
            # Check the actual security-definer wrapper too, not only a body.
            assert prior._equivalent(cur, network='perf557')[1]['provider'] == 'deepline'
            conn.rollback()
    finally:
        store.close()


@pytest.mark.parametrize('mode', ['live', 'shadow'])
def test_global_null_and_explicit_scope_select_same_genuine_candidate(database, mode):
    store, rid, candidate = prior._closed_call(database, 'scope557'+mode)
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE lab_arena_rounds SET configuration_doc=configuration_doc || "
                        "jsonb_build_object('mode',%s::text,'network_name','modes557') WHERE round_id=%s", (mode,rid))
            for scope in ('', None, rid):
                for cursor in (0, None, candidate['uncertain_entry_id'], 9223372036854775807):
                    result = prior._equivalent(cur, scope, cursor, mode=mode, network='modes557')
                    assert result[0]['uncertain_entry_id'] == candidate['uncertain_entry_id']
                    assert result[1]['provider'] == 'deepline'
    finally:
        store.close()


def test_null_call_binding_cannot_qualify_any_full_helper_or_selector(database):
    store, rid, candidate = prior._closed_call(database, 'scope557null')
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            assert all(x['status'] == 'ok' for x in prior._equivalent(cur, rid))
            # Test a corrupt NULL binding even if table CHECKs normally reject
            # it. All catalog changes below stay in this disposable rollback.
            cur.execute("SELECT conname FROM pg_constraint WHERE conrelid='lab_arena_ledger'::regclass "
                        "AND contype='c' AND pg_get_constraintdef(oid) LIKE '%call_identity%'")
            for (name,) in cur.fetchall():
                assert re.fullmatch('[a-zA-Z0-9_]+', name)
                cur.execute('ALTER TABLE lab_arena_ledger DROP CONSTRAINT '+name)
            cur.execute('UPDATE lab_arena_ledger SET call_identity=NULL WHERE entry_id=%s',
                        (candidate['uncertain_entry_id'],))
            for name in ('dynamic', 'host_score_success', 'miner_score_success', 'adaptation'):
                helper = {'dynamic':'lab_arena__closed_score_dynamic_uncertainty_v1',
                          'host_score_success':'lab_arena__closed_host_score_success_uncertainty_v1',
                          'miner_score_success':'lab_arena__closed_miner_score_success_uncertainty_v1',
                          'adaptation':'lab_arena__closed_score_adaptation_uncertainty_v1'}[name]
                cur.execute('SELECT public.'+helper+'(%s)', (candidate['uncertain_entry_id'],))
                assert cur.fetchone() == (False,)
            assert prior._equivalent(cur, rid) == [{'status':'none'}]*2
            conn.rollback()
    finally:
        store.close()
