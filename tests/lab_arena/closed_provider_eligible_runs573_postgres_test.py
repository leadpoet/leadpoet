"""Materialize eligible score runs once without changing any closed bill."""
from pathlib import Path
import hashlib
import json
import re

import pytest

from tests.lab_arena import closed_provider_score_scope560_postgres_test as prior
from tests.lab_arena import closed_provider_candidate_query_postgres_test as old
from tests.lab_arena.flat_per_icp_cost500_postgres_test import catalog, rows

ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / 'scripts/573-lab-arena-closed-provider-eligible-run-materialization.sql'
SIG = prior.SIG
PRE = prior.POST
POST = 'ce8691e166d24a87ff6bbf475dca8cc0eb526b6688f3af22fad6ad8b0a8b6c89'
OUTER = 'public.' + old.NAMES[1] + old.SUFFIX
OUTER_SHA = '89a779452595cbab4537fec883dfaa0191eb327244a311421d297c73fecf1476'


def definition(cur, signature=SIG):
    return prior.definition(cur, signature)


def old_source(source):
    sql = MIGRATION.read_text()
    before = re.search(r'\$old_ctes573\$(.*?)\$old_ctes573\$', sql, re.S)[1]
    after = re.search(r'\$new_ctes573\$(.*?)\$new_ctes573\$', sql, re.S)[1]
    assert source.count(after) == 1
    source = source.replace(after, before, 1)
    assert hashlib.sha256(source.encode()).hexdigest() == PRE
    return source


@pytest.fixture(scope='module')
def database():
    gen = prior.database.__wrapped__()
    db = next(gen)
    try:
        conn = db[0].connect(**db[1])
        conn.autocommit = True
        with conn.cursor() as cur:
            # The existing production index is part of both plans, not a new
            # optimization in this migration.
            cur.execute((ROOT / 'scripts/559-lab-arena-deepline-uncertain-round-entry-index.sql').read_text())
            assert hashlib.sha256(definition(cur).encode()).hexdigest() == PRE
            assert hashlib.sha256(definition(cur, OUTER).encode()).hexdigest() == OUTER_SHA
            # Replace the historical fixture oracle with exact deployed560.
            # The outer oracle must call that same N-1 inner implementation.
            for name in old.NAMES:
                source = definition(cur, 'public.' + name + old.SUFFIX)
                source = source.replace('FUNCTION public.' + name + '(',
                                        'FUNCTION public.test_before_' + name + '(', 1)
                if name == old.NAMES[1]:
                    source = source.replace('public.' + old.NAMES[0] + '(',
                                            'public.test_before_' + old.NAMES[0] + '(')
                cur.execute(source)
                cur.execute('ALTER FUNCTION public.test_before_' + name + old.SUFFIX + ' OWNER TO lab_arena_owner')
            functions, data, relations = catalog(cur), rows(cur), old._catalog(cur)[1]
            before = definition(cur)
            cur.execute(MIGRATION.read_text())
            after = definition(cur)
            assert old_source(after) == before
            tail = '  SELECT COALESCE('
            assert before.split(tail, 1)[1] == after.split(tail, 1)[1]
            current = catalog(cur)
            assert len(functions) == len(current)
            changed = []
            for a, b in zip(functions, current):
                assert a[0] == b[0] and a[2:] == b[2:]
                if a[1] != b[1]:
                    changed.append(a[0])
            assert changed == [SIG.removeprefix('public.')]
            assert rows(cur) == data and old._catalog(cur)[1] == relations
            assert hashlib.sha256(after.encode()).hexdigest() == POST
            cur.execute(MIGRATION.read_text())
            assert catalog(cur) == current and rows(cur) == data
        conn.close()
        yield db
    finally:
        gen.close()


# Full native lifecycle fixtures exercise both the inner selector and unchanged
# provider wrapper against the exact N-1 definitions under identical snapshots.
test_scope_policy_cause_and_proof_controls = prior.test_scope_policy_cause_and_proof_controls
test_cursor_latest_head_and_settlement_controls = prior.test_cursor_latest_head_and_settlement_controls
test_host_success_bindings = prior.test_host_success_bindings
test_openrouter_independent = prior.test_openrouter_independent
test_mixed_provider_global_cursor = prior.test_mixed_provider_global_cursor
test_flat_policy_and_null_scope_equivalence = prior.test_flat_policy_and_null_scope_equivalence
test_active_rounds_malformed_metadata_and_failed_unknowns = prior.test_active_rounds_malformed_metadata_and_failed_unknowns
test_closed_bill_full_transition = prior.test_closed_bill_full_transition
test_global_null_and_explicit_scope_select_same_genuine_candidate = prior.test_global_null_and_explicit_scope_select_same_genuine_candidate
test_null_call_binding_cannot_qualify_any_full_helper_or_selector = prior.test_null_call_binding_cannot_qualify_any_full_helper_or_selector
test_newer_execution_head_still_suppresses_eligible_score_across_rounds = prior.test_newer_execution_head_still_suppresses_eligible_score_across_rounds


@pytest.mark.parametrize('installed', [False, True])
@pytest.mark.parametrize('drift', ['source', 'owner', 'acl', 'security', 'volatility', 'path',
    'helper_dynamic', 'helper_host', 'helper_miner', 'helper_adaptation', 'helper_cost', 'helper_outer', 'target'])
def test_exact_source_security_and_dependency_guards_precede_replay(database, installed, drift):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        if not installed:
            cur.execute(old_source(definition(cur)))
        if drift == 'source':
            cur.execute(definition(cur).replace('WITH eligible_runs AS' if installed else 'WITH uncertainties AS', '-- unexpected\n  WITH eligible_runs AS' if installed else '-- unexpected\n  WITH uncertainties AS', 1))
        elif drift.startswith('helper_'):
            signature = {
                'helper_dynamic': 'public.lab_arena__closed_score_dynamic_uncertainty_v1(bigint)',
                'helper_host': 'public.lab_arena__closed_host_score_success_uncertainty_v1(bigint)',
                'helper_miner': 'public.lab_arena__closed_miner_score_success_uncertainty_v1(bigint)',
                'helper_adaptation': 'public.lab_arena__closed_score_adaptation_uncertainty_v1(bigint)',
                'helper_cost': 'public.lab_arena__deepline_cost_binding_v1(jsonb,jsonb,text)',
                'helper_outer': OUTER,
            }[drift]
            cur.execute(definition(cur, signature).replace('AS $function$\n', 'AS $function$\n-- unexpected\n', 1))
        elif drift != 'target':
            cur.execute({
                'owner': 'ALTER FUNCTION ' + SIG + ' OWNER TO postgres',
                'acl': 'GRANT EXECUTE ON FUNCTION ' + SIG + ' TO anon',
                'security': 'ALTER FUNCTION ' + SIG + ' SECURITY INVOKER',
                'volatility': 'ALTER FUNCTION ' + SIG + ' VOLATILE',
                'path': 'ALTER FUNCTION ' + SIG + ' SET search_path=public,pg_catalog',
            }[drift])
        sql = MIGRATION.read_text()
        if drift == 'target':
            sql = sql.replace(POST, '0' * 64)
        body = sql.split('DO $closed_provider_eligible_runs573$')[1].split('$closed_provider_eligible_runs573$;')[0]
        cur.execute('SAVEPOINT hostile')
        with pytest.raises(database[0].Error, match='Closed provider eligible runs'):
            cur.execute('DO $test$' + body + '$test$')
        cur.execute('ROLLBACK TO SAVEPOINT hostile')
        conn.rollback()


def test_owner_member_without_owner_schema_create_replays_without_catalog_or_data_changes(database):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        functions, data = catalog(cur), rows(cur)
        cur.execute(old_source(definition(cur)))
        cur.execute('CREATE ROLE closed573_migrator NOLOGIN NOSUPERUSER INHERIT')
        cur.execute('GRANT lab_arena_owner TO closed573_migrator')
        cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO closed573_migrator')
        cur.execute('REVOKE CREATE ON SCHEMA public FROM lab_arena_owner')
        cur.execute("SELECT nspacl::text,has_schema_privilege('lab_arena_owner','public','CREATE') FROM pg_namespace WHERE nspname='public'")
        acl, owner_create = cur.fetchone()
        assert owner_create is False
        cur.execute('SET ROLE closed573_migrator')
        cur.execute('SELECT rolsuper FROM pg_roles WHERE rolname=current_user')
        assert cur.fetchone() == (False,)
        cur.execute(MIGRATION.read_text())
        cur.execute(MIGRATION.read_text())
        cur.execute('RESET ROLE')
        assert catalog(cur) == functions and rows(cur) == data
        cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
        assert cur.fetchone() == (acl,)


def test_exact_wrapper_service_acl_and_cursor_walk_same_run_and_null_run_controls(database):
    first, second = old._closed_call(database, '573samea'), old._closed_call(database, '573sameb')
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            # Two complete call lifecycles share a run. The run primary key
            # makes the new join cardinality identical to the former EXISTS.
            cur.execute('UPDATE lab_arena_ledger SET run_id=%s,round_id=%s,submission_id=l.submission_id,miner_hotkey=l.miner_hotkey,stage=l.stage FROM lab_arena_ledger l WHERE l.entry_id=%s AND lab_arena_ledger.call_identity=%s',
                (first[2]['run_id'], first[1], first[2]['uncertain_entry_id'], second[2]['call_identity']))
            cur.execute("UPDATE lab_arena_rounds SET configuration_doc=configuration_doc||'{\"network_name\":\"same573\"}'::jsonb WHERE round_id=ANY(%s)", ([first[1], second[1]],))
            ids = [first[2]['uncertain_entry_id'], second[2]['uncertain_entry_id']]
            for cursor, selected in [(None, ids[0]), (0, ids[0]), (ids[0], ids[1]), (ids[1], ids[0]), (2**63-1, ids[0])]:
                cur.execute('SET LOCAL ROLE lab_arena_service')
                result = old._equivalent(cur, network='same573', cursor=cursor)
                assert all(x['uncertain_entry_id'] == selected for x in result)
                cur.execute('RESET ROLE')
            cur.execute('SAVEPOINT null_run')
            cur.execute('UPDATE lab_arena_ledger SET run_id=NULL WHERE call_identity=%s', (second[2]['call_identity'],))
            assert all(x['uncertain_entry_id'] == ids[0] for x in old._equivalent(cur, network='same573'))
            cur.execute('ROLLBACK TO SAVEPOINT null_run')
            conn.rollback()
    finally:
        first[0].close()
        second[0].close()


def nodes(plan):
    yield plan
    for child in plan.get('Plans', []):
        yield from nodes(child)


def test_generic_plan_materializes_run_predicate_once_and_measures_narrow_scopes(database, tmp_path):
    """Real generic plans, not elapsed-time thresholds or an eligibility mock.

    Accumulated calls span 5000 runs and 33 closed rounds. Most runs execute;
    one round has no eligible score runs. Every full billing helper remains the
    real deployed function, and the original valid bill stays selectable.
    """
    store, rid, candidate = old._closed_call(database, 'perf573')
    pg, dsn = database
    conn = pg.connect(**dsn)
    try:
        cur = conn.cursor()
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("SET LOCAL statement_timeout='20s'")
        cur.execute("UPDATE lab_arena_rounds SET configuration_doc=configuration_doc||'{\"network_name\":\"perf573\"}'::jsonb WHERE round_id=%s", (rid,))

        def clone(table, alias, replacements, count, where, params):
            cur.execute("SELECT attname FROM pg_attribute WHERE attrelid=%s::regclass AND attnum>0 AND NOT attisdropped AND attgenerated='' ORDER BY attnum", (table,))
            columns = [x[0] for x in cur.fetchall()]
            cur.execute('INSERT INTO ' + table + '(' + ','.join(columns) + ') SELECT ' +
                ','.join(replacements.get(k, alias + '.' + k) for k in columns) +
                ' FROM ' + table + ' ' + alias + ' CROSS JOIN generate_series(1,' + str(count) + ') s WHERE ' + where, params)

        clone('lab_arena_rounds', 'r', {
            'round_id': "'arena-2026-10-20-perf573'||s", 'reward_basis_hash': 'NULL',
            'effective_reward_epoch': 'NULL',
            'status': "CASE WHEN s%%2=0 THEN 'cancelled' ELSE 'published' END",
        }, 33, 'r.round_id=%s', (rid,))
        clone('lab_arena_runs', 'r', {
            'run_id': "'perf573-run-'||s", 'assignment_id': "'perf573-assignment-'||s",
            'round_id': "'arena-2026-10-20-perf573'||(1+s%%33)",
            'kind': "CASE WHEN s%%100<97 OR s%%33=0 THEN 'execute' ELSE 'score' END",
            'status': "'accepted'", 'terminal_cause': "'accepted'", 'claim_request_id': 'NULL',
        }, 5000, 'r.run_id=%s', (candidate['run_id'],))
        cur.execute('SELECT max(entry_id) FROM lab_arena_ledger')
        maximum = cur.fetchone()[0]
        run_number = '(1+(s-1)%%5000)'
        clone('lab_arena_ledger', 'l', {
            'entry_id': '%s+s', 'run_id': "'perf573-run-'||" + run_number,
            'round_id': "'arena-2026-10-20-perf573'||(1+(" + run_number + ')%%33)',
            'call_identity': "'sha256:'||md5('perf573-'||s::text)||md5('perf573-'||s::text)",
            'entry_doc': "'{}'::jsonb",
        }, 20000, 'l.entry_id=%s', (maximum, candidate['uncertain_entry_id']))
        conn.commit()
        conn.autocommit = True
        for table in ('lab_arena_ledger', 'lab_arena_runs', 'lab_arena_rounds'):
            cur.execute('VACUUM(ANALYZE) ' + table)
        cur.execute('SELECT count(*),max(entry_id),sum(entry_id) FROM lab_arena_ledger')
        ledger_before = cur.fetchone()
        source_before = definition(cur)
        plans, results, summary = {}, {}, {}
        for label, source in [('before', old_source(source_before)), ('after', source_before)]:
            body = source.split('AS $function$', 1)[1].split('$function$', 1)[0].strip().rstrip(';')
            for n, key in enumerate(('p_mode', 'p_network_name', 'p_netuid', 'p_round_id', 'p_after_entry_id'), 1):
                body = body.replace(key, '$' + str(n))
            cur.execute("SET statement_timeout='15s'")
            cur.execute('SET plan_cache_mode=force_generic_plan')
            cur.execute('SET jit=off')
            cur.execute('SET ROLE lab_arena_owner')
            cur.execute('PREPARE eligible573(text,text,integer,text,bigint) AS ' + body)
            for name, scope in [('global', ''), ('pinned', rid), ('missing', 'absent573'),
                                ('no_eligible', 'arena-2026-10-20-perf5731')]:
                cur.execute('EXPLAIN(ANALYZE,BUFFERS,FORMAT JSON,TIMING FALSE) EXECUTE eligible573(%s,%s,%s,%s,%s)',
                    ('live', 'perf573', 71, scope, 0))
                plan = cur.fetchone()[0][0]
                cur.execute('EXECUTE eligible573(%s,%s,%s,%s,%s)', ('live', 'perf573', 71, scope, 0))
                result = cur.fetchone()[0]
                plans[label + '_' + name] = plan
                results[label + '_' + name] = result
                run_probes = [p for p in nodes(plan['Plan']) if p.get('Relation Name') == 'lab_arena_runs' and p.get('Alias') == 'scoped_score_run']
                ctes = {p['Subplan Name']: {'rows': p['Actual Rows'], 'loops': p['Actual Loops']}
                    for p in nodes(plan['Plan']) if p.get('Subplan Name') in ('CTE eligible_runs', 'CTE uncertainties')}
                summary[label + '_' + name] = {
                    'ms': plan['Execution Time'], 'run_probes': sum(p['Actual Loops'] for p in run_probes),
                    'hit_blocks': plan['Plan']['Shared Hit Blocks'], 'read_blocks': plan['Plan']['Shared Read Blocks'], 'ctes': ctes,
                }
            cur.execute('DEALLOCATE eligible573')
            cur.execute('RESET ROLE')
        for name in ('global', 'pinned', 'missing', 'no_eligible'):
            assert results['before_' + name] == results['after_' + name]
        assert results['after_global']['status'] == results['after_pinned']['status'] == 'ok'
        assert results['after_missing'] == results['after_no_eligible'] == {'status': 'none'}
        # This invariant fails on exact N-1: eligibility is looked up per call
        # there, versus one real run scan in the proposed materialization.
        assert summary['before_global']['run_probes'] >= 20000
        assert summary['after_global']['run_probes'] == 0
        assert summary['after_global']['ctes']['CTE eligible_runs']['loops'] == 1
        assert summary['before_global']['ctes']['CTE uncertainties']['rows'] == summary['after_global']['ctes']['CTE uncertainties']['rows']
        cur.execute('SELECT count(*),max(entry_id),sum(entry_id) FROM lab_arena_ledger')
        assert cur.fetchone() == ledger_before
        assert definition(cur) == source_before
        (tmp_path / 'eligible-run573-plans.json').write_text(json.dumps({'plans': plans, 'summary': summary, 'selected': results}, indent=2))
        print('ELIGIBLE_RUN573_PLAN_PROOF ' + json.dumps(summary), flush=True)
    finally:
        conn.close()
        store.close()
