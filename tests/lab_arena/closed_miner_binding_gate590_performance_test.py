"""Current full-proof CPU stress for deliberately invalid execution keys."""
from functools import partial
import hashlib
import json
import os
import platform
import time

import pytest

from tests import postgres_migration_harness as pg_harness
from tests.lab_arena import lab_arena_pg_harness as arena_pg
from tests.lab_arena import closed_provider_candidate_query_postgres_test as prior
from tests.lab_arena import closed_openrouter_billing_postgres_test as router
from tests.lab_arena import closed_miner_binding_gate590_postgres_test as seam

ROOT = seam.ROOT
SIGNATURE = seam.prior.SIGNATURE
# Production source17.6 remains distinct from the target17.9 historical plan.
POSTGRES_IMAGE = 'postgres:17.6@sha256:00bc86618629af00d2937fdc5a5d63db3ff8450acf52f0636ec813c7f4902929'


@pytest.fixture
def plan_database():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(arena_pg, '_database_with_migrations',
                      partial(pg_harness._database_with_migrations, image=POSTGRES_IMAGE))
        generator = seam.database.__wrapped__()
        try:
            database = next(generator)
            with database[0].connect(**database[1]) as connection, connection.cursor() as cursor:
                cursor.execute(seam.MIGRATION.read_text())
            yield database
        finally:
            generator.close()


def test_invalid_execution_keys_keep_bounded_closed_proof_cpu(plan_database, tmp_path):
    """Adversarial receipts cannot become bills or consume unbounded proof CPU.

    Deliberately relabel copied OpenRouter documents as Deepline: successful
    receipt booleans are true, but required execution keys are absent. This
    preserves the original 500k stress case separately from the historical507
    benchmark's valid native broker receipts and two PG17.9 planner profiles.
    """
    database = plan_database
    pg, dsn = database
    store, rid, run, cid, _ = router._call(database, '507plan')
    try:
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("SET LOCAL statement_timeout='20s'")
            cur.execute("SET LOCAL work_mem='12MB'")
            cur.execute('SET LOCAL jit=off')
            cur.execute('SET LOCAL random_page_cost=1.1')
            # Sixty closed rounds mirror the observed production cardinality.
            cur.execute("UPDATE lab_arena_rounds SET configuration_doc=configuration_doc || "
                        "'{\"network_name\":\"plan507\"}'::jsonb WHERE round_id=%s", (rid,))
            cur.execute("SELECT attname FROM pg_attribute WHERE attrelid='lab_arena_rounds'::regclass "
                        "AND attnum>0 AND NOT attisdropped AND attgenerated='' ORDER BY attnum")
            cols = [r[0] for r in cur.fetchall()]
            repl = {'round_id':"'arena-2026-10-20-plan507'||s", 'reward_basis_hash':'NULL',
                    'effective_reward_epoch':'NULL'}
            cur.execute('INSERT INTO lab_arena_rounds ('+','.join(cols)+') SELECT '+
                        ','.join(repl.get(k,'r.'+k) for k in cols)+
                        ' FROM lab_arena_rounds r CROSS JOIN generate_series(1,59) s WHERE round_id=%s', (rid,))
            cur.execute("SELECT attname FROM pg_attribute WHERE attrelid='lab_arena_runs'::regclass "
                        "AND attnum>0 AND NOT attisdropped AND attgenerated='' ORDER BY attnum")
            cols = [r[0] for r in cur.fetchall()]
            repl = {'run_id':"repeat('r',48)||'plan507-'||s", 'assignment_id':"repeat('r',48)||'plan507-'||s",
                    'round_id':"CASE WHEN s%%60=0 THEN r.round_id ELSE 'arena-2026-10-20-plan507'||(s%%60) END",
                    'status':"'failed'", 'terminal_cause':"'provider_error'", 'claim_request_id':'NULL'}
            cur.execute('INSERT INTO lab_arena_runs ('+','.join(cols)+') SELECT '+
                        ','.join(repl.get(k,'r.'+k) for k in cols)+
                        ' FROM lab_arena_runs r CROSS JOIN generate_series(1,40000) s WHERE run_id=%s',
                        (run['run_id'],))
            cur.execute('SELECT max(entry_id) FROM lab_arena_ledger')
            maximum = cur.fetchone()[0]
            cur.execute("SELECT attname FROM pg_attribute WHERE attrelid='lab_arena_ledger'::regclass "
                        'AND attnum>0 AND NOT attisdropped ORDER BY attnum')
            cols = [r[0] for r in cur.fetchall()]
            repl = {'entry_id':'%s+s', 'run_id':"repeat('r',48)||'plan507-'||(1+((s-1)/3)%%40000)",
                    'round_id':"CASE WHEN (1+((s-1)/3)%%40000)%%60=0 THEN l.round_id ELSE "
                               "'arena-2026-10-20-plan507'||((1+((s-1)/3)%%40000)%%60) END",
                    'call_identity':"'sha256:'||md5(((s-1)/3)::text)||md5(((s-1)/3)::text)",
                    'entry_kind':"CASE WHEN s%%3=1 THEN 'reservation' WHEN s%%3=2 THEN 'dispatch' "
                                 "WHEN ((s-1)/3)%%2=0 THEN 'uncertain' ELSE 'settlement' END",
                    'provider':"CASE WHEN ((s-1)/3)%%3=0 THEN 'openrouter' ELSE 'deepline' END"}
            # Seed identical cardinality in bounded statements. Preserve the
            # measured query budget and every history/plan assertion below.
            for first in range(1, 500001, 100000):
                cur.execute('INSERT INTO lab_arena_ledger SELECT '+','.join(repl.get(k,'l.'+k) for k in cols)+
                            " FROM lab_arena_ledger l CROSS JOIN generate_series(%s,%s) s "
                            "WHERE call_identity=%s AND entry_kind='uncertain'",
                            (maximum, first, first + 99999, cid))
                assert cur.rowcount == 100000
            for table in ('lab_arena_ledger','lab_arena_runs','lab_arena_rounds'):
                cur.execute('ANALYZE '+table)
            cur.execute('SET LOCAL plan_cache_mode=force_generic_plan')
            cur.execute("SELECT current_setting('server_version'), current_setting('server_version_num')")
            server_version, server_version_num = cur.fetchone()
            results = []; plans = {'server_version': server_version,
                                   'server_version_num': server_version_num,
                                   'machine': platform.machine(), 'cpu_count': os.cpu_count()}
            cur.execute("SELECT name,setting,unit FROM pg_settings WHERE name IN "
                        "('work_mem','shared_buffers','effective_cache_size','jit','random_page_cost',"
                        "'plan_cache_mode','max_parallel_workers_per_gather','max_worker_processes') "
                        'ORDER BY name')
            plans['settings'] = cur.fetchall()
            cur.execute("SELECT relname,n_live_tup,n_mod_since_analyze,last_analyze::text "
                        "FROM pg_stat_user_tables WHERE relname IN "
                        "('lab_arena_ledger','lab_arena_runs','lab_arena_rounds') ORDER BY relname")
            plans['table_statistics'] = cur.fetchall()
            cur.execute("SELECT tablename,indexname,indexdef,index_class.reltuples,index_class.relpages,"
                        'index_catalog.indisvalid,index_catalog.indisready '
                        'FROM pg_indexes JOIN pg_class index_class ON index_class.oid='
                        "(quote_ident(schemaname)||'.'||quote_ident(indexname))::regclass "
                        'JOIN pg_index index_catalog ON index_catalog.indexrelid=index_class.oid '
                        "WHERE schemaname='public' AND tablename IN "
                        "('lab_arena_ledger','lab_arena_runs','lab_arena_rounds') ORDER BY indexname")
            plans['indexes'] = cur.fetchall()
            cur.execute('SELECT count(*) FROM lab_arena_ledger')
            plans['ledger_rows'] = cur.fetchone()[0]
            cur.execute('SELECT count(*) FROM lab_arena_runs')
            plans['run_rows'] = cur.fetchone()[0]
            cur.execute('SELECT count(*) FROM lab_arena_rounds')
            plans['round_rows'] = cur.fetchone()[0]
            plans['helper_hashes'] = {}
            for signature in (SIGNATURE,
                    'public.lab_arena__closed_score_dynamic_uncertainty_v1(bigint)',
                    'public.lab_arena__closed_host_score_success_uncertainty_v1(bigint)',
                    'public.lab_arena__closed_miner_score_success_uncertainty_v1(bigint)',
                    'public.lab_arena__closed_score_adaptation_uncertainty_v1(bigint)',
                    'public.lab_arena__deepline_cost_binding_v1(jsonb,jsonb,text)'):
                cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', (signature,))
                plans['helper_hashes'][signature] = hashlib.sha256(cur.fetchone()[0].encode()).hexdigest()
            artifact = (ROOT / 'ci-results' / 'plan590-invalid-keys.json'
                        if os.environ.get('GITHUB_ACTIONS') == 'true' else tmp_path / 'plans.json')
            def persist_plans():
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text(json.dumps(plans, indent=2))
            def measured_execute(statement, phase):
                # Save the estimated plans and all completed work even when the
                # next unchanged acceptance bound cancels its statement.
                plans['phase'] = phase
                persist_plans()
                started = time.monotonic()
                try:
                    cur.execute(statement)
                except pg.Error as error:
                    plans['failure'] = {'phase': phase, 'sqlstate': error.pgcode,
                                        'elapsed_seconds': time.monotonic() - started}
                    persist_plans()
                    print('PLAN590_INVALID_KEYS_FAILURE ' + json.dumps(plans['failure']) +
                          ' artifact=' + str(artifact), flush=True)
                    raise
                plans.setdefault('elapsed_seconds', {})[phase] = time.monotonic() - started
                persist_plans()

            names = ('test_before_'+prior.NAMES[0], prior.NAMES[0])
            # Non-executing plans must precede the timed function: a canceled
            # SQL function otherwise leaves no inspectable generic body plan.
            for index, name in enumerate(names):
                cur.execute('SELECT pg_get_functiondef(%s::regprocedure)', ('public.'+name+prior.SUFFIX,))
                body = cur.fetchone()[0].split('AS $function$',1)[1].split('$function$',1)[0].strip().removesuffix(';')
                literal = body
                for key,value in {'p_mode':"'live'",'p_network_name':"'plan507'",'p_netuid':'71',
                                  'p_round_id':"''",'p_after_entry_id':'0'}.items():
                    literal = literal.replace(key,value)
                cur.execute('EXPLAIN (FORMAT JSON, COSTS TRUE) '+literal)
                plans['literal_'+name] = cur.fetchone()[0][0]
                for number,key in enumerate(('p_mode','p_network_name','p_netuid','p_round_id','p_after_entry_id'),1):
                    body = body.replace(key,'$'+str(number))
                prepared = 'plan507_'+str(index)
                cur.execute('PREPARE '+prepared+'(text,text,integer,text,bigint) AS '+body)
                cur.execute('EXPLAIN (FORMAT JSON, COSTS TRUE) EXECUTE '+
                            prepared+"('live','plan507',71,'',0)")
                plans['estimated_generic_'+name] = cur.fetchone()[0][0]
                persist_plans()
            print('PLAN590_INVALID_KEYS_RUNTIME server_version=%s server_version_num=%s' %
                  (server_version, server_version_num), flush=True)
            for name in names:
                # Only the deliberately obsolete 60-scan reference needs a
                # larger allowance. Optimized507 retains its 20-second bound.
                cur.execute('SET LOCAL statement_timeout=%s',
                            ('60s' if name.startswith('test_before_') else '20s',))
                print('PLAN590_INVALID_KEYS_FUNCTION name=%s statement_timeout=%s' %
                      (name, '60s' if name.startswith('test_before_') else '20s'), flush=True)
                measured_execute('SELECT public.'+name+"('live','plan507',71,'',0)", 'call_'+name)
                results.append(cur.fetchone()[0])
                measured_execute('EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON, TIMING FALSE) SELECT public.'+
                                 name+"('live','plan507',71,'',0)", 'explain_'+name)
                full_plan = cur.fetchone()[0][0]
                plans[name]=full_plan
                persist_plans()
                plan = full_plan['Plan']
                if name.startswith('test_before_'):
                    before_reads = plan['Temp Read Blocks']
                else:
                    after_reads = plan['Temp Read Blocks']
            generic_reads = {}; generic_loops = {}
            def cte_loops(node):
                values = [node.get('Actual Loops',0)] if node.get('CTE Name')=='uncertainties' else []
                for child in node.get('Plans',[]):
                    values.extend(cte_loops(child))
                return values
            for index,name in enumerate(names):
                cur.execute('SET LOCAL statement_timeout=%s',
                            ('60s' if index == 0 else '20s',))
                prepared = 'plan507_'+str(index)
                measured_execute('EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON, TIMING FALSE) EXECUTE '+
                                 prepared+"('live','plan507',71,'',0)", 'generic_'+name)
                full_plan=cur.fetchone()[0][0]
                plans['generic_'+name]=full_plan
                generic_reads[index]=full_plan['Plan']['Temp Read Blocks']
                generic_loops[index]=max(cte_loops(full_plan['Plan']))
                persist_plans()
            assert generic_loops=={0:60,1:1}
            assert generic_reads[0]>0 and generic_reads[1]<generic_reads[0]/10
            assert results == [{'status':'none'}]*2
            assert before_reads > 0 and after_reads < before_reads / 10
            # Keep the history isolated from later callers of this module fixture.
            conn.rollback()
    finally:
        store.close()
