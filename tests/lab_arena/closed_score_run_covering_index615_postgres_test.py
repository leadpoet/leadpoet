"""One scalar covering index; exact 573/578 authority and selection stay intact."""

import hashlib
import json
from pathlib import Path
import re

import pytest

from tests.lab_arena import closed_provider_order578_postgres_test as prior
from tests.lab_arena import closed_provider_candidate_query_postgres_test as closed
from tests.lab_arena import closed_openrouter_billing_postgres_test as router
from tests.lab_arena.flat_per_icp_cost500_postgres_test import catalog, rows

ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / 'scripts/615-lab-arena-closed-score-run-covering-index.sql'
DDL = MIGRATION.read_text()
INDEX = 'lab_arena_runs_closed_score_covering_idx'
EXPECTED = (
    'CREATE INDEX ' + INDEX + ' ON public.lab_arena_runs USING btree '
    '(round_id, run_id) INCLUDE (status, terminal_cause, submission_id, miner_hotkey, stage) '
    "WHERE ((kind = 'score'::text) AND (status = ANY (ARRAY['accepted'::text, 'failed'::text])))"
)
SIGNATURES = ['public.' + name + closed.SUFFIX for name in closed.NAMES]
FUNCTION_HASHES = [
    'ce8691e166d24a87ff6bbf475dca8cc0eb526b6688f3af22fad6ad8b0a8b6c89',
    '91a5e8ebb464aaf2252a34f4b80e00411fdb0decac2f267d25c099c9bf2d9d17',
]


def assert_index(cur):
    cur.execute(
        'SELECT i.indisvalid,i.indisready,i.indisunique,pg_get_userbyid(c.relowner),'
        'pg_get_indexdef(c.oid) FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid '
        'WHERE c.oid=to_regclass(%s)', ('public.' + INDEX,),
    )
    assert cur.fetchone() == (True, True, False, 'lab_arena_owner', EXPECTED)


@pytest.fixture(scope='module')
def database():
    gen = prior.database.__wrapped__()
    db = next(gen)
    try:
        conn = db[0].connect(**db[1])
        conn.autocommit = True
        with conn.cursor() as cur:
            before = catalog(cur), rows(cur)
            cur.execute(DDL)
            assert_index(cur)
            cur.execute(DDL)
            assert_index(cur)
            assert (catalog(cur), rows(cur)) == before
        conn.close()
        yield db
    finally:
        gen.close()


def test_migration_is_one_concurrent_statement():
    sql = re.sub(r'--[^\n]*', '', DDL).strip()
    assert sql.count(';') == 1
    assert sql.startswith('CREATE INDEX CONCURRENTLY IF NOT EXISTS ' + INDEX)
    assert 'terminal_cause' not in sql.split('WHERE', 1)[1]


@pytest.mark.parametrize('drift', ['invalid', 'unready', 'owner', 'definition', 'narrow_predicate'])
def test_required_readback_detects_unusable_or_wrong_existing_index(database, drift):
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        assert_index(cur)
        if drift in {'invalid', 'unready'}:
            column = 'indisvalid' if drift == 'invalid' else 'indisready'
            cur.execute('UPDATE pg_index SET ' + column + '=false WHERE indexrelid=%s::regclass',
                        ('public.' + INDEX,))
        elif drift == 'owner':
            cur.execute('ALTER TABLE public.lab_arena_runs OWNER TO postgres')
        else:
            cur.execute('DROP INDEX public.' + INDEX)
            if drift == 'definition':
                cur.execute('CREATE INDEX ' + INDEX + ' ON public.lab_arena_runs(run_id)')
            else:
                cur.execute('CREATE INDEX ' + INDEX + ' ON public.lab_arena_runs(round_id,run_id) '
                    'INCLUDE(status,terminal_cause,submission_id,miner_hotkey,stage) '
                    "WHERE kind='score' AND status IN('accepted','failed') "
                    "AND terminal_cause IN('accepted','judge_error')")
        with pytest.raises(AssertionError):
            assert_index(cur)
        conn.rollback()


def test_service_keeps_wider_failed_openrouter_cause_and_cursor_scope(database):
    first = router._call(database, 'c615ok')
    second = router._call(database, 'c615bad', run_status='failed')
    try:
        with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            cur.execute("UPDATE lab_arena_runs SET terminal_cause='credential_error' WHERE run_id=%s",
                        (second[2]['run_id'],))
            cur.execute("UPDATE lab_arena_rounds SET configuration_doc=configuration_doc||"
                        "'{\"network_name\":\"cover615\"}'::jsonb WHERE round_id=ANY(%s)",
                        ([first[1], second[1]],))
            ids = [first[4]['uncertain_entry_id'], second[4]['uncertain_entry_id']]
            for scope in (None, ''):
                for cursor in (None, 0, *ids, 2**63 - 1):
                    result = prior.equivalent(cur, rid=scope, cursor=cursor, network='cover615')
                    assert result['uncertain_entry_id'] == min([i for i in ids if i > (cursor or 0)] or ids)
            assert prior.equivalent(cur, rid=second[1], network='cover615')['uncertain_entry_id'] == ids[1]
            assert prior.equivalent(cur, rid='absent615', network='cover615') == {'status': 'none'}
            assert_index(cur)
    finally:
        first[0].close()
        second[0].close()


def clone(cur, table, alias, replacements, count, where, args):
    cur.execute("SELECT attname FROM pg_attribute WHERE attrelid=%s::regclass AND attnum>0 "
                "AND NOT attisdropped AND attgenerated='' ORDER BY attnum", (table,))
    columns = [row[0] for row in cur.fetchall()]
    cur.execute('INSERT INTO ' + table + '(' + ','.join(columns) + ') SELECT '
        + ','.join(replacements.get(key, alias + '.' + key) for key in columns)
        + ' FROM ' + table + ' ' + alias + ' CROSS JOIN generate_series(1,' + str(count)
        + ') s WHERE ' + where, args)


def nodes(plan):
    yield plan
    for child in plan.get('Plans', []):
        yield from nodes(child)


def plans(cur, sources):
    result = {}
    for mode in ('force_generic_plan', 'force_custom_plan'):
        cur.execute('SET plan_cache_mode=' + mode)
        cur.execute("SET statement_timeout='20s'")
        cur.execute('SET jit=off')
        cur.execute("SET work_mem='4MB'")
        for name, source in zip(closed.NAMES, sources):
            body = source.split('AS $function$', 1)[1].split('$function$', 1)[0].strip().rstrip(';')
            for i, key in enumerate(('p_mode', 'p_network_name', 'p_netuid', 'p_round_id', 'p_after_entry_id'), 1):
                body = re.sub(r'\b' + key + r'\b', '$' + str(i), body)
            cur.execute('SET ROLE lab_arena_owner')
            cur.execute('PREPARE cover615(text,text,integer,text,bigint) AS ' + body)
            cur.execute("EXPLAIN(ANALYZE,BUFFERS,FORMAT JSON,TIMING FALSE) "
                        "EXECUTE cover615('live','cover615perf',71,'',0)")
            plan = cur.fetchone()[0][0]
            cur.execute("EXECUTE cover615('live','cover615perf',71,'',0)")
            selected = cur.fetchone()[0]
            cur.execute('DEALLOCATE cover615')
            cur.execute('RESET ROLE')
            result[(mode, name)] = {'plan': plan, 'selected': selected}
    return result


def test_wide_runs_use_index_only_under_both_plan_modes_and_partial_visibility(tmp_path):
    """Fail-before access/buffer regression, not a wall-clock guarantee.

    The separate 47K offline receipt measures representative 98.8 percent
    visibility. Keep this regression smaller and preserve the exact real helpers.
    """
    gen = prior.database.__wrapped__()
    database = next(gen)
    pg, dsn = database
    store = deepline_store = None
    try:
        store, rid, run, identity, candidate = router._call(database, 'cover615perf')
        deepline_store, deepline_rid, _ = closed._closed_call(database, 'c615perf')
        conn = pg.connect(**dsn)
        conn.autocommit = True
        cur = conn.cursor()
        sources = [prior.definition(cur, signature) for signature in SIGNATURES]
        assert [hashlib.sha256(s.encode()).hexdigest() for s in sources] == FUNCTION_HASHES
        cur.execute('SET session_replication_role=replica')
        cur.execute("UPDATE lab_arena_rounds SET configuration_doc=configuration_doc||"
                    "'{\"network_name\":\"cover615perf\"}'::jsonb WHERE round_id=ANY(%s)", ([rid, deepline_rid],))
        clone(cur, 'lab_arena_rounds', 'r', {
            'round_id': "'arena-2026-10-20-cover615'||s", 'reward_basis_hash': 'NULL',
            'effective_reward_epoch': 'NULL',
            'status': "CASE WHEN s%%2=0 THEN 'cancelled' ELSE 'published' END",
        }, 64, 'r.round_id=%s', (rid,))
        clone(cur, 'lab_arena_runs', 'r', {
            'run_id': "'cover615-run-'||lpad(s::text,6,'0')",
            'assignment_id': "'cover615-assignment-'||s", 'claim_request_id': 'NULL',
            'round_id': "'arena-2026-10-20-cover615'||(1+(s-1)%%64)",
            'kind': "CASE WHEN s%%100<47 THEN 'score' ELSE 'execute' END",
            'status': "CASE WHEN s%%10<5 THEN 'accepted' ELSE 'failed' END",
            'terminal_cause': "CASE WHEN s%%10<5 THEN 'accepted' ELSE 'credential_error' END",
        }, 12000, 'r.run_id=%s', (run['run_id'],))
        cur.execute('SELECT max(entry_id) FROM lab_arena_ledger')
        maximum = cur.fetchone()[0]
        run_number = '(1+(s-1)%%12000)'
        clone(cur, 'lab_arena_ledger', 'l', {
            'entry_id': str(maximum) + '+s',
            'run_id': "'cover615-run-'||lpad(" + run_number + "::text,6,'0')",
            'round_id': "'arena-2026-10-20-cover615'||(1+((" + run_number + ')-1)%%64)',
            'call_identity': "'sha256:'||md5('cover615-'||s::text)||md5('cover615-'||s::text)",
            'entry_doc': "'{}'::jsonb",
        }, 20000, 'l.entry_id=%s', (candidate['uncertain_entry_id'],))
        cur.execute('RESET session_replication_role')
        for table in ('lab_arena_runs', 'lab_arena_rounds', 'lab_arena_ledger'):
            cur.execute('VACUUM(ANALYZE) ' + table)
        before_catalog = catalog(cur)
        cur.execute('SELECT count(*),max(entry_id),sum(entry_id),sum(amount_microusd) FROM lab_arena_ledger')
        before_ledger = cur.fetchone()
        reports = {}
        for visibility in ('vacuumed', 'partial'):
            if visibility == 'partial':
                cur.execute('SET session_replication_role=replica')
                cur.execute("UPDATE lab_arena_runs SET result_doc=COALESCE(result_doc,'{}')||"
                    "'{\"index615_visibility_probe\":true}'::jsonb "
                    "WHERE run_id>='cover615-run-000001' AND run_id<='cover615-run-000064'")
                cur.execute('RESET session_replication_role')
                cur.execute('ANALYZE lab_arena_runs')
            base = plans(cur, sources)
            cur.execute(DDL)
            assert_index(cur)
            indexed = plans(cur, sources)
            cur.execute("SELECT relpages,relallvisible FROM pg_class WHERE oid='lab_arena_runs'::regclass")
            visibility_pages = cur.fetchone()
            if visibility == 'partial':
                assert visibility_pages[1] < visibility_pages[0]
            for key in base:
                assert base[key]['selected'] == indexed[key]['selected']
                after_nodes = [node for node in nodes(indexed[key]['plan']['Plan'])
                               if node.get('Relation Name') == 'lab_arena_runs']
                assert any(node.get('Index Name') == INDEX and node['Node Type'] == 'Index Only Scan'
                           for node in after_nodes)
                before_nodes = [node for node in nodes(base[key]['plan']['Plan'])
                                if node.get('Relation Name') == 'lab_arena_runs']
                assert all(node.get('Index Name') != INDEX for node in before_nodes)
                run_buffers = lambda values: sum(n.get('Shared Hit Blocks', 0) + n.get('Shared Read Blocks', 0)
                                                for n in values)
                assert run_buffers(after_nodes) < run_buffers(before_nodes)
                reports['|'.join((visibility,) + key)] = {
                    'before': base[key], 'after': indexed[key], 'visibility_pages': visibility_pages,
                    'before_run_buffers': run_buffers(before_nodes), 'after_run_buffers': run_buffers(after_nodes),
                }
            assert catalog(cur) == before_catalog
            cur.execute('SELECT count(*),max(entry_id),sum(entry_id),sum(amount_microusd) FROM lab_arena_ledger')
            assert cur.fetchone() == before_ledger
            cur.execute('DROP INDEX public.' + INDEX)
        (tmp_path / 'closed-score-cover615-plans.json').write_text(json.dumps(reports))
        cur.close()
        conn.close()
    finally:
        if store is not None:
            store.close()
        if deepline_store is not None:
            deepline_store.close()
        gen.close()
