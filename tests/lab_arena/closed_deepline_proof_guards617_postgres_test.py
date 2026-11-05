"""Late necessary-condition gates preserve full closed billing authority."""
import hashlib
import json
from pathlib import Path
import re

import pytest

from tests.lab_arena import closed_provider_order578_postgres_test as prior
from tests.lab_arena import closed_provider_candidate_query_postgres_test as old
from tests.lab_arena import closed_provider_score_scope560_postgres_test as scope
from tests.lab_arena import closed_miner_binding_gate590_postgres_test as miner
from tests.lab_arena import closed_score_run_covering_index615_postgres_test as covering

ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / 'scripts/617-lab-arena-closed-deepline-proof-guards.sql'
SQL = MIGRATION.read_text()
SIG = 'public.' + old.NAMES[0] + old.SUFFIX
PRE = 'ce8691e166d24a87ff6bbf475dca8cc0eb526b6688f3af22fad6ad8b0a8b6c89'
POST = '96edb88529b6a4e42077d638a70069adf59d64de52ffe1c6decaee36887ec557'
HELPERS = ('lab_arena__closed_score_dynamic_uncertainty_v1',
           'lab_arena__closed_host_score_success_uncertainty_v1',
           'lab_arena__closed_miner_score_success_uncertainty_v1',
           'lab_arena__closed_score_adaptation_uncertainty_v1')
DEPENDENCIES = re.findall(r"\('([^']+)',\s*'([0-9a-f]{64})',", SQL)
assert len(DEPENDENCIES) == 6
INDEXES = ('lab_arena_ledger_deepline_uncertain_round_entry_idx', covering.INDEX)
ORIGINAL = {}


def sha(source):
    return hashlib.sha256(source.encode()).hexdigest()


def definition(cur, signature=SIG):
    return prior.definition(cur, signature)


def guard_body(sql=SQL):
    return sql.split('DO $closed_deepline_proof_guards617$', 1)[1].split('$closed_deepline_proof_guards617$;', 1)[0]


def snapshot(cur):
    functions, data = miner.snapshot(cur)
    cur.execute('SELECT c.oid,pg_get_indexdef(c.oid),to_jsonb(i),c.relowner '
                'FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid '
                "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public' ORDER BY c.oid")
    return functions, data, cur.fetchall()


@pytest.fixture(scope='module')
def database():
    generator = prior.database.__wrapped__()
    db = next(generator)
    try:
        conn = db[0].connect(**db[1])
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute((ROOT / 'scripts/590-lab-arena-closed-miner-binding-gate.sql').read_text())
            cur.execute(covering.DDL)
            assert sha(definition(cur)) == PRE
            # Exact current full helpers, including the miner's post590 binder.
            for signature, expected in DEPENDENCIES:
                assert sha(definition(cur, signature)) == expected
            for name in old.NAMES:
                original = definition(cur, 'public.' + name + old.SUFFIX)
                ORIGINAL[name] = original
                oracle = original.replace('FUNCTION public.' + name + '(',
                    'FUNCTION public.test617_before_' + name + '(', 1)
                if name == old.NAMES[1]:
                    oracle = oracle.replace('public.' + old.NAMES[0] + '(',
                        'public.test617_before_' + old.NAMES[0] + '(')
                cur.execute(oracle)
                qualified = 'public.test617_before_' + name + old.SUFFIX
                cur.execute('ALTER FUNCTION ' + qualified + ' OWNER TO lab_arena_owner')
                cur.execute('REVOKE ALL ON FUNCTION ' + qualified + ' FROM PUBLIC')
                cur.execute('GRANT EXECUTE ON FUNCTION ' + qualified + ' TO lab_arena_service')
            before = snapshot(cur)
            cur.execute(SQL)
            assert sha(definition(cur)) == POST
            after = snapshot(cur)
            assert before[1:] == after[1:]
            assert [(r[0], r[2]) for r in before[0]] == [(r[0], r[2]) for r in after[0]]
            changed = [(a, b) for a, b in zip(before[0], after[0]) if a != b]
            assert len(changed) == 1 and changed[0][0][1] == ORIGINAL[old.NAMES[0]]
            # Both CTEs and the complete thin ordered OFFSET0 fence stay exact.
            marker = '    ) AS uncertainty\n'
            assert definition(cur).split(marker)[0] == ORIGINAL[old.NAMES[0]].split(marker)[0]
            cur.execute(SQL)
            assert snapshot(cur) == after
        conn.close()
        yield db
    finally:
        generator.close()


def equivalent(cur, rid='', cursor=0, mode='live', network='finney', netuid=71):
    args = (mode, network, netuid, rid, cursor)
    cur.execute('SET LOCAL ROLE lab_arena_service')
    result = []
    for name in old.NAMES:
        cur.execute('SELECT public.' + name + '(%s,%s,%s,%s,%s),'
                    'public.test617_before_' + name + '(%s,%s,%s,%s,%s)', args + args)
        after, before = cur.fetchone()
        assert after == before, (name, args, after, before)
        result.append(after)
    cur.execute('RESET ROLE')
    return result


@pytest.mark.parametrize('installed', [False, True])
@pytest.mark.parametrize('target', [SIG] + [item[0] for item in DEPENDENCIES])
@pytest.mark.parametrize('drift', ['source', 'owner', 'acl', 'security', 'volatility', 'path'])
def test_all_source_and_security_guards_precede_replay(database, installed, target, drift):
    pg, dsn = database
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        if not installed:
            cur.execute(ORIGINAL[old.NAMES[0]])
        if drift == 'source':
            cur.execute(definition(cur, target).replace('AS $function$\n', 'AS $function$\n-- drift\n', 1))
        else:
            cur.execute({
                'owner': 'ALTER FUNCTION ' + target + ' OWNER TO postgres',
                'acl': 'GRANT EXECUTE ON FUNCTION ' + target + ' TO anon',
                'security': 'ALTER FUNCTION ' + target + (' SECURITY INVOKER' if 'cost_binding' not in target else ' SECURITY DEFINER'),
                'volatility': 'ALTER FUNCTION ' + target + ' VOLATILE',
                'path': 'ALTER FUNCTION ' + target + ' SET search_path=public,pg_catalog',
            }[drift])
        before = snapshot(cur)
        cur.execute('SAVEPOINT rejected')
        with pytest.raises(pg.Error, match='Closed Deepline proof guards') as failure:
            cur.execute('DO $test$' + guard_body() + '$test$')
        assert failure.value.pgcode == '55000'
        cur.execute('ROLLBACK TO SAVEPOINT rejected')
        assert snapshot(cur) == before
        conn.rollback()


@pytest.mark.parametrize('installed', [False, True])
@pytest.mark.parametrize('index', INDEXES)
@pytest.mark.parametrize('drift', ['missing', 'invalid', 'unready', 'owner', 'definition'])
def test_both_required_index_shapes_precede_replay(database, installed, index, drift):
    pg, dsn = database
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        if not installed:
            cur.execute(ORIGINAL[old.NAMES[0]])
        if drift == 'missing':
            cur.execute('DROP INDEX public.' + index)
        elif drift in {'invalid', 'unready'}:
            column = 'indisvalid' if drift == 'invalid' else 'indisready'
            cur.execute('UPDATE pg_index SET ' + column + '=false WHERE indexrelid=%s::regclass', ('public.' + index,))
        elif drift == 'owner':
            relation = 'lab_arena_ledger' if index == INDEXES[0] else 'lab_arena_runs'
            cur.execute('ALTER TABLE public.' + relation + ' OWNER TO postgres')
        else:
            cur.execute('DROP INDEX public.' + index)
            relation = 'lab_arena_ledger' if index == INDEXES[0] else 'lab_arena_runs'
            cur.execute('CREATE INDEX ' + index + ' ON public.' + relation + '(run_id)')
        before = snapshot(cur)
        cur.execute('SAVEPOINT rejected')
        with pytest.raises(pg.Error, match='Closed Deepline proof guards index'):
            cur.execute('DO $test$' + guard_body() + '$test$')
        cur.execute('ROLLBACK TO SAVEPOINT rejected')
        assert snapshot(cur) == before
        conn.rollback()


@pytest.mark.parametrize('fault', ['postimage', 'readback', 'post_write'])
def test_full_transaction_rejects_bad_target_and_rolls_back_any_write(database, fault):
    pg, dsn = database
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute(ORIGINAL[old.NAMES[0]])
        conn.commit()
        before = snapshot(cur)
        text = SQL
        if fault == 'postimage':
            text = text.replace(POST, '0' * 64)
        elif fault == 'readback':
            anchor = "       IS DISTINCT FROM '" + POST + "'"
            assert text.count(anchor) == 1
            text = text.replace(anchor, "       IS DISTINCT FROM '" + '0' * 64 + "'", 1)
        else:
            text = text.replace('  EXECUTE updated;', "  EXECUTE updated;\n  RAISE EXCEPTION 'Closed Deepline proof guards injected post-write failure';", 1)
        try:
            with pytest.raises(pg.Error, match='Closed Deepline proof guards'):
                cur.execute(text)
            conn.rollback()
            assert snapshot(cur) == before
            assert sha(definition(cur)) == PRE
        finally:
            cur.execute(SQL)
            conn.commit()


def test_owner_member_applies_and_replays_without_owner_schema_create(database):
    pg, dsn = database
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        before = snapshot(cur)
        cur.execute(ORIGINAL[old.NAMES[0]])
        cur.execute('CREATE ROLE closed617_migrator NOLOGIN NOSUPERUSER INHERIT')
        cur.execute('GRANT lab_arena_owner TO closed617_migrator')
        cur.execute('GRANT USAGE,CREATE ON SCHEMA public TO closed617_migrator')
        cur.execute('REVOKE CREATE ON SCHEMA public FROM lab_arena_owner')
        cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
        acl = cur.fetchone()[0]
        cur.execute('SET ROLE closed617_migrator')
        cur.execute(SQL)
        cur.execute(SQL)
        cur.execute('RESET ROLE')
        assert snapshot(cur) == before
        cur.execute("SELECT nspacl::text FROM pg_namespace WHERE nspname='public'")
        assert cur.fetchone() == (acl,)
        conn.rollback()


def test_native_scope_policy_cursor_latest_head_and_mixed_provider_controls(database, monkeypatch):
    monkeypatch.setattr(old, '_equivalent', equivalent)
    for test in (old.test_miner_policies_causes_cancelled_liabilities_and_scope,
                 old.test_cursor_wrap_latest_head_and_settlement,
                 old.test_mixed_provider_cursor_keeps_one_global_order,
                 scope.test_newer_execution_head_still_suppresses_eligible_score_across_rounds):
        test(database)
    for status in ('accepted', 'failed'):
        for closed in ('published', 'cancelled'):
            old.test_openrouter_lane_remains_independent(database, status, closed)


def positive(cur, candidate, kind):
    cid, entry = candidate['call_identity'], candidate['uncertain_entry_id']
    if kind != 'dynamic':
        miner.positive(cur, cid)
    if kind == 'host_success':
        cur.execute("UPDATE lab_arena_ledger SET funding_source='host' WHERE call_identity=%s", (cid,))
        cur.execute("UPDATE lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call,deepline_job_id}','\"job-valid-1\"') WHERE entry_id=%s", (entry,))
    elif kind == 'adaptation':
        cur.execute("UPDATE lab_arena_runs SET status='accepted',terminal_cause='accepted' WHERE run_id=%s", (candidate['run_id'],))
        cur.execute("UPDATE lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{call}',(entry_doc->'call')||'{\"reason\":\"settle_failure\",\"failure_stage\":\"response_adaptation\",\"error_class\":\"CompatibilityResponseError\",\"call_succeeded\":false}'::jsonb) WHERE entry_id=%s", (entry,))


@pytest.mark.parametrize('kind', ['dynamic', 'host_success', 'miner_success', 'adaptation'])
def test_full_helper_positives_exact_json_order_and_malformed_outcome_parity(database, kind):
    store, rid, candidate = old._closed_call(database, '617' + sha(kind)[:6])
    pg, dsn = database
    try:
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            cur.execute('SET LOCAL session_replication_role=replica')
            positive(cur, candidate, kind)
            cid, entry = candidate['call_identity'], candidate['uncertain_entry_id']
            cur.execute('SELECT ' + ','.join('public.' + name + '(%s)' for name in HELPERS), (entry,) * 4)
            classes = cur.fetchone()
            assert classes[{'dynamic': 0, 'host_success': 1, 'miner_success': 2, 'adaptation': 3}[kind]] is True
            for mode in ('force_generic_plan', 'force_custom_plan'):
                cur.execute('SET LOCAL plan_cache_mode=' + mode)
                for scope in (rid, None, '', 'missing617'):
                    for cursor in (None, -1, 0, entry, 2**63 - 1):
                        equivalent(cur, rid=scope, cursor=cursor)
            if kind == 'dynamic':
                cur.execute('SAVEPOINT non_worker')
                cur.execute("UPDATE lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,'{reason}','\"not_worker_reported\"') WHERE entry_id=%s", (entry,))
                # Do not invent a worker_reported gate: leave this rejection
                # to the unchanged full helper, with exact native parity.
                equivalent(cur, rid=rid)
                cur.execute("SET LOCAL track_functions='all'")
                def dynamic_calls():
                    cur.execute('SELECT COALESCE((SELECT calls FROM pg_stat_xact_user_functions WHERE funcid=%s::regprocedure),0)',
                                ('public.' + HELPERS[0] + '(bigint)',))
                    return cur.fetchone()[0]
                before_calls = dynamic_calls()
                cur.execute('SELECT public.' + old.NAMES[0] + '(%s,%s,%s,%s,%s)', ('live', 'finney', 71, rid, 0))
                cur.fetchone()
                assert dynamic_calls() == before_calls + 1
                cur.execute('ROLLBACK TO SAVEPOINT non_worker')
            if kind in {'miner_success', 'host_success'}:
                for status in ('accepted', 'failed'):
                    cur.execute('SAVEPOINT run_state')
                    cur.execute('UPDATE lab_arena_runs SET status=%s,terminal_cause=%s WHERE run_id=%s',
                                (status, 'accepted' if status == 'accepted' else 'judge_error', candidate['run_id']))
                    for damage in ('none', 'dispatch', 'reservation_payer'):
                        cur.execute('SAVEPOINT binding')
                        if damage == 'dispatch':
                            cur.execute("DELETE FROM lab_arena_ledger WHERE call_identity=%s AND entry_kind='dispatch'", (cid,))
                        elif damage == 'reservation_payer':
                            cur.execute("UPDATE lab_arena_ledger SET funding_source=%s WHERE call_identity=%s AND entry_kind='reservation'",
                                        ('miner_key' if kind == 'host_success' else 'host', cid))
                        assert all(doc['status'] == ('ok' if damage == 'none' else 'none') for doc in equivalent(cur, rid=rid))
                        cur.execute('ROLLBACK TO SAVEPOINT binding')
                    cur.execute('ROLLBACK TO SAVEPOINT run_state')
            # SQL NULL funding and SQL NULL documents exercise UNKNOWN CASE
            # conditions, separately from JSON null and absent JSON paths.
            for field in ('funding_source', 'entry_doc'):
                cur.execute('SAVEPOINT sql_null')
                if field == 'entry_doc':
                    cur.execute('ALTER TABLE lab_arena_ledger ALTER COLUMN entry_doc DROP NOT NULL')
                cur.execute('UPDATE lab_arena_ledger SET ' + field + '=NULL WHERE entry_id=%s', (entry,))
                assert equivalent(cur, rid=rid) == [{'status': 'none'}] * 2
                cur.execute('ROLLBACK TO SAVEPOINT sql_null')
            # A JSON string is not a JSON boolean/status number. Start from a
            # genuine full-proof positive so another bad field cannot mask it.
            paths = [('call', 'call_succeeded')]
            if kind in {'host_success', 'miner_success'}:
                paths.append(('call', 'provider_status'))
            for path in paths:
                cur.execute('SAVEPOINT scalar_type')
                value = '200' if path[-1] == 'provider_status' else 'true' if kind.endswith('success') else 'false'
                cur.execute('UPDATE lab_arena_ledger SET entry_doc=jsonb_set(entry_doc,%s,%s::jsonb) WHERE entry_id=%s',
                            (list(path), json.dumps(value), entry))
                assert equivalent(cur, rid=rid) == [{'status': 'none'}] * 2
                cur.execute('ROLLBACK TO SAVEPOINT scalar_type')
            documents = [None, [], True, 'bad', {}, {'call': None}] + [
                {'reason': 'worker_reported', 'call': {'call_succeeded': value,
                 'provider_status': 'bad', 'reason': 'missing_provider_cost'}}
                for value in (None, False, True, 'false', 'true', 'not-bool', 0, 1, {}, [])]
            assert len(documents) == 16
            for document in documents:
                cur.execute('SAVEPOINT malformed')
                cur.execute('UPDATE lab_arena_ledger SET entry_doc=%s::jsonb WHERE entry_id=%s', (json.dumps(document), entry))
                outcomes = []
                for prefix in ('test617_before_', ''):
                    cur.execute('SAVEPOINT outcome')
                    try:
                        cur.execute('SELECT public.' + prefix + old.NAMES[0] + '(%s,%s,%s,%s,%s)', ('live', 'finney', 71, rid, 0))
                        outcome = ('ok', cur.fetchone()[0])
                    except pg.Error as error:
                        outcome = ('error', error.pgcode)
                    cur.execute('ROLLBACK TO SAVEPOINT outcome')
                    outcomes.append(outcome)
                assert outcomes[0] == outcomes[1], (kind, document, outcomes)
                cur.execute('ROLLBACK TO SAVEPOINT malformed')
    finally:
        store.close()


def stats(cur):
    cur.execute('SELECT funcname,calls FROM pg_stat_user_functions WHERE funcname=ANY(%s)', (list(HELPERS),))
    result = dict.fromkeys(HELPERS, 0)
    result.update(cur.fetchall())
    return result


def measured_call(cur, *, prefix, network):
    # Flush any prior EXPLAIN work before reset; it must not enter this count.
    cur.execute('SELECT pg_stat_force_next_flush()')
    cur.execute('SELECT pg_stat_reset()')
    cur.execute('SELECT public.' + prefix + old.NAMES[0] + "('live',%s,71,'',0)", (network,))
    result = cur.fetchone()[0]
    cur.execute('SELECT pg_stat_force_next_flush()')
    cur.execute('SELECT pg_stat_clear_snapshot()')
    return result, stats(cur)


def body_plan(cur, source, network, mode):
    body = source.split('AS $function$', 1)[1].rsplit('$function$', 1)[0].strip().rstrip(';')
    for position, name in enumerate(('p_mode', 'p_network_name', 'p_netuid', 'p_round_id', 'p_after_entry_id'), 1):
        body = re.sub(r'\b' + name + r'\b', '$' + str(position), body)
    cur.execute('SET plan_cache_mode=' + mode)
    cur.execute('SET ROLE lab_arena_owner')
    cur.execute('PREPARE proof617(text,text,integer,text,bigint) AS ' + body)
    cur.execute("EXPLAIN(ANALYZE,BUFFERS,FORMAT JSON,TIMING FALSE) EXECUTE proof617('live',%s,71,'',0)", (network,))
    plan = cur.fetchone()[0][0]
    cur.execute('DEALLOCATE proof617')
    cur.execute('RESET ROLE')
    return plan


def test_real_helper_calls_skip_impossible_lanes_and_keep_late_index_only_plan(database, tmp_path):
    store, rid, candidate = old._closed_call(database, '617plan')
    pg, dsn = database
    conn = pg.connect(**dsn)
    conn.autocommit = True
    count, network = 1200, 'plan617'
    try:
        with conn.cursor() as cur:
            cur.execute("SET track_functions='all'")
            cur.execute('SET jit=off')
            cur.execute("SET statement_timeout='20s'")
            cur.execute('SET session_replication_role=replica')
            cur.execute("UPDATE lab_arena_rounds SET configuration_doc=configuration_doc||jsonb_build_object('network_name',%s::text) WHERE round_id=%s", (network, rid))
            cur.execute('SELECT max(entry_id) FROM lab_arena_ledger')
            maximum = cur.fetchone()[0]
            cur.execute("SELECT entry_kind,entry_id FROM lab_arena_ledger WHERE call_identity=%s AND entry_kind IN('reservation','dispatch','uncertain')", (candidate['call_identity'],))
            templates = dict(cur.fetchall())
            for offset, kind in enumerate(('reservation', 'dispatch', 'uncertain')):
                covering.clone(cur, 'lab_arena_ledger', 'l', {
                    'entry_id': str(maximum) + '+3*s+' + str(offset),
                    'call_identity': "'sha256:'||md5('proof617-'||s::text)||md5('proof617-'||s::text)",
                }, count, 'l.entry_id=%s', (templates[kind],))
            cur.execute("SELECT setval(pg_get_serial_sequence('lab_arena_ledger','entry_id'),(SELECT max(entry_id) FROM lab_arena_ledger)+1)")
            cur.execute("UPDATE lab_arena_ledger SET entry_doc='{}'::jsonb WHERE run_id=%s AND entry_kind='uncertain'", (candidate['run_id'],))
            cur.execute('RESET session_replication_role')
            for table in ('lab_arena_ledger', 'lab_arena_runs', 'lab_arena_rounds'):
                cur.execute('VACUUM(ANALYZE) ' + table)
            preserved = snapshot(cur)
            reports = {}
            for shape in ('all_negative', 'mixed', 'first_valid', 'last_valid'):
                cur.execute('SET session_replication_role=replica')
                cur.execute("UPDATE lab_arena_ledger SET entry_doc='{}'::jsonb WHERE run_id=%s AND entry_kind='uncertain'", (candidate['run_id'],))
                expected = {'status': 'none'}
                if shape == 'mixed':
                    # Gates remain only necessary: these rows have no matching
                    # success/adaptation receipt binding and cannot pass proof.
                    cur.execute("UPDATE lab_arena_ledger SET entry_doc=jsonb_build_object('reason','worker_reported','call',CASE entry_id%%4 WHEN 0 THEN '{\"call_succeeded\":false}'::jsonb WHEN 1 THEN '{\"call_succeeded\":true,\"provider_status\":200,\"reason\":\"missing_provider_cost\"}'::jsonb WHEN 2 THEN '{\"call_succeeded\":false,\"reason\":\"settle_failure\",\"failure_stage\":\"response_adaptation\",\"error_class\":\"CompatibilityResponseError\"}'::jsonb ELSE '{}'::jsonb END) WHERE run_id=%s AND entry_kind='uncertain'", (candidate['run_id'],))
                    cur.execute("UPDATE lab_arena_ledger SET entry_doc=entry_doc||'{\"reserve_remaining_budget\":false}'::jsonb WHERE run_id=%s AND entry_kind='reservation'", (candidate['run_id'],))
                elif shape in {'first_valid', 'last_valid'}:
                    order = 'ASC' if shape == 'first_valid' else 'DESC'
                    cur.execute("SELECT call_identity,entry_id FROM lab_arena_ledger WHERE run_id=%s AND entry_kind='uncertain' ORDER BY entry_id " + order + ' LIMIT 1', (candidate['run_id'],))
                    cid, selected = cur.fetchone()
                    miner.positive(cur, cid)
                    expected = {'status': 'ok', 'uncertain_entry_id': selected, 'round_id': rid, 'run_id': candidate['run_id']}
                cur.execute('RESET session_replication_role')
                cur.execute('VACUUM(ANALYZE) lab_arena_ledger')
                for mode in ('force_generic_plan', 'force_custom_plan'):
                    cur.execute('SET plan_cache_mode=' + mode)
                    before, old_calls = measured_call(cur, prefix='test617_before_', network=network)
                    after, new_calls = measured_call(cur, prefix='', network=network)
                    assert before == after == expected
                    # Native573 invokes all doomed proof lanes. This assertion
                    # fails if617 leaves its original unguarded call sites.
                    if shape == 'all_negative':
                        assert all(value == count + 1 for value in old_calls.values())
                        assert sum(new_calls.values()) == 0
                    elif shape == 'mixed':
                        assert sum(new_calls.values()) < sum(old_calls.values()) // 2
                        assert new_calls[HELPERS[1]] == 0  # miner-funded fixture
                        assert new_calls[HELPERS[0]] > 0 and new_calls[HELPERS[2]] > 0 and new_calls[HELPERS[3]] > 0
                    else:
                        assert new_calls[HELPERS[2]] == 1
                        assert sum(new_calls.values()) == 1
                        plan = body_plan(cur, definition(cur), network, mode)
                        nodes = list(covering.nodes(plan['Plan']))
                        index_only = [node for node in nodes if node.get('Index Name') == INDEXES[0]]
                        assert index_only and all(node['Node Type'] == 'Index Only Scan' for node in index_only)
                        metadata = [node for node in nodes if node.get('Alias') == 'metadata']
                        assert len(metadata) == 1 and metadata[0]['Node Type'] == 'Index Scan'
                        assert metadata[0]['Index Name'] == 'lab_arena_ledger_pkey'
                        assert metadata[0]['Actual Loops'] == (1 if shape == 'first_valid' else count + 1)
                        reports[shape + '|' + mode] = plan
                    reports[shape + '|' + mode + '|calls'] = {'before': old_calls, 'after': new_calls}
            assert [(row[0], row[2]) for row in snapshot(cur)[0]] == [(row[0], row[2]) for row in preserved[0]]
            assert snapshot(cur)[2] == preserved[2]
            (tmp_path / 'closed-proof617-plans.json').write_text(json.dumps(reports))
    finally:
        conn.close()
        store.close()
