"""The large frozen catalog does not alter company judgment validation."""
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.lab_arena import unmeasured_terminal_publication547_postgres_test as prior
from tests.lab_arena import incomplete_participant_publication540_postgres_test as base540

SQL = Path(__file__).parents[2] / 'scripts/548-lab-arena-company-judgment-validation-projection.sql'
KEY = 'sha256:' + 'a' * 64
ROUND = 'arena-2099-01-01-test'


def sample():
    catalog = []
    for i in range(1800):
        seed = ''.join(hashlib.sha256(f'{i}:{j}'.encode()).hexdigest() for j in range(5))
        catalog.append(seed * 3 + seed[:64])
    cfg = {
        'integrity_policy': 'arena_integrity_v1',
        'company_judgment_policy': 'company_judgment_v1',
        'scorer_policy': {'company_judgment_policy': 'company_judgment_v1',
                          'scoring_adapter_version': 'qualification_integrity_v2'},
        'scorer_image_digest': KEY, 'scorer_image_reference': 'image@' + KEY,
        'deepline_catalog': catalog,
    }
    scope = {'cache_key': KEY, 'company_input_hash': KEY, 'round_id': ROUND,
             'network_name': 'finney', 'netuid': 71, 'evaluation_date': '2099-01-01',
             'integrity_policy': 'arena_integrity_v1',
             'company_judgment_policy': 'company_judgment_v1',
             'scorer_image_digest': KEY, 'scorer_image_reference': 'image@' + KEY}
    refs = [{'schema_version': 'leadpoet.lab_arena.company_judgment_ref.v1',
             'company_index': i, 'cache_key': KEY, 'company_input_hash': KEY,
             'scope_doc': scope} for i in range(5)]
    return cfg, refs


def verdict(cur, cfg, refs, *, projected=False):
    configuration = ("CASE WHEN jsonb_typeof(%s::jsonb)='object' "
                     "THEN %s::jsonb || '{}'::jsonb ELSE %s::jsonb END"
                     if projected else '%s::jsonb')
    params = ((json.dumps(refs), ROUND, json.dumps(cfg), json.dumps(cfg), json.dumps(cfg))
              if projected else (json.dumps(refs), ROUND, json.dumps(cfg)))
    cur.execute("SELECT public.lab_arena__company_judgment_policy_v1(r),"
                "public.lab_arena__company_judgment_refs_v1(r,%s::jsonb) "
                "FROM (SELECT jsonb_populate_record(NULL::public.lab_arena_rounds,"
                "jsonb_build_object('round_id',%s,'arena_network_name','finney',"
                f"'arena_netuid',71,'evaluation_date','2099-01-01','configuration_doc',{configuration})) r) s",
                params)
    return cur.fetchone()


def ready_matrix(cur, round_id):
    cur.execute('SELECT public.lab_arena__all_judgments_ready_v1(%s)', (round_id,))
    ready = cur.fetchone()[0]
    cur.execute("SELECT run_id,company_judgment_refs,scoring_generation_floor "
                "FROM public.lab_arena_runs WHERE round_id=%s AND kind='score' "
                "AND status='accepted' ORDER BY run_id LIMIT 1", (round_id,))
    run_id, refs, floor = cur.fetchone()
    cur.execute('SET LOCAL session_replication_role=replica')
    cur.execute("UPDATE public.lab_arena_runs SET company_judgment_refs='[{}]'::jsonb "
                "WHERE run_id=%s", (run_id,))
    cur.execute('SET LOCAL session_replication_role=origin')
    cur.execute('SELECT public.lab_arena__all_judgments_ready_v1(%s)', (round_id,))
    wrong_refs = cur.fetchone()[0]
    cur.execute('SET LOCAL session_replication_role=replica')
    cur.execute('UPDATE public.lab_arena_runs SET company_judgment_refs=%s::jsonb,'
                'scoring_generation_floor=%s WHERE run_id=%s',
                (json.dumps(refs), floor + 1, run_id))
    cur.execute('SET LOCAL session_replication_role=origin')
    cur.execute('SELECT public.lab_arena__all_judgments_ready_v1(%s)', (round_id,))
    wrong_floor = cur.fetchone()[0]
    cur.execute('SET LOCAL session_replication_role=replica')
    cur.execute('UPDATE public.lab_arena_runs SET scoring_generation_floor=%s WHERE run_id=%s',
                (floor, run_id))
    cur.execute('SET LOCAL session_replication_role=origin')
    return ready, wrong_refs, wrong_floor


def ready_ms(cur, round_id):
    cur.execute('EXPLAIN (ANALYZE, FORMAT JSON) '
                'SELECT public.lab_arena__all_judgments_ready_v1(%s)', (round_id,))
    return cur.fetchone()[0][0]['Execution Time']


def validation_ms(cur, refs, *, copied=False):
    source = ("jsonb_populate_record(p,jsonb_build_object('configuration_doc',"
              "CASE WHEN jsonb_typeof(p.configuration_doc)='object' "
              "THEN p.configuration_doc || '{}'::jsonb "
              "ELSE p.configuration_doc END))" if copied else 'p')
    cur.execute("EXPLAIN (ANALYZE, FORMAT JSON) WITH frozen AS MATERIALIZED "
                f"(SELECT {source} AS p FROM public.lab_arena_rounds p WHERE round_id=%s) "
                "SELECT bool_and(public.lab_arena__company_judgment_refs_v1(frozen.p,%s::jsonb)) "
                "FROM frozen CROSS JOIN generate_series(1,10)", (ROUND, json.dumps(refs)))
    return cur.fetchone()[0][0]['Execution Time']


def unchanged_state(cur, round_id):
    result = []
    for table in ('lab_arena_rounds', 'lab_arena_runs', 'lab_arena_ledger'):
        cur.execute(f"SELECT md5(COALESCE(jsonb_agg(to_jsonb(t) ORDER BY to_jsonb(t)::text)::text,'null')) "
                    f'FROM public.{table} t WHERE round_id=%s', (round_id,))
        result.append(cur.fetchone()[0])
    cur.execute("SELECT pg_get_userbyid(p.proowner),p.proacl::text,p.prosecdef,"
                "p.provolatile,p.proconfig,l.lanname FROM pg_proc p "
                "JOIN pg_language l ON l.oid=p.prolang WHERE p.oid="
                "'public.lab_arena__all_judgments_ready_v1(text)'::regprocedure")
    result.append(cur.fetchone())
    return result


@pytest.fixture(scope='module')
def database():
    gen = prior.database.__wrapped__()
    db = next(gen)
    try:
        yield db
    finally:
        gen.close()


@pytest.fixture(scope='module')
def completed(database, tmp_path_factory):
    pg, dsn = database
    harness = base540.IntegrityHarness(
        lambda: pg.connect(**dsn), tmp_path_factory.mktemp('validation548'),
        challengers=['Complete'], runners=['alpha'])
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        execution_sequence_from='2000-01-01T00:00:00Z',
        company_judgment_from='2000-01-01T00:00:00Z',
        per_icp_cost_policy=True, rewards_enabled=False, benchmark_icp_count=10)
    harness.service.config.daily_icp_source = lambda **kw: {
        'status': 'ready', 'set_id': kw['set_id'], 'icps': base540.daily_icps()[:10]}
    base540._enable_verified_proxy_runtime(harness)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(base540.flows, 'deterministic_scorer', base540._baseline_first_judge)
    monkey.setitem(base540.scoring.DEFAULT_JUDGE_MODELS, 'role_batch_check',
                   'google/gemini-2.5-flash')
    prices = base540._BASE_PRICE_TABLE
    monkey.setattr(base540.flows, 'price_table', lambda: prices(
        (*base540.flows.PRICED_MODELS, 'google/gemini-2.5-flash')))
    harness.sandbox.empty_outputs.update((flavor, position)
        for flavor in (*harness.challengers, 'PublicBaseline') for position in range(10))
    try:
        assert base540.flows._start_round(harness, day=22, epoch=62548) == 2
        harness.clock.advance_to(harness.schedule()['stage_1_start'])
        harness.advance_until('scored', runners=1)
        with pg.connect(**dsn) as conn, conn.cursor() as cur:
            cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s',
                        (harness.round_id,))
            row = cur.fetchone()[0]
        yield harness, row
    finally:
        monkey.undo()
        harness.service.store.close()


def test_large_frozen_configuration_preserves_validation_and_security(database, completed):
    cfg, refs = sample()
    cases = [(cfg, refs)]
    for path, value in [
        (('integrity_policy',), 'other'),
        (('company_judgment_policy',), None),
        (('scorer_policy', 'company_judgment_policy'), 'other'),
        (('scorer_policy', 'scoring_adapter_version'), None),
        (('contact_policy',), None),
        (('company_quality_policy',), None),
        (('scorer_policy', 'contact_policy'), None),
        (('scorer_policy', 'company_quality_policy'), None),
        (('scorer_policy',), []),
    ]:
        candidate = copy.deepcopy(cfg)
        node = candidate
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        cases.append((candidate, refs))
    for key, value in [('round_id', 'other'), ('network_name', 'testnet'),
                       ('netuid', 72), ('evaluation_date', 'other'),
                       ('scorer_image_digest', 'other'),
                       ('scorer_image_reference', 'other'),
                       ('company_quality_policy', None)]:
        candidate = copy.deepcopy(refs)
        candidate[0]['scope_doc'][key] = value
        cases.append((cfg, candidate))
    for bad_refs in [None, {}, refs[:1] * 6, [{**refs[0], 'company_index': 2}],
                     [{**refs[0], 'cache_key': 'bad'}]]:
        cases.append((cfg, bad_refs))
    cases += [(bad_cfg, refs) for bad_cfg in ([], 'malformed', None)]
    assert len(json.dumps(cfg)) > 1_800_000
    with database[0].connect(**database[1]) as conn, conn.cursor() as cur:
        before = [verdict(cur, c, r) for c, r in cases]
        cur.execute("SELECT p.oid::regprocedure::text,pg_get_userbyid(p.proowner),p.proacl::text,"
                    "p.prosecdef,p.provolatile,p.proconfig,l.lanname FROM pg_proc p "
                    "JOIN pg_language l ON l.oid=p.prolang WHERE p.oid IN "
                    "('public.lab_arena__company_judgment_policy_v1(public.lab_arena_rounds)'::regprocedure,"
                    "'public.lab_arena__company_judgment_refs_v1(public.lab_arena_rounds,jsonb)'::regprocedure) "
                    "ORDER BY 1")
        metadata = cur.fetchall()
        cur.execute('ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER ALL')
        cur.execute("UPDATE public.lab_arena_rounds SET configuration_doc="
                    "jsonb_set(configuration_doc,'{deepline_catalog}',%s::jsonb) "
                    "WHERE round_id=%s", (json.dumps(cfg['deepline_catalog']),
                                           completed[1]['round_id']))
        cur.execute('ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER ALL')
        # This disposable fixture deliberately corrupts immutable accepted rows
        # to exercise the read-only finality guard on both function versions.
        cur.execute('ALTER TABLE public.lab_arena_runs DISABLE TRIGGER ALL')
        live = completed[1]
        fixture_refs = copy.deepcopy(refs)
        for ref in fixture_refs:
            ref['scope_doc'].update(round_id=live['round_id'],
                                    network_name=live['arena_network_name'],
                                    netuid=live['arena_netuid'],
                                    evaluation_date=live['evaluation_date'],
                                    scorer_image_digest=live['configuration_doc']['scorer_image_digest'],
                                    scorer_image_reference=live['configuration_doc']['scorer_image_reference'])
        cur.execute("UPDATE public.lab_arena_runs SET company_judgment_refs=%s::jsonb "
                    "WHERE round_id=%s AND kind='score' AND status='accepted'",
                    (json.dumps(fixture_refs), live['round_id']))
        assert cur.rowcount >= 10
        cur.execute("SELECT count(*),count(*) FILTER (WHERE status='accepted'),"
                    "count(*) FILTER (WHERE status='accepted' AND terminal_cause='accepted' "
                    "AND per_icp_score IS NOT NULL AND output_ref IS NOT NULL) "
                    "FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute'",
                    (live['round_id'],))
        total, accepted, scored = cur.fetchone()
        assert total >= 10 and accepted == scored == total
        cur.execute("SELECT count(DISTINCT icp_position) FROM public.lab_arena_runs "
                    "WHERE round_id=%s AND kind='execute' AND status='accepted'",
                    (live['round_id'],))
        assert cur.fetchone() == (10,)
        cur.execute("SELECT count(*) FILTER (WHERE public.lab_arena__miner_credit_judgment_terminal_v1(run_id)),"
                    "count(*) FILTER (WHERE NOT public.lab_arena__miner_credit_judgment_terminal_v1(run_id)) "
                    "FROM public.lab_arena_runs WHERE round_id=%s AND kind='execute' AND status='accepted'",
                    (live['round_id'],))
        assert cur.fetchone() == (0, total)
        cur.execute("SELECT pg_column_size(configuration_doc),octet_length(configuration_doc::text),"
                    "(SELECT count(*) FROM jsonb_array_elements(participants) p "
                    "WHERE public.lab_arena__participant_incomplete_reason_v1(%s,p->>'submission_id') IS NOT NULL) "
                    "FROM public.lab_arena_rounds WHERE round_id=%s",
                    (live['round_id'], live['round_id']))
        stored_size, text_size, incomplete = cur.fetchone()
        assert 400_000 < stored_size < 1_000_000
        assert text_size > 1_800_000 and incomplete == 0
        before_ready = ready_matrix(cur, completed[1]['round_id'])
        old_ready_ms = ready_ms(cur, completed[1]['round_id'])
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("INSERT INTO public.lab_arena_rounds(round_id,configuration_doc,evaluation_date) "
                    "VALUES (%s,%s::jsonb,'2099-01-01')", (ROUND, json.dumps(cfg)))
        cur.execute('SET LOCAL session_replication_role=origin')
        old_ms = validation_ms(cur, refs)
        copied_ms = validation_ms(cur, refs, copied=True)
        assert copied_ms < old_ms / 3, (old_ms, copied_ms)
        state_before = unchanged_state(cur, completed[1]['round_id'])
        cur.execute(SQL.read_text())
        after = [verdict(cur, c, r, projected=True) for c, r in cases]
        assert before == after
        assert before[0] == (True, True)
        assert all(not result[0] or not result[1] for result in before[1:])
        cur.execute("SELECT p.oid::regprocedure::text,pg_get_userbyid(p.proowner),p.proacl::text,"
                    "p.prosecdef,p.provolatile,p.proconfig,l.lanname FROM pg_proc p "
                    "JOIN pg_language l ON l.oid=p.prolang WHERE p.oid IN "
                    "('public.lab_arena__company_judgment_policy_v1(public.lab_arena_rounds)'::regprocedure,"
                    "'public.lab_arena__company_judgment_refs_v1(public.lab_arena_rounds,jsonb)'::regprocedure) "
                    "ORDER BY 1")
        assert cur.fetchall() == metadata
        assert ready_matrix(cur, completed[1]['round_id']) == before_ready
        assert before_ready == (True, False, False)
        new_ready_ms = ready_ms(cur, completed[1]['round_id'])
        assert new_ready_ms < old_ready_ms / 3, (old_ready_ms, new_ready_ms)
        cur.execute('ALTER TABLE public.lab_arena_runs ENABLE TRIGGER ALL')
        assert unchanged_state(cur, completed[1]['round_id']) == state_before
        cur.execute(SQL.read_text())
        assert unchanged_state(cur, completed[1]['round_id']) == state_before
        cur.execute("SELECT public.lab_arena__all_judgments_ready_v1(%s)", (ROUND,))
        assert cur.fetchone() == (True,)
        print(f'large config, ten five-ref checks: {old_ms:.1f} ms full; '
              f'{copied_ms:.1f} ms copied; ten-position full guard '
              f'{old_ready_ms:.1f} ms -> {new_ready_ms:.1f} ms; '
              f'{stored_size} stored bytes')


def test_ten_icp_publication_promotion_and_rewards_after_migration(tmp_path, monkeypatch):
    gen = prior.database.__wrapped__()
    db = next(gen)
    try:
        with db[0].connect(**db[1]) as conn, conn.cursor() as cur:
            cur.execute(SQL.read_text())
        base540.test_positive_ten_icp_large_cohort_publishes_promotes_and_derives_rewards(
            db, tmp_path, monkeypatch)
    finally:
        gen.close()
