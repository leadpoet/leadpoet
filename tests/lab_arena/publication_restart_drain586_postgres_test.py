"""The saved publication commit can finish under an automatic restart drain."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.lab_arena import restore_recorded_results538_postgres_test as prior
from tests.lab_arena import test_lab_arena_service_round as flows
from tests.lab_arena.baseline_scored_first_postgres_test import (
    _baseline_first_judge, _enable_verified_proxy_runtime,
)
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.test_integrity_round import IntegrityHarness
from lab_arena import scoring
from lab_arena.store import ArenaStoreError

SQL = Path(__file__).parents[2] / 'scripts/586-lab-arena-publication-restart-drain.sql'
SIGNATURE = 'public.lab_arena_operator_hold_transition_guard_v1()'
BEFORE = 'a1180771aa74f37a92fedd26d35a6e3fadc6fa474a50eda5da08df26b5e00268'
AFTER = '754b81efbc162250f062f03b920a732d4439ce0ecc6aae590edae5c30218dffc'


@pytest.fixture(scope='module')
def database():
    gen = prior.database.__wrapped__()
    db = next(gen)
    try:
        yield db
    finally:
        gen.close()


def _definition(cur):
    cur.execute('SELECT pg_get_functiondef(%s::regprocedure),to_jsonb(p)-\'prosrc\' '
                'FROM pg_proc p WHERE p.oid=%s::regprocedure', (SIGNATURE, SIGNATURE))
    return cur.fetchone()


def test_saved_publication_commit_and_preserved_holds(database, tmp_path, monkeypatch):
    pg, dsn = database
    harness = IntegrityHarness(lambda: pg.connect(**dsn), tmp_path,
                               challengers=['Control'], runners=['alpha'])
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        execution_sequence_from='2000-01-01T00:00:00Z',
        company_judgment_from='2000-01-01T00:00:00Z',
        per_icp_cost_policy=True, rewards_enabled=False, benchmark_icp_count=2)
    harness.service.config.daily_icp_source = lambda **kwargs: {
        'status': 'ready', 'set_id': kwargs['set_id'], 'icps': daily_icps()[:2]}
    _enable_verified_proxy_runtime(harness)
    monkeypatch.setattr(flows, 'deterministic_scorer', _baseline_first_judge)
    monkeypatch.setitem(scoring.DEFAULT_JUDGE_MODELS, 'role_batch_check',
                        'google/gemini-2.5-flash')
    original_prices = flows.price_table
    monkeypatch.setattr(flows, 'price_table', lambda: original_prices(
        (*flows.PRICED_MODELS, 'google/gemini-2.5-flash')))
    harness.sandbox.empty_outputs.update((flavor, position)
        for flavor in ('Control', 'PublicBaseline') for position in (0, 1))
    assert flows._start_round(harness, day=22, epoch=62517) == 2
    harness.clock.advance_to(harness.schedule()['stage_1_start'])
    harness.advance_until('scored', runners=1)
    round_id = harness.round_id
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        before = _definition(cur)
        assert hashlib.sha256(before[0].encode()).hexdigest() == BEFORE
        cur.execute("SELECT pg_get_functiondef('public.lab_arena_operator_hold_score_guard_v1()'::regprocedure)")
        score_guard_before = cur.fetchone()[0]
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=false,"
                    "guard_commitment='sha256:'||repeat('a',64),"
                    "owner_commitment='sha256:'||repeat('b',64),"
                    "guard_generation=702,guard_expires_at=clock_timestamp()+interval '1 hour',"
                    "candidate_commit=repeat('c',40),restart_scope='gateway',restart_phase='draining' "
                    "WHERE singleton")
        conn.commit()
        with pytest.raises(ArenaStoreError, match='lab_arena_round_progression_paused'):
            harness.service.publish(round_id)
        conn.rollback()
        cur.execute('SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton')
        guard_before = cur.fetchone()[0]
        cur.execute(SQL.read_text())
        after = _definition(cur)
        assert hashlib.sha256(after[0].encode()).hexdigest() == AFTER
        assert after[1] == before[1]
        cur.execute("SELECT pg_get_functiondef('public.lab_arena_operator_hold_score_guard_v1()'::regprocedure)")
        assert cur.fetchone()[0] == score_guard_before
        cur.execute('SELECT public.lab_arena_operator_hold_active_v1()')
        assert cur.fetchone() == (True,)
        cur.execute(SQL.read_text())
        assert _definition(cur) == after
        result = harness.service.publish(round_id)
        assert result['status'] == 'ok'
        cur.execute('SELECT status,publication_doc,published_at FROM public.lab_arena_rounds WHERE round_id=%s', (round_id,))
        status, document, published_at = cur.fetchone()
        assert status == 'published' and document['round_id'] == round_id and published_at
        cur.execute('SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton')
        assert cur.fetchone()[0] == guard_before

        # The publication exception does not bypass the independent baseline
        # readiness check on the same production RPC.
        cur.execute('BEGIN')
        missing = 'arena-2099-10-28-incomplete586'
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("INSERT INTO public.lab_arena_rounds"
                    "(round_id,status,configuration_doc,participants,finalists,benchmark_ref,rewards_enabled) "
                    "VALUES(%s,'scored','{}','[]',%s::jsonb,'benchmark',false)",
                    (missing, json.dumps(document['finalists'])))
        cur.execute('SET LOCAL session_replication_role=origin')
        cur.execute('SAVEPOINT missing_baseline')
        publication = dict(document, round_id=missing)
        patch = {'publication_doc': publication, 'published_at': publication['published_at']}
        with pytest.raises(pg.Error, match='lab_arena_final_baseline_incomplete'):
            cur.execute('SELECT public.lab_arena_transition_round(%s,%s,%s,%s::jsonb)',
                        (missing, 'scored', 'published', json.dumps(patch)))
        cur.execute('ROLLBACK TO SAVEPOINT missing_baseline')

        # Exercise the same security-definer trigger on a small disposable row.
        # This isolates the hold decision from unrelated publication validators.
        cur.execute('CREATE TEMP TABLE hold_probe586(round_id text,status text)')
        cur.execute('CREATE TRIGGER hold_probe586 BEFORE UPDATE OF status ON hold_probe586 '
                    'FOR EACH ROW EXECUTE FUNCTION '
                    'public.lab_arena_operator_hold_transition_guard_v1()')
        for old, new in [('committed', 'stage1'), ('stage2_judged', 'scored'),
                         ('scored', 'cancelled')]:
            cur.execute('TRUNCATE hold_probe586')
            cur.execute('INSERT INTO hold_probe586 VALUES(%s,%s)', (missing, old))
            cur.execute('SAVEPOINT blocked_progression')
            with pytest.raises(pg.Error, match='lab_arena_round_progression_paused'):
                cur.execute('UPDATE hold_probe586 SET status=%s', (new,))
            cur.execute('ROLLBACK TO SAVEPOINT blocked_progression')

        cur.execute('CREATE TEMP TABLE score_probe586('
                    'kind text,round_id text,run_id text,per_icp_score numeric,'
                    'qualification_doc jsonb,updated_at timestamptz)')
        cur.execute('CREATE TRIGGER score_probe586 BEFORE UPDATE OF per_icp_score '
                    'ON score_probe586 FOR EACH ROW EXECUTE FUNCTION '
                    'public.lab_arena_operator_hold_score_guard_v1()')
        cur.execute("INSERT INTO score_probe586 VALUES('execute',%s,'probe',1,NULL,now())",
                    (missing,))
        cur.execute('SAVEPOINT blocked_score')
        with pytest.raises(pg.Error, match='lab_arena_round_progression_paused'):
            cur.execute('UPDATE score_probe586 SET per_icp_score=2')
        cur.execute('ROLLBACK TO SAVEPOINT blocked_score')

        cur.execute('TRUNCATE hold_probe586')
        cur.execute("INSERT INTO hold_probe586 VALUES(%s,'scored')", (missing,))
        cur.execute("UPDATE public.lab_arena_restart_claim_control SET operator_paused=true "
                    "WHERE singleton")
        cur.execute('SAVEPOINT explicit_pause')
        with pytest.raises(pg.Error, match='lab_arena_round_progression_paused'):
            cur.execute("UPDATE hold_probe586 SET status='published'")
        cur.execute('ROLLBACK TO SAVEPOINT explicit_pause')
        cur.execute('UPDATE public.lab_arena_restart_claim_control SET operator_paused=false '
                    'WHERE singleton')
        cur.execute('SAVEPOINT missing_control')
        cur.execute('DELETE FROM public.lab_arena_restart_claim_control WHERE singleton')
        with pytest.raises(pg.Error, match='lab_arena_round_progression_paused'):
            cur.execute("UPDATE hold_probe586 SET status='published'")
        cur.execute('ROLLBACK TO SAVEPOINT missing_control')
        cur.execute('SELECT to_jsonb(c) FROM public.lab_arena_restart_claim_control c WHERE singleton')
        assert cur.fetchone()[0] == guard_before

        # A changed function body or privilege must make replay fail closed.
        cur.execute('SAVEPOINT function_drift')
        cur.execute(after[0].replace('  v_operator_paused BOOLEAN;',
                                     '  v_operator_paused BOOLEAN;\n  -- unreviewed change', 1))
        with pytest.raises(pg.Error, match='Arena publication586 preimage differs'):
            cur.execute(SQL.read_text())
        cur.execute('ROLLBACK TO SAVEPOINT function_drift')
        assert _definition(cur) == after
        cur.execute('SAVEPOINT privilege_drift')
        cur.execute('GRANT EXECUTE ON FUNCTION ' + SIGNATURE + ' TO anon')
        with pytest.raises(pg.Error, match='Arena publication586 security or trigger shape differs'):
            cur.execute(SQL.read_text())
        cur.execute('ROLLBACK TO SAVEPOINT privilege_drift')
        assert _definition(cur) == after
