"""A measured baseline is not a training-eligible challenger."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import itertools
from pathlib import Path

import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.total_emissions_reward_slots681_postgres_test import (
    MIGRATION as PRIOR_MIGRATION, POLICY, old_slots, preimage_database,
)
from tests.lab_arena.reward_slots_postgres_test import _basis, _insert, _publication

MIGRATION = Path(__file__).resolve().parents[2] / 'scripts/698-lab-arena-baseline-training-reward-comparison.sql'
CHAINS = itertools.count(46980)


@pytest.fixture(scope='module')
def database():
    for pg, dsn in preimage_database():
        with pg.connect(**dsn) as c:
            with c.cursor() as q:
                q.execute(PRIOR_MIGRATION.read_text())
                q.execute(MIGRATION.read_text())
        yield pg, dsn


@pytest.fixture
def state(database):
    pg, dsn = database
    c = pg.connect(**dsn)
    c.autocommit = True
    store = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
    try:
        yield store, c, next(CHAINS)
    finally:
        store.close()
        c.close()


def publication(rid):
    p = _publication(rid, 11.55, baseline_score=28.975)
    p['final_ranking'][0]['final_score'] = 40.525
    p['final_ranking'][0]['training_data_eligibility'] = {
        'eligible': True, 'reasons': [],
        'capture': {'eligible': True, 'counts': {'complete_runs': 8, 'selected_runs': 10}},
    }
    p['final_ranking'][1].update(
        eligible=False, eligibility_reason='training_data_incomplete',
        training_data_eligibility={
            'eligible': False, 'reasons': ['training_coverage_incomplete'],
            'capture': {'eligible': False, 'counts': {'complete_runs': 6, 'selected_runs': 10}},
            'source_review': {'eligible': True, 'reasons': [], 'exemption': 'organizer_baseline'},
        },
    )
    return p


def test_baseline_capture_gap_does_not_block_verified_champion_rewards(state):
    store, c, chain = state
    rid = f'arena-2026-10-08-{chain}current'
    p = publication(rid)
    _insert(c, chain, 'current', publication=p, activated=False)
    slots = store.reward_slot_snapshot(rid, POLICY)
    assert slots[0] is None
    assert [s['round_id'] for s in slots[1:]] == [rid, rid]
    assert all(s['baseline_score'] == 28.975 and s['winner_score'] == 40.525 for s in slots[1:])
    basis, key = _basis(rid, slots, policy=POLICY)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: store.activate_reward(rid, basis, key), range(2)))
    assert sorted(r['status'] for r in results) == ['activated', 'existing']
    assert store.get_round(rid)['publication_doc'] == p
    assert store.get_round(rid)['reward_basis_doc'] == basis
    assert store.reward_slot_snapshot(rid, POLICY) == slots


@pytest.mark.parametrize('fault', [
    'winner_ineligible', 'baseline_execution_incomplete', 'baseline_cost_failure',
    'missing_training_result', 'training_not_assessed', 'missing_source_review',
    'source_review_failed', 'wrong_exemption', 'missing_baseline_score',
    'duplicate_baseline', 'wrong_identity', 'not_promoted',
])
def test_invalid_comparison_or_challenger_stays_rejected(state, fault):
    store, c, chain = state
    rid = f'arena-2026-10-08-{chain}bad'
    p = publication(rid)
    winner, baseline = p['final_ranking']
    training = baseline['training_data_eligibility']
    if fault == 'winner_ineligible': winner['eligible'] = False
    elif fault == 'baseline_execution_incomplete': baseline['eligibility_reason'] = 'execution_incomplete'
    elif fault == 'baseline_cost_failure': baseline['eligibility_reason'] = 'cost_unresolved'
    elif fault == 'missing_training_result': baseline.pop('training_data_eligibility')
    elif fault == 'training_not_assessed': training['eligible'] = None
    elif fault == 'missing_source_review': training.pop('source_review')
    elif fault == 'source_review_failed': training['source_review']['eligible'] = False
    elif fault == 'wrong_exemption': training['source_review']['exemption'] = 'miner'
    elif fault == 'missing_baseline_score': baseline['final_score'] = None
    elif fault == 'duplicate_baseline': p['final_ranking'].append(deepcopy(baseline))
    elif fault == 'wrong_identity': p['participants'][1]['miner_hotkey'] = p['participants'][0]['miner_hotkey']
    _insert(c, chain, 'bad', publication=p, activated=False, promoted=fault != 'not_promoted')
    if fault == 'not_promoted':
        assert store.reward_slot_snapshot(rid, POLICY) == [None, None, None]
    else:
        with pytest.raises(ArenaStoreError, match='current_achievement_invalid'):
            store.reward_slot_snapshot(rid, POLICY)
    assert store.get_round(rid)['reward_activated_at'] is None


def test_migration_repeat_preserves_identity_and_published_records(state):
    store, c, chain = state
    rid = _insert(c, chain, 'old', activated=False, delta=16)
    before = store.reward_slot_snapshot(rid, POLICY)
    def identity():
        with c.cursor() as q:
            q.execute("SELECT pg_get_functiondef(oid),proowner,proacl::text,prosecdef,provolatile,proconfig FROM pg_proc WHERE oid='public.lab_arena_reward_slot_snapshot(text,jsonb)'::regprocedure")
            return q.fetchone()
    original = identity()
    with c.cursor() as q: q.execute(MIGRATION.read_text())
    assert identity() == original
    assert store.reward_slot_snapshot(rid, POLICY) == before


def test_current_win_resets_only_qualifying_slots_and_carries_verified_history(state):
    store, c, chain = state
    old, _ = old_slots(store, c, chain)
    rid = f'arena-2026-10-08-{chain}current'
    p = publication(rid)
    _insert(c, chain, 'current', publication=p, activated=False)
    slots = store.reward_slot_snapshot(rid, POLICY)
    assert slots[0]['round_id'] == old[0]['round_id']
    assert slots[0]['start_epoch'] == old[0]['start_epoch']
    assert [s['start_epoch'] for s in slots[1:]] == [None, None]
    basis, key = _basis(rid, slots, policy=POLICY, epoch=1000)
    assert store.activate_reward(rid, basis, key)['status'] == 'activated'
    next_id = _insert(c, chain, 'next', day='2026-10-09', delta=4, activated=False)
    carried = store.reward_slot_snapshot(next_id, POLICY)
    assert [s['round_id'] for s in carried] == [old[0]['round_id'], rid, rid]
    assert [s['start_epoch'] for s in carried[1:]] == [1000, 1000]


@pytest.mark.parametrize('drift', ['definition', 'acl'])
def test_migration_rejects_unreviewed_definition_or_permissions(state, drift):
    _, c, _ = state
    with c.cursor() as q:
        q.execute('BEGIN')
        try:
            if drift == 'definition':
                q.execute("SELECT pg_get_functiondef('public.lab_arena_reward_slot_snapshot(text,jsonb)'::regprocedure)")
                definition = q.fetchone()[0]
                q.execute(definition.replace('DECLARE', 'DECLARE\n  -- unreviewed change', 1))
            else:
                q.execute('GRANT EXECUTE ON FUNCTION public.lab_arena_reward_slot_snapshot(text,jsonb) TO anon')
            with pytest.raises(Exception, match='698 reward slot (preimage|security) differs'):
                q.execute(MIGRATION.read_text())
        finally:
            q.execute('ROLLBACK')
