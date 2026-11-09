"""Literal681 proof: prospective ordinal carry and verified current replacement."""
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import re

import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration
from tests.lab_arena.slot_decay_postgres_test import MIGRATIONS as DECAY_MIGRATIONS, POLICY as PRE_OCT23
from tests.lab_arena.reward_slots_postgres_test import _insert, _publication

ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / 'scripts/681-lab-arena-total-emissions-reward-slots.sql'
POLICY = dict(deepcopy(PRE_OCT23), pool_percent=75, allocation_basis='total_emissions', tiers=[
    {'minimum_improvement':15,'allocation_percent':35},
    {'minimum_improvement':10,'allocation_percent':25},
    {'minimum_improvement':5,'allocation_percent':15},
])
OLD_POLICY = dict(deepcopy(PRE_OCT23), pool_percent=50, tiers=[
    {'minimum_improvement':20,'allocation_percent':40},
    {'minimum_improvement':15,'allocation_percent':30},
    {'minimum_improvement':8,'allocation_percent':30},
])
CHAINS = itertools.count(46810)


def preimage_database():
    for pg,dsn in database_with_lab_arena_migration(DECAY_MIGRATIONS + ('591-lab-arena-legacy-baseline-reward-comparison.sql',)):
        c=pg.connect(**dsn); c.autocommit=True
        with c.cursor() as cur:
            body=(ROOT/'scripts/592-arena-2026-10-29-authorized-champion.sql').read_text()
            # Production-specific592 data correction is excluded. Apply its exact
            # literal function patch so681 has the deployed function preimage.
            cur.execute(re.search(r'DO \$oct29_snapshot592\$.*?\$oct29_snapshot592\$;',body,re.S)[0])
        c.close()
        yield pg,dsn


@pytest.fixture(scope='module')
def database():
    for pg,dsn in preimage_database():
        c=pg.connect(**dsn);c.autocommit=True
        with c.cursor() as cur:cur.execute(MIGRATION.read_text())
        c.close()
        yield pg,dsn


@pytest.fixture()
def state(database):
    pg,dsn=database
    c=pg.connect(**dsn);c.autocommit=True
    store=ArenaStore(PsycopgTransport(lambda:pg.connect(**dsn)))
    try:yield store,c,next(CHAINS)
    finally:store.close();c.close()


def winner(c,chain,name,delta,*,day='2026-10-08',activated=True,epoch=None):
    rid=f'arena-{day}-{chain}{name}'
    return _insert(c,chain,name,day=day,delta=delta,activated=activated,epoch=epoch,
                   publication=_publication(rid,delta))


def seed(c,rid,slots,policy=OLD_POLICY):
    # Fixture-only installation of signed-basis columns on published source rows.
    with c.cursor() as cur:
        cur.execute('SELECT effective_reward_epoch FROM public.lab_arena_rounds WHERE round_id=%s',(rid,))
        epoch=cur.fetchone()[0]
        doc={'schema_version':'leadpoet.lab_arena.reward_basis.v2','round_id':rid,
             'effective_reward_epoch':epoch,'slot_policy':policy,'reward_slots':slots}
        cur.execute('BEGIN')
        cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute('UPDATE public.lab_arena_rounds SET reward_basis_doc=%s::jsonb WHERE round_id=%s',(json.dumps(doc),rid))
        cur.execute('COMMIT')


def signature(cur):
    cur.execute("SELECT pg_get_functiondef('public.lab_arena_reward_slot_snapshot(text,jsonb)'::regprocedure)")
    return cur.fetchone()[0]


def old_slots(store,c,chain):
    a=winner(c,chain,'top',25,day='2026-10-05',epoch=100)
    b=winner(c,chain,'mid',17,day='2026-10-06',epoch=200)
    d=winner(c,chain,'low',9,day='2026-10-07',epoch=300)
    slots=store.reward_slot_snapshot(d,OLD_POLICY)
    seed(c,d,slots)
    return slots,[a,b,d]


def test_literal_idempotence_security_and_legacy_unchanged(state):
    store,c,chain=state
    slots,ids=old_slots(store,c,chain)
    with c.cursor() as cur:
        before=signature(cur)
        cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE arena_netuid=%s ORDER BY round_id',(chain,))
        rows=cur.fetchall()
        cur.execute(MIGRATION.read_text());cur.execute(MIGRATION.read_text())
        assert signature(cur)==before
        assert 'frozen_legacy_baseline_training_exempt' in before
        assert "c.round_id = 'arena-2026-10-29'" in before
        cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE arena_netuid=%s ORDER BY round_id',(chain,))
        assert cur.fetchall()==rows
    assert store.reward_slot_snapshot(ids[-1],OLD_POLICY)==slots


@pytest.mark.parametrize('delta,indices',[(4.999,[]),(5,[2]),(10,[1,2]),(15,[0,1,2])])
def test_new_winner_exact_boundaries_and_ordinal_carry(state,delta,indices):
    store,c,chain=state
    slots,ids=old_slots(store,c,chain)
    target=winner(c,chain,'target',delta,activated=False)
    actual=store.reward_slot_snapshot(target,POLICY)
    assert [x['round_id'] for x in actual]==[target if i in indices else ids[i] for i in range(3)]
    assert [x['start_epoch'] for x in actual]==[None if i in indices else [100,200,300][i] for i in range(3)]
    for i in set(range(3))-set(indices):
        assert actual[i]==dict(slots[i],start_epoch=[100,200,300][i])


def test_no_new_winner_carries_without_reviving_displaced_history(state):
    store,c,chain=state
    winner(c,chain,'displaced',30,day='2026-10-04',epoch=90)
    slots,ids=old_slots(store,c,chain)
    slots[0]=None;seed(c,ids[-1],slots)
    target=winner(c,chain,'target',4.999,activated=False)
    actual=store.reward_slot_snapshot(target,POLICY)
    assert actual[0] is None
    assert [x['round_id'] for x in actual[1:]]==ids[1:]


def test_pre_oct23_normalizes_only_current_represented_holders(state):
    store,c,chain=state
    winner(c,chain,'displaced',35,day='2026-10-03',epoch=10)
    a=winner(c,chain,'current',22,day='2026-10-04',epoch=20)
    b=winner(c,chain,'low',6,day='2026-10-05',epoch=30)
    slots=store.reward_slot_snapshot(b,PRE_OCT23)
    seed(c,b,slots,PRE_OCT23)
    target=winner(c,chain,'target',4.999,activated=False)
    assert [x['round_id'] for x in store.reward_slot_snapshot(target,POLICY)]==[a,a,a]
    seed(c,b,[None,None,slots[-1]],PRE_OCT23)
    assert store.reward_slot_snapshot(target,POLICY)==[None,None,None]


@pytest.mark.parametrize('mutate',[
    lambda p:p.pop('pool_percent'), lambda p:p.update(allocation_basis=None),
    lambda p:p.update(allocation_basis='miner_pool'),lambda p:p.update(pool_percent=74),
    lambda p:p.update(extra=True),lambda p:p['tiers'][0].update(allocation_percent=35.0),
])
def test_malformed_policy_rejected(state,mutate):
    store,c,chain=state
    target=winner(c,chain,'target',5,activated=False)
    policy=deepcopy(POLICY);mutate(policy)
    with pytest.raises(ArenaStoreError,match='slot_policy_invalid'):store.reward_slot_snapshot(target,policy)


@pytest.mark.parametrize('mutate',[
    lambda s:s.pop(),lambda s:s.__setitem__(0,{}),
    lambda s:s[0].update(winner_score=99),lambda s:s[0].update(start_epoch=-1),
    lambda s:s[0].update(start_epoch=None),lambda s:s[0].update(extra=True),
    lambda s:s[0].pop('start_epoch'),
])
def test_malformed_carried_achievement_rejected(state,mutate):
    store,c,chain=state
    slots,ids=old_slots(store,c,chain)
    mutate(slots);seed(c,ids[-1],slots)
    target=winner(c,chain,'target',4.999,activated=False)
    with pytest.raises(ArenaStoreError,match='carried_slots_invalid'):store.reward_slot_snapshot(target,POLICY)


def test_cross_chain_future_and_missing_v2_cannot_seed(state):
    store,c,chain=state
    old_slots(store,c,chain+100000)
    target=winner(c,chain,'target',4.999,activated=False)
    assert store.reward_slot_snapshot(target,POLICY)==[None,None,None]
    future=winner(c,chain,'future',25,day='2026-10-09',epoch=500)
    seed(c,future,store.reward_slot_snapshot(future,OLD_POLICY))
    assert store.reward_slot_snapshot(target,POLICY)==[None,None,None]


def test_regression_fails_before_literal_migration_then_passes():
    for pg,dsn in preimage_database():
        c=pg.connect(**dsn);c.autocommit=True
        store=ArenaStore(PsycopgTransport(lambda:pg.connect(**dsn)))
        try:
            with c.cursor() as cur:
                assert hashlib.sha256(signature(cur).encode()).hexdigest()=='0870c5ab466388cdc89b8396e7565e547f4278b2870ea18c3e8bd2bb4dfca5c6'
            target=winner(c,99988,'regression',10,activated=False)
            with pytest.raises(ArenaStoreError,match='slot_policy_invalid'):
                store.reward_slot_snapshot(target,POLICY)
            # Compare every legacy policy variant on the same stored history
            # before and after literal681, including signed-epoch shape.
            winner(c,99988,'old',25,day='2026-10-06',epoch=10)
            winner(c,99988,'oldmid',17,day='2026-10-07',epoch=20)
            policies=[]
            for mode in ('all_qualifying','highest_only'):
                for decay in (False,True):
                    for pool in (False,True):
                        legacy=deepcopy(OLD_POLICY);legacy['assignment_mode']=mode
                        if not decay:legacy.pop('decay')
                        if not pool:legacy.pop('pool_percent')
                        policies.append(legacy)
            legacy_outputs=[store.reward_slot_snapshot(target,p) for p in policies]
            with c.cursor() as cur:cur.execute(MIGRATION.read_text())
            assert [store.reward_slot_snapshot(target,p) for p in policies]==legacy_outputs
            slots=store.reward_slot_snapshot(target,POLICY)
            assert slots[0] is None
            assert [s['round_id'] for s in slots[1:]]==[target,target]
        finally:store.close();c.close()


def test_activation_binds_snapshot_and_preserves_achievement_epochs(state):
    from tests.lab_arena.reward_slots_postgres_test import _basis
    store,c,chain=state
    _,ids=old_slots(store,c,chain)
    target=winner(c,chain,'target',10,activated=False)
    slots=store.reward_slot_snapshot(target,POLICY)
    forged=deepcopy(slots);forged[1]['start_epoch']=400
    basis,key=_basis(target,forged,policy=POLICY,epoch=400)
    with pytest.raises(ArenaStoreError,match='reward_slots_mismatch'):store.activate_reward(target,basis,key)
    basis,key=_basis(target,slots,policy=POLICY,epoch=400)
    assert store.activate_reward(target,basis,key)['status']=='activated'
    assert store.reward_slot_snapshot(target,POLICY)==slots
    later=winner(c,chain,'later',4.999,day='2026-10-09',activated=False)
    carried=store.reward_slot_snapshot(later,POLICY)
    assert [s['round_id'] for s in carried]==[ids[0],target,target]
    assert [s['start_epoch'] for s in carried]==[100,400,400]
    assert store.get_round(target)['reward_basis_doc']==basis


@pytest.mark.parametrize('mutation',['winner_ineligible','baseline_ineligible','wrong_hotkey','duplicate_winner'])
def test_current_winner_keeps_existing_source_proof_guards(state,mutation):
    store,c,chain=state
    old_slots(store,c,chain)
    rid=f'arena-2026-10-08-{chain}target'
    pub=_publication(rid,15)
    if mutation=='winner_ineligible':pub['final_ranking'][0]['eligible']=False
    elif mutation=='baseline_ineligible':pub['final_ranking'][1]['eligible']=False
    elif mutation=='wrong_hotkey':pub['participants'][0]['miner_hotkey']='5'+'C'*47
    else:pub['final_ranking'].append(deepcopy(pub['final_ranking'][0]))
    target=_insert(c,chain,'target',activated=False,publication=pub)
    with pytest.raises(ArenaStoreError,match='current_achievement_invalid'):store.reward_slot_snapshot(target,POLICY)


def test_literal_rejects_changed_function_or_execute_acl(state):
    _,c,_=state
    with c.cursor() as cur:
        original=signature(cur)
        try:
            cur.execute(original.replace('-- 681 prospective allocation:', '-- changed681 prospective allocation:'))
            with pytest.raises(Exception,match='681 reward slot preimage differs'):cur.execute(MIGRATION.read_text())
            cur.execute('ROLLBACK');cur.execute(original)
            cur.execute('GRANT EXECUTE ON FUNCTION public.lab_arena_reward_slot_snapshot(text,jsonb) TO anon')
            with pytest.raises(Exception,match='681 reward slot security differs'):cur.execute(MIGRATION.read_text())
            cur.execute('ROLLBACK')
        finally:
            cur.execute('ROLLBACK')
            cur.execute('REVOKE EXECUTE ON FUNCTION public.lab_arena_reward_slot_snapshot(text,jsonb) FROM anon')
            cur.execute(original)


def test_non_crowned_target_carries_current_signed_holders(state):
    store,c,chain=state
    _,ids=old_slots(store,c,chain)
    rid=f'arena-2026-10-08-{chain}defended'
    pub=_publication(rid,1)
    pub['king_decision'].update(outcome='defended',winner_submission_id=None)
    target=_insert(c,chain,'defended',activated=False,publication=pub)
    assert [s['round_id'] for s in store.reward_slot_snapshot(target,POLICY)]==ids


def test_reopened_governing_basis_cannot_revive_displaced_older_owner(state):
    store,c,chain=state
    a=winner(c,chain,'top',25,day='2026-10-05',epoch=100)
    seed(c,a,store.reward_slot_snapshot(a,OLD_POLICY))
    b=winner(c,chain,'mid',17,day='2026-10-06',epoch=200)
    seed(c,b,store.reward_slot_snapshot(b,OLD_POLICY))
    d=winner(c,chain,'low',9,day='2026-10-07',epoch=300)
    seed(c,d,store.reward_slot_snapshot(d,OLD_POLICY))
    target=winner(c,chain,'target',4.999,activated=False)
    before=store.reward_slot_snapshot(target,POLICY)
    assert [s['round_id'] for s in before]==[a,b,d]
    with c.cursor() as cur:
        cur.execute('BEGIN');cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET status='scored' WHERE round_id=%s",(d,))
        cur.execute('COMMIT')
        cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE arena_netuid=%s ORDER BY round_id',(chain,))
        reopened=cur.fetchall()
    with pytest.raises(ArenaStoreError,match='carried_source_unavailable'):store.reward_slot_snapshot(target,POLICY)
    with c.cursor() as cur:
        cur.execute('SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE arena_netuid=%s ORDER BY round_id',(chain,))
        assert cur.fetchall()==reopened
        cur.execute('BEGIN');cur.execute('SET LOCAL session_replication_role=replica')
        cur.execute("UPDATE public.lab_arena_rounds SET status='published' WHERE round_id=%s",(d,))
        cur.execute('COMMIT')
    assert store.reward_slot_snapshot(target,POLICY)==before
