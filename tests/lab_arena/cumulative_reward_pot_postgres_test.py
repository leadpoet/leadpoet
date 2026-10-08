"""Real PostgreSQL proof for the optional, signed cumulative miner pot."""

from copy import deepcopy
import itertools
from pathlib import Path

import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration
from tests.lab_arena.reward_slots_postgres_test import (
    MIGRATIONS as SLOT_MIGRATIONS, POLICY as OLD_POLICY,
    _basis, _insert, _publication,
)


MIGRATION = Path(__file__).parents[2] / "scripts/429-lab-arena-cumulative-reward-pot.sql"
MIGRATIONS = SLOT_MIGRATIONS + (MIGRATION.name,)
CHAIN_IDS = itertools.count(1500)
ALICE, BOB, CAROL = ["5" + letter * 47 for letter in "BCD"]


def _legacy_cumulative_policy():
    # This fixture deliberately stops at migration 429, before optional decay.
    return dict(deepcopy(OLD_POLICY), assignment_mode="all_qualifying", pool_percent=30)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(MIGRATIONS)


@pytest.fixture()
def state(database):
    psycopg, dsn = database
    control = psycopg.connect(**dsn)
    control.autocommit = True
    store = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    try:
        yield store, control, next(CHAIN_IDS), psycopg, dsn
    finally:
        store.close()
        control.close()


def _winner(control, chain, name, *, day, delta, hotkey, baseline_score=40,
            activated=True, eligible=True):
    round_id = "arena-%s-%s%s" % (day, chain, name)
    publication = _publication(
        round_id, delta, winner=hotkey, baseline_score=baseline_score,
    )
    publication["final_ranking"][0]["eligible"] = eligible
    return _insert(
        control, chain, name, day=day, activated=activated,
        publication=publication,
    )


def test_cumulative_slots_follow_each_published_win_across_days(state):
    store, control, chain, _, _ = state
    a = _winner(control, chain, "a", day="2026-10-05", delta=11, hotkey=ALICE)
    b = _winner(control, chain, "b", day="2026-10-06", delta=6,
                hotkey=BOB, baseline_score=60)
    c = _winner(control, chain, "c", day="2026-10-07", delta=2,
                hotkey=CAROL, baseline_score=80)
    policy = _legacy_cumulative_policy()
    assert [x["round_id"] for x in store.reward_slot_snapshot(a, policy)] == [a, a, a]
    assert [x["round_id"] for x in store.reward_slot_snapshot(b, policy)] == [a, b, b]
    slots = store.reward_slot_snapshot(c, policy)
    assert [x["round_id"] for x in slots] == [a, b, c]
    assert [(x["baseline_score"], x["winner_score"]) for x in slots] == [
        (40, 51), (60, 66), (80, 82),
    ]
    assert [x["miner_hotkey"] for x in slots] == [ALICE, BOB, CAROL]


@pytest.mark.parametrize("delta,expected", [
    (10, [True, True, True]),
    (9.999999, [False, True, True]),
    (5, [False, True, True]),
    (4.999999, [False, False, True]),
    (1, [False, False, True]),
    (0.999999, [False, False, False]),
    (0, [False, False, False]),
])
def test_exact_boundaries_and_below_threshold_have_no_new_holder(state, delta, expected):
    store, control, chain, _, _ = state
    target = _winner(control, chain, "boundary", day="2026-10-07",
                     delta=delta, hotkey=ALICE, activated=False)
    slots = store.reward_slot_snapshot(target, _legacy_cumulative_policy())
    assert [slot is not None for slot in slots] == expected


def test_tie_and_ineligible_newer_winner_do_not_displace_holders(state):
    store, control, chain, _, _ = state
    a = _winner(control, chain, "a", day="2026-10-05", delta=11, hotkey=ALICE)
    _winner(control, chain, "ineligible", day="2026-10-06", delta=20,
            hotkey=BOB, eligible=False)
    tied = _winner(control, chain, "tie", day="2026-10-07", delta=0,
                   hotkey=CAROL, activated=False)
    assert [slot["round_id"] for slot in
            store.reward_slot_snapshot(tied, _legacy_cumulative_policy())] == [a, a, a]


def test_activation_persists_signed_policy_and_slots_across_store_reopen(state):
    store, control, chain, psycopg, dsn = state
    a = _winner(control, chain, "a", day="2026-10-05", delta=11, hotkey=ALICE)
    b = _winner(control, chain, "b", day="2026-10-06", delta=6, hotkey=BOB)
    c = _winner(control, chain, "c", day="2026-10-07", delta=2,
                hotkey=CAROL, activated=False)
    policy = _legacy_cumulative_policy()
    slots = store.reward_slot_snapshot(c, policy)
    assert [slot["round_id"] for slot in slots] == [a, b, c]
    basis, signing_key = _basis(c, slots, policy=policy, hotkey=CAROL)
    assert store.activate_reward(c, basis, signing_key)["status"] == "activated"
    assert store.activate_reward(c, basis, signing_key)["status"] == "existing"
    reopened = ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))
    try:
        saved = reopened.get_round(c)
        assert saved["reward_basis_doc"] == basis
        assert saved["reward_basis_doc"]["slot_policy"] == policy
        assert reopened.reward_slot_snapshot(c, policy) == slots
    finally:
        reopened.close()
    assert store.get_round(a)["reward_basis_doc"] == {"king_outcome": "no_king"}


def test_old_two_field_policy_remains_valid_after_migration(state):
    store, control, chain, _, _ = state
    target = _winner(control, chain, "old", day="2026-10-07", delta=11,
                     hotkey=ALICE, activated=False)
    slots = store.reward_slot_snapshot(target, OLD_POLICY)
    assert [slot is not None for slot in slots] == [True, False, False]
    basis, key = _basis(target, slots, policy=OLD_POLICY, hotkey=ALICE)
    assert store.activate_reward(target, basis, key)["status"] == "activated"
    assert store.get_round(target)["reward_basis_doc"]["slot_policy"] == OLD_POLICY


@pytest.mark.parametrize("bad", [True, None, -1, 101, 30.0, "30"])
def test_invalid_optional_pool_fails_database_validation(state, bad):
    store, control, chain, _, _ = state
    target = _winner(control, chain, "invalid", day="2026-10-07", delta=11,
                     hotkey=ALICE, activated=False)
    policy = deepcopy(_legacy_cumulative_policy())
    policy["pool_percent"] = bad
    with pytest.raises(ArenaStoreError, match="slot_policy_invalid"):
        store.reward_slot_snapshot(target, policy)


def test_unknown_policy_key_fails_and_zero_and_hundred_are_accepted(state):
    store, control, chain, _, _ = state
    target = _winner(control, chain, "policy", day="2026-10-07", delta=11,
                     hotkey=ALICE, activated=False)
    policy = deepcopy(_legacy_cumulative_policy())
    policy["extra"] = 1
    with pytest.raises(ArenaStoreError, match="slot_policy_invalid"):
        store.reward_slot_snapshot(target, policy)
    for pool in (0, 100):
        policy = deepcopy(_legacy_cumulative_policy())
        policy["pool_percent"] = pool
        assert [slot is not None for slot in store.reward_slot_snapshot(target, policy)] == [True] * 3


def test_literal_migration_non_superuser_replay_preserves_shape_and_acl():
    database = database_with_lab_arena_migration(SLOT_MIGRATIONS)
    psycopg, dsn = next(database)
    control = psycopg.connect(**dsn)
    control.autocommit = True
    try:
        with control.cursor() as cursor:
            cursor.execute("CREATE ROLE arena_reward429_migrator NOLOGIN NOSUPERUSER INHERIT")
            cursor.execute("GRANT lab_arena_owner TO arena_reward429_migrator")
            cursor.execute("GRANT USAGE, CREATE ON SCHEMA public TO arena_reward429_migrator WITH GRANT OPTION")
            cursor.execute("REVOKE CREATE ON SCHEMA public FROM lab_arena_owner")
            cursor.execute("SELECT pg_catalog.pg_get_functiondef('public.lab_arena_reward_slot_snapshot(text,jsonb)'::regprocedure)")
            before = cursor.fetchone()[0]
            cursor.execute("SELECT nspacl FROM pg_catalog.pg_namespace WHERE nspname='public'")
            schema_acl = cursor.fetchone()[0]
            cursor.execute("SELECT proowner,proacl,prosecdef FROM pg_catalog.pg_proc WHERE oid='public.lab_arena_reward_slot_snapshot(text,jsonb)'::regprocedure")
            function_acl = cursor.fetchone()
            cursor.execute("SET ROLE arena_reward429_migrator")
            cursor.execute("SELECT rolsuper FROM pg_catalog.pg_roles WHERE rolname=current_user")
            assert cursor.fetchone()[0] is False
            for _ in range(2):
                cursor.execute(MIGRATION.read_text())
                cursor.execute("SELECT pg_catalog.pg_get_functiondef('public.lab_arena_reward_slot_snapshot(text,jsonb)'::regprocedure)")
                after = cursor.fetchone()[0]
                cursor.execute("SELECT nspacl FROM pg_catalog.pg_namespace WHERE nspname='public'")
                assert cursor.fetchone()[0] == schema_acl
                cursor.execute("SELECT proowner,proacl,prosecdef FROM pg_catalog.pg_proc WHERE oid='public.lab_arena_reward_slot_snapshot(text,jsonb)'::regprocedure")
                assert cursor.fetchone() == function_acl
            cursor.execute("RESET ROLE")
            old_shape = "  IF (SELECT pg_catalog.count(*) FROM pg_catalog.jsonb_object_keys(p_slot_policy)) <> 2"
            new_shape = "  IF (SELECT pg_catalog.count(*) FROM pg_catalog.jsonb_object_keys(p_slot_policy - 'pool_percent')) <> 2"
            old_tiers = "  IF pg_catalog.jsonb_array_length(p_slot_policy -> 'tiers') <> 3 THEN"
            new_tiers = """  IF p_slot_policy ? 'pool_percent' AND (
       pg_catalog.jsonb_typeof(p_slot_policy -> 'pool_percent') IS DISTINCT FROM 'number'
       OR COALESCE(p_slot_policy ->> 'pool_percent', '') !~ '^(0|[1-9][0-9]?)$|^100$'
     ) THEN
    RAISE EXCEPTION 'lab_arena_slot_policy_invalid' USING ERRCODE = '22023';
  END IF;
  IF pg_catalog.jsonb_array_length(p_slot_policy -> 'tiers') <> 3 THEN"""
            assert before.count(old_shape) == 1
            assert before.count(old_tiers) == 1
            assert after == before.replace(old_shape, new_shape).replace(old_tiers, new_tiers)
            for role in ("anon", "authenticated", "service_role"):
                cursor.execute("SELECT has_function_privilege(%s,'public.lab_arena_reward_slot_snapshot(text,jsonb)','EXECUTE')", (role,))
                assert cursor.fetchone()[0] is False
            cursor.execute("SELECT has_function_privilege('lab_arena_service','public.lab_arena_reward_slot_snapshot(text,jsonb)','EXECUTE')")
            assert cursor.fetchone()[0] is True
    finally:
        control.close()
        database.close()
