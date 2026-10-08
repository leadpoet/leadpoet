"""Real PostgreSQL proof for signed slot achievement epochs and optional decay."""

from copy import deepcopy
from fractions import Fraction
import itertools
import json
from pathlib import Path

import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from leadpoet_canonical import lab_arena_rewards as kernel
from tests.lab_arena.cumulative_reward_pot_postgres_test import (
    MIGRATIONS as POT_MIGRATIONS, ALICE, BOB, CAROL, _legacy_cumulative_policy,
)
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration
from tests.lab_arena.reward_slots_postgres_test import _basis, _insert, _publication


MIGRATION = Path(__file__).parents[2] / "scripts/431-lab-arena-slot-decay.sql"
MIGRATIONS = POT_MIGRATIONS + (MIGRATION.name,)
CHAIN_IDS = itertools.count(2100)
POLICY = dict(_legacy_cumulative_policy(), decay={"epochs_per_halving": 140, "max_halvings": 4})
LEGACY_FIELDS = {"round_id", "submission_id", "miner_hotkey", "baseline_submission_id",
                 "baseline_score", "winner_score"}


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
        yield store, control, next(CHAIN_IDS)
    finally:
        store.close()
        control.close()


def _winner(control, chain, name, *, day, delta, hotkey, epoch=None, activated=True):
    round_id = f"arena-{day}-{chain}{name}"
    return _insert(control, chain, name, day=day, epoch=epoch, activated=activated,
                   publication=_publication(round_id, delta, winner=hotkey))


def test_source_epoch_survives_daily_refresh_and_week_boundary_and_replacement(state):
    store, control, chain = state
    a = _winner(control, chain, "a", day="2026-10-05", delta=11, hotkey=ALICE, epoch=100)
    b = _winner(control, chain, "b", day="2026-10-06", delta=6, hotkey=BOB, epoch=240)
    c = _winner(control, chain, "c", day="2026-10-07", delta=2, hotkey=CAROL, activated=False)
    slots = store.reward_slot_snapshot(c, POLICY)
    assert [s["round_id"] for s in slots] == [a, b, c]
    assert [s["start_epoch"] for s in slots] == [100, 240, None]
    basis, key = _basis(c, slots, policy=POLICY, hotkey=CAROL, epoch=380)
    assert store.activate_reward(c, basis, key)["status"] == "activated"
    assert store.activate_reward(c, basis, key)["status"] == "existing"
    # The current source stays NULL in its own signed snapshot after activation.
    assert store.reward_slot_snapshot(c, POLICY) == slots
    assert store.get_round(c)["reward_basis_doc"]["reward_slots"] == slots
    hotkeys = [ALICE, BOB, CAROL]
    # Decay does not extend the separate 45-epoch signed-basis freshness gate.
    assert kernel.slot_allocations(basis, 425, hotkeys)
    assert kernel.slot_allocations(basis, 426, hotkeys) == {}
    refresh = _winner(control, chain, "refresh", day="2026-10-08", delta=0.5,
                      hotkey=CAROL, activated=False)
    refreshed = store.reward_slot_snapshot(refresh, POLICY)
    assert [s["round_id"] for s in refreshed] == [a, b, c]
    assert [s["start_epoch"] for s in refreshed] == [100, 240, 380]
    refresh_basis, _ = _basis(refresh, refreshed, policy=POLICY, hotkey=CAROL, epoch=519)
    assert kernel.slot_allocations(refresh_basis, 519, hotkeys) == {
        ALICE: Fraction(3, 80), BOB: Fraction(9, 200), CAROL: Fraction(3, 50),
    }
    assert kernel.slot_allocations(refresh_basis, 520, hotkeys) == {
        ALICE: Fraction(3, 160), BOB: Fraction(9, 400), CAROL: Fraction(3, 100),
    }
    # A fresh achievement resets only the tiers that it actually replaces.
    replacement = _winner(control, chain, "replacement", day="2026-10-09", delta=6,
                          hotkey=BOB, activated=False)
    replaced = store.reward_slot_snapshot(replacement, POLICY)
    assert [s["round_id"] for s in replaced] == [a, replacement, replacement]
    assert [s["start_epoch"] for s in replaced] == [100, None, None]
    doc, key = _basis(replacement, replaced, policy=POLICY, hotkey=BOB, epoch=520)
    assert store.activate_reward(replacement, doc, key)["status"] == "activated"
    assert store.reward_slot_snapshot(replacement, POLICY) == replaced
    later = _winner(control, chain, "later", day="2026-10-10", delta=0.5,
                    hotkey=CAROL, activated=False)
    assert [s["start_epoch"] for s in store.reward_slot_snapshot(later, POLICY)] == [100, 520, 520]


@pytest.mark.parametrize("source", ["current", "historical"])
def test_activation_rejects_forged_start_epoch(state, source):
    store, control, chain = state
    if source == "historical":
        _winner(control, chain, "old", day="2026-10-07", delta=11, hotkey=ALICE, epoch=100)
    target = _winner(control, chain, "tamper", day="2026-10-08",
                     delta=11 if source == "current" else 0.5, hotkey=ALICE, activated=False)
    slots = store.reward_slot_snapshot(target, POLICY)
    slots[0]["start_epoch"] = 200
    basis, key = _basis(target, slots, policy=POLICY, hotkey=ALICE, epoch=240)
    with pytest.raises(ArenaStoreError, match="reward_slots_mismatch"):
        store.activate_reward(target, basis, key)
    assert store.get_round(target)["reward_basis_doc"] is None


@pytest.mark.parametrize("pool", [False, True])
def test_legacy_snapshot_shape_and_activation_remain_valid(state, pool):
    store, control, chain = state
    policy = _legacy_cumulative_policy()
    if not pool:
        policy.pop("pool_percent")
    target = _winner(control, chain, "legacy", day="2026-10-08", delta=11,
                     hotkey=ALICE, activated=False)
    slots = store.reward_slot_snapshot(target, policy)
    assert all(set(slot) == LEGACY_FIELDS for slot in slots)
    basis, key = _basis(target, slots, policy=policy, hotkey=ALICE)
    assert store.activate_reward(target, basis, key)["status"] == "activated"
    assert store.reward_slot_snapshot(target, policy) == slots


@pytest.mark.parametrize("bad", [
    None, True, [], {}, {"epochs_per_halving": 140},
    {"epochs_per_halving": 140, "max_halvings": 4, "extra": 0},
    *[{"epochs_per_halving": value, "max_halvings": 4}
      for value in (None, True, 0, -1, 140.0, "140", 1000001, 10**100)],
    *[{"epochs_per_halving": 140, "max_halvings": value}
      for value in (None, True, -1, 13, 4.0, "4", 10**100)],
])
def test_malformed_decay_policy_is_rejected(state, bad):
    store, control, chain = state
    target = _winner(control, chain, "invalid", day="2026-10-08", delta=11,
                     hotkey=ALICE, activated=False)
    policy = dict(deepcopy(POLICY), decay=bad)
    with pytest.raises(ArenaStoreError, match="slot_policy_invalid"):
        store.reward_slot_snapshot(target, policy)


@pytest.mark.parametrize("epochs,halvings", [(1, 0), (1000000, 12)])
def test_decay_policy_bounds_are_accepted(state, epochs, halvings):
    store, control, chain = state
    target = _winner(control, chain, "bounds", day="2026-10-08", delta=11,
                     hotkey=ALICE, activated=False)
    policy = dict(deepcopy(POLICY), decay={"epochs_per_halving": epochs, "max_halvings": halvings})
    assert [s["start_epoch"] for s in store.reward_slot_snapshot(target, policy)] == [None] * 3


@pytest.mark.parametrize("bad_epoch", [None, -1])
def test_selected_historical_invalid_epoch_fails_but_unselected_history_does_not(state, bad_epoch):
    store, control, chain = state
    old = _winner(control, chain, "old", day="2026-10-07", delta=11, hotkey=ALICE, epoch=100)
    target = _winner(control, chain, "target", day="2026-10-08", delta=0.5,
                     hotkey=ALICE, activated=False)
    # Deliberately corrupt an isolated fixture to prove fail-closed read behavior.
    with control.cursor() as cursor:
        cursor.execute("BEGIN")
        try:
            cursor.execute("SET LOCAL session_replication_role = replica")
            if bad_epoch == -1:
                cursor.execute("ALTER TABLE public.lab_arena_rounds DROP CONSTRAINT lab_arena_rounds_effective_reward_epoch_check")
            cursor.execute("UPDATE public.lab_arena_rounds SET effective_reward_epoch=%s WHERE round_id=%s", (bad_epoch, old))
            cursor.execute("SELECT public.lab_arena_reward_slot_snapshot(%s,%s::jsonb)",
                           (target, json.dumps(_legacy_cumulative_policy())))
            assert cursor.fetchone()[0]["reward_slots"][0]["round_id"] == old
            with pytest.raises(Exception, match="slot_start_epoch_invalid"):
                cursor.execute("SELECT public.lab_arena_reward_slot_snapshot(%s,%s::jsonb)",
                               (target, json.dumps(POLICY)))
        finally:
            cursor.execute("ROLLBACK")
    newer = _winner(control, chain, "newer", day="2026-10-09", delta=11,
                    hotkey=BOB, activated=False)
    with control.cursor() as cursor:
        cursor.execute("BEGIN")
        try:
            cursor.execute("SET LOCAL session_replication_role = replica")
            if bad_epoch == -1:
                cursor.execute("ALTER TABLE public.lab_arena_rounds DROP CONSTRAINT lab_arena_rounds_effective_reward_epoch_check")
            cursor.execute("UPDATE public.lab_arena_rounds SET effective_reward_epoch=%s WHERE round_id=%s", (bad_epoch, old))
            cursor.execute("SELECT public.lab_arena_reward_slot_snapshot(%s,%s::jsonb)",
                           (newer, json.dumps(POLICY)))
            assert [s["round_id"] for s in cursor.fetchone()[0]["reward_slots"]] == [newer] * 3
        finally:
            cursor.execute("ROLLBACK")


def test_decay_rpc_executes_as_service_and_does_not_write(state):
    store, control, chain = state
    target = _winner(control, chain, "service", day="2026-10-08", delta=11,
                     hotkey=ALICE, activated=False)
    before = store.get_round(target)
    with control.cursor() as cursor:
        cursor.execute("SET ROLE lab_arena_service")
        try:
            cursor.execute("SELECT public.lab_arena_reward_slot_snapshot(%s,%s::jsonb)",
                           (target, json.dumps(POLICY)))
            slots = cursor.fetchone()[0]["reward_slots"]
            assert [s["start_epoch"] for s in slots] == [None] * 3
        finally:
            cursor.execute("RESET ROLE")
    assert store.get_round(target) == before


def test_literal_migration_non_superuser_replay_preserves_definition_and_acl():
    database = database_with_lab_arena_migration(POT_MIGRATIONS)
    psycopg, dsn = next(database)
    control = psycopg.connect(**dsn)
    control.autocommit = True
    try:
        with control.cursor() as cursor:
            cursor.execute("CREATE ROLE arena_reward431_migrator NOLOGIN NOSUPERUSER INHERIT")
            cursor.execute("GRANT lab_arena_owner TO arena_reward431_migrator")
            cursor.execute("GRANT USAGE, CREATE ON SCHEMA public TO arena_reward431_migrator WITH GRANT OPTION")
            cursor.execute("REVOKE CREATE ON SCHEMA public FROM lab_arena_owner")
            cursor.execute("SELECT nspacl FROM pg_catalog.pg_namespace WHERE nspname='public'")
            schema_acl = cursor.fetchone()[0]
            cursor.execute("SELECT proowner,proacl,prosecdef FROM pg_catalog.pg_proc WHERE oid='public.lab_arena_reward_slot_snapshot(text,jsonb)'::regprocedure")
            function_acl = cursor.fetchone()
            cursor.execute("SELECT pg_catalog.pg_get_functiondef('public.lab_arena_activate_reward(text,jsonb,jsonb)'::regprocedure)")
            activation = cursor.fetchone()[0]
            cursor.execute("SET ROLE arena_reward431_migrator")
            cursor.execute("SELECT rolsuper FROM pg_catalog.pg_roles WHERE rolname=current_user")
            assert cursor.fetchone()[0] is False
            after = None
            for _ in range(2):
                cursor.execute(MIGRATION.read_text())
                cursor.execute("SELECT pg_catalog.pg_get_functiondef('public.lab_arena_reward_slot_snapshot(text,jsonb)'::regprocedure)")
                definition = cursor.fetchone()[0]
                if after is not None:
                    assert definition == after
                after = definition
                cursor.execute("SELECT nspacl FROM pg_catalog.pg_namespace WHERE nspname='public'")
                assert cursor.fetchone()[0] == schema_acl
                cursor.execute("SELECT proowner,proacl,prosecdef FROM pg_catalog.pg_proc WHERE oid='public.lab_arena_reward_slot_snapshot(text,jsonb)'::regprocedure")
                assert cursor.fetchone() == function_acl
                cursor.execute("SELECT pg_catalog.pg_get_functiondef('public.lab_arena_activate_reward(text,jsonb,jsonb)'::regprocedure)")
                assert cursor.fetchone()[0] == activation
            cursor.execute("RESET ROLE")
            for role in ("anon", "authenticated", "service_role"):
                cursor.execute("SELECT has_function_privilege(%s,'public.lab_arena_reward_slot_snapshot(text,jsonb)','EXECUTE')", (role,))
                assert cursor.fetchone()[0] is False
            cursor.execute("SELECT has_function_privilege('lab_arena_service','public.lab_arena_reward_slot_snapshot(text,jsonb)','EXECUTE')")
            assert cursor.fetchone()[0] is True
    finally:
        control.close()
        database.close()


def test_migration_hash_guard_rejects_drift(state):
    _, control, _ = state
    with control.cursor() as cursor:
        cursor.execute("BEGIN")
        try:
            cursor.execute("SELECT pg_catalog.pg_get_functiondef('public.lab_arena_reward_slot_snapshot(text,jsonb)'::regprocedure)")
            definition = cursor.fetchone()[0]
            cursor.execute(definition.replace("DECLARE", "-- unauthorized drift\nDECLARE", 1))
            with pytest.raises(Exception, match="431 reward slot definition differs"):
                cursor.execute(MIGRATION.read_text())
        finally:
            cursor.execute("ROLLBACK")
