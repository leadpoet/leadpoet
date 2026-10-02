"""Disposable PostgreSQL proof for late historical publication authority."""

import json
from pathlib import Path

import pytest

from lab_arena import rewards, signing
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_reward_migration_postgres import _basis


MIGRATION = Path(__file__).parents[2] / "scripts/377-lab-arena-monotonic-day-authority.sql"
MIGRATIONS = CURRENT_SERVICE_MIGRATIONS + (
    "342-lab-arena-reward-predecessor-barrier.sql",
    MIGRATION.name,
)
BASELINE = "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"
MINER = "5FbuYLc9XA5nCyERcFCj4wjMyECLyPdpbM61M9cSy8QjKF1Q"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(MIGRATIONS)


def _round(cursor, day, suffix, *, netuid=71, outcome="crowned", promoted=False):
    round_id = f"arena-{day}-{suffix}"
    published = f"{day}T12:00:00Z"
    config = {
        "mode": "live", "network_name": "finney", "netuid": netuid,
        "rewards_enabled": True, "baseline_hotkey": BASELINE,
        "reward_constants": rewards.reward_constants_document(),
    }
    publication = {
        "published_at": published,
        "king_decision": {
            "outcome": outcome,
            "king_hotkey": MINER if outcome == "crowned" else "",
        },
    }
    cursor.execute(
        "INSERT INTO public.lab_arena_rounds "
        "(round_id,status,configuration_doc,rewards_enabled,evaluation_date,"
        "publication_doc,king_outcome,king_hotkey,published_at,"
        "promotion_required,baseline_promoted_at,created_at) "
        "VALUES (%s,'published',%s::jsonb,TRUE,%s,%s::jsonb,%s,%s,"
        "%s::timestamptz,TRUE,%s::timestamptz,%s::timestamptz)",
        (
            round_id, json.dumps(config), day, json.dumps(publication), outcome,
            MINER if outcome == "crowned" else None, published,
            published if promoted else None, f"{day}T00:00:00Z",
        ),
    )
    return round_id, published


def _store(psycopg, dsn):
    return ArenaStore(PsycopgTransport(lambda: psycopg.connect(**dsn)))


def test_late_prior_day_cannot_prepare_git_or_activate_reward(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            old, _ = _round(cursor, "2026-10-01", "late")
            newer, _ = _round(cursor, "2026-10-02", "published", promoted=True)
    store = _store(psycopg, dsn)
    try:
        assert store.prepare_promotion(old, {}) == {
            "status": "superseded", "newer_round_id": newer,
        }
        assert store.activate_reward(old, {}, {}) == {
            "status": "superseded", "newer_round_id": newer,
        }
        row = store.get_round(old)
        assert row["promotion_doc"] is None
        assert row["baseline_promoted_at"] is None
        assert row["reward_activated_at"] is None
        assert row["reward_basis_doc"] is None
        assert row["publication_doc"]["king_decision"]["outcome"] == "crowned"
    finally:
        store.close()


def test_newer_day_activation_ignores_superseded_older_promotion(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            _round(cursor, "2026-10-03", "oldpending")
            current, published = _round(
                cursor, "2026-10-04", "current", outcome="no_king"
            )
    signer = signing.LocalSigner.generate()
    key = signing.signing_key_document(signer.public_key_der)
    basis = _basis(signer, current, published, 1000, "no_king", "")
    store = _store(psycopg, dsn)
    try:
        assert store.activate_reward(current, basis, key)["status"] == "activated"
        assert store.get_round(current)["reward_activated_at"] is not None
    finally:
        store.close()


def test_same_day_and_other_chain_are_not_superseded(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            first, published = _round(cursor, "2026-10-05", "first", netuid=72)
            same, _ = _round(cursor, "2026-10-05", "sameday", netuid=72)
            _round(cursor, "2026-10-06", "otherchain", netuid=73)
    store = _store(psycopg, dsn)
    try:
        signer = signing.LocalSigner.generate()
        key = signing.signing_key_document(signer.public_key_der)
        basis = _basis(signer, first, published, 1001, "crowned", MINER)
        with pytest.raises(ArenaStoreError, match="reward_waiting_for_promotion"):
            store.activate_reward(first, basis, key)
        assert store.activate_reward(same, {}, {})["status"] == "waiting_for_older_round"
    finally:
        store.close()


def test_replay_and_definition_guard(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION.read_text())
            for signature in (
                "public.lab_arena_prepare_promotion(text,jsonb)",
                "public.lab_arena_activate_reward(text,jsonb,jsonb)",
                "public.lab_arena_reward_requires_promotion_v1()",
            ):
                cursor.execute("SELECT pg_catalog.pg_get_functiondef(%s::regprocedure)",
                               (signature,))
                assert "377 monotonic day authority" in cursor.fetchone()[0]


def test_replay_rejects_mutated_authority_function(database):
    psycopg, dsn = database
    with psycopg.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute("BEGIN")
            cursor.execute(
                "SELECT pg_catalog.pg_get_functiondef("
                "'public.lab_arena_prepare_promotion(text,jsonb)'::regprocedure)"
            )
            definition = cursor.fetchone()[0]
            cursor.execute(definition.replace(
                "-- 377 monotonic day authority",
                "-- 377 monotonic day authority\n  -- unexpected change",
            ))
            with pytest.raises(psycopg.Error, match="377 promotion replay differs"):
                cursor.execute(MIGRATION.read_text())
            cursor.execute("ROLLBACK")
