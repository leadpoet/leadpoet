"""Focused PostgreSQL tests for durable Arena baseline promotion state."""

from __future__ import annotations

import json

import pytest

from lab_arena import rewards, signing
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    LAB_ARENA_BASELINE_PROMOTION_MIGRATION,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_reward_migration_postgres import (
    BASELINE,
    MINER_A,
    _basis,
    _publish,
)
from tests.test_source_add_end_to_end_postgres import SCRIPTS


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration()


@pytest.fixture()
def connections(database):
    psycopg2, dsn = database
    control = psycopg2.connect(**dsn)
    control.autocommit = True
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    try:
        yield ArenaStore(transport), control
    finally:
        transport.close()
        control.close()


def _plan(seed: str = "1") -> dict:
    return {
        "commit": seed * 40,
        "main_before": "2" * 40,
        "lab_before": "3" * 40,
        "timestamp": "2026-09-07T12:34:56.123456+00:00",
    }


def _crowned_round(store, control, round_id: str, *, miner=MINER_A):
    published_at = _publish(
        store,
        control,
        round_id,
        miner=miner,
        baseline_score=50,
        miner_score=60,
        crowned=True,
    )
    with control.cursor() as cursor:
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions "
            "(submission_id, round_id, miner_hotkey, status, is_king) "
            "VALUES (%s, %s, %s, 'frozen', FALSE)",
            (round_id + "-miner", round_id, miner),
        )
    return published_at


@pytest.mark.parametrize(
    "baseline_score,miner_score,crowned",
    [(50.0, 50.0, False), (50.0, 50.999999, False), (50.0, 51.0, True), (70.1, 71.1, True)],
)
def test_publication_rpc_enforces_the_exact_one_point_boundary(
    connections, baseline_score, miner_score, crowned
):
    store, control = connections
    # Both the allowed decision and its forbidden opposite go through the
    # actual SQL publication RPC and trigger, not only the Python comparator.
    _publish(
        store, control, "arena-2026-09-07-boundarygood", miner=MINER_A,
        baseline_score=baseline_score, miner_score=miner_score, crowned=crowned,
    )
    with pytest.raises(ArenaStoreError, match="publication_winner_"):
        _publish(
            store, control, "arena-2026-09-07-boundarybad", miner=MINER_A,
            baseline_score=baseline_score, miner_score=miner_score, crowned=not crowned,
        )


def test_prepare_and_complete_are_exactly_idempotent(connections):
    store, control = connections
    round_id = "arena-2026-09-07-promoa"
    _crowned_round(store, control, round_id)
    plan = _plan("a")

    assert store.prepare_promotion(round_id, plan) == {"status": "prepared", "plan": plan}
    assert store.prepare_promotion(round_id, plan) == {"status": "existing", "plan": plan}
    with pytest.raises(ArenaStoreError, match="promotion_plan_mismatch"):
        store.prepare_promotion(round_id, _plan("b"))

    completed = store.complete_promotion(round_id, plan)
    assert completed["status"] == "promoted" and completed["baseline_promoted_at"]
    retry = store.complete_promotion(round_id, plan)
    assert retry["status"] == "existing"
    assert retry["baseline_promoted_at"] == completed["baseline_promoted_at"]
    with pytest.raises(ArenaStoreError, match="promotion_plan_mismatch"):
        store.complete_promotion(round_id, _plan("b"))


def test_oldest_pending_winner_prepares_first(connections):
    store, control = connections
    first = "arena-2026-09-07-promob"
    second = "arena-2026-09-07-promoc"
    _crowned_round(store, control, first)
    _crowned_round(store, control, second)
    assert store.prepare_promotion(second, _plan("c"))["status"] == "waiting_for_older_promotion"
    assert [row["round_id"] for row in store.pending_promotions(limit=10)][:2] == [first, second]
    first_plan = _plan("b")
    assert store.prepare_promotion(first, first_plan)["status"] == "prepared"
    store.complete_promotion(first, first_plan)
    assert store.prepare_promotion(second, _plan("c"))["status"] == "prepared"
    store.complete_promotion(second, _plan("c"))


def test_no_king_rows_before_limit_cannot_starve_pending_winner(connections):
    store, control = connections
    with control.cursor() as cursor:
        for index in range(101):
            round_id = "arena-2026-09-05-n%03d" % index
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds "
                "(round_id, status, configuration_doc, rewards_enabled, publication_doc, "
                "king_outcome, published_at) VALUES (%s, 'published', %s::jsonb, FALSE, "
                "%s::jsonb, 'no_king', pg_catalog.clock_timestamp())",
                (
                    round_id,
                    json.dumps({"mode": "live", "rewards_enabled": False}),
                    json.dumps({"king_decision": {"outcome": "no_king"}}),
                ),
            )
    winner = "arena-2026-09-07-starve"
    _crowned_round(store, control, winner)
    assert [row["round_id"] for row in store.pending_promotions(limit=100)] == [winner]


def test_reward_activation_waits_for_new_crowned_promotion_but_not_no_king(connections):
    store, control = connections
    signer = signing.LocalSigner.generate()
    key = signing.signing_key_document(signer.public_key_der)
    crowned = "arena-2026-09-07-promod"
    crowned_at = _crowned_round(store, control, crowned)
    basis = _basis(signer, crowned, crowned_at, 1000, "crowned", MINER_A)
    with pytest.raises(ArenaStoreError, match="reward_waiting_for_promotion"):
        store.activate_reward(crowned, basis, key)
    plan = _plan("d")
    store.prepare_promotion(crowned, plan)
    store.complete_promotion(crowned, plan)
    assert store.activate_reward(crowned, basis, key)["status"] == "activated"

    no_king = "arena-2026-09-07-promoe"
    no_king_at = _publish(
        store,
        control,
        no_king,
        miner=MINER_A,
        baseline_score=50,
        miner_score=40,
        crowned=False,
    )
    defended = _basis(signer, no_king, no_king_at, 1001, "defended", MINER_A, 1000)
    assert store.activate_reward(no_king, defended, key)["status"] == "activated"


def test_migration_preserves_historical_round_and_applies_twice():
    generator = database_with_lab_arena_migration(DEFAULT_MIGRATIONS[:-2])
    psycopg2, dsn = next(generator)
    control = psycopg2.connect(**dsn)
    control.autocommit = True
    try:
        with control.cursor() as cursor:
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds "
                "(round_id, status, configuration_doc, rewards_enabled, publication_doc, "
                "king_outcome, published_at) VALUES (%s, 'published', %s::jsonb, TRUE, "
                "%s::jsonb, 'crowned', pg_catalog.clock_timestamp())",
                (
                    "arena-2026-09-06-legacy",
                    json.dumps({"mode": "live", "rewards_enabled": True}),
                    json.dumps({"king_decision": {"outcome": "crowned"}}),
                ),
            )
            cursor.execute(
                "INSERT INTO public.lab_arena_rounds "
                "(round_id, status, configuration_doc, rewards_enabled) VALUES "
                "(%s, 'open', %s::jsonb, FALSE), "
                "(%s, 'scored', %s::jsonb, FALSE), "
                "(%s, 'cancelled', %s::jsonb, FALSE)",
                (
                    "arena-2026-09-06-active",
                    json.dumps({"mode": "live", "rewards_enabled": False}),
                    "arena-2026-09-06-scored",
                    json.dumps({"mode": "live", "rewards_enabled": False}),
                    "arena-2026-09-06-cancelled",
                    json.dumps({"mode": "live", "rewards_enabled": False}),
                ),
            )
            migration = (SCRIPTS / LAB_ARENA_BASELINE_PROMOTION_MIGRATION).read_text(
                encoding="utf-8"
            )
            cursor.execute(migration)
            cursor.execute(migration)
            cursor.execute(
                "SELECT round_id, promotion_required, promotion_doc, baseline_promoted_at "
                "FROM public.lab_arena_rounds "
                "WHERE round_id LIKE 'arena-2026-09-06-%' ORDER BY round_id"
            )
            assert cursor.fetchall() == [
                ("arena-2026-09-06-active", True, None, None),
                ("arena-2026-09-06-cancelled", False, None, None),
                ("arena-2026-09-06-legacy", False, None, None),
                ("arena-2026-09-06-scored", True, None, None),
            ]
    finally:
        control.close()
        generator.close()
