"""PostgreSQL proof for nonpaying reward-predecessor barriers."""

from __future__ import annotations

import json

import pytest

from lab_arena import rewards, signing
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_reward_migration_postgres import (
    BASELINE,
    MINER_A,
    MINER_B,
    _basis,
)
from tests.postgres_migration_harness import SCRIPTS


MIGRATION = "342-lab-arena-reward-predecessor-barrier.sql"
PATCHED_DEFINITION_SHA256 = (
    "7481e80080726d65844f3b101db19745b232304e8f95dfe0c33cd91d71faf090"
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS + (MIGRATION,)
    )


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


def _configuration(netuid: int) -> dict:
    return {
        "mode": "live",
        "network_name": "test",
        "netuid": netuid,
        "rewards_enabled": True,
        "baseline_hotkey": BASELINE,
        "reward_constants": rewards.reward_constants_document(),
    }


def _publication(round_id: str, published_at: str, hotkey: str = "") -> dict:
    crowned = bool(hotkey)
    return {
        "round_id": round_id,
        "published_at": published_at,
        "king_decision": {
            "outcome": "crowned" if crowned else "no_king",
            "king_hotkey": hotkey,
            "winner_submission_id": round_id + "-winner" if crowned else None,
        },
    }


def _insert_activated(
    cursor,
    *,
    round_id: str,
    configuration: dict,
    publication: dict,
    basis: dict,
    signing_key: dict,
    created_at: str,
) -> None:
    cursor.execute(
        "INSERT INTO public.lab_arena_rounds ("
        "round_id,status,configuration_doc,rewards_enabled,publication_doc,"
        "king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,"
        "reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,"
        "published_at,baseline_promoted_at,created_at) VALUES ("
        "%s,'published',%s::jsonb,TRUE,%s::jsonb,%s,%s,%s,%s,%s,%s::jsonb,"
        "%s::jsonb,%s::timestamptz,%s::timestamptz,%s::timestamptz,"
        "%s::timestamptz)",
        (
            round_id,
            json.dumps(configuration),
            json.dumps(publication),
            basis["king_outcome"],
            basis["king_hotkey"] or None,
            basis["king_start_epoch"],
            basis["effective_reward_epoch"],
            basis["reward_basis_hash"],
            json.dumps(basis),
            json.dumps(signing_key),
            publication["published_at"],
            publication["published_at"],
            (
                publication["published_at"]
                if publication["king_decision"]["outcome"] == "crowned"
                else None
            ),
            created_at,
        ),
    )


def _insert_pending(
    cursor,
    *,
    round_id: str,
    configuration: dict,
    publication: dict,
    created_at: str,
) -> None:
    decision = publication["king_decision"]
    cursor.execute(
        "INSERT INTO public.lab_arena_rounds ("
        "round_id,status,configuration_doc,rewards_enabled,publication_doc,"
        "king_outcome,king_hotkey,published_at,baseline_promoted_at,created_at) VALUES ("
        "%s,'published',%s::jsonb,TRUE,%s::jsonb,%s,%s,%s::timestamptz,"
        "%s::timestamptz,%s::timestamptz)",
        (
            round_id,
            json.dumps(configuration),
            json.dumps(publication),
            decision["outcome"],
            decision["king_hotkey"] or None,
            publication["published_at"],
            publication["published_at"] if decision["outcome"] == "crowned" else None,
            created_at,
        ),
    )


def test_migration_is_idempotent_and_hash_bound(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute((SCRIPTS / MIGRATION).read_text(encoding="utf-8"))
            cursor.execute(
                "SELECT pg_catalog.encode(extensions.digest("
                "pg_catalog.pg_get_functiondef("
                "'public.lab_arena_activate_reward(text,jsonb,jsonb)'::"
                "pg_catalog.regprocedure),'sha256'),'hex')"
            )
            assert cursor.fetchone()[0] == PATCHED_DEFINITION_SHA256


@pytest.mark.parametrize(
    ("barrier_outcome", "netuid", "suffix"),
    (
        ("no_king", 401, "nk"),
        ("retained_ineligible", 402, "ri"),
    ),
)
def test_latest_nonpaying_basis_blocks_revival_but_allows_a_new_winner(
    connections, barrier_outcome, netuid, suffix
):
    store, control = connections
    signer = signing.LocalSigner.generate()
    signing_key = signing.signing_key_document(signer.public_key_der)
    configuration = _configuration(netuid)

    crown_id = "arena-2026-10-01-%scrown" % suffix
    crown_at = "2026-10-01T00:00:00Z"
    crown_publication = _publication(crown_id, crown_at, MINER_A)
    crown_basis = _basis(signer, crown_id, crown_at, 100, "crowned", MINER_A)

    ordinary_id = "arena-2026-10-02-%snormal" % suffix
    ordinary_at = "2026-10-02T00:00:00Z"
    ordinary_publication = _publication(ordinary_id, ordinary_at)

    barrier_id = "arena-2026-10-03-%sbarrier" % suffix
    barrier_at = "2026-10-03T00:00:00Z"
    barrier_publication = _publication(barrier_id, barrier_at)

    later_id = "arena-2026-10-04-%slater" % suffix
    later_at = "2026-10-04T00:00:00Z"
    later_publication = _publication(later_id, later_at)

    winner_id = "arena-2026-10-05-%swinner" % suffix
    winner_at = "2026-10-05T00:00:00Z"
    winner_publication = _publication(winner_id, winner_at, MINER_B)

    with control.cursor() as cursor:
        _insert_activated(
            cursor,
            round_id=crown_id,
            configuration=configuration,
            publication=crown_publication,
            basis=crown_basis,
            signing_key=signing_key,
            created_at="2026-10-01T00:00:01Z",
        )
        _insert_pending(
            cursor,
            round_id=ordinary_id,
            configuration=configuration,
            publication=ordinary_publication,
            created_at="2026-10-02T00:00:01Z",
        )

    ordinary_defense = _basis(
        signer, ordinary_id, ordinary_at, 101, "defended", MINER_A, 100
    )
    assert store.activate_reward(
        ordinary_id, ordinary_defense, signing_key
    )["status"] == "activated"

    barrier_basis = _basis(
        signer,
        barrier_id,
        barrier_at,
        102,
        barrier_outcome,
        MINER_A if barrier_outcome == "retained_ineligible" else "",
        100 if barrier_outcome == "retained_ineligible" else None,
    )
    with control.cursor() as cursor:
        _insert_activated(
            cursor,
            round_id=barrier_id,
            configuration=configuration,
            publication=barrier_publication,
            basis=barrier_basis,
            signing_key=signing_key,
            created_at="2026-10-03T00:00:01Z",
        )
        _insert_pending(
            cursor,
            round_id=later_id,
            configuration=configuration,
            publication=later_publication,
            created_at="2026-10-04T00:00:01Z",
        )

    still_no_king = _basis(signer, later_id, later_at, 103, "no_king", "")
    assert store.activate_reward(
        later_id, still_no_king, signing_key
    )["status"] == "activated"

    with control.cursor() as cursor:
        _insert_pending(
            cursor,
            round_id=winner_id,
            configuration=configuration,
            publication=winner_publication,
            created_at="2026-10-05T00:00:01Z",
        )
    new_winner = _basis(signer, winner_id, winner_at, 104, "crowned", MINER_B)
    assert store.activate_reward(
        winner_id, new_winner, signing_key
    )["status"] == "activated"

    with control.cursor() as cursor:
        cursor.execute(
            "SELECT round_id,reward_basis_doc->>'king_outcome',"
            "reward_basis_doc->>'king_hotkey',king_start_epoch "
            "FROM public.lab_arena_rounds "
            "WHERE arena_network_name='test' AND arena_netuid=%s "
            "ORDER BY effective_reward_epoch",
            (netuid,),
        )
        assert cursor.fetchall() == [
            (crown_id, "crowned", MINER_A, 100),
            (ordinary_id, "defended", MINER_A, 100),
            (
                barrier_id,
                barrier_outcome,
                MINER_A if barrier_outcome == "retained_ineligible" else "",
                100 if barrier_outcome == "retained_ineligible" else 0,
            ),
            (later_id, "no_king", "", 0),
            (winner_id, "crowned", MINER_B, 104),
        ]
