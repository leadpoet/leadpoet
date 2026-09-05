"""PostgreSQL coverage for miner-only Arena reward activation."""

from __future__ import annotations

import json

import pytest

from lab_arena import contracts, rewards, signing
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    LAB_ARENA_DAILY_COMPETITION_MIGRATION,
    LAB_ARENA_MIGRATION,
    database_with_lab_arena_migration,
)

REWARD_MIGRATION = "183-lab-arena-miner-reward-basis.sql"
BASELINE = "5" + "A" * 47
MINER_A = "5" + "B" * 47
MINER_B = "5" + "C" * 47


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        (LAB_ARENA_MIGRATION, LAB_ARENA_DAILY_COMPETITION_MIGRATION, REWARD_MIGRATION)
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


def _publish(
    store: ArenaStore,
    control,
    round_id: str,
    *,
    miner: str,
    baseline_score: float,
    miner_score: float,
    crowned: bool,
) -> str:
    constants = rewards.reward_constants_document()
    assert store.create_round(
        round_id,
        {
            "mode": "live",
            "rewards_enabled": True,
            "baseline_hotkey": BASELINE,
            "reward_constants": constants,
        },
    )["status"] == "created"
    participants = [
        {"submission_id": round_id + "-baseline", "miner_hotkey": BASELINE, "is_king": True},
        {"submission_id": round_id + "-miner", "miner_hotkey": miner, "is_king": False},
    ]
    with control.cursor() as cursor:
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status = 'scored', "
            "participants = %s::jsonb, finalists = '[]'::jsonb WHERE round_id = %s",
            (json.dumps(participants), round_id),
        )
    winner_id = round_id + "-miner"
    baseline_id = round_id + "-baseline"
    ranking = [
        {
            "rank": 1 if crowned else 2,
            "submission_id": winner_id,
            "final_score": miner_score,
            "is_baseline": False,
        },
        {
            "rank": 2 if crowned else 1,
            "submission_id": baseline_id,
            "final_score": baseline_score,
            "is_baseline": True,
        },
    ]
    ranking.sort(key=lambda row: row["rank"])
    decision = {
        "outcome": "crowned" if crowned else "no_king",
        "king_submission_id": winner_id if crowned else None,
        "king_hotkey": miner if crowned else "",
        "winner_submission_id": winner_id if crowned else None,
    }
    published_at = "2026-09-04T00:00:00Z"
    publication = {
        "schema_version": contracts.PUBLICATION_SCHEMA_VERSION,
        "round_id": round_id,
        "participants": [
            {"submission_id": baseline_id, "miner_hotkey": BASELINE, "is_baseline": True},
            {"submission_id": winner_id, "miner_hotkey": miner, "is_baseline": False},
        ],
        "stage1_ranking": [],
        "finalists": [],
        "final_ranking": ranking,
        "king_decision": decision,
        "published_at": published_at,
    }
    result = store.transition_round(
        round_id,
        "scored",
        "published",
        {"publication_doc": publication, "published_at": published_at},
    )
    assert result["status"] == "ok"
    return published_at


def _basis(
    signer: signing.LocalSigner,
    round_id: str,
    published_at: str,
    effective_epoch: int,
    outcome: str,
    hotkey: str,
    previous_start=None,
):
    return signing.sign_document(
        signer,
        rewards.reward_basis_document(
            round_id=round_id,
            published_at=published_at,
            finalized_epoch=effective_epoch - 1,
            king_outcome=outcome,
            king_hotkey=hotkey,
            previous_king_start_epoch=previous_start,
            reward_constants=rewards.reward_constants_document(),
        ),
        hash_field="reward_basis_hash",
    )


def test_activation_carries_the_miner_and_never_pays_the_baseline(connections):
    store, control = connections
    signer = signing.LocalSigner.generate()
    key = signing.signing_key_document(signer.public_key_der)

    first = "arena-2026-09-04-rewarda"
    first_at = _publish(
        store, control, first, miner=MINER_A, baseline_score=50, miner_score=60, crowned=True
    )
    first_basis = _basis(signer, first, first_at, 100, "crowned", MINER_A)
    assert store.activate_reward(first, first_basis, key)["status"] == "activated"

    second = "arena-2026-09-04-rewardb"
    second_at = _publish(
        store, control, second, miner=MINER_B, baseline_score=50, miner_score=40, crowned=False
    )
    baseline_basis = _basis(signer, second, second_at, 101, "crowned", BASELINE)
    with pytest.raises(ArenaStoreError, match="reward_activation_invalid"):
        store.activate_reward(second, baseline_basis, key)
    defended = _basis(signer, second, second_at, 101, "defended", MINER_A, 100)
    assert store.activate_reward(second, defended, key)["status"] == "activated"

    third = "arena-2026-09-04-rewardc"
    third_at = _publish(
        store, control, third, miner=MINER_A, baseline_score=50, miner_score=60, crowned=True
    )
    same_miner = _basis(signer, third, third_at, 102, "defended", MINER_A, 100)
    assert store.activate_reward(third, same_miner, key)["status"] == "activated"

    fourth = "arena-2026-09-04-rewardd"
    fourth_at = _publish(
        store, control, fourth, miner=MINER_B, baseline_score=50, miner_score=60, crowned=True
    )
    new_miner = _basis(signer, fourth, fourth_at, 103, "crowned", MINER_B)
    assert store.activate_reward(fourth, new_miner, key)["status"] == "activated"

    with control.cursor() as cursor:
        cursor.execute(
            "SELECT round_id, king_outcome, king_hotkey, king_start_epoch "
            "FROM public.lab_arena_reward_basis_v1 ORDER BY effective_reward_epoch"
        )
        assert cursor.fetchall() == [
            (first, "crowned", MINER_A, 100),
            (second, "defended", MINER_A, 100),
            (third, "defended", MINER_A, 100),
            (fourth, "crowned", MINER_B, 103),
        ]


def test_publication_rejects_no_winner_when_a_miner_beat_the_baseline(connections):
    store, control = connections
    with pytest.raises(ArenaStoreError, match="publication_winner_missing"):
        _publish(
            store,
            control,
            "arena-2026-09-04-badwinner",
            miner=MINER_A,
            baseline_score=50,
            miner_score=60,
            crowned=False,
        )
