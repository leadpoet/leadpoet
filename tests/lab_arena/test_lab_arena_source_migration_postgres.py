"""Focused PostgreSQL coverage for Arena source admission migration 181."""

from __future__ import annotations

from typing import Any, Dict

import pytest
from bittensor_wallet import Keypair

from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    LAB_ARENA_DAILY_COMPETITION_MIGRATION,
    LAB_ARENA_MIGRATION,
    LAB_ARENA_SOURCE_SUBMISSIONS_MIGRATION,
    database_with_lab_arena_migration,
)
from tests.test_source_add_end_to_end_postgres import SCRIPTS

MIGRATIONS = (
    LAB_ARENA_MIGRATION,
    LAB_ARENA_DAILY_COMPETITION_MIGRATION,
    LAB_ARENA_SOURCE_SUBMISSIONS_MIGRATION,
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(MIGRATIONS)


@pytest.fixture()
def store(database):
    psycopg2, dsn = database
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    yield ArenaStore(transport)
    transport.close()


def _hotkey(label: str) -> str:
    return Keypair.create_from_uri("//" + label).ss58_address


def _round_config(round_id: str, *, cutoff: str) -> Dict[str, Any]:
    return {
        "round_id": round_id,
        "mode": "shadow",
        "rewards_enabled": False,
        "schedule": {
            "submission_open": "2000-01-01T00:00:00Z",
            "submission_cutoff": cutoff,
        },
        "runner_hotkeys": [_hotkey("source-runner")],
        "baseline_hotkey": _hotkey("source-baseline"),
    }


def _source_doc(round_id: str, submission_id: str, seed: str) -> Dict[str, Any]:
    return {
        "source_ref": "arena/%s/sources/%s.tar.gz" % (round_id, submission_id),
        "source_sha256": "sha256:" + seed * 64,
        "source_size_bytes": 123,
        "consent": {"public_rerun": True},
    }


def test_one_active_slot_reuses_a_matching_upload_reservation(store):
    round_id = "arena-2098-01-01-source"
    miner = _hotkey("source-miner-one")
    assert store.create_round(
        round_id, _round_config(round_id, cutoff="2099-01-01T00:00:00Z")
    )["status"] == "created"
    first = store.register_submission(
        round_id, "sub-first", miner, _source_doc(round_id, "sub-first", "a")
    )
    assert first["status"] == "registered"
    retry = store.register_submission(
        round_id, "sub-retry", miner, _source_doc(round_id, "sub-retry", "a")
    )
    assert retry == {
        "status": "existing",
        "submission_status": "uploading",
        "submission_id": "sub-first",
        "source_ref": "arena/%s/sources/sub-first.tar.gz" % round_id,
    }
    with pytest.raises(ArenaStoreError, match="lab_arena_submission_conflict"):
        store.register_submission(
            round_id,
            "sub-other",
            miner,
            _source_doc(round_id, "sub-other", "b"),
        )
    assert store.update_submission(
        round_id, "sub-first", "uploading", "accepted"
    )["status"] == "ok"


def test_public_baseline_can_use_the_same_source_admission_after_cutoff(store):
    round_id = "arena-2001-01-01-source"
    assert store.create_round(
        round_id, _round_config(round_id, cutoff="2001-01-01T01:00:00Z")
    )["status"] == "created"
    miner_result = store.register_submission(
        round_id,
        "sub-late",
        _hotkey("source-late-miner"),
        _source_doc(round_id, "sub-late", "c"),
    )
    assert miner_result["status"] == "window_closed"
    baseline_doc = _source_doc(round_id, "baseline-2001-01-01", "d")
    baseline_doc["is_king"] = True
    baseline = store.register_submission(
        round_id,
        "baseline-2001-01-01",
        _hotkey("source-baseline"),
        baseline_doc,
    )
    assert baseline["status"] == "registered"
    assert store.update_submission(
        round_id, "baseline-2001-01-01", "uploading", "accepted"
    )["status"] == "ok"


def test_source_migration_applies_twice(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                (SCRIPTS / LAB_ARENA_SOURCE_SUBMISSIONS_MIGRATION).read_text(
                    encoding="utf-8"
                )
            )
    finally:
        connection.close()
