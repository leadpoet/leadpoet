"""Raise only the former daily admission default without rewriting frozen work."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_arena.store import ArenaStoreError
from tests.lab_arena.hotkey_admission_postgres_test import (
    _accept, _config, _register, _store, hotkey,
)
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)


MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "scripts/395-lab-arena-per-miner-daily-admission.sql"
).read_text()


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _apply(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION)


def test_upgrade_admits_twenty_first_without_changing_existing_entries(database):
    store = _store(database)
    round_id = "arena-2098-10-04"
    config = _config(round_id, cap=20)
    assert store.create_round(round_id, config)["status"] == "created"
    owner = hotkey("daily-admission-owner")
    entries = []
    for index in range(21):
        submission_id = f"sub-daily-admission-{index}"
        miner = hotkey(f"daily-admission-{index}")
        assert _register(store, round_id, submission_id, miner, owner)["status"] == "registered"
        entries.append((submission_id, miner))
        if index < 20:
            assert _accept(store, round_id, submission_id, miner)["status"] == "ok"
    with pytest.raises(ArenaStoreError, match="lab_arena_round_full"):
        _accept(store, round_id, *entries[-1])
    rows_before = store.list_submissions(round_id)
    credentials_before = [
        store.get_submission_credential(submission_id, miner, "openrouter")
        for submission_id, miner in entries[:-1]
    ]

    _apply(database)
    _apply(database)
    assert store.get_round(round_id)["configuration_doc"] == {
        **config, "max_challengers": 256,
    }
    assert store.list_submissions(round_id) == rows_before
    assert [
        store.get_submission_credential(submission_id, miner, "openrouter")
        for submission_id, miner in entries[:-1]
    ] == credentials_before
    assert _accept(store, round_id, *entries[-1])["status"] == "ok"
    # A retry by the same miner still resolves to its existing model.
    assert _register(store, round_id, "duplicate-daily-entry", entries[0][1], owner)["submission_id"] == entries[0][0]
    accepted = store.list_submissions(round_id, status="accepted")
    assert len(accepted) == len({row["miner_hotkey"] for row in accepted}) == 21

    store._transport.close()
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT tgenabled FROM pg_trigger "
            "WHERE tgrelid='public.lab_arena_rounds'::regclass "
            "AND tgname='lab_arena_rounds_write_once'"
        )
        assert cursor.fetchone() == ("O",)
        with pytest.raises(psycopg2.Error, match="write-once"):
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET configuration_doc="
                "jsonb_set(configuration_doc,'{max_challengers}','100') "
                "WHERE round_id=%s", (round_id,),
            )


def test_full_miner_set_keeps_one_current_entry_per_hotkey_and_cap(database):
    store = _store(database)
    round_id = "arena-2098-10-05"
    assert store.create_round(round_id, _config(round_id, cap=256))["status"] == "created"
    owner = hotkey("full-daily-admission-owner")
    first = None
    for index in range(256):
        submission_id = f"sub-full-daily-{index}"
        miner = hotkey(f"full-daily-miner-{index}")
        assert _register(store, round_id, submission_id, miner, owner)["status"] == "registered"
        assert _accept(store, round_id, submission_id, miner)["status"] == "ok"
        if index == 0:
            first = (submission_id, miner)
    assert first is not None
    assert _register(store, round_id, "sub-full-daily-retry", first[1], owner)["submission_id"] == first[0]
    extra = "sub-full-daily-overflow"
    extra_miner = hotkey("full-daily-overflow-miner")
    assert _register(store, round_id, extra, extra_miner, owner)["status"] == "registered"
    with pytest.raises(ArenaStoreError, match="lab_arena_round_full"):
        _accept(store, round_id, extra, extra_miner)
    accepted = store.list_submissions(round_id, status="accepted")
    assert len(accepted) == len({row["miner_hotkey"] for row in accepted}) == 256
    store._transport.close()


def test_upgrade_preserves_started_expired_custom_and_nonproduction_rounds(database):
    store = _store(database)
    expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    cases = [
        ({"status": "committed"}, {}),
        ({"status": "published"}, {}),
        ({"status": "cancelled"}, {}),
        ({"benchmark_ref": "frozen-bank.json"}, {}),
        ({"participants": [{"submission_id": "frozen-miner"}]}, {}),
        ({}, {"max_challengers": 8}),
        ({}, {"max_challengers": 256}),
        ({}, {"mode": "shadow"}),
        ({}, {"network_name": "test", "netuid": 401}),
        ({}, {"schedule": {"submission_cutoff": expired}}),
        ({}, {"schedule": {}}),
    ]
    ids = []
    psycopg2, dsn = database
    for index, (row_updates, config_updates) in enumerate(cases, start=1):
        round_id = f"arena-2098-11-{index:02d}"
        assert store.create_round(round_id, _config(round_id, cap=20, **config_updates))["status"] == "created"
        if row_updates:
            with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
                cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER")
                for column, value in row_updates.items():
                    # Column names come only from the fixed fixture cases above.
                    if column == "participants":
                        from psycopg2.extras import Json
                        value = Json(value)
                    cursor.execute(f"UPDATE public.lab_arena_rounds SET {column}=%s WHERE round_id=%s", (value, round_id))
                cursor.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER")
        ids.append(round_id)

    custom_id = "arena-2098-11-20-custom"
    assert store.create_round(custom_id, _config(custom_id))["status"] == "created"
    ids.append(custom_id)
    run_round = "arena-2098-11-21"
    assert store.create_round(run_round, _config(run_round))["status"] == "created"
    miner = hotkey("daily-admission-started-run")
    owner = hotkey("daily-admission-started-owner")
    assert _register(store, run_round, "sub-daily-started", miner, owner)["status"] == "registered"
    assert _accept(store, run_round, "sub-daily-started", miner)["status"] == "ok"
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO public.lab_arena_runs (run_id, assignment_id, round_id, "
            "submission_id, miner_hotkey, stage, icp_position, attempt) "
            "VALUES ('run-daily-started','assignment-daily-started',%s,'sub-daily-started',%s,1,0,1)",
            (run_round, miner),
        )
    ids.append(run_round)
    before = {round_id: store.get_round(round_id) for round_id in ids}
    _apply(database)
    assert {round_id: store.get_round(round_id) for round_id in ids} == before
    store._transport.close()
