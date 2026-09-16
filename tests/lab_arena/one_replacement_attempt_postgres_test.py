"""Durable single replacement reservation on the current Arena PostgreSQL schema."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_arena.owner_admission import OwnerAdmission
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.hotkey_admission_postgres_test import (
    BLOCK_HASH,
    _config as _integrity_config,
)
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.queued_submission_replacement_postgres_test import (
    _accept,
    _config,
    _doc,
)
from tests.lab_arena.test_lab_arena_source_migration_postgres import _hotkey


MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "scripts/262-lab-arena-one-replacement-attempt.sql"
).read_text()


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


@pytest.fixture
def store(database):
    psycopg2, dsn = database
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    yield ArenaStore(transport)
    transport.close()


def _round(store, round_id):
    assert store.create_round(
        round_id, _config(round_id, cutoff_after=timedelta(hours=3))
    )["status"] == "created"


def _first(store, round_id, hotkey, submission_id):
    assert store.register_submission(
        round_id, submission_id, hotkey, _doc(round_id, submission_id, 123)
    )["status"] == "registered"
    assert _accept(store, round_id, submission_id, hotkey)["status"] == "ok"


def test_rejected_reservation_consumes_allowance_and_preserves_fallback(store, database):
    round_id, hotkey = "arena-2098-06-01-once", _hotkey("once-fallback")
    _round(store, round_id)
    _first(store, round_id, hotkey, "once-original")
    assert store.submission_replacement_schema() == {
        "schema_version": "leadpoet.lab_arena.submission_replacement_schema.v1",
        "version": 258,
        "replacement_freeze_seconds": 3600,
        "max_replacement_attempts": 1,
    }
    assert store.register_submission(
        round_id, "once-candidate", hotkey,
        _doc(round_id, "once-candidate", 124, checksum=b"candidate"),
    )["status"] == "registered"
    # A transport retry of the pending source uses its exact reservation.
    retry = store.register_submission(
        round_id, "once-retry", hotkey,
        _doc(round_id, "once-retry", 124, checksum=b"candidate"),
    )
    assert (retry["status"], retry["submission_id"]) == ("existing", "once-candidate")
    assert store.get_submission("once-retry") is None
    assert store.update_submission(
        round_id, "once-candidate", "uploading", "rejected",
        {"rejection_rule": "source_checksum_mismatch"},
    )["status"] == "ok"
    for submission_id, checksum in (("once-same-terminal", b"candidate"), ("once-distinct", b"different")):
        result = store.register_submission(
            round_id, submission_id, hotkey,
            _doc(round_id, submission_id, 124, checksum=checksum),
        )
        assert result["status"] == "replacement_limit_reached"
        assert result["max_replacement_attempts"] == 1
        assert store.get_submission(submission_id) is None
    original = store.get_submission("once-original")
    candidate = store.get_submission("once-candidate")
    assert original["status"] == "accepted"
    assert candidate["replaces_submission_id"] == "once-original"
    assert candidate["status"] == "rejected"
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM public.lab_arena_submissions "
            "WHERE round_id=%s AND miner_hotkey=%s AND replaces_submission_id IS NOT NULL",
            (round_id, hotkey),
        )
        assert cursor.fetchone()[0] == 1


def test_rejected_original_is_not_a_new_first_submission(store):
    round_id, hotkey = "arena-2098-06-02-root", _hotkey("once-rejected-original")
    _round(store, round_id)
    assert store.register_submission(
        round_id, "root-invalid", hotkey, _doc(round_id, "root-invalid", 123)
    )["status"] == "registered"
    assert store.update_submission(
        round_id, "root-invalid", "uploading", "rejected",
        {"rejection_rule": "source_checksum_mismatch"},
    )["status"] == "ok"
    assert store.register_submission(
        round_id, "root-replacement", hotkey,
        _doc(round_id, "root-replacement", 124),
    )["status"] == "registered"
    assert store.get_submission("root-replacement")["replaces_submission_id"] == "root-invalid"
    assert _accept(store, round_id, "root-replacement", hotkey)["status"] == "ok"
    assert store.register_submission(
        round_id, "root-third", hotkey, _doc(round_id, "root-third", 125)
    )["status"] == "replacement_limit_reached"
    assert store.get_submission("root-invalid")["status"] == "rejected"
    assert store.get_submission("root-replacement")["status"] == "accepted"


def test_accepted_replacement_retry_and_fresh_daily_round(store):
    hotkey = _hotkey("once-new-day")
    old_round, new_round = "arena-2098-06-03-daily", "arena-2098-06-04-daily"
    _round(store, old_round)
    _first(store, old_round, hotkey, "daily-original")
    assert store.register_submission(
        old_round, "daily-replacement", hotkey,
        _doc(old_round, "daily-replacement", 124),
    )["status"] == "registered"
    assert _accept(store, old_round, "daily-replacement", hotkey)["status"] == "ok"
    same = store.register_submission(
        old_round, "daily-transport-retry", hotkey,
        _doc(old_round, "daily-transport-retry", 124),
    )
    assert (same["status"], same["submission_id"]) == ("existing", "daily-replacement")
    assert store.register_submission(
        old_round, "daily-third", hotkey,
        _doc(old_round, "daily-third", 125),
    )["status"] == "replacement_limit_reached"
    _round(store, new_round)
    _first(store, new_round, hotkey, "newday-original")
    assert store.register_submission(
        new_round, "newday-replacement", hotkey,
        _doc(new_round, "newday-replacement", 124),
    )["status"] == "registered"
    assert store.get_submission("newday-replacement")["replaces_submission_id"] == "newday-original"


def test_concurrent_distinct_contenders_reserve_only_one(store, database):
    round_id, hotkey = "arena-2098-06-05-race", _hotkey("once-concurrent")
    _round(store, round_id)
    _first(store, round_id, hotkey, "race-original")
    psycopg2, dsn = database

    def contend(index):
        transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
        contender = ArenaStore(transport)
        submission_id = f"race-candidate-{index}"
        try:
            return contender.register_submission(
                round_id, submission_id, hotkey,
                _doc(round_id, submission_id, 124 + index),
            )
        finally:
            transport.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(contend, (1, 2)))
    assert sorted(result["status"] for result in results) == [
        "registered", "replacement_limit_reached",
    ]
    linked = [store.get_submission(f"race-candidate-{index}") for index in (1, 2)]
    assert sum(row is not None for row in linked) == 1
    assert store.get_submission("race-original")["status"] == "accepted"


def test_owner_change_and_missing_checksum_cannot_reset_allowance(store):
    round_id, hotkey = "arena-2098-06-06-owner", _hotkey("once-owner-hotkey")
    owner, other = _hotkey("once-owner-coldkey"), _hotkey("once-owner-other-coldkey")
    config = _integrity_config(
        round_id,
        contact_policy="contacts_v1",
        intent_details_policy="intent_details_v1",
        scorer_policy={"scoring_adapter_version": "qualification_contacts_v3"},
    )
    config["schedule"]["submission_cutoff"] = (
        datetime.now(timezone.utc) + timedelta(hours=3)
    ).isoformat().replace("+00:00", "Z")
    store.create_round(round_id, config)
    admission = OwnerAdmission(owner, 123, BLOCK_HASH)
    changed = OwnerAdmission(other, 124, BLOCK_HASH)
    assert store.register_submission(
        round_id, "owner-original", hotkey,
        _doc(round_id, "owner-original", 123), owner_admission=admission,
    )["status"] == "registered"
    assert _accept(store, round_id, "owner-original", hotkey)["status"] == "ok"
    assert store.register_submission(
        round_id, "owner-reserved", hotkey,
        _doc(round_id, "owner-reserved", 124, checksum=b"digest"),
        owner_admission=admission,
    )["status"] == "registered"
    assert store.update_submission(
        round_id, "owner-reserved", "uploading", "rejected",
        {"rejection_rule": "source_checksum_mismatch"},
    )["status"] == "ok"
    with pytest.raises(ArenaStoreError, match="lab_arena_submission_owner_changed"):
        store.register_submission(
            round_id, "owner-changed", hotkey,
            _doc(round_id, "owner-changed", 124), owner_admission=changed,
        )
    assert store.register_submission(
        round_id, "owner-no-checksum", hotkey,
        _doc(round_id, "owner-no-checksum", 124), owner_admission=admission,
    )["status"] == "replacement_limit_reached"


def test_legacy_multiple_attempts_remain_and_migration_replay_is_safe():
    # Build real pre-limit history through migration 258, then upgrade in place.
    old_migrations = CURRENT_SERVICE_MIGRATIONS[
        :CURRENT_SERVICE_MIGRATIONS.index("262-lab-arena-one-replacement-attempt.sql")
    ]
    with _legacy_database(old_migrations) as database:
        psycopg2, dsn = database
        transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
        store = ArenaStore(transport)
        round_id, hotkey = "arena-2098-06-08-legacy", _hotkey("once-legacy")
        _round(store, round_id)
        _first(store, round_id, hotkey, "legacy-original")
        for index in (1, 2):
            submission_id = f"legacy-replacement-{index}"
            assert store.register_submission(
                round_id, submission_id, hotkey,
                _doc(round_id, submission_id, 123 + index),
            )["status"] == "registered"
            assert store.update_submission(
                round_id, submission_id, "uploading", "rejected",
                {"rejection_rule": "source_checksum_mismatch"},
            )["status"] == "ok"
        with psycopg2.connect(**dsn) as connection:
            connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT count(*) FROM public.lab_arena_submissions "
                    "WHERE round_id=%s AND replaces_submission_id IS NOT NULL",
                    (round_id,),
                )
                assert cursor.fetchone()[0] == 2
                cursor.execute(MIGRATION)
                cursor.execute(MIGRATION)
                cursor.execute(
                    "SELECT count(*) FROM public.lab_arena_submissions "
                    "WHERE round_id=%s AND replaces_submission_id IS NOT NULL",
                    (round_id,),
                )
                assert cursor.fetchone()[0] == 2
                cursor.execute(
                    "SELECT has_function_privilege('lab_arena_service', "
                    "'public.lab_arena_submission_replacement_schema_v1()', 'EXECUTE'), "
                    "has_function_privilege('service_role', "
                    "'public.lab_arena_submission_replacement_schema_v1()', 'EXECUTE')"
                )
                assert cursor.fetchone() == (True, False)
        assert store.register_submission(
            round_id, "legacy-third", hotkey,
            _doc(round_id, "legacy-third", 126),
        )["status"] == "replacement_limit_reached"
        assert store.get_submission("legacy-original")["status"] == "accepted"
        transport.close()


@contextmanager
def _legacy_database(migrations):
    generator = database_with_lab_arena_migration(migrations)
    try:
        yield next(generator)
    finally:
        generator.close()
