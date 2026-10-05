"""Current-schema PostgreSQL proof of source-aware Arena admission."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import timedelta
from pathlib import Path

import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration,
)
from tests.lab_arena.queued_submission_replacement_postgres_test import _config, _doc
from tests.lab_arena.test_lab_arena_source_migration_postgres import _hotkey


MIGRATION = "405-lab-arena-submission-duplicates.sql"
CREDS = {"openrouter": "dGVzdA==", "deepline": "dGVzdA=="}


@pytest.fixture(scope="module")
def database():
    migrations = tuple(value for value in CURRENT_SERVICE_MIGRATIONS if value != MIGRATION)
    yield from database_with_lab_arena_migration(migrations + (MIGRATION,))


@pytest.fixture
def store(database):
    psycopg2, dsn = database
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    yield ArenaStore(transport)
    transport.close()


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _current_config(round_id: str):
    config = _config(round_id, cutoff_after=timedelta(hours=3))
    config.update(stage_1_icp_count=10, stage_2_icp_count=10)
    return config


def _register(store, round_id: str, submission_id: str, hotkey: str):
    size = 123 + hashlib.sha256(submission_id.encode()).digest()[0]
    assert store.register_submission(
        round_id, submission_id, hotkey, _doc(round_id, submission_id, size)
    )["status"] == "registered"


def _accept(store, round_id: str, submission_id: str, hotkey: str,
            archive: str, normalized: str):
    return store.accept_submission_source_with_credentials(
        round_id, submission_id, hotkey, CREDS, _hash(archive), _hash(normalized)
    )


def test_duplicate_rejection_keeps_same_hotkey_fallback(store):
    round_id = "arena-2098-07-01-duplicate"
    a, b, c = (_hotkey(value) for value in
               ("duplicate-a", "duplicate-b", "duplicate-c"))
    store.create_round(round_id, _current_config(round_id))
    assert store.submission_similarity_schema()["version"] == 405
    _register(store, round_id, "sub-duplicate-a", a)
    first = _accept(store, round_id, "sub-duplicate-a", a, "archive-a", "same-code")
    assert first["status"] == "ok"
    initial = store.get_submission("sub-duplicate-a")
    assert initial["accepted_at"] is not None
    assert initial["source_archive_sha256"] == _hash("archive-a")
    assert _accept(store, round_id, "sub-duplicate-a", a, "archive-a", "same-code")["status"] == "existing"
    with pytest.raises(ArenaStoreError, match="lab_arena_source_digest_conflict"):
        _accept(store, round_id, "sub-duplicate-a", a, "changed", "same-code")

    _register(store, round_id, "sub-duplicate-b", b)
    assert _accept(store, round_id, "sub-duplicate-b", b, "different-tar", "same-code")["status"] == "rejected_duplicate"
    rejected = store.get_submission("sub-duplicate-b")
    assert (rejected["status"], rejected["rejection_rule"]) == ("rejected", "duplicate_submission")
    assert rejected["source_archive_sha256"] is None
    assert store.get_submission_credential("sub-duplicate-b", b, "openrouter") is None
    with pytest.raises(ArenaStoreError, match="lab_arena_submission_not_uploading"):
        _accept(store, round_id, "sub-duplicate-b", b, "different-tar", "same-code")

    _register(store, round_id, "sub-duplicate-c-good", c)
    assert _accept(store, round_id, "sub-duplicate-c-good", c, "new-tar", "new-code")["status"] == "ok"
    _register(store, round_id, "sub-duplicate-c-replacement", c)
    assert _accept(store, round_id, "sub-duplicate-c-replacement", c, "duplicate-tar", "same-code")["status"] == "rejected_duplicate"
    assert store.get_submission("sub-duplicate-c-good")["status"] == "accepted"
    assert store.get_submission_credential("sub-duplicate-c-good", c, "openrouter") is not None


def test_same_hotkey_replacement_and_concurrent_cross_hotkey_winner(store, database):
    round_id = "arena-2098-07-02-duplicate"
    a, b, c = (_hotkey(value) for value in ("winner-a", "winner-b", "winner-c"))
    store.create_round(round_id, _current_config(round_id))
    _register(store, round_id, "sub-winner-a-old", a)
    assert _accept(store, round_id, "sub-winner-a-old", a, "old-archive", "old-code")["status"] == "ok"
    _register(store, round_id, "sub-winner-a-new", a)
    assert _accept(store, round_id, "sub-winner-a-new", a, "new-archive", "new-code")["status"] == "ok"
    assert store.get_submission("sub-winner-a-old")["status"] == "rejected"
    _register(store, round_id, "sub-winner-b", b)
    _register(store, round_id, "sub-winner-c", c)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda item: _accept(store, round_id, item[0], item[1], item[2], "race-code"),
            (("sub-winner-b", b, "race-b"), ("sub-winner-c", c, "race-c")),
        ))
    assert sorted(result["status"] for result in results) == ["ok", "rejected_duplicate"]
    assert sum(store.get_submission(sid)["status"] == "accepted"
               for sid in ("sub-winner-b", "sub-winner-c")) == 1
    winner = next(store.get_submission(sid) for sid in ("sub-winner-b", "sub-winner-c")
                  if store.get_submission(sid)["status"] == "accepted")
    assert winner["accepted_at"] is not None


def test_migration_replay_and_service_only_acl(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(Path(__file__).resolve().parents[2].joinpath("scripts", MIGRATION).read_text())
        cursor.execute("SELECT has_function_privilege('anon', %s, 'EXECUTE')", (
            "public.lab_arena_accept_submission_source_with_credentials(text,text,text,jsonb,text,text)",
        ))
        assert cursor.fetchone()[0] is False
        cursor.execute("SELECT has_function_privilege('lab_arena_service', %s, 'EXECUTE')", (
            "public.lab_arena_accept_submission_source_with_credentials(text,text,text,jsonb,text,text)",
        ))
        assert cursor.fetchone()[0] is True
        cursor.execute("SELECT has_function_privilege('anon', %s, 'EXECUTE')", (
            "public.lab_arena_submission_similarity_champion(text)",
        ))
        assert cursor.fetchone()[0] is False
        cursor.execute("SELECT has_function_privilege('lab_arena_service', %s, 'EXECUTE')", (
            "public.lab_arena_submission_similarity_champion(text)",
        ))
        assert cursor.fetchone()[0] is True


def test_champion_lookup_and_admission_time_immutable(store, database):
    round_id = "arena-2098-07-04-duplicate"
    miner = _hotkey("admission-time-owner")
    store.create_round(round_id, _current_config(round_id))
    assert store.submission_similarity_champion(round_id) == {
        "status": "ready", "submission_id": None,
    }
    _register(store, round_id, "sub-admission-time", miner)
    assert _accept(store, round_id, "sub-admission-time", miner, "time-tar", "time-code")["status"] == "ok"
    psycopg2, dsn = database
    accepted_at = store.get_submission("sub-admission-time")["accepted_at"]
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE public.lab_arena_submissions SET accepted_at = "
            "accepted_at + INTERVAL '1 second' WHERE submission_id=%s",
            ("sub-admission-time",),
        )
    assert store.get_submission("sub-admission-time")["accepted_at"] == accepted_at
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET champion_funding_frozen=TRUE "
            "WHERE round_id=%s", (round_id,),
        )
    assert store.submission_similarity_champion(round_id) == {
        "status": "ready", "submission_id": None,
    }


def test_champion_lookup_waits_for_eligible_promotion(store, database):
    prior_id = "arena-2098-07-05-duplicate"
    current_id = "arena-2098-07-06-duplicate"
    store.create_round(prior_id, _current_config(prior_id))
    store.create_round(current_id, _current_config(current_id))
    prior_hotkey = _hotkey("promoted-prior-owner")
    _register(store, prior_id, "sub-promoted-prior", prior_hotkey)
    assert _accept(store, prior_id, "sub-promoted-prior", prior_hotkey,
                   "prior-tar", "prior-code")["status"] == "ok"
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE public.lab_arena_submissions SET status='frozen', "
            "code_review_status='passed', code_review_attempts=1, "
            "code_review_claim=%s, code_review_started_at=clock_timestamp(), "
            "code_review_doc=%s::jsonb, frozen_at=clock_timestamp() "
            "WHERE submission_id=%s",
            ("sha256:" + "0" * 64, json.dumps({"passed": True}),
             "sub-promoted-prior"),
        )
        cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='published', "
            "promotion_required=TRUE, "
            "publication_doc=%s::jsonb, published_at=clock_timestamp() "
            "WHERE round_id=%s",
            (json.dumps({"king_decision": {"outcome": "crowned",
                "winner_submission_id": "sub-promoted-prior",
                "king_hotkey": prior_hotkey}}), prior_id),
        )
        cursor.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER")
    assert store.submission_similarity_champion(current_id) == {
        "status": "promotion_pending", "submission_id": None,
    }
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET baseline_promoted_at=clock_timestamp() "
            "WHERE round_id=%s", (prior_id,),
        )
        cursor.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER")
    assert store.submission_similarity_champion(current_id) == {
        "status": "ready", "submission_id": "sub-promoted-prior",
    }
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET configuration_doc="
            "jsonb_set(configuration_doc,'{mode}','\"live\"'::jsonb) WHERE round_id=%s",
            (current_id,),
        )
        cursor.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER")
    assert store.submission_similarity_champion(current_id) == {
        "status": "ready", "submission_id": None,
    }


def test_migration_preserves_existing_accepted_row_and_review_permissions():
    migrations = tuple(value for value in CURRENT_SERVICE_MIGRATIONS if value != MIGRATION)
    with closing(database_with_lab_arena_migration(migrations)) as fixture:
        psycopg2, dsn = next(fixture)
        transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
        store = ArenaStore(transport)
        round_id = "arena-2098-07-03-duplicate"
        miner = _hotkey("legacy-source-owner")
        store.create_round(round_id, _current_config(round_id))
        _register(store, round_id, "sub-legacy-digest", miner)
        assert store.accept_submission_with_credentials(
            round_id, "sub-legacy-digest", miner, CREDS
        )["status"] == "ok"
        before = store.get_submission("sub-legacy-digest")
        frozen_hotkey = _hotkey("legacy-frozen-owner")
        _register(store, round_id, "sub-legacy-frozen", frozen_hotkey)
        assert store.accept_submission_with_credentials(
            round_id, "sub-legacy-frozen", frozen_hotkey, CREDS
        )["status"] == "ok"
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_submissions SET status='frozen', "
                "code_review_status='passed', code_review_attempts=1, "
                "code_review_claim=%s, code_review_started_at=clock_timestamp(), "
                "code_review_doc=%s::jsonb, frozen_at=clock_timestamp() "
                "WHERE submission_id=%s",
                ("sha256:" + "0" * 64, json.dumps({"passed": True}),
                 "sub-legacy-frozen"),
            )
        frozen_before = store.get_submission("sub-legacy-frozen")
        migration = Path(__file__).resolve().parents[2].joinpath("scripts", MIGRATION).read_text()
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute(migration)
            cursor.execute("SELECT has_function_privilege('anon', %s, 'EXECUTE')", (
                "public.lab_arena_submission_duplicate_schema_v1()",
            ))
            assert cursor.fetchone()[0] is False
            cursor.execute("SELECT has_function_privilege('lab_arena_service', %s, 'EXECUTE')", (
                "public.lab_arena_submission_duplicate_schema_v1()",
            ))
            assert cursor.fetchone()[0] is True
        after = store.get_submission("sub-legacy-digest")
        assert after["status"] == before["status"] == "accepted"
        assert after["source_ref"] == before["source_ref"]
        assert after["source_archive_sha256"] is None
        assert after["source_normalized_sha256"] is None
        assert after["accepted_at"] == before["accepted_at"]
        assert store.get_submission_credential("sub-legacy-digest", miner, "openrouter") is not None
        frozen_after = store.get_submission("sub-legacy-frozen")
        assert frozen_after["status"] == frozen_before["status"] == "frozen"
        assert frozen_after["source_ref"] == frozen_before["source_ref"]
        assert frozen_after["accepted_at"] == frozen_before["accepted_at"]
        transport.close()
