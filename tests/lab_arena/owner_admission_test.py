"""Coldkey admission for opt-in Arena integrity-policy rounds."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from lab_arena.chain import MetagraphSnapshot
from lab_arena.owner_admission import OwnerAdmission, resolve_finalized_owner
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_source_migration_postgres import (
    _hotkey,
    _round_config,
)

MIGRATION = "211-lab-arena-owner-admission.sql"
BLOCK_HASH = "0x" + "a" * 64


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(DEFAULT_MIGRATIONS + (MIGRATION,))


@pytest.fixture()
def store(database):
    psycopg2, dsn = database
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    yield ArenaStore(transport)
    transport.close()


def _doc(round_id: str, submission_id: str, size: int = 123):
    return {
        "source_ref": f"arena/{round_id}/sources/{submission_id}.tar.gz",
        "source_size_bytes": size,
        "consent": {"public_rerun": True},
    }


def _open(store: ArenaStore, suffix: str, *, integrity: bool = True, cutoff: str = "2099-01-01T00:00:00Z"):
    round_id = f"arena-2098-02-01-{suffix}"
    config = _round_config(round_id, cutoff=cutoff)
    if integrity:
        config["integrity_policy"] = "arena_integrity_v1"
    assert store.create_round(round_id, config)["status"] == "created"
    return round_id


def _admission(owner: str, *, number: int = 123, block_hash: str = BLOCK_HASH):
    return OwnerAdmission(owner, number, block_hash)


def test_finalized_owner_resolution_returns_snapshot_reference():
    hotkey = _hotkey("owner-resolution-hotkey")
    owner = _hotkey("owner-resolution-coldkey")
    snapshot = MetagraphSnapshot(
        netuid=71,
        block_number=123,
        block_hash=BLOCK_HASH,
        hotkeys=(hotkey,),
        coldkeys=(owner,),
        validator_permit=(False,),
    )
    calls = []
    chain = SimpleNamespace(
        metagraph=lambda *, finalized: calls.append(finalized) or snapshot
    )
    assert resolve_finalized_owner(chain, hotkey) == _admission(owner)
    assert calls == [True]


def test_service_uses_finalized_owner_only_for_opted_in_round():
    hotkey = _hotkey("owner-service-hotkey")
    owner = _hotkey("owner-service-coldkey")
    snapshot = MetagraphSnapshot(
        netuid=71,
        block_number=123,
        block_hash=BLOCK_HASH,
        hotkeys=(hotkey,),
        coldkeys=(owner,),
        validator_permit=(False,),
    )
    captured = {}

    class Store:
        @staticmethod
        def register_submission(*args, **kwargs):
            captured.update(kwargs)
            return {
                "status": "registered",
                "submission_id": args[1],
                "source_ref": args[3]["source_ref"],
            }

    service = object.__new__(ArenaService)
    service._clock = lambda: datetime(2098, 1, 1, tzinfo=timezone.utc)
    service._store = Store()
    service._objects = SimpleNamespace(
        presign_put=lambda *_args, **_kwargs: {
            "upload_url": "https://uploads.example/source",
            "upload_headers": {},
            "expires_in_seconds": 900,
        }
    )
    service._config = SimpleNamespace(
        chain=SimpleNamespace(
            metagraph=lambda *, finalized: snapshot,
            uid_for_hotkey=lambda _hotkey: pytest.fail("legacy lookup used"),
        )
    )
    service._request_round = lambda *_args, **_kwargs: (
        {
            "hotkey": hotkey,
            "body": {
                "source_size_bytes": 123,
                "consent": {"public_rerun": True},
            },
        },
        {
            "round_id": "arena-2098-01-01",
            "status": "open",
            "configuration_doc": {
                "integrity_policy": "arena_integrity_v1",
                "baseline_hotkey": _hotkey("owner-service-baseline"),
                "schedule": {
                    "submission_open": "2097-01-01T00:00:00Z",
                    "submission_cutoff": "2099-01-01T00:00:00Z",
                },
            },
        },
    )
    service.handle_submission_presign({})
    assert captured["owner_admission"] == _admission(owner)


def test_historical_round_keeps_legacy_registration(store, database):
    round_id = _open(store, "historical", integrity=False)
    hotkey = _hotkey("owner-historical-hotkey")
    assert store.register_submission(
        round_id, "sub-historical", hotkey, _doc(round_id, "sub-historical")
    )["status"] == "registered"
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT owner_coldkey, owner_block_number, owner_block_hash "
            "FROM public.lab_arena_submissions WHERE submission_id=%s",
            ("sub-historical",),
        )
        assert cursor.fetchone() == (None, None, None)


def test_integrity_round_rejects_legacy_and_null_owner_rpc(store):
    round_id = _open(store, "nullbypass")
    hotkey = _hotkey("owner-null-hotkey")
    doc = _doc(round_id, "sub-null")
    with pytest.raises(ArenaStoreError, match="owner_admission_required"):
        store.register_submission(round_id, "sub-null", hotkey, doc)
    forged_baseline = _doc(round_id, "sub-forged-baseline")
    forged_baseline["is_king"] = True
    with pytest.raises(ArenaStoreError, match="baseline_identity_invalid"):
        store.register_submission(
            round_id, "sub-forged-baseline", hotkey, forged_baseline
        )
    with pytest.raises(ArenaStoreError, match="owner_admission_required"):
        store._transport.rpc(
            "lab_arena_register_submission_v2",
            {
                "p_round_id": round_id,
                "p_submission_id": "sub-null-v2",
                "p_miner_hotkey": hotkey,
                "p_doc": _doc(round_id, "sub-null-v2"),
                "p_owner_coldkey": None,
                "p_owner_block_number": None,
                "p_owner_block_hash": None,
            },
        )
    assert store.list_submissions(round_id) == []


def test_one_owner_cannot_enter_through_two_hotkeys(store):
    round_id = _open(store, "siblings")
    owner = _hotkey("owner-siblings-coldkey")
    first = _hotkey("owner-siblings-first")
    second = _hotkey("owner-siblings-second")
    assert store.register_submission(
        round_id,
        "sub-sibling-first",
        first,
        _doc(round_id, "sub-sibling-first"),
        owner_admission=_admission(owner),
    )["status"] == "registered"
    with pytest.raises(ArenaStoreError, match="owner_active_submission"):
        store.register_submission(
            round_id,
            "sub-sibling-second",
            second,
            _doc(round_id, "sub-sibling-second"),
            owner_admission=_admission(owner, number=124, block_hash="0x" + "b" * 64),
        )


def test_retry_and_replacement_keep_first_owner_reference(store, database):
    round_id = _open(store, "fixed")
    hotkey = _hotkey("owner-fixed-hotkey")
    owner = _hotkey("owner-fixed-coldkey")
    original = _admission(owner)
    assert store.register_submission(
        round_id, "sub-fixed-first", hotkey, _doc(round_id, "sub-fixed-first"),
        owner_admission=original,
    )["status"] == "registered"
    retry = store.register_submission(
        round_id, "sub-fixed-retry", hotkey, _doc(round_id, "sub-fixed-retry"),
        owner_admission=_admission(owner, number=124, block_hash="0x" + "b" * 64),
    )
    assert retry["status"] == "existing"
    replacement = store.register_submission(
        round_id, "sub-fixed-next", hotkey, _doc(round_id, "sub-fixed-next", 456),
        owner_admission=_admission(owner, number=125, block_hash="0x" + "c" * 64),
    )
    assert replacement["status"] == "registered"
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT owner_coldkey, owner_block_number, owner_block_hash "
            "FROM public.lab_arena_submissions WHERE submission_id=%s",
            ("sub-fixed-next",),
        )
        assert cursor.fetchone() == (owner, 123, BLOCK_HASH)


def test_owner_change_cannot_replace_or_mutate_reservation(store, database):
    round_id = _open(store, "immutable")
    hotkey = _hotkey("owner-immutable-hotkey")
    owner = _hotkey("owner-immutable-coldkey")
    other_owner = _hotkey("owner-immutable-other")
    store.register_submission(
        round_id, "sub-immutable", hotkey, _doc(round_id, "sub-immutable"),
        owner_admission=_admission(owner),
    )
    with pytest.raises(ArenaStoreError, match="submission_owner_changed"):
        store.register_submission(
            round_id, "sub-immutable-next", hotkey, _doc(round_id, "sub-immutable-next", 456),
            owner_admission=_admission(other_owner, number=124),
        )
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        with pytest.raises(psycopg2.Error, match="submission_owner_immutable"):
            cursor.execute(
                "UPDATE public.lab_arena_submissions SET owner_coldkey=%s "
                "WHERE submission_id=%s",
                (other_owner, "sub-immutable"),
            )


def test_sibling_hotkey_race_admits_exactly_one_owner_entry(store, database):
    round_id = _open(store, "race")
    owner = _hotkey("owner-race-coldkey")
    psycopg2, dsn = database

    def submit(index: int):
        transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
        contender = ArenaStore(transport)
        try:
            submission_id = f"sub-owner-race-{index}"
            return contender.register_submission(
                round_id,
                submission_id,
                _hotkey(f"owner-race-hotkey-{index}"),
                _doc(round_id, submission_id),
                owner_admission=_admission(owner),
            )
        except ArenaStoreError as exc:
            return {"error": str(exc)}
        finally:
            transport.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, range(2)))
    assert sum(row.get("status") == "registered" for row in results) == 1
    assert sum("owner_active_submission" in row.get("error", "") for row in results) == 1
    assert len(store.list_submissions(round_id, status="uploading")) == 1


def test_database_cutoff_check_wins_over_owner_admission(store):
    round_id = _open(store, "cutoff", cutoff="2001-01-01T00:00:00Z")
    result = store.register_submission(
        round_id,
        "sub-cutoff",
        _hotkey("owner-cutoff-hotkey"),
        _doc(round_id, "sub-cutoff"),
        owner_admission=_admission(_hotkey("owner-cutoff-coldkey")),
    )
    assert result["status"] == "window_closed"
    assert store.list_submissions(round_id) == []


def test_migration_replay_preserves_rows_and_private_rpc_grant(database):
    psycopg2, dsn = database
    migration = (
        Path(__file__).resolve().parents[2] / "scripts" / MIGRATION
    ).read_text(encoding="utf-8")
    with psycopg2.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(migration)
            cursor.execute(migration)
            cursor.execute(
                "SELECT owner_coldkey, owner_block_number, owner_block_hash "
                "FROM public.lab_arena_submissions WHERE submission_id=%s",
                ("sub-historical",),
            )
            assert cursor.fetchone() == (None, None, None)
            signature = (
                "public.lab_arena_register_submission_v2"
                "(text,text,text,jsonb,text,bigint,text)"
            )
            cursor.execute(
                "SELECT has_function_privilege('lab_arena_service', %s, 'EXECUTE'), "
                "has_function_privilege('anon', %s, 'EXECUTE'), "
                "has_function_privilege('authenticated', %s, 'EXECUTE')",
                (signature, signature, signature),
            )
            assert cursor.fetchone() == (True, False, False)
