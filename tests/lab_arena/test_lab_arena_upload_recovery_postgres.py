"""Source reservation recovery, including replacement/finalize races."""

import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from lab_arena.service import ArenaService, ServiceError
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration
from tests.lab_arena.test_lab_arena_source_migration_postgres import _hotkey, _round_config


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration()


@pytest.fixture
def store(database):
    psycopg2, dsn = database
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    yield ArenaStore(transport)
    transport.close()


def _doc(round_id, submission_id, payload=b"first source"):
    return {
        "source_ref": f"arena/{round_id}/sources/{submission_id}.tar.gz",
        "source_size_bytes": len(payload),
        "source_content_md5": base64.b64encode(hashlib.md5(payload).digest()).decode(),
        "consent": {"public_rerun": True},
    }


def _open(store, suffix):
    round_id = f"arena-2098-01-01-{suffix}"
    store.create_round(round_id, _round_config(round_id, cutoff="2099-01-01T00:00:00Z"))
    return round_id, _hotkey(suffix)


@pytest.mark.parametrize("payload", [b"other source", b"larger replacement source"])
def test_changed_unfinished_upload_uses_new_object_and_preserves_old_row(store, payload):
    round_id, miner = _open(store, "replace" + str(len(payload)))
    first_id = "sub-first-" + str(len(payload))
    new_id = "sub-new-" + str(len(payload))
    original = _doc(round_id, first_id)
    store.register_submission(round_id, first_id, miner, original)
    retry = store.register_submission(round_id, "sub-retry", miner, _doc(round_id, "sub-retry"))
    assert retry["submission_id"] == first_id
    replaced = store.register_submission(round_id, new_id, miner, _doc(round_id, new_id, payload))
    assert replaced["status"] == "registered" and replaced["submission_id"] == new_id
    old = store.get_submission(first_id)
    assert old["status"] == "rejected" and old["rejection_rule"] == "source_replaced"
    assert old["source_ref"] == original["source_ref"]
    assert old["submission_doc"] == original
    assert store.get_submission(new_id)["source_ref"] != old["source_ref"]
    with pytest.raises(ArenaStoreError, match="submission_not_uploading"):
        store.accept_submission_with_credentials(round_id, first_id, miner, {"openrouter": "dGVzdA==", "deepline": "dGVzdA=="})


def test_accepted_submission_cannot_be_replaced_even_at_same_size(store):
    round_id, miner = _open(store, "accepted")
    store.register_submission(round_id, "sub-accepted", miner, _doc(round_id, "sub-accepted"))
    store.accept_submission_with_credentials(round_id, "sub-accepted", miner, {"openrouter": "dGVzdA==", "deepline": "dGVzdA=="})
    assert store.register_submission(round_id, "sub-retry", miner, _doc(round_id, "sub-retry"))["submission_id"] == "sub-accepted"
    with pytest.raises(ArenaStoreError, match="submission_conflict"):
        store.register_submission(round_id, "sub-new", miner, _doc(round_id, "sub-new", b"other source"))
    assert store.get_submission("sub-accepted")["status"] == "accepted"


def test_legacy_unfinished_reservation_can_recover_with_checksum(store):
    round_id, miner = _open(store, "legacy")
    old = _doc(round_id, "sub-legacy")
    old.pop("source_content_md5")
    store.register_submission(round_id, "sub-legacy", miner, old)
    result = store.register_submission(round_id, "sub-upgraded", miner, _doc(round_id, "sub-upgraded"))
    assert result["submission_id"] == "sub-upgraded"
    assert store.get_submission("sub-legacy")["rejection_rule"] == "source_replaced"


def test_concurrent_retries_share_one_reservation(store, database):
    round_id, miner = _open(store, "parallel")
    psycopg2, dsn = database
    def submit(index):
        transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
        try:
            submission_id = f"sub-parallel-{index}"
            return ArenaStore(transport).register_submission(round_id, submission_id, miner, _doc(round_id, submission_id))
        finally:
            transport.close()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(submit, range(4)))
    assert len({row["submission_id"] for row in results}) == 1
    assert len(store.list_submissions(round_id, status="uploading")) == 1


def test_same_size_stale_object_fails_checksum_before_archive_or_credentials():
    service = object.__new__(ArenaService)
    service._objects = SimpleNamespace(get_bounded=lambda *_args: b"other source")
    row = {**_doc("arena-2098-01-01", "sub-stale"), "submission_doc": _doc("arena-2098-01-01", "sub-stale")}
    with pytest.raises(ServiceError) as caught:
        service._validate_uploaded_source(row)
    assert caught.value.code == "submission_rejected:source_checksum_mismatch"
