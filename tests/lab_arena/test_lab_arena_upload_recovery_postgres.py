"""Source reservation recovery, including replacement/finalize races."""

import base64
import hashlib
import io
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from pathlib import Path

import pytest

from lab_arena.service import ArenaService, ServiceError
from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS, LAB_ARENA_UPLOAD_RECOVERY_MIGRATION,
    database_with_lab_arena_migration,
)
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


@pytest.mark.parametrize("finalize_first", [True, False])
def test_replacement_and_finalize_serialize_without_replacing_accepted_source(store, database, finalize_first):
    round_id, miner = _open(store, "race" + str(int(finalize_first)))
    old_id, new_id = "sub-old-" + str(int(finalize_first)), "sub-next-" + str(int(finalize_first))
    store.register_submission(round_id, old_id, miner, _doc(round_id, old_id))
    psycopg2, dsn = database
    blocker = psycopg2.connect(**dsn)
    observer = psycopg2.connect(**dsn)
    observer.autocommit = True
    def operation(finalize):
        label = "upload-race-finalize" if finalize else "upload-race-replace"
        transport = PsycopgTransport(lambda: psycopg2.connect(**dsn, application_name=label))
        contender = ArenaStore(transport)
        try:
            if finalize:
                return contender.accept_submission_with_credentials(round_id, old_id, miner, {"openrouter": "dGVzdA==", "deepline": "dGVzdA=="})
            return contender.register_submission(round_id, new_id, miner, _doc(round_id, new_id, b"other source"))
        except ArenaStoreError as exc:
            return {"error": str(exc)}
        finally:
            transport.close()
    def wait_for_lock(finalize):
        label = "upload-race-finalize" if finalize else "upload-race-replace"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with observer.cursor() as cursor:
                cursor.execute("SELECT count(*) FROM pg_catalog.pg_stat_activity WHERE application_name=%s AND wait_event_type='Lock'", (label,))
                if cursor.fetchone()[0]:
                    return
            time.sleep(0.01)
        raise AssertionError("contender did not enter the round lock queue")
    try:
        with blocker.cursor() as cursor:
            cursor.execute("SELECT round_id FROM public.lab_arena_rounds WHERE round_id=%s FOR UPDATE", (round_id,))
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(operation, finalize_first)
            try:
                wait_for_lock(finalize_first)
                second = pool.submit(operation, not finalize_first)
                wait_for_lock(not finalize_first)
            finally:
                blocker.commit()
            first_result, second_result = first.result(timeout=10), second.result(timeout=10)
        assert "error" not in first_result
        assert "error" in second_result
        old = store.get_submission(old_id)
        assert old["status"] == ("accepted" if finalize_first else "rejected")
        assert len([s for s in store.list_submissions(round_id) if s["status"] in ("accepted", "uploading")]) == 1
        if finalize_first:
            assert store.get_submission(new_id) is None
        else:
            assert store.get_submission(new_id)["status"] == "uploading"
    finally:
        blocker.close()
        observer.close()


def test_same_size_stale_object_fails_checksum_before_archive_or_credentials():
    service = object.__new__(ArenaService)
    service._objects = SimpleNamespace(get_bounded=lambda *_args: b"other source")
    row = {**_doc("arena-2098-01-01", "sub-stale"), "submission_doc": _doc("arena-2098-01-01", "sub-stale")}
    with pytest.raises(ServiceError) as caught:
        service._validate_uploaded_source(row)
    assert caught.value.code == "submission_rejected:source_checksum_mismatch"


def test_gateway_source_error_names_file_without_returning_credentials():
    from fastapi.testclient import TestClient
    from lab_arena.api import create_app

    secret = "synthetic-private-api-key"
    name = ".env." + secret
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.size = 4
        archive.addfile(member, io.BytesIO(b"test"))
    payload = raw.getvalue()
    service = object.__new__(ArenaService)
    service._objects = SimpleNamespace(get_bounded=lambda *_args: payload)
    def finalize(_submission_id, _envelope):
        service._validate_uploaded_source(
            {"source_ref": "arena/source", "source_size_bytes": len(payload)},
            forbidden_values=(secret,),
        )
    service.handle_submission_finalize = finalize
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/arena/v1/submissions/sub-path/finalize",
            json={"body": {"submission_id": "sub-path"}},
        )
    assert response.status_code == 400
    assert response.json()["code"] == "submission_rejected:source_contains_credentials"
    assert response.json()["source_path"] == ".env.[REDACTED]"
    assert secret not in response.text


def test_upload_migration_193_is_idempotent_under_hosted_owner(database):
    psycopg2, dsn = database
    migration = (Path(__file__).resolve().parents[2] / "scripts/193-lab-arena-upload-recovery.sql").read_text()
    with psycopg2.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute("CREATE ROLE upload_migrator LOGIN CREATEROLE INHERIT; GRANT lab_arena_owner TO upload_migrator; ALTER SCHEMA public OWNER TO upload_migrator")
            cursor.execute("SELECT count(*) FROM public.lab_arena_submissions")
            before = cursor.fetchone()[0]
            cursor.execute("SET ROLE upload_migrator")
            cursor.execute(migration)
            cursor.execute(migration)
            cursor.execute("RESET ROLE; SELECT count(*) FROM public.lab_arena_submissions")
            assert cursor.fetchone()[0] == before
            cursor.execute("SET ROLE lab_arena_service; SELECT public.lab_arena_schema_version_v1()")
            assert cursor.fetchone()[0]["version"] == 193


def test_schema190_upload_reservations_survive193_upgrade():
    migrations = DEFAULT_MIGRATIONS[:DEFAULT_MIGRATIONS.index(LAB_ARENA_UPLOAD_RECOVERY_MIGRATION)]
    previous = database_with_lab_arena_migration(migrations)
    psycopg2, dsn = next(previous)
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    store = ArenaStore(transport)
    try:
        round_id, miner = _open(store, "upgrade")
        original = _doc(round_id, "sub-before")
        original.pop("source_content_md5")
        store.register_submission(round_id, "sub-before", miner, original)
        with pytest.raises(ArenaStoreError, match="submission_conflict"):
            store.register_submission(round_id, "sub-after", miner, _doc(round_id, "sub-after", b"larger replacement source"))
        with psycopg2.connect(**dsn) as admin:
            admin.autocommit = True
            with admin.cursor() as cursor:
                cursor.execute((Path(__file__).resolve().parents[2] / "scripts" / LAB_ARENA_UPLOAD_RECOVERY_MIGRATION).read_text())
        recovered = store.register_submission(round_id, "sub-after", miner, _doc(round_id, "sub-after", b"larger replacement source"))
        assert recovered["submission_id"] == "sub-after"
        assert store.get_submission("sub-before")["submission_doc"] == original
        assert store.get_submission("sub-before")["rejection_rule"] == "source_replaced"
        store.accept_submission_with_credentials(round_id, "sub-after", miner, {"openrouter": "dGVzdA==", "deepline": "dGVzdA=="})
        assert store.get_submission("sub-after")["status"] == "accepted"
    finally:
        transport.close()
        previous.close()
