"""Current-schema PostgreSQL proof of atomic own-hotkey source replacement."""

from __future__ import annotations

import base64
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport, new_lease_token
from lab_arena.owner_admission import OwnerAdmission
from tests.lab_arena.hotkey_admission_postgres_test import (
    BLOCK_HASH,
    _config as _integrity_config,
)
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_source_migration_postgres import _hotkey, _round_config


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


@pytest.fixture
def store(database):
    psycopg2, dsn = database
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    yield ArenaStore(transport)
    transport.close()


def _config(round_id: str, *, cutoff_after: timedelta):
    now = datetime.now(timezone.utc)
    config = _round_config(
        round_id,
        cutoff=(now + cutoff_after).isoformat().replace("+00:00", "Z"),
    )
    return config


def _doc(round_id: str, submission_id: str, size: int, *, checksum: bytes | None = None):
    doc = {
        "source_ref": f"arena/{round_id}/sources/{submission_id}.tar.gz",
        "source_size_bytes": size,
        "consent": {"public_rerun": True},
    }
    if checksum is not None:
        doc["source_content_md5"] = base64.b64encode(hashlib.md5(checksum).digest()).decode()
    return doc


def _accept(store, round_id, submission_id, miner):
    return store.accept_submission_with_credentials(
        round_id, submission_id, miner,
        {"openrouter": "dGVzdA==", "deepline": "dGVzdA=="},
    )


def test_replacement_keeps_fallback_until_atomic_valid_finalize(store, database):
    round_id = "arena-2098-05-01-replace"
    miner = _hotkey("queued-replacement-miner")
    assert store.create_round(round_id, _config(round_id, cutoff_after=timedelta(hours=3)))["status"] == "created"
    assert store.submission_replacement_schema()["version"] == 258

    assert store.register_submission(round_id, "sub-original", miner, _doc(round_id, "sub-original", 123))["status"] == "registered"
    assert _accept(store, round_id, "sub-original", miner)["status"] == "ok"
    credential_before = store.get_submission_credential("sub-original", miner, "openrouter")

    reservation = store.register_submission(round_id, "sub-new", miner, _doc(round_id, "sub-new", 124))
    assert reservation["status"] == "registered"
    assert store.get_submission("sub-new")["replaces_submission_id"] == "sub-original"
    assert store.get_submission("sub-original")["status"] == "accepted"
    assert store.get_submission_credential("sub-original", miner, "openrouter") == credential_before

    assert _accept(store, round_id, "sub-new", miner)["status"] == "ok"
    old, new = store.get_submission("sub-original"), store.get_submission("sub-new")
    assert (old["status"], old["rejection_rule"], old["replaced_by_submission_id"]) == (
        "rejected", "source_replaced", "sub-new",
    )
    assert new["status"] == "accepted"
    assert new["replaces_submission_id"] == "sub-original"
    assert old["source_ref"] == _doc(round_id, "sub-original", 123)["source_ref"]
    assert store.get_submission_credential("sub-original", miner, "openrouter") is None
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM public.lab_arena_submission_credentials "
            "WHERE submission_id='sub-original'"
        )
        assert cursor.fetchone()[0] == 2
    assert len(store.list_submissions(round_id, status="accepted")) == 1
    with pytest.raises(ArenaStoreError, match="submission_not_uploading"):
        _accept(store, round_id, "sub-original", miner)


def test_rejected_pending_replacement_does_not_remove_accepted_fallback(store):
    round_id = "arena-2098-05-02-fallback"
    miner = _hotkey("queued-fallback-miner")
    store.create_round(round_id, _config(round_id, cutoff_after=timedelta(hours=3)))
    store.register_submission(round_id, "sub-first", miner, _doc(round_id, "sub-first", 123))
    _accept(store, round_id, "sub-first", miner)
    store.register_submission(round_id, "sub-invalid", miner, _doc(round_id, "sub-invalid", 124))
    assert store.update_submission(
        round_id, "sub-invalid", "uploading", "rejected",
        {"rejection_rule": "source_checksum_mismatch"},
    )["status"] == "ok"
    assert store.get_submission("sub-first")["status"] == "accepted"
    assert store.register_submission(round_id, "sub-third", miner, _doc(round_id, "sub-third", 125))["status"] == "replacement_limit_reached"
    assert store.get_submission("sub-third") is None


def test_legacy_same_size_new_checksum_gets_new_pending_object(store):
    round_id = "arena-2098-05-06-samesize"
    miner = _hotkey("queued-same-size-miner")
    store.create_round(round_id, _config(round_id, cutoff_after=timedelta(hours=3)))
    store.register_submission(round_id, "sub-no-md5", miner, _doc(round_id, "sub-no-md5", 123))
    _accept(store, round_id, "sub-no-md5", miner)
    new = store.register_submission(
        round_id, "sub-with-md5", miner,
        _doc(round_id, "sub-with-md5", 123, checksum=b"different bytes"),
    )
    assert new["status"] == "registered" and new["submission_id"] == "sub-with-md5"
    assert store.get_submission("sub-with-md5")["replaces_submission_id"] == "sub-no-md5"
    assert store.get_submission("sub-no-md5")["status"] == "accepted"


def test_replacement_preserves_occupied_slot_at_twenty_challenger_cap(store):
    round_id = "arena-2098-05-07-fullcap"
    config = _config(round_id, cutoff_after=timedelta(hours=3))
    config["max_challengers"] = 20
    store.create_round(round_id, config)
    for index in range(20):
        miner = _hotkey(f"queued-cap-miner-{index}")
        submission_id = f"sub-cap-{index}"
        store.register_submission(round_id, submission_id, miner, _doc(round_id, submission_id, 123))
        assert _accept(store, round_id, submission_id, miner)["status"] == "ok"
    miner = _hotkey("queued-cap-miner-0")
    store.register_submission(round_id, "sub-cap-replacement", miner, _doc(round_id, "sub-cap-replacement", 124))
    assert _accept(store, round_id, "sub-cap-replacement", miner)["status"] == "ok"
    assert len(store.list_submissions(round_id, status="accepted")) == 20


def test_integrity_owner_and_current_output_policies_remain_bound_to_hotkey(store):
    round_id = "arena-2098-05-14-integrity"
    miner = _hotkey("queued-integrity-miner")
    sibling = _hotkey("queued-integrity-sibling")
    foreign = _hotkey("queued-integrity-foreign")
    owner = _hotkey("queued-integrity-owner")
    wrong_owner = _hotkey("queued-integrity-wrong-owner")
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
    changed_owner = OwnerAdmission(wrong_owner, 123, BLOCK_HASH)

    assert store.register_submission(
        round_id, "sub-integrity-a", miner,
        _doc(round_id, "sub-integrity-a", 123), owner_admission=admission,
    )["status"] == "registered"
    assert _accept(store, round_id, "sub-integrity-a", miner)["status"] == "ok"
    with pytest.raises(ArenaStoreError, match="lab_arena_submission_owner_changed"):
        store.register_submission(
            round_id, "sub-integrity-wrong", miner,
            _doc(round_id, "sub-integrity-wrong", 124),
            owner_admission=changed_owner,
        )
    assert store.get_submission("sub-integrity-wrong") is None

    assert store.register_submission(
        round_id, "sub-integrity-b", miner,
        _doc(round_id, "sub-integrity-b", 124), owner_admission=admission,
    )["status"] == "registered"
    with pytest.raises(ArenaStoreError, match="lab_arena_submission_missing"):
        _accept(store, round_id, "sub-integrity-b", foreign)
    assert store.get_submission("sub-integrity-a")["status"] == "accepted"
    assert store.get_submission("sub-integrity-b")["status"] == "uploading"

    # The former coldkey collision was removed in migration 228. A sibling
    # hotkey with the same owner has its own independent active slot.
    assert store.register_submission(
        round_id, "sub-integrity-sibling", sibling,
        _doc(round_id, "sub-integrity-sibling", 125), owner_admission=admission,
    )["status"] == "registered"
    assert _accept(store, round_id, "sub-integrity-sibling", sibling)["status"] == "ok"
    assert _accept(store, round_id, "sub-integrity-b", miner)["status"] == "ok"
    accepted = store.list_submissions(round_id, status="accepted")
    assert {row["miner_hotkey"] for row in accepted} == {miner, sibling}
    assert store.get_submission("sub-integrity-b")["owner_coldkey"] == owner
    assert store.get_round(round_id)["configuration_doc"] == config


def test_existing_hotkey_changes_close_one_hour_before_cutoff(store):
    round_id = "arena-2098-05-03-freeze"
    miner = _hotkey("queued-freeze-miner")
    newcomer = _hotkey("queued-newcomer-miner")
    store.create_round(round_id, _config(round_id, cutoff_after=timedelta(minutes=30)))
    assert store.register_submission(round_id, "sub-first-freeze", miner, _doc(round_id, "sub-first-freeze", 123))["status"] == "registered"
    assert _accept(store, round_id, "sub-first-freeze", miner)["status"] == "ok"
    late = store.register_submission(round_id, "sub-late-freeze", miner, _doc(round_id, "sub-late-freeze", 124))
    assert late["status"] == "replacement_closed"
    assert store.get_submission("sub-first-freeze")["status"] == "accepted"
    assert store.get_submission("sub-late-freeze") is None
    assert store.register_submission(round_id, "sub-newcomer", newcomer, _doc(round_id, "sub-newcomer", 123))["status"] == "registered"


def test_review_claim_is_deferred_without_ledger_spend_before_replacement_freeze(store):
    round_id = "arena-2098-05-04-review"
    miner = _hotkey("queued-review-miner")
    store.create_round(round_id, _config(round_id, cutoff_after=timedelta(hours=3)))
    store.register_submission(round_id, "sub-review", miner, _doc(round_id, "sub-review", 123))
    _accept(store, round_id, "sub-review", miner)
    result = store.begin_submission_review(
        "sub-review", miner, new_lease_token(), 50_000,
        "openai/gpt-5", 3, 1200,
    )
    assert result["status"] == "deferred"
    assert store.get_submission("sub-review")["code_review_status"] == "pending"
    assert store.list_ledger(submission_id="sub-review") == []


def test_completed_old_review_cost_and_receipt_survive_replacement(store, database):
    round_id = "arena-2098-05-12-oldreview"
    miner = _hotkey("queued-old-review-miner")
    store.create_round(round_id, _config(round_id, cutoff_after=timedelta(minutes=30)))
    store.register_submission(round_id, "sub-reviewed", miner, _doc(round_id, "sub-reviewed", 123))
    _accept(store, round_id, "sub-reviewed", miner)
    token = new_lease_token()
    claimed = store.begin_submission_review(
        "sub-reviewed", miner, token, 50_000, "openai/gpt-5", 3, 1200,
    )
    assert claimed["status"] == "claimed"
    completed = store.finish_submission_review(
        "sub-reviewed", miner, token, "passed",
        {
            "schema_version": "leadpoet.lab_arena.code_review.result.v1",
            "passed": True, "verdict": "pass", "model": "openai/gpt-5",
            "file_count": 3, "source_bytes": 1200, "categories": [],
        }, 31_250,
    )
    assert completed["status"] == "passed"
    old_review = store.get_submission("sub-reviewed")["code_review_doc"]
    old_ledger = store.list_ledger(submission_id="sub-reviewed")
    assert old_review["review_cost_microusd"] == 31_250

    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET configuration_doc = "
            "pg_catalog.jsonb_set(configuration_doc, '{schedule,submission_cutoff}', "
            "to_jsonb((pg_catalog.clock_timestamp() + INTERVAL '3 hours')::TEXT)) "
            "WHERE round_id=%s", (round_id,),
        )
        cursor.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once")

    store.register_submission(round_id, "sub-new-reviewed", miner, _doc(round_id, "sub-new-reviewed", 124))
    assert _accept(store, round_id, "sub-new-reviewed", miner)["status"] == "ok"
    old = store.get_submission("sub-reviewed")
    assert old["status"] == "rejected" and old["code_review_status"] == "passed"
    assert old["code_review_doc"] == old_review
    assert store.list_ledger(submission_id="sub-reviewed") == old_ledger
    assert store.list_ledger(submission_id="sub-new-reviewed") == []


def test_pending_replacement_cannot_finalize_after_freeze(store, database):
    round_id = "arena-2098-05-05-finalfreeze"
    miner = _hotkey("queued-final-freeze-miner")
    store.create_round(round_id, _config(round_id, cutoff_after=timedelta(hours=3)))
    store.register_submission(round_id, "sub-fallback", miner, _doc(round_id, "sub-fallback", 123))
    _accept(store, round_id, "sub-fallback", miner)
    store.register_submission(round_id, "sub-pending", miner, _doc(round_id, "sub-pending", 124))

    # Advance only this disposable round's schedule with its write-once guard
    # disabled. Production uses the same post-lock database clock comparison.
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET configuration_doc = "
            "pg_catalog.jsonb_set(configuration_doc, '{schedule,submission_cutoff}', "
            "to_jsonb((pg_catalog.clock_timestamp() + INTERVAL '30 minutes')::TEXT)) "
            "WHERE round_id=%s", (round_id,),
        )
        cursor.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once")

    assert _accept(store, round_id, "sub-pending", miner)["status"] == "replacement_closed"
    assert store.register_submission(
        round_id, "sub-pending-retry", miner,
        _doc(round_id, "sub-pending-retry", 124),
    )["status"] == "replacement_closed"
    assert store.get_submission("sub-fallback")["status"] == "accepted"
    assert store.get_submission("sub-pending")["status"] == "uploading"


def test_migration_replay_is_safe_and_capability_is_service_only(database):
    psycopg2, dsn = database
    migration = (
        Path(__file__).resolve().parents[2]
        / "scripts/262-lab-arena-one-replacement-attempt.sql"
    ).read_text()
    with psycopg2.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM public.lab_arena_submissions")
            before = cursor.fetchone()[0]
            cursor.execute(migration)
            cursor.execute(migration)
            cursor.execute("SELECT count(*) FROM public.lab_arena_submissions")
            assert cursor.fetchone()[0] == before
            cursor.execute(
                "SELECT has_function_privilege('lab_arena_service', "
                "'public.lab_arena_submission_replacement_schema_v1()', 'EXECUTE'), "
                "has_function_privilege('service_role', "
                "'public.lab_arena_submission_replacement_schema_v1()', 'EXECUTE')"
            )
            assert cursor.fetchone() == (True, False)


def _contend(database, label, operation):
    psycopg2, dsn = database
    transport = PsycopgTransport(
        lambda: psycopg2.connect(**dsn, application_name=label)
    )
    contender = ArenaStore(transport)
    try:
        return operation(contender)
    except ArenaStoreError as exc:
        return {"error": str(exc)}
    finally:
        transport.close()


def _wait_for_lock(observer, label):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with observer.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM pg_catalog.pg_stat_activity "
                "WHERE application_name=%s AND wait_event_type='Lock'",
                (label,),
            )
            if cursor.fetchone()[0]:
                return
        time.sleep(0.01)
    raise AssertionError(f"{label} did not wait on the round lock")


def test_extra_registration_cannot_supersede_the_one_pending_replacement(store, database):
    round_id = "arena-2098-05-08-newer"
    miner = _hotkey("queued-newer-miner")
    store.create_round(round_id, _config(round_id, cutoff_after=timedelta(hours=3)))
    store.register_submission(round_id, "sub-a", miner, _doc(round_id, "sub-a", 123))
    _accept(store, round_id, "sub-a", miner)
    store.register_submission(round_id, "sub-b", miner, _doc(round_id, "sub-b", 124))
    psycopg2, dsn = database
    blocker = psycopg2.connect(**dsn)
    observer = psycopg2.connect(**dsn)
    observer.autocommit = True
    try:
        with blocker.cursor() as cursor:
            cursor.execute(
                "SELECT round_id FROM public.lab_arena_rounds WHERE round_id=%s FOR UPDATE",
                (round_id,),
            )
        with ThreadPoolExecutor(max_workers=2) as pool:
            newer = pool.submit(
                _contend, database, "queued-newer-register",
                lambda s: s.register_submission(round_id, "sub-c", miner, _doc(round_id, "sub-c", 125)),
            )
            _wait_for_lock(observer, "queued-newer-register")
            stale = pool.submit(
                _contend, database, "queued-newer-finalize",
                lambda s: _accept(s, round_id, "sub-b", miner),
            )
            _wait_for_lock(observer, "queued-newer-finalize")
            blocker.commit()
            new_result, stale_result = newer.result(timeout=10), stale.result(timeout=10)
        assert new_result["status"] == "replacement_limit_reached"
        assert stale_result["status"] == "ok"
        assert store.get_submission("sub-b")["status"] == "accepted"
        assert store.get_submission("sub-c") is None
        assert store.get_submission("sub-a")["status"] == "rejected"
    finally:
        blocker.close()
        observer.close()


def test_parallel_finalize_one_accepted_candidate_and_one_credential_set(store, database):
    round_id = "arena-2098-05-09-parallel"
    miner = _hotkey("queued-parallel-finalize-miner")
    store.create_round(round_id, _config(round_id, cutoff_after=timedelta(hours=3)))
    store.register_submission(round_id, "sub-a-parallel", miner, _doc(round_id, "sub-a-parallel", 123))
    _accept(store, round_id, "sub-a-parallel", miner)
    store.register_submission(round_id, "sub-b-parallel", miner, _doc(round_id, "sub-b-parallel", 124))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda index: _contend(
                database, f"queued-parallel-finalize-{index}",
                lambda s: _accept(s, round_id, "sub-b-parallel", miner),
            ), range(2),
        ))
    assert sorted(result["status"] for result in results) == ["existing", "ok"]
    assert len(store.list_submissions(round_id, status="accepted")) == 1
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM public.lab_arena_submission_credentials "
            "WHERE submission_id='sub-b-parallel'"
        )
        assert cursor.fetchone()[0] == 2


def test_round_lock_wait_rechecks_replacement_freeze_and_review_claim(store, database):
    round_id = "arena-2098-05-10-lockfreeze"
    miner = _hotkey("queued-lock-freeze-miner")
    config = _config(round_id, cutoff_after=timedelta(hours=1, seconds=2))
    store.create_round(round_id, config)
    store.register_submission(round_id, "sub-lock-a", miner, _doc(round_id, "sub-lock-a", 123))
    _accept(store, round_id, "sub-lock-a", miner)
    freeze_at = datetime.fromisoformat(
        config["schedule"]["submission_cutoff"].replace("Z", "+00:00")
    ) - timedelta(hours=1)
    psycopg2, dsn = database
    blocker = psycopg2.connect(**dsn)
    observer = psycopg2.connect(**dsn)
    observer.autocommit = True
    try:
        with blocker.cursor() as cursor:
            cursor.execute(
                "SELECT round_id FROM public.lab_arena_rounds WHERE round_id=%s FOR UPDATE",
                (round_id,),
            )
        with ThreadPoolExecutor(max_workers=2) as pool:
            replacement = pool.submit(
                _contend, database, "queued-boundary-replace",
                lambda s: s.register_submission(
                    round_id, "sub-lock-b", miner, _doc(round_id, "sub-lock-b", 124),
                ),
            )
            review = pool.submit(
                _contend, database, "queued-boundary-review",
                lambda s: s.begin_submission_review(
                    "sub-lock-a", miner, new_lease_token(), 50_000,
                    "openai/gpt-5", 3, 1200,
                ),
            )
            _wait_for_lock(observer, "queued-boundary-replace")
            _wait_for_lock(observer, "queued-boundary-review")
            time.sleep(max(0, (freeze_at - datetime.now(timezone.utc)).total_seconds() + 0.06))
            blocker.commit()
            replacement_result = replacement.result(timeout=10)
            review_result = review.result(timeout=10)
        assert replacement_result["status"] == "replacement_closed"
        assert review_result["status"] == "claimed"
        assert store.get_submission("sub-lock-a")["status"] == "accepted"
        assert store.get_submission("sub-lock-b") is None
        assert [row["entry_kind"] for row in store.list_ledger(submission_id="sub-lock-a")] == [
            "reservation", "dispatch",
        ]
    finally:
        blocker.close()
        observer.close()


def test_round_lock_wait_rechecks_midnight_for_first_finalize(store, database):
    round_id = "arena-2098-05-11-midnight"
    miner = _hotkey("queued-midnight-miner")
    config = _config(round_id, cutoff_after=timedelta(seconds=2))
    store.create_round(round_id, config)
    store.register_submission(round_id, "sub-midnight", miner, _doc(round_id, "sub-midnight", 123))
    cutoff = datetime.fromisoformat(
        config["schedule"]["submission_cutoff"].replace("Z", "+00:00")
    )
    psycopg2, dsn = database
    blocker = psycopg2.connect(**dsn)
    observer = psycopg2.connect(**dsn)
    observer.autocommit = True
    try:
        with blocker.cursor() as cursor:
            cursor.execute(
                "SELECT round_id FROM public.lab_arena_rounds WHERE round_id=%s FOR UPDATE",
                (round_id,),
            )
        with ThreadPoolExecutor(max_workers=1) as pool:
            finalize = pool.submit(
                _contend, database, "queued-midnight-finalize",
                lambda s: _accept(s, round_id, "sub-midnight", miner),
            )
            _wait_for_lock(observer, "queued-midnight-finalize")
            time.sleep(max(0, (cutoff - datetime.now(timezone.utc)).total_seconds() + 0.06))
            blocker.commit()
            result = finalize.result(timeout=10)
        assert result["status"] == "window_closed"
        assert store.get_submission("sub-midnight")["status"] == "uploading"
        assert store.list_ledger(submission_id="sub-midnight") == []
    finally:
        blocker.close()
        observer.close()


@pytest.mark.parametrize("replacement", [True, False])
def test_credential_insert_wait_rolls_back_admission_after_deadline(
    store, database, replacement,
):
    suffix = "swapwait" if replacement else "firstwait"
    round_id = f"arena-2098-05-13-{suffix}"
    miner = _hotkey(f"queued-credential-wait-{suffix}")
    wait_time = timedelta(hours=1, seconds=2) if replacement else timedelta(seconds=2)
    config = _config(round_id, cutoff_after=wait_time)
    store.create_round(round_id, config)
    candidate_id = f"sub-wait-b-{suffix}"
    if replacement:
        store.register_submission(round_id, "sub-wait-a", miner, _doc(round_id, "sub-wait-a", 123))
        _accept(store, round_id, "sub-wait-a", miner)
    store.register_submission(round_id, candidate_id, miner, _doc(round_id, candidate_id, 124))
    deadline = datetime.fromisoformat(
        config["schedule"]["submission_cutoff"].replace("Z", "+00:00")
    )
    if replacement:
        deadline -= timedelta(hours=1)
    psycopg2, dsn = database
    blocker = psycopg2.connect(**dsn)
    observer = psycopg2.connect(**dsn)
    observer.autocommit = True
    label = f"queued-credential-wait-{suffix}"
    try:
        with blocker.cursor() as cursor:
            cursor.execute(
                "LOCK TABLE public.lab_arena_submission_credentials IN SHARE MODE"
            )
        with ThreadPoolExecutor(max_workers=1) as pool:
            finalize = pool.submit(
                _contend, database, label,
                lambda s: _accept(s, round_id, candidate_id, miner),
            )
            _wait_for_lock(observer, label)
            time.sleep(max(0, (deadline - datetime.now(timezone.utc)).total_seconds() + 0.06))
            blocker.commit()
            result = finalize.result(timeout=10)
        expected = (
            "lab_arena_submission_replacement_closed"
            if replacement else "lab_arena_submission_window_closed"
        )
        assert expected in result["error"]
        assert store.get_submission(candidate_id)["status"] == "uploading"
        if replacement:
            old = store.get_submission("sub-wait-a")
            assert old["status"] == "accepted"
            assert old["replaced_by_submission_id"] is None
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_submission_credentials "
                "WHERE submission_id=%s", (candidate_id,)
            )
            assert cursor.fetchone()[0] == 0
    finally:
        blocker.close()
        observer.close()
