"""Focused PostgreSQL coverage for Arena source migrations 181 and 182."""

from __future__ import annotations

from typing import Any, Dict

import pytest
from bittensor_wallet import Keypair

from lab_arena import contracts
from lab_arena.store import (
    ArenaStore,
    ArenaStoreError,
    PsycopgTransport,
    hash_lease_token,
    new_lease_token,
)
from tests.lab_arena.lab_arena_pg_harness import (
    LAB_ARENA_DAILY_COMPETITION_MIGRATION,
    LAB_ARENA_MIGRATION,
    LAB_ARENA_SOURCE_EXECUTION_MIGRATION,
    LAB_ARENA_SOURCE_SUBMISSIONS_MIGRATION,
    database_with_lab_arena_migration,
)
from tests.test_source_add_end_to_end_postgres import SCRIPTS

MIGRATIONS = (
    LAB_ARENA_MIGRATION,
    LAB_ARENA_DAILY_COMPETITION_MIGRATION,
    LAB_ARENA_SOURCE_SUBMISSIONS_MIGRATION,
    LAB_ARENA_SOURCE_EXECUTION_MIGRATION,
)


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(MIGRATIONS)


@pytest.fixture(scope="module")
def pre_source_execution_database():
    """A disposable database stopped at 181 for the 182 cutover test."""

    yield from database_with_lab_arena_migration(MIGRATIONS[:-1])


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
        "scorer_image_digest": "sha256:" + "f" * 64,
        "scorer_image_reference": "registry.example/scorer@sha256:" + "f" * 64,
    }


def _source_doc(round_id: str, submission_id: str) -> Dict[str, Any]:
    return {
        "source_ref": "arena/%s/sources/%s.tar.gz" % (round_id, submission_id),
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
        round_id, "sub-first", miner, _source_doc(round_id, "sub-first")
    )
    assert first["status"] == "registered"
    retry = store.register_submission(
        round_id, "sub-retry", miner, _source_doc(round_id, "sub-retry")
    )
    assert retry == {
        "status": "existing",
        "submission_status": "uploading",
        "submission_id": "sub-first",
        "source_ref": "arena/%s/sources/sub-first.tar.gz" % round_id,
    }
    with pytest.raises(ArenaStoreError, match="lab_arena_submission_conflict"):
        conflict = _source_doc(round_id, "sub-other")
        conflict["source_size_bytes"] = 124
        store.register_submission(
            round_id,
            "sub-other",
            miner,
            conflict,
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
        _source_doc(round_id, "sub-late"),
    )
    assert miner_result["status"] == "window_closed"
    baseline_doc = _source_doc(round_id, "baseline-2001-01-01")
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


def test_claim_persists_source_for_execution_and_nulls_it_for_scoring(store, database):
    round_id = "arena-2098-01-02-source"
    runner = _hotkey("source-runner")
    miner = _hotkey("source-claim-miner")
    submission_id = "sub-source-claim"
    source = _source_doc(round_id, submission_id)
    assert store.create_round(
        round_id, _round_config(round_id, cutoff="2099-01-01T00:00:00Z")
    )["status"] == "created"
    assert store.register_submission(round_id, submission_id, miner, source)[
        "status"
    ] == "registered"
    assert store.update_submission(
        round_id, submission_id, "uploading", "accepted"
    )["status"] == "ok"
    assert store.update_submission(
        round_id, submission_id, "accepted", "frozen"
    )["status"] == "ok"
    participants = [
        {"submission_id": submission_id, "miner_hotkey": miner, "is_king": False}
    ]
    assert store.transition_round(
        round_id,
        "open",
        "committed",
        {
            "participants": participants,
            "benchmark_ref": "arena/%s/benchmark.json" % round_id,
            "evaluation_date": "2098-01-02",
        },
    )["status"] == "ok"
    assert store.open_stage(round_id, 1, participants, list(range(10)))[
        "status"
    ] == "ok"

    execute_token = new_lease_token()
    execute_request_id = contracts.new_request_id()
    execute_request_hash = contracts.document_hash({"request": "execute"})
    execute = store.claim_assignment(
        round_id=round_id,
        runner_hotkey=runner,
        declared_parallelism=2,
        slot_ceiling=2,
        excluded_miner_hotkeys=[],
        request_id=execute_request_id,
        request_hash=execute_request_hash,
        lease_token_hash=hash_lease_token(execute_token),
    )
    assert execute["status"] == "leased" and execute["kind"] == "execute"
    assert {
        key: execute[key]
        for key in ("source_ref", "source_size_bytes")
    } == {
        key: source[key]
        for key in ("source_ref", "source_size_bytes")
    }
    assert "image_digest" not in execute
    replay = store.claim_assignment(
        round_id=round_id,
        runner_hotkey=runner,
        declared_parallelism=2,
        slot_ceiling=2,
        excluded_miner_hotkeys=[],
        request_id=execute_request_id,
        request_hash=execute_request_hash,
        lease_token_hash=hash_lease_token(new_lease_token()),
    )
    assert replay == execute

    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_runs SET kind = 'score' "
                "WHERE round_id = %s AND status = 'pending' AND icp_position = 1",
                (round_id,),
            )
            assert cursor.rowcount == 1
    finally:
        connection.close()

    score = store.claim_assignment(
        round_id=round_id,
        runner_hotkey=runner,
        declared_parallelism=2,
        slot_ceiling=2,
        excluded_miner_hotkeys=[],
        request_id=contracts.new_request_id(),
        request_hash=contracts.document_hash({"request": "score"}),
        lease_token_hash=hash_lease_token(new_lease_token()),
    )
    assert score["status"] == "leased" and score["kind"] == "score"
    assert score["source_ref"] is None
    assert score["source_size_bytes"] is None
    assert "image_digest" not in score

    row = store.get_submission(submission_id)
    assert row is not None
    assert not {
        "submitted_reference",
        "image_reference",
        "image_digest",
        "image_size_bytes",
        "source_sha256",
        "source_cache_key",
    }.intersection(row)


def test_configured_baseline_bypasses_only_the_coldkey_exclusion(store):
    round_id = "arena-2098-01-03-baseline"
    runner = _hotkey("source-runner")
    baseline = _hotkey("source-baseline")
    submission_id = "baseline-2098-01-03-baseline"
    source = _source_doc(round_id, submission_id)
    source["is_king"] = True
    assert store.create_round(
        round_id, _round_config(round_id, cutoff="2099-01-01T00:00:00Z")
    )["status"] == "created"
    assert store.register_submission(round_id, submission_id, baseline, source)[
        "status"
    ] == "registered"
    assert store.update_submission(
        round_id, submission_id, "uploading", "accepted", {"is_king": True}
    )["status"] == "ok"
    assert store.update_submission(
        round_id, submission_id, "accepted", "frozen", {"is_king": True}
    )["status"] == "ok"
    participants = [
        {"submission_id": submission_id, "miner_hotkey": baseline, "is_king": True}
    ]
    assert store.transition_round(
        round_id,
        "open",
        "committed",
        {
            "participants": participants,
            "benchmark_ref": "arena/%s/benchmark.json" % round_id,
            "evaluation_date": "2098-01-03",
        },
    )["status"] == "ok"
    assert store.open_stage(round_id, 1, participants, list(range(10)))[
        "status"
    ] == "ok"

    response = store.claim_assignment(
        round_id=round_id,
        runner_hotkey=runner,
        declared_parallelism=1,
        slot_ceiling=1,
        excluded_miner_hotkeys=[baseline],
        request_id=contracts.new_request_id(),
        request_hash=contracts.document_hash({"request": "baseline"}),
        lease_token_hash=hash_lease_token(new_lease_token()),
    )
    assert response["status"] == "leased"
    assert response["submission_id"] == submission_id
    assert response["source_ref"] == source["source_ref"]


def test_retired_image_only_rows_cannot_become_execute_leases(store, database):
    round_id = "arena-2098-01-04-source"
    runner = _hotkey("source-runner")
    miner = _hotkey("retired-image-miner")
    submission_id = "sub-retired-image"
    assert store.create_round(
        round_id, _round_config(round_id, cutoff="2099-01-01T00:00:00Z")
    )["status"] == "created"
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions "
                "(submission_id, round_id, miner_hotkey, status, is_king, consent, submission_doc) "
                "VALUES (%s, %s, %s, 'frozen', FALSE, '{\"public_rerun\":true}', '{}')",
                (submission_id, round_id, miner),
            )
    finally:
        connection.close()
    participants = [
        {"submission_id": submission_id, "miner_hotkey": miner, "is_king": False}
    ]
    assert store.transition_round(
        round_id,
        "open",
        "committed",
        {
            "participants": participants,
            "benchmark_ref": "arena/%s/benchmark.json" % round_id,
            "evaluation_date": "2098-01-04",
        },
    )["status"] == "ok"
    assert store.open_stage(round_id, 1, participants, list(range(10)))[
        "status"
    ] == "ok"
    response = store.claim_assignment(
        round_id=round_id,
        runner_hotkey=runner,
        declared_parallelism=1,
        slot_ceiling=1,
        excluded_miner_hotkeys=[],
        request_id=contracts.new_request_id(),
        request_hash=contracts.document_hash({"request": "retired"}),
        lease_token_hash=hash_lease_token(new_lease_token()),
    )
    assert response == {"status": "no_pending"}


def test_migration_cancels_an_active_legacy_image_round(
    pre_source_execution_database,
):
    psycopg2, dsn = pre_source_execution_database
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    legacy_store = ArenaStore(transport)
    round_id = "arena-2098-01-05-cutover"
    runner = _hotkey("source-runner")
    miner = _hotkey("legacy-cutover-miner")
    submission_id = "sub-legacy-cutover"
    digest = "sha256:" + "8" * 64
    assert legacy_store.create_round(
        round_id, _round_config(round_id, cutoff="2099-01-01T00:00:00Z")
    )["status"] == "created"
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions "
                "(submission_id, round_id, miner_hotkey, status, is_king, "
                "submitted_reference, image_reference, image_digest, image_size_bytes, "
                "consent, submission_doc, frozen_at) "
                "VALUES (%s, %s, %s, 'frozen', FALSE, %s, %s, %s, 123, "
                "'{\"public_rerun\":true}', '{}', clock_timestamp())",
                (
                    submission_id,
                    round_id,
                    miner,
                    "registry.example/legacy:latest",
                    "registry.example/legacy@" + digest,
                    digest,
                ),
            )
        participants = [
            {
                "submission_id": submission_id,
                "miner_hotkey": miner,
                "is_king": False,
            }
        ]
        assert legacy_store.transition_round(
            round_id,
            "open",
            "committed",
            {
                "participants": participants,
                "benchmark_ref": "arena/%s/benchmark.json" % round_id,
                "evaluation_date": "2098-01-05",
            },
        )["status"] == "ok"
        assert legacy_store.open_stage(round_id, 1, participants, list(range(10)))[
            "status"
        ] == "ok"
        legacy_lease = legacy_store.claim_assignment(
            round_id=round_id,
            runner_hotkey=runner,
            declared_parallelism=1,
            slot_ceiling=1,
            excluded_miner_hotkeys=[],
            request_id=contracts.new_request_id(),
            request_hash=contracts.document_hash({"request": "legacy"}),
            lease_token_hash=hash_lease_token(new_lease_token()),
        )
        assert legacy_lease["status"] == "leased"

        with connection.cursor() as cursor:
            cursor.execute(
                (SCRIPTS / LAB_ARENA_SOURCE_EXECUTION_MIGRATION).read_text(
                    encoding="utf-8"
                )
            )
            cursor.execute(
                "SELECT status, cancel_reason FROM public.lab_arena_rounds "
                "WHERE round_id = %s",
                (round_id,),
            )
            assert cursor.fetchone() == ("cancelled", "source_bundle_cutover")
            cursor.execute(
                "SELECT status, rejection_rule FROM public.lab_arena_submissions "
                "WHERE submission_id = %s",
                (submission_id,),
            )
            assert cursor.fetchone() == (
                "rejected",
                "source_reupload_required",
            )
            cursor.execute(
                "SELECT COUNT(*) FROM public.lab_arena_runs "
                "WHERE round_id = %s AND status = 'failed' "
                "AND terminal_cause = 'stage_closed'",
                (round_id,),
            )
            assert cursor.fetchone()[0] == 10
            cursor.execute(
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_schema = 'public' "
                "AND table_name = 'lab_arena_submissions' "
                "AND column_name IN "
                "('submitted_reference', 'image_reference', 'image_digest', 'image_size_bytes')"
            )
            assert cursor.fetchone()[0] == 0
    finally:
        connection.close()
        transport.close()


def test_source_migrations_apply_again(database):
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
            cursor.execute(
                (SCRIPTS / LAB_ARENA_SOURCE_EXECUTION_MIGRATION).read_text(
                    encoding="utf-8"
                )
            )
            cursor.execute(
                (SCRIPTS / LAB_ARENA_SOURCE_EXECUTION_MIGRATION).read_text(
                    encoding="utf-8"
                )
            )
    finally:
        connection.close()
