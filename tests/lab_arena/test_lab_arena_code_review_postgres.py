"""Disposable-PostgreSQL proof for the Arena pre-scoring code review state."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from lab_arena import contracts
from lab_arena.store import (
    ArenaStore,
    ArenaStoreError,
    PsycopgTransport,
    hash_lease_token,
    new_lease_token,
)
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    encrypted_runtime_credentials,
    hotkey,
    round_config,
    source_submission_doc,
)
from tests.postgres_migration_harness import SCRIPTS


MIGRATION = "207-lab-arena-code-review.sql"
MODEL = "anthropic/claude-sonnet-5"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(DEFAULT_MIGRATIONS)


@pytest.fixture(scope="module")
def connect(database):
    psycopg2, dsn = database

    def _connect():
        return psycopg2.connect(**dsn)

    return _connect


@pytest.fixture()
def store(connect):
    transport = PsycopgTransport(connect)
    yield ArenaStore(transport)
    transport.close()


@pytest.fixture()
def superuser(connect):
    connection = connect()
    connection.autocommit = True
    yield connection
    connection.close()


def accepted_submission(store, suffix, *, is_king=False, miner=None):
    round_id = "arena-2099-02-01-" + suffix.replace("-", "")
    submission_id = suffix + "-submission"
    miner = miner or hotkey("code-review-" + suffix)
    config = round_config(round_id, [hotkey("code-review-runner-" + suffix)])
    assert store.create_round(round_id, config)["status"] == "created"
    source = source_submission_doc(round_id, submission_id, is_king=is_king)
    assert store.register_submission(round_id, submission_id, miner, source)[
        "status"
    ] == "registered"
    assert store.accept_submission_with_credentials(
        round_id,
        submission_id,
        miner,
        encrypted_runtime_credentials(submission_id),
    )["status"] == "ok"
    return round_id, submission_id, miner


def review_doc(verdict, *, file_count=3, source_bytes=1200):
    return {
        "schema_version": "leadpoet.lab_arena.code_review.result.v1",
        "passed": verdict == "pass",
        "verdict": verdict,
        "model": MODEL,
        "file_count": file_count,
        "source_bytes": source_bytes,
        "categories": [],
    }


def test_capability_and_passing_review_gate_freeze_and_track_cost(
    store, superuser
):
    assert store.code_review_schema() == {
        "schema_version": "leadpoet.lab_arena.code_review.v1",
        "version": 207,
        "claim_ttl_seconds": 600,
        "retry_backoff_seconds": 60,
        "max_attempts": 3,
    }
    signatures = (
        "lab_arena_code_review_schema_v1()",
        "lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)",
        "lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)",
    )
    with superuser.cursor() as cursor:
        cursor.execute(
            "SELECT NOT has_schema_privilege("
            "'lab_arena_owner', 'public', 'CREATE')"
        )
        assert cursor.fetchone() == (True,)
        for signature in signatures:
            cursor.execute(
                "SELECT owner.rolname, "
                "has_function_privilege('lab_arena_service', proc.oid, 'EXECUTE'), "
                "has_function_privilege('service_role', proc.oid, 'EXECUTE'), "
                "has_function_privilege('anon', proc.oid, 'EXECUTE'), "
                "has_function_privilege('authenticated', proc.oid, 'EXECUTE') "
                "FROM pg_catalog.pg_proc AS proc "
                "JOIN pg_catalog.pg_roles AS owner ON owner.oid=proc.proowner "
                "WHERE proc.oid=pg_catalog.to_regprocedure(%s)",
                ("public." + signature,),
            )
            assert cursor.fetchone() == (
                "lab_arena_owner", True, False, False, False
            )
    round_id, submission_id, miner = accepted_submission(store, "pass")

    with pytest.raises(ArenaStoreError, match="lab_arena_code_review_required"):
        store.update_submission(round_id, submission_id, "accepted", "frozen")

    token = new_lease_token()
    claimed = store.begin_submission_review(
        submission_id, miner, token, 50_000, MODEL, 3, 1200
    )
    assert claimed["status"] == "claimed"
    assert claimed["attempt"] == 1
    assert claimed["idempotent"] is False
    assert store.begin_submission_review(
        submission_id, miner, token, 50_000, MODEL, 3, 1200
    )["idempotent"] is True

    ledger = store.list_ledger(submission_id=submission_id)
    assert [row["entry_kind"] for row in ledger] == ["reservation", "dispatch"]
    assert {row["run_id"] for row in ledger} == {None}
    assert {row["stage"] for row in ledger} == {None}
    assert {row["operation_id"] for row in ledger} == {
        "openrouter.code_review"
    }
    assert {row["funding_source"] for row in ledger} == {"miner_key"}

    with pytest.raises(ArenaStoreError, match="lab_arena_code_review_input_invalid"):
        store.finish_submission_review(
            submission_id, miner, token, "passed", review_doc("pass"), None
        )
    with pytest.raises(ArenaStoreError, match="lab_arena_code_review_input_invalid"):
        store.finish_submission_review(
            submission_id,
            miner,
            token,
            "passed",
            review_doc("pass", source_bytes=1199),
            31_250,
        )
    with pytest.raises(ArenaStoreError, match="lab_arena_code_review_input_invalid"):
        store.finish_submission_review(
            submission_id, miner, token, "passed", review_doc("reject"), 31_250
        )
    assert store.get_submission(submission_id)["code_review_status"] == "reviewing"
    assert len(store.list_ledger(submission_id=submission_id)) == 2

    finished = store.finish_submission_review(
        submission_id, miner, token, "passed", review_doc("pass"), 31_250
    )
    assert finished["status"] == "passed"
    assert finished["ledger_status"] == "settled"
    assert finished["cost_microusd"] == 31_250
    assert finished["review_cost_microusd"] == 31_250
    row = store.get_submission(submission_id)
    assert row["code_review_status"] == "passed"
    assert row["code_review_doc"]["cost_status"] == "settled"
    assert row["code_review_doc"]["cost_microusd"] == 31_250
    assert row["code_review_doc"]["review_cost_microusd"] == 31_250
    assert store.finish_submission_review(
        submission_id, miner, token, "passed", review_doc("pass"), 31_250
    )["status"] == "existing"
    assert store.update_submission(
        round_id, submission_id, "accepted", "frozen"
    )["status"] == "ok"


def test_parallel_begin_has_one_claim_and_rejected_review_cannot_freeze(connect):
    setup = ArenaStore(PsycopgTransport(connect))
    try:
        round_id, submission_id, miner = accepted_submission(setup, "parallel")
    finally:
        setup.close()
    tokens = [new_lease_token(), new_lease_token()]

    def begin(token):
        contender = ArenaStore(PsycopgTransport(connect))
        try:
            return contender.begin_submission_review(
                submission_id, miner, token, 40_000, MODEL, 4, 2000
            )
        finally:
            contender.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(begin, tokens))
    assert sorted(result["status"] for result in results) == ["busy", "claimed"]
    winner = tokens[[result["status"] for result in results].index("claimed")]

    store = ArenaStore(PsycopgTransport(connect))
    try:
        stale = tokens[1] if winner == tokens[0] else tokens[0]
        assert store.finish_submission_review(
            submission_id, miner, stale, "passed", review_doc("pass"), 1
        )["status"] == "stale"
        assert store.finish_submission_review(
            submission_id,
            miner,
            winner,
            "rejected",
            review_doc("reject", file_count=4, source_bytes=2000),
            20_000,
        )["status"] == "rejected"
        with pytest.raises(ArenaStoreError, match="lab_arena_code_review_required"):
            store.update_submission(round_id, submission_id, "accepted", "frozen")
        assert len(store.list_ledger(submission_id=submission_id)) == 3
    finally:
        store.close()


def test_expired_and_error_attempts_are_bounded_and_never_stale_write(
    store, superuser
):
    _round_id, submission_id, miner = accepted_submission(store, "retry")
    token1, token2, token3 = (new_lease_token() for _ in range(3))
    assert store.begin_submission_review(
        submission_id, miner, token1, 100, MODEL, 2, 1000
    )["attempt"] == 1
    with superuser.cursor() as cursor:
        cursor.execute(
            "UPDATE public.lab_arena_submissions "
            "SET code_review_started_at=clock_timestamp() - interval '601 seconds', "
            "code_review_expires_at=clock_timestamp() - interval '1 second' "
            "WHERE submission_id=%s",
            (submission_id,),
        )
    assert store.begin_submission_review(
        submission_id, miner, token2, 200, MODEL, 2, 1000
    )["attempt"] == 2
    assert store.finish_submission_review(
        submission_id,
        miner,
        token1,
        "passed",
        review_doc("pass", file_count=2, source_bytes=1000),
        1,
    )["status"] == "stale"
    failed = store.finish_submission_review(
        submission_id,
        miner,
        token2,
        "error",
        review_doc("error", file_count=2, source_bytes=1000),
        None,
    )
    assert failed["ledger_status"] == "uncertain"
    assert failed["cost_microusd"] == 200
    assert failed["review_cost_microusd"] == 300
    assert store.begin_submission_review(
        submission_id, miner, token3, 300, MODEL, 2, 1000
    )["status"] == "backoff"
    with superuser.cursor() as cursor:
        cursor.execute(
            "UPDATE public.lab_arena_submissions "
            "SET code_review_started_at=clock_timestamp() - interval '61 seconds' "
            "WHERE submission_id=%s",
            (submission_id,),
        )
    third = store.begin_submission_review(
        submission_id, miner, token3, 300, MODEL, 2, 1000
    )
    assert third["status"] == "claimed" and third["attempt"] == 3
    final = store.finish_submission_review(
        submission_id,
        miner,
        token3,
        "error",
        review_doc("error", file_count=2, source_bytes=1000),
        0,
    )
    assert final["review_cost_microusd"] == 300
    assert store.begin_submission_review(
        submission_id, miner, new_lease_token(), 1, MODEL, 2, 1000
    )["status"] == "exhausted"
    heads = [
        row["entry_kind"]
        for row in store.list_ledger(submission_id=submission_id)
    ]
    assert heads == [
        "reservation", "dispatch", "uncertain",
        "reservation", "dispatch", "uncertain",
        "reservation", "dispatch", "settlement",
    ]


def test_zero_cost_preparation_failure_and_late_result_on_rejected_slot(store):
    round_id, submission_id, miner = accepted_submission(store, "prep-error")
    token = new_lease_token()
    assert store.begin_submission_review(
        submission_id, miner, token, 0, MODEL, 0, 4096
    )["status"] == "claimed"
    with pytest.raises(ArenaStoreError, match="lab_arena_code_review_input_invalid"):
        store.finish_submission_review(
            submission_id,
            miner,
            token,
            "passed",
            review_doc("pass", file_count=0, source_bytes=4096),
            0,
        )
    assert store.get_submission(submission_id)["code_review_status"] == "reviewing"
    assert store.update_submission(
        round_id,
        submission_id,
        "accepted",
        "rejected",
        {"rejection_rule": "code_review_incomplete"},
    )["status"] == "ok"
    result = store.finish_submission_review(
        submission_id,
        miner,
        token,
        "error",
        {
            "schema_version": "leadpoet.lab_arena.code_review.result.v1",
            "verdict": "error",
            "model": MODEL,
            "file_count": 0,
            "source_bytes": 4096,
            "error_code": "source_archive_invalid",
        },
        0,
    )
    assert result["cost_microusd"] == 0
    row = store.get_submission(submission_id)
    assert row["status"] == "rejected"
    assert row["code_review_status"] == "error"


def test_exact_baseline_is_exempt_but_fake_king_is_not(store):
    suffix = "baseline"
    round_id = "arena-2099-02-01-" + suffix
    baseline = hotkey("baseline")
    baseline_id = "baseline-2099-02-01-" + suffix
    config = round_config(round_id, [hotkey("baseline-review-runner")])
    assert store.create_round(round_id, config)["status"] == "created"
    assert store.register_submission(
        round_id,
        baseline_id,
        baseline,
        source_submission_doc(round_id, baseline_id, is_king=True),
    )["status"] == "registered"
    assert store.update_submission(
        round_id, baseline_id, "uploading", "accepted", {"is_king": True}
    )["status"] == "ok"
    assert store.begin_submission_review(
        baseline_id, baseline, new_lease_token(), 1, MODEL, 1, 100
    )["status"] == "baseline"
    assert store.update_submission(
        round_id, baseline_id, "accepted", "frozen", {"is_king": True}
    )["status"] == "ok"

    fake_id = "fake-king-submission"
    fake = hotkey("fake-code-review-king")
    assert store.register_submission(
        round_id,
        fake_id,
        fake,
        source_submission_doc(round_id, fake_id, is_king=True),
    )["status"] == "registered"
    assert store.accept_submission_with_credentials(
        round_id, fake_id, fake, encrypted_runtime_credentials(fake_id)
    )["status"] == "ok"
    with pytest.raises(ArenaStoreError, match="lab_arena_code_review_required"):
        store.update_submission(
            round_id, fake_id, "accepted", "frozen", {"is_king": True}
        )


def test_migration_replay_preserves_rows_and_review_results(store, superuser):
    _round_id, submission_id, miner = accepted_submission(store, "replay")
    token = new_lease_token()
    store.begin_submission_review(
        submission_id, miner, token, 777, MODEL, 5, 2222
    )
    store.finish_submission_review(
        submission_id,
        miner,
        token,
        "passed",
        review_doc("pass", file_count=5, source_bytes=2222),
        555,
    )
    before = store.get_submission(submission_id)
    migration = (SCRIPTS / MIGRATION).read_text(encoding="utf-8")
    with superuser.cursor() as cursor:
        cursor.execute(migration)
        cursor.execute(migration)
    assert store.get_submission(submission_id) == before


def test_review_document_is_bounded(store):
    _round_id, submission_id, miner = accepted_submission(store, "bounded-doc")
    token = new_lease_token()
    store.begin_submission_review(
        submission_id, miner, token, 1, MODEL, 1, 1
    )
    with pytest.raises(ArenaStoreError, match="lab_arena_code_review_input_invalid"):
        store.finish_submission_review(
            submission_id,
            miner,
            token,
            "passed",
            {"summary": "x" * 33_000},
            1,
        )


def test_upgrade_preserves_existing_frozen_row_and_allows_review_recovery():
    migrations_before_review = DEFAULT_MIGRATIONS[
        : DEFAULT_MIGRATIONS.index(MIGRATION)
    ]
    generator = database_with_lab_arena_migration(migrations_before_review)
    connection = None
    transport = None
    try:
        psycopg2, dsn = next(generator)

        def connect():
            return psycopg2.connect(**dsn)

        transport = PsycopgTransport(connect)
        store = ArenaStore(transport)
        round_id, submission_id, miner = accepted_submission(store, "upgrade")
        assert store.update_submission(
            round_id, submission_id, "accepted", "frozen"
        )["status"] == "ok"
        before = store.get_submission(submission_id)
        participant = {
            "submission_id": submission_id,
            "miner_hotkey": miner,
            "is_king": False,
        }
        assert store.transition_round(
            round_id,
            "open",
            "committed",
            {
                "participants": [participant],
                "benchmark_ref": f"arena/{round_id}/benchmark.json",
                "evaluation_date": "2099-02-01",
            },
        )["status"] == "ok"
        assert store.open_stage(
            round_id, 1, [participant], list(contracts.stage_positions(1))
        )["status"] == "ok"

        connection = connect()
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_runs SET kind='score', scored_run_id=run_id "
                "WHERE run_id = ("
                "SELECT run_id FROM public.lab_arena_runs WHERE round_id=%s "
                "ORDER BY icp_position, created_at, assignment_id LIMIT 1)",
                (round_id,),
            )
        migration = (SCRIPTS / MIGRATION).read_text(encoding="utf-8")
        with connection.cursor() as cursor:
            cursor.execute(migration)
            cursor.execute(migration)
        after = store.get_submission(submission_id)
        for key, value in before.items():
            assert after[key] == value
        assert after["code_review_status"] == "pending"
        assert after["code_review_attempts"] == 0

        runner = hotkey("code-review-runner-upgrade")

        def claim(index):
            request_id = f"{index:032x}"
            return store.claim_assignment(
                round_id=round_id,
                runner_hotkey=runner,
                declared_parallelism=20,
                slot_ceiling=20,
                excluded_miner_hotkeys=[],
                request_id=request_id,
                request_hash=contracts.document_hash({"request_id": request_id}),
                lease_token_hash=hash_lease_token(new_lease_token()),
                lease_ttl_seconds=120,
            )

        assert claim(1)["status"] == "no_pending"
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT kind, status FROM public.lab_arena_runs "
                "WHERE round_id=%s ORDER BY kind, status",
                (round_id,),
            )
            run_states = cursor.fetchall()
        assert {kind for kind, _status in run_states} == {"execute", "score"}
        assert all(status == "pending" for _kind, status in run_states)
        assert store.list_ledger(submission_id=submission_id) == []

        token = new_lease_token()
        assert store.begin_submission_review(
            submission_id, miner, token, 999, MODEL, 3, 1200
        )["status"] == "claimed"
        assert store.finish_submission_review(
            submission_id, miner, token, "passed", review_doc("pass"), 900
        )["status"] == "passed"
        recovered = store.get_submission(submission_id)
        assert recovered["status"] == "frozen"
        assert recovered["code_review_status"] == "passed"
        claimed_kinds = set()
        for index in range(2, 12):
            leased = claim(index)
            assert leased["status"] == "leased"
            claimed_kinds.add(leased["kind"])
        assert claimed_kinds == {"execute", "score"}
        with connection.cursor() as cursor:
            with pytest.raises(Exception, match="frozen submission is immutable"):
                cursor.execute(
                    "UPDATE public.lab_arena_submissions SET source_size_bytes=1 "
                    "WHERE submission_id=%s",
                    (submission_id,),
                )
        connection.rollback()
    finally:
        if transport is not None:
            transport.close()
        if connection is not None:
            connection.close()
        generator.close()
