"""The two exact historical 585 migrations preserve each other's contracts."""

import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena import champion_baseline_reasoning585_postgres_test as baseline
from tests.postgres_migration_harness import SCRIPTS


BASELINE = SCRIPTS / "585-lab-arena-champion-baseline-reasoning.sql"
REVIEW = SCRIPTS / "585-lab-arena-review-deadline-recovery.sql"
REVIEW_ROLLBACK = SCRIPTS / "585-lab-arena-review-deadline-recovery-rollback.sql.template"
BASELINE_FUNCTIONS = (
    "public.lab_arena_rounds_write_once_v1()",
    "public.lab_arena_freeze_full_thinking_baseline_v1(text,text,text,text,text,text,text,text)",
)


@pytest.fixture()
def database():
    # The predecessor fixture installs baseline 567 and review 492. Add only
    # the existing review prerequisites, then test both exact migration files below.
    generator = baseline.prior.database.__wrapped__()
    db = next(generator)
    try:
        with db[0].connect(**db[1]) as connection, connection.cursor() as cursor:
            for name in (
                "562-lab-arena-review-context-integrity-reason.sql",
                "563-lab-arena-review-source-digest-reason.sql",
                "566-lab-arena-research-review-result.sql",
            ):
                cursor.execute((SCRIPTS / name).read_text())
        yield db
    finally:
        generator.close()


def _baseline_definitions(cursor):
    result = []
    for signature in BASELINE_FUNCTIONS:
        cursor.execute("SELECT to_jsonb(p) FROM pg_proc p WHERE oid=%s::regprocedure", (signature,))
        result.append(cursor.fetchone()[0])
    return result


@pytest.mark.parametrize("order", [(BASELINE, REVIEW), (REVIEW, BASELINE)])
def test_baseline_snapshot_survives_review_upgrade_and_exact_rollback(database, tmp_path, order):
    connect = lambda: database[0].connect(**database[1])
    store = ArenaStore(PsycopgTransport(connect))
    try:
        with connect() as connection, connection.cursor() as cursor:
            for migration in order:
                cursor.execute(migration.read_text())
            baseline_before = _baseline_definitions(cursor)
        assert store.code_review_schema()["retry_policy"] == "deadline_transient_v2"

        # Reuse the real legacy-champion promotion/source-freeze regression.
        # It checks exact bytes, funding owner, empty legacy capture snapshot,
        # idempotent freeze, and refusal of the old bootstrap substitution.
        baseline.test_legacy_champion_source_and_capture_contract_are_frozen(database, tmp_path)

        with connect() as connection:
            connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.execute("SELECT jsonb_agg(to_jsonb(r) ORDER BY round_id) FROM public.lab_arena_rounds r")
                rounds_before = cursor.fetchone()[0]
                cursor.execute("SELECT count(*) FROM public.lab_arena_rounds WHERE configuration_doc ? 'baseline_reasoning_configuration'")
                assert cursor.fetchone()[0] > 0
                try:
                    cursor.execute(REVIEW_ROLLBACK.read_text())
                    assert _baseline_definitions(cursor) == baseline_before
                    assert store._transport.rpc("lab_arena_code_review_schema_v1", {})["retry_policy"] == "bounded_transient_v1"
                    with pytest.raises(ArenaStoreError, match="code review schema mismatch"):
                        store.code_review_schema()
                    cursor.execute("SELECT jsonb_agg(to_jsonb(r) ORDER BY round_id) FROM public.lab_arena_rounds r")
                    assert cursor.fetchone()[0] == rounds_before
                finally:
                    cursor.execute(REVIEW.read_text())
                assert _baseline_definitions(cursor) == baseline_before
                cursor.execute("SELECT jsonb_agg(to_jsonb(r) ORDER BY round_id) FROM public.lab_arena_rounds r")
                assert cursor.fetchone()[0] == rounds_before
        assert store.code_review_schema()["retry_policy"] == "deadline_transient_v2"
    finally:
        store.close()


def test_current_service_fixture_reaches_atomic_review_capability():
    from tests.lab_arena.lab_arena_pg_harness import current_service_database

    database = current_service_database()
    store = None
    try:
        psycopg2, dsn = next(database)
        store = ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))
        schema = store.code_review_schema()
        assert schema["retry_policy"] == "deadline_transient_v2"
        assert schema["max_transient_attempts"] is None
        assert schema["atomic_restart_guard"] is True
    finally:
        if store is not None:
            store.close()
        database.close()
