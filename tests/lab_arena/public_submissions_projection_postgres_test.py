"""Exercise the submissions projection and authorities against real PostgreSQL."""

from copy import deepcopy

import pytest

from lab_arena import source_disclosure
from lab_arena.store import ArenaStore, PsycopgTransport, PUBLIC_CONFIGURATION_FIELDS
from tests.lab_arena.competition_publication443_postgres_test import _insert_rows
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS, database_with_lab_arena_migration, prepare_current_service_reads,
)
from tests.lab_arena.public_submissions_projection_test import _row, _service


@pytest.fixture()
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def test_real_projection_keeps_rankings_validation_disclosure_and_fresh_generation(database):
    pg, dsn = database
    prepare_current_service_reads(lambda: pg.connect(**dsn), include_run_reads=False)
    row = _row("published")
    row["round_id"] = "arena-2026-10-13"
    row.update(created_at="2026-10-12T00:00:00Z", promotion_required=False,
               baseline_promoted_at=None)
    connection = pg.connect(**dsn)
    connection.autocommit = True
    transport = PsycopgTransport(lambda: pg.connect(**dsn))
    service, fixture_transport = _service(row)
    service._store = ArenaStore(transport)
    try:
        _insert_rows(connection, [row])
        with connection.cursor() as cursor:
            # These synthetic rows test read authorities, not production
            # transitions. Use the same disposable-fixture rule as insertion.
            cursor.execute("SET session_replication_role=replica")
            try:
                cursor.execute("UPDATE public.lab_arena_rounds SET benchmark_ref=%s,status_generation=%s WHERE round_id=%s",
                               (row["benchmark_ref"], row["status_generation"], row["round_id"]))
            finally:
                cursor.execute("SET session_replication_role=origin")
        full = service._round(row["round_id"])
        compact = service._public_submissions_round(row["round_id"])
        assert compact["publication_doc"] == full["publication_doc"]
        assert len(compact["publication_doc"]["final_ranking"]) == 3
        assert compact["configuration_doc"] == {
            key: deepcopy(value) for key, value in full["configuration_doc"].items()
            if key in PUBLIC_CONFIGURATION_FIELDS + ("scorer_policy",)
        }
        for key in ("participants", "benchmark_ref", "published_at", "icp_set_date",
                    "evaluation_date", "cancel_reason", "status", "status_generation"):
            assert compact[key] == full[key]
        for submission in fixture_transport.submissions:
            assert source_disclosure.disclosure_status(submission, service.now(), round_row=compact) == \
                source_disclosure.disclosure_status(submission, service.now(), round_row=full)
        with connection.cursor() as cursor:
            cursor.execute("SET session_replication_role=replica")
            try:
                cursor.execute("UPDATE public.lab_arena_rounds SET status='stage2',status_generation=status_generation+1 WHERE round_id=%s",
                               (row["round_id"],))
            finally:
                cursor.execute("SET session_replication_role=origin")
        fresh = service._public_submissions_round(row["round_id"])
        assert fresh["status"] == "stage2" and fresh["status_generation"] == 8
    finally:
        transport.close()
        connection.close()
