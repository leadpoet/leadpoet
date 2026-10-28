"""Keep historical SQL evidence separate from current reader contracts."""

import pytest

from lab_arena.store import ArenaStore, ArenaStoreError, PsycopgTransport
from tests.lab_arena.historical_service_harness import HistoricalArenaStore
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey, round_config


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration()


def test_historical_projection_keeps_pinned_schema_and_database_errors(database):
    psycopg2, dsn = database
    transport = PsycopgTransport(lambda: psycopg2.connect(**dsn))
    historical = HistoricalArenaStore(transport)
    current = ArenaStore(transport)
    round_id = "arena-2026-09-18-historicalclient"
    configuration = round_config(round_id, [hotkey("historical-client-runner")])
    try:
        assert historical.create_round(round_id, configuration)["status"] == "created"
        with pytest.raises(ArenaStoreError, match="lab_arena_execution_attempt_count"):
            current.list_runs(round_id, columns="*,execution_attempt:lab_arena_execution_attempt_count")
        assert historical.list_runs(round_id, columns="*,execution_attempt:lab_arena_execution_attempt_count") == []
        with pytest.raises(ArenaStoreError, match="scoring_generation_floor"):
            current.get_public_submissions_round(round_id)
        assert historical.get_public_submissions_round(round_id) == current.get_round(round_id)
        assert historical.get_published_results_round(round_id) is None
        with pytest.raises(ArenaStoreError, match="does_not_exist"):
            historical.list_runs(round_id, columns="does_not_exist")
        # No later migration or production fallback was installed by this client.
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute("SELECT to_regprocedure('public.lab_arena_execution_attempt_count(public.lab_arena_runs)')")
            assert cursor.fetchone()[0] is None
            cursor.execute("SELECT count(*) FROM information_schema.columns WHERE table_schema='public' AND table_name='lab_arena_rounds' AND column_name='scoring_generation_floor'")
            assert cursor.fetchone()[0] == 0
    finally:
        transport.close()
