"""Current service flows use the complete schema without migration seed state."""
import pytest
from tests.lab_arena.lab_arena_pg_harness import current_service_database
from tests.lab_arena.complete_baseline_final_authority_postgres_test import (
    test_normal_complete_baseline_publishes_and_activates_with_all_guards as prove_round,
)


@pytest.fixture
def database():
    yield from current_service_database()


def test_current_service_schema_preserves_complete_baseline_final_authority(database, tmp_path, monkeypatch):
    pg, dsn = database
    with pg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute('SELECT count(*) FROM public.lab_arena_rounds')
        assert cur.fetchone() == (0,)
        cur.execute("SELECT to_regprocedure('public.lab_arena_execution_attempt_count(public.lab_arena_runs)'),"
                    "to_regprocedure('public.lab_arena_review_guard_state_v1()')")
        assert all(cur.fetchone())
    prove_round(database, tmp_path, monkeypatch, False, True, 17)
