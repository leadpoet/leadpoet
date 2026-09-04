"""Disposable PostgreSQL checks for the public-baseline state table."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_source_add_end_to_end_postgres import (
    _database_with_migrations,
)


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = "180-public-baseline-rebenchmark.sql"


@pytest.fixture(scope="module")
def database():
    yield from _database_with_migrations((MIGRATION,))


def test_public_baseline_migration_is_repeatable_and_private(database) -> None:
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                (ROOT / "scripts" / MIGRATION).read_text(encoding="utf-8")
            )
            cursor.execute(
                """
                SELECT relrowsecurity
                FROM pg_class
                WHERE oid = 'public.research_lab_daily_rebenchmarks'::regclass
                """
            )
            assert cursor.fetchone() == (True,)
            for role in ("anon", "authenticated"):
                cursor.execute(
                    "SELECT has_table_privilege(%s, %s, 'SELECT')",
                    (role, "public.research_lab_daily_rebenchmarks"),
                )
                assert cursor.fetchone() == (False,)
            for privilege in ("SELECT", "INSERT", "UPDATE"):
                cursor.execute(
                    "SELECT has_table_privilege(%s, %s, %s)",
                    (
                        "service_role",
                        "public.research_lab_daily_rebenchmarks",
                        privilege,
                    ),
                )
                assert cursor.fetchone() == (True,)
            cursor.execute(
                "SELECT has_table_privilege(%s, %s, 'DELETE')",
                ("service_role", "public.research_lab_daily_rebenchmarks"),
            )
            assert cursor.fetchone() == (False,)
    finally:
        connection.close()


def test_completed_row_requires_full_ordinary_result(database) -> None:
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            with pytest.raises(psycopg2.errors.CheckViolation):
                cursor.execute(
                    """
                    INSERT INTO public.research_lab_daily_rebenchmarks (
                        run_id, benchmark_date, baseline_id,
                        baseline_repository, baseline_entrypoint,
                        status, expected_icp_count,
                        completed_icp_count, aggregate_score, worker_ref,
                        completed_at
                    ) VALUES (
                        gen_random_uuid(), DATE '2026-09-03', 'baseline',
                        'https://github.com/leadpoet/pydantic-harness',
                        'harness.run_icp', 'completed', 1, 1,
                        50.0, 'test', now()
                    )
                    """
                )
    finally:
        connection.close()
