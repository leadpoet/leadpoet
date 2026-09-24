"""The publication guard uses the service's exact aggregate precision."""

from decimal import Decimal
from fractions import Fraction
import math
from pathlib import Path

import pytest

from lab_arena import verify
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)


MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "scripts/362-lab-arena-publication-aggregate-double.sql"
)
FUNCTION = "public.lab_arena__per_icp_publication_valid(text,jsonb)"
LIVE_REPLAY_SCORES = [
    Decimal("18"),
    Decimal("17"),
    Decimal("20"),
    Decimal("14.666667"),
    *([Decimal("0")] * 13),
]
NORMAL_SCORE_PATTERN = tuple(
    Decimal(value)
    for value in (
        "18.000000",
        "17.000000",
        "20.000000",
        "14.666667",
        "0.000000",
        "33.333333",
        "60.000000",
        "88.888889",
    )
)
MICRO_SCORE_PATTERN = tuple(
    Decimal(value)
    for value in (
        "0.000000",
        "0.000001",
        "0.000002",
        "0.000003",
        "0.000005",
        "0.999999",
    )
)


@pytest.fixture(scope="module")
def database():
    migrations = tuple(
        name for name in CURRENT_SERVICE_MIGRATIONS if name != MIGRATION.name
    )
    yield from database_with_lab_arena_migration(migrations)


def _definition(cursor) -> str:
    cursor.execute(
        "SELECT pg_get_functiondef(%s::regprocedure)",
        (FUNCTION,),
    )
    return cursor.fetchone()[0]


def _identity(cursor) -> tuple:
    cursor.execute(
        """
        SELECT pg_get_userbyid(proowner), prosecdef, proconfig, proacl
        FROM pg_proc WHERE oid=%s::regprocedure
        """,
        (FUNCTION,),
    )
    return cursor.fetchone()


def test_migration_changes_only_aggregate_precision_and_is_idempotent(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        before = _definition(cursor)
        identity = _identity(cursor)
        expected = before.replace(
            "  v_expected_score NUMERIC;",
            "  v_expected_score DOUBLE PRECISION;",
        ).replace(
            "    THEN runs.per_icp_score ELSE 0 END)\n"
            "  INTO v_score_count, v_accepted_count, v_expected_score",
            "    THEN runs.per_icp_score ELSE 0 END)::DOUBLE PRECISION\n"
            "  INTO v_score_count, v_accepted_count, v_expected_score",
        ).replace(
            "       OR (p_ranking ->> 'final_score')::NUMERIC IS DISTINCT FROM v_expected_score",
            "       OR (p_ranking ->> 'final_score')::DOUBLE PRECISION IS DISTINCT FROM v_expected_score",
        )
        assert expected != before

        cursor.execute(MIGRATION.read_text())
        after = _definition(cursor)
        assert after == expected
        assert _identity(cursor) == identity
        cursor.execute(MIGRATION.read_text())
        assert _definition(cursor) == after
        assert _identity(cursor) == identity


def test_live_replay_vector_rounds_once_and_rejects_next_float(database):
    psycopg2, dsn = database
    expected = verify.stage_score(
        [float(value) for value in LIVE_REPLAY_SCORES],
        len(LIVE_REPLAY_SCORES),
    )
    exact = sum(
        (Fraction(value) for value in LIVE_REPLAY_SCORES), Fraction(0)
    ) / len(LIVE_REPLAY_SCORES)
    assert expected == float(exact) == 4.0980392352941175

    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT avg(value), avg(value)::double precision,
                   avg(value) IS DISTINCT FROM %s::numeric,
                   avg(value)::double precision IS NOT DISTINCT FROM
                     %s::double precision,
                   avg(value)::double precision IS DISTINCT FROM
                     %s::double precision
            FROM unnest(%s::numeric[]) AS score(value)
            """,
            (
                repr(expected),
                repr(expected),
                repr(math.nextafter(expected, math.inf)),
                LIVE_REPLAY_SCORES,
            ),
        )
        numeric_average, double_average, old_rejects, exact_accepts, tamper_rejects = (
            cursor.fetchone()
        )
    # PostgreSQL's NUMERIC average chooses scale 16 here, then the migration
    # rounds that one aggregate to binary64. It reaches the same binary64 value
    # as the service's exact Fraction-to-float conversion.
    assert numeric_average == Decimal("4.0980392352941176")
    assert double_average == expected
    assert old_rejects is True
    assert exact_accepts is True
    assert tamper_rejects is True


def test_supported_denominators_match_service_for_six_decimal_scores(database):
    psycopg2, dsn = database

    def values(pattern, count):
        return [pattern[index % len(pattern)] for index in range(count)]

    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        for count in range(2, 101):
            normal = values(NORMAL_SCORE_PATTERN, count)
            micro = values(MICRO_SCORE_PATTERN, count)
            service_normal = verify.stage_score(
                [float(value) for value in normal], count
            )
            service_micro = verify.stage_score(
                [float(value) for value in micro], count
            )
            cursor.execute(
                """
                SELECT
                  (SELECT avg(value)::double precision
                   FROM unnest(%s::numeric[]) AS score(value)),
                  (SELECT avg(value)::double precision
                   FROM unnest(%s::numeric[]) AS score(value))
                """,
                (normal, micro),
            )
            postgres_normal, postgres_micro = cursor.fetchone()
            assert postgres_normal == service_normal, count
            assert postgres_micro == service_micro, count
