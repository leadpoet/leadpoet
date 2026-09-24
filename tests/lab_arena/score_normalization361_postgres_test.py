"""Exact September 25 score normalization with historical state preserved."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from lab_arena import contracts, scoring
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_contracts import base_round_configuration


MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "scripts/361-arena-2026-09-25-score-normalization.sql"
)
ROUND_ID = "arena-2026-09-25"
NORMALIZATION_KEY = "ARENA_SCORE_NORMALIZATION"
NORMALIZATION_POLICY = "available_intent_cap_v1"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _configuration(day: str = "25", *, company_quality: bool = False) -> dict:
    config = base_round_configuration()
    config.update(
        round_id=f"arena-2026-09-{day}",
        mode="live",
        network_name="finney",
        netuid=71,
        integrity_policy="arena_integrity_v1",
        intent_details_policy="intent_details_v1",
        cost_per_company_microusd=800000,
        stage_1_icp_count=5,
        stage_2_icp_count=5,
        promotion_margin=0.5,
    )
    day_number = int(day)
    config["schedule"] = {
        key: value.replace("2026-09-01", f"2026-09-{day_number - 1:02d}").replace(
            "2026-09-02", f"2026-09-{day_number:02d}"
        )
        for key, value in config["schedule"].items()
    }
    config["scorer_policy"] = scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2",
        company_quality=company_quality,
        intent_details=True,
    )
    if company_quality:
        config["company_quality_policy"] = "company_quality_v1"
    # Keep this fixture pre-migration even if future-round policy construction
    # learns the new default while this one-off migration remains in the tree.
    config["scorer_policy"]["env_bindings"].pop(NORMALIZATION_KEY, None)
    config["scorer_policy"]["env_bindings"]["EXISTING_BINDING"] = "preserved"
    return contracts.validate_round_configuration(config)


def _seed(
    cursor, *, day: str = "25", status: str = "open", company_quality: bool = False
) -> dict:
    config = _configuration(day, company_quality=company_quality)
    cursor.execute("SET session_replication_role=replica")
    cursor.execute(
        """
        INSERT INTO public.lab_arena_rounds
          (round_id, status, configuration_doc, rewards_enabled)
        VALUES (%s, %s, %s::jsonb, false)
        """,
        (config["round_id"], status, json.dumps(config)),
    )
    cursor.execute("SET session_replication_role=origin")
    return config


def _row(cursor, day: str = "25") -> dict:
    cursor.execute(
        "SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s",
        (f"arena-2026-09-{day}",),
    )
    return cursor.fetchone()[0]


def _delete_rounds(cursor) -> None:
    cursor.execute("SET session_replication_role=replica")
    cursor.execute(
        "DELETE FROM public.lab_arena_rounds WHERE round_id IN (%s, %s)",
        ("arena-2026-09-24", ROUND_ID),
    )
    cursor.execute("SET session_replication_role=origin")


@pytest.mark.parametrize("company_quality", [False, True])
def test_migration_is_exact_idempotent_and_preserves_history(
    database, company_quality
):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        _seed(cursor, company_quality=company_quality)
        _seed(cursor, day="24", status="published")
        before = _row(cursor)
        historical = _row(cursor, "24")

        cursor.execute(MIGRATION.read_text())
        after = _row(cursor)
        expected = deepcopy(before["configuration_doc"])
        expected["scorer_policy"]["env_bindings"][NORMALIZATION_KEY] = (
            NORMALIZATION_POLICY
        )

        assert after["configuration_doc"] == expected
        assert contracts.validate_round_configuration(expected) == expected
        assert expected["scorer_policy"]["env_bindings"]["EXISTING_BINDING"] == (
            "preserved"
        )
        assert {
            key: value
            for key, value in after.items()
            if key not in {"configuration_doc", "updated_at"}
        } == {
            key: value
            for key, value in before.items()
            if key not in {"configuration_doc", "updated_at"}
        }
        assert _row(cursor, "24") == historical

        cursor.execute(MIGRATION.read_text())
        assert _row(cursor) == after
        cursor.execute(
            """
            SELECT tgenabled FROM pg_trigger
            WHERE tgrelid='public.lab_arena_rounds'::regclass
              AND tgname='lab_arena_rounds_write_once'
            """
        )
        assert cursor.fetchone()[0] == "O"
        assert MIGRATION.name not in CURRENT_SERVICE_MIGRATIONS
        _delete_rounds(cursor)


@pytest.mark.parametrize("status", ["committed", "stage1", "published"])
def test_migration_refuses_started_or_published_round(database, status):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        with connection.cursor() as cursor:
            _seed(cursor, status=status)
            connection.commit()
            before = _row(cursor)

            with pytest.raises(
                psycopg2.Error, match="open, unfrozen, unstarted"
            ):
                cursor.execute(MIGRATION.read_text())
            connection.rollback()
            assert _row(cursor) == before
            _delete_rounds(cursor)
            connection.commit()
    finally:
        connection.close()


def test_migration_refuses_open_round_with_started_generation(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        with connection.cursor() as cursor:
            _seed(cursor)
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET stage_generation=1 WHERE round_id=%s",
                (ROUND_ID,),
            )
            cursor.execute("SET session_replication_role=origin")
            connection.commit()
            before = _row(cursor)

            with pytest.raises(
                psycopg2.Error, match="open, unfrozen, unstarted"
            ):
                cursor.execute(MIGRATION.read_text())
            connection.rollback()
            assert _row(cursor) == before
            _delete_rounds(cursor)
            connection.commit()
    finally:
        connection.close()


def test_migration_refuses_an_existing_different_binding(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        with connection.cursor() as cursor:
            config = _configuration()
            config["scorer_policy"]["env_bindings"][NORMALIZATION_KEY] = "future_v2"
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                """
                INSERT INTO public.lab_arena_rounds
                  (round_id, status, configuration_doc, rewards_enabled)
                VALUES (%s, 'open', %s::jsonb, false)
                """,
                (ROUND_ID, json.dumps(config)),
            )
            cursor.execute("SET session_replication_role=origin")
            connection.commit()
            before = _row(cursor)

            with pytest.raises(psycopg2.Error, match="binding differs"):
                cursor.execute(MIGRATION.read_text())
            connection.rollback()
            assert _row(cursor) == before
            _delete_rounds(cursor)
            connection.commit()
    finally:
        connection.close()
