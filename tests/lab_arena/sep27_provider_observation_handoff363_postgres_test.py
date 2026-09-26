"""Exact September 27 observation handoff with frozen rounds preserved."""

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


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts/363-arena-2026-09-27-provider-observation-handoff.sql"
ROUND_ID = "arena-2026-09-27"
BINDING = contracts.PROVIDER_OBSERVATION_HANDOFF_BINDING
CAPABILITY = contracts.AUTHENTICATED_PROVIDER_OBSERVATION_HANDOFF


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _configuration(day: str, *, handoff: bool = False) -> dict:
    config = base_round_configuration()
    config.update(
        round_id=f"arena-2026-09-{day}",
        mode="live",
        network_name="finney",
        netuid=71,
        integrity_policy="arena_integrity_v1",
        company_quality_policy="company_quality_v1",
        intent_details_policy="intent_details_v1",
        cost_per_company_microusd=800000,
        stage_1_icp_count=5,
        stage_2_icp_count=5,
        promotion_margin=0.5,
    )
    day_number = int(day)
    config["schedule"] = {
        key: value.replace(
            "2026-09-01", f"2026-09-{day_number - 1:02d}"
        ).replace("2026-09-02", f"2026-09-{day}")
        for key, value in config["schedule"].items()
    }
    config["scorer_policy"] = scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2",
        company_quality=True,
        intent_details=True,
        normalize_intent_scale=True,
        provider_observation_handoff=handoff,
    )
    return contracts.validate_round_configuration(config)


def _seed(cursor, day: str, *, status: str, handoff: bool = False) -> dict:
    config = _configuration(day, handoff=handoff)
    cursor.execute("SET session_replication_role=replica")
    cursor.execute(
        "INSERT INTO public.lab_arena_rounds "
        "(round_id,status,configuration_doc,rewards_enabled) "
        "VALUES (%s,%s,%s::jsonb,false)",
        (config["round_id"], status, json.dumps(config)),
    )
    cursor.execute("SET session_replication_role=origin")
    return config


def _row(cursor, day: str) -> dict:
    cursor.execute(
        "SELECT to_jsonb(r) FROM public.lab_arena_rounds r WHERE round_id=%s",
        (f"arena-2026-09-{day}",),
    )
    return cursor.fetchone()[0]


def _delete_rounds(cursor) -> None:
    cursor.execute("SET session_replication_role=replica")
    cursor.execute(
        "DELETE FROM public.lab_arena_rounds WHERE round_id IN (%s,%s)",
        ("arena-2026-09-26", ROUND_ID),
    )
    cursor.execute("SET session_replication_role=origin")


def test_exact_migration_is_idempotent_and_preserves_active_frozen_policy(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        _seed(cursor, "27", status="open")
        _seed(cursor, "26", status="stage1")
        before = _row(cursor, "27")
        active = _row(cursor, "26")

        cursor.execute(MIGRATION.read_text(encoding="utf-8"))
        after = _row(cursor, "27")
        expected = deepcopy(before["configuration_doc"])
        expected["scorer_policy"]["env_bindings"][BINDING] = CAPABILITY

        assert after["configuration_doc"] == expected
        assert contracts.validate_round_configuration(expected) == expected
        assert _row(cursor, "26") == active
        assert {
            key: value for key, value in after.items()
            if key not in {"configuration_doc", "updated_at"}
        } == {
            key: value for key, value in before.items()
            if key not in {"configuration_doc", "updated_at"}
        }

        cursor.execute(MIGRATION.read_text(encoding="utf-8"))
        assert _row(cursor, "27") == after
        assert MIGRATION.name not in CURRENT_SERVICE_MIGRATIONS
        _delete_rounds(cursor)


def test_migration_replay_preserves_progress_after_capability_is_frozen(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        _seed(cursor, "27", status="committed", handoff=True)
        before = _row(cursor, "27")

        cursor.execute(MIGRATION.read_text(encoding="utf-8"))

        assert _row(cursor, "27") == before
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "DELETE FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND_ID,)
        )
        cursor.execute("SET session_replication_role=origin")


@pytest.mark.parametrize("blocker", ["committed", "frozen_submission"])
def test_legacy_policy_refuses_committed_or_frozen_round_without_mutation(
    database, blocker
):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        with connection.cursor() as cursor:
            _seed(
                cursor,
                "27",
                status="committed" if blocker == "committed" else "open",
            )
            if blocker == "frozen_submission":
                cursor.execute("SET session_replication_role=replica")
                cursor.execute(
                    "INSERT INTO public.lab_arena_submissions "
                    "(submission_id,round_id,miner_hotkey,status,is_king,"
                    "source_ref,source_size_bytes,submission_doc,frozen_at) "
                    "VALUES ('frozen-source',%s,%s,'frozen',false,"
                    "'arena/arena-2026-09-27/sources/frozen-source.tar.gz',1,"
                    "'{}'::jsonb,clock_timestamp())",
                    (ROUND_ID, "5" * 48),
                )
                cursor.execute("SET session_replication_role=origin")
            connection.commit()
            before = _row(cursor, "27")

            with pytest.raises(
                psycopg2.Error, match="open, unfrozen, uncommitted"
            ):
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            connection.rollback()
            assert _row(cursor, "27") == before
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "DELETE FROM public.lab_arena_rounds WHERE round_id=%s",
                (ROUND_ID,),
            )
            cursor.execute("SET session_replication_role=origin")
            connection.commit()
    finally:
        connection.close()
