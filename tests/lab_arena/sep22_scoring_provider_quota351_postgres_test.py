"""Verbatim migration 351 against disposable current PostgreSQL."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lab_arena import contracts
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_contracts import base_round_configuration
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts/351-arena-2026-09-22-scoring-provider-quota-2000.sql"
ROUND = "arena-2026-09-22"
SUBMISSION = "sep22-quota-source"
MINER = hotkey("sep22-scoring-quota")


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _configuration() -> dict:
    configuration = base_round_configuration()
    configuration.update(
        {
            "round_id": ROUND,
            "mode": "live",
            "rewards_enabled": True,
            "scoring_call_quotas": dict(
                contracts.LEGACY_SCORING_CALL_QUOTAS_PER_WORK_ITEM
            ),
            "execution_sequence_policy": contracts.BASELINE_SCORED_FIRST_POLICY,
        }
    )
    configuration["schedule"] = {
        key: value.replace("2026-09-01", "2026-09-21").replace(
            "2026-09-02", "2026-09-22"
        )
        for key, value in configuration["schedule"].items()
    }
    return contracts.validate_round_configuration(configuration)


def _seed(connection, *, frozen=False, with_run=False):
    configuration = _configuration()
    source_ref = f"arena/{ROUND}/sources/{SUBMISSION}.tar.gz"
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "TRUNCATE public.lab_arena_ledger,public.lab_arena_runs,"
            "public.lab_arena_submissions,public.lab_arena_rounds "
            "RESTART IDENTITY CASCADE"
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds("
            "round_id,status,status_generation,stage_generation,"
            "configuration_doc,rewards_enabled) "
            "VALUES (%s,'open',0,0,%s::jsonb,TRUE)",
            (ROUND, json.dumps(configuration)),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions("
            "submission_id,round_id,miner_hotkey,status,is_king,source_ref,"
            "source_size_bytes,submission_doc,frozen_at) VALUES ("
            "%s,%s,%s,%s,FALSE,%s,4096,%s::jsonb,%s)",
            (
                SUBMISSION,
                ROUND,
                MINER,
                "frozen" if frozen else "accepted",
                source_ref,
                json.dumps(
                    {
                        "source_ref": source_ref,
                        "source_size_bytes": 4096,
                        "consent": {"public_rerun": True},
                    }
                ),
                "2026-09-21T23:59:59Z" if frozen else None,
            ),
        )
        if with_run:
            cursor.execute(
                "INSERT INTO public.lab_arena_runs("
                "run_id,assignment_id,round_id,submission_id,miner_hotkey,"
                "stage,icp_position,attempt,kind,status,stage_generation) "
                "VALUES ('sep22-blocking-run','sep22-blocking-assignment',"
                "%s,%s,%s,1,0,1,'execute','pending',0)",
                (ROUND, SUBMISSION, MINER),
            )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()


def _state(connection) -> dict:
    state = {}
    with connection.cursor() as cursor:
        for table, order in (
            ("lab_arena_rounds", "round_id"),
            ("lab_arena_submissions", "submission_id"),
            ("lab_arena_runs", "run_id"),
        ):
            cursor.execute(
                "SELECT COALESCE(jsonb_agg(to_jsonb(rows) ORDER BY "
                f"{order}),'[]'::jsonb) FROM public.{table} AS rows"
            )
            state[table] = cursor.fetchone()[0]
        cursor.execute(
            "SELECT tgenabled FROM pg_catalog.pg_trigger WHERE tgrelid="
            "'public.lab_arena_rounds'::regclass AND "
            "tgname='lab_arena_rounds_write_once' AND NOT tgisinternal"
        )
        state["trigger"] = cursor.fetchone()[0]
    return state


def _execute(connection):
    with connection.cursor() as cursor:
        cursor.execute(MIGRATION.read_text(encoding="utf-8"))
    connection.commit()


def test_unstarted_open_round_changes_only_scoring_quotas_and_replays(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        _seed(connection)
        before = _state(connection)
        _execute(connection)
        after = _state(connection)

        expected = copy.deepcopy(before)
        expected_round = expected["lab_arena_rounds"][0]
        expected_round["configuration_doc"]["scoring_call_quotas"] = dict(
            contracts.SCORING_CALL_QUOTAS_PER_WORK_ITEM
        )
        expected_round["updated_at"] = after["lab_arena_rounds"][0]["updated_at"]
        assert after == expected
        assert contracts.validate_round_configuration(
            after["lab_arena_rounds"][0]["configuration_doc"]
        )["scoring_call_quotas"] == {
            "scrapingdog": 2000,
            "deepline": 2000,
            "openrouter": 2000,
        }

        # Normal progress after the repair cannot make a replay rewrite state.
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_submissions SET status='frozen',"
                "frozen_at=clock_timestamp() WHERE submission_id=%s",
                (SUBMISSION,),
            )
        connection.commit()
        progressed = _state(connection)
        _execute(connection)
        assert _state(connection) == progressed


@pytest.mark.parametrize("blocker", ("frozen", "run"))
def test_old_profile_refuses_started_or_frozen_round_without_mutation(
    database, blocker
):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        _seed(
            connection,
            frozen=blocker == "frozen",
            with_run=blocker == "run",
        )
        before = _state(connection)
        with pytest.raises(
            psycopg2.Error,
            match="requires an open, unfrozen, unstarted round",
        ):
            _execute(connection)
        connection.rollback()
        assert _state(connection) == before
