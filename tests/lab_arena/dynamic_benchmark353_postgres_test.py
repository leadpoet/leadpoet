"""The dynamic benchmark migration against disposable current PostgreSQL."""

from __future__ import annotations

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


MIGRATION = Path(__file__).resolve().parents[2] / "scripts/353-lab-arena-dynamic-benchmark-count.sql"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def test_replay_keeps_installed_definitions_and_frozen_documents(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT has_function_privilege('lab_arena_service', "
                "'public.lab_arena_dynamic_benchmark_schema_v1()', 'EXECUTE'), "
                "has_function_privilege('anon', "
                "'public.lab_arena_dynamic_benchmark_schema_v1()', 'EXECUTE')"
            )
            assert cursor.fetchone() == (True, False)
            cursor.execute(
                "SELECT coalesce(jsonb_agg(configuration_doc ORDER BY round_id),'[]'::jsonb) "
                "FROM public.lab_arena_rounds"
            )
            before = cursor.fetchone()[0]
            cursor.execute(MIGRATION.read_text())
            cursor.execute(MIGRATION.read_text())
            cursor.execute(
                "SELECT coalesce(jsonb_agg(configuration_doc ORDER BY round_id),'[]'::jsonb) "
                "FROM public.lab_arena_rounds"
            )
            assert cursor.fetchone()[0] == before


@pytest.mark.parametrize("count", [10, 15, 20, 30])
def test_frozen_count_opens_baseline_then_miners_at_exact_positions(database, count):
    psycopg2, dsn = database
    round_id = f"arena-2026-09-23-count{count}"
    baseline_id = f"baseline-{count}"
    miner_id = f"miner-{count}"
    baseline_hotkey = hotkey(f"baseline-{count}")
    miner_hotkey = hotkey(f"miner-{count}")
    configuration = base_round_configuration()
    configuration.update({
        "round_id": round_id,
        "network_name": "finney",
        "netuid": 71,
        "stage_1_icp_count": (count + 1) // 2,
        "stage_2_icp_count": count // 2,
        "promotion_margin": 0.5 if count != 20 else 1.0,
        "execution_sequence_policy": contracts.BASELINE_SCORED_FIRST_POLICY,
    })
    configuration = contracts.validate_round_configuration(configuration)
    participants = [
        {"submission_id": baseline_id, "miner_hotkey": baseline_hotkey, "is_king": True},
        {"submission_id": miner_id, "miner_hotkey": miner_hotkey, "is_king": False},
    ]
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT public.lab_arena_dynamic_benchmark_schema_v1()")
        assert cursor.fetchone()[0]["version"] == 353
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds "
            "(round_id,status,configuration_doc,rewards_enabled,participants) "
            "VALUES (%s,'committed',%s::jsonb,FALSE,%s::jsonb)",
            (round_id, json.dumps(configuration), json.dumps(participants)),
        )
        for submission_id, miner_hotkey, is_king in (
            (baseline_id, baseline_hotkey, True),
            (miner_id, miner_hotkey, False),
        ):
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions "
                "(submission_id,round_id,miner_hotkey,status,is_king) "
                "VALUES (%s,%s,%s,'frozen',%s)",
                (submission_id, round_id, miner_hotkey, is_king),
            )
        cursor.execute("SET session_replication_role=origin")
        positions = list(range(count))
        cursor.execute(
            "SELECT public.lab_arena_open_stage(%s,1::smallint,%s::jsonb,%s::integer[])",
            (round_id, json.dumps([participants[0]]), positions),
        )
        assert cursor.fetchone()[0]["assignments"] == count
        cursor.execute(
            "SELECT count(*), min(icp_position), max(icp_position) "
            "FROM public.lab_arena_runs WHERE round_id=%s AND stage=1",
            (round_id,),
        )
        assert cursor.fetchone() == (count, 0, count - 1)
        cursor.execute(
            "SELECT public.lab_arena__integrity_submission_summary(%s,%s,%s::integer[])",
            (round_id, baseline_id, [count - 1]),
        )
        assert cursor.fetchone()[0]["valid"] is False
        for invalid in ([count], [0, 0]):
            cursor.execute("SAVEPOINT invalid_positions")
            with pytest.raises(psycopg2.Error, match="lab_arena_integrity_positions_invalid"):
                cursor.execute(
                    "SELECT public.lab_arena__integrity_submission_summary(%s,%s,%s::integer[])",
                    (round_id, baseline_id, invalid),
                )
            cursor.execute("ROLLBACK TO SAVEPOINT invalid_positions")
            cursor.execute("RELEASE SAVEPOINT invalid_positions")
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='stage1_scored' "
            "WHERE round_id=%s", (round_id,),
        )
        cursor.execute("SET session_replication_role=origin")
        cursor.execute(
            "SELECT public.lab_arena_open_stage(%s,2::smallint,%s::jsonb,%s::integer[])",
            (round_id, json.dumps([participants[1]]), positions),
        )
        assert cursor.fetchone()[0]["assignments"] == count
        cursor.execute(
            "SELECT count(*), min(icp_position), max(icp_position) "
            "FROM public.lab_arena_runs WHERE round_id=%s AND stage=2",
            (round_id,),
        )
        assert cursor.fetchone() == (count, 0, count - 1)


@pytest.mark.parametrize("count", [10, 30])
def test_parallel_legacy_path_uses_frozen_stage_split(database, count):
    psycopg2, dsn = database
    round_id = f"arena-2026-09-23-parallel{count}"
    baseline_id = f"parallel-baseline-{count}"
    miner_id = f"parallel-miner-{count}"
    baseline_hotkey = hotkey(baseline_id)
    miner_hotkey = hotkey(miner_id)
    configuration = base_round_configuration()
    configuration.update({
        "round_id": round_id,
        "network_name": "finney",
        "netuid": 71,
        "stage_1_icp_count": (count + 1) // 2,
        "stage_2_icp_count": count // 2,
        "parallel_twenty_icp_execution": True,
    })
    configuration = contracts.validate_round_configuration(configuration)
    participants = [
        {"submission_id": baseline_id, "miner_hotkey": baseline_hotkey, "is_king": True},
        {"submission_id": miner_id, "miner_hotkey": miner_hotkey, "is_king": False},
    ]
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds "
            "(round_id,status,configuration_doc,rewards_enabled,participants,icp_set_date) "
            "VALUES (%s,'committed',%s::jsonb,FALSE,%s::jsonb,'2026-09-22')",
            (round_id, json.dumps(configuration), json.dumps(participants)),
        )
        for submission_id, miner_hotkey, is_king in (
            (baseline_id, baseline_hotkey, True),
            (miner_id, miner_hotkey, False),
        ):
            cursor.execute(
                "INSERT INTO public.lab_arena_submissions "
                "(submission_id,round_id,miner_hotkey,status,is_king) "
                "VALUES (%s,%s,%s,'frozen',%s)",
                (submission_id, round_id, miner_hotkey, is_king),
            )
        cursor.execute("SET session_replication_role=origin")
        cursor.execute(
            "SELECT public.lab_arena_open_parallel_execution_v1(%s,%s::jsonb)",
            (round_id, json.dumps(participants)),
        )
        assert cursor.fetchone()[0]["assignments"] == 2 * count
        cursor.execute(
            "SELECT stage,count(*),min(icp_position),max(icp_position) "
            "FROM public.lab_arena_runs WHERE round_id=%s GROUP BY stage ORDER BY stage",
            (round_id,),
        )
        assert cursor.fetchall() == [
            (1, 2 * ((count + 1) // 2), 0, (count + 1) // 2 - 1),
            (2, 2 * (count // 2), (count + 1) // 2, count - 1),
        ]
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_runs SET status='accepted', "
            "terminal_cause='accepted', per_icp_score=70 "
            "WHERE round_id=%s AND stage=1", (round_id,),
        )
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='stage1_judged', "
            "stage1_scoring_plan_doc='{}'::jsonb WHERE round_id=%s", (round_id,),
        )
        cursor.execute("SET session_replication_role=origin")
        cursor.execute(
            "SELECT public.lab_arena_transition_round("
            "%s,'stage1_judged','stage1_scored',%s::jsonb)",
            (round_id, json.dumps({"finalists": [miner_id]})),
        )
        assert cursor.fetchone()[0]["round_status"] == "stage1_scored"
