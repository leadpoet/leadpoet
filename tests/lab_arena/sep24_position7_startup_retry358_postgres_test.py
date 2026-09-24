"""Disposable PostgreSQL proof for the exact Sep24 position-7 retry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_arena import scoring
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_contracts import base_round_configuration
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey


ROOT = Path(__file__).parents[2]
MIGRATION = ROOT / "scripts/358-arena-2026-09-24-position7-startup-retry.sql"
ROUND = "arena-2026-09-24"
BASELINE = "baseline-2026-09-24"
BASELINE_HOTKEY = "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"
PUBLIC_RUNNER = "5Chnr6Y72gdfTFdoZnsCvkndKpMk8jAtt9JAYKaNG3LmU4BW"
ASSIGNMENT = f"{ROUND}:{BASELINE}:1:7"
SOURCE_REF = f"arena/{ROUND}/sources/{BASELINE}.tar.gz"
CONFIG_HASH = "sha256:51db3a10993646f6aa07503bd76dea9244c4c97879f54f2a1b67b7eebcd3c8da"
PARTICIPANTS_HASH = "sha256:d8f69ece7505dfd31e3570b2fc21f3589f80aaa49d6b46adb1f82a986232029a"
BANK_HASH = "sha256:48d6387354b9cead518aff20c8b681bb8236aec64efcbe01dc05bcab1f4d6972"

ATTEMPT1_RESULT = {
    "finished_at": "2026-09-24T00:02:09Z",
    "resource_summary": {
        "cpu_seconds": 0.0,
        "max_rss_bytes": 0,
        "provider_call_count": 0,
        "stderr_bytes": 0,
        "stdout_bytes": 0,
        "wall_seconds": 0.0,
    },
    "schema_version": "leadpoet.lab_arena.run_result.v1",
    "started_at": "2026-09-24T00:01:36Z",
    "terminal_status": "model_error",
}
ATTEMPT2_RESULT = {
    "finished_at": "2026-09-24T01:02:14Z",
    "resource_summary": {
        "cpu_seconds": 78.432882,
        "max_rss_bytes": 334307328,
        "provider_call_count": 119,
        "stderr_bytes": 963,
        "stdout_bytes": 0,
        "wall_seconds": 3577.251937283203,
        "web_egress": {
            "active_limit_rejection_count": 0,
            "byte_limit_rejection_count": 0,
            "cleanup_block_rejection_count": 0,
            "connection_count": 20,
            "download_bytes": 25329812,
            "exit_fingerprint": "b8ed2a8279876c4d",
            "failure_count": 0,
            "policy_version": "webshare_parallel_v1",
            "total_limit_rejection_count": 0,
            "upload_bytes": 3579152,
            "worker_slot": 8,
        },
    },
    "schema_version": "leadpoet.lab_arena.run_result.v1",
    "started_at": "2026-09-24T00:02:36Z",
    "terminal_status": "model_error",
}


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(CURRENT_SERVICE_MIGRATIONS)


def _configuration() -> dict:
    config = base_round_configuration()
    config.update(
        round_id=ROUND,
        mode="live",
        network_name="finney",
        netuid=71,
        integrity_policy="arena_integrity_v1",
        intent_details_policy="intent_details_v1",
        execution_sequence_policy="baseline_scored_first_v1",
        sourcing_cost_eligibility_policy="successful_calls_per_icp_v1",
        call_quotas={"deepline": 200, "openrouter": 2000, "scrapingdog": 200},
        scoring_call_quotas={
            "deepline": 2000,
            "openrouter": 2000,
            "scrapingdog": 2000,
        },
        stage_1_icp_count=5,
        stage_2_icp_count=5,
        max_attempts_per_assignment=2,
        icp_wall_clock_seconds=3600,
        runner_slot_ceiling=20,
        execution_icp_cap_microusd=4_000_000,
        cost_per_company_microusd=800_000,
        scorer_policy=scoring.build_scorer_policy(
            scoring_adapter_version="qualification_integrity_v2",
            intent_details=True,
        ),
    )
    config.pop("contact_policy", None)
    config["schedule"] = {
        "benchmark_deadline": "2026-09-24T00:30:00Z",
        "final_scoring_close": "2026-09-24T20:30:02Z",
        "publication_deadline": "2026-09-24T20:30:03Z",
        "stage_1_close": "2026-09-24T04:30:01Z",
        "stage_1_scoring_close": "2026-09-24T11:00:01Z",
        "stage_1_start": "2026-09-24T00:30:01Z",
        "stage_2_close": "2026-09-24T14:00:02Z",
        "stage_2_start": "2026-09-24T11:00:02Z",
        "submission_cutoff": "2026-09-24T00:00:00Z",
        "submission_open": "2026-09-23T00:00:00Z",
    }
    return config


def _participants() -> list[dict]:
    rows = [
        {
            "submission_id": f"fixture-miner-{index}",
            "miner_hotkey": hotkey(f"sep24-358-{index}"),
            "source_ref": f"arena/{ROUND}/sources/fixture-miner-{index}.tar.gz",
            "source_size_bytes": 100_000 + index,
            "is_king": False,
        }
        for index in range(8)
    ]
    rows.append(
        {
            "submission_id": BASELINE,
            "miner_hotkey": BASELINE_HOTKEY,
            "source_ref": SOURCE_REF,
            "source_size_bytes": 817_521,
            "is_king": True,
        }
    )
    return rows


def _bank() -> list[dict]:
    return [{"icp_id": f"sep24-{index}", "prompt": f"ICP {index}"} for index in range(10)]


def _jsonb_hash(cursor, value) -> str:
    cursor.execute(
        "SELECT 'sha256:'||encode(extensions.digest((%s::jsonb)::text,'sha256'),'hex')",
        (json.dumps(value),),
    )
    return cursor.fetchone()[0]


def _migration_sql(cursor) -> str:
    sql = MIGRATION.read_text(encoding="utf-8")
    replacements = {
        CONFIG_HASH: _jsonb_hash(cursor, _configuration()),
        PARTICIPANTS_HASH: _jsonb_hash(cursor, _participants()),
        BANK_HASH: _jsonb_hash(cursor, _bank()),
        "pg_catalog.clock_timestamp() + INTERVAL '60 minutes'":
            "'2026-09-24T02:00:00Z'::TIMESTAMPTZ + INTERVAL '60 minutes'",
    }
    for old, new in replacements.items():
        assert sql.count(old) == 1, old
        sql = sql.replace(old, new)
    return sql


def _seed(cursor) -> None:
    config = _configuration()
    participants = _participants()
    bank = _bank()
    cursor.execute("SET session_replication_role=replica")
    cursor.execute(
        "INSERT INTO public.qualification_private_icp_sets "
        "(set_id,icps,active_from,active_until,is_active) VALUES "
        "(20260923,%s::jsonb,'2026-09-23T00:00:00Z','2026-09-24T00:00:00Z',false)",
        (json.dumps(bank),),
    )
    cursor.execute(
        "INSERT INTO public.lab_arena_rounds "
        "(round_id,status,status_generation,stage_generation,configuration_doc,"
        "rewards_enabled,participants,benchmark_ref,evaluation_date,"
        "promotion_required,icp_set_date,"
        "champion_funding_frozen) VALUES "
        "(%s,'stage1',2,1,%s::jsonb,true,%s::jsonb,%s,'2026-09-24',"
        "true,'2026-09-23',true)",
        (ROUND, json.dumps(config), json.dumps(participants), f"arena/{ROUND}/benchmark.json"),
    )
    submission_doc = {
        "source_ref": SOURCE_REF,
        "source_size_bytes": 817_521,
        "consent": {"public_rerun": True},
        "is_king": True,
    }
    cursor.execute(
        "INSERT INTO public.lab_arena_submissions "
        "(submission_id,round_id,miner_hotkey,status,is_king,submission_doc,"
        "source_ref,source_size_bytes) VALUES (%s,%s,%s,'frozen',true,%s::jsonb,%s,817521)",
        (BASELINE, ROUND, BASELINE_HOTKEY, json.dumps(submission_doc), SOURCE_REF),
    )
    for position in range(6):
        assignment = f"{ROUND}:{BASELINE}:1:{position}"
        cursor.execute(
            "INSERT INTO public.lab_arena_runs "
            "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,status,terminal_cause,result_doc,output_ref,"
            "lease_generation,stage_generation,kind,runner_hotkey) VALUES "
            "(%s,%s,%s,%s,%s,1,%s,1,'accepted','accepted',"
            "'{\"terminal_status\":\"accepted\"}'::jsonb,%s,1,1,'execute',%s)",
            (
                assignment + ":1",
                assignment,
                ROUND,
                BASELINE,
                BASELINE_HOTKEY,
                position,
                f"arena/{ROUND}/outputs/{assignment}:1.json",
                BASELINE_HOTKEY,
            ),
        )
    for attempt, result, runner, previous in (
        (1, ATTEMPT1_RESULT, PUBLIC_RUNNER, None),
        (2, ATTEMPT2_RESULT, BASELINE_HOTKEY, PUBLIC_RUNNER),
    ):
        cursor.execute(
            "INSERT INTO public.lab_arena_runs "
            "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,status,terminal_cause,result_doc,lease_generation,"
            "stage_generation,kind,runner_hotkey,previous_runner_hotkey,lease_token_hash) "
            "VALUES (%s,%s,%s,%s,%s,1,7,%s,'failed','model_error',%s::jsonb,%s,1,"
            "'execute',%s,%s,'sha256:'||repeat(%s,64))",
            (
                f"{ASSIGNMENT}:{attempt}",
                ASSIGNMENT,
                ROUND,
                BASELINE,
                BASELINE_HOTKEY,
                attempt,
                json.dumps(result),
                attempt,
                runner,
                previous,
                str(attempt),
            ),
        )
    for index in range(120):
        identity = "sha256:" + f"{index + 1:064x}"
        amount = 23_868 if index == 119 else 2_000
        for entry_kind in ("reservation", "dispatch", "settlement"):
            cursor.execute(
                "INSERT INTO public.lab_arena_ledger "
                "(entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
                "call_identity,provider,operation_id,funding_source,amount_microusd,"
                "entry_doc,terminal_response) VALUES "
                "(%s,%s,%s,%s,%s,1,%s,'openrouter','openrouter.responses','host',"
                "%s,'{}'::jsonb,%s::jsonb)",
                (
                    entry_kind,
                    BASELINE_HOTKEY,
                    ROUND,
                    BASELINE,
                    f"{ASSIGNMENT}:2",
                    identity,
                    amount,
                    json.dumps({"status": 200}) if entry_kind == "settlement" else None,
                ),
            )
    cursor.execute("SET session_replication_role=origin")


def _cleanup(cursor) -> None:
    cursor.execute("SET session_replication_role=replica")
    cursor.execute("DELETE FROM public.lab_arena_ledger WHERE round_id=%s", (ROUND,))
    cursor.execute("DELETE FROM public.lab_arena_runs WHERE round_id=%s", (ROUND,))
    cursor.execute("DELETE FROM public.lab_arena_submissions WHERE round_id=%s", (ROUND,))
    cursor.execute("DELETE FROM public.lab_arena_rounds WHERE round_id=%s", (ROUND,))
    cursor.execute("DELETE FROM public.qualification_private_icp_sets WHERE set_id=20260923")
    cursor.execute("SET session_replication_role=origin")


def _snapshot(cursor) -> dict:
    result = {}
    for key, table, order in (
        ("round", "lab_arena_rounds", "round_id"),
        ("submissions", "lab_arena_submissions", "submission_id"),
        ("runs", "lab_arena_runs", "run_id"),
        ("ledger", "lab_arena_ledger", "entry_id"),
    ):
        cursor.execute(
            f"SELECT coalesce(jsonb_agg(to_jsonb(x) ORDER BY {order}),'[]'::jsonb) "
            f"FROM public.{table} x WHERE round_id=%s",
            (ROUND,),
        )
        result[key] = cursor.fetchone()[0]
    cursor.execute(
        "SELECT to_jsonb(x) FROM public.qualification_private_icp_sets x WHERE set_id=20260923"
    )
    result["bank"] = cursor.fetchone()[0]
    return result


def test_migration_keeps_live_commitments_and_time_capacity_guard() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "pg_catalog.clock_timestamp() + INTERVAL '60 minutes'" in sql
    assert "status IN ('pending', 'leased', 'submitted')" in sql
    assert CONFIG_HASH in sql and PARTICIPANTS_HASH in sql and BANK_HASH in sql
    assert "f5a95ff38865c0249750ec45f66c06794016abef" in sql
    assert "423c6c6ac369ec96404083852887f5513bed5e891d0140232221da65bd229329" in sql


def test_append_only_retry_preserves_history_costs_and_replays(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        with connection.cursor() as cursor:
            _seed(cursor)
            connection.commit()
            before = _snapshot(cursor)
            sql = _migration_sql(cursor)
            cursor.execute(sql)
            after = _snapshot(cursor)
            assert after["round"] == before["round"]
            assert after["bank"] == before["bank"]
            assert after["submissions"] == before["submissions"]
            assert after["ledger"] == before["ledger"]
            assert after["runs"][:-1] == before["runs"]
            retry = after["runs"][-1]
            assert (
                retry["run_id"],
                retry["assignment_id"],
                retry["attempt"],
                retry["status"],
                retry["stage_generation"],
                retry["previous_runner_hotkey"],
            ) == (ASSIGNMENT + ":3", ASSIGNMENT, 3, "pending", 1, BASELINE_HOTKEY)
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "UPDATE public.lab_arena_runs SET status='accepted',"
                "terminal_cause='accepted',result_doc='{\"terminal_status\":\"accepted\"}'::jsonb,"
                "output_ref=%s WHERE run_id=%s",
                (f"arena/{ROUND}/outputs/{ASSIGNMENT}:3.json", ASSIGNMENT + ":3"),
            )
            cursor.execute("SET session_replication_role=origin")
            connection.commit()
            progressed = _snapshot(cursor)
            cursor.execute(sql)
            assert _snapshot(cursor) == progressed
            _cleanup(cursor)
            connection.commit()
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            "UPDATE public.lab_arena_rounds SET stage1_scoring_plan_doc='{}'::jsonb "
            f"WHERE round_id='{ROUND}'",
            "active unscored stage1",
        ),
        (
            "UPDATE public.qualification_private_icp_sets SET icps="
            "jsonb_set(icps,'{0,prompt}','\"changed\"'::jsonb) WHERE set_id=20260923",
            "source, bank, or policy differs",
        ),
        (
            "UPDATE public.lab_arena_runs SET result_doc='{}'::jsonb "
            f"WHERE run_id='{ASSIGNMENT}:1'",
            "attempts differ",
        ),
        (
            "UPDATE public.lab_arena_ledger SET amount_microusd=amount_microusd+1 "
            f"WHERE entry_id=(SELECT max(entry_id) FROM public.lab_arena_ledger WHERE run_id='{ASSIGNMENT}:2')",
            "accounting differs or is open",
        ),
    ),
)
def test_migration_rejects_drift_atomically(database, mutation, message):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        with connection.cursor() as cursor:
            _seed(cursor)
            connection.commit()
            sql = _migration_sql(cursor)
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(mutation)
            cursor.execute("SET session_replication_role=origin")
            connection.commit()
            before = _snapshot(cursor)
            with pytest.raises(psycopg2.Error, match=message):
                cursor.execute(sql)
            connection.rollback()
            assert _snapshot(cursor) == before
            _cleanup(cursor)
            connection.commit()
    finally:
        connection.close()


def test_migration_rejects_full_capacity_atomically(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    try:
        with connection.cursor() as cursor:
            _seed(cursor)
            connection.commit()
            sql = _migration_sql(cursor)
            cursor.execute("SET session_replication_role=replica")
            cursor.execute(
                "INSERT INTO public.lab_arena_runs "
                "(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
                "icp_position,attempt,status,lease_generation,stage_generation,kind) "
                "SELECT %s||n, %s||n, %s, %s, %s, 1, 8, 1, 'pending', 1, 1, 'execute' "
                "FROM generate_series(1,20) AS n",
                (f"{ROUND}:capacity:", f"{ROUND}:capacity:", ROUND, BASELINE, BASELINE_HOTKEY),
            )
            cursor.execute("SET session_replication_role=origin")
            connection.commit()
            before = _snapshot(cursor)
            with pytest.raises(psycopg2.Error, match="no execution capacity"):
                cursor.execute(sql)
            connection.rollback()
            assert _snapshot(cursor) == before
            _cleanup(cursor)
            connection.commit()
    finally:
        connection.close()
