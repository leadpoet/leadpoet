"""Verbatim migration 352 against disposable current PostgreSQL."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_contracts import base_round_configuration
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey, sha


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts/352-arena-2026-09-21-scoring-provider-quota-2000.sql"
ROUND = "arena-2026-09-21"
SUBMISSION = "sep21-score-quota-submission"
RUN = "sep21-score-quota-active"
MINER = hotkey("sep21-score-quota-miner")
LEASE_TOKEN_HASH = "sha256:" + "a" * 64


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
            "scorer_image_digest": "sha256:"
            "33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6",
            "scorer_image_reference": (
                "493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/"
                "sourcing-model@sha256:"
                "33012bf556b6fe46263ecb80183a8d344017b8232f51e95e582ac1ccf67ef1c6"
            ),
        }
    )
    configuration["schedule"] = {
        key: value.replace("2026-09-01", "2026-09-20").replace(
            "2026-09-02", "2026-09-21"
        )
        for key, value in configuration["schedule"].items()
    }
    return contracts.validate_round_configuration(configuration)


def _seed(connection, *, status="stage1_scoring", wrong_profile=False):
    configuration = _configuration()
    run_stage = 2 if status == "stage2_scoring" else 1
    active_position = 16 if run_stage == 2 else 6
    pending_position = active_position + 1
    if wrong_profile:
        configuration["scoring_call_quotas"]["deepline"] = 41
    participants = [
        {"submission_id": SUBMISSION, "miner_hotkey": MINER, "is_king": False}
    ]
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
            "configuration_doc,rewards_enabled,participants,evaluation_date,"
            "stage1_scoring_plan_doc) VALUES ("
            "%s,%s,9,4,%s::jsonb,TRUE,%s::jsonb,'2026-09-21',%s::jsonb)",
            (
                ROUND,
                status,
                json.dumps(configuration),
                json.dumps(participants),
                json.dumps({"plan": "preserve", "generation": 4}),
            ),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions("
            "submission_id,round_id,miner_hotkey,status,is_king,source_ref,"
            "source_size_bytes,submission_doc,frozen_at) VALUES ("
            "%s,%s,%s,'frozen',FALSE,%s,4096,%s::jsonb,"
            "'2026-09-21T00:00:00Z')",
            (
                SUBMISSION,
                ROUND,
                MINER,
                f"arena/{ROUND}/sources/{SUBMISSION}.tar.gz",
                json.dumps({"source": "preserve", "size": 4096}),
            ),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_runs("
            "run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,"
            "icp_position,attempt,kind,status,runner_hotkey,lease_token_hash,"
            "lease_generation,stage_generation,lease_expires_at,scored_run_id,"
            "judgment_input_hash,judgment_scope_doc) VALUES ("
            "%s,'sep21-score-quota-assignment',%s,%s,%s,%s,%s,1,'score',"
            "'leased',%s,%s,2,4,clock_timestamp()+interval '1 hour',"
            "'sep21-execution-preserved',%s,%s::jsonb),"
            "('sep21-score-quota-pending','sep21-score-quota-pending-assignment',"
            "%s,%s,%s,%s,%s,2,'score','pending',NULL,NULL,0,4,NULL,"
            "'sep21-execution-pending',%s,%s::jsonb)",
            (
                RUN,
                ROUND,
                SUBMISSION,
                MINER,
                run_stage,
                active_position,
                MINER,
                LEASE_TOKEN_HASH,
                "sha256:" + "b" * 64,
                json.dumps({"scorer": "33012bf", "preserve": True}),
                ROUND,
                SUBMISSION,
                MINER,
                run_stage,
                pending_position,
                "sha256:" + "c" * 64,
                json.dumps({"scorer": "33012bf", "preserve": True}),
            ),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_ledger("
            "entry_kind,miner_hotkey,round_id,submission_id,run_id,stage,"
            "call_identity,provider,operation_id,funding_source,"
            "amount_microusd,entry_doc) SELECT 'reservation',%s,%s,%s,%s,%s,"
            "'sha256:'||md5(%s||series::text)||md5('tail-'||%s||series::text),"
            "'deepline','deepline.execute','miner_key',0,'{}'::jsonb "
            "FROM generate_series(1,40) AS series",
            (MINER, ROUND, SUBMISSION, RUN, run_stage, RUN, RUN),
        )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()


def _state(connection):
    state = {}
    with connection.cursor() as cursor:
        for table, order in (
            ("lab_arena_rounds", "round_id"),
            ("lab_arena_submissions", "submission_id"),
            ("lab_arena_runs", "run_id"),
            ("lab_arena_ledger", "entry_id"),
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


@pytest.mark.parametrize("scoring_status", ("stage1_scoring", "stage2_scoring"))
def test_active_score_leases_cross_old_deepline_limit_without_state_rewrite(
    database, scoring_status,
):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        _seed(connection, status=scoring_status)
        store = ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))
        try:
            assert store.run_quota_snapshot(RUN, LEASE_TOKEN_HASH)["providers"][
                "deepline"
            ] == {"limit": 40, "used": 40, "remaining": 0, "inflight": 40}
            before = _state(connection)
            _execute(connection)
            after = _state(connection)

            expected = copy.deepcopy(before)
            expected_round = expected["lab_arena_rounds"][0]
            expected_round["configuration_doc"]["scoring_call_quotas"] = dict(
                contracts.SCORING_CALL_QUOTAS_PER_WORK_ITEM
            )
            expected_round["updated_at"] = after["lab_arena_rounds"][0][
                "updated_at"
            ]
            assert after == expected
            assert store.run_quota_snapshot(RUN, LEASE_TOKEN_HASH)["providers"][
                "deepline"
            ] == {
                "limit": 2000,
                "used": 40,
                "remaining": 1960,
                "inflight": 40,
            }

            identity = contracts.provider_call_identity(
                attempt=1,
                assignment_id="sep21-score-quota-assignment",
                icp_position=16 if scoring_status == "stage2_scoring" else 6,
                action_sequence=41,
                operation_id="deepline.execute",
                request_hash=sha("sep21-deepline-41"),
            )
            admitted = store.reserve_call(
                run_id=RUN,
                lease_token_hash=LEASE_TOKEN_HASH,
                call_identity=identity,
                operation_id="deepline.execute",
                provider="deepline",
                funding_source="miner_key",
                amount_microusd=0,
                call_doc={},
            )
            assert admitted["status"] == "reserved"

            replay_state = _state(connection)
            _execute(connection)
            assert _state(connection) == replay_state
        finally:
            store.close()


@pytest.mark.parametrize("status", ("stage1", "published", "cancelled"))
def test_non_scoring_or_terminal_round_refuses_without_mutation(database, status):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        _seed(connection, status=status)
        before = _state(connection)
        with pytest.raises(
            psycopg2.Error,
            match="requires an active unpublished scoring round",
        ):
            _execute(connection)
        connection.rollback()
        assert _state(connection) == before


def test_unrecognized_scoring_profile_refuses_without_mutation(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        _seed(connection, wrong_profile=True)
        before = _state(connection)
        with pytest.raises(psycopg2.Error, match="profile differs"):
            _execute(connection)
        connection.rollback()
        assert _state(connection) == before
