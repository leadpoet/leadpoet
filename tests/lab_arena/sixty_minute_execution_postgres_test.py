"""The 60-minute lease and unstarted Sep22 transition on disposable PostgreSQL."""

from __future__ import annotations

from dataclasses import replace
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
from tests.lab_arena.test_lab_arena_service_round import Harness, _start_round
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.explicit_90m_lease_postgres_test import (
    _exercise_cost_recovery,
    _service_claim,
)


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
LEASE_MIGRATION = SCRIPTS / "354-lab-arena-60m-lease.sql"
ROUND_MIGRATION = SCRIPTS / "355-arena-2026-09-22-60m-execution.sql"
ROUND = "arena-2026-09-22"


@pytest.fixture(scope="module")
def database():
    migrations = CURRENT_SERVICE_MIGRATIONS + (
        "264-lab-arena-codex-cost-reconciliation.sql",
        "311-lab-arena-per-icp-closed-billing-reconciliation.sql",
        "312-lab-arena-temporary-hold-admission.sql",
        "314-lab-arena-openrouter-web-search-reservation.sql",
        "319-lab-arena-quota-sourcing-cost.sql",
        "321-lab-arena-confirmed-cost-admission.sql",
    )
    for psycopg2, dsn in database_with_lab_arena_migration(migrations):
        with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
            cursor.execute((SCRIPTS / "329-lab-arena-explicit-90m-lease.sql").read_text())
            cursor.execute(LEASE_MIGRATION.read_text())
        yield psycopg2, dsn


def _configuration(round_id=ROUND, *, count=10):
    config = base_round_configuration()
    config.update(
        round_id=round_id,
        mode="live",
        checkpoint_deadline_policy=contracts.CHECKPOINT_DEADLINE_POLICY,
        icp_wall_clock_seconds=2700,
        lease_ttl_seconds=3600,
        stage_1_icp_count=(count + 1) // 2,
        stage_2_icp_count=count // 2,
    )
    config["schedule"] = {
        key: value.replace("2026-09-01", "2026-09-21").replace(
            "2026-09-02", "2026-09-22"
        ) for key, value in config["schedule"].items()
    }
    return contracts.validate_round_configuration(config)


def _seed(connection, *, with_run=False, frozen=False):
    config = _configuration()
    source_ref = f"arena/{ROUND}/sources/sep22-source.tar.gz"
    with connection.cursor() as cursor:
        cursor.execute("SET session_replication_role=replica")
        cursor.execute(
            "TRUNCATE public.lab_arena_ledger,public.lab_arena_runs,"
            "public.lab_arena_submissions,public.lab_arena_rounds RESTART IDENTITY CASCADE"
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds "
            "(round_id,status,status_generation,stage_generation,configuration_doc,rewards_enabled) "
            "VALUES (%s,'open',0,0,%s::jsonb,FALSE)",
            (ROUND, json.dumps(config)),
        )
        historical = dict(config, round_id="arena-2026-09-21")
        cursor.execute(
            "INSERT INTO public.lab_arena_rounds "
            "(round_id,status,status_generation,stage_generation,configuration_doc,rewards_enabled) "
            "VALUES ('arena-2026-09-21','published',1,2,%s::jsonb,FALSE)",
            (json.dumps(historical),),
        )
        cursor.execute(
            "INSERT INTO public.lab_arena_submissions "
            "(submission_id,round_id,miner_hotkey,status,is_king,source_ref,"
            "source_size_bytes,submission_doc,frozen_at) VALUES "
            "('sep22-source',%s,%s,%s,FALSE,%s,4096,'{}'::jsonb,%s)",
            (
                ROUND, hotkey("sep22-60m"),
                "frozen" if frozen else "accepted",
                source_ref,
                "2026-09-21T23:59:59Z" if frozen else None,
            ),
        )
        if with_run:
            cursor.execute(
                "INSERT INTO public.lab_arena_runs "
                "(run_id,assignment_id,round_id,submission_id,miner_hotkey,"
                "stage,icp_position,attempt,kind,status,stage_generation) VALUES "
                "('sep22-60m-run','sep22-60m-assignment',%s,'sep22-source',%s,"
                "1,0,1,'execute','pending',0)",
                (ROUND, hotkey("sep22-60m")),
            )
        cursor.execute("SET session_replication_role=origin")
    connection.commit()
    return config


def _read_configuration(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id=%s",
            (ROUND,),
        )
        return cursor.fetchone()[0]


def test_exact_4500_guard_preserves_rpc_privileges_and_replays(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        signatures = (
            "public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)",
            "public.lab_arena_mark_uncertain(text,text,text,jsonb,integer)",
            "public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)",
            "public.lab_arena_settle_call(text,text,text,bigint,jsonb,integer)",
        )
        cursor.execute(
            "SELECT p.oid::regprocedure::text, p.proowner, p.proacl "
            "FROM pg_proc p WHERE p.oid=ANY(%s::regprocedure[]) ORDER BY 1",
            (list(signatures),),
        )
        privileges = cursor.fetchall()
        cursor.execute(LEASE_MIGRATION.read_text())
        cursor.execute(LEASE_MIGRATION.read_text())
        cursor.execute(
            "SELECT p.oid::regprocedure::text, p.proowner, p.proacl "
            "FROM pg_proc p WHERE p.oid=ANY(%s::regprocedure[]) ORDER BY 1",
            (list(signatures),),
        )
        assert cursor.fetchall() == privileges
        for signature in signatures:
            cursor.execute("SELECT pg_get_functiondef(%s::regprocedure)", (signature,))
            definition = cursor.fetchone()[0]
            assert definition.count("AND COALESCE(p_lease_ttl_seconds, 0) <> 4500") == 1
            assert definition.count("AND COALESCE(p_lease_ttl_seconds, 0) <> 6300") == 1


def test_new_60m_round_claim_and_paid_call_use_exact_4500_lease(database, tmp_path):
    psycopg2, dsn = database
    connect = lambda: psycopg2.connect(**dsn)
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        benchmark_icp_count=10,
        checkpoint_deadline_enabled=True,
        runner_slot_ceiling=20,
    )
    harness.service.config.daily_icp_source = lambda **kwargs: {
        "status": "ready", "set_id": int(kwargs["set_id"]), "icps": daily_icps()[:10],
    }
    _start_round(harness, day=30, epoch=32800)
    config = harness.service.store.get_round(harness.round_id)["configuration_doc"]
    assert (
        config["checkpoint_deadline_policy"],
        config["icp_wall_clock_seconds"],
        config["lease_ttl_seconds"],
        contracts.benchmark_icp_count(config),
    ) == (contracts.CHECKPOINT_60M_DEADLINE_POLICY, 3600, 4500, 10)
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    lease = _service_claim(harness)
    assert lease["icp_wall_clock_seconds"] == 3600
    assert lease["lease_ttl_seconds"] == 4500
    _exercise_cost_recovery(harness.service.store, lease, 4500, "60m-call")


def test_sep22_changes_only_deadline_and_preserves_dynamic_ten_icps(database):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection, connection.cursor() as cursor:
        before = _seed(connection)
        cursor.execute(
            "SELECT to_jsonb(r) FROM public.lab_arena_rounds r "
            "WHERE round_id='arena-2026-09-21'"
        )
        historical = cursor.fetchone()[0]
        cursor.execute(ROUND_MIGRATION.read_text())
        after = _read_configuration(connection)
        assert {
            key: after.pop(key)
            for key in (
                "checkpoint_deadline_policy", "icp_wall_clock_seconds",
                "lease_ttl_seconds",
            )
        } == {
            "checkpoint_deadline_policy": contracts.CHECKPOINT_60M_DEADLINE_POLICY,
            "icp_wall_clock_seconds": 3600,
            "lease_ttl_seconds": 4500,
        }
        for key in (
            "checkpoint_deadline_policy", "icp_wall_clock_seconds",
            "lease_ttl_seconds",
        ):
            before.pop(key)
        assert after == before
        assert contracts.benchmark_icp_count(after) == 10
        cursor.execute(
            "SELECT to_jsonb(r) FROM public.lab_arena_rounds r "
            "WHERE round_id='arena-2026-09-21'"
        )
        assert cursor.fetchone()[0] == historical
        cursor.execute(ROUND_MIGRATION.read_text())


@pytest.mark.parametrize("blocker", ["run", "frozen"])
def test_sep22_refuses_started_or_frozen_round(database, blocker):
    psycopg2, dsn = database
    with psycopg2.connect(**dsn) as connection:
        before = _seed(connection, with_run=blocker == "run", frozen=blocker == "frozen")
        with pytest.raises(psycopg2.Error, match="open, unfrozen, unstarted"):
            with connection.cursor() as cursor:
                cursor.execute(ROUND_MIGRATION.read_text())
        connection.rollback()
        assert _read_configuration(connection) == before
