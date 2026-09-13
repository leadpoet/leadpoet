"""Static and disposable-PostgreSQL checks for migration 232."""

from __future__ import annotations

import glob
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "scripts" / "232-retire-legacy-lead-and-research-lab-data.sql"

TARGET_TABLES = (
    "leads_private",
    "leads_private_backup",
    "miner_test_leads",
    "ops_research_lab_event_monitor_state",
    "ops_research_lab_event_notifications",
    "research_lab_auto_research_loop_events",
    "research_lab_provider_cost_events",
    "research_lab_provider_outcome_checkpoints_v2",
    "research_lab_scoring_category_results",
    "research_lab_scoring_dispatch_events",
    "research_lab_scoring_icp_events",
    "research_lab_scoring_icp_executions",
    "research_lab_scoring_run_events",
    "research_lab_scoring_runs",
    "test_leads_for_miners",
)

PRESERVED_TABLES = (
    "lab_arena_accepted_weight_states",
    "lab_arena_chain_outcomes",
    "lab_arena_company_judgment_reservations",
    "lab_arena_company_judgments",
    "lab_arena_judgment_cache",
    "lab_arena_ledger",
    "lab_arena_restart_claim_control",
    "lab_arena_rounds",
    "lab_arena_runs",
    "lab_arena_submission_credentials",
    "lab_arena_submissions",
    "qualification_baselines",
    "qualification_private_icp_sets",
    "research_lab_official_baseline_action_attempts_v1",
    "research_lab_official_baseline_action_terminals_v1",
    "research_lab_official_baseline_runs_v1",
    "research_lab_official_baseline_unit_closures_v1",
    "research_lab_attested_ancestry_activations_v2",
    "research_lab_attested_ancestry_checkpoints_v2",
    "research_lab_attested_artifact_links_v2",
    "research_lab_attested_boot_identities_v2",
    "research_lab_attested_business_artifact_links_v2",
    "research_lab_attested_execution_receipts_v2",
    "research_lab_attested_execution_results_v2",
    "research_lab_attested_host_operations_v2",
    "research_lab_attested_receipt_edges_v2",
    "research_lab_attested_receipt_transport_v2",
    "research_lab_attested_transport_attempts_v2",
)


def _without_comments(sql: str) -> str:
    return "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )


def test_migration_is_exact_and_fail_closed() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    executable = _without_comments(sql).upper()
    assert "CASCADE" not in executable
    assert "DROP TABLE IF EXISTS PUBLIC.RESEARCH_LAB_SCORING_" not in executable.replace(
        "RESEARCH_LAB_SCORING_CATEGORY_RESULTS", ""
    ).replace("RESEARCH_LAB_SCORING_DISPATCH_EVENTS", "").replace(
        "RESEARCH_LAB_SCORING_ICP_EVENTS", ""
    ).replace("RESEARCH_LAB_SCORING_ICP_EXECUTIONS", "").replace(
        "RESEARCH_LAB_SCORING_RUN_EVENTS", ""
    ).replace("RESEARCH_LAB_SCORING_RUNS", "")
    for table in TARGET_TABLES:
        assert f"DROP TABLE IF EXISTS public.{table}" in sql
    for table in PRESERVED_TABLES:
        assert f"DROP TABLE IF EXISTS public.{table}" not in sql
    assert "validation_evidence_private_lead_id_fkey" in sql
    assert "refresh-miner-test-leads" in sql
    assert "pg_catalog.md5(p.prosrc)" in sql


def _find_pg_bindir() -> str | None:
    candidates: list[str] = []
    initdb = shutil.which("initdb")
    if initdb:
        candidates.append(str(Path(initdb).parent))
    candidates += sorted(glob.glob("/opt/homebrew/opt/postgresql@*/bin"), reverse=True)
    candidates += sorted(glob.glob("/usr/local/opt/postgresql@*/bin"), reverse=True)
    candidates += sorted(glob.glob("/usr/lib/postgresql/*/bin"), reverse=True)
    for candidate in candidates:
        if all(
            (Path(candidate) / name).is_file()
            for name in ("initdb", "pg_ctl", "postgres", "createdb")
        ):
            return candidate
    return None


PG_BINDIR = _find_pg_bindir()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.fixture(scope="module")
def postgres_databases():
    if PG_BINDIR is None:
        pytest.skip("no PostgreSQL server binaries available")
    pytest.importorskip("psycopg2")
    bindir = Path(PG_BINDIR)
    data_dir = Path(tempfile.mkdtemp(prefix="legacy-data-retire-pg-"))
    socket_dir = Path(tempfile.mkdtemp(prefix="legacy-data-retire-socket-"))
    port = _free_port()
    started = False
    try:
        subprocess.run(
            [
                str(bindir / "initdb"),
                "-D",
                str(data_dir),
                "-U",
                "postgres",
                "--auth=trust",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )
        subprocess.run(
            [
                str(bindir / "pg_ctl"),
                "-D",
                str(data_dir),
                "-w",
                "-t",
                "30",
                "-l",
                str(data_dir / "server.log"),
                "-o",
                f"-p {port} -c listen_addresses=127.0.0.1 "
                f"-c unix_socket_directories={socket_dir}",
                "start",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
        )
        started = True
        databases = (
            "retire_success",
            "retire_drift",
            "retire_dependency",
            "retire_private_caller",
            "retire_cron_caller",
            "retire_guard_drift",
        )
        for database in databases:
            subprocess.run(
                [
                    str(bindir / "createdb"),
                    "-h",
                    "127.0.0.1",
                    "-p",
                    str(port),
                    "-U",
                    "postgres",
                    database,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        yield {
            database: {
                "host": "127.0.0.1",
                "port": port,
                "user": "postgres",
                "dbname": database,
            }
            for database in databases
        }
    finally:
        if started:
            subprocess.run(
                [
                    str(bindir / "pg_ctl"),
                    "-D",
                    str(data_dir),
                    "-w",
                    "-t",
                    "20",
                    "stop",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        shutil.rmtree(data_dir, ignore_errors=True)
        shutil.rmtree(socket_dir, ignore_errors=True)


def _setup_sql() -> str:
    preserved = "\n".join(
        f"CREATE TABLE public.{name} (id TEXT PRIMARY KEY, value TEXT NOT NULL);"
        f" INSERT INTO public.{name} VALUES ('kept', 'preserve');"
        for name in PRESERVED_TABLES
    )
    return f"""
CREATE SCHEMA cron;
CREATE TABLE cron.job (
    jobid BIGINT PRIMARY KEY,
    jobname TEXT UNIQUE NOT NULL,
    command TEXT NOT NULL
);
CREATE FUNCTION cron.unschedule(p_jobid BIGINT) RETURNS BOOLEAN
LANGUAGE plpgsql AS $$
BEGIN
    DELETE FROM cron.job WHERE jobid = p_jobid;
    RETURN FOUND;
END;
$$;
INSERT INTO cron.job VALUES
    (15, 'process-rep-scores', 'SELECT net.http_post()'),
    (17, 'refresh-dashboard-precalc', 'SELECT public.refresh_dashboard_precalc()'),
    (46, 'refresh-miner-test-leads', 'SELECT public.refresh_miner_test_leads()'),
    (47, 'reset-miner-rate-limits-daily', 'SELECT public.reset_miner_rate_limits_daily()');

{preserved}

CREATE TABLE public.research_loop_start_payments (ticket_id UUID);
CREATE TABLE public.research_loop_start_credit_events (ticket_id UUID);
CREATE TABLE public.research_loop_run_queue_events (ticket_id UUID);
CREATE TABLE public.research_loop_receipts (ticket_id UUID);
CREATE TABLE public.research_lab_candidate_artifacts (ticket_id UUID);

CREATE FUNCTION public.research_lab_ticket_has_unpaid_lifecycle_evidence(
    target_ticket_id UUID
)
RETURNS BOOLEAN
LANGUAGE plpgsql
VOLATILE
SET search_path = ''
AS $$
DECLARE
    legacy_evidence_exists BOOLEAN := FALSE;
BEGIN
    IF EXISTS (
        SELECT 1 FROM public.research_loop_start_payments p
        WHERE p.ticket_id = target_ticket_id
    )
    OR EXISTS (
        SELECT 1 FROM public.research_loop_start_credit_events ce
        WHERE ce.ticket_id = target_ticket_id
    )
    OR EXISTS (
        SELECT 1 FROM public.research_loop_run_queue_events q
        WHERE q.ticket_id = target_ticket_id
    )
    OR EXISTS (
        SELECT 1 FROM public.research_loop_receipts r
        WHERE r.ticket_id = target_ticket_id
    )
    OR EXISTS (
        SELECT 1 FROM public.research_lab_auto_research_loop_events l
        WHERE l.ticket_id = target_ticket_id
    )
    OR EXISTS (
        SELECT 1 FROM public.research_lab_candidate_artifacts c
        WHERE c.ticket_id = target_ticket_id
    )
    THEN
        RETURN TRUE;
    END IF;

    IF pg_catalog.to_regclass('public.research_loop_start_credits') IS NOT NULL THEN
        EXECUTE
            'SELECT EXISTS (SELECT 1 FROM public.research_loop_start_credits c '
            || 'WHERE c.ticket_id = $1)'
            INTO legacy_evidence_exists
            USING target_ticket_id;
        IF legacy_evidence_exists THEN
            RETURN TRUE;
        END IF;
    END IF;

    IF pg_catalog.to_regclass('public.research_loop_balance_ledger') IS NOT NULL THEN
        EXECUTE
            'SELECT EXISTS (SELECT 1 FROM public.research_loop_balance_ledger bl '
            || 'WHERE bl.ticket_id = $1)'
            INTO legacy_evidence_exists
            USING target_ticket_id;
        IF legacy_evidence_exists THEN
            RETURN TRUE;
        END IF;
    END IF;

    RETURN FALSE;
END;
$$;

CREATE TABLE public.leads_private (lead_id UUID PRIMARY KEY);
CREATE SEQUENCE public.leads_private_queue_position_seq;
CREATE TABLE public.validation_evidence_private (
    evidence_id UUID PRIMARY KEY,
    lead_id UUID REFERENCES public.leads_private(lead_id),
    value TEXT NOT NULL
);
INSERT INTO public.leads_private VALUES ('00000000-0000-4000-8000-000000000001');
INSERT INTO public.validation_evidence_private VALUES (
    '00000000-0000-4000-8000-000000000002',
    '00000000-0000-4000-8000-000000000001',
    'preserve'
);
CREATE TABLE public.leads_private_backup (lead_id UUID PRIMARY KEY);
CREATE TABLE public.miner_test_leads (id BIGSERIAL PRIMARY KEY);
CREATE TABLE public.test_leads_for_miners (id BIGSERIAL PRIMARY KEY);
CREATE TABLE public.ops_research_lab_event_monitor_state (id TEXT PRIMARY KEY);
CREATE TABLE public.ops_research_lab_event_notifications (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_auto_research_loop_events (
    id TEXT PRIMARY KEY,
    ticket_id UUID
);
CREATE TABLE public.research_lab_provider_outcome_checkpoints_v2 (id TEXT PRIMARY KEY);

CREATE TABLE public.research_lab_scoring_runs (id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_scoring_run_events (
    id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES public.research_lab_scoring_runs(id)
);
CREATE TABLE public.research_lab_scoring_icp_executions (
    id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES public.research_lab_scoring_runs(id)
);
CREATE TABLE public.research_lab_scoring_icp_events (
    id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES public.research_lab_scoring_runs(id),
    execution_id TEXT REFERENCES public.research_lab_scoring_icp_executions(id)
);
CREATE TABLE public.research_lab_provider_cost_events (
    id TEXT PRIMARY KEY,
    execution_id TEXT REFERENCES public.research_lab_scoring_icp_executions(id)
);
CREATE TABLE public.research_lab_scoring_dispatch_events (
    id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES public.research_lab_scoring_runs(id)
);
CREATE TABLE public.research_lab_scoring_category_results (
    id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES public.research_lab_scoring_runs(id)
);

CREATE VIEW public.research_lab_auto_research_loop_current AS
    SELECT id AS marker FROM public.research_lab_auto_research_loop_events;
CREATE VIEW public.research_lab_scoring_run_current AS
    SELECT id AS marker FROM public.research_lab_scoring_runs;
CREATE VIEW public.research_lab_scoring_icp_execution_current AS
    SELECT id AS marker FROM public.research_lab_scoring_icp_executions;
CREATE VIEW public.research_lab_scoring_dashboard_telemetry_v2 AS
    SELECT marker FROM public.research_lab_scoring_icp_execution_current;
CREATE VIEW public.research_lab_scoring_dashboard_telemetry_legacy AS
    SELECT id AS marker FROM public.research_lab_provider_cost_events;
CREATE VIEW public.research_lab_scoring_dashboard_telemetry AS
    SELECT marker FROM public.research_lab_scoring_dashboard_telemetry_v2
    UNION ALL
    SELECT marker FROM public.research_lab_scoring_dashboard_telemetry_legacy;
CREATE VIEW public.research_lab_private_benchmark_dashboard_telemetry AS
    SELECT marker FROM public.research_lab_scoring_dashboard_telemetry;
"""


def _connect(dsn):
    import psycopg2

    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    return connection


def test_migration_deletes_targets_and_preserves_shared_state(postgres_databases) -> None:
    connection = _connect(postgres_databases["retire_success"])
    try:
        with connection.cursor() as cursor:
            cursor.execute(_setup_sql())
            sql = MIGRATION.read_text(encoding="utf-8")
            cursor.execute(sql)
            cursor.execute(sql)
            cursor.execute(
                """
                SELECT
                    (SELECT count(*) FROM public.validation_evidence_private),
                    (SELECT count(*) FROM cron.job),
                    (SELECT count(*) FROM public.lab_arena_ledger),
                    (SELECT count(*) FROM public.qualification_private_icp_sets),
                    (SELECT count(*) FROM public.research_lab_attested_execution_receipts_v2),
                    to_regclass('public.leads_private') IS NULL,
                    to_regclass('public.research_lab_scoring_runs') IS NULL,
                    to_regclass('public.research_lab_provider_cost_events') IS NULL
                """
            )
            assert cursor.fetchone() == (1, 3, 1, 1, 1, True, True, True)
            cursor.execute(
                """
                SELECT count(*)
                FROM pg_constraint
                WHERE conrelid = 'public.validation_evidence_private'::regclass
                  AND conname = 'validation_evidence_private_lead_id_fkey'
                """
            )
            assert cursor.fetchone()[0] == 0
            ticket_id = "00000000-0000-4000-8000-000000000010"
            cursor.execute(
                "INSERT INTO public.research_loop_receipts VALUES (%s)",
                (ticket_id,),
            )
            cursor.execute(
                "SELECT public.research_lab_ticket_has_unpaid_lifecycle_evidence(%s)",
                (ticket_id,),
            )
            assert cursor.fetchone()[0] is True
    finally:
        connection.close()


def test_changed_reviewed_routine_rolls_back(postgres_databases) -> None:
    connection = _connect(postgres_databases["retire_drift"])
    try:
        with connection.cursor() as cursor:
            cursor.execute(_setup_sql())
            cursor.execute(
                """
                CREATE FUNCTION public.extract_lead_blob_fields()
                RETURNS TRIGGER LANGUAGE plpgsql AS $$ BEGIN RETURN NEW; END $$
                """
            )
            with pytest.raises(Exception, match="changed after review"):
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute("ROLLBACK")
            cursor.execute(
                "SELECT to_regclass('public.leads_private'), count(*) FROM cron.job"
            )
            assert cursor.fetchone() == ("leads_private", 4)
    finally:
        connection.close()


def test_unknown_dependency_rolls_back(postgres_databases) -> None:
    connection = _connect(postgres_databases["retire_dependency"])
    try:
        with connection.cursor() as cursor:
            cursor.execute(_setup_sql())
            cursor.execute(
                "CREATE VIEW public.unreviewed_lead_dependency AS "
                "SELECT lead_id FROM public.leads_private"
            )
            with pytest.raises(Exception, match="depend on it"):
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute("ROLLBACK")
            cursor.execute(
                "SELECT to_regclass('public.leads_private'), count(*) FROM cron.job"
            )
            assert cursor.fetchone() == ("leads_private", 4)
    finally:
        connection.close()


def test_private_dynamic_function_caller_rolls_back(postgres_databases) -> None:
    connection = _connect(postgres_databases["retire_private_caller"])
    try:
        with connection.cursor() as cursor:
            cursor.execute(_setup_sql())
            cursor.execute(
                """
                CREATE SCHEMA private;
                CREATE FUNCTION private.unreviewed_target_reader()
                RETURNS BIGINT LANGUAGE plpgsql AS $$
                DECLARE row_count BIGINT;
                BEGIN
                    EXECUTE 'SELECT count(*) FROM public.leads_private'
                        INTO row_count;
                    RETURN row_count;
                END;
                $$
                """
            )
            with pytest.raises(Exception, match="remaining functions reference"):
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute("ROLLBACK")
            cursor.execute(
                "SELECT to_regclass('public.leads_private'), count(*) FROM cron.job"
            )
            assert cursor.fetchone() == ("leads_private", 4)
    finally:
        connection.close()


def test_unreviewed_cron_caller_rolls_back(postgres_databases) -> None:
    connection = _connect(postgres_databases["retire_cron_caller"])
    try:
        with connection.cursor() as cursor:
            cursor.execute(_setup_sql())
            cursor.execute(
                "INSERT INTO cron.job VALUES "
                "(99, 'unreviewed-target-job', "
                "'SELECT count(*) FROM public.leads_private')"
            )
            with pytest.raises(Exception, match="remaining cron jobs reference"):
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute("ROLLBACK")
            cursor.execute(
                "SELECT to_regclass('public.leads_private'), count(*) FROM cron.job"
            )
            assert cursor.fetchone() == ("leads_private", 5)
    finally:
        connection.close()


def test_idempotent_rerun_rejects_shared_guard_drift(postgres_databases) -> None:
    connection = _connect(postgres_databases["retire_guard_drift"])
    try:
        with connection.cursor() as cursor:
            cursor.execute(_setup_sql())
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute(
                """
                CREATE OR REPLACE FUNCTION
                    public.research_lab_ticket_has_unpaid_lifecycle_evidence(
                        target_ticket_id UUID
                    )
                RETURNS BOOLEAN LANGUAGE plpgsql AS $$ BEGIN RETURN FALSE; END $$
                """
            )
            with pytest.raises(Exception, match="shared ticket evidence guard changed"):
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute("ROLLBACK")
            cursor.execute(
                """
                SELECT
                    to_regclass('public.leads_private') IS NULL,
                    md5(prosrc) <> '12321f60dfacab18ead8342c9656a9e8'
                FROM pg_proc
                WHERE oid =
                    'public.research_lab_ticket_has_unpaid_lifecycle_evidence(uuid)'::regprocedure
                """
            )
            assert cursor.fetchone() == (True, True)
    finally:
        connection.close()
