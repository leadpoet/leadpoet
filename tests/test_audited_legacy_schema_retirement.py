"""Static and disposable-PostgreSQL checks for migration 239."""

from __future__ import annotations

import glob
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "scripts" / "239-retire-audited-legacy-schema.sql"
MIGRATION_233 = ROOT / "scripts" / "233-retire-final-legacy-tables.sql"
TARGET_TABLES = (
    "banned_hotkeys",
    "contributor_attestations",
    "dashboard_miner_stats",
    "dashboard_precalc",
    "engine_trace_mappings",
    "epoch_audit_logs",
    "evidence_bundles",
    "execution_traces",
    "merkle_checkpoints",
    "miner_rate_limits",
    "ops_alert_current",
    "ops_alert_delivery_events",
    "ops_alert_events",
    "ops_alert_monitor_state",
    "ops_validator_registry",
    "qualification_baselines",
    "qualification_model_rate_limits",
    "qualification_models",
    "qualification_payments",
    "research_evaluation_score_bundle_events",
    "research_evaluation_score_bundles",
    "research_island_participation_snapshots",
    "research_lab_allocator_selection_records",
    "research_lab_attested_artifact_links",
    "research_lab_attested_execution_receipts",
    "research_lab_candidate_artifacts",
    "research_lab_candidate_claim",
    "research_lab_candidate_evaluation_events",
    "research_lab_candidate_promotion_events",
    "research_lab_company_label_examples",
    "research_lab_corpus_complete",
    "research_lab_gateway_control_events",
    "research_lab_inner_loop_activation_events",
    "research_lab_maintenance_lease",
    "research_lab_openrouter_key_refs",
    "research_lab_openrouter_privacy_proof_events",
    "research_lab_private_model_benchmark_bundles",
    "research_lab_private_model_benchmark_events",
    "research_lab_private_model_version_events",
    "research_lab_private_model_versions",
    "research_lab_private_repo_commit_events",
    "research_lab_provider_credential_envelopes_v2",
    "research_lab_provider_registry",
    "research_lab_provider_usage_ledger",
    "research_lab_public_benchmark_report_events",
    "research_lab_public_benchmark_reports",
    "research_lab_rejected_companies",
    "research_lab_results_ledger",
    "research_lab_rolling_icp_windows",
    "research_lab_score_calibration",
    "research_lab_shadow_monitor_windows",
    "research_lab_signed_transition_commands_v2",
    "research_lab_trace_pointer_quarantine",
    "research_loop_receipt_events",
    "research_loop_receipts",
    "research_loop_run_claim",
    "research_loop_run_queue_events",
    "research_loop_start_credit_events",
    "research_loop_start_payments",
    "research_loop_ticket_events",
    "research_loop_tickets",
    "research_trajectories",
    "research_trajectory_events",
    "validator_attestations",
    "validator_sourcing_epoch_inputs_v2",
)

RETAINED_TABLES = (
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
    "published_weight_bundles",
    "qualification_private_icp_sets",
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
    "research_lab_provider_evidence_cache_v2",
    "research_lab_stateful_subnet_epoch_boundaries_v1",
    "research_lab_stateful_subnet_epoch_candidates_v1",
    "research_lab_stateful_subnet_epoch_cutover_state_v1",
    "research_lab_stateful_subnet_epoch_cutovers_v1",
    "research_lab_stateful_subnet_epoch_snapshots_v1",
)


def _targets() -> tuple[str, ...]:
    return TARGET_TABLES


def _without_comments(sql: str) -> str:
    return "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )


def test_migration_has_exact_authorized_scope_and_bounded_guards() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    executable = _without_comments(sql).upper()
    targets = _targets()
    assert len(targets) == 65
    assert len(RETAINED_TABLES) == 30
    assert "CASCADE" not in executable
    assert "SET LOCAL lock_timeout = '5s'" in sql
    assert "SET LOCAL statement_timeout = '120s'" in sql
    for table in targets:
        assert sql.count(f"DROP TABLE IF EXISTS public.{table} RESTRICT;") == 1
    for table in RETAINED_TABLES:
        assert f"DROP TABLE IF EXISTS public.{table}" not in sql
    assert "IF actual_md5 IS NULL OR actual_md5 NOT IN (" in sql
    assert "archived_row_count = 21660" in sql
    assert "source_row_fingerprint = '5331decb81b3e4a11e0ebcc3d78ae606'" in sql
    assert "source_add_marker_row_count = 152" in sql
    assert "WHERE jobid IN (8,15,47)" in sql
    assert sql.count("routine.prosrc ~* closure_pattern") == 2
    assert sql.count("FROM pg_catalog.pg_depend AS dependency") == 2


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
    data_dir = Path(tempfile.mkdtemp(prefix="audited-retire-pg-"))
    socket_dir = Path(tempfile.mkdtemp(prefix="audited-retire-socket-"))
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
        names = (
            "retire_success",
            "retire_new_rows",
            "retire_dependency",
            "retire_sql_body_dependency",
            "retire_fk",
            "retire_missing_fence",
            "retire_routine_drift",
        )
        for name in names:
            subprocess.run(
                [
                    str(bindir / "createdb"),
                    "-h",
                    "127.0.0.1",
                    "-p",
                    str(port),
                    "-U",
                    "postgres",
                    name,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        yield {
            name: {
                "host": "127.0.0.1",
                "port": port,
                "user": "postgres",
                "dbname": name,
            }
            for name in names
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


def _current_fence_sql() -> str:
    sql = MIGRATION_233.read_text(encoding="utf-8")
    marker = (
        "CREATE OR REPLACE FUNCTION "
        "public.enforce_research_lab_stateful_epoch_fence_v1()"
    )
    start = sql.index(marker)
    body_start = sql.index("AS $$", start)
    end = sql.index("$$;", body_start + len("AS $$")) + len("$$;")
    return sql[start:end]


def _setup_sql(*, include_fence: bool = True) -> str:
    retained = "\n".join(
        f"CREATE TABLE public.{name} (id TEXT PRIMARY KEY, value TEXT NOT NULL);"
        f" INSERT INTO public.{name} VALUES ('kept', 'preserve');"
        for name in RETAINED_TABLES
    )
    targets = "\n".join(
        f"CREATE TABLE public.{name} (id TEXT PRIMARY KEY);"
        + (
            ""
            if name
            in {
                "research_lab_signed_transition_commands_v2",
                "validator_sourcing_epoch_inputs_v2",
            }
            else f" INSERT INTO public.{name} VALUES ('obsolete');"
        )
        for name in _targets()
    )
    fence = _current_fence_sql() if include_fence else ""
    return f"""
SET check_function_bodies = off;
CREATE SCHEMA storage;
CREATE FUNCTION storage.update_updated_at_column() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;
CREATE TABLE storage.objects (
    id TEXT PRIMARY KEY,
    updated_at TIMESTAMPTZ
);
CREATE TRIGGER update_objects_updated_at
BEFORE UPDATE ON storage.objects
FOR EACH ROW EXECUTE FUNCTION storage.update_updated_at_column();
CREATE SCHEMA source_add_history;
CREATE TABLE source_add_history.legacy_audit_rows (
    source_pk BIGINT PRIMARY KEY,
    row_doc JSONB NOT NULL
);
INSERT INTO source_add_history.legacy_audit_rows
VALUES (41960931, '{{"event_type":"SOURCE_ADD"}}');
CREATE TABLE source_add_history.archive_manifests (
    source_relation TEXT PRIMARY KEY,
    archive_floor BIGINT NOT NULL,
    historical_ceiling BIGINT NOT NULL,
    historical_row_count BIGINT NOT NULL,
    archived_row_count BIGINT NOT NULL,
    min_source_pk BIGINT NOT NULL,
    max_source_pk BIGINT NOT NULL,
    source_row_fingerprint TEXT NOT NULL,
    audit_row_count BIGINT NOT NULL,
    source_add_marker_row_count BIGINT NOT NULL,
    missing_nonzero_parent_count BIGINT NOT NULL,
    missing_parent_fingerprint TEXT NOT NULL
);
INSERT INTO source_add_history.archive_manifests VALUES (
    'public.transparency_log', 41960931, 41982623, 21660, 21660,
    41960931, 41982623, '5331decb81b3e4a11e0ebcc3d78ae606',
    324, 152, 4, 'e8fb898746467eab12a07bdc296bfc18'
);
{retained}
{targets}
{fence}
"""


def _connect(dsn):
    import psycopg2

    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    return connection


def _assert_rollback(connection, expected_message: str) -> None:
    with connection.cursor() as cursor:
        with pytest.raises(Exception, match=expected_message):
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
        cursor.execute("ROLLBACK")
        cursor.execute(
            "SELECT to_regclass('public.banned_hotkeys'), "
            "(SELECT count(*) FROM source_add_history.legacy_audit_rows)"
        )
        assert cursor.fetchone() == ("banned_hotkeys", 1)


def test_verbatim_migration_removes_65_and_preserves_30_and_archive(
    postgres_databases,
) -> None:
    connection = _connect(postgres_databases["retire_success"])
    try:
        with connection.cursor() as cursor:
            cursor.execute(_setup_sql())
            migration = MIGRATION.read_text(encoding="utf-8")
            cursor.execute(migration)
            cursor.execute(migration)
            cursor.execute(
                """
                SELECT
                    count(*) FILTER (WHERE namespace.nspname = 'public'),
                    count(*) FILTER (WHERE namespace.nspname = 'source_add_history')
                FROM pg_catalog.pg_class AS relation
                JOIN pg_catalog.pg_namespace AS namespace
                  ON namespace.oid = relation.relnamespace
                WHERE relation.relkind IN ('r','p')
                  AND namespace.nspname IN ('public','source_add_history')
                """
            )
            assert cursor.fetchone() == (30, 2)
            cursor.execute(
                """
                SELECT
                    (SELECT count(*) FROM public.lab_arena_ledger),
                    (SELECT count(*) FROM public.research_lab_attested_execution_receipts_v2),
                    (SELECT count(*) FROM public.research_lab_provider_evidence_cache_v2),
                    (SELECT count(*) FROM source_add_history.legacy_audit_rows),
                    to_regclass('public.banned_hotkeys') IS NULL,
                    to_regclass('public.research_trajectories') IS NULL,
                    to_regprocedure('storage.update_updated_at_column()') IS NOT NULL,
                    EXISTS (
                        SELECT 1 FROM pg_catalog.pg_trigger
                        WHERE tgrelid = 'storage.objects'::regclass
                          AND tgname = 'update_objects_updated_at'
                          AND NOT tgisinternal
                    )
                """
            )
            assert cursor.fetchone() == (1, 1, 1, 1, True, True, True, True)
    finally:
        connection.close()


def test_new_rows_in_protected_graph_child_roll_back(postgres_databases) -> None:
    connection = _connect(postgres_databases["retire_new_rows"])
    try:
        with connection.cursor() as cursor:
            cursor.execute(_setup_sql())
            cursor.execute(
                "INSERT INTO public.research_lab_signed_transition_commands_v2 "
                "VALUES ('new')"
            )
        _assert_rollback(connection, "signed transition rows appeared")
    finally:
        connection.close()


def test_unknown_view_dependency_rolls_back(postgres_databases) -> None:
    connection = _connect(postgres_databases["retire_dependency"])
    try:
        with connection.cursor() as cursor:
            cursor.execute(_setup_sql())
            cursor.execute(
                "CREATE VIEW public.unreviewed_retired_reader AS "
                "SELECT id FROM public.execution_traces"
            )
        _assert_rollback(connection, "unexpected view depends")
    finally:
        connection.close()


def test_parsed_sql_body_dependency_rolls_back(postgres_databases) -> None:
    connection = _connect(postgres_databases["retire_sql_body_dependency"])
    try:
        with connection.cursor() as cursor:
            cursor.execute(_setup_sql())
            cursor.execute(
                """
                CREATE FUNCTION storage.unreviewed_sql_dependency()
                RETURNS text
                LANGUAGE SQL
                BEGIN ATOMIC
                    SELECT id FROM public.execution_traces LIMIT 1;
                END
                """
            )
        _assert_rollback(connection, "unexpected routine depends")
    finally:
        connection.close()


def test_retained_fk_to_target_rolls_back(postgres_databases) -> None:
    connection = _connect(postgres_databases["retire_fk"])
    try:
        with connection.cursor() as cursor:
            cursor.execute(_setup_sql())
            cursor.execute(
                "ALTER TABLE public.lab_arena_ledger ADD COLUMN banned_id TEXT; "
                "ALTER TABLE public.lab_arena_ledger ADD CONSTRAINT unreviewed_fk "
                "FOREIGN KEY (banned_id) REFERENCES public.banned_hotkeys(id)"
            )
        _assert_rollback(connection, "retained table has an FK")
    finally:
        connection.close()


def test_missing_shared_fence_rolls_back(postgres_databases) -> None:
    connection = _connect(postgres_databases["retire_missing_fence"])
    try:
        with connection.cursor() as cursor:
            cursor.execute(_setup_sql(include_fence=False))
        _assert_rollback(connection, "shared epoch fence changed")
    finally:
        connection.close()


def test_changed_reviewed_routine_rolls_back(postgres_databases) -> None:
    connection = _connect(postgres_databases["retire_routine_drift"])
    try:
        with connection.cursor() as cursor:
            cursor.execute(_setup_sql())
            cursor.execute(
                "CREATE FUNCTION public.reset_daily_stats() RETURNS void "
                "LANGUAGE sql AS $$ SELECT $$"
            )
        _assert_rollback(connection, "reviewed routine .* changed after audit")
    finally:
        connection.close()
