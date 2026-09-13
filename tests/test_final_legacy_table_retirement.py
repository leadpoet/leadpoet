"""Static and disposable-PostgreSQL checks for migration 233."""

from __future__ import annotations

import glob
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "scripts" / "233-retire-final-legacy-tables.sql"
STATEFUL_MIGRATION = ROOT / "scripts" / "101-stateful-subnet-epoch-authority.sql"

TARGET_TABLES = (
    "company_information_table",
    "early_access_emails",
    "outreach_email_verifications",
    "research_lab_official_baseline_action_attempts_v1",
    "research_lab_official_baseline_action_terminals_v1",
    "research_lab_official_baseline_runs_v1",
    "research_lab_official_baseline_unit_closures_v1",
    "research_lab_public_loop_card_events",
    "research_lab_public_loop_cards",
    "suppression_ledger",
    "transparency_log",
    "validation_evidence_private",
)

PRESERVED_TABLES = (
    "dashboard_miner_stats",
    "dashboard_precalc",
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
    "merkle_checkpoints",
    "published_weight_bundles",
    "qualification_baselines",
    "qualification_private_icp_sets",
    "research_evaluation_score_bundles",
    "research_lab_attested_execution_receipts",
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
    "research_lab_stateful_subnet_epoch_boundaries_v1",
    "research_lab_stateful_subnet_epoch_candidates_v1",
    "research_lab_stateful_subnet_epoch_cutover_state_v1",
    "research_lab_stateful_subnet_epoch_cutovers_v1",
    "research_lab_stateful_subnet_epoch_snapshots_v1",
)


def _without_comments(sql: str) -> str:
    return "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )


def test_migration_has_an_exact_fail_closed_allowlist() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    executable = _without_comments(sql).upper()
    assert "CASCADE" not in executable
    assert "SOURCE_ADD_ARCHIVE_SELECTION" not in sql
    for table in TARGET_TABLES:
        assert f"DROP TABLE IF EXISTS public.{table}" in sql
    for table in PRESERVED_TABLES:
        assert f"DROP TABLE IF EXISTS public.{table}" not in sql
    assert "<> 12" in sql
    assert "pg_catalog.md5(routine.prosrc)" in sql
    assert "remaining routine references final retirement closure" in sql
    assert "remaining cron job references final retirement closure" in sql
    assert "source_add_history.legacy_audit_rows" in sql


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
    data_dir = Path(tempfile.mkdtemp(prefix="final-legacy-retire-pg-"))
    socket_dir = Path(tempfile.mkdtemp(prefix="final-legacy-retire-socket-"))
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
        import psycopg2

        admin = psycopg2.connect(
            host="127.0.0.1", port=port, user="postgres", dbname="postgres"
        )
        admin.autocommit = True
        with admin.cursor() as cursor:
            cursor.execute("CREATE ROLE anon NOLOGIN")
            cursor.execute("CREATE ROLE authenticated NOLOGIN")
            cursor.execute("CREATE ROLE service_role NOLOGIN")
        admin.close()
        databases = (
            "retire_success",
            "retire_drift",
            "retire_dynamic_dependency",
            "retire_view_dependency",
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


def _old_stateful_fence_sql() -> str:
    sql = STATEFUL_MIGRATION.read_text(encoding="utf-8")
    marker = (
        "CREATE OR REPLACE FUNCTION "
        "public.enforce_research_lab_stateful_epoch_fence_v1()"
    )
    start = sql.index(marker)
    end = sql.index("\n$$;", start) + len("\n$$;")
    return sql[start:end]


def _setup_sql() -> str:
    generic_preserved = "\n".join(
        f"CREATE TABLE public.{name} (id TEXT PRIMARY KEY, value TEXT NOT NULL);"
        f" INSERT INTO public.{name} VALUES ('kept', 'preserve');"
        for name in PRESERVED_TABLES
        if name
        not in {
            "merkle_checkpoints",
            "research_lab_stateful_subnet_epoch_boundaries_v1",
            "research_lab_stateful_subnet_epoch_candidates_v1",
            "research_lab_stateful_subnet_epoch_cutover_state_v1",
            "research_lab_stateful_subnet_epoch_cutovers_v1",
            "research_lab_stateful_subnet_epoch_snapshots_v1",
        }
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

{generic_preserved}

CREATE TABLE public.research_lab_stateful_subnet_epoch_cutover_state_v1 (
    singleton BOOLEAN PRIMARY KEY,
    lifecycle_state TEXT NOT NULL,
    mapping_hash TEXT,
    network_genesis_hash TEXT,
    netuid INTEGER,
    first_settlement_epoch_id INTEGER,
    candidate_snapshot_hash TEXT,
    candidate_receipt_hash TEXT,
    cutover_authority_hash TEXT,
    cutover_receipt_hash TEXT,
    initialization_nonce UUID,
    initialization_payload_hash TEXT,
    last_legacy_finalization_receipt_hash TEXT
);
INSERT INTO public.research_lab_stateful_subnet_epoch_cutover_state_v1
    (singleton, lifecycle_state, mapping_hash, first_settlement_epoch_id)
VALUES (TRUE, 'stateful_active', 'sha256:' || repeat('a', 64), 100);

CREATE TABLE public.research_lab_stateful_subnet_epoch_boundaries_v1 (
    epoch_id INTEGER PRIMARY KEY
);
CREATE TABLE public.research_lab_stateful_subnet_epoch_candidates_v1 (
    proposed_settlement_epoch_id INTEGER,
    network_genesis_hash TEXT,
    netuid INTEGER,
    mapping_hash TEXT,
    snapshot_hash TEXT,
    chain_state_receipt_hash TEXT
);
CREATE TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1 (
    first_settlement_epoch_id INTEGER,
    mapping_hash TEXT,
    cutover_authority_hash TEXT,
    cutover_receipt_hash TEXT
);
CREATE TABLE public.research_lab_stateful_subnet_epoch_snapshots_v1 (
    epoch_id INTEGER PRIMARY KEY
);

CREATE TABLE public.merkle_checkpoints (
    id BIGINT PRIMARY KEY,
    seq_start BIGINT NOT NULL,
    seq_end BIGINT NOT NULL
);
INSERT INTO public.merkle_checkpoints VALUES (1, 1, 1453696);

CREATE TABLE public.company_information_table (
    id BIGSERIAL PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE public.early_access_emails (id TEXT PRIMARY KEY);
CREATE TABLE public.outreach_email_verifications (id TEXT PRIMARY KEY);
CREATE TABLE public.suppression_ledger (id TEXT PRIMARY KEY);
CREATE TABLE public.validation_evidence_private (id TEXT PRIMARY KEY);

CREATE TABLE public.research_lab_official_baseline_runs_v1 (
    run_sha256 TEXT PRIMARY KEY
);
CREATE TABLE public.research_lab_official_baseline_action_attempts_v1 (
    attempt_key TEXT PRIMARY KEY,
    run_sha256 TEXT REFERENCES public.research_lab_official_baseline_runs_v1
);
CREATE TABLE public.research_lab_official_baseline_action_terminals_v1 (
    attempt_key TEXT PRIMARY KEY REFERENCES
        public.research_lab_official_baseline_action_attempts_v1
);
CREATE TABLE public.research_lab_official_baseline_unit_closures_v1 (
    closure_key TEXT PRIMARY KEY,
    run_sha256 TEXT REFERENCES public.research_lab_official_baseline_runs_v1
);

CREATE TABLE public.research_lab_public_loop_cards (card_id TEXT PRIMARY KEY);
CREATE TABLE public.research_lab_public_loop_card_events (
    event_id TEXT PRIMARY KEY,
    card_id TEXT REFERENCES public.research_lab_public_loop_cards
);
CREATE VIEW public.research_lab_public_loop_card_current AS
    SELECT card.card_id, event.event_id
      FROM public.research_lab_public_loop_cards AS card
      LEFT JOIN public.research_lab_public_loop_card_events AS event
        ON event.card_id = card.card_id;

CREATE TABLE public.transparency_log (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    actor_hotkey TEXT NOT NULL,
    nonce UUID NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    payload_hash TEXT NOT NULL,
    build_id TEXT NOT NULL,
    signature TEXT NOT NULL,
    payload JSONB NOT NULL,
    merkle_leaf_hash TEXT,
    merkle_index BIGINT,
    created_at TIMESTAMPTZ DEFAULT now(),
    arweave_tx_id TEXT,
    arweave_confirmed_at TIMESTAMPTZ,
    tee_sequence BIGINT,
    tee_buffered_at TIMESTAMPTZ,
    tee_buffer_size INTEGER,
    email_hash TEXT,
    linkedin_combo_hash TEXT,
    event_hash TEXT UNIQUE,
    enclave_pubkey TEXT,
    boot_id TEXT,
    monotonic_seq BIGINT,
    prev_event_hash TEXT,
    netuid INTEGER,
    epoch_id INTEGER,
    signed_log_entry JSONB
);
INSERT INTO public.transparency_log
    (id, event_type, actor_hotkey, nonce, ts, payload_hash, build_id,
     signature, payload, merkle_leaf_hash, merkle_index, created_at,
     arweave_tx_id, arweave_confirmed_at, tee_sequence, tee_buffered_at,
     tee_buffer_size, email_hash, linkedin_combo_hash, event_hash,
     enclave_pubkey, boot_id, monotonic_seq, prev_event_hash, netuid, epoch_id,
     signed_log_entry)
WITH source_ids AS (
    SELECT id,
           row_number() OVER (ORDER BY id)::BIGINT AS ordinal
    FROM generate_series(41960931::BIGINT, 41982623::BIGINT) AS id
    WHERE id NOT BETWEEN 41961931 AND 41961963
)
SELECT id,
       CASE
           WHEN ordinal <= 324 THEN 'RESEARCH_LAB_EPOCH_AUDIT'
           WHEN ordinal <= 648 THEN 'WEIGHT_SUBMISSION'
           ELSE 'CHAIN_EVENT'
       END,
       'fixture-hotkey',
       '00000000-0000-0000-0000-000000000001'::UUID,
       '2026-07-20 00:00:00+00'::TIMESTAMPTZ
           + ordinal * INTERVAL '1 second',
       'payload-' || ordinal,
       'fixture-build',
       'signature-' || ordinal,
       CASE
           WHEN ordinal <= 324 THEN pg_catalog.jsonb_build_object(
               'purpose', CASE WHEN ordinal <= 152
                               THEN 'SOURCE_ADD'
                               ELSE 'OTHER' END,
               'weights', pg_catalog.jsonb_build_object(
                   'weight_submission_event_hash', 'event-' || (ordinal + 324)
               ),
               'ordinal', ordinal
           )
           ELSE pg_catalog.jsonb_build_object('ordinal', ordinal)
       END,
       NULL,
       ordinal,
       '2026-07-20 00:00:00+00'::TIMESTAMPTZ
           + ordinal * INTERVAL '1 second',
       NULL,
       NULL,
       ordinal,
       NULL,
       NULL,
       NULL,
       NULL,
       'event-' || ordinal,
       'fixture-pubkey',
       'fixture-boot',
       ordinal,
       CASE
           WHEN ordinal IN (1, 100, 200, 300) THEN 'missing-' || ordinal
           ELSE 'event-' || (ordinal - 1)
       END,
       71,
       25000,
       pg_catalog.jsonb_build_object('ordinal', ordinal)
FROM source_ids;

INSERT INTO public.company_information_table (value) VALUES ('retire');
INSERT INTO public.early_access_emails VALUES ('retire');
INSERT INTO public.outreach_email_verifications VALUES ('retire');
INSERT INTO public.suppression_ledger VALUES ('retire');
INSERT INTO public.validation_evidence_private VALUES ('retire');
INSERT INTO public.research_lab_official_baseline_runs_v1 VALUES ('run');
INSERT INTO public.research_lab_official_baseline_action_attempts_v1
VALUES ('attempt', 'run');
INSERT INTO public.research_lab_official_baseline_action_terminals_v1
VALUES ('attempt');
INSERT INTO public.research_lab_official_baseline_unit_closures_v1
VALUES ('closure', 'run');
INSERT INTO public.research_lab_public_loop_cards VALUES ('card');
INSERT INTO public.research_lab_public_loop_card_events VALUES ('event', 'card');
"""


def _connect(dsn):
    import psycopg2

    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    return connection


def _setup(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(_setup_sql())
        cursor.execute(_old_stateful_fence_sql())
        cursor.execute(
            """
            CREATE TRIGGER enforce_research_lab_stateful_epoch_fence_v1
            BEFORE INSERT ON public.research_lab_stateful_subnet_epoch_boundaries_v1
            FOR EACH ROW EXECUTE FUNCTION
                public.enforce_research_lab_stateful_epoch_fence_v1()
            """
        )


def test_migration_archives_source_add_and_preserves_shared_state(
    postgres_databases,
) -> None:
    connection = _connect(postgres_databases["retire_success"])
    try:
        _setup(connection)
        sql = MIGRATION.read_text(encoding="utf-8")
        with connection.cursor() as cursor:
            cursor.execute(sql)
            cursor.execute(sql)
            cursor.execute(
                """
                SELECT
                    (SELECT count(*) FROM source_add_history.legacy_audit_rows),
                    (SELECT count(*) FROM public.lab_arena_ledger),
                    (SELECT count(*) FROM public.dashboard_precalc),
                    (SELECT count(*) FROM public.merkle_checkpoints),
                    md5(proc.prosrc),
                    to_regclass('public.transparency_log') IS NULL,
                    to_regclass('public.research_lab_official_baseline_runs_v1')
                        IS NULL
                FROM pg_proc AS proc
                WHERE proc.oid =
                    'public.enforce_research_lab_stateful_epoch_fence_v1()'
                        ::regprocedure
                """
            )
            assert cursor.fetchone() == (
                21660,
                1,
                1,
                1,
                "5b48baac95474f877b66f84deb78d8f4",
                True,
                True,
            )
            cursor.execute(
                """
                SELECT archived_row_count,
                       historical_row_count,
                       audit_row_count,
                       source_add_marker_row_count,
                       missing_nonzero_parent_count,
                       jsonb_array_length(source_columns),
                       source_row_fingerprint = (
                           SELECT md5(string_agg(
                               md5(row_doc::TEXT), '' ORDER BY source_pk::BIGINT
                           ))
                           FROM source_add_history.legacy_audit_rows
                       )
                FROM source_add_history.archive_manifests
                WHERE source_relation = 'public.transparency_log'
                """
            )
            assert cursor.fetchone() == (21660, 21660, 324, 152, 4, 27, True)
            cursor.execute(
                "INSERT INTO public.research_lab_stateful_subnet_epoch_boundaries_v1 "
                "VALUES (101)"
            )
            cursor.execute(
                "UPDATE public.research_lab_stateful_subnet_epoch_cutover_state_v1 "
                "SET lifecycle_state = 'cutover_fenced' WHERE singleton"
            )
            with pytest.raises(Exception, match="rejects boundary/snapshot writes"):
                cursor.execute(
                    "INSERT INTO public.research_lab_stateful_subnet_epoch_boundaries_v1 "
                    "VALUES (102)"
                )
            with pytest.raises(Exception, match="append-only"):
                cursor.execute(
                    "UPDATE source_add_history.legacy_audit_rows "
                    "SET source_pk = source_pk"
                )
    finally:
        connection.close()


def test_changed_reviewed_routine_rolls_back(postgres_databases) -> None:
    connection = _connect(postgres_databases["retire_drift"])
    try:
        _setup(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                "CREATE FUNCTION public.refresh_dashboard_precalc() "
                "RETURNS void LANGUAGE plpgsql AS $$ BEGIN NULL; END $$"
            )
            with pytest.raises(Exception, match="changed after review"):
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute("ROLLBACK")
            cursor.execute("SELECT to_regclass('public.transparency_log')")
            assert cursor.fetchone()[0] == "transparency_log"
    finally:
        connection.close()


def test_private_dynamic_caller_rolls_back(postgres_databases) -> None:
    connection = _connect(postgres_databases["retire_dynamic_dependency"])
    try:
        _setup(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE SCHEMA private;
                CREATE FUNCTION private.unreviewed_transparency_reader()
                RETURNS BIGINT LANGUAGE plpgsql AS $$
                DECLARE row_count BIGINT;
                BEGIN
                    EXECUTE 'SELECT count(*) FROM public.transparency_log'
                       INTO row_count;
                    RETURN row_count;
                END;
                $$;
                """
            )
            with pytest.raises(Exception, match="unexpected routine depends"):
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute("ROLLBACK")
            cursor.execute("SELECT to_regclass('public.transparency_log')")
            assert cursor.fetchone()[0] == "transparency_log"
    finally:
        connection.close()


def test_unknown_view_dependency_rolls_back(postgres_databases) -> None:
    connection = _connect(postgres_databases["retire_view_dependency"])
    try:
        _setup(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                "CREATE VIEW public.unreviewed_transparency_view AS "
                "SELECT id FROM public.transparency_log"
            )
            with pytest.raises(Exception, match="unexpected view depends"):
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute("ROLLBACK")
            cursor.execute("SELECT to_regclass('public.transparency_log')")
            assert cursor.fetchone()[0] == "transparency_log"
    finally:
        connection.close()
