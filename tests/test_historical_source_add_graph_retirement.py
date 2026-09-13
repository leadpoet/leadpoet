"""Focused static and disposable-PostgreSQL checks for migration 242."""

from __future__ import annotations

import glob
import re
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "scripts" / "242-retire-historical-source-add-graph.sql"
TABLES = (
    "published_weight_bundles",
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
    "research_lab_stateful_subnet_epoch_snapshots_v1",
)
ROUTINES = (
    ("persist_research_lab_ancestry_checkpoint_v2", "checkpoint jsonb"),
    ("research_lab_active_model_replay_contract_v2", ""),
    ("research_lab_ancestry_disclosure_lookup_contract_v1", ""),
    ("research_lab_ancestry_checkpoint_bootstrap_contract_v2", ""),
    ("research_lab_attested_transport_purpose_contract_v2", ""),
    ("research_lab_attested_transport_terminal_contract_v2", ""),
    ("research_lab_candidate_hybrid_purpose_contract_v1", ""),
    ("research_lab_compact_checkpoint_graph_contract_v1", ""),
    ("validate_research_lab_compact_checkpoint_sidecars_v1", ""),
    ("validate_research_lab_stateful_subnet_epoch_v1", ""),
)


def _without_comments(sql: str) -> str:
    return "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )


def test_exact_scope_archive_gate_and_no_cascade() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    executable = _without_comments(sql).upper()
    assert len(TABLES) == 15
    assert "CASCADE" not in executable
    assert re.search(r"archive_uri CONSTANT TEXT := '[^']+';", sql)
    assert re.search(r"archive_sha256 CONSTANT TEXT := '[^']+';", sql)
    assert "verified historical archive URI and SHA-256 are required" in sql
    assert "SET LOCAL lock_timeout = '5s'" in sql
    assert "SET LOCAL statement_timeout = '120s'" in sql
    for table in TABLES:
        assert sql.count(f"DROP TABLE IF EXISTS public.{table} RESTRICT;") == 1
    assert "DROP TABLE IF EXISTS public.research_lab_provider_evidence_cache_v2" not in sql
    assert "DROP TABLE IF EXISTS public.research_lab_stateful_subnet_epoch_cutovers_v1" not in sql
    assert "DROP TABLE IF EXISTS public.research_lab_stateful_subnet_epoch_cutover_state_v1" not in sql
    assert sql.count("DROP CONSTRAINT IF EXISTS research_lab_stateful_subnet_") == 5
    assert "historical cutover registration is retired; existing mappings are read-only" in sql


def _find_pg_bindir() -> str | None:
    candidates: list[str] = [
        "/opt/homebrew/opt/postgresql@15/bin",
        "/usr/local/opt/postgresql@15/bin",
        "/usr/lib/postgresql/15/bin",
    ]
    initdb = shutil.which("initdb")
    if initdb:
        candidates.append(str(Path(initdb).parent))
    candidates += sorted(glob.glob("/opt/homebrew/opt/postgresql@*/bin"), reverse=True)
    candidates += sorted(glob.glob("/usr/local/opt/postgresql@*/bin"), reverse=True)
    candidates += sorted(glob.glob("/usr/lib/postgresql/*/bin"), reverse=True)
    for candidate in candidates:
        if all((Path(candidate) / name).is_file() for name in ("initdb", "pg_ctl", "createdb")):
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
    data_dir = Path(tempfile.mkdtemp(prefix="history-retire-pg-"))
    socket_dir = Path(tempfile.mkdtemp(prefix="history-retire-socket-"))
    port = _free_port()
    started = False
    try:
        subprocess.run(
            [str(bindir / "initdb"), "-D", str(data_dir), "-U", "postgres", "--auth=trust"],
            check=True, capture_output=True, text=True, timeout=90,
        )
        subprocess.run(
            [str(bindir / "pg_ctl"), "-D", str(data_dir), "-w", "-t", "30",
             "-l", str(data_dir / "server.log"), "-o",
             f"-p {port} -c listen_addresses=127.0.0.1 -c unix_socket_directories={socket_dir}",
             "start"],
            check=True, capture_output=True, text=True, timeout=45,
        )
        started = True
        names = ("success", "view_dependency", "routine_dependency", "archive_oid_drift", "archive_write_drift", "parsed_dependency", "transitive_dependency")
        for name in names:
            subprocess.run(
                [str(bindir / "createdb"), "-h", "127.0.0.1", "-p", str(port),
                 "-U", "postgres", name],
                check=True, capture_output=True, text=True, timeout=30,
            )
        yield {name: {"host": "127.0.0.1", "port": port, "user": "postgres", "dbname": name} for name in names}
    finally:
        if started:
            subprocess.run(
                [str(bindir / "pg_ctl"), "-D", str(data_dir), "-w", "-t", "20", "stop"],
                check=False, capture_output=True, text=True, timeout=30,
            )
        shutil.rmtree(data_dir, ignore_errors=True)
        shutil.rmtree(socket_dir, ignore_errors=True)


def _connect(dsn):
    import psycopg2

    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    return connection


def _setup(connection) -> None:
    simple = "\n".join(
        f"CREATE TABLE public.{name} (id TEXT PRIMARY KEY); INSERT INTO public.{name} VALUES ('old');"
        for name in TABLES
        if name not in {
            "research_lab_attested_execution_receipts_v2",
            "research_lab_stateful_subnet_epoch_candidates_v1",
        }
    )
    obsolete = "\n".join(
        (
            f"CREATE FUNCTION public.{name}(checkpoint JSONB) RETURNS void LANGUAGE plpgsql AS "
            "$$ BEGIN PERFORM 1; END $$;"
            if args
            else f"CREATE FUNCTION public.{name}() RETURNS JSONB LANGUAGE plpgsql AS "
                 "$$ BEGIN RETURN '{}'::jsonb; END $$;"
        )
        for name, args in ROUTINES
    )
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            {simple}
            CREATE TABLE public.research_lab_attested_execution_receipts_v2 (
                receipt_hash TEXT PRIMARY KEY
            );
            INSERT INTO public.research_lab_attested_execution_receipts_v2 VALUES ('receipt');
            CREATE TABLE public.research_lab_stateful_subnet_epoch_candidates_v1 (
                snapshot_hash TEXT PRIMARY KEY
            );
            INSERT INTO public.research_lab_stateful_subnet_epoch_candidates_v1 VALUES ('snapshot');
            CREATE TABLE public.research_lab_stateful_subnet_epoch_cutover_state_v1 (
                singleton BOOLEAN PRIMARY KEY,
                lifecycle_state TEXT NOT NULL,
                mapping_hash TEXT,
                cutover_authority_hash TEXT,
                cutover_receipt_hash TEXT,
                initialization_nonce UUID,
                initialization_payload_hash TEXT,
                first_settlement_epoch_id INTEGER
            );
            INSERT INTO public.research_lab_stateful_subnet_epoch_cutover_state_v1 VALUES
                (true, 'stateful_active', 'mapping', 'authority', 'receipt',
                 '00000000-0000-0000-0000-000000000001', 'payload', 1);
            CREATE TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1 (
                cutover_authority_hash TEXT PRIMARY KEY,
                mapping_hash TEXT NOT NULL,
                cutover_receipt_hash TEXT NOT NULL,
                first_settlement_epoch_id INTEGER NOT NULL,
                first_epoch_ref TEXT NOT NULL,
                first_snapshot_hash TEXT NOT NULL,
                first_snapshot_receipt_hash TEXT NOT NULL,
                last_legacy_finalization_receipt_hash TEXT,
                predecessor_receipt_hash TEXT
            );
            INSERT INTO public.research_lab_stateful_subnet_epoch_cutovers_v1 VALUES
                ('authority', 'mapping', 'receipt', 1, 'epoch', 'snapshot',
                 'receipt', 'receipt', 'receipt');
            ALTER TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1
                ADD CONSTRAINT research_lab_stateful_subnet__last_legacy_finalization_rec_fkey
                    FOREIGN KEY (last_legacy_finalization_receipt_hash)
                    REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash) ON DELETE RESTRICT,
                ADD CONSTRAINT research_lab_stateful_subnet_e_first_snapshot_receipt_hash_fkey
                    FOREIGN KEY (first_snapshot_receipt_hash)
                    REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash) ON DELETE RESTRICT,
                ADD CONSTRAINT research_lab_stateful_subnet_epoc_predecessor_receipt_hash_fkey
                    FOREIGN KEY (predecessor_receipt_hash)
                    REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash) ON DELETE RESTRICT,
                ADD CONSTRAINT research_lab_stateful_subnet_epoch_cu_cutover_receipt_hash_fkey
                    FOREIGN KEY (cutover_receipt_hash)
                    REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash) ON DELETE RESTRICT,
                ADD CONSTRAINT research_lab_stateful_subnet_epoch_cut_first_snapshot_hash_fkey
                    FOREIGN KEY (first_snapshot_hash)
                    REFERENCES public.research_lab_stateful_subnet_epoch_candidates_v1(snapshot_hash) ON DELETE RESTRICT;
            CREATE TABLE public.research_lab_provider_evidence_cache_v2 (id TEXT PRIMARY KEY);
            INSERT INTO public.research_lab_provider_evidence_cache_v2 VALUES ('cache');
            CREATE SCHEMA storage;
            CREATE TABLE storage.published_weight_bundles (id TEXT PRIMARY KEY);
            INSERT INTO storage.published_weight_bundles VALUES ('keep');

            CREATE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1()
            RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER SET search_path TO ''
            AS $$ BEGIN RETURN NEW; END $$;
            CREATE FUNCTION public.validate_research_lab_stateful_epoch_cutover_v2()
            RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER SET search_path TO ''
            AS $$ BEGIN RETURN NEW; END $$;
            CREATE FUNCTION public.prevent_research_lab_attested_v2_mutation()
            RETURNS TRIGGER LANGUAGE plpgsql AS $$ BEGIN RETURN NEW; END $$;
            CREATE FUNCTION public.put_research_lab_provider_evidence_cache_v2(cache_row JSONB)
            RETURNS JSONB LANGUAGE sql AS $$ SELECT cache_row $$;
            CREATE FUNCTION public.research_lab_stateful_subnet_epoch_cutover_public_state_v1()
            RETURNS JSONB LANGUAGE sql AS $$ SELECT '{{}}'::jsonb $$;
            CREATE TRIGGER enforce_research_lab_stateful_epoch_fence_v1
                BEFORE INSERT ON public.research_lab_stateful_subnet_epoch_cutovers_v1
                FOR EACH ROW EXECUTE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1();
            CREATE TRIGGER validate_research_lab_stateful_epoch_cutover_v1
                BEFORE INSERT ON public.research_lab_stateful_subnet_epoch_cutovers_v1
                FOR EACH ROW EXECUTE FUNCTION public.validate_research_lab_stateful_epoch_cutover_v2();
            CREATE TRIGGER preserve_provider_cache_mutation
                BEFORE UPDATE ON public.research_lab_provider_evidence_cache_v2
                FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_attested_v2_mutation();
            {obsolete}
            """
        )


def _migration_for_fixture(connection) -> str:
    sql = MIGRATION.read_text(encoding="utf-8")
    sql = re.sub(
        r"archive_uri CONSTANT TEXT := '[^']+';",
        "archive_uri CONSTANT TEXT := 's3://private/archive/history.dump';",
        sql,
        count=1,
    )
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_stat_force_next_flush()")
        cursor.execute(
            "SELECT relname, relid, n_tup_ins, n_tup_upd, n_tup_del "
            "FROM pg_stat_user_tables WHERE schemaname='public' AND relname=ANY(%s) "
            "ORDER BY relname",
            (list(TABLES),),
        )
        stats = cursor.fetchall()
    stats_sql = "INSERT INTO _retire_242_source_stats VALUES\n" + ",\n".join(
        f"    ('{name}', {oid}, {inserted}, {updated}, {deleted})"
        for name, oid, inserted, updated, deleted in stats
    ) + ";"
    sql = re.sub(
        r"-- TEST_FIXTURE_STATS_BEGIN\n.*?\n-- TEST_FIXTURE_STATS_END",
        "-- TEST_FIXTURE_STATS_BEGIN\n" + stats_sql + "\n-- TEST_FIXTURE_STATS_END",
        sql,
        count=1,
        flags=re.S,
    )
    sql = re.sub(
        r"archive_sha256 CONSTANT TEXT := '[^']+';",
        f"archive_sha256 CONSTANT TEXT := '{'a' * 64}';",
        sql,
        count=1,
    )
    names = [name for name, _ in ROUTINES] + [
        "enforce_research_lab_stateful_epoch_fence_v1",
        "validate_research_lab_stateful_epoch_cutover_v2",
    ]
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT proname, md5(prosrc) FROM pg_proc p JOIN pg_namespace n "
            "ON n.oid=p.pronamespace WHERE n.nspname='public' AND proname=ANY(%s)",
            (names,),
        )
        actual = dict(cursor.fetchall())
    production_hashes = {
        "persist_research_lab_ancestry_checkpoint_v2": "266871ca5d1f1ce4e0ae22e7da76a3e6",
        "research_lab_active_model_replay_contract_v2": "b9950de24ed91e7d3718570c6554c220",
        "research_lab_ancestry_disclosure_lookup_contract_v1": "881ae11e2e3e02659d820b07ac80bd2d",
        "research_lab_ancestry_checkpoint_bootstrap_contract_v2": "67b952a7d04124501d68554ca4392373",
        "research_lab_attested_transport_purpose_contract_v2": "b940563a3e0f9b15df459f6ab3b47a4a",
        "research_lab_attested_transport_terminal_contract_v2": "1b5f105d75a5c21c726ee70a6bfef31e",
        "research_lab_candidate_hybrid_purpose_contract_v1": "246c0992c7a64f715ec4c5875bffa2d1",
        "research_lab_compact_checkpoint_graph_contract_v1": "c982d13b2eddf870fd19bbde9b2b0264",
        "validate_research_lab_compact_checkpoint_sidecars_v1": "0f42e80c1c38aaf72b1f70b0923d4047",
        "validate_research_lab_stateful_subnet_epoch_v1": "7259893de35bbc17e0483af31a3ca654",
        "enforce_research_lab_stateful_epoch_fence_v1": "5b97ae04b866110b43b6cf8ec159463b",
        "validate_research_lab_stateful_epoch_cutover_v2": "bc22df15eda65827f8aade10a2e0f5b4",
    }
    for name, expected in production_hashes.items():
        sql = sql.replace(expected, actual[name])
    return sql


def test_migration_is_idempotent_preserves_retained_objects_and_rejects_insert(postgres_databases) -> None:
    connection = _connect(postgres_databases["success"])
    try:
        _setup(connection)
        migration = _migration_for_fixture(connection)
        with connection.cursor() as cursor:
            cursor.execute(migration)
            cursor.execute(migration)
            cursor.execute(
                "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname='public' AND c.relname=ANY(%s)",
                (list(TABLES),),
            )
            assert cursor.fetchone()[0] == 0
            cursor.execute(
                "SELECT (SELECT count(*) FROM public.research_lab_stateful_subnet_epoch_cutovers_v1), "
                "(SELECT count(*) FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1), "
                "(SELECT count(*) FROM public.research_lab_provider_evidence_cache_v2), "
                "(SELECT count(*) FROM storage.published_weight_bundles), "
                "public.research_lab_stateful_subnet_epoch_cutover_public_state_v1()"
            )
            assert cursor.fetchone() == (1, 1, 1, 1, {})
            with pytest.raises(Exception, match="historical cutover registration is retired"):
                cursor.execute(
                    "INSERT INTO public.research_lab_stateful_subnet_epoch_cutovers_v1 VALUES "
                    "('authority', 'mapping', 'receipt', 1, 'epoch', 'snapshot', "
                    "'receipt', 'receipt', 'receipt')"
                )
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("database", "dependency_sql", "message"),
    (
        (
            "view_dependency",
            "CREATE VIEW storage.unexpected_history AS SELECT id FROM public.published_weight_bundles",
            "unexpected view depends on historical set",
        ),
        (
            "routine_dependency",
            "CREATE FUNCTION storage.unexpected_history() RETURNS TEXT LANGUAGE plpgsql AS "
            "$$ BEGIN PERFORM id FROM public.published_weight_bundles; RETURN 'x'; END $$",
            "unexpected routine text references historical set",
        ),
        (
            "parsed_dependency",
            "CREATE FUNCTION storage.unexpected_history() RETURNS TEXT LANGUAGE sql "
            "BEGIN ATOMIC SELECT id FROM public.published_weight_bundles LIMIT 1; END",
            "unexpected parsed routine",
        ),
        (
            "transitive_dependency",
            "CREATE FUNCTION storage.unexpected_history() RETURNS void LANGUAGE plpgsql AS "
            "$$ BEGIN PERFORM public.research_lab_ancestry_disclosure_lookup_contract_v1(); END $$",
            "unexpected.*routine",
        ),
    ),
)
def test_unexpected_nonpublic_dependency_fails_closed(
    postgres_databases, database, dependency_sql, message
) -> None:
    connection = _connect(postgres_databases[database])
    try:
        _setup(connection)
        migration = _migration_for_fixture(connection)
        with connection.cursor() as cursor:
            cursor.execute(dependency_sql)
            with pytest.raises(Exception, match=message):
                cursor.execute(migration)
            cursor.execute("ROLLBACK")
            cursor.execute("SELECT count(*) FROM public.published_weight_bundles")
            assert cursor.fetchone()[0] == 1
    finally:
        connection.close()


@pytest.mark.parametrize("database,field", (("archive_oid_drift", "expected_oid"), ("archive_write_drift", "n_tup_ins")))
def test_archive_identity_or_write_counter_drift_aborts_all_deletions(postgres_databases, database, field) -> None:
    connection = _connect(postgres_databases[database])
    try:
        _setup(connection)
        migration = _migration_for_fixture(connection)
        drift_sql = f"UPDATE _retire_242_source_stats SET {field} = {field}::bigint + 1 WHERE table_name = 'published_weight_bundles';\n"
        migration = migration.replace("DO $source_snapshot_guard$", drift_sql + "DO $source_snapshot_guard$", 1)
        with connection.cursor() as cursor:
            with pytest.raises(Exception, match="historical source changed after archive baseline"):
                cursor.execute(migration)
            cursor.execute("ROLLBACK")
            cursor.execute("SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public' AND c.relname=ANY(%s)", (list(TABLES),))
            assert cursor.fetchone()[0] == 15
            cursor.execute("SELECT count(*) FROM public.published_weight_bundles")
            assert cursor.fetchone()[0] == 1
    finally:
        connection.close()
