"""Static and real-PostgreSQL proof for Fulfillment schema retirement."""

from __future__ import annotations

import glob
import os
import re
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "scripts" / "226-retire-legacy-fulfillment-schema.sql"

FULFILLMENT_RELATIONS = (
    "fulfillment_score_consensus",
    "fulfillment_scores",
    "fulfillment_submissions",
    "fulfillment_requests",
    "role_translations",
)

FULFILLMENT_ROUTINES = (
    "fulfillment_accept_commit(uuid,text,jsonb)",
    "fulfillment_close_window(uuid,text)",
    "fulfillment_release_lifecycle_lock(bigint)",
    "fulfillment_try_lifecycle_lock(bigint)",
    "fulfillment_upsert_consensus(jsonb)",
    "fulfillment_upsert_scores(jsonb,text)",
    "fulfillment_claim_finalization(uuid,bigint,timestamp with time zone)",
    "get_chain_held_count(uuid)",
    "get_chain_root_num_leads(uuid)",
    "get_chain_summaries(uuid[])",
    "get_chain_winners(uuid)",
    "get_fulfillment_graph_summary(uuid[])",
    "get_fulfillment_rejection_stats()",
    "get_rejection_reason_histogram(uuid[])",
)

HISTORICAL_FULFILLMENT_SQL = (
    "scripts/09-fulfillment-internal-label-column.sql",
    "scripts/10-fulfillment-company-column.sql",
    "scripts/11-fulfillment-expired-status.sql",
    "scripts/12-fulfillment-pending-status.sql",
    "scripts/13-fulfillment-chain-held-column.sql",
    "scripts/14-fulfillment-continued-open-and-partially-fulfilled.sql",
    "scripts/15-backfill-chain-company-and-label.sql",
    "scripts/16-fulfillment-intent-breakdown-column.sql",
    "scripts/17-role-translations-cache.sql",
    "scripts/18-fulfillment-deep-research-columns.sql",
    "scripts/20-fulfillment-attribute-verification-columns.sql",
    "scripts/21-backfill-required-attributes.sql",
    "scripts/22-backfill-required-attributes-label-based.sql",
    "scripts/23-split-region-migration.sql",
    "scripts/sql/bruce_callahan_5_icp_fix.sql",
    "sql/add_chain_successor_index.sql",
)


def _without_comments(sql: str) -> str:
    return "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )


def test_retirement_sql_is_exact_narrow_and_fail_closed() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    executable = _without_comments(sql).upper()

    assert "CASCADE" not in executable
    assert "PRONAME LIKE" not in executable
    for relation in FULFILLMENT_RELATIONS:
        assert f"DROP TABLE IF EXISTS public.{relation}" in sql
    for routine in FULFILLMENT_ROUTINES:
        compact_sql = re.sub(r"\s+", "", sql.lower())
        drop_signature = routine.replace("timestamp with time zone", "timestamptz")
        assert (
            f"dropfunctionifexistspublic.{drop_signature}".replace(" ", "")
            in compact_sql
        )

    for shared in (
        "lab_arena_rounds",
        "lab_arena_submissions",
        "lab_arena_runs",
        "lab_arena_ledger",
        "lab_arena_accepted_weight_states",
        "miner_test_leads",
        "refresh_miner_test_leads",
    ):
        assert f"DROP TABLE IF EXISTS public.{shared}" not in sql
        assert f"DROP FUNCTION IF EXISTS public.{shared}" not in sql


def test_obsolete_fulfillment_sql_paths_are_removed() -> None:
    assert not [path for path in HISTORICAL_FULFILLMENT_SQL if (ROOT / path).exists()]
    assert list((ROOT / "scripts").glob("[0-9]*-fulfillment-*.sql")) == [MIGRATION]


def _find_pg_bindir() -> str | None:
    candidates: list[str] = []
    initdb = shutil.which("initdb")
    if initdb:
        candidates.append(str(Path(initdb).parent))
    candidates += sorted(glob.glob("/opt/homebrew/opt/postgresql@*/bin"), reverse=True)
    candidates += sorted(glob.glob("/usr/local/opt/postgresql@*/bin"), reverse=True)
    candidates += sorted(glob.glob("/usr/lib/postgresql/*/bin"), reverse=True)
    for candidate in candidates:
        if all((Path(candidate) / name).is_file() for name in ("initdb", "pg_ctl", "postgres", "psql")):
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
    data_dir = Path(tempfile.mkdtemp(prefix="fulfillment-retire-pg-"))
    socket_dir = Path(tempfile.mkdtemp(prefix="fulfillment-retire-socket-"))
    port = _free_port()
    env = {**os.environ, "PGTZ": "UTC", "TZ": "UTC"}
    started = False

    def binary(name: str) -> str:
        return str(bindir / name)

    try:
        subprocess.run(
            [
                binary("initdb"),
                "-D",
                str(data_dir),
                "-U",
                "postgres",
                "--auth=trust",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=90,
        )
        subprocess.run(
            [
                binary("pg_ctl"),
                "-D",
                str(data_dir),
                "-w",
                "-t",
                "30",
                "-l",
                str(data_dir / "server.log"),
                "-o",
                "-p %d -c listen_addresses=127.0.0.1 -c unix_socket_directories=%s"
                % (port, socket_dir),
                "start",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=45,
        )
        started = True
        for database in ("retire_success", "retire_guard"):
            subprocess.run(
                [
                    binary("createdb"),
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
                env=env,
                timeout=30,
            )
        yield {
            name: {
                "host": "127.0.0.1",
                "port": port,
                "user": "postgres",
                "dbname": name,
            }
            for name in ("retire_success", "retire_guard")
        }
    finally:
        if started:
            subprocess.run(
                [
                    binary("pg_ctl"),
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
                env=env,
                timeout=30,
            )
        shutil.rmtree(data_dir, ignore_errors=True)
        shutil.rmtree(socket_dir, ignore_errors=True)


FIXTURE_SQL = r"""
CREATE TABLE public.lab_arena_rounds (id TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE public.lab_arena_submissions (id TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE public.lab_arena_runs (id TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE public.lab_arena_ledger (id TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE public.lab_arena_accepted_weight_states (id TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE public.miner_test_leads (id INTEGER PRIMARY KEY, business TEXT NOT NULL);
CREATE TABLE public.qualification_private_icp_sets (id TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE public.research_lab_provider_usage_ledger (id TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE public.transparency_log (id TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE public.banned_hotkeys (hotkey TEXT PRIMARY KEY);
CREATE SCHEMA cron;
CREATE TABLE cron.job (
    jobid INTEGER PRIMARY KEY,
    jobname TEXT NOT NULL,
    command TEXT NOT NULL
);

INSERT INTO public.lab_arena_rounds VALUES ('round', 'preserve');
INSERT INTO public.lab_arena_submissions VALUES ('submission', 'preserve');
INSERT INTO public.lab_arena_runs VALUES ('run', 'preserve');
INSERT INTO public.lab_arena_ledger VALUES ('provider-call', 'preserve');
INSERT INTO public.lab_arena_accepted_weight_states VALUES ('weight', 'preserve');
INSERT INTO public.miner_test_leads VALUES (1, 'Qualification Lead');
INSERT INTO public.qualification_private_icp_sets VALUES ('icp', 'preserve');
INSERT INTO public.research_lab_provider_usage_ledger VALUES ('provider', 'preserve');
INSERT INTO public.transparency_log VALUES ('event', 'preserve');
INSERT INTO public.banned_hotkeys VALUES ('shared-hotkey');
INSERT INTO cron.job VALUES
    (46, 'refresh-miner-test-leads', 'SELECT public.refresh_miner_test_leads();'),
    (47, 'reset-miner-rate-limits-daily', 'SELECT 1;');

CREATE TABLE public.fulfillment_requests (
    request_id UUID PRIMARY KEY,
    successor_request_id UUID REFERENCES public.fulfillment_requests(request_id),
    payload JSONB NOT NULL DEFAULT '{}'::JSONB
);
CREATE TABLE public.fulfillment_submissions (
    submission_id UUID PRIMARY KEY,
    request_id UUID NOT NULL REFERENCES public.fulfillment_requests(request_id),
    payload JSONB NOT NULL DEFAULT '{}'::JSONB
);
CREATE TABLE public.fulfillment_scores (
    score_id UUID PRIMARY KEY,
    request_id UUID NOT NULL REFERENCES public.fulfillment_requests(request_id),
    submission_id UUID NOT NULL REFERENCES public.fulfillment_submissions(submission_id),
    payload JSONB NOT NULL DEFAULT '{}'::JSONB
);
CREATE TABLE public.fulfillment_score_consensus (
    consensus_id UUID PRIMARY KEY,
    request_id UUID NOT NULL REFERENCES public.fulfillment_requests(request_id),
    submission_id UUID NOT NULL REFERENCES public.fulfillment_submissions(submission_id),
    payload JSONB NOT NULL DEFAULT '{}'::JSONB
);
CREATE TABLE public.role_translations (
    role_original TEXT PRIMARY KEY,
    translated_en TEXT NOT NULL
);

INSERT INTO public.fulfillment_requests(request_id)
VALUES ('10000000-0000-4000-8000-000000000001');
INSERT INTO public.fulfillment_submissions(submission_id, request_id)
VALUES (
    '20000000-0000-4000-8000-000000000001',
    '10000000-0000-4000-8000-000000000001'
);
INSERT INTO public.fulfillment_scores(score_id, request_id, submission_id)
VALUES (
    '30000000-0000-4000-8000-000000000001',
    '10000000-0000-4000-8000-000000000001',
    '20000000-0000-4000-8000-000000000001'
);
INSERT INTO public.fulfillment_score_consensus(
    consensus_id, request_id, submission_id
) VALUES (
    '40000000-0000-4000-8000-000000000001',
    '10000000-0000-4000-8000-000000000001',
    '20000000-0000-4000-8000-000000000001'
);
INSERT INTO public.role_translations VALUES ('Directeur', 'Director');

CREATE FUNCTION public.shared_touch_row()
RETURNS TRIGGER LANGUAGE plpgsql AS $$ BEGIN RETURN NEW; END $$;
CREATE TRIGGER fulfillment_requests_shared_touch
BEFORE UPDATE ON public.fulfillment_requests
FOR EACH ROW EXECUTE FUNCTION public.shared_touch_row();

CREATE FUNCTION public.fulfillment_accept_commit(UUID, TEXT, JSONB)
RETURNS UUID LANGUAGE SQL AS $$ SELECT $1 $$;
CREATE FUNCTION public.fulfillment_close_window(UUID, TEXT)
RETURNS VOID LANGUAGE plpgsql AS $$ BEGIN RETURN; END $$;
CREATE FUNCTION public.fulfillment_release_lifecycle_lock(BIGINT)
RETURNS VOID LANGUAGE plpgsql AS $$ BEGIN RETURN; END $$;
CREATE FUNCTION public.fulfillment_try_lifecycle_lock(BIGINT)
RETURNS BOOLEAN LANGUAGE SQL AS $$ SELECT TRUE $$;
CREATE FUNCTION public.fulfillment_upsert_consensus(JSONB)
RETURNS VOID LANGUAGE plpgsql AS $$ BEGIN RETURN; END $$;
CREATE FUNCTION public.fulfillment_upsert_scores(JSONB, TEXT)
RETURNS VOID LANGUAGE plpgsql AS $$ BEGIN RETURN; END $$;
CREATE FUNCTION public.fulfillment_claim_finalization(UUID, BIGINT, TIMESTAMPTZ)
RETURNS BOOLEAN LANGUAGE SQL AS $$ SELECT TRUE $$;
CREATE FUNCTION public.get_chain_held_count(UUID)
RETURNS INTEGER LANGUAGE SQL AS $$ SELECT 1 $$;
CREATE FUNCTION public.get_chain_root_num_leads(UUID)
RETURNS INTEGER LANGUAGE SQL AS $$ SELECT 1 $$;
CREATE FUNCTION public.get_chain_summaries(UUID[])
RETURNS TABLE(request_id UUID) LANGUAGE SQL AS $$ SELECT unnest($1) $$;
CREATE FUNCTION public.get_chain_winners(UUID)
RETURNS TABLE(request_id UUID) LANGUAGE SQL AS $$ SELECT $1 $$;
CREATE FUNCTION public.get_fulfillment_graph_summary(UUID[])
RETURNS TABLE(request_id UUID) LANGUAGE SQL AS $$ SELECT unnest($1) $$;
CREATE FUNCTION public.get_fulfillment_rejection_stats()
RETURNS JSONB LANGUAGE SQL AS $$ SELECT '{}'::JSONB $$;
CREATE FUNCTION public.get_rejection_reason_histogram(UUID[])
RETURNS TABLE(reason TEXT, count BIGINT) LANGUAGE SQL
AS $$ SELECT 'test'::TEXT, 1::BIGINT $$;

CREATE FUNCTION public.refresh_miner_test_leads()
RETURNS VOID LANGUAGE plpgsql AS $$ BEGIN RETURN; END $$;
CREATE FUNCTION public.shared_metric_projection()
RETURNS JSONB LANGUAGE SQL
AS $$ SELECT jsonb_build_object('fulfillment_rate', 1.0) $$;
"""


def _bootstrap(connection) -> None:
    connection.autocommit = True
    with connection.cursor() as cursor:
        cursor.execute(FIXTURE_SQL)


def test_upgrade_second_apply_and_shared_data_survival(postgres_databases) -> None:
    import psycopg2

    connection = psycopg2.connect(**postgres_databases["retire_success"])
    try:
        _bootstrap(connection)
        migration = MIGRATION.read_text(encoding="utf-8")
        with connection.cursor() as cursor:
            cursor.execute(migration)
            cursor.execute(migration)

            for relation in FULFILLMENT_RELATIONS:
                cursor.execute("SELECT to_regclass(%s)", ("public." + relation,))
                assert cursor.fetchone() == (None,)
            for routine in FULFILLMENT_ROUTINES:
                cursor.execute("SELECT to_regprocedure(%s)", ("public." + routine,))
                assert cursor.fetchone() == (None,)

            cursor.execute(
                """
                SELECT
                    (SELECT value FROM public.lab_arena_rounds WHERE id = 'round'),
                    (SELECT value FROM public.lab_arena_submissions WHERE id = 'submission'),
                    (SELECT value FROM public.lab_arena_runs WHERE id = 'run'),
                    (SELECT value FROM public.lab_arena_ledger WHERE id = 'provider-call'),
                    (SELECT value FROM public.lab_arena_accepted_weight_states WHERE id = 'weight'),
                    (SELECT business FROM public.miner_test_leads WHERE id = 1),
                    (SELECT value FROM public.qualification_private_icp_sets WHERE id = 'icp'),
                    (SELECT value FROM public.research_lab_provider_usage_ledger WHERE id = 'provider'),
                    (SELECT value FROM public.transparency_log WHERE id = 'event'),
                    (SELECT hotkey FROM public.banned_hotkeys WHERE hotkey = 'shared-hotkey'),
                    to_regprocedure('public.refresh_miner_test_leads()') IS NOT NULL,
                    to_regprocedure('public.shared_touch_row()') IS NOT NULL,
                    to_regprocedure('public.shared_metric_projection()') IS NOT NULL,
                    (SELECT count(*) FROM cron.job)
                """
            )
            assert cursor.fetchone() == (
                "preserve",
                "preserve",
                "preserve",
                "preserve",
                "preserve",
                "Qualification Lead",
                "preserve",
                "preserve",
                "preserve",
                "shared-hotkey",
                True,
                True,
                True,
                2,
            )
    finally:
        connection.close()


def test_unknown_shared_dependency_rolls_back_every_drop(postgres_databases) -> None:
    import psycopg2

    connection = psycopg2.connect(**postgres_databases["retire_guard"])
    try:
        _bootstrap(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE public.shared_fulfillment_reference (
                    request_id UUID PRIMARY KEY REFERENCES public.fulfillment_requests(request_id)
                );
                INSERT INTO public.shared_fulfillment_reference
                VALUES ('10000000-0000-4000-8000-000000000001');
                """
            )
            with pytest.raises(psycopg2.errors.DependentObjectsStillExist):
                cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            cursor.execute("ROLLBACK")

            cursor.execute(
                """
                SELECT
                    to_regclass('public.fulfillment_requests') IS NOT NULL,
                    to_regclass('public.fulfillment_scores') IS NOT NULL,
                    to_regprocedure(
                        'public.fulfillment_upsert_scores(jsonb,text)'
                    ) IS NOT NULL,
                    to_regclass(
                        'public.shared_fulfillment_reference'
                    ) IS NOT NULL,
                    (SELECT count(*) FROM public.lab_arena_rounds)
                """
            )
            assert cursor.fetchone() == (True, True, True, True, 1)
    finally:
        connection.close()
