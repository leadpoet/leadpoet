"""Safety and PostgreSQL tests for the fixed failure-funnel migration runner."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import socket
import subprocess
import time
from uuid import uuid4

import pytest

from scripts import apply_failure_funnel_migrations as runner


DOCKER = shutil.which("docker")
MIGRATION_150_PATH = Path(runner.MIGRATION_150)
MIGRATION_151_PATH = Path(runner.MIGRATION_151)


def _migration(path: Path) -> runner.VerifiedMigration:
    content = path.read_bytes()
    return runner.VerifiedMigration(
        path=path.as_posix(),
        sha256=hashlib.sha256(content).hexdigest(),
        sql=content.decode("utf-8"),
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_source_verification_binds_tracked_origin_commit_paths_and_hashes(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Migration Test")
    scripts = repo / "scripts"
    scripts.mkdir()
    (repo / "AGENTS.md").write_text("same\n", encoding="utf-8")
    (repo / "CLAUDE.md").write_text("same\n", encoding="utf-8")
    for source in (MIGRATION_150_PATH, MIGRATION_151_PATH):
        (scripts / source.name).write_bytes(source.read_bytes())
    _git(repo, "add", "AGENTS.md", "CLAUDE.md", "scripts")
    _git(repo, "commit", "-m", "fixture")
    commit = _git(repo, "rev-parse", "HEAD")
    remote = (tmp_path / "authorized-origin.git").as_posix()
    _git(repo, "remote", "add", "origin", remote)
    _git(repo, "update-ref", "refs/remotes/origin/main", commit)
    monkeypatch.setattr(runner, "AUTHORIZED_REMOTE_URLS", {remote})

    hashes = tuple(
        hashlib.sha256((repo / path).read_bytes()).hexdigest()
        for path in runner.MIGRATION_PATHS
    )
    verified = runner._verify_repository_source(
        repo,
        commit=commit,
        expected_hashes=hashes,
        enforce_canonical=False,
        refresh_origin=False,
    )
    assert tuple(item.path for item in verified) == runner.MIGRATION_PATHS

    (repo / "untracked.txt").write_text("local-only\n", encoding="utf-8")
    runner._verify_repository_source(
        repo,
        commit=commit,
        expected_hashes=hashes,
        enforce_canonical=False,
        refresh_origin=False,
    )

    (repo / "AGENTS.md").write_text("unstaged edit\n", encoding="utf-8")
    with pytest.raises(runner.MigrationApplyError, match="tracked repository"):
        runner._verify_repository_source(
            repo,
            commit=commit,
            expected_hashes=hashes,
            enforce_canonical=False,
            refresh_origin=False,
        )
    _git(repo, "add", "AGENTS.md")
    with pytest.raises(runner.MigrationApplyError, match="tracked repository"):
        runner._verify_repository_source(
            repo,
            commit=commit,
            expected_hashes=hashes,
            enforce_canonical=False,
            refresh_origin=False,
        )


def test_source_verification_rejects_wrong_hash_and_noncanonical_root(
    tmp_path, monkeypatch
):
    with pytest.raises(runner.MigrationApplyError, match="canonical repository"):
        runner._require_canonical_repo(tmp_path)

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Migration Test")
    (repo / "scripts").mkdir()
    (repo / "AGENTS.md").write_text("same\n", encoding="utf-8")
    (repo / "CLAUDE.md").write_text("same\n", encoding="utf-8")
    for source in (MIGRATION_150_PATH, MIGRATION_151_PATH):
        (repo / "scripts" / source.name).write_bytes(source.read_bytes())
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    commit = _git(repo, "rev-parse", "HEAD")
    remote = (tmp_path / "authorized-origin.git").as_posix()
    _git(repo, "remote", "add", "origin", remote)
    _git(repo, "update-ref", "refs/remotes/origin/main", commit)
    monkeypatch.setattr(runner, "AUTHORIZED_REMOTE_URLS", {remote})
    hashes = [
        hashlib.sha256((repo / path).read_bytes()).hexdigest()
        for path in runner.MIGRATION_PATHS
    ]
    hashes[0] = "0" * 64
    with pytest.raises(runner.MigrationApplyError, match="operator-bound"):
        runner._verify_repository_source(
            repo,
            commit=commit,
            expected_hashes=hashes,
            enforce_canonical=False,
            refresh_origin=False,
        )


def test_statement_and_connection_contracts_remain_fail_closed(monkeypatch):
    migration_150 = _migration(MIGRATION_150_PATH)
    migration_151 = _migration(MIGRATION_151_PATH)
    statements_150, statements_151 = runner._validated_statement_sets(
        migration_150, migration_151
    )
    assert [runner._statement_command(item) for item in statements_150] == [
        "SET",
        "CREATE",
        "CREATE",
        "CREATE",
        "DO",
        "RESET",
    ]
    assert runner._statement_command(statements_151[0]) == "BEGIN"
    assert runner._statement_command(statements_151[-1]) == "COMMIT"

    monkeypatch.setenv(
        "SUPABASE_DB_URL",
        "postgresql://postgres.%s:secret@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
        % runner.PROJECT_REF,
    )
    with pytest.raises(runner.MigrationApplyError, match="port 5432"):
        runner._database_url_from_environment()
    monkeypatch.setenv(
        "SUPABASE_DB_URL",
        "postgresql://postgres:secret@db.other.supabase.co:5432/postgres",
    )
    with pytest.raises(runner.MigrationApplyError, match="authorized Supabase project"):
        runner._database_url_from_environment()
    valid_session_dsn = (
        "postgresql://postgres.%s:fixture-password@"
        "aws-0-us-east-1.pooler.supabase.com:5432/postgres?sslmode=require"
        % runner.PROJECT_REF
    )
    monkeypatch.setenv("SUPABASE_DB_URL", valid_session_dsn)
    assert runner._database_url_from_environment() == valid_session_dsn
    monkeypatch.setenv(
        "SUPABASE_DB_URL",
        valid_session_dsn + "&host=db.other.supabase.co",
    )
    with pytest.raises(runner.MigrationApplyError, match="unauthorized options"):
        runner._database_url_from_environment()

    monkeypatch.setattr(
        runner,
        "_active_target_builds",
        lambda _connection: ["public.table[index]"],
    )
    with pytest.raises(runner.MigrationApplyError, match="build is active"):
        runner._require_no_active_builds(object())


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.mark.skipif(DOCKER is None, reason="Docker is unavailable")
def test_direct_autocommit_runner_resumes_partial_indexes_idempotently():
    psycopg2 = pytest.importorskip("psycopg2")
    port = _free_port()
    container = "failure-funnel-runner-%s" % uuid4().hex[:12]
    started = False
    connection = None
    try:
        result = subprocess.run(
            [
                str(DOCKER),
                "run",
                "--rm",
                "--detach",
                "--name",
                container,
                "--cpus",
                "0.5",
                "--memory",
                "256m",
                "--shm-size",
                "64m",
                "--tmpfs",
                "/var/lib/postgresql/data:rw,size=128m",
                "--env",
                "POSTGRES_PASSWORD=postgres",
                "--publish",
                "127.0.0.1:%d:5432" % port,
                "postgres:15",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            pytest.skip("PostgreSQL container could not start")
        started = True
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            ready = subprocess.run(
                [str(DOCKER), "exec", container, "pg_isready", "-U", "postgres"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if ready.returncode == 0:
                break
            time.sleep(0.25)
        else:
            pytest.fail("PostgreSQL container did not become ready")

        connection = psycopg2.connect(
            host="127.0.0.1",
            port=port,
            user="postgres",
            password="postgres",
            dbname="postgres",
        )
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE ROLE anon NOLOGIN;
                CREATE ROLE authenticated NOLOGIN;
                CREATE ROLE service_role NOLOGIN;

                CREATE TABLE public.research_evaluation_score_bundles (
                    score_bundle_id TEXT PRIMARY KEY,
                    run_id UUID NOT NULL,
                    ticket_id UUID,
                    bundle_status TEXT NOT NULL,
                    private_model_manifest_hash TEXT NOT NULL,
                    score_bundle_doc JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE VIEW public.research_evaluation_score_bundle_current AS
                SELECT b.*, b.bundle_status AS current_event_status
                FROM public.research_evaluation_score_bundles b;

                CREATE TABLE public.research_lab_company_label_examples (
                    label_id UUID PRIMARY KEY,
                    ticket_id UUID,
                    candidate_id TEXT,
                    icp_ref TEXT NOT NULL DEFAULT 'icp-1',
                    model_side TEXT NOT NULL DEFAULT 'candidate',
                    capture_state TEXT NOT NULL DEFAULT 'captured_unreviewed',
                    final_score DOUBLE PRECISION NOT NULL DEFAULT 0,
                    failure_reason TEXT,
                    failure_stage TEXT,
                    capture_doc JSONB NOT NULL DEFAULT '{}'::JSONB,
                    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE TABLE public.research_lab_scoring_runs (
                    scoring_run_id UUID PRIMARY KEY,
                    ticket_id UUID,
                    candidate_id TEXT,
                    run_type TEXT NOT NULL DEFAULT 'candidate_scoring',
                    run_attempt INTEGER NOT NULL DEFAULT 0,
                    expected_icp_count INTEGER NOT NULL,
                    current_run_status TEXT DEFAULT 'completed',
                    current_telemetry_degraded BOOLEAN NOT NULL DEFAULT FALSE,
                    score_bundle_id TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE VIEW public.research_lab_scoring_run_current AS
                SELECT * FROM public.research_lab_scoring_runs;
                CREATE TABLE public.research_lab_scoring_icp_executions (
                    icp_execution_id UUID PRIMARY KEY,
                    scoring_run_id UUID NOT NULL,
                    icp_ref TEXT NOT NULL DEFAULT 'icp-1',
                    model_role TEXT NOT NULL DEFAULT 'candidate',
                    attempt_ordinal INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE TABLE public.research_lab_scoring_icp_events (
                    event_id UUID PRIMARY KEY,
                    icp_execution_id UUID NOT NULL,
                    event_type TEXT NOT NULL,
                    retryable BOOLEAN,
                    failure_category TEXT,
                    telemetry_degraded BOOLEAN NOT NULL DEFAULT FALSE,
                    event_ordinal BIGINT NOT NULL,
                    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                -- This intentionally mirrors the broader valid production
                -- index that predates migration 150.  The migration's own
                -- catalog contract accepts it and creates only missing names.
                CREATE INDEX idx_research_eval_score_bundles_ticket_created
                    ON public.research_evaluation_score_bundles(
                        ticket_id, created_at DESC
                    );
                GRANT SELECT ON ALL TABLES IN SCHEMA public TO service_role;
                """
            )

        migration_150 = _migration(MIGRATION_150_PATH)
        migration_151 = _migration(MIGRATION_151_PATH)
        initial = runner._apply_verified_migrations(
            connection, migration_150, migration_151
        )
        assert {name: state for name, _table, state in initial} == {
            "idx_research_eval_score_bundles_ticket_created": "ready",
            "idx_research_lab_company_labels_ticket_candidate": "missing",
            "idx_research_lab_scoring_runs_ticket_candidate": "missing",
        }
        assert all(
            state == "ready"
            for _name, _table, state in runner._index_states(connection)
        )

        resumed = runner._apply_verified_migrations(
            connection, migration_150, migration_151
        )
        assert all(state == "ready" for _name, _table, state in resumed)
        runner._require_reporting_function(connection)

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT pg_catalog.obj_description(
                    'public.get_research_lab_failure_funnel(uuid,text)'::REGPROCEDURE,
                    'pg_proc'
                )
                """
            )
            original_comment = cursor.fetchone()[0]
        broken_sql = migration_151.sql.replace(
            "NOTIFY pgrst, 'reload schema';",
            """
            COMMENT ON FUNCTION public.get_research_lab_failure_funnel(UUID, TEXT)
                IS 'this transaction must roll back';
            SELECT 1 / 0;
            NOTIFY pgrst, 'reload schema';
            """,
        )
        broken_151 = runner.VerifiedMigration(
            path=migration_151.path,
            sha256=hashlib.sha256(broken_sql.encode("utf-8")).hexdigest(),
            sql=broken_sql,
        )
        with pytest.raises(runner.MigrationApplyError, match="migration 151 failed"):
            runner._apply_verified_migrations(connection, migration_150, broken_151)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT pg_catalog.obj_description(
                    'public.get_research_lab_failure_funnel(uuid,text)'::REGPROCEDURE,
                    'pg_proc'
                )
                """
            )
            assert cursor.fetchone()[0] == original_comment

        with connection.cursor() as cursor:
            cursor.execute(
                "DROP INDEX public.idx_research_lab_scoring_runs_ticket_candidate"
            )
            cursor.execute(
                """
                CREATE INDEX idx_research_lab_scoring_runs_ticket_candidate
                    ON public.research_lab_company_label_examples(ticket_id)
                """
            )
        with pytest.raises(runner.MigrationApplyError, match="exact catalog contract"):
            runner._apply_verified_migrations(connection, migration_150, migration_151)
    finally:
        if connection is not None:
            connection.close()
        if started:
            subprocess.run(
                [str(DOCKER), "rm", "--force", container],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
