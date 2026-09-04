"""Disposable PostgreSQL for Lab Arena tests.

Prefers the repository's Docker harness (``_database_with_migrations`` from
``tests/test_source_add_end_to_end_postgres.py``), which is what CI runs.
When ``LAB_ARENA_PG_LOCAL=1`` is set, or Docker is unavailable, a local
PostgreSQL server (``initdb``/``pg_ctl``) is used instead with the same
Supabase shim, so the migration is exercised identically. Both paths yield
``(psycopg2, dsn)`` exactly like the Docker harness.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.test_source_add_end_to_end_postgres import SCRIPTS, _database_with_migrations

LAB_ARENA_MIGRATION = "179-lab-arena-v1.sql"
LAB_ARENA_DAILY_COMPETITION_MIGRATION = "180-lab-arena-daily-competition.sql"
LAB_ARENA_SOURCE_SUBMISSIONS_MIGRATION = "181-lab-arena-source-submissions.sql"
LAB_ARENA_SOURCE_EXECUTION_MIGRATION = "182-lab-arena-source-execution.sql"
LAB_ARENA_MINER_REWARD_MIGRATION = "183-lab-arena-miner-reward-basis.sql"
LAB_ARENA_SCORING_ISOLATION_MIGRATION = "184-lab-arena-scoring-failure-isolation.sql"
LAB_ARENA_MINER_CREDENTIALS_MIGRATION = "185-lab-arena-miner-credentials.sql"
DEFAULT_MIGRATIONS = (
    LAB_ARENA_MIGRATION,
    LAB_ARENA_DAILY_COMPETITION_MIGRATION,
    LAB_ARENA_SOURCE_SUBMISSIONS_MIGRATION,
    LAB_ARENA_SOURCE_EXECUTION_MIGRATION,
    LAB_ARENA_MINER_REWARD_MIGRATION,
    LAB_ARENA_SCORING_ISOLATION_MIGRATION,
    LAB_ARENA_MINER_CREDENTIALS_MIGRATION,
)

_SHIM_SQL = """
CREATE SCHEMA IF NOT EXISTS extensions;
CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA extensions;
CREATE ROLE anon NOLOGIN;
CREATE ROLE authenticated NOLOGIN;
CREATE ROLE service_role NOLOGIN;
"""

_DAILY_SOURCE_SHIM_SQL = """
CREATE TABLE public.qualification_private_icp_sets (
  set_id BIGINT PRIMARY KEY,
  icps JSONB NOT NULL,
  active_from TIMESTAMPTZ,
  active_until TIMESTAMPTZ,
  is_active BOOLEAN NOT NULL DEFAULT FALSE
);
ALTER TABLE public.qualification_private_icp_sets ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.qualification_private_icp_sets
  FROM PUBLIC, anon, authenticated;
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _server_bindir() -> Path | None:
    configured = os.environ.get("LAB_ARENA_PG_BINDIR")
    candidates = [Path(configured)] if configured else []
    which = shutil.which("postgres")
    if which:
        candidates.append(Path(which).parent)
    candidates.extend(
        sorted(Path("/opt/homebrew/opt").glob("postgresql@*/bin"), reverse=True)
    )
    candidates.extend(sorted(Path("/usr/lib/postgresql").glob("*/bin"), reverse=True))
    for candidate in candidates:
        if (candidate / "postgres").exists() and (candidate / "initdb").exists():
            return candidate
    return None


def _local_database(migrations):
    psycopg2 = pytest.importorskip("psycopg2")
    bindir = _server_bindir()
    if bindir is None:
        pytest.skip("no local PostgreSQL server binaries and Docker not used")
    datadir = Path(tempfile.mkdtemp(prefix="lab-arena-pg-"))
    sockdir = Path(tempfile.mkdtemp(prefix="/tmp/lapgs"))
    port = _free_port()
    # PostgreSQL 17 on macOS aborts with "postmaster became multithreaded
    # during startup" when the inherited locale environment loads threaded
    # frameworks, so the server runs under a scrubbed C-locale environment.
    env = {"PATH": "%s:/usr/bin:/bin" % bindir, "HOME": str(datadir), "LC_ALL": "C", "LANG": "C", "PGTZ": "UTC", "TZ": "UTC"}
    started = False
    try:
        subprocess.run(
            [str(bindir / "initdb"), "-D", str(datadir), "-U", "postgres", "--auth=trust"],
            check=True, capture_output=True, text=True, env=env, timeout=120,
        )
        subprocess.run(
            [
                str(bindir / "pg_ctl"), "-D", str(datadir), "-w", "-t", "60",
                "-l", str(datadir / "server.log"), "-o",
                "-p %d -c listen_addresses=127.0.0.1 -c unix_socket_directories=%s -c fsync=off" % (port, sockdir),
                "start",
            ],
            check=True, capture_output=True, text=True, env=env, timeout=90,
        )
        started = True
        dsn = {"host": "127.0.0.1", "port": port, "user": "postgres", "dbname": "postgres"}
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(_SHIM_SQL)
            cursor.execute(_DAILY_SOURCE_SHIM_SQL)
            for migration in migrations:
                cursor.execute((SCRIPTS / migration).read_text(encoding="utf-8"))
        connection.close()
        yield psycopg2, dsn
    finally:
        if started:
            subprocess.run(
                [str(bindir / "pg_ctl"), "-D", str(datadir), "-w", "-t", "30", "-m", "immediate", "stop"],
                capture_output=True, text=True, env=env, timeout=60,
            )
        shutil.rmtree(datadir, ignore_errors=True)
        shutil.rmtree(sockdir, ignore_errors=True)


def database_with_lab_arena_migration(migrations=DEFAULT_MIGRATIONS):
    """Yield ``(psycopg2, dsn)`` for a disposable database with the migration applied."""

    use_local = os.environ.get("LAB_ARENA_PG_LOCAL") == "1" or shutil.which("docker") is None
    if use_local:
        yield from _local_database(migrations)
        return
    # The Docker harness polls pg_isready with short fixed timeouts; on a loaded
    # host that first poll can time out before the container answers. Retry the
    # whole start a bounded number of times so a slow Docker daemon is not
    # mistaken for a migration failure.
    last_error: BaseException | None = None
    for _attempt in range(3):
        generator = _database_with_migrations(
            migrations, setup_sql=_DAILY_SOURCE_SHIM_SQL
        )
        try:
            value = next(generator)
        except (subprocess.TimeoutExpired, pytest.fail.Exception) as exc:  # type: ignore[attr-defined]
            last_error = exc
            generator.close()
            continue
        try:
            yield value
        finally:
            generator.close()
        return
    raise RuntimeError("disposable PostgreSQL did not start after 3 attempts: %r" % (last_error,))
