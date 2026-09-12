"""Disposable PostgreSQL for Lab Arena tests.

Prefers the repository's Docker harness (``_database_with_migrations`` from
``tests/postgres_migration_harness.py``), which is what CI runs.
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

from tests.postgres_migration_harness import SCRIPTS, _database_with_migrations

LAB_ARENA_MIGRATION = "179-lab-arena-v1.sql"
LAB_ARENA_DAILY_COMPETITION_MIGRATION = "180-lab-arena-daily-competition.sql"
LAB_ARENA_SOURCE_SUBMISSIONS_MIGRATION = "181-lab-arena-source-submissions.sql"
LAB_ARENA_SOURCE_EXECUTION_MIGRATION = "182-lab-arena-source-execution.sql"
LAB_ARENA_MINER_REWARD_MIGRATION = "183-lab-arena-miner-reward-basis.sql"
LAB_ARENA_SCORING_ISOLATION_MIGRATION = "184-lab-arena-scoring-failure-isolation.sql"
LAB_ARENA_MINER_CREDENTIALS_MIGRATION = "185-lab-arena-miner-credentials.sql"
LAB_ARENA_PROMOTION_THRESHOLD_MIGRATION = "187-lab-arena-promotion-threshold.sql"
LAB_ARENA_BASELINE_PROMOTION_MIGRATION = "188-lab-arena-baseline-promotion.sql"
LAB_ARENA_NETWORK_SCOPE_MIGRATION = "189-lab-arena-round-network-scope.sql"
LAB_ARENA_RESTART_CLAIM_DRAIN_MIGRATION = "190-lab-arena-restart-claim-drain.sql"
LAB_ARENA_UPLOAD_RECOVERY_MIGRATION = "193-lab-arena-upload-recovery.sql"
LAB_ARENA_OPEN_SCORER_REFRESH_MIGRATION = "194-lab-arena-open-scorer-refresh.sql"
LAB_ARENA_REWARD_CHAIN_SCOPE_MIGRATION = "197-lab-arena-reward-chain-scope.sql"
LAB_ARENA_SOURCE_DISCLOSURE_MIGRATION = "199-lab-arena-source-disclosure-time.sql"
LAB_ARENA_NEXT_DAY_ICP_MIGRATION = "200-lab-arena-next-day-icp-disclosure.sql"
LAB_ARENA_ACCEPTED_WEIGHT_STATE_MIGRATION = "202-arena-accepted-weight-state.sql"
LAB_ARENA_RETIRED_INCENTIVE_BRIDGE_MIGRATION = "203-retire-legacy-incentive-weight-bridge.sql"
LAB_ARENA_OPTIONAL_SCRAPINGDOG_CREDENTIAL_MIGRATION = "205-lab-arena-optional-scrapingdog-credential.sql"
LAB_ARENA_COMBINED_PROVIDER_BUDGET_MIGRATION = "206-lab-arena-combined-provider-budget.sql"
LAB_ARENA_CODE_REVIEW_MIGRATION = "207-lab-arena-code-review.sql"
LAB_ARENA_VALIDATOR_SCORING_AUTHORITY_MIGRATION = "208-lab-arena-validator-scoring-authority.sql"
LAB_ARENA_UNCERTAIN_COST_ELIGIBILITY_MIGRATION = "209-lab-arena-uncertain-cost-eligibility.sql"
DEFAULT_MIGRATIONS = (
    LAB_ARENA_MIGRATION,
    LAB_ARENA_DAILY_COMPETITION_MIGRATION,
    LAB_ARENA_SOURCE_SUBMISSIONS_MIGRATION,
    LAB_ARENA_SOURCE_EXECUTION_MIGRATION,
    LAB_ARENA_MINER_REWARD_MIGRATION,
    LAB_ARENA_SCORING_ISOLATION_MIGRATION,
    LAB_ARENA_MINER_CREDENTIALS_MIGRATION,
    LAB_ARENA_PROMOTION_THRESHOLD_MIGRATION,
    LAB_ARENA_BASELINE_PROMOTION_MIGRATION,
    LAB_ARENA_NETWORK_SCOPE_MIGRATION,
    LAB_ARENA_RESTART_CLAIM_DRAIN_MIGRATION,
    LAB_ARENA_UPLOAD_RECOVERY_MIGRATION,
    LAB_ARENA_OPEN_SCORER_REFRESH_MIGRATION,
    LAB_ARENA_REWARD_CHAIN_SCOPE_MIGRATION,
    LAB_ARENA_SOURCE_DISCLOSURE_MIGRATION,
    LAB_ARENA_NEXT_DAY_ICP_MIGRATION,
    "201-lab-arena-daily-capacity.sql",
    LAB_ARENA_ACCEPTED_WEIGHT_STATE_MIGRATION,
    LAB_ARENA_RETIRED_INCENTIVE_BRIDGE_MIGRATION,
    LAB_ARENA_OPTIONAL_SCRAPINGDOG_CREDENTIAL_MIGRATION,
    LAB_ARENA_COMBINED_PROVIDER_BUDGET_MIGRATION,
    LAB_ARENA_CODE_REVIEW_MIGRATION,
    LAB_ARENA_VALIDATOR_SCORING_AUTHORITY_MIGRATION,
    LAB_ARENA_UNCERTAIN_COST_ELIGIBILITY_MIGRATION,
)
# Keep the historical default intact: several migration tests intentionally
# exercise intermediate schemas. PostgREST round tests need the current
# integrity, contact, and company-quality RPCs and their prerequisites.
POSTGREST_MIGRATIONS = DEFAULT_MIGRATIONS + (
    "211-lab-arena-owner-admission.sql",
    "212-lab-arena-accepted-judgment-cache.sql",
    "213-lab-arena-score-integrity.sql",
    "214-lab-arena-prior-credential-refusal.sql",
    "215-lab-arena-contacts.sql",
    "216-lab-arena-validator-participation.sql",
    "217-lab-arena-company-judgments.sql",
    "218-lab-arena-cross-provider-credential-refusal.sql",
    "221-lab-arena-participation-original-judgments.sql",
    "223-lab-arena-cancelled-call-late-settlement.sql",
    "225-lab-arena-openrouter-delayed-cost-reconciliation.sql",
    "227-lab-arena-champion-funding.sql",
    "228-lab-arena-hotkey-admission.sql",
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
