"""Real-PostgreSQL integration test for the egress-reduction migrations.

Mocked-RPC and SQL-string tests prove call shape and lock-down, but not that the
functions actually EXECUTE against a live engine. This test stands up a
throwaway PostgreSQL, applies migrations 115-118 verbatim, and exercises every
RPC's runtime behavior. It already caught a real bug (`pg_catalog.coalesce` is
not a function -- COALESCE/GREATEST are keyword expressions), which mocked tests
could never surface.

Skips only when no PostgreSQL server binaries are found (initdb/pg_ctl/psql);
CI's ubuntu image and Homebrew both provide them.
"""

from __future__ import annotations

import glob
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _find_pg_bindir() -> str | None:
    """Locate a directory containing initdb + pg_ctl + postgres + psql."""
    candidates: list[str] = []
    # PATH first.
    initdb_on_path = shutil.which("initdb")
    if initdb_on_path:
        candidates.append(str(Path(initdb_on_path).parent))
    # Linux (GitHub ubuntu images) and Homebrew (macOS).
    candidates += sorted(glob.glob("/usr/lib/postgresql/*/bin"), reverse=True)
    candidates += sorted(glob.glob("/opt/homebrew/opt/postgresql@*/bin"), reverse=True)
    candidates += sorted(glob.glob("/usr/local/opt/postgresql@*/bin"), reverse=True)
    candidates += sorted(glob.glob("/usr/pgsql-*/bin"), reverse=True)
    for d in candidates:
        if all((Path(d) / b).exists() for b in ("initdb", "pg_ctl", "postgres", "psql")):
            return d
    return None


_PG_BINDIR = _find_pg_bindir()
pytestmark = pytest.mark.skipif(
    _PG_BINDIR is None,
    reason="no PostgreSQL server binaries (initdb/pg_ctl/postgres/psql) available",
)


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# Supabase-compat shims (roles the GRANT/RLS statements reference) + the minimal
# dependency relations migrations 115/116/118 reference.
_SETUP_SQL = """
CREATE ROLE anon NOLOGIN;
CREATE ROLE authenticated NOLOGIN;
CREATE ROLE service_role NOLOGIN;

CREATE TABLE public.research_lab_provider_usage_ledger (
    usage_row_id        UUID PRIMARY KEY,
    schema_version      TEXT NOT NULL DEFAULT '1.0',
    utc_day             TEXT,
    recorded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    provider_id         TEXT,
    endpoint_class      TEXT,
    request_fingerprint TEXT,
    evidence            TEXT,
    status              INTEGER,
    est_cost_microusd   BIGINT,
    caller_doc          JSONB NOT NULL DEFAULT '{}'::JSONB
);

CREATE TABLE public.research_trajectories (trajectory_id UUID PRIMARY KEY);

CREATE TABLE public.research_lab_candidate_evaluation_current (
    candidate_id             TEXT,
    run_id                   UUID,
    ticket_id                UUID,
    current_candidate_status TEXT,
    current_reason           TEXT,
    current_status_at        TIMESTAMPTZ
);
"""

# Runtime assertions. Each DO block RAISEs on failure; the psql -v ON_ERROR_STOP=1
# flag makes that a non-zero exit, which the test asserts against.
_ASSERT_SQL = r"""
DO $$
DECLARE r JSONB; n INT;
BEGIN
    -- 115: batched insert is conflict-safe and idempotent.
    r := public.insert_research_lab_provider_usage_ledger_rows(
        '[{"usage_row_id":"11111111-1111-4111-8111-111111111111","provider_id":"scrapingdog"},
          {"usage_row_id":"22222222-2222-4222-8222-222222222222","provider_id":"exa"}]'::JSONB);
    IF (r->>'inserted')::INT <> 2 THEN RAISE EXCEPTION '115 first insert expected 2, got %', r->>'inserted'; END IF;
    r := public.insert_research_lab_provider_usage_ledger_rows(
        '[{"usage_row_id":"11111111-1111-4111-8111-111111111111","provider_id":"scrapingdog"},
          {"usage_row_id":"33333333-3333-4333-8333-333333333333","provider_id":"deepline"}]'::JSONB);
    IF (r->>'inserted')::INT <> 1 THEN RAISE EXCEPTION '115 idempotent insert expected 1, got %', r->>'inserted'; END IF;
    SELECT count(*) INTO n FROM public.research_lab_provider_usage_ledger;
    IF n <> 3 THEN RAISE EXCEPTION '115 total rows expected 3, got %', n; END IF;
    BEGIN
        r := public.insert_research_lab_provider_usage_ledger_rows('{"not":"array"}'::JSONB);
        RAISE EXCEPTION '115 should reject a non-array batch';
    EXCEPTION WHEN sqlstate '22023' THEN NULL; END;

    -- 116: anti-join returns only the ids absent from research_trajectories.
    INSERT INTO public.research_trajectories(trajectory_id) VALUES ('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa');
    SELECT count(*) INTO n FROM public.research_lab_missing_trajectory_ids(ARRAY[
        'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'::UUID,
        'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb'::UUID,
        'cccccccc-cccc-4ccc-8ccc-cccccccccccc'::UUID]);
    IF n <> 2 THEN RAISE EXCEPTION '116 expected 2 missing, got %', n; END IF;

    -- 118: single eligible queued candidate, staleness filter applied, LIMIT 1.
    INSERT INTO public.research_lab_candidate_evaluation_current
        (candidate_id, run_id, ticket_id, current_candidate_status, current_reason, current_status_at)
    VALUES
        ('c-old','10000000-0000-4000-8000-000000000001','20000000-0000-4000-8000-000000000001','queued','',now()-interval '10 min'),
        ('c-new','10000000-0000-4000-8000-000000000002','20000000-0000-4000-8000-000000000002','queued','',now()-interval '1 min'),
        ('c-hold','10000000-0000-4000-8000-000000000003','20000000-0000-4000-8000-000000000003','queued','baseline_not_ready',now()),
        ('c-done','10000000-0000-4000-8000-000000000004','20000000-0000-4000-8000-000000000004','scored','',now()-interval '99 min');
    SELECT count(*) INTO n FROM public.claim_next_research_lab_candidate(900, 300);
    IF n <> 1 THEN RAISE EXCEPTION '118 expected exactly 1 row, got %', n; END IF;
    IF (SELECT candidate_id FROM public.claim_next_research_lab_candidate(900, 300)) <> 'c-old'
        THEN RAISE EXCEPTION '118 expected c-old (oldest eligible) first'; END IF;
    RAISE NOTICE 'RPC-BEHAVIOR-OK';
END $$;

DO $$
DECLARE r JSONB;
        tok_a TEXT := 'research-lab-worker-1#hostx#111#aaa';
        tok_b TEXT := 'research-lab-worker-1#hostx#222#bbb';
        lease TEXT := 'hosted_worker_maintenance';
BEGIN
    -- 117: two processes with the SAME human-readable name but DIFFERENT lease
    -- tokens -- only one may hold the lease (the identity-bug regression guard).
    r := public.research_lab_acquire_maintenance_lease(lease, tok_a, 180);
    IF (r->>'acquired')::BOOL IS NOT TRUE THEN RAISE EXCEPTION '117 A should acquire'; END IF;
    r := public.research_lab_acquire_maintenance_lease(lease, tok_b, 180);
    IF (r->>'acquired')::BOOL IS NOT FALSE THEN RAISE EXCEPTION '117 B must NOT acquire while A holds'; END IF;
    IF (r->>'holder_ref') <> tok_a THEN RAISE EXCEPTION '117 holder should still be A'; END IF;
    r := public.research_lab_acquire_maintenance_lease(lease, tok_a, 180);
    IF (r->>'acquired')::BOOL IS NOT TRUE THEN RAISE EXCEPTION '117 A should renew'; END IF;
    UPDATE public.research_lab_maintenance_lease SET expires_at = now() - interval '1 s' WHERE lease_name = lease;
    r := public.research_lab_acquire_maintenance_lease(lease, tok_b, 180);
    IF (r->>'acquired')::BOOL IS NOT TRUE THEN RAISE EXCEPTION '117 B should take an expired lease'; END IF;
    BEGIN
        r := public.research_lab_acquire_maintenance_lease(lease, tok_a, 0);
        RAISE EXCEPTION '117 should reject ttl=0';
    EXCEPTION WHEN sqlstate '22023' THEN NULL; END;
    RAISE NOTICE 'LEASE-IDENTITY-OK';
END $$;
"""


@pytest.fixture(scope="module")
def pg_db():
    bindir = Path(_PG_BINDIR)
    datadir = Path(tempfile.mkdtemp(prefix="egress-pg-"))
    # A short socket dir (postgres needs a valid one even for TCP-only clients);
    # the datadir path can be too long for the 103-byte socket limit.
    sockdir = Path(tempfile.mkdtemp(prefix="/tmp/pgs"))
    port = _free_port()
    env = {**os.environ, "PGTZ": "UTC", "TZ": "UTC"}

    def _bin(name: str) -> str:
        return str(bindir / name)

    def psql(args: list[str], input_sql: str | None = None):
        return subprocess.run(
            [_bin("psql"), "-v", "ON_ERROR_STOP=1", "-h", "127.0.0.1",
             "-p", str(port), "-U", "postgres", "-d", "migtest", "-q", *args],
            input=input_sql, capture_output=True, text=True, env=env, timeout=60,
        )

    started = False
    try:
        subprocess.run(
            [_bin("initdb"), "-D", str(datadir), "-U", "postgres", "--auth=trust"],
            check=True, capture_output=True, text=True, env=env, timeout=90,
        )
        # -l is required: without a logfile the postgres daemon inherits
        # pg_ctl's stdout pipe and holds it open, so capture_output would block
        # on EOF forever (the daemon never exits).
        subprocess.run(
            [_bin("pg_ctl"), "-D", str(datadir), "-w", "-t", "30",
             "-l", str(datadir / "server.log"), "-o",
             f"-p {port} -c listen_addresses=127.0.0.1 "
             f"-c unix_socket_directories={sockdir}",
             "start"],
            check=True, capture_output=True, text=True, env=env, timeout=45,
        )
        started = True
        subprocess.run(
            [_bin("psql"), "-h", "127.0.0.1", "-p", str(port), "-U", "postgres",
             "-d", "postgres", "-c", "CREATE DATABASE migtest"],
            check=True, capture_output=True, text=True, env=env, timeout=30,
        )
        yield psql
    finally:
        if started:
            subprocess.run([_bin("pg_ctl"), "-D", str(datadir), "-w", "-t", "20", "stop"],
                           capture_output=True, text=True, env=env, timeout=30)
        shutil.rmtree(datadir, ignore_errors=True)
        shutil.rmtree(sockdir, ignore_errors=True)


def test_migrations_115_to_118_apply_and_execute(pg_db) -> None:
    # Shims + dependency stubs.
    setup = pg_db(["-c", _SETUP_SQL])
    assert setup.returncode == 0, setup.stderr

    # Apply each migration verbatim.
    for num in ("115", "116", "117", "118"):
        matches = sorted(SCRIPTS.glob(f"{num}-*.sql"))
        assert matches, f"migration {num} not found"
        applied = pg_db(["-f", str(matches[0])])
        assert applied.returncode == 0, f"migration {num} failed:\n{applied.stderr}"

    # Exercise the RPCs' runtime behavior.
    result = pg_db(["-c", _ASSERT_SQL])
    assert result.returncode == 0, result.stderr
    assert "RPC-BEHAVIOR-OK" in result.stderr
    assert "LEASE-IDENTITY-OK" in result.stderr
