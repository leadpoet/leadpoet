"""Real-PostgreSQL integration test for the egress-reduction migrations.

Mocked-RPC and SQL-string tests prove call shape and lock-down, but not that the
functions actually EXECUTE against a live engine. This test stands up a
throwaway PostgreSQL, applies the migrations verbatim, and exercises every RPC's
runtime behavior -- batch-insert idempotency, the trajectory anti-join, the
single-owner lease identity, the deterministic-id reproduction + projection
deltas, and the atomic candidate/run claims (including concurrent no-double-
assign via real threads). It already caught a real bug (`pg_catalog.coalesce`
is not a function -- COALESCE/GREATEST are keyword expressions).

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
import threading
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MIGRATIONS = ("115", "116", "117", "119", "120", "121", "122")

# A known trajectory_id_for_run value (verified against Python and 300 live runs)
# pins the in-SQL uuid5(canonical_hash(...)) reproduction.
_KNOWN_RUN_ID = "3f2a1c00-0000-4000-8000-000000000001"
_KNOWN_TRAJECTORY_ID = "1544660e-89e4-52e6-84be-8bc3d0e01907"


def _find_pg_bindir() -> str | None:
    candidates: list[str] = []
    initdb_on_path = shutil.which("initdb")
    if initdb_on_path:
        candidates.append(str(Path(initdb_on_path).parent))
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


# Supabase-compat shims (roles the GRANT/RLS statements reference; pgcrypto in
# the extensions schema, matching Supabase) + the minimal dependency relations
# the migrations reference.
_SETUP_SQL = """
CREATE ROLE anon NOLOGIN;
CREATE ROLE authenticated NOLOGIN;
CREATE ROLE service_role NOLOGIN;
CREATE SCHEMA IF NOT EXISTS extensions;
CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA extensions;

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
CREATE TABLE public.execution_traces (run_id UUID PRIMARY KEY);

-- Watermark source tables (minimal stubs matching the columns the
-- research_lab_corpus_source_watermark function aggregates).
CREATE TABLE public.research_lab_auto_research_loop_events (
    run_id UUID, seq INTEGER, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE public.research_lab_candidate_artifacts (
    candidate_id TEXT, run_id UUID
);
CREATE TABLE public.research_lab_candidate_evaluation_events (
    candidate_id TEXT, run_id UUID, seq INTEGER
);
CREATE TABLE public.research_lab_candidate_promotion_events (
    candidate_id TEXT, private_model_version_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE public.research_lab_private_model_version_events (
    private_model_version_id UUID, seq INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE public.research_evaluation_score_bundles (
    score_bundle_id TEXT, run_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE public.research_loop_run_queue_current (
    run_id                   UUID,
    ticket_id                UUID,
    queue_priority           INTEGER,
    current_event_hash       TEXT,
    current_status_at        TIMESTAMPTZ,
    current_queue_status     TEXT
);

CREATE TABLE public.research_lab_candidate_evaluation_current (
    candidate_id               TEXT,
    run_id                     UUID,
    ticket_id                  UUID,
    private_model_manifest_doc JSONB,
    candidate_patch_manifest   JSONB,
    miner_hotkey               TEXT,
    candidate_kind             TEXT,
    candidate_model_manifest_doc JSONB,
    candidate_build_doc        JSONB,
    candidate_source_diff_hash TEXT,
    candidate_patch_hash       TEXT,
    parent_artifact_hash       TEXT,
    receipt_id                 UUID,
    island                     TEXT,
    hypothesis_doc             JSONB,
    redacted_public_summary    TEXT,
    current_candidate_status   TEXT,
    current_reason             TEXT,
    current_status_at          TIMESTAMPTZ
);
"""

# Sequential runtime assertions (single psql session).
_ASSERT_SQL = r"""
DO $$
DECLARE r JSONB; n INT; c1 TEXT; c2 TEXT; ru1 UUID; ru2 UUID;
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

    -- 116: anti-join returns only the ids absent from research_trajectories.
    INSERT INTO public.research_trajectories(trajectory_id) VALUES ('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa');
    SELECT count(*) INTO n FROM public.research_lab_missing_trajectory_ids(ARRAY[
        'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'::UUID,
        'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb'::UUID]);
    IF n <> 1 THEN RAISE EXCEPTION '116 expected 1 missing, got %', n; END IF;

    -- 119: in-SQL deterministic id matches Python (pins uuid5(canonical_hash)).
    IF public.research_lab_trajectory_id('3f2a1c00-0000-4000-8000-000000000001')
       <> '1544660e-89e4-52e6-84be-8bc3d0e01907' THEN
        RAISE EXCEPTION '119 trajectory_id reproduction drifted from Python';
    END IF;

    -- 119 delta: seed one projected + one unprojected terminal run.
    INSERT INTO public.research_loop_run_queue_current
        (run_id, ticket_id, queue_priority, current_event_hash, current_status_at, current_queue_status)
    VALUES
        ('d0000000-0000-4000-8000-000000000001','e0000000-0000-4000-8000-000000000001',0,'h',now()-interval '5 min','completed'),
        ('d0000000-0000-4000-8000-000000000002','e0000000-0000-4000-8000-000000000002',0,'h',now()-interval '1 min','completed');
    INSERT INTO public.research_trajectories(trajectory_id)
        VALUES (public.research_lab_trajectory_id('d0000000-0000-4000-8000-000000000001'));  -- run 1 projected
    SELECT count(*) INTO n FROM public.research_lab_next_unprojected_terminal_runs(25, false);
    IF n <> 1 THEN RAISE EXCEPTION '119 expected 1 unprojected run, got %', n; END IF;
    IF (SELECT run_id FROM public.research_lab_next_unprojected_terminal_runs(25, false))
       <> 'd0000000-0000-4000-8000-000000000002' THEN
        RAISE EXCEPTION '119 unprojected delta returned the wrong run';
    END IF;
    -- run 1 is projected but has no execution trace -> flagged as missing-trace.
    SELECT count(*) INTO n FROM public.research_lab_terminal_runs_missing_traces(25, true);
    IF n <> 1 THEN RAISE EXCEPTION '119 expected 1 missing-trace run, got %', n; END IF;
    INSERT INTO public.execution_traces(run_id)
        VALUES (public.research_lab_execution_trace_id('d0000000-0000-4000-8000-000000000001'));
    SELECT count(*) INTO n FROM public.research_lab_terminal_runs_missing_traces(25, true);
    IF n <> 0 THEN RAISE EXCEPTION '119 traced run should no longer be flagged, got %', n; END IF;

    -- 122 corpus-completeness discovery. Project run 2 now (119's unprojected
    -- assertions already ran) so both terminal runs are projected; BOTH need
    -- corpus until a completeness marker exists (captures ANY missing corpus,
    -- not just a missing trace).
    INSERT INTO public.research_trajectories(trajectory_id)
        VALUES (public.research_lab_trajectory_id('d0000000-0000-4000-8000-000000000002'));
    SELECT count(*) INTO n FROM public.research_lab_terminal_runs_needing_corpus(25, true);
    IF n <> 2 THEN RAISE EXCEPTION '122 expected 2 runs needing corpus, got %', n; END IF;
    -- Mark run 1 complete at its CURRENT source watermark.
    PERFORM public.research_lab_mark_corpus_complete(
        public.research_lab_trajectory_id('d0000000-0000-4000-8000-000000000001'),
        'd0000000-0000-4000-8000-000000000001',
        public.research_lab_corpus_source_watermark('d0000000-0000-4000-8000-000000000001'));
    SELECT count(*) INTO n FROM public.research_lab_terminal_runs_needing_corpus(25, true);
    IF n <> 1 THEN RAISE EXCEPTION '122 marked run must drop from discovery, got %', n; END IF;
    IF (SELECT run_id FROM public.research_lab_terminal_runs_needing_corpus(25, true))
       <> 'd0000000-0000-4000-8000-000000000002' THEN
        RAISE EXCEPTION '122 wrong run left needing corpus';
    END IF;

    -- 122 REQUIRED late-event rediscovery: append a late LOOP event to the
    -- marked run -> its source watermark changes -> it is DISCOVERED again.
    INSERT INTO public.research_lab_auto_research_loop_events(run_id, seq)
        VALUES ('d0000000-0000-4000-8000-000000000001', 7);
    SELECT count(*) INTO n FROM public.research_lab_terminal_runs_needing_corpus(25, true);
    IF n <> 2 THEN RAISE EXCEPTION '122 late loop event must force rediscovery, got %', n; END IF;
    -- Same for a late PROMOTION + VERSION event chain (candidate -> promotion -> version).
    PERFORM public.research_lab_mark_corpus_complete(
        public.research_lab_trajectory_id('d0000000-0000-4000-8000-000000000001'),
        'd0000000-0000-4000-8000-000000000001',
        public.research_lab_corpus_source_watermark('d0000000-0000-4000-8000-000000000001'));
    SELECT count(*) INTO n FROM public.research_lab_terminal_runs_needing_corpus(25, true);
    IF n <> 1 THEN RAISE EXCEPTION '122 re-mark at new watermark must settle, got %', n; END IF;
    INSERT INTO public.research_lab_candidate_artifacts(candidate_id, run_id)
        VALUES ('cand-late', 'd0000000-0000-4000-8000-000000000001');
    INSERT INTO public.research_lab_candidate_promotion_events(candidate_id, private_model_version_id)
        VALUES ('cand-late', 'ab000000-0000-4000-8000-000000000009');
    INSERT INTO public.research_lab_private_model_version_events(private_model_version_id, seq)
        VALUES ('ab000000-0000-4000-8000-000000000009', 0);
    SELECT count(*) INTO n FROM public.research_lab_terminal_runs_needing_corpus(25, true);
    IF n <> 2 THEN RAISE EXCEPTION '122 late promotion/version events must force rediscovery, got %', n; END IF;
    -- And a late SCORE BUNDLE after settling again.
    PERFORM public.research_lab_mark_corpus_complete(
        public.research_lab_trajectory_id('d0000000-0000-4000-8000-000000000001'),
        'd0000000-0000-4000-8000-000000000001',
        public.research_lab_corpus_source_watermark('d0000000-0000-4000-8000-000000000001'));
    INSERT INTO public.research_evaluation_score_bundles(score_bundle_id, run_id)
        VALUES ('sb-late', 'd0000000-0000-4000-8000-000000000001');
    SELECT count(*) INTO n FROM public.research_lab_terminal_runs_needing_corpus(25, true);
    IF n <> 2 THEN RAISE EXCEPTION '122 late score bundle must force rediscovery, got %', n; END IF;

    -- 120: atomic candidate claim -- sequential claims yield DISTINCT candidates.
    INSERT INTO public.research_lab_candidate_evaluation_current
        (candidate_id, run_id, ticket_id, current_candidate_status, current_reason, current_status_at)
    VALUES
        ('cand-a','a1000000-0000-4000-8000-000000000001','b1000000-0000-4000-8000-000000000001','queued','',now()-interval '10 min'),
        ('cand-b','a1000000-0000-4000-8000-000000000002','b1000000-0000-4000-8000-000000000002','queued','',now()-interval '5 min'),
        ('cand-hold','a1000000-0000-4000-8000-000000000003','b1000000-0000-4000-8000-000000000003','queued','baseline_not_ready',now());
    SELECT candidate_id INTO c1 FROM public.claim_next_research_lab_candidate('w1',120,900,300);
    SELECT candidate_id INTO c2 FROM public.claim_next_research_lab_candidate('w2',120,900,300);
    IF c1 IS NULL OR c2 IS NULL OR c1 = c2 THEN
        RAISE EXCEPTION '120 sequential claims must be distinct, got % and %', c1, c2;
    END IF;
    IF c1 <> 'cand-a' THEN RAISE EXCEPTION '120 oldest eligible should be claimed first, got %', c1; END IF;
    -- fresh baseline_not_ready is excluded; both eligible now claimed -> none left.
    IF (SELECT count(*) FROM public.claim_next_research_lab_candidate('w3',120,900,300)) <> 0 THEN
        RAISE EXCEPTION '120 no eligible candidate should remain';
    END IF;

    -- 121: atomic run claim -- sequential claims yield DISTINCT runs, priority order.
    INSERT INTO public.research_loop_run_queue_current
        (run_id, ticket_id, queue_priority, current_event_hash, current_status_at, current_queue_status)
    VALUES
        ('c1000000-0000-4000-8000-000000000001','f1000000-0000-4000-8000-000000000001',5,'h',now()-interval '3 min','queued'),
        ('c1000000-0000-4000-8000-000000000002','f1000000-0000-4000-8000-000000000002',1,'h',now()-interval '1 min','queued');
    SELECT run_id INTO ru1 FROM public.claim_next_research_loop_run('w1',120);
    SELECT run_id INTO ru2 FROM public.claim_next_research_loop_run('w2',120);
    IF ru1 IS NULL OR ru2 IS NULL OR ru1 = ru2 THEN
        RAISE EXCEPTION '121 sequential run claims must be distinct, got % and %', ru1, ru2;
    END IF;
    -- lower queue_priority sorts first (matches the legacy order).
    IF ru1 <> 'c1000000-0000-4000-8000-000000000002' THEN
        RAISE EXCEPTION '121 priority order wrong, got %', ru1;
    END IF;

    RAISE NOTICE 'RPC-BEHAVIOR-OK';
END $$;

DO $$
DECLARE r JSONB;
        tok_a TEXT := 'research-lab-worker-1#hostx#111#aaa';
        tok_b TEXT := 'research-lab-worker-1#hostx#222#bbb';
        lease TEXT := 'hosted_worker_maintenance';
BEGIN
    -- 117: two processes, same human-readable name, DIFFERENT tokens -- only one holds.
    r := public.research_lab_acquire_maintenance_lease(lease, tok_a, 180);
    IF (r->>'acquired')::BOOL IS NOT TRUE THEN RAISE EXCEPTION '117 A should acquire'; END IF;
    r := public.research_lab_acquire_maintenance_lease(lease, tok_b, 180);
    IF (r->>'acquired')::BOOL IS NOT FALSE THEN RAISE EXCEPTION '117 B must NOT acquire while A holds'; END IF;
    r := public.research_lab_acquire_maintenance_lease(lease, tok_a, 180);
    IF (r->>'acquired')::BOOL IS NOT TRUE THEN RAISE EXCEPTION '117 A should renew'; END IF;
    UPDATE public.research_lab_maintenance_lease SET expires_at = now() - interval '1 s' WHERE lease_name = lease;
    r := public.research_lab_acquire_maintenance_lease(lease, tok_b, 180);
    IF (r->>'acquired')::BOOL IS NOT TRUE THEN RAISE EXCEPTION '117 B should take an expired lease'; END IF;
    RAISE NOTICE 'LEASE-IDENTITY-OK';
END $$;
"""


@pytest.fixture(scope="module")
def pg():
    bindir = Path(_PG_BINDIR)
    datadir = Path(tempfile.mkdtemp(prefix="egress-pg-"))
    sockdir = Path(tempfile.mkdtemp(prefix="/tmp/pgs"))
    port = _free_port()
    env = {**os.environ, "PGTZ": "UTC", "TZ": "UTC"}

    def _bin(name: str) -> str:
        return str(bindir / name)

    def psql(args, input_sql=None):
        return subprocess.run(
            [_bin("psql"), "-v", "ON_ERROR_STOP=1", "-h", "127.0.0.1",
             "-p", str(port), "-U", "postgres", "-d", "migtest", "-q", *args],
            input=input_sql, capture_output=True, text=True, env=env, timeout=120,
        )

    started = False
    try:
        subprocess.run([_bin("initdb"), "-D", str(datadir), "-U", "postgres", "--auth=trust"],
                       check=True, capture_output=True, text=True, env=env, timeout=90)
        subprocess.run(
            [_bin("pg_ctl"), "-D", str(datadir), "-w", "-t", "30",
             "-l", str(datadir / "server.log"), "-o",
             f"-p {port} -c listen_addresses=127.0.0.1 -c unix_socket_directories={sockdir}", "start"],
            check=True, capture_output=True, text=True, env=env, timeout=45)
        started = True
        subprocess.run(
            [_bin("psql"), "-h", "127.0.0.1", "-p", str(port), "-U", "postgres",
             "-d", "postgres", "-c", "CREATE DATABASE migtest"],
            check=True, capture_output=True, text=True, env=env, timeout=30)
        # Shims + apply migrations once for the module.
        setup = psql(["-c", _SETUP_SQL])
        assert setup.returncode == 0, setup.stderr
        for num in MIGRATIONS:
            matches = sorted(SCRIPTS.glob(f"{num}-*.sql"))
            assert matches, f"migration {num} not found"
            applied = psql(["-f", str(matches[0])])
            assert applied.returncode == 0, f"migration {num} failed:\n{applied.stderr}"
        yield {"psql": psql, "port": port, "env": env}
    finally:
        if started:
            subprocess.run([_bin("pg_ctl"), "-D", str(datadir), "-w", "-t", "20", "stop"],
                           capture_output=True, text=True, env=env, timeout=30)
        shutil.rmtree(datadir, ignore_errors=True)
        shutil.rmtree(sockdir, ignore_errors=True)


def test_migrations_apply_and_execute(pg) -> None:
    result = pg["psql"](["-c", _ASSERT_SQL])
    assert result.returncode == 0, result.stderr
    assert "RPC-BEHAVIOR-OK" in result.stderr
    assert "LEASE-IDENTITY-OK" in result.stderr


def test_atomic_claims_never_double_assign_under_concurrency(pg) -> None:
    # Item 12d: N threads each call the claim RPC once against the SAME live
    # server; the advisory lock + claim row must hand every thread a DISTINCT
    # candidate/run -- zero duplicates.
    # In CI the concurrency proof must RUN, not silently skip: deploy-checks
    # installs psycopg2-binary, so a missing driver there is a hard failure.
    try:
        import psycopg2
    except ImportError:
        if os.environ.get("CI"):
            pytest.fail("psycopg2 is required in CI (installed by deploy-checks) — the no-double-assign proof must not skip")
        pytest.skip("psycopg2 not installed locally")
    dsn = dict(host="127.0.0.1", port=pg["port"], user="postgres", dbname="migtest")

    seed = psycopg2.connect(**dsn)
    seed.autocommit = True
    cur = seed.cursor()
    # Fresh, distinct-named rows so this test is independent of the sequential one.
    for i in range(24):
        cur.execute(
            "INSERT INTO research_lab_candidate_evaluation_current"
            "(candidate_id,run_id,ticket_id,current_candidate_status,current_reason,current_status_at)"
            " VALUES (%s,gen_random_uuid(),gen_random_uuid(),'queued','',now()-(%s||' sec')::interval)",
            (f"cc-{i:02d}", i),
        )
        cur.execute(
            "INSERT INTO research_loop_run_queue_current"
            "(run_id,ticket_id,queue_priority,current_event_hash,current_status_at,current_queue_status)"
            " VALUES (gen_random_uuid(),gen_random_uuid(),0,'h',now()-(%s||' sec')::interval,'queued')",
            (i,),
        )
    seed.close()

    def hammer(sql: str, holder_args):
        out, lock = [], threading.Lock()
        barrier = threading.Barrier(20)

        def one(k):
            conn = psycopg2.connect(**dsn)
            conn.autocommit = True
            c = conn.cursor()
            barrier.wait()  # maximize real overlap
            c.execute(sql, (f"w{k}", *holder_args))
            row = c.fetchone()
            with lock:
                out.append(str(row[0]) if row else None)
            conn.close()

        threads = [threading.Thread(target=one, args=(k,)) for k in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return [r for r in out if r]

    cand = hammer("SELECT candidate_id FROM claim_next_research_lab_candidate(%s,120,900,300)", ())
    assert len(cand) == len(set(cand)), f"candidate double-assign: {cand}"
    assert len(cand) == 20  # 24 available, 20 claimers

    runs = hammer("SELECT run_id FROM claim_next_research_loop_run(%s,120)", ())
    assert len(runs) == len(set(runs)), f"run double-assign: {runs}"
    assert len(runs) == 20
