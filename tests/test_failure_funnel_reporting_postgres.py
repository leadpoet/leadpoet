"""PostgreSQL execution test for the private failure-funnel function."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import pytest


DOCKER = shutil.which("docker")
MIGRATION = Path("scripts/150-research-lab-failure-funnel-reporting.sql")
pytestmark = pytest.mark.skipif(DOCKER is None, reason="Docker is unavailable")

TICKET_ID = "11111111-1111-4111-8111-111111111111"
EMPTY_TICKET_ID = "22222222-2222-4222-8222-222222222222"
PARTIAL_TICKET_ID = "88888888-8888-4888-8888-888888888888"
CANDIDATE_ID = "candidate:" + "c" * 64
PARTIAL_CANDIDATE_ID = "candidate:" + "d" * 64


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.fixture(scope="module")
def database():
    psycopg2 = pytest.importorskip("psycopg2")
    port = _free_port()
    container = "failure-funnel-%s" % uuid4().hex[:12]
    started = False
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
            pytest.skip(
                "PostgreSQL container could not start: %s" % result.stderr[-300:]
            )
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

        dsn = {
            "host": "127.0.0.1",
            "port": port,
            "user": "postgres",
            "password": "postgres",
            "dbname": "postgres",
        }
        connect_deadline = time.monotonic() + 15
        while True:
            try:
                connection = psycopg2.connect(**dsn)
                break
            except psycopg2.OperationalError:
                if time.monotonic() >= connect_deadline:
                    raise
                time.sleep(0.25)
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
                    model_side TEXT NOT NULL DEFAULT 'candidate',
                    capture_state TEXT NOT NULL DEFAULT 'captured_unreviewed',
                    final_score DOUBLE PRECISION NOT NULL DEFAULT 0,
                    failure_reason TEXT,
                    failure_stage TEXT,
                    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                CREATE TABLE public.research_lab_scoring_runs (
                    scoring_run_id UUID PRIMARY KEY,
                    ticket_id UUID,
                    candidate_id TEXT,
                    run_type TEXT NOT NULL DEFAULT 'candidate_scoring',
                    run_attempt INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
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
                    failure_category TEXT,
                    telemetry_degraded BOOLEAN NOT NULL DEFAULT FALSE,
                    event_ordinal BIGINT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                GRANT SELECT ON ALL TABLES IN SCHEMA public TO service_role;
                """
            )
            cursor.execute(MIGRATION.read_text(encoding="utf-8"))
        connection.close()
        yield psycopg2, dsn
    finally:
        if started:
            subprocess.run(
                [str(DOCKER), "rm", "--force", container],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )


def _seed_report(cursor) -> None:
    bundle_doc = {
        "candidate_model_manifest_hash": "sha256:candidate-model",
        "serving_model_version": {"candidate_id": CANDIDATE_ID},
        "aggregates": {
            "per_icp_results": [
                {
                    "funnel": {
                        "sourced": 5,
                        "fit_pass": 3,
                        "verified": 2,
                        "intent_valid": 1,
                        "scored": 1,
                    }
                },
                {
                    "funnel": {
                        "sourced": 0,
                        "fit_pass": 0,
                        "verified": 0,
                        "intent_valid": 0,
                        "scored": 0,
                    }
                },
            ]
        },
    }
    cursor.execute(
        """
        INSERT INTO public.research_evaluation_score_bundles (
            score_bundle_id, run_id, ticket_id, bundle_status,
            private_model_manifest_hash, score_bundle_doc
        ) VALUES (%s, %s, %s, 'scored', %s, %s::JSONB)
        """,
        (
            "bundle-1",
            "33333333-3333-4333-8333-333333333333",
            TICKET_ID,
            "sha256:model",
            json.dumps(bundle_doc),
        ),
    )
    reasons = (
        "employee_count_mismatch",
        "company_stage_mismatch",
        "company verification failed",
        "intent_fabricated",
    )
    for index, reason in enumerate(reasons, start=1):
        cursor.execute(
            """
            INSERT INTO public.research_lab_company_label_examples (
                label_id, ticket_id, candidate_id, failure_reason
            ) VALUES (%s, %s, %s, %s)
            """,
            (
                "%08d-4444-4444-8444-444444444444" % index,
                TICKET_ID,
                CANDIDATE_ID,
                reason,
            ),
        )
    cursor.execute(
        """
        INSERT INTO public.research_lab_company_label_examples (
            label_id, ticket_id, candidate_id, model_side, failure_reason
        ) VALUES (%s, %s, %s, 'champion', 'employee_count_mismatch')
        """,
        ("00000005-4444-4444-8444-444444444444", TICKET_ID, CANDIDATE_ID),
    )
    cursor.execute(
        """
        INSERT INTO public.research_lab_scoring_runs (
            scoring_run_id, ticket_id, candidate_id
        ) VALUES (%s, %s, %s);
        INSERT INTO public.research_lab_scoring_icp_executions (
            icp_execution_id, scoring_run_id
        ) VALUES (%s, %s);
        INSERT INTO public.research_lab_scoring_icp_events (
            event_id, icp_execution_id, event_type, failure_category, event_ordinal
        ) VALUES (%s, %s, 'failed', 'provider_timeout', 1);
        """,
        (
            "55555555-5555-4555-8555-555555555555",
            TICKET_ID,
            CANDIDATE_ID,
            "66666666-6666-4666-8666-666666666666",
            "55555555-5555-4555-8555-555555555555",
            "77777777-7777-4777-8777-777777777777",
            "66666666-6666-4666-8666-666666666666",
        ),
    )


def _stage(report: dict, name: str) -> dict:
    return next(row for row in report["stages"] if row["stage"] == name)


def test_failure_funnel_executes_and_balances_stages(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        _seed_report(cursor)
        cursor.execute("SET ROLE service_role")
        cursor.execute(
            "SELECT public.get_research_lab_failure_funnel(%s, %s)",
            (TICKET_ID, CANDIDATE_ID),
        )
        report = cursor.fetchone()[0]
    connection.close()

    assert report["schema_version"] == "research_lab_failure_funnel.v1"
    assert report["telemetry"]["status"] == "complete"
    assert report["telemetry"]["company_label_count"] == 4
    assert report["model_revisions"] == ["sha256:candidate-model"]
    assert _stage(report, "sourcing") == {
        "stage": "sourcing",
        "unit": "icp_attempts",
        "reviewed": 2,
        "passed": 1,
        "rejected": 1,
    }
    assert _stage(report, "firmographic")["reviewed"] == 5
    assert _stage(report, "firmographic")["passed"] == 3
    assert _stage(report, "verifier")["rejected"] == 1
    assert _stage(report, "intent")["rejected"] == 1
    assert _stage(report, "scoring")["passed"] == 1
    assert any(
        row["stage"] == "sourcing"
        and row["reason_code"] == "provider_timeout"
        and row["unit"] == "icp_attempts"
        for row in report["rejections"]
    )


def test_failure_funnel_missing_state_and_permissions(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT public.get_research_lab_failure_funnel(%s, NULL)",
            (EMPTY_TICKET_ID,),
        )
        missing = cursor.fetchone()[0]
        cursor.execute(
            """
            SELECT
                has_function_privilege('service_role', 'public.get_research_lab_failure_funnel(uuid,text)', 'EXECUTE'),
                has_function_privilege('anon', 'public.get_research_lab_failure_funnel(uuid,text)', 'EXECUTE'),
                has_function_privilege('authenticated', 'public.get_research_lab_failure_funnel(uuid,text)', 'EXECUTE')
            """
        )
        privileges = cursor.fetchone()
    connection.close()

    assert missing["telemetry"]["status"] == "missing"
    assert missing["stages"] == []
    assert privileges == (True, False, False)


def test_failure_funnel_marks_mixed_valid_and_missing_rows_partial(database):
    psycopg2, dsn = database
    bundle_doc = {
        "serving_model_version": {"candidate_id": PARTIAL_CANDIDATE_ID},
        "aggregates": {
            "per_icp_results": [
                {
                    "funnel": {
                        "sourced": 1,
                        "fit_pass": 1,
                        "verified": 1,
                        "intent_valid": 1,
                        "scored": 1,
                    }
                },
                {
                    "funnel": {
                        "sourced": 2,
                        "fit_pass": 1,
                        "intent_valid": 0,
                        "scored": 0,
                    }
                },
            ]
        },
    }
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO public.research_evaluation_score_bundles (
                score_bundle_id, run_id, ticket_id, bundle_status,
                private_model_manifest_hash, score_bundle_doc
            ) VALUES (%s, %s, %s, 'scored', %s, %s::JSONB)
            """,
            (
                "bundle-partial",
                "99999999-9999-4999-8999-999999999999",
                PARTIAL_TICKET_ID,
                "sha256:partial",
                json.dumps(bundle_doc),
            ),
        )
        cursor.execute("SET ROLE service_role")
        cursor.execute(
            "SELECT public.get_research_lab_failure_funnel(%s, %s)",
            (PARTIAL_TICKET_ID, PARTIAL_CANDIDATE_ID),
        )
        report = cursor.fetchone()[0]
    connection.close()

    assert report["telemetry"]["status"] == "partial"
    assert report["telemetry"]["invalid_funnel_row_count"] == 1
    assert report["telemetry"]["funnel_row_count"] == 1
