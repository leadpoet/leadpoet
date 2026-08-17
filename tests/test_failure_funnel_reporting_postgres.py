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
INDEX_MIGRATION = Path(
    "scripts/150-research-lab-failure-funnel-indexes.concurrent.sql"
)
MIGRATION = Path("scripts/151-research-lab-failure-funnel-reporting.sql")
pytestmark = pytest.mark.skipif(DOCKER is None, reason="Docker is unavailable")

TICKET_ID = "11111111-1111-4111-8111-111111111111"
EMPTY_TICKET_ID = "22222222-2222-4222-8222-222222222222"
PARTIAL_TICKET_ID = "88888888-8888-4888-8888-888888888888"
INTERRUPTED_TICKET_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
MISSING_POSITIVE_TICKET_ID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
INFRA_TICKET_ID = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
ZERO_EXPECTED_TICKET_ID = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
HEARTBEAT_TICKET_ID = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
INCOMPLETE_BUNDLE_TICKET_ID = "ffffffff-ffff-4fff-8fff-ffffffffffff"
DEGRADED_HEALTH_TICKET_ID = "12121212-1212-4212-8212-121212121212"
INVALID_OUTPUT_TICKET_ID = "13131313-1313-4313-8313-131313131313"
MISSING_HEALTH_TICKET_ID = "14141414-1414-4414-8414-141414141414"
STALE_BUNDLE_TICKET_ID = "15151515-1515-4515-8515-151515151515"
DETERMINISTIC_FAILURE_TICKET_ID = "16161616-1616-4616-8616-161616161616"
RETRYABLE_FAILURE_TICKET_ID = "17171717-1717-4717-8717-171717171717"
STALE_LABEL_TICKET_ID = "18181818-1818-4818-8818-181818181818"
CANDIDATE_ID = "candidate:" + "c" * 64
PARTIAL_CANDIDATE_ID = "candidate:" + "d" * 64
INTERRUPTED_CANDIDATE_ID = "candidate:" + "e" * 64
MISSING_POSITIVE_CANDIDATE_ID = "candidate:" + "f" * 64
INFRA_CANDIDATE_ID = "candidate:" + "a" * 64
ZERO_EXPECTED_CANDIDATE_ID = "candidate:" + "b" * 64
HEARTBEAT_CANDIDATE_ID = "candidate:" + "1" * 64
INCOMPLETE_BUNDLE_CANDIDATE_ID = "candidate:" + "2" * 64
DEGRADED_HEALTH_CANDIDATE_ID = "candidate:" + "3" * 64
INVALID_OUTPUT_CANDIDATE_ID = "candidate:" + "4" * 64
MISSING_HEALTH_CANDIDATE_ID = "candidate:" + "5" * 64
STALE_BUNDLE_CANDIDATE_ID = "candidate:" + "6" * 64
DETERMINISTIC_FAILURE_CANDIDATE_ID = "candidate:" + "7" * 64
RETRYABLE_FAILURE_CANDIDATE_ID = "candidate:" + "8" * 64
STALE_LABEL_CANDIDATE_ID = "candidate:" + "9" * 64


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
                    created_at TIMESTAMPTZ NOT NULL
                        DEFAULT (NOW() - INTERVAL '1 day')
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

                GRANT SELECT ON ALL TABLES IN SCHEMA public TO service_role;
                """
            )
            indexed = subprocess.run(
                [
                    str(DOCKER),
                    "exec",
                    "--interactive",
                    container,
                    "psql",
                    "-U",
                    "postgres",
                    "--set",
                    "ON_ERROR_STOP=1",
                ],
                input=INDEX_MIGRATION.read_text(encoding="utf-8"),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if indexed.returncode != 0:
                pytest.fail(
                    "concurrent index migration failed: %s"
                    % indexed.stderr[-1000:]
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
        "candidate_model_manifest_hash": "sha256:" + "c" * 64,
        "candidate_artifact_hash": "sha256:" + "c" * 64,
        "serving_model_version": {"candidate_id": CANDIDATE_ID},
        "scoring_health": {
            "schema_version": "1.0",
            "health_status": "healthy",
            "icp_count": 2,
            "failure_class_counts": {
                "candidate_model_zero_companies": 1
            },
        },
        "aggregates": {
            "per_icp_results": [
                {
                    "icp_ref": "icp-1",
                    "funnel": {
                        "sourced": 5,
                        "fit_pass": 3,
                        "verified": 2,
                        "intent_valid": 1,
                        "scored": 1,
                    }
                },
                {
                    "icp_ref": "icp-2",
                    "failure_reason": "candidate_model_zero_companies",
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
                label_id, ticket_id, candidate_id, failure_reason, capture_doc
            ) VALUES (%s, %s, %s, %s, %s::JSONB)
            """,
            (
                "%08d-4444-4444-8444-444444444444" % index,
                TICKET_ID,
                CANDIDATE_ID,
                reason,
                json.dumps(
                    {
                        "scoring_run_id": (
                            "55555555-5555-4555-8555-555555555555"
                        )
                    }
                ),
            ),
        )
    cursor.execute(
        """
        INSERT INTO public.research_lab_company_label_examples (
            label_id, ticket_id, candidate_id, final_score, capture_doc
        ) VALUES (%s, %s, %s, 42, %s::JSONB)
        """,
        (
            "00000006-4444-4444-8444-444444444444",
            TICKET_ID,
            CANDIDATE_ID,
            json.dumps(
                {"scoring_run_id": "55555555-5555-4555-8555-555555555555"}
            ),
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
            scoring_run_id, ticket_id, candidate_id, expected_icp_count,
            score_bundle_id
        ) VALUES (%s, %s, %s, 2, 'bundle-1');
        INSERT INTO public.research_lab_scoring_icp_executions (
            icp_execution_id, scoring_run_id, icp_ref
        ) VALUES (%s, %s, 'icp-1'), (%s, %s, 'icp-2');
        INSERT INTO public.research_lab_scoring_icp_events (
            event_id, icp_execution_id, event_type, failure_category, event_ordinal
        ) VALUES
            (%s, %s, 'completed', NULL, 1),
            (%s, %s, 'completed', NULL, 1);
        """,
        (
            "55555555-5555-4555-8555-555555555555",
            TICKET_ID,
            CANDIDATE_ID,
            "66666666-6666-4666-8666-666666666666",
            "55555555-5555-4555-8555-555555555555",
            "66666667-6666-4666-8666-666666666666",
            "55555555-5555-4555-8555-555555555555",
            "77777777-7777-4777-8777-777777777777",
            "66666666-6666-4666-8666-666666666666",
            "77777778-7777-4777-8777-777777777777",
            "66666667-6666-4666-8666-666666666666",
        ),
    )


def _stage(report: dict, name: str) -> dict:
    return next(row for row in report["stages"] if row["stage"] == name)


def _insert_score_bundle(
    cursor,
    *,
    bundle_id: str,
    run_id: str,
    ticket_id: str,
    candidate_id: str,
    funnels: list[dict],
    scoring_health: dict | None = None,
    include_scoring_health: bool = True,
    failure_reasons: list[str] | None = None,
) -> None:
    reasons = list(failure_reasons or ())
    bundle_doc = {
        "candidate_model_manifest_hash": "sha256:" + "d" * 64,
        "candidate_artifact_hash": "sha256:" + "d" * 64,
        "serving_model_version": {"candidate_id": candidate_id},
        "aggregates": {
            "per_icp_results": [
                {
                    "icp_ref": f"icp-{index}",
                    "failure_reason": (
                        reasons[index - 1] if index <= len(reasons) else ""
                    ),
                    "funnel": row,
                }
                for index, row in enumerate(funnels, start=1)
            ]
        },
    }
    if include_scoring_health:
        bundle_doc["scoring_health"] = scoring_health or {
            "schema_version": "1.0",
            "health_status": "healthy",
            "icp_count": len(funnels),
            "failure_class_counts": {},
        }
    cursor.execute(
        """
        INSERT INTO public.research_evaluation_score_bundles (
            score_bundle_id, run_id, ticket_id, bundle_status,
            private_model_manifest_hash, score_bundle_doc
        ) VALUES (%s, %s, %s, 'scored', %s, %s::JSONB)
        """,
        (
            bundle_id,
            run_id,
            ticket_id,
            "sha256:" + "e" * 64,
            json.dumps(bundle_doc),
        ),
    )


def _insert_scoring_execution(
    cursor,
    *,
    scoring_run_id: str,
    ticket_id: str,
    candidate_id: str,
    expected_icp_count: int,
    execution_id: str,
    event_id: str,
    score_bundle_id: str,
) -> None:
    cursor.execute(
        """
        INSERT INTO public.research_lab_scoring_runs (
            scoring_run_id, ticket_id, candidate_id, expected_icp_count,
            score_bundle_id
        ) VALUES (%s, %s, %s, %s, %s);
        INSERT INTO public.research_lab_scoring_icp_executions (
            icp_execution_id, scoring_run_id, icp_ref
        ) VALUES (%s, %s, 'icp-1');
        INSERT INTO public.research_lab_scoring_icp_events (
            event_id, icp_execution_id, event_type, event_ordinal
        ) VALUES (%s, %s, 'completed', 1);
        """,
        (
            scoring_run_id,
            ticket_id,
            candidate_id,
            expected_icp_count,
            score_bundle_id,
            execution_id,
            scoring_run_id,
            event_id,
            execution_id,
        ),
    )


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
    assert report["telemetry"]["company_label_count"] == 5
    assert report["telemetry"]["company_positive_count"] == 1
    assert report["telemetry"]["expected_execution_count"] == 2
    assert report["telemetry"]["terminal_execution_count"] == 2
    assert report["telemetry"]["infrastructure_failure_count"] == 0
    assert report["model_revisions"] == ["sha256:" + "c" * 64]
    assert _stage(report, "sourcing") == {
        "stage": "sourcing",
        "unit": "icp_attempts",
        "reviewed": 2,
        "passed": 1,
        "rejected": 1,
    }
    assert _stage(report, "company_fit")["reviewed"] == 5
    assert _stage(report, "company_fit")["passed"] == 3
    assert _stage(report, "verifier")["rejected"] == 1
    assert _stage(report, "intent")["rejected"] == 1
    assert _stage(report, "scoring")["passed"] == 1
    assert any(
        row["stage"] == "firmographic"
        and row["reason_code"] == "employee_count_mismatch"
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


def test_failure_funnel_marks_interrupted_execution_coverage_partial(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        _insert_score_bundle(
            cursor,
            bundle_id="bundle-interrupted",
            run_id="aaaaaaaa-1111-4111-8111-111111111111",
            ticket_id=INTERRUPTED_TICKET_ID,
            candidate_id=INTERRUPTED_CANDIDATE_ID,
            funnels=[
                {"sourced": 0, "fit_pass": 0, "verified": 0, "intent_valid": 0, "scored": 0},
                {"sourced": 0, "fit_pass": 0, "verified": 0, "intent_valid": 0, "scored": 0},
            ],
        )
        _insert_scoring_execution(
            cursor,
            scoring_run_id="aaaaaaaa-2222-4222-8222-222222222222",
            ticket_id=INTERRUPTED_TICKET_ID,
            candidate_id=INTERRUPTED_CANDIDATE_ID,
            expected_icp_count=2,
            execution_id="aaaaaaaa-3333-4333-8333-333333333333",
            event_id="aaaaaaaa-4444-4444-8444-444444444444",
            score_bundle_id="bundle-interrupted",
        )
        cursor.execute("SET ROLE service_role")
        cursor.execute(
            "SELECT public.get_research_lab_failure_funnel(%s, %s)",
            (INTERRUPTED_TICKET_ID, INTERRUPTED_CANDIDATE_ID),
        )
        report = cursor.fetchone()[0]
    connection.close()

    assert report["telemetry"]["status"] == "partial"
    assert report["telemetry"]["expected_execution_count"] == 2
    assert report["telemetry"]["terminal_execution_count"] == 1


def test_failure_funnel_does_not_certify_a_zero_expected_run(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        _insert_score_bundle(
            cursor,
            bundle_id="bundle-zero-expected",
            run_id="dddddddd-1111-4111-8111-111111111111",
            ticket_id=ZERO_EXPECTED_TICKET_ID,
            candidate_id=ZERO_EXPECTED_CANDIDATE_ID,
            funnels=[
                {
                    "sourced": 0,
                    "fit_pass": 0,
                    "verified": 0,
                    "intent_valid": 0,
                    "scored": 0,
                }
            ],
        )
        cursor.execute(
            """
            INSERT INTO public.research_lab_scoring_runs (
                scoring_run_id, ticket_id, candidate_id, expected_icp_count,
                score_bundle_id
            ) VALUES (%s, %s, %s, 0, 'bundle-zero-expected')
            """,
            (
                "dddddddd-2222-4222-8222-222222222222",
                ZERO_EXPECTED_TICKET_ID,
                ZERO_EXPECTED_CANDIDATE_ID,
            ),
        )
        cursor.execute("SET ROLE service_role")
        cursor.execute(
            "SELECT public.get_research_lab_failure_funnel(%s, %s)",
            (ZERO_EXPECTED_TICKET_ID, ZERO_EXPECTED_CANDIDATE_ID),
        )
        report = cursor.fetchone()[0]
    connection.close()

    assert report["telemetry"]["status"] == "partial"
    assert report["telemetry"]["expected_execution_count"] == 0
    assert report["telemetry"]["terminal_execution_count"] == 0


def test_failure_funnel_uses_event_time_not_heartbeat_ordinal(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        _insert_score_bundle(
            cursor,
            bundle_id="bundle-heartbeat-order",
            run_id="eeeeeeee-1111-4111-8111-111111111111",
            ticket_id=HEARTBEAT_TICKET_ID,
            candidate_id=HEARTBEAT_CANDIDATE_ID,
            funnels=[
                {
                    "sourced": 0,
                    "fit_pass": 0,
                    "verified": 0,
                    "intent_valid": 0,
                    "scored": 0,
                }
            ],
        )
        cursor.execute(
            """
            INSERT INTO public.research_lab_scoring_runs (
                scoring_run_id, ticket_id, candidate_id, expected_icp_count,
                score_bundle_id
            ) VALUES (%s, %s, %s, 1, 'bundle-heartbeat-order');
            INSERT INTO public.research_lab_scoring_icp_executions (
                icp_execution_id, scoring_run_id, icp_ref
            ) VALUES (%s, %s, 'icp-1');
            INSERT INTO public.research_lab_scoring_icp_events (
                event_id, icp_execution_id, event_type, event_ordinal,
                occurred_at
            ) VALUES
                (%s, %s, 'heartbeat', 999999,
                    '2026-08-16T10:00:00+00:00'),
                (%s, %s, 'completed', 0,
                    '2026-08-16T10:01:00+00:00');
            """,
            (
                "eeeeeeee-2222-4222-8222-222222222222",
                HEARTBEAT_TICKET_ID,
                HEARTBEAT_CANDIDATE_ID,
                "eeeeeeee-3333-4333-8333-333333333333",
                "eeeeeeee-2222-4222-8222-222222222222",
                "eeeeeeee-4444-4444-8444-444444444444",
                "eeeeeeee-3333-4333-8333-333333333333",
                "eeeeeeee-5555-4555-8555-555555555555",
                "eeeeeeee-3333-4333-8333-333333333333",
            ),
        )
        cursor.execute("SET ROLE service_role")
        cursor.execute(
            "SELECT public.get_research_lab_failure_funnel(%s, %s)",
            (HEARTBEAT_TICKET_ID, HEARTBEAT_CANDIDATE_ID),
        )
        report = cursor.fetchone()[0]
    connection.close()

    assert report["telemetry"]["status"] == "complete"
    assert report["telemetry"]["terminal_execution_count"] == 1
    assert report["telemetry"]["nonterminal_execution_count"] == 0


def test_failure_funnel_requires_one_bundle_row_per_expected_icp(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        _insert_score_bundle(
            cursor,
            bundle_id="bundle-incomplete-icp-coverage",
            run_id="ffffffff-1111-4111-8111-111111111111",
            ticket_id=INCOMPLETE_BUNDLE_TICKET_ID,
            candidate_id=INCOMPLETE_BUNDLE_CANDIDATE_ID,
            funnels=[
                {
                    "sourced": 0,
                    "fit_pass": 0,
                    "verified": 0,
                    "intent_valid": 0,
                    "scored": 0,
                }
            ],
        )
        cursor.execute(
            """
            INSERT INTO public.research_lab_scoring_runs (
                scoring_run_id, ticket_id, candidate_id, expected_icp_count,
                score_bundle_id
            ) VALUES (%s, %s, %s, 2, 'bundle-incomplete-icp-coverage');
            INSERT INTO public.research_lab_scoring_icp_executions (
                icp_execution_id, scoring_run_id, icp_ref
            ) VALUES
                (%s, %s, 'icp-1'),
                (%s, %s, 'icp-2');
            INSERT INTO public.research_lab_scoring_icp_events (
                event_id, icp_execution_id, event_type, event_ordinal
            ) VALUES
                (%s, %s, 'completed', 0),
                (%s, %s, 'completed', 0);
            """,
            (
                "ffffffff-2222-4222-8222-222222222222",
                INCOMPLETE_BUNDLE_TICKET_ID,
                INCOMPLETE_BUNDLE_CANDIDATE_ID,
                "ffffffff-3333-4333-8333-333333333333",
                "ffffffff-2222-4222-8222-222222222222",
                "ffffffff-4444-4444-8444-444444444444",
                "ffffffff-2222-4222-8222-222222222222",
                "ffffffff-5555-4555-8555-555555555555",
                "ffffffff-3333-4333-8333-333333333333",
                "ffffffff-6666-4666-8666-666666666666",
                "ffffffff-4444-4444-8444-444444444444",
            ),
        )
        cursor.execute("SET ROLE service_role")
        cursor.execute(
            "SELECT public.get_research_lab_failure_funnel(%s, %s)",
            (INCOMPLETE_BUNDLE_TICKET_ID, INCOMPLETE_BUNDLE_CANDIDATE_ID),
        )
        report = cursor.fetchone()[0]
    connection.close()

    assert report["telemetry"]["status"] == "partial"
    assert report["telemetry"]["expected_execution_count"] == 2
    assert report["telemetry"]["terminal_execution_count"] == 2
    assert report["telemetry"]["icp_row_count"] == 1
    assert report["telemetry"]["funnel_row_count"] == 1
    assert report["telemetry"]["coverage_mismatch_count"] > 0


@pytest.mark.parametrize(
    (
        "ticket_id",
        "candidate_id",
        "bundle_id",
        "run_id",
        "scoring_run_id",
        "execution_id",
        "event_id",
        "scoring_health",
        "include_scoring_health",
        "expected_status",
        "expected_invalid",
        "expected_degraded",
    ),
    [
        (
            DEGRADED_HEALTH_TICKET_ID,
            DEGRADED_HEALTH_CANDIDATE_ID,
            "bundle-degraded-health",
            "12121212-1111-4111-8111-111111111111",
            "12121212-2222-4222-8222-222222222222",
            "12121212-3333-4333-8333-333333333333",
            "12121212-4444-4444-8444-444444444444",
            {
                "schema_version": "1.0",
                "health_status": "degraded",
                "icp_count": 1,
                "provider_error_count": 1,
                "failure_class_counts": {
                    "candidate_model_runtime_provider_error": 1,
                    "candidate_model_zero_companies": 1,
                },
            },
            True,
            "partial",
            0,
            1,
        ),
        (
            INVALID_OUTPUT_TICKET_ID,
            INVALID_OUTPUT_CANDIDATE_ID,
            "bundle-invalid-output",
            "13131313-1111-4111-8111-111111111111",
            "13131313-2222-4222-8222-222222222222",
            "13131313-3333-4333-8333-333333333333",
            "13131313-4444-4444-8444-444444444444",
            {
                "schema_version": "1.0",
                "health_status": "healthy",
                "icp_count": 1,
                "candidate_runtime_failure_count": 1,
                "invalid_output_count": 1,
                "failure_class_counts": {
                    "candidate_model_runtime_invalid_output": 1,
                    "candidate_model_zero_companies": 1,
                },
            },
            True,
            "complete",
            0,
            0,
        ),
        (
            MISSING_HEALTH_TICKET_ID,
            MISSING_HEALTH_CANDIDATE_ID,
            "bundle-missing-health",
            "14141414-1111-4111-8111-111111111111",
            "14141414-2222-4222-8222-222222222222",
            "14141414-3333-4333-8333-333333333333",
            "14141414-4444-4444-8444-444444444444",
            None,
            False,
            "partial",
            1,
            0,
        ),
    ],
)
def test_failure_funnel_requires_valid_healthy_signed_scoring_health(
    database,
    ticket_id,
    candidate_id,
    bundle_id,
    run_id,
    scoring_run_id,
    execution_id,
    event_id,
    scoring_health,
    include_scoring_health,
    expected_status,
    expected_invalid,
    expected_degraded,
):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        _insert_score_bundle(
            cursor,
            bundle_id=bundle_id,
            run_id=run_id,
            ticket_id=ticket_id,
            candidate_id=candidate_id,
            funnels=[
                {
                    "sourced": 0,
                    "fit_pass": 0,
                    "verified": 0,
                    "intent_valid": 0,
                    "scored": 0,
                }
            ],
            scoring_health=scoring_health,
            include_scoring_health=include_scoring_health,
            failure_reasons=(
                ["candidate_model_runtime_provider_error;candidate_model_zero_companies"]
                if ticket_id == DEGRADED_HEALTH_TICKET_ID
                else ["candidate_model_runtime_invalid_output;candidate_model_zero_companies"]
                if ticket_id == INVALID_OUTPUT_TICKET_ID
                else ["candidate_model_zero_companies"]
            ),
        )
        _insert_scoring_execution(
            cursor,
            scoring_run_id=scoring_run_id,
            ticket_id=ticket_id,
            candidate_id=candidate_id,
            expected_icp_count=1,
            execution_id=execution_id,
            event_id=event_id,
            score_bundle_id=bundle_id,
        )
        cursor.execute("SET ROLE service_role")
        cursor.execute(
            "SELECT public.get_research_lab_failure_funnel(%s, %s)",
            (ticket_id, candidate_id),
        )
        report = cursor.fetchone()[0]
    connection.close()

    assert report["telemetry"]["status"] == expected_status
    assert report["telemetry"]["invalid_scoring_health_count"] == expected_invalid
    assert report["telemetry"]["degraded_scoring_health_count"] == expected_degraded
    assert report["telemetry"]["scoring_health_icp_count"] == (
        0 if expected_invalid else 1
    )
    if ticket_id == DEGRADED_HEALTH_TICKET_ID:
        assert any(
            row["stage"] == "infrastructure"
            and row["reason_code"] == "candidate_model_runtime_provider_error"
            and row["count"] == 1
            for row in report["rejections"]
        )
    if ticket_id == INVALID_OUTPUT_TICKET_ID:
        assert any(
            row["stage"] == "scoring"
            and row["reason_code"] == "candidate_model_runtime_invalid_output"
            and row["count"] == 1
            for row in report["rejections"]
        )


def test_failure_funnel_does_not_pair_new_run_with_stale_bundle(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        _insert_score_bundle(
            cursor,
            bundle_id="bundle-stale-pair",
            run_id="15151515-1111-4111-8111-111111111111",
            ticket_id=STALE_BUNDLE_TICKET_ID,
            candidate_id=STALE_BUNDLE_CANDIDATE_ID,
            funnels=[
                {
                    "sourced": 0,
                    "fit_pass": 0,
                    "verified": 0,
                    "intent_valid": 0,
                    "scored": 0,
                }
            ],
            failure_reasons=["candidate_model_zero_companies"],
        )
        cursor.execute(
            """
            INSERT INTO public.research_lab_scoring_runs (
                scoring_run_id, ticket_id, candidate_id, expected_icp_count,
                score_bundle_id, created_at
            ) VALUES
                (%s, %s, %s, 1, 'bundle-stale-pair',
                    '2026-08-16T10:00:00+00:00'),
                (%s, %s, %s, 1, NULL,
                    '2026-08-16T10:02:00+00:00');
            INSERT INTO public.research_lab_scoring_icp_executions (
                icp_execution_id, scoring_run_id, icp_ref
            ) VALUES
                (%s, %s, 'icp-1'),
                (%s, %s, 'icp-1');
            INSERT INTO public.research_lab_scoring_icp_events (
                event_id, icp_execution_id, event_type, event_ordinal,
                occurred_at
            ) VALUES
                (%s, %s, 'completed', 0,
                    '2026-08-16T10:01:00+00:00'),
                (%s, %s, 'completed', 0,
                    '2026-08-16T10:03:00+00:00');
            """,
            (
                "15151515-2222-4222-8222-222222222222",
                STALE_BUNDLE_TICKET_ID,
                STALE_BUNDLE_CANDIDATE_ID,
                "15151515-3333-4333-8333-333333333333",
                STALE_BUNDLE_TICKET_ID,
                STALE_BUNDLE_CANDIDATE_ID,
                "15151515-4444-4444-8444-444444444444",
                "15151515-2222-4222-8222-222222222222",
                "15151515-5555-4555-8555-555555555555",
                "15151515-3333-4333-8333-333333333333",
                "15151515-6666-4666-8666-666666666666",
                "15151515-4444-4444-8444-444444444444",
                "15151515-7777-4777-8777-777777777777",
                "15151515-5555-4555-8555-555555555555",
            ),
        )
        cursor.execute("SET ROLE service_role")
        cursor.execute(
            "SELECT public.get_research_lab_failure_funnel(%s, %s)",
            (STALE_BUNDLE_TICKET_ID, STALE_BUNDLE_CANDIDATE_ID),
        )
        report = cursor.fetchone()[0]
    connection.close()

    assert report["telemetry"]["status"] == "partial"
    assert report["telemetry"]["expected_execution_count"] == 1
    assert report["telemetry"]["terminal_execution_count"] == 1
    assert report["telemetry"]["coverage_mismatch_count"] > 0


def test_failure_funnel_does_not_reuse_labels_from_an_older_run(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        _insert_score_bundle(
            cursor,
            bundle_id="bundle-stale-label",
            run_id="18181818-1111-4111-8111-111111111111",
            ticket_id=STALE_LABEL_TICKET_ID,
            candidate_id=STALE_LABEL_CANDIDATE_ID,
            funnels=[
                {
                    "sourced": 1,
                    "fit_pass": 0,
                    "verified": 0,
                    "intent_valid": 0,
                    "scored": 0,
                }
            ],
        )
        cursor.execute(
            """
            INSERT INTO public.research_lab_company_label_examples (
                label_id, ticket_id, candidate_id, icp_ref, failure_reason,
                captured_at, capture_doc
            ) VALUES (
                %s, %s, %s, 'icp-1', 'employee_count_mismatch',
                '2026-08-16T11:00:00+00:00', %s::JSONB
            );
            INSERT INTO public.research_lab_scoring_runs (
                scoring_run_id, ticket_id, candidate_id, expected_icp_count,
                score_bundle_id, created_at
            ) VALUES (
                %s, %s, %s, 1, 'bundle-stale-label',
                '2026-08-16T10:00:00+00:00'
            );
            INSERT INTO public.research_lab_scoring_icp_executions (
                icp_execution_id, scoring_run_id, icp_ref
            ) VALUES (%s, %s, 'icp-1');
            INSERT INTO public.research_lab_scoring_icp_events (
                event_id, icp_execution_id, event_type, event_ordinal,
                occurred_at
            ) VALUES (
                %s, %s, 'completed', 0,
                '2026-08-16T10:01:00+00:00'
            );
            """,
            (
                "18181818-2222-4222-8222-222222222222",
                STALE_LABEL_TICKET_ID,
                STALE_LABEL_CANDIDATE_ID,
                json.dumps(
                    {"scoring_run_id": "18181818-aaaaaaaa-4aaa-8aaa-aaaaaaaaaaaa"}
                ),
                "18181818-3333-4333-8333-333333333333",
                STALE_LABEL_TICKET_ID,
                STALE_LABEL_CANDIDATE_ID,
                "18181818-4444-4444-8444-444444444444",
                "18181818-3333-4333-8333-333333333333",
                "18181818-5555-4555-8555-555555555555",
                "18181818-4444-4444-8444-444444444444",
            ),
        )
        cursor.execute("SET ROLE service_role")
        cursor.execute(
            "SELECT public.get_research_lab_failure_funnel(%s, %s)",
            (STALE_LABEL_TICKET_ID, STALE_LABEL_CANDIDATE_ID),
        )
        report = cursor.fetchone()[0]
    connection.close()

    assert report["telemetry"]["status"] == "partial"
    assert report["telemetry"]["company_label_count"] == 0
    assert report["telemetry"]["detailed_reason_gap_count"] == 1
    assert not any(
        row["reason_code"] == "employee_count_mismatch"
        for row in report["rejections"]
    )


@pytest.mark.parametrize(
    (
        "ticket_id",
        "candidate_id",
        "bundle_id",
        "run_id",
        "scoring_run_id",
        "execution_id",
        "event_id",
        "retryable",
        "failure_category",
        "expected_stage",
        "expected_infrastructure_count",
    ),
    [
        (
            DETERMINISTIC_FAILURE_TICKET_ID,
            DETERMINISTIC_FAILURE_CANDIDATE_ID,
            "bundle-deterministic-failure",
            "16161616-1111-4111-8111-111111111111",
            "16161616-2222-4222-8222-222222222222",
            "16161616-3333-4333-8333-333333333333",
            "16161616-4444-4444-8444-444444444444",
            False,
            "candidate_runtime_error",
            "scoring",
            0,
        ),
        (
            RETRYABLE_FAILURE_TICKET_ID,
            RETRYABLE_FAILURE_CANDIDATE_ID,
            "bundle-retryable-failure",
            "17171717-1111-4111-8111-111111111111",
            "17171717-2222-4222-8222-222222222222",
            "17171717-3333-4333-8333-333333333333",
            "17171717-4444-4444-8444-444444444444",
            True,
            "conditional_validation_retryable_failure",
            "infrastructure",
            1,
        ),
    ],
)
def test_failure_funnel_uses_authoritative_execution_retryability(
    database,
    ticket_id,
    candidate_id,
    bundle_id,
    run_id,
    scoring_run_id,
    execution_id,
    event_id,
    retryable,
    failure_category,
    expected_stage,
    expected_infrastructure_count,
):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        _insert_score_bundle(
            cursor,
            bundle_id=bundle_id,
            run_id=run_id,
            ticket_id=ticket_id,
            candidate_id=candidate_id,
            funnels=[
                {
                    "sourced": 0,
                    "fit_pass": 0,
                    "verified": 0,
                    "intent_valid": 0,
                    "scored": 0,
                }
            ],
            failure_reasons=["candidate_model_zero_companies"],
        )
        cursor.execute(
            """
            INSERT INTO public.research_lab_scoring_runs (
                scoring_run_id, ticket_id, candidate_id, expected_icp_count,
                score_bundle_id
            ) VALUES (%s, %s, %s, 1, %s);
            INSERT INTO public.research_lab_scoring_icp_executions (
                icp_execution_id, scoring_run_id, icp_ref
            ) VALUES (%s, %s, 'icp-1');
            INSERT INTO public.research_lab_scoring_icp_events (
                event_id, icp_execution_id, event_type, retryable,
                failure_category, event_ordinal
            ) VALUES (%s, %s, 'failed', %s, %s, 0);
            """,
            (
                scoring_run_id,
                ticket_id,
                candidate_id,
                bundle_id,
                execution_id,
                scoring_run_id,
                event_id,
                execution_id,
                retryable,
                failure_category,
            ),
        )
        cursor.execute("SET ROLE service_role")
        cursor.execute(
            "SELECT public.get_research_lab_failure_funnel(%s, %s)",
            (ticket_id, candidate_id),
        )
        report = cursor.fetchone()[0]
    connection.close()

    assert report["telemetry"]["status"] == "partial"
    assert report["telemetry"]["infrastructure_failure_count"] == (
        expected_infrastructure_count
    )
    assert any(
        row["stage"] == expected_stage
        and row["reason_code"] == failure_category
        and row["unit"] == "icp_attempts"
        for row in report["rejections"]
    )


def test_failure_funnel_marks_missing_positive_capture_partial(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        _insert_score_bundle(
            cursor,
            bundle_id="bundle-missing-positive",
            run_id="bbbbbbbb-1111-4111-8111-111111111111",
            ticket_id=MISSING_POSITIVE_TICKET_ID,
            candidate_id=MISSING_POSITIVE_CANDIDATE_ID,
            funnels=[
                {"sourced": 1, "fit_pass": 1, "verified": 1, "intent_valid": 1, "scored": 1}
            ],
        )
        _insert_scoring_execution(
            cursor,
            scoring_run_id="bbbbbbbb-2222-4222-8222-222222222222",
            ticket_id=MISSING_POSITIVE_TICKET_ID,
            candidate_id=MISSING_POSITIVE_CANDIDATE_ID,
            expected_icp_count=1,
            execution_id="bbbbbbbb-3333-4333-8333-333333333333",
            event_id="bbbbbbbb-4444-4444-8444-444444444444",
            score_bundle_id="bundle-missing-positive",
        )
        cursor.execute("SET ROLE service_role")
        cursor.execute(
            "SELECT public.get_research_lab_failure_funnel(%s, %s)",
            (MISSING_POSITIVE_TICKET_ID, MISSING_POSITIVE_CANDIDATE_ID),
        )
        report = cursor.fetchone()[0]
    connection.close()

    assert report["telemetry"]["status"] == "partial"
    assert report["telemetry"]["company_positive_count"] == 0
    assert report["telemetry"]["detailed_pass_gap_count"] == 1
    assert report["telemetry"]["company_label_gap_count"] == 1


def test_failure_funnel_preserves_explicit_stage_and_authoritative_infrastructure(database):
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    with connection.cursor() as cursor:
        _insert_score_bundle(
            cursor,
            bundle_id="bundle-infrastructure",
            run_id="cccccccc-1111-4111-8111-111111111111",
            ticket_id=INFRA_TICKET_ID,
            candidate_id=INFRA_CANDIDATE_ID,
            funnels=[
                {"sourced": 2, "fit_pass": 0, "verified": 0, "intent_valid": 0, "scored": 0}
            ],
        )
        cursor.execute(
            """
            INSERT INTO public.research_lab_company_label_examples (
                label_id, ticket_id, candidate_id, final_score,
                failure_reason, failure_stage, capture_doc
            ) VALUES (%s, %s, %s, 0, %s, 'verifier', %s::JSONB)
            """,
            (
                "cccccccc-5555-4555-8555-555555555555",
                INFRA_TICKET_ID,
                INFRA_CANDIDATE_ID,
                "company verification failed",
                json.dumps(
                    {
                        "retryable_infrastructure_failure": True,
                        "scoring_run_id": (
                            "cccccccc-2222-4222-8222-222222222222"
                        ),
                    }
                ),
            ),
        )
        cursor.execute(
            """
            INSERT INTO public.research_lab_company_label_examples (
                label_id, ticket_id, candidate_id, final_score,
                failure_reason, failure_stage, capture_doc
            ) VALUES (%s, %s, %s, 0, %s, 'candidate_runtime_error', %s::JSONB)
            """,
            (
                "cccccccc-6666-4666-8666-666666666666",
                INFRA_TICKET_ID,
                INFRA_CANDIDATE_ID,
                "candidate returned deterministic invalid output",
                json.dumps(
                    {
                        "retryable_infrastructure_failure": False,
                        "scoring_run_id": (
                            "cccccccc-2222-4222-8222-222222222222"
                        ),
                    }
                ),
            ),
        )
        _insert_scoring_execution(
            cursor,
            scoring_run_id="cccccccc-2222-4222-8222-222222222222",
            ticket_id=INFRA_TICKET_ID,
            candidate_id=INFRA_CANDIDATE_ID,
            expected_icp_count=1,
            execution_id="cccccccc-3333-4333-8333-333333333333",
            event_id="cccccccc-4444-4444-8444-444444444444",
            score_bundle_id="bundle-infrastructure",
        )
        cursor.execute("SET ROLE service_role")
        cursor.execute(
            "SELECT public.get_research_lab_failure_funnel(%s, %s)",
            (INFRA_TICKET_ID, INFRA_CANDIDATE_ID),
        )
        report = cursor.fetchone()[0]
    connection.close()

    assert report["telemetry"]["status"] == "partial"
    assert report["telemetry"]["infrastructure_failure_count"] == 1
    assert any(
        row["stage"] == "verifier"
        and row["reason_code"] == "infrastructure_failure"
        and row["count"] == 1
        for row in report["rejections"]
    )
    assert any(
        row["stage"] == "candidate_runtime_error"
        and row["reason_code"] == "other"
        and row["count"] == 1
        for row in report["rejections"]
    )


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
