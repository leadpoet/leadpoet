"""Disposable PostgreSQL checks for the public-baseline state table."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_source_add_end_to_end_postgres import (
    _database_with_migrations,
)


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = "180-public-baseline-rebenchmark.sql"
PROTOTYPE_SCHEMA = """
CREATE TABLE public.research_lab_daily_rebenchmarks (
    run_id UUID PRIMARY KEY,
    benchmark_date DATE NOT NULL,
    baseline_id TEXT NOT NULL,
    baseline_repository TEXT NOT NULL,
    baseline_entrypoint TEXT NOT NULL,
    rolling_window_hash TEXT NOT NULL,
    window_doc JSONB NOT NULL DEFAULT '{}'::jsonb,
    evaluation_epoch BIGINT,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    expected_icp_count INTEGER NOT NULL CHECK (expected_icp_count > 0),
    completed_icp_count INTEGER NOT NULL DEFAULT 0 CHECK (completed_icp_count >= 0),
    aggregate_score DOUBLE PRECISION,
    per_icp_results JSONB NOT NULL DEFAULT '[]'::jsonb,
    usage_doc JSONB NOT NULL DEFAULT '{}'::jsonb,
    score_summary_doc JSONB NOT NULL DEFAULT '{}'::jsonb,
    public_report_doc JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_doc JSONB NOT NULL DEFAULT '{}'::jsonb,
    worker_ref TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    UNIQUE (benchmark_date, baseline_id, rolling_window_hash)
);
INSERT INTO public.research_lab_daily_rebenchmarks (
    run_id, benchmark_date, baseline_id, baseline_repository,
    baseline_entrypoint, rolling_window_hash, status,
    expected_icp_count, worker_ref
) VALUES (
    '00000000-0000-0000-0000-000000000180', DATE '2026-09-02',
    'leadpoet/pydantic-harness',
    'https://github.com/leadpoet/pydantic-harness.git',
    'harness.run_icp', 'sha256:prototype', 'running', 20, 'prototype-worker'
);
"""


@pytest.fixture(scope="module")
def database():
    yield from _database_with_migrations((MIGRATION,))


@pytest.fixture(scope="module")
def prototype_database():
    yield from _database_with_migrations(
        (MIGRATION,),
        setup_sql=PROTOTYPE_SCHEMA,
    )


def test_migration_upgrades_the_prototype_daily_identity(prototype_database) -> None:
    psycopg2, dsn = prototype_database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name
                  FROM information_schema.columns
                 WHERE table_schema = 'public'
                   AND table_name = 'research_lab_daily_rebenchmarks'
                   AND column_name = 'rolling_window_hash'
                """
            )
            assert cursor.fetchone() is None
            cursor.execute(
                """
                SELECT attempt_count, benchmark_input_doc
                  FROM public.research_lab_daily_rebenchmarks
                 WHERE run_id = '00000000-0000-0000-0000-000000000180'
                """
            )
            assert cursor.fetchone() == (1, {})
            with pytest.raises(psycopg2.errors.UniqueViolation):
                cursor.execute(
                    """
                    INSERT INTO public.research_lab_daily_rebenchmarks (
                        run_id, benchmark_date, baseline_id,
                        baseline_repository, baseline_entrypoint,
                        status, expected_icp_count, worker_ref
                    ) VALUES (
                        '00000000-0000-0000-0000-000000000181',
                        DATE '2026-09-02', 'leadpoet/pydantic-harness',
                        'https://github.com/leadpoet/pydantic-harness.git',
                        'harness.run_icp', 'running', 20, 'new-worker'
                    )
                    """
                )
    finally:
        connection.close()


def test_public_baseline_migration_is_repeatable_and_private(database) -> None:
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "GRANT DELETE ON public.research_lab_daily_rebenchmarks TO service_role"
            )
            cursor.execute(
                (ROOT / "scripts" / MIGRATION).read_text(encoding="utf-8")
            )
            cursor.execute(
                """
                SELECT relrowsecurity
                FROM pg_class
                WHERE oid = 'public.research_lab_daily_rebenchmarks'::regclass
                """
            )
            assert cursor.fetchone() == (True,)
            for role in ("anon", "authenticated"):
                cursor.execute(
                    "SELECT has_table_privilege(%s, %s, 'SELECT')",
                    (role, "public.research_lab_daily_rebenchmarks"),
                )
                assert cursor.fetchone() == (False,)
            for privilege in ("SELECT", "INSERT", "UPDATE"):
                cursor.execute(
                    "SELECT has_table_privilege(%s, %s, %s)",
                    (
                        "service_role",
                        "public.research_lab_daily_rebenchmarks",
                        privilege,
                    ),
                )
                assert cursor.fetchone() == (True,)
            cursor.execute(
                "SELECT has_table_privilege(%s, %s, 'DELETE')",
                ("service_role", "public.research_lab_daily_rebenchmarks"),
            )
            assert cursor.fetchone() == (False,)
    finally:
        connection.close()


def test_completed_row_requires_full_ordinary_result(database) -> None:
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            with pytest.raises(psycopg2.errors.CheckViolation):
                cursor.execute(
                    """
                    INSERT INTO public.research_lab_daily_rebenchmarks (
                        run_id, benchmark_date, baseline_id,
                        baseline_repository, baseline_entrypoint,
                        status, expected_icp_count,
                        completed_icp_count, aggregate_score, per_icp_results,
                        score_summary_doc, public_report_doc, worker_ref, completed_at
                    ) VALUES (
                        gen_random_uuid(), DATE '2026-09-03', 'baseline',
                        'https://github.com/leadpoet/pydantic-harness',
                        'harness.run_icp', 'completed', 20, 20, 50.0,
                        (
                          SELECT jsonb_agg(jsonb_build_object(
                            'icp_ref', 'icp-' || value::text,
                            'status', 'completed'
                          ))
                          FROM generate_series(1, 19) AS value
                        ),
                        '{"aggregate_score":50}'::jsonb,
                        '{"aggregate_score":50}'::jsonb,
                        'test', now()
                    )
                    """
                )
            cursor.execute(
                """
                INSERT INTO public.research_lab_daily_rebenchmarks (
                    run_id, benchmark_date, baseline_id,
                    baseline_repository, baseline_entrypoint,
                    status, expected_icp_count,
                    completed_icp_count, aggregate_score, per_icp_results,
                    score_summary_doc, public_report_doc, worker_ref, completed_at
                ) VALUES (
                    gen_random_uuid(), DATE '2026-09-03', 'baseline',
                    'https://github.com/leadpoet/pydantic-harness',
                    'harness.run_icp', 'completed', 20, 20, 50.0,
                    (
                      SELECT jsonb_agg(jsonb_build_object(
                        'icp_ref', 'icp-' || value::text,
                        'status', 'completed'
                      ))
                      FROM generate_series(1, 20) AS value
                    ),
                    '{"aggregate_score":50}'::jsonb,
                    '{"aggregate_score":50}'::jsonb,
                    'test', now()
                )
                RETURNING completed_icp_count, jsonb_array_length(per_icp_results)
                """
            )
            assert cursor.fetchone() == (20, 20)
    finally:
        connection.close()


def test_failed_run_has_only_one_fenced_whole_run_retry(database) -> None:
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO public.research_lab_daily_rebenchmarks (
                    run_id, benchmark_date, baseline_id,
                    baseline_repository, baseline_entrypoint,
                    status, expected_icp_count, completed_icp_count,
                    per_icp_results, error_doc, worker_ref, completed_at
                ) VALUES (
                    '11111111-1111-1111-1111-111111111111',
                    DATE '2026-09-04', 'baseline',
                    'https://github.com/leadpoet/pydantic-harness',
                    'harness.run_icp', 'failed', 20, 1,
                    '[{"icp_ref":"icp-1","status":"failed"}]'::jsonb,
                    '{"code":"baseline_run_failed"}'::jsonb,
                    'failed-worker', now()
                )
                """
            )
            cursor.execute(
                """
                SELECT public.research_lab_retry_daily_rebenchmark(
                    '11111111-1111-1111-1111-111111111111',
                    1, 'retry-claim', 'retry-worker', 300
                )
                """
            )
            retry = cursor.fetchone()[0]
            assert retry["retry_status"] == "retried"
            assert retry["run"]["attempt_count"] == 2
            assert retry["run"]["status"] == "running"
            assert retry["run"]["per_icp_results"] == []

            cursor.execute(
                """
                SELECT public.research_lab_retry_daily_rebenchmark(
                    '11111111-1111-1111-1111-111111111111',
                    1, 'stale-claim', 'stale-worker', 300
                )
                """
            )
            assert cursor.fetchone()[0] == {"retry_status": "stale"}

            cursor.execute(
                """
                UPDATE public.research_lab_daily_rebenchmarks
                   SET status = 'failed',
                       error_doc = '{"code":"baseline_run_failed"}'::jsonb,
                       claim_token = '',
                       lease_expires_at = NULL,
                       completed_at = now()
                 WHERE run_id = '11111111-1111-1111-1111-111111111111'
                """
            )
            cursor.execute(
                """
                SELECT public.research_lab_retry_daily_rebenchmark(
                    '11111111-1111-1111-1111-111111111111',
                    2, 'third-claim', 'third-worker', 300
                )
                """
            )
            assert cursor.fetchone()[0] == {"retry_status": "exhausted"}
    finally:
        connection.close()


def test_daily_claim_uses_one_retry_then_exhausts_after_two_expiries(database) -> None:
    psycopg2, dsn = database
    connection = psycopg2.connect(**dsn)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO public.research_lab_daily_rebenchmarks (
                    run_id, benchmark_date, baseline_id,
                    baseline_repository, baseline_entrypoint,
                    status, expected_icp_count, completed_icp_count,
                    per_icp_results, usage_doc, worker_ref,
                    claim_token, lease_expires_at
                ) VALUES (
                    '22222222-2222-2222-2222-222222222222',
                    DATE '2026-09-05', 'baseline',
                    'https://github.com/leadpoet/pydantic-harness',
                    'harness.run_icp', 'running', 20, 1,
                    '[{"icp_ref":"icp-1","status":"completed"}]'::jsonb,
                    '{"provider_calls":1}'::jsonb, 'owner',
                    'owner-claim', now() + interval '5 minutes'
                )
                """
            )
            cursor.execute(
                """
                SELECT public.research_lab_claim_daily_rebenchmark(
                    '22222222-2222-2222-2222-222222222222',
                    'new-claim', 'new-worker', 300
                )
                """
            )
            assert cursor.fetchone()[0] == {"claim_status": "busy"}

            cursor.execute(
                """
                UPDATE public.research_lab_daily_rebenchmarks
                   SET lease_expires_at = now() - interval '1 second'
                 WHERE run_id = '22222222-2222-2222-2222-222222222222'
                """
            )
            cursor.execute(
                """
                SELECT public.research_lab_claim_daily_rebenchmark(
                    '22222222-2222-2222-2222-222222222222',
                    'new-claim', 'new-worker', 300
                )
                """
            )
            takeover = cursor.fetchone()[0]
            assert takeover["claim_status"] == "claimed"
            assert takeover["run"]["claim_token"] == "new-claim"
            assert takeover["run"]["attempt_count"] == 2
            assert takeover["run"]["completed_icp_count"] == 0
            assert takeover["run"]["per_icp_results"] == []
            assert takeover["run"]["usage_doc"] == {}

            cursor.execute(
                """
                UPDATE public.research_lab_daily_rebenchmarks
                   SET completed_icp_count = 1
                 WHERE run_id = '22222222-2222-2222-2222-222222222222'
                   AND status = 'running'
                   AND claim_token = 'owner-claim'
                """
            )
            assert cursor.rowcount == 0

            cursor.execute(
                """
                UPDATE public.research_lab_daily_rebenchmarks
                   SET lease_expires_at = now() - interval '1 second'
                 WHERE run_id = '22222222-2222-2222-2222-222222222222'
                """
            )
            cursor.execute(
                """
                SELECT public.research_lab_claim_daily_rebenchmark(
                    '22222222-2222-2222-2222-222222222222',
                    'third-claim', 'third-worker', 300
                )
                """
            )
            exhausted = cursor.fetchone()[0]
            assert exhausted["claim_status"] == "exhausted"
            assert exhausted["run"]["status"] == "failed"
            assert exhausted["run"]["attempt_count"] == 2
            assert exhausted["run"]["error_doc"]["code"] == (
                "daily_rebenchmark_lease_exhausted"
            )
    finally:
        connection.close()
