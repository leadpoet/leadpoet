-- Ordinary daily state for the public open-source sourcing baseline.
-- This table intentionally has no model digest, receipt, manifest, signature,
-- Git identity, or attestation fields.

BEGIN;

CREATE TABLE IF NOT EXISTS public.research_lab_daily_rebenchmarks (
    run_id UUID PRIMARY KEY,
    benchmark_date DATE NOT NULL,
    baseline_id TEXT NOT NULL,
    baseline_repository TEXT NOT NULL,
    baseline_entrypoint TEXT NOT NULL,
    window_doc JSONB NOT NULL DEFAULT '{}'::jsonb,
    benchmark_input_doc JSONB NOT NULL DEFAULT '{}'::jsonb,
    evaluation_epoch BIGINT,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    attempt_count SMALLINT NOT NULL DEFAULT 1 CHECK (attempt_count BETWEEN 1 AND 2),
    expected_icp_count INTEGER NOT NULL CHECK (expected_icp_count = 20),
    completed_icp_count INTEGER NOT NULL DEFAULT 0 CHECK (completed_icp_count >= 0),
    aggregate_score DOUBLE PRECISION CHECK (
        aggregate_score IS NULL OR (aggregate_score >= 0 AND aggregate_score <= 100)
    ),
    per_icp_results JSONB NOT NULL DEFAULT '[]'::jsonb,
    usage_doc JSONB NOT NULL DEFAULT '{}'::jsonb,
    score_summary_doc JSONB NOT NULL DEFAULT '{}'::jsonb,
    public_report_doc JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_doc JSONB NOT NULL DEFAULT '{}'::jsonb,
    worker_ref TEXT NOT NULL,
    claim_token TEXT NOT NULL DEFAULT '',
    lease_expires_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    CONSTRAINT research_lab_daily_rebenchmarks_day_baseline_key
        UNIQUE (benchmark_date, baseline_id),
    CHECK (jsonb_typeof(window_doc) = 'object'),
    CHECK (jsonb_typeof(benchmark_input_doc) = 'object'),
    CHECK (jsonb_typeof(per_icp_results) = 'array'),
    CHECK (jsonb_typeof(usage_doc) = 'object'),
    CHECK (jsonb_typeof(score_summary_doc) = 'object'),
    CHECK (jsonb_typeof(public_report_doc) = 'object'),
    CHECK (jsonb_typeof(error_doc) = 'object'),
    CHECK (completed_icp_count <= expected_icp_count),
    CHECK (
        status <> 'completed'
        OR (
            completed_icp_count = expected_icp_count
            AND aggregate_score IS NOT NULL
            AND jsonb_array_length(per_icp_results) = expected_icp_count
            AND score_summary_doc <> '{}'::jsonb
            AND public_report_doc <> '{}'::jsonb
            AND error_doc = '{}'::jsonb
            AND completed_at IS NOT NULL
        )
    ),
    CHECK (
        status <> 'failed'
        OR (error_doc <> '{}'::jsonb AND completed_at IS NOT NULL)
    )
);

ALTER TABLE public.research_lab_daily_rebenchmarks
    ADD COLUMN IF NOT EXISTS benchmark_input_doc JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE public.research_lab_daily_rebenchmarks
    ADD COLUMN IF NOT EXISTS claim_token TEXT NOT NULL DEFAULT '';
ALTER TABLE public.research_lab_daily_rebenchmarks
    ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ;
ALTER TABLE public.research_lab_daily_rebenchmarks
    ADD COLUMN IF NOT EXISTS attempt_count SMALLINT NOT NULL DEFAULT 1;

-- Remove the prototype rolling-window identity. The daily date and public
-- baseline are the only run identity in the open competition.
ALTER TABLE public.research_lab_daily_rebenchmarks
    DROP COLUMN IF EXISTS rolling_window_hash;

DO $upgrade$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM pg_catalog.pg_constraint
         WHERE conrelid = 'public.research_lab_daily_rebenchmarks'::regclass
           AND conname = 'research_lab_daily_rebenchmarks_day_baseline_key'
    ) THEN
        ALTER TABLE public.research_lab_daily_rebenchmarks
            ADD CONSTRAINT research_lab_daily_rebenchmarks_day_baseline_key
            UNIQUE (benchmark_date, baseline_id);
    END IF;
END
$upgrade$;

ALTER TABLE public.research_lab_daily_rebenchmarks
    DROP CONSTRAINT IF EXISTS research_lab_daily_rebenchmarks_expected_icp_count_check;
ALTER TABLE public.research_lab_daily_rebenchmarks
    ADD CONSTRAINT research_lab_daily_rebenchmarks_expected_icp_count_check
    CHECK (expected_icp_count = 20);
ALTER TABLE public.research_lab_daily_rebenchmarks
    DROP CONSTRAINT IF EXISTS research_lab_daily_rebenchmarks_attempt_count_check;
ALTER TABLE public.research_lab_daily_rebenchmarks
    ADD CONSTRAINT research_lab_daily_rebenchmarks_attempt_count_check
    CHECK (attempt_count BETWEEN 1 AND 2);

CREATE OR REPLACE FUNCTION public.research_lab_claim_daily_rebenchmark(
    p_run_id UUID,
    p_claim_token TEXT,
    p_worker_ref TEXT,
    p_lease_seconds INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    v_row public.research_lab_daily_rebenchmarks;
    v_now TIMESTAMPTZ := pg_catalog.statement_timestamp();
BEGIN
    IF p_claim_token IS NULL OR p_claim_token = ''
       OR p_worker_ref IS NULL OR p_worker_ref = ''
       OR p_lease_seconds IS NULL
       OR p_lease_seconds < 60 OR p_lease_seconds > 7200 THEN
        RAISE EXCEPTION 'daily_rebenchmark_claim_invalid' USING ERRCODE = '22023';
    END IF;
    SELECT *
      INTO v_row
      FROM public.research_lab_daily_rebenchmarks
     WHERE run_id = p_run_id
     FOR UPDATE;
    IF NOT FOUND OR v_row.status <> 'running' THEN
        RETURN pg_catalog.jsonb_build_object('claim_status', 'busy');
    END IF;

    IF v_row.lease_expires_at IS NOT NULL AND v_row.lease_expires_at > v_now THEN
        IF v_row.claim_token <> p_claim_token THEN
            RETURN pg_catalog.jsonb_build_object('claim_status', 'busy');
        END IF;
        UPDATE public.research_lab_daily_rebenchmarks
           SET worker_ref = p_worker_ref,
               lease_expires_at = v_now
                   + pg_catalog.make_interval(secs => p_lease_seconds),
               updated_at = v_now
         WHERE run_id = p_run_id
        RETURNING * INTO v_row;
        RETURN pg_catalog.jsonb_build_object(
            'claim_status', 'claimed',
            'run', pg_catalog.to_jsonb(v_row)
        );
    END IF;

    IF v_row.attempt_count >= 2 THEN
        UPDATE public.research_lab_daily_rebenchmarks
           SET status = 'failed',
               error_doc = pg_catalog.jsonb_build_object(
                   'code', 'daily_rebenchmark_lease_exhausted',
                   'message', 'daily public rebenchmark lease expired twice'
               ),
               worker_ref = p_worker_ref,
               claim_token = '',
               lease_expires_at = NULL,
               updated_at = v_now,
               completed_at = v_now
         WHERE run_id = p_run_id
        RETURNING * INTO v_row;
        RETURN pg_catalog.jsonb_build_object(
            'claim_status', 'exhausted',
            'run', pg_catalog.to_jsonb(v_row)
        );
    END IF;

    UPDATE public.research_lab_daily_rebenchmarks
       SET attempt_count = attempt_count + 1,
           completed_icp_count = 0,
           aggregate_score = NULL,
           per_icp_results = '[]'::jsonb,
           usage_doc = '{}'::jsonb,
           score_summary_doc = '{}'::jsonb,
           public_report_doc = '{}'::jsonb,
           error_doc = '{}'::jsonb,
           claim_token = p_claim_token,
           worker_ref = p_worker_ref,
           lease_expires_at = v_now
               + pg_catalog.make_interval(secs => p_lease_seconds),
           started_at = v_now,
           updated_at = v_now,
           completed_at = NULL
     WHERE run_id = p_run_id
    RETURNING * INTO v_row;
    RETURN pg_catalog.jsonb_build_object(
        'claim_status', 'claimed',
        'run', pg_catalog.to_jsonb(v_row)
    );
END
$function$;

REVOKE ALL ON FUNCTION public.research_lab_claim_daily_rebenchmark(
    UUID, TEXT, TEXT, INTEGER
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_claim_daily_rebenchmark(
    UUID, TEXT, TEXT, INTEGER
) TO service_role;

CREATE OR REPLACE FUNCTION public.research_lab_retry_daily_rebenchmark(
    p_run_id UUID,
    p_expected_attempt INTEGER,
    p_claim_token TEXT,
    p_worker_ref TEXT,
    p_lease_seconds INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    v_row public.research_lab_daily_rebenchmarks;
    v_status TEXT;
    v_attempt_count INTEGER;
BEGIN
    IF p_expected_attempt IS NULL OR p_expected_attempt < 1
       OR p_claim_token IS NULL OR p_claim_token = ''
       OR p_worker_ref IS NULL OR p_worker_ref = ''
       OR p_lease_seconds IS NULL
       OR p_lease_seconds < 60 OR p_lease_seconds > 7200 THEN
        RAISE EXCEPTION 'daily_rebenchmark_retry_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.research_lab_daily_rebenchmarks
       SET status = 'running',
           attempt_count = attempt_count + 1,
           completed_icp_count = 0,
           aggregate_score = NULL,
           per_icp_results = '[]'::jsonb,
           usage_doc = '{}'::jsonb,
           score_summary_doc = '{}'::jsonb,
           public_report_doc = '{}'::jsonb,
           error_doc = '{}'::jsonb,
           worker_ref = p_worker_ref,
           claim_token = p_claim_token,
           lease_expires_at = pg_catalog.statement_timestamp()
               + pg_catalog.make_interval(secs => p_lease_seconds),
           started_at = pg_catalog.statement_timestamp(),
           updated_at = pg_catalog.statement_timestamp(),
           completed_at = NULL
     WHERE run_id = p_run_id
       AND status = 'failed'
       AND attempt_count = p_expected_attempt
       AND attempt_count < 2
    RETURNING * INTO v_row;
    IF FOUND THEN
        RETURN pg_catalog.jsonb_build_object(
            'retry_status', 'retried',
            'run', pg_catalog.to_jsonb(v_row)
        );
    END IF;
    SELECT status, attempt_count
      INTO v_status, v_attempt_count
      FROM public.research_lab_daily_rebenchmarks
     WHERE run_id = p_run_id;
    IF v_status = 'failed' AND v_attempt_count >= 2 THEN
        RETURN pg_catalog.jsonb_build_object('retry_status', 'exhausted');
    END IF;
    RETURN pg_catalog.jsonb_build_object('retry_status', 'stale');
END
$function$;

REVOKE ALL ON FUNCTION public.research_lab_retry_daily_rebenchmark(
    UUID, INTEGER, TEXT, TEXT, INTEGER
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_retry_daily_rebenchmark(
    UUID, INTEGER, TEXT, TEXT, INTEGER
) TO service_role;

ALTER TABLE public.research_lab_daily_rebenchmarks ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON TABLE public.research_lab_daily_rebenchmarks
    FROM PUBLIC, anon, authenticated, service_role;
GRANT SELECT, INSERT, UPDATE ON TABLE public.research_lab_daily_rebenchmarks TO service_role;

COMMENT ON TABLE public.research_lab_daily_rebenchmarks IS
    'Daily results and restart progress for the public sourcing baseline.';

NOTIFY pgrst, 'reload schema';

COMMIT;
