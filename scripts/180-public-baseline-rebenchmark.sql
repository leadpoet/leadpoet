-- Ordinary daily state for the public open-source sourcing baseline.
-- This table intentionally has no model digest, receipt, manifest, signature,
-- Git identity, or attestation fields.

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
    expected_icp_count INTEGER NOT NULL CHECK (expected_icp_count > 0),
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
BEGIN
    IF p_claim_token IS NULL OR p_claim_token = ''
       OR p_worker_ref IS NULL OR p_worker_ref = ''
       OR p_lease_seconds < 60 OR p_lease_seconds > 7200 THEN
        RAISE EXCEPTION 'daily_rebenchmark_claim_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.research_lab_daily_rebenchmarks
       SET claim_token = p_claim_token,
           worker_ref = p_worker_ref,
           lease_expires_at = pg_catalog.statement_timestamp()
               + pg_catalog.make_interval(secs => p_lease_seconds),
           updated_at = pg_catalog.statement_timestamp()
     WHERE run_id = p_run_id
       AND status = 'running'
       AND (
           claim_token = p_claim_token
           OR lease_expires_at IS NULL
           OR lease_expires_at <= pg_catalog.statement_timestamp()
       )
    RETURNING * INTO v_row;
    IF NOT FOUND THEN
        RETURN pg_catalog.jsonb_build_object('claim_status', 'busy');
    END IF;
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

ALTER TABLE public.research_lab_daily_rebenchmarks ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON TABLE public.research_lab_daily_rebenchmarks FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT, UPDATE ON TABLE public.research_lab_daily_rebenchmarks TO service_role;

COMMENT ON TABLE public.research_lab_daily_rebenchmarks IS
    'Daily results and restart progress for the public sourcing baseline.';

NOTIFY pgrst, 'reload schema';
