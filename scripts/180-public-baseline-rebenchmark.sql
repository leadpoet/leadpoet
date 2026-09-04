-- Ordinary daily state for the public open-source sourcing baseline.
-- This table intentionally has no model digest, receipt, manifest, signature,
-- Git identity, or attestation fields.

CREATE TABLE IF NOT EXISTS public.research_lab_daily_rebenchmarks (
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
    aggregate_score DOUBLE PRECISION CHECK (
        aggregate_score IS NULL OR (aggregate_score >= 0 AND aggregate_score <= 100)
    ),
    per_icp_results JSONB NOT NULL DEFAULT '[]'::jsonb,
    usage_doc JSONB NOT NULL DEFAULT '{}'::jsonb,
    score_summary_doc JSONB NOT NULL DEFAULT '{}'::jsonb,
    public_report_doc JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_doc JSONB NOT NULL DEFAULT '{}'::jsonb,
    worker_ref TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    UNIQUE (benchmark_date, baseline_id, rolling_window_hash),
    CHECK (jsonb_typeof(window_doc) = 'object'),
    CHECK (jsonb_typeof(per_icp_results) = 'array'),
    CHECK (jsonb_typeof(usage_doc) = 'object'),
    CHECK (jsonb_typeof(score_summary_doc) = 'object'),
    CHECK (jsonb_typeof(public_report_doc) = 'object'),
    CHECK (jsonb_typeof(error_doc) = 'object'),
    CHECK (completed_icp_count <= expected_icp_count)
);

ALTER TABLE public.research_lab_daily_rebenchmarks ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON TABLE public.research_lab_daily_rebenchmarks FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT, UPDATE ON TABLE public.research_lab_daily_rebenchmarks TO service_role;

COMMENT ON TABLE public.research_lab_daily_rebenchmarks IS
    'Daily results and restart progress for the public sourcing baseline.';

