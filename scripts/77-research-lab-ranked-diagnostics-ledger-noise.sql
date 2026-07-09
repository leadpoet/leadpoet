-- Research Lab ranked fallback diagnostics, ledger serving stamps, and noise report.
--
-- Additive only. Does not change scoring, promotion, rewards, weights,
-- fulfillment, ICP selection, capped top-5 behavior, or provider-cost math.

BEGIN;

ALTER TABLE public.research_lab_results_ledger
    ADD COLUMN IF NOT EXISTS serving_model_version_hash TEXT NOT NULL DEFAULT '';

ALTER TABLE public.research_lab_results_ledger
    ADD COLUMN IF NOT EXISTS serving_model_manifest_hash TEXT NOT NULL DEFAULT '';

ALTER TABLE public.research_lab_results_ledger
    ADD COLUMN IF NOT EXISTS serving_model_artifact_hash TEXT NOT NULL DEFAULT '';

ALTER TABLE public.research_lab_results_ledger
    ADD COLUMN IF NOT EXISTS private_model_version_id TEXT NOT NULL DEFAULT '';

ALTER TABLE public.research_lab_results_ledger
    ADD COLUMN IF NOT EXISTS candidate_id TEXT NOT NULL DEFAULT '';

ALTER TABLE public.research_lab_results_ledger
    ADD COLUMN IF NOT EXISTS score_bundle_id TEXT NOT NULL DEFAULT '';

ALTER TABLE public.research_lab_results_ledger
    ADD COLUMN IF NOT EXISTS serving_model_version_doc JSONB NOT NULL DEFAULT '{}'::JSONB;

ALTER TABLE public.research_lab_results_ledger
    DROP CONSTRAINT IF EXISTS research_lab_results_ledger_serving_model_doc_safe_check;

ALTER TABLE public.research_lab_results_ledger
    ADD CONSTRAINT research_lab_results_ledger_serving_model_doc_safe_check
    CHECK (
        jsonb_typeof(serving_model_version_doc) = 'object'
        AND serving_model_version_doc::TEXT !~* '(sk-or-|openrouter_api_key|openrouter_management_key|scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|service_role|hidden_prompt|provider_output|request_body|response_body|page_content|raw_content|judge_prompt|private_repo|proxy[_-]?url|://[^/]+:[^/@]+@)'
    );

CREATE INDEX IF NOT EXISTS idx_research_lab_results_ledger_serving_model
    ON public.research_lab_results_ledger(serving_model_artifact_hash, created_at DESC)
    WHERE serving_model_artifact_hash <> '';

CREATE INDEX IF NOT EXISTS idx_research_lab_results_ledger_score_bundle
    ON public.research_lab_results_ledger(score_bundle_id, created_at DESC)
    WHERE score_bundle_id <> '';

CREATE INDEX IF NOT EXISTS idx_research_lab_results_ledger_candidate
    ON public.research_lab_results_ledger(candidate_id, created_at DESC)
    WHERE candidate_id <> '';

CREATE OR REPLACE VIEW public.research_lab_daily_noise_budget_report_current
WITH (security_invoker = true) AS
SELECT
    b.benchmark_bundle_id,
    b.benchmark_date,
    b.rolling_window_hash,
    b.benchmark_attempt,
    b.benchmark_quality,
    b.private_model_artifact_hash,
    b.private_model_manifest_hash,
    b.evaluation_epoch,
    COALESCE(NULLIF(n.noise->>'aggregate_score', '')::DOUBLE PRECISION, b.aggregate_score)
        AS aggregate_score,
    NULLIF(n.noise->>'icp_count', '')::INTEGER AS icp_count,
    NULLIF(n.noise->>'mean_icp_score', '')::DOUBLE PRECISION AS mean_icp_score,
    NULLIF(n.noise->>'sample_sd', '')::DOUBLE PRECISION AS sample_sd,
    NULLIF(n.noise->>'standard_error', '')::DOUBLE PRECISION AS standard_error,
    NULLIF(n.noise->'confidence_band_95'->>'lower', '')::DOUBLE PRECISION
        AS confidence_band_95_lower,
    NULLIF(n.noise->'confidence_band_95'->>'upper', '')::DOUBLE PRECISION
        AS confidence_band_95_upper,
    NULLIF(n.noise->>'zero_score_count', '')::INTEGER AS zero_score_count,
    COALESCE((n.noise->>'high_volatility')::BOOLEAN, FALSE) AS high_volatility,
    COALESCE((n.noise->>'observability_only')::BOOLEAN, TRUE) AS observability_only,
    n.noise AS daily_noise_budget_doc,
    b.benchmark_bundle_hash,
    b.created_at,
    b.current_event_seq,
    b.current_event_type,
    b.current_benchmark_status,
    b.current_status_at
FROM public.research_lab_private_model_benchmark_current b
CROSS JOIN LATERAL (
    SELECT b.score_summary_doc->'daily_noise_budget' AS noise
) n
WHERE b.benchmark_quality = 'passed'
  AND b.current_benchmark_status = 'completed'
  AND jsonb_typeof(n.noise) = 'object';

REVOKE ALL ON TABLE public.research_lab_daily_noise_budget_report_current FROM anon, authenticated;
GRANT SELECT ON TABLE public.research_lab_daily_noise_budget_report_current TO service_role;

COMMENT ON COLUMN public.research_lab_results_ledger.serving_model_version_hash IS
    'Join-safe Research Lab serving version hash for the model that produced this ledger row.';
COMMENT ON COLUMN public.research_lab_results_ledger.serving_model_manifest_hash IS
    'Join-safe private model manifest hash for the model that produced this ledger row.';
COMMENT ON COLUMN public.research_lab_results_ledger.serving_model_artifact_hash IS
    'Join-safe model artifact hash for the model that produced this ledger row.';
COMMENT ON COLUMN public.research_lab_results_ledger.private_model_version_id IS
    'Active private model version id when known; empty for historical rows.';
COMMENT ON COLUMN public.research_lab_results_ledger.candidate_id IS
    'Candidate id associated with the ledger row when known.';
COMMENT ON COLUMN public.research_lab_results_ledger.score_bundle_id IS
    'Score bundle id associated with the ledger row when known.';
COMMENT ON COLUMN public.research_lab_results_ledger.serving_model_version_doc IS
    'Sanitized serving-model join metadata only; raw manifests, prompts, provider responses, and secrets are forbidden.';
COMMENT ON VIEW public.research_lab_daily_noise_budget_report_current IS
    'Queryable Research Lab daily benchmark noise budget extracted from completed benchmark bundles. Observability only; not a promotion or reward gate.';

COMMIT;
