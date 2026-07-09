-- Research Lab: per-ICP churn/reversal observability view.
--
-- This view is Research Lab-only observability. It exposes ICP refs/hashes,
-- dates, scores, and volatility metrics from published benchmark summaries.
-- It deliberately does not expose private ICP text, provider payloads, prompts,
-- companies, or raw traces.

BEGIN;

CREATE OR REPLACE VIEW public.research_lab_icp_churn_reversal_report AS
WITH baseline_rows AS (
    SELECT
        b.benchmark_date,
        b.created_at,
        COALESCE(b.rolling_window_hash, b.score_summary_doc ->> 'rolling_window_hash') AS rolling_window_hash,
        item ->> 'icp_ref' AS icp_ref,
        item ->> 'icp_hash' AS icp_hash,
        NULLIF(item ->> 'score', '')::double precision AS score
    FROM public.research_lab_private_model_benchmark_bundles b
    CROSS JOIN LATERAL jsonb_array_elements(
        COALESCE(b.score_summary_doc -> 'per_icp_summaries', '[]'::jsonb)
    ) AS item
    WHERE b.benchmark_quality = 'passed'
),
score_steps AS (
    SELECT
        *,
        score - LAG(score) OVER (
            PARTITION BY COALESCE(NULLIF(icp_hash, ''), NULLIF(icp_ref, ''))
            ORDER BY benchmark_date, created_at
        ) AS score_delta
    FROM baseline_rows
    WHERE COALESCE(NULLIF(icp_hash, ''), NULLIF(icp_ref, '')) IS NOT NULL
),
delta_steps AS (
    SELECT
        *,
        LAG(score_delta) OVER (
            PARTITION BY COALESCE(NULLIF(icp_hash, ''), NULLIF(icp_ref, ''))
            ORDER BY benchmark_date, created_at
        ) AS previous_score_delta
    FROM score_steps
)
SELECT
    COALESCE(NULLIF(icp_hash, ''), NULLIF(icp_ref, '')) AS icp_key,
    MAX(NULLIF(icp_ref, '')) AS icp_ref,
    MAX(NULLIF(icp_hash, '')) AS icp_hash,
    MIN(benchmark_date) AS first_benchmark_date,
    MAX(benchmark_date) AS last_benchmark_date,
    COUNT(*)::integer AS sample_count,
    ROUND(AVG(score)::numeric, 6)::double precision AS mean_score,
    MIN(score) AS min_score,
    MAX(score) AS max_score,
    ROUND(STDDEV_SAMP(score)::numeric, 6)::double precision AS score_stddev,
    COUNT(*) FILTER (
        WHERE previous_score_delta IS NOT NULL
          AND score_delta IS NOT NULL
          AND previous_score_delta <> 0
          AND score_delta <> 0
          AND SIGN(previous_score_delta) <> SIGN(score_delta)
    )::integer AS reversal_count,
    ARRAY_AGG(DISTINCT rolling_window_hash ORDER BY rolling_window_hash)
        FILTER (WHERE rolling_window_hash IS NOT NULL) AS rolling_window_hashes
FROM delta_steps
GROUP BY COALESCE(NULLIF(icp_hash, ''), NULLIF(icp_ref, ''));

COMMENT ON VIEW public.research_lab_icp_churn_reversal_report IS
    'Research Lab observability view for repeated per-ICP score reversals/volatility. Contains only refs/hashes and aggregate scores; no private ICP text or provider payloads.';

COMMIT;
