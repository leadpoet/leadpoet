-- Private, read-only failure-funnel reporting for Research Lab scoring.
--
-- This migration does not create another source of truth. It projects the
-- append-only score bundles, company-label rows, and scoring execution events
-- that Lab already stores. Historical rows with coarse funnels remain useful;
-- missing detailed labels are reported as partial telemetry, never as zero.

BEGIN;

CREATE INDEX IF NOT EXISTS idx_research_eval_score_bundles_ticket_created
    ON public.research_evaluation_score_bundles(ticket_id, created_at DESC)
    WHERE ticket_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_research_lab_company_labels_ticket_candidate
    ON public.research_lab_company_label_examples(ticket_id, candidate_id, captured_at DESC)
    WHERE ticket_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_research_lab_scoring_runs_ticket_candidate
    ON public.research_lab_scoring_runs(ticket_id, candidate_id, created_at DESC)
    WHERE ticket_id IS NOT NULL;

CREATE OR REPLACE FUNCTION public.get_research_lab_failure_funnel(
    p_ticket_id UUID,
    p_candidate_id TEXT DEFAULT NULL
)
RETURNS JSONB
LANGUAGE SQL
STABLE
SECURITY INVOKER
SET search_path = ''
AS $$
WITH bundle_candidates AS (
    SELECT
        b.score_bundle_id,
        COALESCE(
            NULLIF(b.score_bundle_doc ->> 'candidate_model_manifest_hash', ''),
            b.private_model_manifest_hash
        ) AS model_revision_hash,
        b.score_bundle_doc,
        b.created_at,
        COALESCE(
            NULLIF(b.score_bundle_doc #>> '{serving_model_version,candidate_id}', ''),
            NULLIF(b.score_bundle_doc ->> 'candidate_id', ''),
            b.run_id::TEXT
        ) AS reporting_candidate_id
    FROM public.research_evaluation_score_bundle_current b
    WHERE b.ticket_id = p_ticket_id
      AND COALESCE(b.current_event_status, b.bundle_status) <> 'tombstoned'
),
latest_bundles AS (
    SELECT DISTINCT ON (reporting_candidate_id)
        score_bundle_id,
        model_revision_hash,
        score_bundle_doc,
        created_at,
        reporting_candidate_id
    FROM bundle_candidates
    WHERE p_candidate_id IS NULL OR reporting_candidate_id = p_candidate_id
    ORDER BY reporting_candidate_id, created_at DESC, score_bundle_id DESC
),
raw_icps AS (
    SELECT
        b.reporting_candidate_id,
        item,
        item -> 'funnel' AS funnel
    FROM latest_bundles b
    CROSS JOIN LATERAL jsonb_array_elements(
        CASE
            WHEN jsonb_typeof(b.score_bundle_doc #> '{aggregates,per_icp_results}') = 'array'
                THEN b.score_bundle_doc #> '{aggregates,per_icp_results}'
            ELSE '[]'::JSONB
        END
    ) item
),
parsed_funnels AS (
    SELECT
        reporting_candidate_id,
        CASE WHEN jsonb_typeof(funnel -> 'sourced') = 'number'
            THEN (funnel ->> 'sourced')::INTEGER END AS sourced,
        CASE WHEN jsonb_typeof(funnel -> 'fit_pass') = 'number'
            THEN (funnel ->> 'fit_pass')::INTEGER END AS fit_pass,
        CASE WHEN jsonb_typeof(funnel -> 'verified') = 'number'
            THEN (funnel ->> 'verified')::INTEGER END AS verified,
        CASE WHEN jsonb_typeof(funnel -> 'intent_valid') = 'number'
            THEN (funnel ->> 'intent_valid')::INTEGER END AS intent_valid,
        CASE WHEN jsonb_typeof(funnel -> 'scored') = 'number'
            THEN (funnel ->> 'scored')::INTEGER END AS scored
    FROM raw_icps
),
valid_funnels AS (
    SELECT *
    FROM parsed_funnels
    WHERE sourced >= fit_pass
      AND fit_pass >= verified
      AND verified >= intent_valid
      AND intent_valid >= scored
      AND scored >= 0
),
stage_totals AS (
    SELECT
        COALESCE(COUNT(*), 0)::BIGINT AS icp_count,
        COALESCE(COUNT(*) FILTER (WHERE sourced > 0), 0)::BIGINT AS sourced_icp_count,
        COALESCE(COUNT(*) FILTER (WHERE sourced = 0), 0)::BIGINT AS zero_source_icp_count,
        COALESCE(SUM(sourced), 0)::BIGINT AS sourced,
        COALESCE(SUM(fit_pass), 0)::BIGINT AS fit_pass,
        COALESCE(SUM(verified), 0)::BIGINT AS verified,
        COALESCE(SUM(intent_valid), 0)::BIGINT AS intent_valid,
        COALESCE(SUM(scored), 0)::BIGINT AS scored
    FROM valid_funnels
),
stage_rows AS (
    SELECT * FROM (
        VALUES
            (1, 'sourcing'::TEXT, 'icp_attempts'::TEXT,
                (SELECT icp_count FROM stage_totals),
                (SELECT sourced_icp_count FROM stage_totals),
                (SELECT zero_source_icp_count FROM stage_totals)),
            (2, 'firmographic', 'companies',
                (SELECT sourced FROM stage_totals),
                (SELECT fit_pass FROM stage_totals),
                (SELECT sourced - fit_pass FROM stage_totals)),
            (3, 'verifier', 'companies',
                (SELECT fit_pass FROM stage_totals),
                (SELECT verified FROM stage_totals),
                (SELECT fit_pass - verified FROM stage_totals)),
            (4, 'intent', 'companies',
                (SELECT verified FROM stage_totals),
                (SELECT intent_valid FROM stage_totals),
                (SELECT verified - intent_valid FROM stage_totals)),
            (5, 'scoring', 'companies',
                (SELECT intent_valid FROM stage_totals),
                (SELECT scored FROM stage_totals),
                (SELECT intent_valid - scored FROM stage_totals))
    ) AS stages(stage_order, stage, unit, reviewed, passed, rejected)
),
label_failures AS (
    SELECT
        CASE
            WHEN LOWER(REPLACE(COALESCE(l.failure_reason, ''), '_', ' ')) LIKE '%employee count%'
              OR LOWER(REPLACE(COALESCE(l.failure_reason, ''), '_', ' ')) LIKE '%company stage%'
                THEN 'firmographic'
            WHEN LOWER(REPLACE(COALESCE(l.failure_reason, ''), '_', ' ')) LIKE '%company verification%'
                THEN 'verifier'
            WHEN LOWER(REPLACE(COALESCE(l.failure_reason, ''), '_', ' ')) LIKE '%fabricat%'
              OR LOWER(REPLACE(COALESCE(l.failure_reason, ''), '_', ' ')) LIKE '%llm scoring error%'
                THEN 'intent'
            WHEN LOWER(REPLACE(COALESCE(l.failure_reason, ''), '_', ' ')) LIKE '%pre-check%'
                THEN 'identity'
            WHEN LOWER(REPLACE(COALESCE(l.failure_reason, ''), '_', ' ')) LIKE '%duplicate%'
                THEN 'uniqueness'
            ELSE COALESCE(NULLIF(l.failure_stage, ''), 'unclassified')
        END AS stage,
        CASE
            WHEN LOWER(REPLACE(COALESCE(l.failure_reason, ''), '_', ' ')) LIKE '%employee count mismatch%'
                THEN 'employee_count_mismatch'
            WHEN LOWER(REPLACE(COALESCE(l.failure_reason, ''), '_', ' ')) LIKE '%missing employee count%'
                THEN 'employee_count_missing'
            WHEN LOWER(REPLACE(COALESCE(l.failure_reason, ''), '_', ' ')) LIKE '%company stage mismatch%'
                THEN 'company_stage_mismatch'
            WHEN LOWER(REPLACE(COALESCE(l.failure_reason, ''), '_', ' ')) LIKE '%missing company stage%'
                THEN 'company_stage_missing'
            WHEN LOWER(REPLACE(COALESCE(l.failure_reason, ''), '_', ' ')) LIKE '%company verification%'
                THEN 'company_unverifiable'
            WHEN LOWER(REPLACE(COALESCE(l.failure_reason, ''), '_', ' ')) LIKE '%fabricat%'
                THEN 'intent_fabricated'
            WHEN LOWER(REPLACE(COALESCE(l.failure_reason, ''), '_', ' ')) LIKE '%llm scoring error%'
                THEN 'scoring_error'
            WHEN LOWER(REPLACE(COALESCE(l.failure_reason, ''), '_', ' ')) LIKE '%pre-check%'
                THEN 'failed_prechecks'
            WHEN LOWER(REPLACE(COALESCE(l.failure_reason, ''), '_', ' ')) LIKE '%duplicate%'
                THEN 'duplicate_company'
            ELSE 'other'
        END AS reason_code,
        COUNT(*)::BIGINT AS failure_count
    FROM public.research_lab_company_label_examples l
    WHERE l.ticket_id = p_ticket_id
      AND l.capture_state <> 'tombstoned'
      AND l.model_side = 'candidate'
      AND (p_candidate_id IS NULL OR l.candidate_id = p_candidate_id)
      AND (l.final_score <= 0 OR COALESCE(l.failure_reason, '') <> '')
    GROUP BY 1, 2
),
latest_scoring_runs AS (
    SELECT DISTINCT ON (r.candidate_id)
        r.scoring_run_id,
        r.candidate_id
    FROM public.research_lab_scoring_runs r
    WHERE r.ticket_id = p_ticket_id
      AND r.run_type = 'candidate_scoring'
      AND (p_candidate_id IS NULL OR r.candidate_id = p_candidate_id)
    ORDER BY r.candidate_id, r.created_at DESC, r.run_attempt DESC
),
scoring_executions AS (
    SELECT DISTINCT ON (e.scoring_run_id, e.icp_ref)
        e.icp_execution_id
    FROM public.research_lab_scoring_icp_executions e
    JOIN latest_scoring_runs r
      ON r.scoring_run_id = e.scoring_run_id
    WHERE e.model_role = 'candidate'
    ORDER BY e.scoring_run_id, e.icp_ref, e.attempt_ordinal DESC, e.created_at DESC
),
latest_execution_events AS (
    SELECT DISTINCT ON (e.icp_execution_id)
        e.icp_execution_id,
        e.event_type,
        e.failure_category,
        e.telemetry_degraded
    FROM public.research_lab_scoring_icp_events e
    JOIN scoring_executions x ON x.icp_execution_id = e.icp_execution_id
    ORDER BY e.icp_execution_id, e.event_ordinal DESC, e.created_at DESC
),
execution_failures AS (
    SELECT
        CASE
            WHEN COALESCE(failure_category, '') ~ '(source|provider|runtime)' THEN 'sourcing'
            WHEN COALESCE(failure_category, '') LIKE '%verif%' THEN 'verifier'
            WHEN COALESCE(failure_category, '') LIKE '%intent%' THEN 'intent'
            ELSE 'scoring'
        END AS stage,
        COALESCE(NULLIF(failure_category, ''), 'execution_failed') AS reason_code,
        COUNT(*)::BIGINT AS failure_count
    FROM latest_execution_events
    WHERE event_type IN ('failed', 'cancelled', 'skipped')
    GROUP BY 1, 2
),
reason_rows AS (
    SELECT stage, reason_code, 'companies'::TEXT AS unit, failure_count
    FROM label_failures
    UNION ALL
    SELECT stage, reason_code, 'icp_attempts'::TEXT AS unit, failure_count
    FROM execution_failures
),
coverage AS (
    SELECT
        (SELECT COUNT(*) FROM latest_bundles)::BIGINT AS bundle_count,
        (SELECT COUNT(*) FROM raw_icps)::BIGINT AS icp_row_count,
        (SELECT COUNT(*) FROM valid_funnels)::BIGINT AS funnel_row_count,
        (SELECT COUNT(*) FROM public.research_lab_company_label_examples l
          WHERE l.ticket_id = p_ticket_id
            AND l.capture_state <> 'tombstoned'
            AND l.model_side = 'candidate'
            AND (p_candidate_id IS NULL OR l.candidate_id = p_candidate_id))::BIGINT AS company_label_count,
        (SELECT COALESCE(SUM(failure_count), 0) FROM label_failures)::BIGINT
            AS company_failure_count,
        (SELECT COALESCE(SUM(failure_count), 0) FROM label_failures
          WHERE reason_code = 'other' OR stage = 'unclassified')::BIGINT
            AS unclassified_failure_count,
        (SELECT COUNT(*) FROM latest_execution_events)::BIGINT AS execution_count,
        (SELECT COUNT(*) FROM parsed_funnels p
          WHERE (
              p.sourced >= p.fit_pass
              AND p.fit_pass >= p.verified
              AND p.verified >= p.intent_valid
              AND p.intent_valid >= p.scored
              AND p.scored >= 0
          ) IS NOT TRUE)::BIGINT AS invalid_funnel_row_count,
        (SELECT COUNT(*) FROM latest_execution_events
          WHERE event_type NOT IN ('completed', 'failed', 'cancelled', 'skipped'))::BIGINT
            AS nonterminal_execution_count,
        (SELECT COUNT(*) FROM latest_execution_events
          WHERE telemetry_degraded)::BIGINT AS degraded_execution_count
),
telemetry AS (
    SELECT
        c.*,
        CASE
            WHEN c.bundle_count = 0 AND c.company_label_count = 0 AND c.execution_count = 0
                THEN 'missing'
            WHEN c.funnel_row_count = 0
              OR c.invalid_funnel_row_count > 0
              OR c.nonterminal_execution_count > 0
              OR c.degraded_execution_count > 0
              OR c.company_failure_count <> (SELECT sourced - scored FROM stage_totals)
              OR c.unclassified_failure_count > 0
                THEN 'partial'
            ELSE 'complete'
        END AS status
    FROM coverage c
)
SELECT jsonb_build_object(
    'schema_version', 'research_lab_failure_funnel.v1',
    'ticket_id', p_ticket_id,
    'candidate_id', p_candidate_id,
    'stages', CASE
        WHEN (SELECT funnel_row_count FROM telemetry) = 0 THEN '[]'::JSONB
        ELSE COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
                'stage', stage,
                'unit', unit,
                'reviewed', reviewed,
                'passed', passed,
                'rejected', rejected
            ) ORDER BY stage_order)
            FROM stage_rows
        ), '[]'::JSONB)
    END,
    'rejections', COALESCE((
        SELECT jsonb_agg(jsonb_build_object(
            'stage', stage,
            'reason_code', reason_code,
            'unit', unit,
            'count', failure_count
        ) ORDER BY stage, reason_code, unit)
        FROM reason_rows
    ), '[]'::JSONB),
    'model_revisions', COALESCE((
        SELECT jsonb_agg(model_revision_hash ORDER BY model_revision_hash)
        FROM (
            SELECT DISTINCT model_revision_hash
            FROM latest_bundles
        ) revisions
    ), '[]'::JSONB),
    'telemetry', (
        SELECT jsonb_build_object(
            'status', status,
            'bundle_count', bundle_count,
            'icp_row_count', icp_row_count,
            'funnel_row_count', funnel_row_count,
            'company_label_count', company_label_count,
            'company_failure_count', company_failure_count,
            'detailed_reason_gap_count', GREATEST(
                0,
                (SELECT sourced - scored FROM stage_totals) - company_failure_count
            ),
            'detailed_reason_excess_count', GREATEST(
                0,
                company_failure_count - (SELECT sourced - scored FROM stage_totals)
            ),
            'unclassified_failure_count', unclassified_failure_count,
            'execution_count', execution_count,
            'invalid_funnel_row_count', invalid_funnel_row_count,
            'nonterminal_execution_count', nonterminal_execution_count,
            'degraded_execution_count', degraded_execution_count
        )
        FROM telemetry
    )
);
$$;

REVOKE ALL ON FUNCTION public.get_research_lab_failure_funnel(UUID, TEXT)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.get_research_lab_failure_funnel(UUID, TEXT)
    TO service_role;

COMMENT ON FUNCTION public.get_research_lab_failure_funnel(UUID, TEXT) IS
    'Service-only counts report for Research Lab sourcing, firmographic, verifier, intent, and scoring failures. Missing historical detail is marked partial.';

COMMIT;
