-- Private, read-only failure-funnel reporting for Research Lab scoring.
--
-- This migration does not create another source of truth. It projects the
-- append-only score bundles, company-label rows, and scoring execution events
-- that Lab already stores. Historical rows with coarse funnels remain useful;
-- missing detailed labels are reported as partial telemetry, never as zero.

BEGIN;

DO $prerequisite$
DECLARE
    invalid_indexes TEXT;
BEGIN
    SELECT pg_catalog.string_agg(expected.index_name, ', ' ORDER BY expected.index_name)
    INTO invalid_indexes
    FROM (
        VALUES
            (
                'idx_research_eval_score_bundles_ticket_created',
                'research_evaluation_score_bundles'
            ),
            (
                'idx_research_lab_company_labels_ticket_candidate',
                'research_lab_company_label_examples'
            ),
            (
                'idx_research_lab_scoring_runs_ticket_candidate',
                'research_lab_scoring_runs'
            )
    ) AS expected(index_name, table_name)
    WHERE NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_class index_relation
        JOIN pg_catalog.pg_namespace index_namespace
          ON index_namespace.oid = index_relation.relnamespace
        JOIN pg_catalog.pg_index index_meta
          ON index_meta.indexrelid = index_relation.oid
        JOIN pg_catalog.pg_class table_relation
          ON table_relation.oid = index_meta.indrelid
        JOIN pg_catalog.pg_namespace table_namespace
          ON table_namespace.oid = table_relation.relnamespace
        WHERE index_namespace.nspname = 'public'
          AND table_namespace.nspname = 'public'
          AND index_relation.relname = expected.index_name
          AND table_relation.relname = expected.table_name
          AND index_relation.relkind = 'i'
          AND index_meta.indisvalid
          AND index_meta.indisready
          AND index_meta.indislive
    );

    IF invalid_indexes IS NOT NULL THEN
        RAISE EXCEPTION
            'failure-funnel prerequisite indexes are missing or invalid: %',
            invalid_indexes;
    END IF;
END;
$prerequisite$;

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
        NULLIF(b.score_bundle_doc ->> 'candidate_artifact_hash', '')
            AS model_revision_hash,
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
bundle_health_inputs AS (
    SELECT
        b.reporting_candidate_id,
        b.score_bundle_doc -> 'scoring_health' AS scoring_health,
        CASE
            WHEN jsonb_typeof(
                b.score_bundle_doc #> '{aggregates,per_icp_results}'
            ) = 'array'
                THEN jsonb_array_length(
                    b.score_bundle_doc #> '{aggregates,per_icp_results}'
                )
            ELSE 0
        END AS bundle_icp_count
    FROM latest_bundles b
),
bundle_health_parsed AS (
    SELECT
        reporting_candidate_id,
        scoring_health,
        bundle_icp_count,
        CASE
            WHEN jsonb_typeof(scoring_health) = 'object'
              AND scoring_health ->> 'schema_version' = '1.0'
              AND scoring_health ->> 'health_status' IN ('healthy', 'degraded')
              AND jsonb_typeof(scoring_health -> 'failure_class_counts') = 'object'
              AND COALESCE(scoring_health ->> 'icp_count', '') ~ '^(0|[1-9][0-9]*)$'
                THEN (scoring_health ->> 'icp_count')::NUMERIC
            ELSE NULL
        END AS health_icp_count
    FROM bundle_health_inputs
),
bundle_health AS (
    SELECT
        *,
        health_icp_count IS NOT NULL
            AND health_icp_count = bundle_icp_count AS health_contract_valid
    FROM bundle_health_parsed
),
raw_icps AS (
    SELECT
        b.reporting_candidate_id,
        NULLIF(TRIM(item ->> 'icp_ref'), '') AS icp_ref,
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
        icp_ref,
        funnel,
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
            (2, 'company_fit', 'companies',
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
latest_scoring_runs AS (
    SELECT DISTINCT ON (r.candidate_id)
        r.scoring_run_id,
        r.candidate_id,
        r.expected_icp_count,
        r.current_run_status,
        r.current_telemetry_degraded,
        r.score_bundle_id
    FROM public.research_lab_scoring_run_current r
    WHERE r.ticket_id = p_ticket_id
      AND r.run_type = 'candidate_scoring'
      AND (p_candidate_id IS NULL OR r.candidate_id = p_candidate_id)
    ORDER BY r.candidate_id, r.created_at DESC, r.run_attempt DESC
),
label_inputs AS (
    SELECT
        l.candidate_id,
        NULLIF(TRIM(l.icp_ref), '') AS icp_ref,
        l.final_score,
        COALESCE(l.failure_reason, '') AS failure_reason,
        LOWER(REPLACE(COALESCE(l.failure_reason, ''), '_', ' ')) AS reason_text,
        LOWER(TRIM(COALESCE(l.failure_stage, ''))) AS explicit_stage,
        LOWER(COALESCE(
            l.capture_doc ->> 'retryable_infrastructure_failure',
            ''
        )) IN ('true', '1', 'yes') AS retryable_infrastructure_failure
    FROM public.research_lab_company_label_examples l
    JOIN latest_scoring_runs r
      ON r.candidate_id = l.candidate_id
     AND NULLIF(l.capture_doc ->> 'scoring_run_id', '')
            = r.scoring_run_id::TEXT
    WHERE l.ticket_id = p_ticket_id
      AND l.capture_state <> 'tombstoned'
      AND l.model_side = 'candidate'
      AND (p_candidate_id IS NULL OR l.candidate_id = p_candidate_id)
),
label_outcomes AS (
    SELECT
        candidate_id,
        icp_ref,
        CASE
            -- New rows carry the scorer's authoritative stage. Preserve it;
            -- reason parsing below exists only for historical missing stages.
            WHEN explicit_stage <> '' THEN explicit_stage
            WHEN retryable_infrastructure_failure
                THEN 'infrastructure'
            WHEN reason_text LIKE '%intent verification unavailable:%'
              OR reason_text LIKE '%llm scoring error:%'
              OR reason_text LIKE '%company verification error:%'
              OR reason_text LIKE '%company verification failed: website unreachable:%'
              OR reason_text LIKE '%company verification failed: website fetch error:%'
              OR reason_text LIKE '%company verification unavailable:%'
              OR reason_text LIKE '%company fit pre-check unavailable:%'
              OR reason_text LIKE '%company web re-verification unavailable:%'
              OR reason_text LIKE '%providerclientv2error%'
              OR reason_text LIKE '%runner external request must use https%'
              OR reason_text LIKE '%provider error%'
              OR reason_text LIKE '%provider timeout%'
              OR reason_text LIKE '%http 429%'
              OR reason_text LIKE '%no openrouter key%'
              OR reason_text LIKE '%no_openrouter_key%'
                THEN 'infrastructure'
            WHEN reason_text LIKE '%employee count%'
              OR reason_text LIKE '%company stage%'
                THEN 'firmographic'
            WHEN reason_text LIKE '%company fit%'
                THEN 'company_fit'
            WHEN reason_text LIKE '%required attribute%'
                THEN 'attribute'
            WHEN reason_text LIKE '%company verification%'
                THEN 'verifier'
            WHEN reason_text LIKE '%fabricat%'
                THEN 'intent'
            WHEN reason_text LIKE '%llm scoring error%'
                THEN 'scoring'
            WHEN reason_text LIKE '%pre-check%'
                THEN 'identity'
            WHEN reason_text LIKE '%duplicate%'
                THEN 'uniqueness'
            ELSE 'unclassified'
        END AS stage,
        CASE
            WHEN retryable_infrastructure_failure
              OR explicit_stage ~ '(^|_)(provider|transport|infrastructure)(_|$)'
              OR reason_text LIKE '%intent verification unavailable:%'
              OR reason_text LIKE '%llm scoring error:%'
              OR reason_text LIKE '%company verification error:%'
              OR reason_text LIKE '%company verification failed: website unreachable:%'
              OR reason_text LIKE '%company verification failed: website fetch error:%'
              OR reason_text LIKE '%company verification unavailable:%'
              OR reason_text LIKE '%company fit pre-check unavailable:%'
              OR reason_text LIKE '%company web re-verification unavailable:%'
              OR reason_text LIKE '%providerclientv2error%'
              OR reason_text LIKE '%runner external request must use https%'
              OR reason_text LIKE '%provider error%'
              OR reason_text LIKE '%provider timeout%'
              OR reason_text LIKE '%http 429%'
              OR reason_text LIKE '%no openrouter key%'
              OR reason_text LIKE '%no_openrouter_key%'
                THEN 'infrastructure_failure'
            WHEN reason_text LIKE '%employee count mismatch%'
                THEN 'employee_count_mismatch'
            WHEN reason_text LIKE '%missing employee count%'
                THEN 'employee_count_missing'
            WHEN reason_text LIKE '%company stage mismatch%'
                THEN 'company_stage_mismatch'
            WHEN reason_text LIKE '%missing company stage%'
                THEN 'company_stage_missing'
            WHEN reason_text LIKE '%company fit%'
                THEN 'company_fit_not_proven'
            WHEN reason_text LIKE '%required attribute%'
                THEN 'required_attribute_not_proven'
            WHEN reason_text LIKE '%company verification%'
                THEN 'company_unverifiable'
            WHEN reason_text LIKE '%fabricat%'
                THEN 'intent_fabricated'
            WHEN reason_text LIKE '%llm scoring error%'
                THEN 'scoring_error'
            WHEN reason_text LIKE '%pre-check%'
                THEN 'failed_prechecks'
            WHEN reason_text LIKE '%duplicate%'
                THEN 'duplicate_company'
            ELSE 'other'
        END AS reason_code,
        (final_score <= 0 OR failure_reason <> '') AS is_failure,
        (final_score > 0 AND failure_reason = '') AS is_positive
    FROM label_inputs
),
label_failures AS (
    SELECT stage, reason_code, COUNT(*)::BIGINT AS failure_count
    FROM label_outcomes
    WHERE is_failure
    GROUP BY 1, 2
),
label_coverage AS (
    SELECT
        COUNT(*)::BIGINT AS company_label_count,
        COUNT(*) FILTER (WHERE is_failure)::BIGINT AS company_failure_count,
        COUNT(*) FILTER (WHERE is_positive)::BIGINT AS company_positive_count,
        COUNT(*) FILTER (
            WHERE is_failure AND reason_code = 'infrastructure_failure'
        )::BIGINT AS infrastructure_label_failure_count
    FROM label_outcomes
),
scoring_executions AS (
    SELECT DISTINCT ON (e.scoring_run_id, e.icp_ref)
        e.icp_execution_id,
        e.scoring_run_id,
        e.icp_ref,
        r.candidate_id
    FROM public.research_lab_scoring_icp_executions e
    JOIN latest_scoring_runs r
      ON r.scoring_run_id = e.scoring_run_id
    WHERE e.model_role = 'candidate'
    ORDER BY e.scoring_run_id, e.icp_ref, e.attempt_ordinal DESC, e.created_at DESC
),
latest_execution_events AS (
    SELECT DISTINCT ON (e.icp_execution_id)
        e.icp_execution_id,
        x.candidate_id,
        x.icp_ref,
        e.event_type,
        e.retryable,
        e.failure_category,
        e.telemetry_degraded
    FROM public.research_lab_scoring_icp_events e
    JOIN scoring_executions x ON x.icp_execution_id = e.icp_execution_id
    -- Heartbeats use minute-based ordinals while ordinary terminal events use
    -- the append-event default. Event ordinal is therefore not chronology.
    -- Match the canonical current view's timestamp/identity ordering so a
    -- heartbeat can never mask a later completion or failure.
    ORDER BY e.icp_execution_id, e.occurred_at DESC, e.event_id DESC
),
execution_failures AS (
    SELECT
        CASE
            WHEN retryable IS TRUE
              OR telemetry_degraded
              OR COALESCE(failure_category, '') ~ '(^|_)(infra|provider|transport|supabase|persistence|persist|network|timeout|http)(_|$)'
                THEN 'infrastructure'
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
bundle_failure_reason_inputs AS (
    SELECT
        CASE
            WHEN LOWER(COALESCE(item ->> 'provider_excluded', '')) IN ('true', '1', 'yes')
              OR LOWER(COALESCE(item ->> 'provider_cost_cap_blocked', '')) IN ('true', '1', 'yes')
              OR LOWER(COALESCE(item ->> 'provider_cost_tracking_failed', '')) IN ('true', '1', 'yes')
              OR COALESCE(item ->> 'failure_reason', '') ~
                    '(^|;)(candidate_model_runtime_(provider_error|timeout)|reference_model_runtime_provider_error)(;|$)'
                THEN 'infrastructure'
            WHEN COALESCE(item ->> 'failure_reason', '') LIKE
                    '%candidate_model_runtime_skipped_after_%'
                THEN 'sourcing'
            WHEN COALESCE(item ->> 'failure_reason', '') ~
                    '(^|;)candidate_model_runtime_(invalid_json|adapter_failed|invalid_output)(;|$)'
                THEN 'scoring'
            WHEN COALESCE(item ->> 'failure_reason', '') ~
                    '(^|;)candidate_model_zero_scoreable_companies(;|$)'
                THEN 'scoring'
            WHEN COALESCE(item ->> 'failure_reason', '') ~
                    '(^|;)candidate_model_zero_companies(;|$)'
                THEN 'sourcing'
            ELSE NULL
        END AS stage,
        CASE
            -- Keep stable, sanitized scorer receipt codes for the internal
            -- report. The signed-miner projection coarsens provider details.
            WHEN LOWER(COALESCE(item ->> 'provider_cost_tracking_failed', ''))
                    IN ('true', '1', 'yes')
                THEN 'provider_cost_tracking_failed'
            WHEN LOWER(COALESCE(item ->> 'provider_cost_cap_blocked', ''))
                    IN ('true', '1', 'yes')
                THEN 'provider_cost_cap_blocked'
            WHEN LOWER(COALESCE(item ->> 'provider_excluded', ''))
                    IN ('true', '1', 'yes')
                THEN 'provider_excluded'
            WHEN COALESCE(item ->> 'failure_reason', '') ~
                    '(^|;)reference_model_runtime_provider_error(;|$)'
                THEN 'reference_model_runtime_provider_error'
            WHEN COALESCE(item ->> 'failure_reason', '') ~
                    '(^|;)candidate_model_runtime_provider_error(;|$)'
                THEN 'candidate_model_runtime_provider_error'
            WHEN COALESCE(item ->> 'failure_reason', '') ~
                    '(^|;)candidate_model_runtime_timeout(;|$)'
                THEN 'candidate_model_runtime_timeout'
            WHEN COALESCE(item ->> 'failure_reason', '') ~
                    '(^|;)candidate_model_runtime_invalid_json(;|$)'
                THEN 'candidate_model_runtime_invalid_json'
            WHEN COALESCE(item ->> 'failure_reason', '') ~
                    '(^|;)candidate_model_runtime_adapter_failed(;|$)'
                THEN 'candidate_model_runtime_adapter_failed'
            WHEN COALESCE(item ->> 'failure_reason', '') ~
                    '(^|;)candidate_model_runtime_invalid_output(;|$)'
                THEN 'candidate_model_runtime_invalid_output'
            WHEN COALESCE(item ->> 'failure_reason', '') ~
                    '(^|;)candidate_model_runtime_skipped_after_timeout(;|$)'
                THEN 'candidate_model_runtime_skipped_after_timeout'
            WHEN COALESCE(item ->> 'failure_reason', '') ~
                    '(^|;)candidate_model_runtime_skipped_after_invalid_json(;|$)'
                THEN 'candidate_model_runtime_skipped_after_invalid_json'
            WHEN COALESCE(item ->> 'failure_reason', '') ~
                    '(^|;)candidate_model_runtime_skipped_after_adapter_failed(;|$)'
                THEN 'candidate_model_runtime_skipped_after_adapter_failed'
            WHEN COALESCE(item ->> 'failure_reason', '') ~
                    '(^|;)candidate_model_runtime_skipped_after_invalid_output(;|$)'
                THEN 'candidate_model_runtime_skipped_after_invalid_output'
            WHEN COALESCE(item ->> 'failure_reason', '') ~
                    '(^|;)candidate_model_zero_scoreable_companies(;|$)'
                THEN 'candidate_model_zero_scoreable_companies'
            WHEN COALESCE(item ->> 'failure_reason', '') ~
                    '(^|;)candidate_model_zero_companies(;|$)'
                THEN 'candidate_model_zero_companies'
            ELSE NULL
        END AS reason_code,
        COALESCE(item ->> 'failure_reason', '') AS failure_reason
    FROM raw_icps
),
bundle_failure_reasons AS (
    SELECT stage, reason_code, COUNT(*)::BIGINT AS failure_count
    FROM bundle_failure_reason_inputs
    WHERE stage IS NOT NULL AND reason_code IS NOT NULL
    GROUP BY stage, reason_code
    HAVING COUNT(*) > 0
),
reason_rows AS (
    SELECT stage, reason_code, 'companies'::TEXT AS unit, failure_count
    FROM label_failures
    UNION ALL
    SELECT stage, reason_code, 'icp_attempts'::TEXT AS unit, failure_count
    FROM execution_failures
    UNION ALL
    SELECT stage, reason_code, 'icp_attempts'::TEXT AS unit, failure_count
    FROM bundle_failure_reasons
),
bundle_candidate_coverage AS (
    SELECT
        b.reporting_candidate_id AS candidate_id,
        b.score_bundle_id,
        COUNT(p.funnel)::BIGINT AS bundle_icp_count,
        COUNT(p.funnel) FILTER (
            WHERE p.sourced >= p.fit_pass
              AND p.fit_pass >= p.verified
              AND p.verified >= p.intent_valid
              AND p.intent_valid >= p.scored
              AND p.scored >= 0
        )::BIGINT AS valid_funnel_count,
        MAX(bh.health_icp_count) FILTER (
            WHERE bh.health_contract_valid
        ) AS health_icp_count
    FROM latest_bundles b
    LEFT JOIN parsed_funnels p
      ON p.reporting_candidate_id = b.reporting_candidate_id
    LEFT JOIN bundle_health bh
      ON bh.reporting_candidate_id = b.reporting_candidate_id
    GROUP BY b.reporting_candidate_id, b.score_bundle_id
),
run_candidate_coverage AS (
    SELECT
        r.candidate_id,
        r.expected_icp_count,
        r.current_run_status,
        r.current_telemetry_degraded,
        r.score_bundle_id,
        COUNT(x.icp_execution_id)::BIGINT AS execution_ref_count,
        COUNT(x.icp_execution_id) FILTER (
            WHERE event.event_type IN ('completed', 'failed', 'cancelled', 'skipped')
        )::BIGINT AS terminal_execution_count
    FROM latest_scoring_runs r
    LEFT JOIN scoring_executions x
      ON x.scoring_run_id = r.scoring_run_id
    LEFT JOIN latest_execution_events event
      ON event.icp_execution_id = x.icp_execution_id
    GROUP BY
        r.candidate_id,
        r.expected_icp_count,
        r.current_run_status,
        r.current_telemetry_degraded,
        r.score_bundle_id
),
candidate_coverage_mismatches AS (
    SELECT COUNT(*)::BIGINT AS mismatch_count
    FROM bundle_candidate_coverage bundle
    FULL OUTER JOIN run_candidate_coverage run
      ON run.candidate_id = bundle.candidate_id
    WHERE COALESCE(bundle.candidate_id, run.candidate_id, '') = ''
       OR bundle.candidate_id IS NULL
       OR run.candidate_id IS NULL
       OR run.current_run_status IS DISTINCT FROM 'completed'
       OR COALESCE(run.current_telemetry_degraded, FALSE)
       OR run.score_bundle_id IS DISTINCT FROM bundle.score_bundle_id
       OR bundle.bundle_icp_count <> run.expected_icp_count
       OR bundle.valid_funnel_count <> run.expected_icp_count
       OR COALESCE(bundle.health_icp_count, -1) <> run.expected_icp_count
       OR run.execution_ref_count <> run.expected_icp_count
       OR run.terminal_execution_count <> run.expected_icp_count
),
bundle_icp_coverage AS (
    SELECT
        reporting_candidate_id AS candidate_id,
        icp_ref,
        COUNT(*)::BIGINT AS bundle_row_count,
        COUNT(*) FILTER (
            WHERE sourced >= fit_pass
              AND fit_pass >= verified
              AND verified >= intent_valid
              AND intent_valid >= scored
              AND scored >= 0
        )::BIGINT AS valid_funnel_count,
        COALESCE(SUM(sourced), 0)::BIGINT AS sourced_count
    FROM parsed_funnels
    GROUP BY reporting_candidate_id, icp_ref
),
execution_icp_coverage AS (
    SELECT
        x.candidate_id,
        NULLIF(TRIM(x.icp_ref), '') AS icp_ref,
        COUNT(*)::BIGINT AS execution_count,
        COUNT(*) FILTER (
            WHERE event.event_type IN ('completed', 'failed', 'cancelled', 'skipped')
        )::BIGINT AS terminal_count
    FROM scoring_executions x
    LEFT JOIN latest_execution_events event
      ON event.icp_execution_id = x.icp_execution_id
    GROUP BY x.candidate_id, NULLIF(TRIM(x.icp_ref), '')
),
label_icp_coverage AS (
    SELECT
        candidate_id,
        icp_ref,
        COUNT(*)::BIGINT AS label_count
    FROM label_outcomes
    GROUP BY candidate_id, icp_ref
),
icp_coverage_joined AS (
    SELECT
        COALESCE(bundle.candidate_id, execution.candidate_id, label.candidate_id)
            AS candidate_id,
        COALESCE(bundle.icp_ref, execution.icp_ref, label.icp_ref) AS icp_ref,
        bundle.bundle_row_count,
        bundle.valid_funnel_count,
        bundle.sourced_count,
        execution.execution_count,
        execution.terminal_count,
        label.label_count
    FROM bundle_icp_coverage bundle
    FULL OUTER JOIN execution_icp_coverage execution
      ON execution.candidate_id = bundle.candidate_id
     AND execution.icp_ref IS NOT DISTINCT FROM bundle.icp_ref
    FULL OUTER JOIN label_icp_coverage label
      ON label.candidate_id = COALESCE(bundle.candidate_id, execution.candidate_id)
     AND label.icp_ref IS NOT DISTINCT FROM COALESCE(bundle.icp_ref, execution.icp_ref)
),
icp_coverage_mismatches AS (
    SELECT COUNT(*)::BIGINT AS mismatch_count
    FROM icp_coverage_joined
    WHERE COALESCE(candidate_id, '') = ''
       OR COALESCE(icp_ref, '') = ''
       OR COALESCE(bundle_row_count, 0) <> 1
       OR COALESCE(valid_funnel_count, 0) <> 1
       OR COALESCE(execution_count, 0) <> 1
       OR COALESCE(terminal_count, 0) <> 1
       OR COALESCE(label_count, 0) <> COALESCE(sourced_count, 0)
),
coverage AS (
    SELECT
        (SELECT COUNT(*) FROM latest_bundles)::BIGINT AS bundle_count,
        (
            (SELECT mismatch_count FROM candidate_coverage_mismatches)
            + (SELECT mismatch_count FROM icp_coverage_mismatches)
        )::BIGINT AS coverage_mismatch_count,
        (SELECT COUNT(*) FROM bundle_health
          WHERE NOT health_contract_valid)::BIGINT AS invalid_scoring_health_count,
        (SELECT COUNT(*) FROM bundle_health
          WHERE health_contract_valid
            AND scoring_health ->> 'health_status' = 'degraded')::BIGINT
            AS degraded_scoring_health_count,
        (SELECT COALESCE(SUM(health_icp_count), 0) FROM bundle_health
          WHERE health_contract_valid) AS scoring_health_icp_count,
        (SELECT COUNT(*) FROM raw_icps)::BIGINT AS icp_row_count,
        (SELECT COUNT(*) FROM valid_funnels)::BIGINT AS funnel_row_count,
        (SELECT company_label_count FROM label_coverage) AS company_label_count,
        (SELECT company_failure_count FROM label_coverage) AS company_failure_count,
        (SELECT company_positive_count FROM label_coverage) AS company_positive_count,
        (SELECT COALESCE(SUM(failure_count), 0) FROM label_failures
          WHERE reason_code = 'other' OR stage = 'unclassified')::BIGINT
            AS unclassified_failure_count,
        (
            (SELECT infrastructure_label_failure_count FROM label_coverage)
            + (SELECT COALESCE(SUM(failure_count), 0) FROM execution_failures
               WHERE stage = 'infrastructure')
            + (SELECT COALESCE(SUM(failure_count), 0)
               FROM bundle_failure_reasons
               WHERE stage = 'infrastructure')
        )::BIGINT AS infrastructure_failure_count,
        (SELECT COUNT(*) FROM bundle_failure_reason_inputs
          WHERE reason_code IS NULL
            AND (
                failure_reason LIKE '%candidate_model_%'
                OR failure_reason LIKE '%reference_model_%'
            ))::BIGINT
            AS unclassified_icp_failure_count,
        (SELECT COUNT(*) FROM latest_scoring_runs)::BIGINT AS scoring_run_count,
        (SELECT COALESCE(SUM(expected_icp_count), 0) FROM latest_scoring_runs)::BIGINT
            AS expected_execution_count,
        (SELECT COUNT(*) FROM latest_execution_events)::BIGINT AS execution_count,
        (SELECT COUNT(*) FROM latest_execution_events
          WHERE event_type IN ('completed', 'failed', 'cancelled', 'skipped'))::BIGINT
            AS terminal_execution_count,
        (SELECT COUNT(*) FROM latest_execution_events
          WHERE event_type IN ('failed', 'cancelled'))::BIGINT
            AS failed_execution_count,
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
              OR c.coverage_mismatch_count > 0
              OR c.invalid_scoring_health_count > 0
              OR c.degraded_scoring_health_count > 0
              OR c.scoring_run_count = 0
              OR c.expected_execution_count = 0
              OR c.icp_row_count <> c.expected_execution_count
              OR c.funnel_row_count <> c.expected_execution_count
              OR c.scoring_health_icp_count <> c.expected_execution_count
              OR c.expected_execution_count <> c.terminal_execution_count
              OR c.nonterminal_execution_count > 0
              OR c.failed_execution_count > 0
              OR c.degraded_execution_count > 0
              OR c.infrastructure_failure_count > 0
              OR c.unclassified_icp_failure_count > 0
              OR c.company_failure_count <> (SELECT sourced - scored FROM stage_totals)
              OR c.company_positive_count <> (SELECT scored FROM stage_totals)
              OR c.company_label_count <> (SELECT sourced FROM stage_totals)
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
            WHERE model_revision_hash IS NOT NULL
        ) revisions
    ), '[]'::JSONB),
    'telemetry', (
        SELECT jsonb_build_object(
            'status', status,
            'bundle_count', bundle_count,
            'coverage_mismatch_count', coverage_mismatch_count,
            'invalid_scoring_health_count', invalid_scoring_health_count,
            'degraded_scoring_health_count', degraded_scoring_health_count,
            'scoring_health_icp_count', scoring_health_icp_count,
            'icp_row_count', icp_row_count,
            'funnel_row_count', funnel_row_count,
            'company_label_count', company_label_count,
            'company_failure_count', company_failure_count,
            'company_positive_count', company_positive_count,
            'detailed_reason_gap_count', GREATEST(
                0,
                (SELECT sourced - scored FROM stage_totals) - company_failure_count
            ),
            'detailed_reason_excess_count', GREATEST(
                0,
                company_failure_count - (SELECT sourced - scored FROM stage_totals)
            ),
            'detailed_pass_gap_count', GREATEST(
                0,
                (SELECT scored FROM stage_totals) - company_positive_count
            ),
            'detailed_pass_excess_count', GREATEST(
                0,
                company_positive_count - (SELECT scored FROM stage_totals)
            ),
            'company_label_gap_count', GREATEST(
                0,
                (SELECT sourced FROM stage_totals) - company_label_count
            ),
            'company_label_excess_count', GREATEST(
                0,
                company_label_count - (SELECT sourced FROM stage_totals)
            ),
            'unclassified_failure_count', unclassified_failure_count,
            'infrastructure_failure_count', infrastructure_failure_count,
            'unclassified_icp_failure_count', unclassified_icp_failure_count,
            'scoring_run_count', scoring_run_count,
            'expected_execution_count', expected_execution_count,
            'execution_count', execution_count,
            'terminal_execution_count', terminal_execution_count,
            'failed_execution_count', failed_execution_count,
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
    'Service-only counts report for Research Lab sourcing, company-fit, verifier, intent, and scoring failures. Missing historical detail is marked partial.';

NOTIFY pgrst, 'reload schema';

COMMIT;
