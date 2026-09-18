-- Permit the service's documented null final score only when a non-baseline
-- finalist has no accepted execution among its latest scored attempts.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

DO $requires_289$
DECLARE
  v_schema JSONB;
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena__per_icp_publication_valid(text,jsonb)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_per_icp_cost_schema_v1()'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply migration 289 before migration 292';
  END IF;
  SELECT public.lab_arena_per_icp_cost_schema_v1() INTO v_schema;
  IF v_schema ->> 'schema_version' IS DISTINCT FROM
       'leadpoet.lab_arena.per_icp_cost_schema.v1'
     OR (v_schema ->> 'version')::INTEGER <> 289
     OR v_schema ->> 'policy' IS DISTINCT FROM
       'successful_calls_per_icp_v1' THEN
    RAISE EXCEPTION 'migration 289 schema differs';
  END IF;
END;
$requires_289$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena__per_icp_publication_valid(
  p_round_id TEXT, p_ranking JSONB
)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $per_icp_publication_valid$
DECLARE
  v_submission_id TEXT := p_ranking ->> 'submission_id';
  v_cost JSONB := p_ranking -> 'cost_summary';
  v_row JSONB;
  v_summary JSONB;
  v_expected JSONB;
  v_position INTEGER;
  v_expected_score NUMERIC;
  v_hard_reason TEXT;
  v_total_spend BIGINT := 0;
  v_total_qualified BIGINT := 0;
  v_total_returned BIGINT := 0;
  v_eligible_icps BIGINT := 0;
  v_expected_execution JSONB;
  v_expected_judge JSONB;
  v_score_count BIGINT;
  v_accepted_count BIGINT;
  v_is_baseline BOOLEAN;
BEGIN
  IF v_cost ->> 'sourcing_cost_eligibility_policy'
       IS DISTINCT FROM 'successful_calls_per_icp_v1'
     OR pg_catalog.jsonb_typeof(v_cost -> 'per_icp') IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(v_cost -> 'per_icp') <> 20 THEN
    RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch' USING ERRCODE = '22023';
  END IF;
  FOR v_position IN 0..19 LOOP
    SELECT value INTO v_row FROM pg_catalog.jsonb_array_elements(v_cost -> 'per_icp')
      WHERE value ->> 'icp_position' = v_position::TEXT;
    IF v_row IS NULL THEN
      RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch' USING ERRCODE = '22023';
    END IF;
    v_summary := public.lab_arena__integrity_submission_summary(
      p_round_id, v_submission_id, ARRAY[v_position]
    );
    IF NOT COALESCE((v_summary ->> 'valid')::BOOLEAN, FALSE) THEN
      RAISE EXCEPTION 'lab_arena_publication_scoring_incomplete'
        USING ERRCODE = '22023';
    END IF;
    v_expected := public.lab_arena_icp_cost_eligibility(
      p_round_id, v_submission_id, v_position,
      (v_summary ->> 'qualified_company_count')::INTEGER
    );
    IF (v_row ->> 'qualified_company_count')::BIGINT IS DISTINCT FROM
         (v_summary ->> 'qualified_company_count')::BIGINT
       OR (v_row ->> 'eligible')::BOOLEAN IS DISTINCT FROM
         (v_expected ->> 'eligible')::BOOLEAN
       OR v_row ->> 'eligibility_reason' IS DISTINCT FROM
         v_expected ->> 'eligibility_reason'
       OR (v_row ->> 'competition_sourcing_microusd')::BIGINT IS DISTINCT FROM
         (v_expected ->> 'competition_sourcing_microusd')::BIGINT
       OR (v_row ->> 'eligibility_cap_microusd')::BIGINT IS DISTINCT FROM
         (v_expected ->> 'eligibility_cap_microusd')::BIGINT
       OR (v_row ->> 'execution_icp_cap_microusd')::BIGINT IS DISTINCT FROM
         (v_expected ->> 'execution_icp_cap_microusd')::BIGINT
       OR (v_row ->> 'cost_per_company_cap_microusd')::BIGINT IS DISTINCT FROM
         (v_expected ->> 'cost_per_company_cap_microusd')::BIGINT
       OR v_row -> 'execution' IS DISTINCT FROM v_expected -> 'execution' THEN
      RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch' USING ERRCODE = '22023';
    END IF;
    v_total_spend := v_total_spend
      + (v_expected ->> 'competition_sourcing_microusd')::BIGINT;
    v_total_qualified := v_total_qualified
      + (v_summary ->> 'qualified_company_count')::BIGINT;
    IF pg_catalog.jsonb_typeof(v_row -> 'returned_company_count') IS DISTINCT FROM 'number'
       OR (v_row ->> 'returned_company_count')::NUMERIC NOT BETWEEN 0 AND 5
       OR (v_row ->> 'returned_company_count')::NUMERIC <>
         pg_catalog.trunc((v_row ->> 'returned_company_count')::NUMERIC) THEN
      RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch' USING ERRCODE = '22023';
    END IF;
    v_total_returned := v_total_returned
      + (v_row ->> 'returned_company_count')::BIGINT;
    IF (v_expected ->> 'eligible')::BOOLEAN THEN
      v_eligible_icps := v_eligible_icps + 1;
    END IF;
    IF v_hard_reason IS NULL AND v_expected ->> 'eligibility_reason' IN (
      'provider_calls_inflight','provider_cost_uncertain'
    ) THEN v_hard_reason := v_expected ->> 'eligibility_reason'; END IF;
  END LOOP;
  -- lab_arena_per_icp_null_final_score_v1: match the service's selected
  -- latest scored attempts, while deriving baseline identity from stored data.
  SELECT COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE)
  INTO v_is_baseline
  FROM public.lab_arena_rounds AS round_row
  CROSS JOIN LATERAL pg_catalog.jsonb_array_elements(
    round_row.participants
  ) AS participant
  WHERE round_row.round_id = p_round_id
    AND participant ->> 'submission_id' = v_submission_id
  LIMIT 1;
  IF v_is_baseline IS NULL THEN
    RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch'
      USING ERRCODE = '22023';
  END IF;
  v_expected_execution := public.lab_arena__cost_kind_summary_v1(
    v_submission_id, 'execute'
  );
  v_expected_judge := public.lab_arena__cost_kind_summary_v1(
    v_submission_id, 'score'
  );
  SELECT pg_catalog.count(*),
         pg_catalog.count(*) FILTER (WHERE runs.terminal_cause = 'accepted'),
         pg_catalog.avg(CASE WHEN
      (public.lab_arena_icp_cost_eligibility(
        p_round_id, v_submission_id, runs.icp_position,
        (public.lab_arena__integrity_submission_summary(
          p_round_id, v_submission_id, ARRAY[runs.icp_position]
        ) ->> 'qualified_company_count')::INTEGER
      ) ->> 'eligible')::BOOLEAN
    THEN runs.per_icp_score ELSE 0 END)
  INTO v_score_count, v_accepted_count, v_expected_score
  FROM (
    SELECT DISTINCT ON (icp_position)
      icp_position, per_icp_score, terminal_cause
    FROM public.lab_arena_runs
    WHERE round_id = p_round_id AND submission_id = v_submission_id
      AND kind = 'execute' AND per_icp_score IS NOT NULL
    ORDER BY icp_position, attempt DESC
  ) AS runs;
  IF v_score_count <> 20
     OR NOT (p_ranking ? 'final_score')
     OR pg_catalog.jsonb_typeof(p_ranking -> 'final_score')
          NOT IN ('number', 'null')
     OR (v_accepted_count = 0 AND (
       v_is_baseline
       OR p_ranking -> 'final_score' IS DISTINCT FROM 'null'::JSONB
     ))
     OR (v_accepted_count > 0 AND (
       pg_catalog.jsonb_typeof(p_ranking -> 'final_score')
         IS DISTINCT FROM 'number'
       OR (p_ranking ->> 'final_score')::NUMERIC IS DISTINCT FROM v_expected_score
     ))
     OR (p_ranking ->> 'eligible')::BOOLEAN IS DISTINCT FROM (v_hard_reason IS NULL)
     OR p_ranking ->> 'eligibility_reason' IS DISTINCT FROM COALESCE(v_hard_reason,'eligible')
     OR (v_cost ->> 'competition_sourcing_microusd')::BIGINT IS DISTINCT FROM v_total_spend
     OR (v_cost ->> 'qualified_company_count')::BIGINT IS DISTINCT FROM v_total_qualified
     OR (v_cost ->> 'returned_company_count')::BIGINT IS DISTINCT FROM v_total_returned
     OR (v_cost ->> 'eligible_icp_count')::BIGINT IS DISTINCT FROM v_eligible_icps
     OR (v_cost ->> 'execution_icp_cap_microusd')::BIGINT IS DISTINCT FROM
       (SELECT (configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
        FROM public.lab_arena_rounds WHERE round_id = p_round_id)
     OR (v_cost ->> 'cost_per_company_cap_microusd')::BIGINT IS DISTINCT FROM
       (SELECT (configuration_doc ->> 'cost_per_company_microusd')::BIGINT
        FROM public.lab_arena_rounds WHERE round_id = p_round_id)
     OR v_cost -> 'execution' IS DISTINCT FROM v_expected_execution
     OR v_cost -> 'judge' IS DISTINCT FROM v_expected_judge THEN
    RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch' USING ERRCODE = '22023';
  END IF;
  RETURN TRUE;
END;
$per_icp_publication_valid$;
ALTER FUNCTION public.lab_arena__per_icp_publication_valid(TEXT,JSONB)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__per_icp_publication_valid(TEXT,JSONB)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

DO $verify_patch$
DECLARE
  v_definition TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena__per_icp_publication_valid(text,jsonb)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_per_icp_null_final_score_v1'
     ) = 0
     OR pg_catalog.strpos(v_definition, 'v_accepted_count') = 0
     OR pg_catalog.strpos(v_definition, 'v_is_baseline') = 0 THEN
    RAISE EXCEPTION 'lab_arena per-ICP null-score publication patch failed';
  END IF;
END;
$verify_patch$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
