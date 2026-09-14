-- Apply the same final cost-efficiency consequence to the daily baseline as
-- to challengers. The raw accepted per-ICP scores and provider ledger remain
-- immutable; only the published baseline score becomes zero when its exact,
-- receipt-backed cost result exceeds either frozen cost limit.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

DO $requires_251$
DECLARE
  v_schema JSONB;
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_integrity_publication_guard_v1()'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_twenty_icp_promotion_schema_v1()'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_successful_call_cost_schema_v1()'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply migrations 230 and 251 before migration 254';
  END IF;
  SELECT public.lab_arena_twenty_icp_promotion_schema_v1() INTO v_schema;
  IF v_schema ->> 'schema_version' IS DISTINCT FROM
       'leadpoet.lab_arena.twenty_icp_promotion_schema.v1'
     OR (v_schema ->> 'version')::INTEGER < 251 THEN
    RAISE EXCEPTION 'migration 251 schema differs';
  END IF;
  SELECT public.lab_arena_successful_call_cost_schema_v1() INTO v_schema;
  IF v_schema ->> 'schema_version' IS DISTINCT FROM
       'leadpoet.lab_arena.successful_call_cost_schema.v1'
     OR (v_schema ->> 'version')::INTEGER < 230
     OR v_schema ->> 'policy' IS DISTINCT FROM 'successful_calls_v1' THEN
    RAISE EXCEPTION 'migration 230 schema differs';
  END IF;
END;
$requires_251$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- Preserve migration 251's complete publication guard and replace only its
-- score comparison. The database recomputes eligibility before it permits a
-- zero baseline; a caller cannot forge the published reason or score.
DO $baseline_cost_score$
DECLARE
  v_definition TEXT;
  v_declaration_old TEXT := $old$
  v_allowed_failure BOOLEAN;
$old$;
  v_declaration_new TEXT := $new$
  v_allowed_failure BOOLEAN;
  v_expected_score NUMERIC;
$new$;
  v_old TEXT := $old$
    v_main := public.lab_arena__integrity_submission_summary(
      NEW.round_id, v_submission_id,
      ARRAY(SELECT pg_catalog.generate_series(0, 19))
    );
    IF NOT COALESCE((v_main ->> 'valid')::BOOLEAN, FALSE)
       OR NOT (v_ranking ? 'final_score')
       OR (v_main -> 'score' = 'null'::JSONB
           AND v_ranking -> 'final_score' <> 'null'::JSONB)
       OR (v_main -> 'score' <> 'null'::JSONB AND (
         pg_catalog.jsonb_typeof(v_ranking -> 'final_score')
           IS DISTINCT FROM 'number'
         OR (v_ranking ->> 'final_score')::NUMERIC
           IS DISTINCT FROM (v_main ->> 'score')::NUMERIC
       )) THEN
      RAISE EXCEPTION 'lab_arena_final_score_mismatch'
        USING ERRCODE = '22023';
    END IF;
    v_eligibility := public.lab_arena__integrity_eligibility(
      NEW.round_id, v_submission_id,
      ARRAY(SELECT pg_catalog.generate_series(0, 19))
    );
$old$;
  v_new TEXT := $new$
    v_main := public.lab_arena__integrity_submission_summary(
      NEW.round_id, v_submission_id,
      ARRAY(SELECT pg_catalog.generate_series(0, 19))
    );
    v_eligibility := public.lab_arena__integrity_eligibility(
      NEW.round_id, v_submission_id,
      ARRAY(SELECT pg_catalog.generate_series(0, 19))
    );
    -- lab_arena_baseline_cost_ineligible_score_v1
    v_expected_score := (v_main ->> 'score')::NUMERIC;
    IF NOT COALESCE((v_main ->> 'valid')::BOOLEAN, FALSE)
       OR NOT (v_ranking ? 'final_score')
       OR (v_main -> 'score' = 'null'::JSONB AND (
         v_submission_id = v_baseline_id
         OR v_ranking -> 'final_score' <> 'null'::JSONB
       ))
       OR (v_main -> 'score' <> 'null'::JSONB AND (
         pg_catalog.jsonb_typeof(v_ranking -> 'final_score')
           IS DISTINCT FROM 'number'
         OR (v_ranking ->> 'final_score')::NUMERIC IS DISTINCT FROM
           CASE WHEN v_submission_id = v_baseline_id AND COALESCE(
             v_eligibility ->> 'eligibility_reason' IN (
               'cost_per_company_exceeded', 'execution_cap_exceeded'
             ), FALSE
           ) THEN 0 ELSE v_expected_score END
       )) THEN
      RAISE EXCEPTION 'lab_arena_final_score_mismatch'
        USING ERRCODE = '22023';
    END IF;
$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure(
    'public.lab_arena_integrity_publication_guard_v1()'
  )) INTO v_definition;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_baseline_cost_ineligible_score_v1'
     ) > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
    RAISE EXCEPTION 'lab_arena_integrity_publication_guard_shape_unexpected';
  END IF;
  IF pg_catalog.strpos(v_definition, v_declaration_old) = 0 THEN
    RAISE EXCEPTION 'lab_arena_integrity_publication_guard_declaration_unexpected';
  END IF;
  v_definition := pg_catalog.replace(
    v_definition, v_declaration_old, v_declaration_new
  );
  EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
END;
$baseline_cost_score$;

DO $verify_guard$
DECLARE
  v_definition TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure(
    'public.lab_arena_integrity_publication_guard_v1()'
  )) INTO v_definition;
  IF v_definition IS NULL
     OR pg_catalog.strpos(
       v_definition, 'lab_arena_baseline_cost_ineligible_score_v1'
     ) = 0
     OR pg_catalog.strpos(v_definition, 'cost_per_company_exceeded') = 0
     OR pg_catalog.strpos(v_definition, 'execution_cap_exceeded') = 0
     OR pg_catalog.strpos(
       v_definition, 'lab_arena__integrity_eligibility'
     ) = 0
     OR pg_catalog.strpos(
       v_definition, 'lab_arena__integrity_submission_summary'
     ) = 0
     OR pg_catalog.strpos(v_definition, 'v_expected_score NUMERIC') = 0
     OR pg_catalog.strpos(v_definition,
       $expected$CASE WHEN v_submission_id = v_baseline_id AND COALESCE(
             v_eligibility ->> 'eligibility_reason' IN (
               'cost_per_company_exceeded', 'execution_cap_exceeded'
             ), FALSE
           ) THEN 0 ELSE v_expected_score END$expected$
     ) = 0 THEN
    RAISE EXCEPTION 'lab_arena_baseline_cost_eligibility_guard_invalid';
  END IF;
END;
$verify_guard$;

-- This additive marker leaves the old release's startup check valid while the
-- new release fails closed if the receipt-backed trigger is not installed.
-- Migration 254 and the new publisher are a required pair: after migration,
-- an old publisher's nonzero cost-ineligible baseline is rejected safely.
CREATE OR REPLACE FUNCTION public.lab_arena_baseline_cost_eligibility_schema_v1()
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog
AS $schema$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version',
      'leadpoet.lab_arena.baseline_cost_eligibility_schema.v1',
    'version', 254,
    'policy', 'zero_baseline_cost_ineligible_v1'
  );
$schema$;
ALTER FUNCTION public.lab_arena_baseline_cost_eligibility_schema_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_baseline_cost_eligibility_schema_v1()
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_baseline_cost_eligibility_schema_v1()
  TO lab_arena_service;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
