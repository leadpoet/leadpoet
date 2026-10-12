-- Preserve terminal full-context legacy review rows in the existing failed
-- artifact. No score, cache row, or retry allowance is created by a checkpoint.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

CREATE OR REPLACE FUNCTION public.lab_arena__legacy_partial_bound(
  p_run public.lab_arena_runs, p_receipt JSONB, p_output_ref TEXT
) RETURNS BOOLEAN
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $legacy_partial_bound$
  SELECT COALESCE(
    p_run.kind = 'score'
    AND pg_catalog.jsonb_typeof(p_receipt) = 'object'
    AND p_receipt ->> 'schema_version' = 'leadpoet.lab_arena.legacy_partial.v1'
    AND p_receipt ->> 'source_score_run_id' = p_run.run_id
    AND p_receipt ->> 'scored_run_id' = p_run.scored_run_id
    AND p_receipt ->> 'round_id' = p_run.round_id
    AND p_receipt ->> 'submission_id' = p_run.submission_id
    AND p_receipt ->> 'miner_hotkey' = p_run.miner_hotkey
    AND p_receipt ->> 'stage' = p_run.stage::TEXT
    AND p_receipt ->> 'icp_position' = p_run.icp_position::TEXT
    AND p_receipt ->> 'source_runner_hotkey' = p_run.runner_hotkey
    AND p_receipt ->> 'source_claim_request_id' = p_run.claim_request_id
    AND p_receipt ->> 'source_claim_request_hash' = p_run.claim_request_hash
    AND p_receipt ->> 'source_lease_generation' = p_run.lease_generation::TEXT
    AND pg_catalog.jsonb_typeof(p_receipt -> 'runner_authority_exclusions') = 'array'
    AND p_receipt -> 'runner_authority_exclusions' = p_run.claim_response -> 'runner_authority_exclusions'
    AND p_receipt ->> 'source_output_ref' = p_output_ref
    AND p_output_ref = 'arena/' || p_run.round_id || '/scores/items/' || p_run.run_id || '.json'
    AND p_receipt ->> 'source_output_hash' ~ '^sha256:[0-9a-f]{64}$'
    AND p_receipt ->> 'scoring_input_hash' ~ '^sha256:[0-9a-f]{64}$'
    AND p_receipt ->> 'execution_output_hash' ~ '^sha256:[0-9a-f]{64}$'
    AND p_receipt ->> 'source_completion_request_hash' ~ '^sha256:[0-9a-f]{64}$'
    AND pg_catalog.jsonb_typeof(p_receipt -> 'terminal_count') = 'number'
    AND p_receipt ->> 'terminal_count' ~ '^[1-9][0-9]{0,2}$'
    AND (p_receipt ->> 'terminal_count')::INTEGER <= 200
    AND EXISTS (
      SELECT 1 FROM public.lab_arena_rounds AS frozen
      WHERE frozen.round_id = p_run.round_id
        AND frozen.configuration_doc ->> 'integrity_policy' = 'arena_integrity_v1'
        AND NOT frozen.configuration_doc ? 'company_quality_policy'
        AND NOT (frozen.configuration_doc -> 'scorer_policy') ? 'company_quality_policy'
        AND p_receipt -> 'scorer_policy' = frozen.configuration_doc -> 'scorer_policy'
        AND p_receipt ->> 'scorer_image_digest' = frozen.configuration_doc ->> 'scorer_image_digest'
        AND p_receipt ->> 'scorer_image_reference' = frozen.configuration_doc ->> 'scorer_image_reference'
        AND p_receipt ->> 'evaluation_date' = frozen.evaluation_date::TEXT
        AND frozen.configuration_doc #>> '{scorer_policy,scoring_adapter_version}' IN
            ('qualification_integrity_v2', 'qualification_contacts_v3')
    ), FALSE);
$legacy_partial_bound$;
ALTER FUNCTION public.lab_arena__legacy_partial_bound(public.lab_arena_runs,JSONB,TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__legacy_partial_bound(public.lab_arena_runs,JSONB,TEXT) FROM PUBLIC;

-- Migration 465's quality allowance stays intact. The service injects this
-- receipt after full-input validation; workers cannot submit result_doc fields.
-- Existing RPC lease locks, accounting, immutable terminals, and retry limits
-- still control whether the receipt is committed.
DO $legacy_partial_completion$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$AND pg_catalog.jsonb_typeof(partial_source.company_judgment_refs) = 'array'$old$;
  v_new TEXT := $new$AND (
                 pg_catalog.jsonb_typeof(partial_source.company_judgment_refs) = 'array'
                 OR public.lab_arena__legacy_partial_bound(
                   partial_source, p_result -> 'legacy_partial_checkpoint', p_output_ref)
               )$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef('public.lab_arena_complete_attempt(text,text,jsonb,text,text)'::regprocedure)
  INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'p_result -> ''legacy_partial_checkpoint''') = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
      RAISE EXCEPTION 'lab_arena_legacy_partial_completion_requires_465';
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  END IF;
END;
$legacy_partial_completion$;

-- Owner credit retry admits only the bound failed legacy artifact. Preserve
-- every existing owner, proof, deadline, ledger, attempt, and race gate.
DO $legacy_partial_credit_retry$
DECLARE
  v_definition TEXT;
  v_old TEXT := 'previous.output_ref IS NULL';
  v_new TEXT := $new$(previous.output_ref IS NULL
        OR public.lab_arena__legacy_partial_bound(
          previous, previous.result_doc -> 'legacy_partial_checkpoint', previous.output_ref))$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef('public.lab_arena_retry_credit_failures_v1(text,text,text,text)'::regprocedure)
  INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'previous.result_doc -> ''legacy_partial_checkpoint''') = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
      RAISE EXCEPTION 'lab_arena_legacy_partial_credit_patch_missing';
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  END IF;
END;
$legacy_partial_credit_retry$;
NOTIFY pgrst, 'reload schema';
COMMIT;
