-- Recover exact audit charges for completed host-funded judge requests after
-- their score run failed. Keep unknown amounts, admission and publication intact.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena__closed_host_score_success_uncertainty_v1(
  p_uncertain_entry_id BIGINT
)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $closed_host_score_success_uncertainty$
  SELECT COALESCE((
    SELECT
      uncertainty.entry_kind = 'uncertain'
      AND uncertainty.provider = 'deepline'
      AND uncertainty.funding_source = 'host'
      AND runs.kind = 'score'
      AND runs.status = 'failed'
      AND runs.terminal_cause IN (
        'lease_expired', 'worker_lost', 'result_rejected', 'provider_error',
        'stage_closed', 'judge_error', 'judge_timeout'
      )
      AND uncertainty.entry_doc ->> 'reason' = 'worker_reported'
      AND uncertainty.entry_doc #>> '{call,reason}' = 'missing_provider_cost'
      AND uncertainty.entry_doc #> '{call,call_succeeded}' = 'true'::JSONB
      AND uncertainty.entry_doc #> '{call,provider_status}' = '200'::JSONB
      AND reservation.entry_doc ->> 'deepline_execution_key' =
          'arena:' || pg_catalog.substr(uncertainty.call_identity, 8)
      AND uncertainty.entry_doc #> '{call,deepline_execution_key}' =
          reservation.entry_doc -> 'deepline_execution_key'
      AND pg_catalog.jsonb_typeof(
            uncertainty.entry_doc #> '{call,deepline_job_id}'
          ) = 'string'
      AND uncertainty.entry_doc #>> '{call,deepline_job_id}' ~
          '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'
      AND uncertainty.entry_doc #>> '{call,deepline_job_id}' NOT LIKE 'ctx-tool-%'
      -- The existing cost binding is authority for identity, not for a charge.
      -- Guard its typed worker result before evaluating its legacy casts.
      AND CASE WHEN uncertainty.entry_doc #> '{call,call_succeeded}' = 'true'::JSONB
        THEN public.lab_arena__deepline_cost_binding_v1(
          uncertainty.entry_doc, reservation.entry_doc, uncertainty.call_identity
        ) ELSE FALSE END
      AND EXISTS (
        SELECT 1
        FROM public.lab_arena_ledger AS dispatch
        WHERE dispatch.call_identity = uncertainty.call_identity
          AND dispatch.entry_kind = 'dispatch'
          AND dispatch.run_id = uncertainty.run_id
          AND dispatch.round_id = uncertainty.round_id
          AND dispatch.submission_id = uncertainty.submission_id
          AND dispatch.miner_hotkey = uncertainty.miner_hotkey
          AND dispatch.stage = uncertainty.stage
          AND dispatch.provider = uncertainty.provider
          AND dispatch.operation_id = uncertainty.operation_id
          AND dispatch.funding_source = uncertainty.funding_source
      )
    FROM public.lab_arena_ledger AS uncertainty
    JOIN public.lab_arena_ledger AS reservation
      ON reservation.call_identity = uncertainty.call_identity
     AND reservation.entry_kind = 'reservation'
     AND reservation.run_id = uncertainty.run_id
     AND reservation.round_id = uncertainty.round_id
     AND reservation.submission_id = uncertainty.submission_id
     AND reservation.miner_hotkey = uncertainty.miner_hotkey
     AND reservation.stage = uncertainty.stage
     AND reservation.provider = uncertainty.provider
     AND reservation.operation_id = uncertainty.operation_id
     AND reservation.funding_source = uncertainty.funding_source
    JOIN public.lab_arena_runs AS runs
      ON runs.run_id = uncertainty.run_id
     AND runs.round_id = uncertainty.round_id
     AND runs.submission_id = uncertainty.submission_id
     AND runs.miner_hotkey = uncertainty.miner_hotkey
     AND runs.stage = uncertainty.stage
    WHERE uncertainty.entry_id = p_uncertain_entry_id
  ), FALSE);
$closed_host_score_success_uncertainty$;
ALTER FUNCTION public.lab_arena__closed_host_score_success_uncertainty_v1(BIGINT)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__closed_host_score_success_uncertainty_v1(BIGINT)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

DO $closed_host_score_billing$
DECLARE
  v_signature TEXT;
  v_definition TEXT;
  v_argument TEXT;
  v_old TEXT;
  v_new TEXT;
  v_helper TEXT := 'public.lab_arena__closed_host_score_success_uncertainty_v1';
BEGIN
  -- Three closed-billing gates, including both currently deployed list and
  -- settlement versions. No admission or scoring function is changed.
  FOREACH v_signature IN ARRAY ARRAY[
    'public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)',
    'public.lab_arena_list_deepline_cost_reconciliations_v2(text,text,bigint,integer,boolean)',
    'public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)',
    'public.lab_arena_reconcile_deepline_cost_v1(text,text,text,bigint,text,text,text,bigint,text)',
    'public.lab_arena_reconcile_deepline_cost_v2(text,text,text,bigint,text,text,text,bigint,text,text,text)'
  ] LOOP
    SELECT pg_catalog.pg_get_functiondef(v_signature::pg_catalog.regprocedure)
      INTO v_definition;
    IF pg_catalog.strpos(v_signature, 'list_deepline') > 0 THEN
      v_argument := E'\n                      uncertainty.entry_id\n                    ';
    ELSIF pg_catalog.strpos(v_signature, 'next_closed') > 0 THEN
      v_argument := 'uncertainty.entry_id';
    ELSE
      -- Preserve replay eligibility using the original uncertainty after a
      -- delayed settlement. This grants no new permission to change a round.
      v_argument := $argument$
         CASE
           WHEN v_head.entry_kind = 'uncertain' THEN v_head.entry_id
           WHEN v_head.entry_kind = 'settlement'
             AND v_head.entry_doc ->> 'deepline_delayed_reconciliation' = 'true'
             AND v_head.entry_doc ->> 'reconciled_uncertainty_entry_id' ~ '^[0-9]{1,18}$'
           THEN (v_head.entry_doc ->> 'reconciled_uncertainty_entry_id')::BIGINT
           ELSE NULL
         END
       $argument$;
    END IF;
    v_old := 'public.lab_arena__closed_score_dynamic_uncertainty_v1(' || v_argument || ')';
    v_new := '(' || v_old || ' OR ' || v_helper || '(' || v_argument || '))';
    IF pg_catalog.strpos(v_definition, v_helper) > 0 THEN
      IF (pg_catalog.length(v_definition) - pg_catalog.length(
          pg_catalog.replace(v_definition, v_new, ''))) / pg_catalog.length(v_new) <> 1
         OR (pg_catalog.length(v_definition) - pg_catalog.length(
           pg_catalog.replace(v_definition, v_helper, ''))) / pg_catalog.length(v_helper) <> 1 THEN
        RAISE EXCEPTION 'closed host billing applied shape unexpected: %', v_signature;
      END IF;
      CONTINUE;
    END IF;
    IF pg_catalog.strpos(v_definition, 'lab_arena_closed_scoring_billing_reconciliation') = 0
       AND pg_catalog.strpos(v_signature, 'next_closed') = 0 THEN
      RAISE EXCEPTION 'closed billing prerequisite missing: %', v_signature;
    END IF;
    IF (pg_catalog.length(v_definition) - pg_catalog.length(
        pg_catalog.replace(v_definition, v_old, ''))) / pg_catalog.length(v_old) <> 1 THEN
      RAISE EXCEPTION 'closed host billing preimage unexpected: %', v_signature;
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  END LOOP;
END;
$closed_host_score_billing$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
