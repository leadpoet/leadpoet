-- Admit the exact worker settlement-failure head to the existing Deepline
-- billing reconciler. This migration changes only candidate eligibility; it
-- never settles an uncertain call or guesses a provider charge.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $settlement_failure_shape$
DECLARE
  v_list TEXT;
  v_settle TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)'::pg_catalog.regprocedure
  ) INTO v_list;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_reconcile_deepline_cost_v1(text,text,text,bigint,text,text,text,bigint,text)'::pg_catalog.regprocedure
  ) INTO v_settle;
  IF pg_catalog.strpos(v_list, 'lab_arena_deepline_interrupted_cost_reconciliation') = 0
     OR pg_catalog.strpos(v_list, 'lab_arena_deepline_native_billing_identity') = 0
     OR pg_catalog.strpos(v_settle, 'lab_arena_deepline_interrupted_cost_reconciliation') = 0
     OR pg_catalog.strpos(v_settle, 'lab_arena_deepline_native_billing_identity') = 0 THEN
    RAISE EXCEPTION 'lab_arena_settlement_failure_reconciliation_shape_unexpected';
  END IF;
END;
$settlement_failure_shape$;

CREATE OR REPLACE FUNCTION public.lab_arena__deepline_cost_binding_v1(
  p_uncertainty_doc JSONB,
  p_reservation_doc JSONB,
  p_call_identity TEXT
)
RETURNS BOOLEAN
LANGUAGE sql
IMMUTABLE
SET search_path = pg_catalog, public
AS $lab_arena__deepline_cost_binding_v1$
  SELECT COALESCE((
    pg_catalog.jsonb_typeof(p_uncertainty_doc) = 'object'
    AND pg_catalog.jsonb_typeof(p_reservation_doc) = 'object'
    AND p_call_identity ~ '^sha256:[0-9a-f]{64}$'
    AND p_reservation_doc ->> 'deepline_request_id' ~
        '^ctx-tool-[0-9a-f]{32}$'
    AND p_reservation_doc ->> 'deepline_request_id' =
        'ctx-tool-' || pg_catalog.substr(p_call_identity, 8, 32)
    AND p_reservation_doc ->> 'tool' ~ '^[a-z0-9_]{1,64}$'
    AND p_reservation_doc ->> 'credential_fingerprint' ~
        '^sha256:[0-9a-f]{64}$'
    AND CASE p_uncertainty_doc ->> 'reason'
      WHEN 'worker_reported' THEN
        CASE p_uncertainty_doc #>> '{call,reason}'
          WHEN 'transport_failure' THEN
            pg_catalog.jsonb_typeof(
              p_uncertainty_doc #> '{call,call_succeeded}'
            ) = 'boolean'
            AND (p_uncertainty_doc #>> '{call,call_succeeded}')::BOOLEAN IS FALSE
            AND p_uncertainty_doc #>> '{call,deepline_request_id}' =
                p_reservation_doc ->> 'deepline_request_id'
            AND p_uncertainty_doc #>> '{call,deepline_operation}' =
                p_reservation_doc ->> 'tool'
            AND p_uncertainty_doc #>> '{call,credential_fingerprint}' =
                p_reservation_doc ->> 'credential_fingerprint'
          WHEN 'missing_provider_cost' THEN
            pg_catalog.jsonb_typeof(
              p_uncertainty_doc #> '{call,call_succeeded}'
            ) = 'boolean'
            AND p_uncertainty_doc #>> '{call,deepline_request_id}' =
                p_reservation_doc ->> 'deepline_request_id'
            AND p_uncertainty_doc #>> '{call,deepline_operation}' =
                p_reservation_doc ->> 'tool'
            AND p_uncertainty_doc #>> '{call,credential_fingerprint}' =
                p_reservation_doc ->> 'credential_fingerprint'
          WHEN 'settle_failure' THEN
            -- The old worker reported only this bounded failure class. The
            -- native provider receipt is unavailable, so bind to the exact
            -- deterministic ID in the immutable reservation. New workers may
            -- retain all three binding fields and a native job receipt.
            pg_catalog.jsonb_typeof(p_uncertainty_doc -> 'call') = 'object'
            AND p_uncertainty_doc #>> '{call,failure_stage}' = 'settlement'
            AND p_uncertainty_doc #>> '{call,error_class}' = 'ArenaStoreError'
            AND pg_catalog.jsonb_typeof(
              p_uncertainty_doc #> '{call,call_succeeded}'
            ) = 'boolean'
            AND (
              (NOT (p_uncertainty_doc -> 'call' ? 'deepline_request_id')
               AND NOT (p_uncertainty_doc -> 'call' ? 'deepline_operation')
               AND NOT (p_uncertainty_doc -> 'call' ? 'credential_fingerprint')
               AND NOT (p_uncertainty_doc -> 'call' ? 'deepline_job_id'))
              OR
              (p_uncertainty_doc #>> '{call,deepline_request_id}' =
                   p_reservation_doc ->> 'deepline_request_id'
               AND p_uncertainty_doc #>> '{call,deepline_operation}' =
                   p_reservation_doc ->> 'tool'
               AND p_uncertainty_doc #>> '{call,credential_fingerprint}' =
                   p_reservation_doc ->> 'credential_fingerprint'
               AND (NOT (p_uncertainty_doc -> 'call' ? 'deepline_job_id')
                    OR p_uncertainty_doc #>> '{call,deepline_job_id}' ~
                      '^[a-z0-9]{3,8}::[a-z0-9]{1,16}-[0-9]{13}-[a-f0-9]{12,64}$'))
            )
          ELSE FALSE
        END
      WHEN 'lease_expired' THEN
        pg_catalog.jsonb_typeof(
          p_uncertainty_doc #> '{call,call_succeeded}'
        ) = 'boolean'
        AND (p_uncertainty_doc #>> '{call,call_succeeded}')::BOOLEAN IS FALSE
      WHEN 'stage_closed' THEN
        pg_catalog.jsonb_typeof(
          p_uncertainty_doc #> '{call,call_succeeded}'
        ) = 'boolean'
        AND (p_uncertainty_doc #>> '{call,call_succeeded}')::BOOLEAN IS FALSE
      WHEN 'round_cancelled' THEN
        pg_catalog.jsonb_typeof(
          p_uncertainty_doc #> '{call,call_succeeded}'
        ) = 'boolean'
        AND (p_uncertainty_doc #>> '{call,call_succeeded}')::BOOLEAN IS FALSE
      ELSE FALSE
    END
  ), FALSE);
$lab_arena__deepline_cost_binding_v1$;
ALTER FUNCTION public.lab_arena__deepline_cost_binding_v1(JSONB, JSONB, TEXT)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION
  public.lab_arena__deepline_cost_binding_v1(JSONB, JSONB, TEXT)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

COMMIT;
