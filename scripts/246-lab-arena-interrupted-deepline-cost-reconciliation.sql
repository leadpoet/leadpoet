-- Reconcile exact Deepline billing after the database closes a dispatched
-- call whose worker lease expired, stage closed, or round was cancelled.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

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
        p_uncertainty_doc #>> '{call,reason}' IN
          ('transport_failure', 'missing_provider_cost')
        AND pg_catalog.jsonb_typeof(
              p_uncertainty_doc #> '{call,call_succeeded}'
            ) = 'boolean'
        AND (
          p_uncertainty_doc #>> '{call,reason}' <> 'transport_failure'
          OR (p_uncertainty_doc #>> '{call,call_succeeded}')::BOOLEAN IS FALSE
        )
        AND p_uncertainty_doc #>> '{call,deepline_request_id}' =
            p_reservation_doc ->> 'deepline_request_id'
        AND p_uncertainty_doc #>> '{call,deepline_operation}' =
            p_reservation_doc ->> 'tool'
        AND p_uncertainty_doc #>> '{call,credential_fingerprint}' =
            p_reservation_doc ->> 'credential_fingerprint'
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

DO $lab_arena_246_list_interrupted$
DECLARE
  v_definition TEXT;
  v_start TEXT := $start$            AND uncertainty.entry_doc ->> 'reason' = 'worker_reported'
$start$;
  v_end TEXT := $end$            AND NOT EXISTS (
              SELECT 1
              FROM public.lab_arena_ledger AS later
$end$;
  v_new TEXT := $new$            -- lab_arena_deepline_interrupted_cost_reconciliation
            AND public.lab_arena__deepline_cost_binding_v1(
                  uncertainty.entry_doc,
                  reservation.entry_doc,
                  uncertainty.call_identity
                )
$new$;
  v_start_at INTEGER;
  v_end_at INTEGER;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_deepline_interrupted_cost_reconciliation'
     ) > 0 THEN
    RETURN;
  END IF;
  v_start_at := pg_catalog.strpos(v_definition, v_start);
  v_end_at := pg_catalog.strpos(v_definition, v_end);
  IF v_start_at = 0 OR v_end_at <= v_start_at THEN
    RAISE EXCEPTION 'lab_arena_deepline_list_interrupted_shape_unexpected';
  END IF;
  EXECUTE pg_catalog.substr(v_definition, 1, v_start_at - 1)
    || v_new || pg_catalog.substr(v_definition, v_end_at);
END;
$lab_arena_246_list_interrupted$;

DO $lab_arena_246_settle_interrupted$
DECLARE
  v_definition TEXT;
  v_start TEXT := $start$     OR v_head.entry_doc ->> 'reason' IS DISTINCT FROM 'worker_reported'
$start$;
  v_end TEXT := $end$     OR v_head.entry_doc #>> '{call,credential_fingerprint}'
        IS DISTINCT FROM p_credential_fingerprint THEN
$end$;
  v_new TEXT := $new$     -- lab_arena_deepline_interrupted_cost_reconciliation
     OR NOT public.lab_arena__deepline_cost_binding_v1(
          v_head.entry_doc,
          pg_catalog.jsonb_build_object(
            'deepline_request_id', p_request_id,
            'tool', p_operation,
            'credential_fingerprint', p_credential_fingerprint
          ),
          p_call_identity
        ) THEN
$new$;
  v_start_at INTEGER;
  v_end_at INTEGER;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_reconcile_deepline_cost_v1(text,text,text,bigint,text,text,text,bigint,text)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_deepline_interrupted_cost_reconciliation'
     ) > 0 THEN
    RETURN;
  END IF;
  v_start_at := pg_catalog.strpos(v_definition, v_start);
  v_end_at := pg_catalog.strpos(v_definition, v_end);
  IF v_start_at = 0 OR v_end_at <= v_start_at THEN
    RAISE EXCEPTION 'lab_arena_deepline_settle_interrupted_shape_unexpected';
  END IF;
  v_end_at := v_end_at + pg_catalog.length(v_end);
  v_definition := pg_catalog.substr(v_definition, 1, v_start_at - 1)
    || v_new || pg_catalog.substr(v_definition, v_end_at);
  v_definition := pg_catalog.replace(
    v_definition,
    $old$      'reconciled_uncertainty_reason', 'worker_reported',
$old$,
    $new$      'reconciled_uncertainty_reason', v_head.entry_doc ->> 'reason',
$new$
  );
  EXECUTE v_definition;
END;
$lab_arena_246_settle_interrupted$;

DO $lab_arena_246_claim_interrupted$
DECLARE
  v_definition TEXT;
  v_start TEXT := $start$          AND uncertainty.entry_doc ->> 'reason' = 'worker_reported'
$start$;
  v_end TEXT := $end$          -- lab_arena_deepline_credential_deferral_bypass: a strictly bound
$end$;
  v_new TEXT := $new$          -- lab_arena_deepline_interrupted_retry_deferral
          AND uncertainty.entry_doc ->> 'reason' IN
              ('worker_reported', 'lease_expired')
          AND public.lab_arena__deepline_cost_binding_v1(
                uncertainty.entry_doc,
                reservation.entry_doc,
                uncertainty.call_identity
              )
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
$new$;
  v_start_at INTEGER;
  v_end_at INTEGER;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_deepline_interrupted_retry_deferral'
     ) > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_deepline_credential_deferral_bypass'
     ) = 0 THEN
    RAISE EXCEPTION 'apply migration 245 before migration 246';
  END IF;
  v_start_at := pg_catalog.strpos(v_definition, v_start);
  v_end_at := pg_catalog.strpos(v_definition, v_end);
  IF v_start_at = 0 OR v_end_at <= v_start_at THEN
    RAISE EXCEPTION 'lab_arena_deepline_claim_interrupted_shape_unexpected';
  END IF;
  EXECUTE pg_catalog.substr(v_definition, 1, v_start_at - 1)
    || v_new || pg_catalog.substr(v_definition, v_end_at);
END;
$lab_arena_246_claim_interrupted$;

CREATE OR REPLACE FUNCTION
  public.lab_arena_deepline_cost_reconciliation_schema_v1()
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog
AS $lab_arena_deepline_cost_reconciliation_schema_v1$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version',
      'leadpoet.lab_arena.deepline_cost_reconciliation_schema.v1',
    'version', 246
  );
$lab_arena_deepline_cost_reconciliation_schema_v1$;
ALTER FUNCTION public.lab_arena_deepline_cost_reconciliation_schema_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION
  public.lab_arena_deepline_cost_reconciliation_schema_v1()
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION
  public.lab_arena_deepline_cost_reconciliation_schema_v1()
  TO lab_arena_service;

NOTIFY pgrst, 'reload schema';
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;
