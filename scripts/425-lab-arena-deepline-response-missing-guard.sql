-- A recovered Deepline reply may fill a missing response, never replace a
-- terminal response already retained by the append-only ledger or overlay.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $response_guard$
DECLARE
  v_definition TEXT;
  v_old TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_recover_deepline_response_v1(text,text,text,text,text,text,text,text,bigint,jsonb,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'lab_arena_response_recovery_missing_guard_425') = 0 THEN
    v_old := $old$  IF v_response.call_identity IS NOT NULL THEN
    IF v_response.request_id IS DISTINCT FROM p_request_id THEN$old$;
    IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
      RAISE EXCEPTION 'Deepline response recovery replay preimage differs';
    END IF;
    v_definition := pg_catalog.replace(v_definition, v_old,
      $new$  IF v_response.call_identity IS NOT NULL THEN
    IF v_response.reservation_entry_id IS DISTINCT FROM v_reservation.entry_id
       OR v_response.run_id IS DISTINCT FROM p_run_id
       OR v_response.request_id IS DISTINCT FROM p_request_id
       OR v_response.terminal_response IS DISTINCT FROM p_terminal_response THEN$new$);

    v_old := $old$  IF v_head.entry_kind = 'dispatch' AND p_actual_microusd IS NULL THEN$old$;
    IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
      RAISE EXCEPTION 'Deepline response recovery eligibility preimage differs';
    END IF;
    v_definition := pg_catalog.replace(v_definition, v_old, $new$  -- lab_arena_response_recovery_missing_guard_425: persisted evidence is
  -- authoritative. A service readback alone cannot declare a response missing.
  IF v_response.call_identity IS NULL AND NOT COALESCE((
    (v_head.entry_kind = 'dispatch' AND v_head.terminal_response IS NULL)
    OR (v_head.entry_kind = 'uncertain'
      AND v_head.terminal_response IS NULL
      AND v_head.entry_doc ->> 'reason' = 'worker_reported'
      AND v_head.entry_doc #>> '{call,reason}' IN ('transport_failure', 'missing_provider_cost')
      AND v_head.entry_doc #> '{call,deepline_terminal_response}' IS NULL
      AND v_head.entry_doc #> '{call,account_failure_evidence}' IS NULL
      AND v_head.entry_doc #>> '{call,deepline_execution_key}' = p_execution_key
      AND v_head.entry_doc #>> '{call,credential_fingerprint}' = p_credential_fingerprint
      AND v_head.entry_doc #>> '{call,deepline_operation}' = p_operation
      AND v_head.entry_doc #>> '{call,deepline_request_id}'
        = v_reservation.entry_doc ->> 'deepline_request_id')
    OR (v_head.entry_kind = 'settlement'
      AND v_head.terminal_response -> 'call_succeeded' = 'false'::JSONB
      AND (
        v_head.terminal_response -> 'deepline_response_missing' = 'true'::JSONB
        OR (v_head.entry_doc ->> 'deepline_delayed_reconciliation' = 'true'
          AND EXISTS (SELECT 1 FROM public.lab_arena_ledger AS uncertain
            WHERE uncertain.entry_id = (v_head.entry_doc ->> 'reconciled_uncertainty_entry_id')::BIGINT
              AND uncertain.call_identity = p_call_identity
              AND uncertain.run_id = p_run_id
              AND uncertain.entry_kind = 'uncertain'
              AND uncertain.terminal_response IS NULL
              AND uncertain.entry_doc ->> 'reason' = 'worker_reported'
              AND uncertain.entry_doc #>> '{call,reason}' IN ('transport_failure', 'missing_provider_cost')
              AND uncertain.entry_doc #> '{call,deepline_terminal_response}' IS NULL
              AND uncertain.entry_doc #> '{call,account_failure_evidence}' IS NULL
              AND uncertain.entry_doc #>> '{call,deepline_execution_key}' = p_execution_key
              AND uncertain.entry_doc #>> '{call,credential_fingerprint}' = p_credential_fingerprint
              AND uncertain.entry_doc #>> '{call,deepline_operation}' = p_operation
              AND uncertain.entry_doc #>> '{call,deepline_request_id}'
                = v_reservation.entry_doc ->> 'deepline_request_id'))))
  ), FALSE) THEN
    RETURN pg_catalog.jsonb_build_object('status', 'conflict');
  END IF;
  IF v_head.entry_kind = 'dispatch' AND p_actual_microusd IS NULL THEN$new$);
    EXECUTE v_definition;
  END IF;
END;
$response_guard$;

CREATE OR REPLACE FUNCTION public.lab_arena_deepline_response_schema_v1()
RETURNS JSONB LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $response_readiness$
BEGIN
  IF pg_catalog.to_regprocedure('public.lab_arena_recover_deepline_response_v1(text,text,text,text,text,text,text,text,bigint,jsonb,integer)') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_deepline_call_responses') IS NULL
     OR pg_catalog.strpos(pg_catalog.pg_get_functiondef(
       'public.lab_arena_recover_deepline_response_v1(text,text,text,text,text,text,text,text,bigint,jsonb,integer)'::pg_catalog.regprocedure), 'lab_arena_response_recovery_missing_guard_425') = 0
     OR pg_catalog.strpos(pg_catalog.pg_get_functiondef(
       'public.lab_arena__call_state_view(public.lab_arena_ledger,public.lab_arena_runs)'::pg_catalog.regprocedure), 'lab_arena_deepline_call_responses') = 0
     OR pg_catalog.strpos(pg_catalog.pg_get_functiondef(
       'public.lab_arena__successful_call_cost_state(text,text,text)'::pg_catalog.regprocedure), 'lab_arena_deepline_call_responses') = 0
     OR pg_catalog.strpos(pg_catalog.pg_get_functiondef(
       'public.lab_arena__successful_icp_cost_state(text,text,integer)'::pg_catalog.regprocedure), 'lab_arena_deepline_call_responses') = 0 THEN
    RAISE EXCEPTION 'lab_arena_deepline_response_schema_incomplete' USING ERRCODE = '55000';
  END IF;
  RETURN pg_catalog.jsonb_build_object('status','ok', 'schema_version',
    'leadpoet.lab_arena.deepline_response_schema.v1', 'version',425);
END;
$response_readiness$;
ALTER FUNCTION public.lab_arena_deepline_response_schema_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_deepline_response_schema_v1() FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_deepline_response_schema_v1() TO lab_arena_service;
NOTIFY pgrst, 'reload schema';
COMMIT;
