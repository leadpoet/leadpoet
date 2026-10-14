-- A retained, bound terminal provider rejection closes the worker call, not
-- its price. Allow the existing guarded host expiry; keep the uncertain bill.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $closed_failed_execute_drain476$
DECLARE
  v_expiry_signature CONSTANT TEXT := 'public.lab_arena_expire_leases(text)';
  v_drain_signature CONSTANT TEXT := 'public.lab_arena__restart_drain_state_v1()';
  v_expiry_preimage CONSTANT TEXT := 'b871e809c70fb12dbacf918e12ca5e75f9d7721f2fdbcb02a3261035cc2d45ff';
  v_drain_preimage CONSTANT TEXT := '8d5b985c4186104bb35f4716fe785810d9bb2c580b2f3530cda5acdee3e77bc4';
  v_expiry_postimage CONSTANT TEXT := 'c7f883bae1aa2a85831e620c68870960ae5ad4abe2b652b876c21067e7e3aec3';
  v_drain_postimage CONSTANT TEXT := '5c1d6335fd6160b4240b0cd60b003f2892505850df6b510a4e275057ccf5980c';
  v_expiry TEXT;
  v_drain TEXT;
  v_identity JSONB;
  v_old_gate CONSTANT TEXT := $old_gate$                     AND dispatch.funding_source = heads.funding_source)
             )),FALSE)$old_gate$;
  v_new_gate CONSTANT TEXT := $new_gate$                     AND dispatch.funding_source = heads.funding_source)
             )
             -- lab_arena_closed_failed_execute_host_drain_v1: this is a
             -- completed provider rejection with an unknown fee, not a
             -- free call. Preserve the uncertain head for exact billing.
             OR (v_run.kind = 'execute'
               AND v_round.evaluation_date::DATE >= DATE '2026-10-14'
               AND heads.entry_kind = 'uncertain'
               AND heads.provider = 'deepline'
               AND heads.run_id = v_run.run_id
               AND heads.round_id = v_run.round_id
               AND heads.submission_id = v_run.submission_id
               AND heads.miner_hotkey = v_run.miner_hotkey
               AND heads.stage = v_run.stage
               AND heads.entry_doc ->> 'reason' = 'worker_reported'
               AND heads.entry_doc #>> '{call,reason}' = 'missing_provider_cost'
               AND heads.entry_doc #> '{call,call_succeeded}' = 'false'::JSONB
               AND heads.entry_doc #> '{call,provider_status}' IN
                 ('400'::JSONB,'403'::JSONB,'404'::JSONB,'422'::JSONB)
               AND heads.entry_doc #> '{call,deepline_async_job_ids}' IS NULL
               AND heads.entry_doc #> '{call,deepline_terminal_response,status}' =
                   heads.entry_doc #> '{call,provider_status}'
               AND heads.entry_doc #> '{call,deepline_terminal_response,call_succeeded}' =
                   'false'::JSONB
               AND pg_catalog.jsonb_typeof(heads.entry_doc #>
                   '{call,deepline_terminal_response,headers}') = 'object'
               AND pg_catalog.jsonb_typeof(heads.entry_doc #>
                   '{call,deepline_terminal_response,body_b64}') = 'string'
               AND pg_catalog.length(heads.entry_doc #>>
                   '{call,deepline_terminal_response,body_b64}') > 0
               AND EXISTS (
                 SELECT 1 FROM public.lab_arena_ledger AS reservation
                 WHERE reservation.call_identity = heads.call_identity
                   AND reservation.entry_kind = 'reservation'
                   AND (reservation.run_id,reservation.round_id,
                     reservation.submission_id,reservation.miner_hotkey,
                     reservation.stage,reservation.provider,
                     reservation.operation_id,reservation.funding_source)
                     IS NOT DISTINCT FROM
                     (heads.run_id,heads.round_id,heads.submission_id,
                      heads.miner_hotkey,heads.stage,heads.provider,
                      heads.operation_id,heads.funding_source)
                   AND heads.entry_doc #>> '{call,deepline_execution_key}' =
                       reservation.entry_doc ->> 'deepline_execution_key'
                   AND heads.entry_doc #>> '{call,credential_fingerprint}' =
                       reservation.entry_doc ->> 'credential_fingerprint'
                   AND heads.entry_doc #>> '{call,deepline_request_id}' =
                       reservation.entry_doc ->> 'deepline_request_id'
                   AND heads.entry_doc #>> '{call,deepline_operation}' =
                       reservation.entry_doc ->> 'tool'
                   AND public.lab_arena__deepline_cost_binding_v1(
                     heads.entry_doc,reservation.entry_doc,heads.call_identity)
                   AND EXISTS (
                     SELECT 1 FROM public.lab_arena_ledger AS dispatch
                     WHERE dispatch.call_identity = heads.call_identity
                       AND dispatch.entry_kind = 'dispatch'
                       AND (dispatch.run_id,dispatch.round_id,
                         dispatch.submission_id,dispatch.miner_hotkey,
                         dispatch.stage,dispatch.provider,
                         dispatch.operation_id,dispatch.funding_source)
                         IS NOT DISTINCT FROM
                         (heads.run_id,heads.round_id,heads.submission_id,
                          heads.miner_hotkey,heads.stage,heads.provider,
                          heads.operation_id,heads.funding_source))))
             ),FALSE)$new_gate$;
  v_old_reason CONSTANT TEXT := $old_reason$                ELSE 'authenticated_settled_execute_runtime_host_error' END$old_reason$;
  v_new_reason CONSTANT TEXT := $new_reason$                ELSE CASE WHEN EXISTS (
                  SELECT 1 FROM (
                    SELECT DISTINCT ON (cost.call_identity) cost.entry_kind
                    FROM public.lab_arena_ledger AS cost
                    WHERE cost.run_id = v_run.run_id
                    ORDER BY cost.call_identity,cost.entry_id DESC
                  ) AS heads WHERE heads.entry_kind = 'uncertain')
                  THEN 'authenticated_closed_failed_execute_runtime_host_error'
                  ELSE 'authenticated_settled_execute_runtime_host_error' END END$new_reason$;
  v_old_receipt CONSTANT TEXT := $old_receipt$            OR (runs.kind='execute' AND runs.terminal_doc->>'recovery_reason'=
              'authenticated_settled_execute_runtime_host_error'))$old_receipt$;
  v_new_receipt CONSTANT TEXT := $new_receipt$            OR (runs.kind='execute' AND runs.terminal_doc->>'recovery_reason' IN (
              'authenticated_settled_execute_runtime_host_error',
              'authenticated_closed_failed_execute_runtime_host_error')))$new_receipt$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_expiry,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_expiry_signature::REGPROCEDURE;
  IF v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array('lab_arena_owner',
    '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena closed failed expiry security shape differs' USING ERRCODE='55000';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_drain,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_drain_signature::REGPROCEDURE;
  IF v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array('lab_arena_owner',
    '{lab_arena_owner=X/lab_arena_owner}',TRUE,'s',
    ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena closed failed receipt security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_expiry,'sha256'),'hex')=v_expiry_postimage
     AND pg_catalog.encode(extensions.digest(v_drain,'sha256'),'hex')=v_drain_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_expiry,'sha256'),'hex')<>v_expiry_preimage
     OR pg_catalog.encode(extensions.digest(v_drain,'sha256'),'hex')<>v_drain_preimage
     OR (pg_catalog.length(v_expiry)-pg_catalog.length(
       pg_catalog.replace(v_expiry,v_old_gate,'')))<>pg_catalog.length(v_old_gate)
     OR (pg_catalog.length(v_expiry)-pg_catalog.length(
       pg_catalog.replace(v_expiry,v_old_reason,'')))<>pg_catalog.length(v_old_reason)
     OR (pg_catalog.length(v_drain)-pg_catalog.length(
       pg_catalog.replace(v_drain,v_old_receipt,'')))<>pg_catalog.length(v_old_receipt)
     OR (pg_catalog.length(v_drain)-pg_catalog.length(
       pg_catalog.replace(v_drain,v_old_gate,'')))<>pg_catalog.length(v_old_gate) THEN
    RAISE EXCEPTION 'Arena closed failed drain preimage differs' USING ERRCODE='55000';
  END IF;
  v_expiry:=pg_catalog.replace(pg_catalog.replace(v_expiry,v_old_gate,v_new_gate),
    v_old_reason,v_new_reason);
  v_drain:=pg_catalog.replace(pg_catalog.replace(v_drain,v_old_receipt,v_new_receipt),
    v_old_gate,pg_catalog.replace(pg_catalog.replace(v_new_gate,'v_run.','runs.'),
      'v_round.','recovery_round.'));
  IF pg_catalog.encode(extensions.digest(v_expiry,'sha256'),'hex')<>v_expiry_postimage
     OR pg_catalog.encode(extensions.digest(v_drain,'sha256'),'hex')<>v_drain_postimage THEN
    RAISE EXCEPTION 'Arena closed failed drain postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_expiry;
  EXECUTE v_drain;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
       v_expiry_signature::REGPROCEDURE),'sha256'),'hex')<>v_expiry_postimage
     OR pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
       v_drain_signature::REGPROCEDURE),'sha256'),'hex')<>v_drain_postimage THEN
    RAISE EXCEPTION 'Arena closed failed drain readback differs' USING ERRCODE='55000';
  END IF;
END;
$closed_failed_execute_drain476$;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;
