-- Training-only delivery evidence from the existing private recovery row.
-- Never change the append-only accounting ledger or its terminal response.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

-- Only an exact replay of this migration may replace an existing function.
-- A same-name overload, changed body, owner, security mode or ACL fails closed.
DO $function_preimage$
DECLARE v_function pg_catalog.pg_proc; v_count INTEGER;
BEGIN
  SELECT count(*) INTO v_count FROM pg_catalog.pg_proc AS p
    JOIN pg_catalog.pg_namespace AS n ON n.oid = p.pronamespace
    WHERE n.nspname = 'public' AND p.proname = 'lab_arena_training_deepline_response_v1';
  IF v_count > 0 THEN
    SELECT p.* INTO v_function FROM pg_catalog.pg_proc AS p
      WHERE p.oid = pg_catalog.to_regprocedure(
        'public.lab_arena_training_deepline_response_v1(public.lab_arena_ledger)');
    IF v_count <> 1 OR v_function.oid IS NULL
       OR pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(v_function.prosrc, 'UTF8')), 'hex')
            <> '4e29b9b06ea814c70a3046939a1569da51efb25ec45275f729ff7e314ba97b69'
       OR v_function.proowner <> 'lab_arena_owner'::pg_catalog.regrole
       OR v_function.prolang <> (SELECT oid FROM pg_catalog.pg_language WHERE lanname = 'plpgsql')
       OR v_function.prorettype <> 'jsonb'::pg_catalog.regtype OR v_function.proretset
       OR v_function.prokind <> 'f' OR v_function.proisstrict OR v_function.proleakproof
       OR v_function.pronargdefaults <> 0 OR v_function.prosupport <> 0
       OR v_function.proargnames IS DISTINCT FROM ARRAY['p_ledger']::TEXT[]
       OR v_function.proargmodes IS NOT NULL OR v_function.proallargtypes IS NOT NULL
       OR NOT v_function.prosecdef OR v_function.provolatile <> 's'
       OR v_function.proparallel <> 'u'
       OR v_function.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
       OR (SELECT count(*) FROM pg_catalog.aclexplode(v_function.proacl)) <> 2
       OR EXISTS (SELECT 1 FROM pg_catalog.aclexplode(v_function.proacl) AS privilege
         WHERE privilege.grantee NOT IN ('lab_arena_owner'::pg_catalog.regrole, 'lab_arena_service'::pg_catalog.regrole)
           OR privilege.grantor <> 'lab_arena_owner'::pg_catalog.regrole
           OR privilege.privilege_type <> 'EXECUTE' OR privilege.is_grantable) THEN
      RAISE EXCEPTION 'lab_arena_training_response_preimage_invalid' USING ERRCODE = '55000';
    END IF;
  END IF;
END;
$function_preimage$;

CREATE OR REPLACE FUNCTION public.lab_arena_training_deepline_response_v1(
  p_ledger public.lab_arena_ledger
) RETURNS JSONB LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $training_response$
DECLARE
  v_entry public.lab_arena_ledger;
  v_head public.lab_arena_ledger;
  v_reservation public.lab_arena_ledger;
  v_run public.lab_arena_runs;
  v_response public.lab_arena_deepline_call_responses;
  v_terminal JSONB;
BEGIN
  -- Cheap rejection for ordinary ledger reads. Caller-supplied composites
  -- are never authority: re-read the row and require the exact stored tuple.
  IF p_ledger.provider IS DISTINCT FROM 'deepline'
     OR p_ledger.entry_kind NOT IN ('settlement', 'uncertain') THEN
    RETURN NULL;
  END IF;
  SELECT * INTO v_entry FROM public.lab_arena_ledger WHERE entry_id = p_ledger.entry_id;
  IF v_entry.entry_id IS NULL OR v_entry IS DISTINCT FROM p_ledger THEN
    RETURN NULL;
  END IF;
  v_head := public.lab_arena__ledger_head(v_entry.call_identity);
  IF v_head.entry_id IS DISTINCT FROM v_entry.entry_id THEN
    RETURN NULL;
  END IF;
  SELECT * INTO v_response FROM public.lab_arena_deepline_call_responses
    WHERE call_identity = v_entry.call_identity;
  IF v_response.call_identity IS NULL THEN
    RETURN NULL;
  END IF;
  -- The ordinary ledger receipt already contains this exact response.
  IF v_entry.terminal_response IS NOT DISTINCT FROM v_response.terminal_response THEN
    RETURN NULL;
  END IF;
  SELECT * INTO v_reservation FROM public.lab_arena_ledger
    WHERE entry_id = v_response.reservation_entry_id;
  SELECT * INTO v_run FROM public.lab_arena_runs WHERE run_id = v_entry.run_id;
  IF v_reservation.entry_kind IS DISTINCT FROM 'reservation'
     OR v_reservation.provider IS DISTINCT FROM 'deepline'
     OR v_reservation.call_identity IS DISTINCT FROM v_entry.call_identity
     OR v_reservation.run_id IS DISTINCT FROM v_entry.run_id
     OR v_response.run_id IS DISTINCT FROM v_entry.run_id
     OR v_run.run_id IS DISTINCT FROM v_entry.run_id
     OR v_reservation.round_id IS DISTINCT FROM v_entry.round_id
     OR v_reservation.submission_id IS DISTINCT FROM v_entry.submission_id
     OR v_reservation.miner_hotkey IS DISTINCT FROM v_entry.miner_hotkey
     OR v_reservation.stage IS DISTINCT FROM v_entry.stage
     OR v_reservation.operation_id IS DISTINCT FROM v_entry.operation_id
     OR v_reservation.funding_source IS DISTINCT FROM v_entry.funding_source
     OR v_run.round_id IS DISTINCT FROM v_entry.round_id
     OR v_run.submission_id IS DISTINCT FROM v_entry.submission_id
     OR v_run.miner_hotkey IS DISTINCT FROM v_entry.miner_hotkey
     OR v_run.stage IS DISTINCT FROM v_entry.stage
     OR COALESCE(v_reservation.entry_doc ->> 'request_hash', '') !~ '^sha256:[0-9a-f]{64}$'
     OR COALESCE(v_reservation.entry_doc ->> 'tool', '') = ''
     OR v_reservation.entry_doc ->> 'deepline_execution_key'
          IS DISTINCT FROM 'arena:' || pg_catalog.substr(v_entry.call_identity, 8)
     OR (v_entry.entry_kind = 'settlement'
         AND v_entry.terminal_response #>> '{provider_cost,request_id}' IS DISTINCT FROM v_response.request_id)
     OR (v_entry.entry_kind = 'uncertain'
         AND v_entry.entry_doc #>> '{call,deepline_job_id}' IS NOT NULL
         AND v_entry.entry_doc #>> '{call,deepline_job_id}' IS DISTINCT FROM v_response.request_id)
     OR (v_response.terminal_response #>> '{provider_cost,request_id}' IS NOT NULL
         AND v_response.terminal_response #>> '{provider_cost,request_id}' IS DISTINCT FROM v_response.request_id)
     OR (v_response.terminal_response #>> '{provider_cost,operation}' IS NOT NULL
         AND v_response.terminal_response #>> '{provider_cost,operation}'
               IS DISTINCT FROM v_reservation.entry_doc ->> 'tool') THEN
    RETURN NULL;
  END IF;
  -- Reuse the broker's current delivered-response authority, with the same
  -- original ledger head and run. Do not reconstruct a different overlay.
  v_terminal := public.lab_arena__call_state_view(v_head, v_run) -> 'terminal_response';
  IF v_terminal IS DISTINCT FROM v_response.terminal_response
     OR v_terminal -> 'call_succeeded' IS DISTINCT FROM 'true'::JSONB
     OR COALESCE(v_terminal ->> 'status', '') !~ '^2[0-9]{2}$'
     OR pg_catalog.jsonb_typeof(v_terminal -> 'body_b64') IS DISTINCT FROM 'string' THEN
    RETURN NULL;
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'table', 'lab_arena_deepline_call_responses',
    'call_identity', v_response.call_identity,
    'reservation_entry_id', v_response.reservation_entry_id,
    'run_id', v_response.run_id, 'request_id', v_response.request_id,
    'created_at', v_response.created_at, 'terminal_entry_id', v_entry.entry_id,
    'request_hash', v_reservation.entry_doc ->> 'request_hash',
    'tool', v_reservation.entry_doc ->> 'tool', 'terminal_response', v_terminal);
END;
$training_response$;
-- The canonical migration principal has CREATE. Temporarily grant only the
-- target owner's required schema privilege for the ownership transfer.
DO $ownership$
DECLARE v_acl pg_catalog.aclitem[]; v_had_create BOOLEAN;
BEGIN
  SELECT nspacl, pg_catalog.has_schema_privilege('lab_arena_owner', 'public', 'CREATE')
    INTO v_acl, v_had_create FROM pg_catalog.pg_namespace WHERE nspname = 'public';
  IF NOT v_had_create THEN
    GRANT CREATE ON SCHEMA public TO lab_arena_owner;
  END IF;
  ALTER FUNCTION public.lab_arena_training_deepline_response_v1(public.lab_arena_ledger)
    OWNER TO lab_arena_owner;
  IF NOT v_had_create THEN
    REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  END IF;
  IF (SELECT nspacl FROM pg_catalog.pg_namespace WHERE nspname = 'public')
       IS DISTINCT FROM v_acl THEN
    RAISE EXCEPTION 'lab_arena_training_response_schema_acl_changed';
  END IF;
END;
$ownership$;
REVOKE ALL ON FUNCTION public.lab_arena_training_deepline_response_v1(public.lab_arena_ledger)
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_training_deepline_response_v1(public.lab_arena_ledger)
  TO lab_arena_service;

-- First creation can inherit additional default EXECUTE grants. Reject them
-- transactionally instead of changing unrelated default privileges or roles.
DO $function_postcondition$
DECLARE v_function pg_catalog.pg_proc;
BEGIN
  SELECT p.* INTO v_function FROM pg_catalog.pg_proc AS p
    WHERE p.oid = pg_catalog.to_regprocedure(
      'public.lab_arena_training_deepline_response_v1(public.lab_arena_ledger)');
  IF v_function.oid IS NULL
     OR v_function.proowner <> 'lab_arena_owner'::pg_catalog.regrole
     OR (SELECT count(*) FROM pg_catalog.aclexplode(v_function.proacl)) <> 2
     OR EXISTS (SELECT 1 FROM pg_catalog.aclexplode(v_function.proacl) AS privilege
       WHERE privilege.grantee NOT IN ('lab_arena_owner'::pg_catalog.regrole, 'lab_arena_service'::pg_catalog.regrole)
         OR privilege.grantor <> 'lab_arena_owner'::pg_catalog.regrole
         OR privilege.privilege_type <> 'EXECUTE' OR privilege.is_grantable) THEN
    RAISE EXCEPTION 'lab_arena_training_response_postcondition_invalid' USING ERRCODE = '55000';
  END IF;
END;
$function_postcondition$;

NOTIFY pgrst, 'reload schema';
COMMIT;
