-- Recover authenticated abandoned score leases with completed unpriced replies.
-- Preserve unknown charges, execute recovery, cleanup authority and every lease fence.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
DO $completed_score_recovery450$
DECLARE
  v_definition TEXT;
  v_identity JSONB;
  v_hash TEXT;
  v_index INTEGER;
  v_preimage CONSTANT TEXT := '9aa15b0f1aa66e61a668c708384600954b28bef7ef530781b0a25da357d40c71';
  v_postimage CONSTANT TEXT := '1539836b593cee82a8e9ac237199208464da5aa5bd960e9179275c9f51387f97';
  v_old CONSTANT TEXT[] := ARRAY[$oldguard$           SELECT DISTINCT ON (cost.call_identity) cost.entry_kind
           FROM public.lab_arena_ledger AS cost
           WHERE cost.run_id = v_run.run_id
           ORDER BY cost.call_identity, cost.entry_id DESC
         ) AS heads WHERE heads.entry_kind <> 'settlement'$oldguard$,$oldreason$WHEN 'score' THEN 'authenticated_settled_score_runtime_host_error'$oldreason$];
  v_new CONSTANT TEXT[] := ARRAY[$newguard$           SELECT DISTINCT ON (cost.call_identity) cost.*
           FROM public.lab_arena_ledger AS cost
           WHERE cost.run_id = v_run.run_id
           ORDER BY cost.call_identity, cost.entry_id DESC
         ) AS heads WHERE heads.entry_kind <> 'settlement'
           -- Completion releases a stopped score worker; it does not price a bill.
           AND NOT COALESCE((v_run.kind = 'score'
             AND heads.entry_kind = 'uncertain' AND heads.provider = 'deepline'
             AND heads.round_id = v_run.round_id
             AND heads.submission_id = v_run.submission_id
             AND heads.miner_hotkey = v_run.miner_hotkey AND heads.stage = v_run.stage
             AND heads.entry_doc ->> 'reason' = 'worker_reported'
             AND heads.entry_doc #>> '{call,reason}' = 'missing_provider_cost'
             AND heads.entry_doc #> '{call,call_succeeded}' = 'true'::JSONB
             AND heads.entry_doc #> '{call,provider_status}' = '200'::JSONB
             AND heads.entry_doc #>> '{call,top_level_job_status}' = 'completed'
             AND EXISTS (
               SELECT 1 FROM public.lab_arena_deepline_call_responses AS response
               JOIN public.lab_arena_ledger AS reservation
                 ON reservation.entry_id = response.reservation_entry_id
                AND reservation.entry_kind = 'reservation'
                AND reservation.call_identity = heads.call_identity
                AND reservation.run_id = heads.run_id
                AND reservation.round_id = heads.round_id
                AND reservation.submission_id = heads.submission_id
                AND reservation.miner_hotkey = heads.miner_hotkey
                AND reservation.stage = heads.stage
                AND reservation.provider = heads.provider
                AND reservation.operation_id = heads.operation_id
                AND reservation.funding_source = heads.funding_source
               WHERE response.call_identity = heads.call_identity
                 AND response.run_id = heads.run_id
                 AND response.request_id = heads.entry_doc #>> '{call,deepline_job_id}'
                 AND response.request_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'
                 AND response.request_id NOT LIKE 'ctx-tool-%'
                 AND response.terminal_response -> 'status' = '200'::JSONB
                 AND response.terminal_response -> 'call_succeeded' = 'true'::JSONB
                 AND pg_catalog.jsonb_typeof(response.terminal_response -> 'headers') = 'object'
                 AND pg_catalog.jsonb_typeof(response.terminal_response -> 'body_b64') = 'string'
                 AND pg_catalog.length(response.terminal_response ->> 'body_b64') > 0
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
                     AND dispatch.run_id = heads.run_id
                     AND dispatch.round_id = heads.round_id
                     AND dispatch.submission_id = heads.submission_id
                     AND dispatch.miner_hotkey = heads.miner_hotkey
                     AND dispatch.stage = heads.stage
                     AND dispatch.provider = heads.provider
                     AND dispatch.operation_id = heads.operation_id
                     AND dispatch.funding_source = heads.funding_source)
             )),FALSE)$newguard$,$newreason$WHEN 'score' THEN CASE WHEN EXISTS (
                  SELECT 1 FROM (SELECT DISTINCT ON (cost.call_identity) cost.entry_kind
                    FROM public.lab_arena_ledger AS cost WHERE cost.run_id = v_run.run_id
                    ORDER BY cost.call_identity,cost.entry_id DESC) AS heads
                  WHERE heads.entry_kind = 'uncertain')
                  THEN 'authenticated_completed_unpriced_score_runtime_host_error'
                  ELSE 'authenticated_settled_score_runtime_host_error' END$newreason$];
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,
      p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_roles AS owner ON owner.oid=p.proowner
  WHERE p.oid='public.lab_arena_expire_leases(text)'::REGPROCEDURE;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM
    pg_catalog.jsonb_build_array('lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
      TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena completed score recovery security shape differs'
      USING ERRCODE='55000';
  END IF;
  v_hash:=pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex');
  IF v_hash=v_postimage THEN RETURN; END IF;
  IF v_hash<>v_preimage THEN
    RAISE EXCEPTION 'Arena completed score recovery preimage differs' USING ERRCODE='55000';
  END IF;
  FOR v_index IN 1..pg_catalog.cardinality(v_old) LOOP
    IF (pg_catalog.length(v_definition)-pg_catalog.length(
         pg_catalog.replace(v_definition,v_old[v_index],'')))
         / pg_catalog.length(v_old[v_index]) <> 1 THEN
      RAISE EXCEPTION 'Arena completed score recovery anchor differs' USING ERRCODE='55000';
    END IF;
    v_definition:=pg_catalog.replace(v_definition,v_old[v_index],v_new[v_index]);
  END LOOP;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena completed score recovery postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
    'public.lab_arena_expire_leases(text)'::REGPROCEDURE),'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena completed score recovery readback differs' USING ERRCODE='55000';
  END IF;
END;
$completed_score_recovery450$;
COMMIT;
