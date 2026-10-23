-- Raw tool packets have not passed the scorer's company/evidence validation.
-- Preserve their own immutable ledger replay, but never share them as verdicts.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
DO $tool_reply_own_call_cache$
DECLARE
  v_definition TEXT;
  v_hash TEXT;
  v_input_old CONSTANT TEXT := $old$     OR pg_catalog.octet_length(p_call_doc::TEXT) > 65536$old$;
  v_input_new CONSTANT TEXT := $new$     OR pg_catalog.octet_length(p_call_doc::TEXT) > 65536
     OR (p_call_doc ? 'judgment_cache_own_call_only'
         AND pg_catalog.jsonb_typeof(p_call_doc -> 'judgment_cache_own_call_only')
           IS DISTINCT FROM 'boolean')$new$;
  v_lookup_old CONSTANT TEXT := $old$  -- Consider only direct paid-provider reservations. A cached destination$old$;
  v_lookup_new CONSTANT TEXT := $new$  -- Sharing policy must never change an already reserved call's replay,
  -- pending billing or settlement. That own-head branch above is authoritative.
  IF p_call_doc -> 'judgment_cache_own_call_only' = 'true'::JSONB THEN
    RETURN public.lab_arena_reserve_call(
      p_run_id, p_lease_token_hash, p_call_identity, p_operation_id,
      p_provider, p_funding_source, p_amount_microusd, p_call_doc,
      p_lease_ttl_seconds
    );
  END IF;

  -- Consider only direct paid-provider reservations. A cached destination$new$;
  v_source_old CONSTANT TEXT := $old$      AND NOT reservation.entry_doc ? 'judgment_cache_source_call_identity'$old$;
  v_source_new CONSTANT TEXT := $new$      AND NOT reservation.entry_doc ? 'judgment_cache_source_call_identity'
      AND reservation.entry_doc -> 'judgment_cache_own_call_only'
        IS DISTINCT FROM 'true'::JSONB$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
      pg_catalog.pg_get_functiondef(p.oid),'UTF8')),'hex')
  INTO STRICT v_definition,v_hash
  FROM pg_catalog.pg_proc p
  WHERE p.oid='public.lab_arena_reserve_judgment_call(text,text,text,text,text,text,bigint,jsonb,integer)'::pg_catalog.regprocedure
    AND pg_catalog.pg_get_userbyid(p.proowner)='lab_arena_owner'
    AND p.prosecdef AND p.provolatile='v'
    AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
    AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}';
  IF v_hash='f73e8cd3839372e5bdb0cfefd2269e9bd1abcd578d0502cda14948e81acb843d' THEN RETURN; END IF;
  IF v_hash<>'6970e47b0b3aa9308a3a38471ce0c70fc5498f4336c9277994546134ecb28d3b'
     OR (pg_catalog.length(v_definition)-pg_catalog.length(
       pg_catalog.replace(v_definition,v_input_old,'')))/pg_catalog.length(v_input_old)<>1
     OR (pg_catalog.length(v_definition)-pg_catalog.length(
       pg_catalog.replace(v_definition,v_lookup_old,'')))/pg_catalog.length(v_lookup_old)<>1
     OR (pg_catalog.length(v_definition)-pg_catalog.length(
       pg_catalog.replace(v_definition,v_source_old,'')))/pg_catalog.length(v_source_old)<>1 THEN
    RAISE EXCEPTION 'tool-reply cache source differs';
  END IF;
  EXECUTE pg_catalog.replace(pg_catalog.replace(pg_catalog.replace(
    v_definition,v_input_old,v_input_new),v_lookup_old,v_lookup_new),
    v_source_old,v_source_new);
  SELECT pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
    pg_catalog.pg_get_functiondef('public.lab_arena_reserve_judgment_call(text,text,text,text,text,text,bigint,jsonb,integer)'::pg_catalog.regprocedure),'UTF8')),'hex')
  INTO v_hash;
  IF v_hash<>'f73e8cd3839372e5bdb0cfefd2269e9bd1abcd578d0502cda14948e81acb843d' THEN
    RAISE EXCEPTION 'tool-reply cache postimage differs';
  END IF;
END;
$tool_reply_own_call_cache$;
COMMIT;
