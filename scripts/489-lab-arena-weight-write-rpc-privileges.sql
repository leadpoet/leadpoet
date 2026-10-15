-- Supabase default grants survived202's PUBLIC-only revocation.
-- Restore the two write RPCs' dedicated-service boundary without changing code.
-- Production prerequisite: apply489 before unchanged488 while legacy grants
-- remain.488 deliberately rejects that ACL;489 accepts202 and488 bodies so
-- it can also be replayed after488. Numeric order is not the activation order.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $weight_write_rpc_privileges489$
DECLARE
  v_signatures TEXT[] := ARRAY[
    'public.lab_arena_record_chain_outcome_v1(text,integer,bigint,text,text,text,jsonb)',
    'public.lab_arena_publish_weight_state_v1(text,integer,bigint,text,jsonb)'];
  v_record_images TEXT[] := ARRAY[
    '4b59587948bfe87087c8cc5de1f8dd68b1ef53932b849387a4082b8e70611c65',
    '107dfd90dd15606d3ca169414c8c0376b734d6a813e1e57552b7c7e60a7d65ba'];
  v_publish_image CONSTANT TEXT := 'e2771d8514f16f82a7d5265b9f6cc563d64aa940236af29dc10ffd6115211b87';
  v_legacy_acl CONSTANT TEXT := '{lab_arena_owner=X/lab_arena_owner,anon=X/lab_arena_owner,authenticated=X/lab_arena_owner,service_role=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}';
  v_strict_acl CONSTANT TEXT := '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}';
  v_before TEXT[];
  v_definition TEXT; v_acl TEXT; v_identity JSONB; v_image TEXT; v_i INTEGER;
BEGIN
  -- Validate the whole coupled boundary before revoking any privilege.
  FOR v_i IN 1..2 LOOP
    SELECT pg_catalog.pg_get_functiondef(p.oid),p.proacl::TEXT,
      pg_catalog.jsonb_build_array(owner.rolname,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_definition,v_acl,v_identity FROM pg_catalog.pg_proc p
    JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
    WHERE p.oid=pg_catalog.to_regprocedure(v_signatures[v_i]);
    v_image:=pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex');
    IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
      'lab_arena_owner',TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[])
      OR NOT COALESCE(v_acl IN (v_legacy_acl,v_strict_acl),FALSE)
      OR NOT COALESCE(CASE v_i WHEN 1 THEN v_image=ANY(v_record_images)
        ELSE v_image=v_publish_image END,FALSE) THEN
      RAISE EXCEPTION 'weight write RPC authority differs: %',v_signatures[v_i] USING ERRCODE='55000';
    END IF;
    v_before:=pg_catalog.array_append(v_before,v_definition);
  END LOOP;
  -- No PUBLIC grant exists in either reviewed shape. An unexpected one fails
  -- above, rather than silently extending this revocation to unknown authority.
  REVOKE EXECUTE ON FUNCTION public.lab_arena_record_chain_outcome_v1(
    TEXT,INTEGER,BIGINT,TEXT,TEXT,TEXT,JSONB) FROM anon,authenticated,service_role;
  REVOKE EXECUTE ON FUNCTION public.lab_arena_publish_weight_state_v1(
    TEXT,INTEGER,BIGINT,TEXT,JSONB) FROM anon,authenticated,service_role;
  FOR v_i IN 1..2 LOOP
    SELECT pg_catalog.pg_get_functiondef(p.oid),p.proacl::TEXT,
      pg_catalog.jsonb_build_array(owner.rolname,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_definition,v_acl,v_identity FROM pg_catalog.pg_proc p
    JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
    WHERE p.oid=pg_catalog.to_regprocedure(v_signatures[v_i]);
    IF v_definition IS DISTINCT FROM v_before[v_i]
      OR v_acl IS DISTINCT FROM v_strict_acl
      OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
        'lab_arena_owner',TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[])
      OR pg_catalog.has_function_privilege('anon',v_signatures[v_i],'EXECUTE')
      OR pg_catalog.has_function_privilege('authenticated',v_signatures[v_i],'EXECUTE')
      OR pg_catalog.has_function_privilege('service_role',v_signatures[v_i],'EXECUTE')
      OR NOT pg_catalog.has_function_privilege('lab_arena_service',v_signatures[v_i],'EXECUTE')
      OR NOT pg_catalog.has_schema_privilege('lab_arena_service','public','USAGE') THEN
      RAISE EXCEPTION 'weight write RPC privilege readback differs: %',v_signatures[v_i] USING ERRCODE='55000';
    END IF;
  END LOOP;
END;
$weight_write_rpc_privileges489$;
NOTIFY pgrst,'reload schema';
COMMIT;
