-- Promotion is a canonical Arena service operation. Supabase default grants
-- survived migration188's PUBLIC-only revoke. Change only these two ACLs.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $promotion_privileges583$
DECLARE
  v_signature TEXT;
  v_image TEXT;
  v_oid OID;
  v_definition TEXT;
  v_identity JSONB;
  v_metadata JSONB;
BEGIN
  FOR v_signature,v_image IN SELECT * FROM (VALUES
    ('public.lab_arena_prepare_promotion(text,jsonb)',
     'ef6c100a7f55aecc0c96be62309ace478a09f5b6da2bb4bdb7092c79b0a77e20'),
    ('public.lab_arena_complete_promotion(text,jsonb)',
     '30ae36884002f9a3d67ba9803345ff1f7b39ec3a9a70243e4cff640fe46d543d')
  ) AS reviewed(signature,image) LOOP
    v_oid := pg_catalog.to_regprocedure(v_signature);
    SELECT pg_catalog.pg_get_functiondef(p.oid),
      pg_catalog.jsonb_build_array(owner.rolname,p.prosecdef,p.provolatile,p.proconfig),
      pg_catalog.to_jsonb(p)-'proacl'
    INTO v_definition,v_identity,v_metadata
    FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
    WHERE p.oid=v_oid;
    IF v_definition IS NULL
      OR pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')
        IS DISTINCT FROM v_image
      OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
        'lab_arena_owner',TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
      RAISE EXCEPTION 'promotion583 authority differs: %',v_signature USING ERRCODE='55000';
    END IF;
    EXECUTE pg_catalog.format('REVOKE EXECUTE ON FUNCTION %s FROM PUBLIC,anon,authenticated,service_role',v_signature);
    EXECUTE pg_catalog.format('GRANT EXECUTE ON FUNCTION %s TO lab_arena_service',v_signature);
    IF (SELECT pg_catalog.to_jsonb(p)-'proacl' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
        IS DISTINCT FROM v_metadata
      OR (SELECT p.proacl::TEXT FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
        IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
      OR pg_catalog.has_function_privilege('anon',v_oid,'EXECUTE')
      OR pg_catalog.has_function_privilege('authenticated',v_oid,'EXECUTE')
      OR pg_catalog.has_function_privilege('service_role',v_oid,'EXECUTE')
      OR NOT pg_catalog.has_function_privilege('lab_arena_owner',v_oid,'EXECUTE')
      OR NOT pg_catalog.has_function_privilege('lab_arena_service',v_oid,'EXECUTE') THEN
      RAISE EXCEPTION 'promotion583 privilege readback differs: %',v_signature USING ERRCODE='55000';
    END IF;
  END LOOP;
END;
$promotion_privileges583$;
NOTIFY pgrst,'reload schema';
COMMIT;
