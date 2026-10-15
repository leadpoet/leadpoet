-- A retained scoring backup has no caller/date gate. It is not a public RPC.
-- Keep it owner-only, like the other retained backup. Do not change its body,
-- current scoring RPCs, table permissions, or any round/run data.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $legacy_scoring_privileges491$
DECLARE
  v_signature CONSTANT TEXT :=
    'public.lab_arena_open_scoring_sep15_baseline_only_v1(text,smallint,jsonb)';
  v_images CONSTANT TEXT[] := ARRAY[
    'c71e43caf400c28fb7ece5172686cef09fd5542a95945b2a8a79beb6a43ba750',
    'f8dea3dbd0a59d4e6f788dc85b1f4faa664a8b2658c1f368529c0332223f4410'];
  v_legacy_acl CONSTANT TEXT :=
    '{lab_arena_owner=X/lab_arena_owner,anon=X/lab_arena_owner,authenticated=X/lab_arena_owner,service_role=X/lab_arena_owner}';
  v_owner_acl CONSTANT TEXT := '{lab_arena_owner=X/lab_arena_owner}';
  v_oid OID;
  v_definition TEXT;
  v_acl TEXT;
  v_identity JSONB;
  v_metadata JSONB;
BEGIN
  v_oid := pg_catalog.to_regprocedure(v_signature);
  SELECT pg_catalog.pg_get_functiondef(p.oid),p.proacl::TEXT,
    pg_catalog.jsonb_build_array(owner.rolname,p.prosecdef,p.provolatile,p.proconfig),
    pg_catalog.to_jsonb(p)-'proacl'
  INTO v_definition,v_acl,v_identity,v_metadata
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_oid;
  IF v_definition IS NULL
    OR NOT COALESCE(pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')
      = ANY(v_images),FALSE)
    OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
      'lab_arena_owner',TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[])
    OR NOT COALESCE(v_acl IN (v_legacy_acl,v_owner_acl),FALSE) THEN
    RAISE EXCEPTION 'legacy scoring491 authority differs' USING ERRCODE='55000';
  END IF;
  REVOKE EXECUTE ON FUNCTION public.lab_arena_open_scoring_sep15_baseline_only_v1(
    TEXT,SMALLINT,JSONB) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
  IF (SELECT p.proacl::TEXT FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
      IS DISTINCT FROM v_owner_acl
    OR (SELECT pg_catalog.to_jsonb(p)-'proacl' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
      IS DISTINCT FROM v_metadata
    OR pg_catalog.has_function_privilege('anon',v_oid,'EXECUTE')
    OR pg_catalog.has_function_privilege('authenticated',v_oid,'EXECUTE')
    OR pg_catalog.has_function_privilege('service_role',v_oid,'EXECUTE')
    OR pg_catalog.has_function_privilege('lab_arena_service',v_oid,'EXECUTE')
    OR NOT pg_catalog.has_function_privilege('lab_arena_owner',v_oid,'EXECUTE') THEN
    RAISE EXCEPTION 'legacy scoring491 privilege readback differs' USING ERRCODE='55000';
  END IF;
END;
$legacy_scoring_privileges491$;
NOTIFY pgrst,'reload schema';
COMMIT;
