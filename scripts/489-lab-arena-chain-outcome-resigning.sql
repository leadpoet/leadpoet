-- Acknowledge a valid re-signature of the same immutable chain observation.
-- Authentication stays in the service; preserve the first receipt and proof.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $chain_outcome_resigning489$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_record_chain_outcome_v1(text,integer,bigint,text,text,text,jsonb)';
  v_preimage CONSTANT TEXT := '107dfd90dd15606d3ca169414c8c0376b734d6a813e1e57552b7c7e60a7d65ba';
  v_postimage CONSTANT TEXT := '78af80199274864ccdbe9ee07a820609c1788cd5e29f3865f94a6f8331570190';
  v_definition TEXT; v_identity JSONB;
  v_old CONSTANT TEXT := $old489$  -- An existing request ID retains its original exact-document contract.
  IF existing.outcome_doc IS DISTINCT FROM p_outcome_doc AND
    (existing.request_id = p_request_id OR
      (existing.outcome_doc - ARRAY['observed_at','request_id','signature'])
        IS DISTINCT FROM
      (p_outcome_doc - ARRAY['observed_at','request_id','signature'])) THEN$old489$;
  v_new CONSTANT TEXT := $new489$  -- Re-signing the same request is also equivalent after authentication.
  IF (existing.outcome_doc - ARRAY['observed_at','request_id','signature'])
    IS DISTINCT FROM
    (p_outcome_doc - ARRAY['observed_at','request_id','signature']) THEN$new489$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.jsonb_build_array(
    owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'chain outcome retry security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN RETURN; END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage
    OR (pg_catalog.length(v_definition)-pg_catalog.length(
      pg_catalog.replace(v_definition,v_old,'')))<>pg_catalog.length(v_old) THEN
    RAISE EXCEPTION 'chain outcome retry preimage differs' USING ERRCODE='55000';
  END IF;
  v_definition:=pg_catalog.replace(v_definition,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'chain outcome retry postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
    v_signature::REGPROCEDURE),'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'chain outcome retry readback differs' USING ERRCODE='55000';
  END IF;
END;
$chain_outcome_resigning489$;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst,'reload schema';
COMMIT;
