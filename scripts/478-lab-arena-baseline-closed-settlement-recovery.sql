-- A final charged provider response is closed even when the error response
-- omits call_succeeded. Retain the separate inflight and unresolved-success
-- cost checks, and leave the uncertain-success gate unchanged.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $baseline_closed_settlement478$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_recover_baseline_v1(text,jsonb)';
  v_preimage CONSTANT TEXT := 'cc87055b6200b9560a488b1d73f8681ae6dd09bac2e98deb29690ffa8e5dc046';
  v_postimage CONSTANT TEXT := 'fc708bc7e5727bf5fd049ba8381d3a23be07c595a4c36de8ed9dfc0099fef147';
  v_old_gate CONSTANT TEXT := $old$ OR
            (entry_kind='settlement' AND jsonb_typeof(terminal_response->'call_succeeded') IS DISTINCT FROM 'boolean')$old$;
  v_definition TEXT;
  v_identity JSONB;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,
      p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_signature::REGPROCEDURE;
  IF v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array('lab_arena_owner',
    '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena baseline settlement security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage
     OR (pg_catalog.length(v_definition)-pg_catalog.length(
       pg_catalog.replace(v_definition,v_old_gate,'')))<>pg_catalog.length(v_old_gate) THEN
    RAISE EXCEPTION 'Arena baseline settlement preimage differs' USING ERRCODE='55000';
  END IF;
  v_definition:=pg_catalog.replace(v_definition,v_old_gate,'');
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena baseline settlement postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
       v_signature::REGPROCEDURE),'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena baseline settlement readback differs' USING ERRCODE='55000';
  END IF;
END;
$baseline_closed_settlement478$;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;
