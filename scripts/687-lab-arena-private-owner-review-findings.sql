-- Keep bounded, already source-validated own excerpts for a signed owner read.
-- No historical row, admission verdict, retry, ledger, or public grant changes.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $owner687$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)';
  v_preimage CONSTANT TEXT := '54414a7f9d07d00111206bbd02884896e2fee89c12f80eabdd35c344b0767530';
  v_postimage CONSTANT TEXT := 'a88ac51cd235480d73630e9a426004a58bdbac59699c88c2de31accc83c6a584';
  v_oid OID;
  v_definition TEXT;
  v_updated TEXT;
  v_metadata JSONB;
  v_identity JSONB;
  v_old TEXT;
  v_new TEXT;
BEGIN
  v_oid:=pg_catalog.to_regprocedure(v_signature);
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.to_jsonb(p)-'prosrc',
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_metadata,v_identity
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_oid;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena owner687 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN RETURN; END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena owner687 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_old:=$old687a$  v_effective_status := p_status;$old687a$;
  v_new:=$new687a$  -- Optional diagnostics cannot introduce provider prose or peer source.
  -- The host has already proved every excerpt against this submission's source.
  IF p_review_doc ? 'owner_findings' THEN
    IF p_status<>'rejected'
       OR pg_catalog.jsonb_typeof(p_review_doc->'owner_findings') IS DISTINCT FROM 'object'
       OR pg_catalog.octet_length((p_review_doc->'owner_findings')::TEXT)>20000 THEN
      RAISE EXCEPTION 'lab_arena_code_review_input_invalid' USING ERRCODE='22023';
    END IF;
    IF (SELECT count(*) FROM pg_catalog.jsonb_object_keys(p_review_doc->'owner_findings'))<>3
       OR NOT (p_review_doc->'owner_findings') ?& ARRAY['findings','truncated','redacted']
       OR pg_catalog.jsonb_typeof(p_review_doc#>'{owner_findings,truncated}') IS DISTINCT FROM 'boolean'
       OR pg_catalog.jsonb_typeof(p_review_doc#>'{owner_findings,redacted}') IS DISTINCT FROM 'boolean'
       OR pg_catalog.jsonb_typeof(p_review_doc#>'{owner_findings,findings}') IS DISTINCT FROM 'array' THEN
      RAISE EXCEPTION 'lab_arena_code_review_input_invalid' USING ERRCODE='22023';
    END IF;
    IF pg_catalog.jsonb_array_length(p_review_doc#>'{owner_findings,findings}')>8 THEN
      RAISE EXCEPTION 'lab_arena_code_review_input_invalid' USING ERRCODE='22023';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_catalog.jsonb_array_elements(p_review_doc#>'{owner_findings,findings}') f
       WHERE pg_catalog.jsonb_typeof(f) IS DISTINCT FROM 'object') THEN
      RAISE EXCEPTION 'lab_arena_code_review_input_invalid' USING ERRCODE='22023';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_catalog.jsonb_array_elements(p_review_doc#>'{owner_findings,findings}') f
       WHERE (SELECT count(*) FROM pg_catalog.jsonb_object_keys(f))<>3
          OR NOT f ?& ARRAY['category','file','evidence']
          OR pg_catalog.jsonb_typeof(f->'category') IS DISTINCT FROM 'string'
          OR NOT (p_review_doc->'categories') @> pg_catalog.jsonb_build_array(f->>'category')
          OR pg_catalog.jsonb_typeof(f->'file') IS DISTINCT FROM 'string'
          OR pg_catalog.octet_length(f->>'file') NOT BETWEEN 1 AND 512
          OR pg_catalog.jsonb_typeof(f->'evidence') IS DISTINCT FROM 'string'
          OR pg_catalog.octet_length(f->>'evidence') NOT BETWEEN 1 AND 1000) THEN
      RAISE EXCEPTION 'lab_arena_code_review_input_invalid' USING ERRCODE='22023';
    END IF;
  END IF;
  v_effective_status := p_status;$new687a$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena owner687 validation anchor differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old687b$      'categories', p_review_doc -> 'categories'
    )$old687b$;
  v_new:=$new687b$      'categories', p_review_doc -> 'categories'
    ) || CASE WHEN p_status='rejected' AND p_review_doc ? 'owner_findings'
        THEN pg_catalog.jsonb_build_object('owner_findings',p_review_doc->'owner_findings')
        ELSE '{}'::JSONB END$new687b$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena owner687 projection anchor differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena owner687 postimage differs: %',pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex') USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid) IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena owner687 readback differs' USING ERRCODE='55000';
  END IF;
END;
$owner687$;
COMMIT;
