-- Accept the existing permanent source-digest diagnostic. No verdict, retry,
-- accounting, claim, or historical submission state changes.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $review_source_digest563$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)';
  v_preimage CONSTANT TEXT := 'f8b5171ec7584721ae879a66e450136967e63a91e7a80c03ea9fb97e2d15c1d8';
  v_postimage CONSTANT TEXT := '573ee0b15a4952e4d25e2943c79578e44993c5e8117d2fdcc2a208fbb623cba1';
  v_oid OID;
  v_definition TEXT;
  v_updated TEXT;
  v_identity JSONB;
  v_metadata JSONB;
BEGIN
  v_oid:=pg_catalog.to_regprocedure(v_signature);
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.jsonb_build_array(
    owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig),
    pg_catalog.to_jsonb(p)-'prosrc'
  INTO v_definition,v_identity,v_metadata
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_oid;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner',
    '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena review563 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena review563 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_definition,
    $old563$'code_review_source_size_mismatch',$old563$,
    $new563$'code_review_source_size_mismatch',
         'code_review_source_digest_mismatch',$new563$);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena review563 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')
      <>v_postimage OR (SELECT pg_catalog.to_jsonb(p)-'prosrc'
        FROM pg_catalog.pg_proc p WHERE p.oid=v_oid) IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena review563 readback differs' USING ERRCODE='55000';
  END IF;
END;
$review_source_digest563$;

COMMIT;
