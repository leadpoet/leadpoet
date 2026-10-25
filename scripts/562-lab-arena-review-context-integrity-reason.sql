-- Accept the existing typed fail-closed review diagnostic. No verdict, retry,
-- accounting, claim, or historical submission state changes.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $review_context_integrity562$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)';
  v_preimage CONSTANT TEXT := '7db9233a48f24be3cf2fa61572546774cccefc1d69d95af2ad3c5dc10bea8083';
  v_postimage CONSTANT TEXT := 'f8b5171ec7584721ae879a66e450136967e63a91e7a80c03ea9fb97e2d15c1d8';
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
    RAISE EXCEPTION 'Arena review562 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena review562 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_definition,
    $old562$'finding_evidence_mismatch', 'finding_explanation'$old562$,
    $new562$'finding_evidence_mismatch', 'finding_explanation',
             'context_integrity'$new562$);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena review562 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')
      <>v_postimage OR (SELECT pg_catalog.to_jsonb(p)-'prosrc'
        FROM pg_catalog.pg_proc p WHERE p.oid=v_oid) IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena review562 readback differs' USING ERRCODE='55000';
  END IF;
END;
$review_context_integrity562$;

COMMIT;
