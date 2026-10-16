-- Recover the native identity behind a keyed legacy Deepline edge locator.
-- Keep immutable request, credential, dispatch and append-only settlement guards.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $deepline_keyed_edge_hint494$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_reconcile_deepline_cost_v2(text,text,text,bigint,text,text,text,bigint,text,text,text)';
  v_preimage CONSTANT TEXT := 'd21a2c6f2f7eb645ecffce3a2de5a5c0fe35268f55b860a69463cf873554dd92';
  v_postimage CONSTANT TEXT := '5c67e1d48e19c22cc0de51de32173e9e16fe53cc194cec9550e40406bbba9113';
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
    RAISE EXCEPTION 'Arena Deepline edge494 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena Deepline edge494 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_definition,$old494$         AND v_head.entry_doc #>> '{call,deepline_job_id}'
             IS DISTINCT FROM p_recovered_request_id) THEN$old494$,
    $new494$         AND v_head.entry_doc #>> '{call,deepline_job_id}'
             IS DISTINCT FROM p_recovered_request_id
         AND NOT (
           -- lab_arena_deepline_keyed_edge_hint494: keep the candidate and
           -- immutable key bound while replacing only a legacy edge locator.
           p_execution_key IS NOT NULL
           AND v_head.entry_doc #>> '{call,deepline_job_id}' = p_request_id
           AND p_request_id ~
             '^[a-z0-9]{3,8}::[a-z0-9]{1,16}-[0-9]{13}-[a-f0-9]{12,64}$'
         )) THEN$new494$);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena Deepline edge494 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')
      <>v_postimage OR (SELECT pg_catalog.to_jsonb(p)-'prosrc'
        FROM pg_catalog.pg_proc p WHERE p.oid=v_oid) IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena Deepline edge494 readback differs' USING ERRCODE='55000';
  END IF;
END;
$deepline_keyed_edge_hint494$;

COMMIT;
