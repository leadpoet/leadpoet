-- Forward-only readiness correction for the exact669 review wrapper.
-- Preserve every legacy direct-body probe, capability field, grant and stored row.
-- Future delegated implementations need a separately reviewed probe update.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
DO $probe670$
DECLARE
  v_target OID := pg_catalog.to_regprocedure('public.lab_arena_submission_replacement_schema_v1()');
  v_definition TEXT;
  v_updated TEXT;
  v_metadata JSONB;
  v_identity JSONB;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.to_jsonb(p)-'prosrc',
    pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),
      p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_metadata,v_identity FROM pg_catalog.pg_proc p WHERE p.oid=v_target;
  IF v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
      'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
      TRUE,'s',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena probe670 security differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')='078a88402b352debb9a3a03ad8e07df908da693b662f6fe9f542883e06ff8081' THEN RETURN; END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex') IS DISTINCT FROM
       '42d80fc942a8c8a931804be622328e8672cb8a2c1703b384d6c8b61c0c266ff2' THEN
    RAISE EXCEPTION 'Arena probe670 preimage differs' USING ERRCODE='55000';
  END IF;
  IF length(v_definition)-length(replace(v_definition,$decl670$  v_review TEXT;$decl670$,''))<>length($decl670$  v_review TEXT;$decl670$)
     OR length(v_definition)-length(replace(v_definition,$resolve670$  ) INTO v_review;$resolve670$,''))<>length($resolve670$  ) INTO v_review;$resolve670$) THEN
    RAISE EXCEPTION 'Arena probe670 anchors differ' USING ERRCODE='55000';
  END IF;
  v_updated:=replace(v_definition,$decl670$  v_review TEXT;$decl670$,$newdecl670$  v_review TEXT;
  v_delegate TEXT;
  v_delegate_oid OID;
  v_review_identity JSONB;
  v_delegate_identity JSONB;$newdecl670$);
  v_updated:=replace(v_updated,$resolve670$  ) INTO v_review;$resolve670$,$newresolve670$  ) INTO v_review;
  -- Validate any additive v2 before accepting either callable entrypoint.
  -- A legacy direct body can coexist with it after a source rollback.
  v_delegate_oid := pg_catalog.to_regprocedure(
    'public.lab_arena_begin_submission_review_v2(text,text,text,bigint,text,integer,bigint,text)'
  );
  IF v_delegate_oid IS NOT NULL
     OR pg_catalog.strpos(pg_catalog.lower(v_review), 'lab_arena_begin_submission_review_v2') > 0 THEN
    SELECT pg_catalog.pg_get_functiondef(p.oid),
      pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),
        p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_delegate,v_delegate_identity FROM pg_catalog.pg_proc p WHERE p.oid = v_delegate_oid;
    IF pg_catalog.encode(extensions.digest(v_delegate,'sha256'),'hex') IS DISTINCT FROM
         '00b9b8c403a5f446eef02e1839eec7d6f96c681bf55c5712b6bb6a2c05f8cbc5'
       OR v_delegate_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
         'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
         TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
      RAISE EXCEPTION 'lab_arena_queued_replacement_schema_incomplete';
    END IF;
  END IF;
  -- Only a verified wrapper can transfer the deferral proof to its delegate.
  -- A copied marker comment in an altered wrapper cannot establish this link.
  IF pg_catalog.strpos(pg_catalog.lower(v_review), 'lab_arena_begin_submission_review_v2') > 0 THEN
    SELECT pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),
      p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_review_identity FROM pg_catalog.pg_proc p WHERE p.oid =
      'public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)'::pg_catalog.regprocedure;
    IF pg_catalog.encode(extensions.digest(v_review,'sha256'),'hex') IS DISTINCT FROM
         '2ade4eb49ebeddead5cf0356f6f94e273a0a4d3d5de47de294c1576ca3c8cbb8'
       OR v_review_identity IS DISTINCT FROM v_delegate_identity THEN
      RAISE EXCEPTION 'lab_arena_queued_replacement_schema_incomplete';
    END IF;
    v_review := v_delegate;
  END IF;$newresolve670$);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>'078a88402b352debb9a3a03ad8e07df908da693b662f6fe9f542883e06ff8081' THEN
    RAISE EXCEPTION 'Arena probe670 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_target),'sha256'),'hex')<>'078a88402b352debb9a3a03ad8e07df908da693b662f6fe9f542883e06ff8081'
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_target) IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena probe670 readback differs' USING ERRCODE='55000';
  END IF;
END;
$probe670$;
NOTIFY pgrst,'reload schema';
COMMIT;
