-- Controlled rollback of migration585's RPC behavior before an old service restart.
-- Keep additive integer/state storage; never rewrite reviews, costs or participants.
-- Stop the review worker and drain exact claims through the canonical workflow first.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $rollback585_lab_arena_begin_submission_review$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)';
  v_preimage CONSTANT TEXT := '136f980e76b1ece72fb72986cfdfdcf8cb2384e96d5b44892aec2a5af9f43644';
  v_postimage CONSTANT TEXT := '7a50473ae8d77d6708191a9bfb2037a7b91a933f8fdef97ee550d2442676fd7a';
  v_oid OID;
  v_definition TEXT;
  v_updated TEXT;
  v_metadata JSONB;
  v_identity JSONB;
BEGIN
  v_oid:=pg_catalog.to_regprocedure(v_signature);
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.to_jsonb(p)-'prosrc',
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_metadata,v_identity
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_oid;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner',
    '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena rollback585 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena rollback585 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_updated:=pg_catalog.replace(v_updated,$old585r0$IF v_attempt_limit IS NOT NULL
       AND v_submission.code_review_attempts >= v_attempt_limit THEN$old585r0$,$new585r0$IF v_submission.code_review_attempts >= v_attempt_limit THEN$new585r0$);
  v_updated:=pg_catalog.replace(v_updated,$old585r1$WHEN v_retryable IS TRUE THEN NULL$old585r1$,$new585r1$WHEN v_retryable IS TRUE THEN 6$new585r1$);
  v_updated:=pg_catalog.replace(v_updated,$old585r2$      WHEN v_submission.code_review_doc ->> 'error_code' IN (
        'code_review_preparation_unavailable', 'code_review_provider_rate_limited',
        'code_review_provider_timeout', 'code_review_provider_unavailable',
        'code_review_transport_error', 'review_response_invalid',
        'code_review_research_unverified'
      ) THEN TRUE
      WHEN v_submission.code_review_doc -> 'retryable' = 'false'::JSONB THEN FALSE
      ELSE NULL$old585r2$,$new585r2$      WHEN pg_catalog.jsonb_typeof(v_submission.code_review_doc -> 'retryable')
             = 'boolean'
      THEN (v_submission.code_review_doc ->> 'retryable')::BOOLEAN
      ELSE NULL$new585r2$);
  v_updated:=pg_catalog.replace(v_updated,$old585r3$  v_attempt INTEGER;
  v_attempt_limit INTEGER;$old585r3$,$new585r3$  v_attempt SMALLINT;
  v_attempt_limit SMALLINT;$new585r3$);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena rollback585 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena rollback585 readback differs' USING ERRCODE='55000';
  END IF;
END;
$rollback585_lab_arena_begin_submission_review$;

DO $rollback585_lab_arena_code_review_schema_v1$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_code_review_schema_v1()';
  v_preimage CONSTANT TEXT := 'c2bcebe5ab7187a810871418fa15bc4d410108625e5b4d01b60ece7361c0deba';
  v_postimage CONSTANT TEXT := '92b0cc683936514628ee381568b92a5df04d42e8b993eaf17c4438e57e46b542';
  v_oid OID;
  v_definition TEXT;
  v_updated TEXT;
  v_metadata JSONB;
  v_identity JSONB;
BEGIN
  v_oid:=pg_catalog.to_regprocedure(v_signature);
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.to_jsonb(p)-'prosrc',
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_metadata,v_identity
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_oid;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner',
    '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'s',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena rollback585 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena rollback585 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_updated:=pg_catalog.replace(v_updated,$old585r0$'max_transient_attempts', NULL::INTEGER$old585r0$,$new585r0$'max_transient_attempts', 6$new585r0$);
  v_updated:=pg_catalog.replace(v_updated,$old585r1$'retry_policy', 'deadline_transient_v2'$old585r1$,$new585r1$'retry_policy', 'bounded_transient_v1'$new585r1$);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena rollback585 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena rollback585 readback differs' USING ERRCODE='55000';
  END IF;
END;
$rollback585_lab_arena_code_review_schema_v1$;

DO $rollback585_lab_arena_finish_submission_review$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)';
  v_preimage CONSTANT TEXT := '90f66612d21b2cde658c4b9107bc63025eae9a96a3872ca5dbb9c18da60cdb20';
  v_postimage CONSTANT TEXT := '19c1b1f10e9967cac66681999473cb0e1863c8c29f25064fdfee9ccab87adb14';
  v_oid OID;
  v_definition TEXT;
  v_updated TEXT;
  v_metadata JSONB;
  v_identity JSONB;
BEGIN
  v_oid:=pg_catalog.to_regprocedure(v_signature);
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.to_jsonb(p)-'prosrc',
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_metadata,v_identity
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_oid;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner',
    '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena rollback585 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena rollback585 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_updated:=pg_catalog.replace(v_updated,$old585r0$          'code_review_research_unverified', 'review_response_invalid'
        ))$old585r0$,$new585r0$          'code_review_research_unverified'
        ))$new585r0$);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena rollback585 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena rollback585 readback differs' USING ERRCODE='55000';
  END IF;
END;
$rollback585_lab_arena_finish_submission_review$;

NOTIFY pgrst, 'reload schema';
COMMIT;
