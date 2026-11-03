-- Retry unusable provider reviews until the existing admission deadline.
-- No historical verdicts, costs, claims or participants are rewritten.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
DO $review585_state_guard$
DECLARE
  v_hash TEXT;
  v_type TEXT;
BEGIN
  SELECT pg_catalog.encode(extensions.digest(pg_catalog.pg_get_constraintdef(oid),'sha256'),'hex')
  INTO v_hash FROM pg_catalog.pg_constraint
  WHERE conrelid='public.lab_arena_submissions'::pg_catalog.regclass
    AND conname='lab_arena_submissions_code_review_state_check';
  SELECT pg_catalog.format_type(atttypid,atttypmod) INTO v_type
  FROM pg_catalog.pg_attribute
  WHERE attrelid='public.lab_arena_submissions'::pg_catalog.regclass
    AND attname='code_review_attempts' AND NOT attisdropped AND attnotnull;
  IF (v_hash,v_type) IS DISTINCT FROM
       ('75b85768aedaacea496a7298a16a17f8d01896aae7346b316f806d3004786f50','smallint')
     AND (v_hash,v_type) IS DISTINCT FROM
       ('4ef46ef50d2a1ceb6a5be98c94664b2a1f4ea1ce58073deced064f77fe2be455','integer') THEN
    RAISE EXCEPTION 'Arena review585 state shape differs' USING ERRCODE='55000';
  END IF;
END;
$review585_state_guard$;

DO $review585_lab_arena_begin_submission_review$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)';
  v_preimage CONSTANT TEXT := '7a50473ae8d77d6708191a9bfb2037a7b91a933f8fdef97ee550d2442676fd7a';
  v_postimage CONSTANT TEXT := '136f980e76b1ece72fb72986cfdfdcf8cb2384e96d5b44892aec2a5af9f43644';
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
    RAISE EXCEPTION 'Arena review585 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena review585 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_updated:=pg_catalog.replace(v_updated,$old5850$  v_attempt SMALLINT;
  v_attempt_limit SMALLINT;$old5850$,$new5850$  v_attempt INTEGER;
  v_attempt_limit INTEGER;$new5850$);
  v_updated:=pg_catalog.replace(v_updated,$old5851$      WHEN pg_catalog.jsonb_typeof(v_submission.code_review_doc -> 'retryable')
             = 'boolean'
      THEN (v_submission.code_review_doc ->> 'retryable')::BOOLEAN
      ELSE NULL$old5851$,$new5851$      WHEN v_submission.code_review_doc ->> 'error_code' IN (
        'code_review_preparation_unavailable', 'code_review_provider_rate_limited',
        'code_review_provider_timeout', 'code_review_provider_unavailable',
        'code_review_transport_error', 'review_response_invalid',
        'code_review_research_unverified'
      ) THEN TRUE
      WHEN v_submission.code_review_doc -> 'retryable' = 'false'::JSONB THEN FALSE
      ELSE NULL$new5851$);
  v_updated:=pg_catalog.replace(v_updated,$old5852$WHEN v_retryable IS TRUE THEN 6$old5852$,$new5852$WHEN v_retryable IS TRUE THEN NULL$new5852$);
  v_updated:=pg_catalog.replace(v_updated,$old5853$IF v_submission.code_review_attempts >= v_attempt_limit THEN$old5853$,$new5853$IF v_attempt_limit IS NOT NULL
       AND v_submission.code_review_attempts >= v_attempt_limit THEN$new5853$);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena review585 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena review585 readback differs' USING ERRCODE='55000';
  END IF;
END;
$review585_lab_arena_begin_submission_review$;

DO $review585_lab_arena_code_review_schema_v1$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_code_review_schema_v1()';
  v_preimage CONSTANT TEXT := '92b0cc683936514628ee381568b92a5df04d42e8b993eaf17c4438e57e46b542';
  v_postimage CONSTANT TEXT := 'c2bcebe5ab7187a810871418fa15bc4d410108625e5b4d01b60ece7361c0deba';
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
    RAISE EXCEPTION 'Arena review585 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena review585 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_updated:=pg_catalog.replace(v_updated,$old5850$'retry_policy', 'bounded_transient_v1'$old5850$,$new5850$'retry_policy', 'deadline_transient_v2'$new5850$);
  v_updated:=pg_catalog.replace(v_updated,$old5851$'max_transient_attempts', 6$old5851$,$new5851$'max_transient_attempts', NULL::INTEGER$new5851$);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena review585 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena review585 readback differs' USING ERRCODE='55000';
  END IF;
END;
$review585_lab_arena_code_review_schema_v1$;

DO $review585_lab_arena_finish_submission_review$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)';
  v_preimage CONSTANT TEXT := '19c1b1f10e9967cac66681999473cb0e1863c8c29f25064fdfee9ccab87adb14';
  v_postimage CONSTANT TEXT := '90f66612d21b2cde658c4b9107bc63025eae9a96a3872ca5dbb9c18da60cdb20';
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
    RAISE EXCEPTION 'Arena review585 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena review585 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_updated:=pg_catalog.replace(v_updated,$old5850$          'code_review_research_unverified'
        ))$old5850$,$new5850$          'code_review_research_unverified', 'review_response_invalid'
        ))$new5850$);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena review585 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena review585 readback differs' USING ERRCODE='55000';
  END IF;
END;
$review585_lab_arena_finish_submission_review$;

-- Attempts are an audit counter. The deadline and capped backoff bound dispatch.
ALTER TABLE public.lab_arena_submissions
  ALTER COLUMN code_review_attempts TYPE INTEGER;
ALTER TABLE public.lab_arena_submissions
  DROP CONSTRAINT lab_arena_submissions_code_review_state_check;
ALTER TABLE public.lab_arena_submissions
  ADD CONSTRAINT lab_arena_submissions_code_review_state_check CHECK (
    (code_review_status='pending' AND code_review_attempts=0
      AND code_review_doc IS NULL AND code_review_claim IS NULL
      AND code_review_started_at IS NULL AND code_review_expires_at IS NULL)
    OR (code_review_status='reviewing' AND code_review_attempts>=1
      AND code_review_doc IS NULL AND code_review_claim IS NOT NULL
      AND code_review_started_at IS NOT NULL
      AND code_review_expires_at>code_review_started_at)
    OR (code_review_status IN ('passed','rejected','error') AND code_review_attempts>=1
      AND code_review_doc IS NOT NULL AND code_review_claim IS NOT NULL
      AND code_review_started_at IS NOT NULL AND code_review_expires_at IS NULL)
  );
NOTIFY pgrst, 'reload schema';
COMMIT;
