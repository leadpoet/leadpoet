-- Serialize review reservation with retained restart-guard ownership.
-- No review decisions, participants, costs or historical claims are rewritten.
-- INSTALL PREREQUISITE: stop and drain every review worker and verify no old
-- begin_submission_review invocation remains in flight. CREATE OR REPLACE does
-- not fence an already-running old function body. The coordinating release
-- must prove quiescence before applying this migration or trusting its marker.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
LOCK TABLE public.lab_arena_rounds, public.lab_arena_submissions
  IN ACCESS EXCLUSIVE MODE NOWAIT;

DO $review587_0$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)';
  v_preimage CONSTANT TEXT := '136f980e76b1ece72fb72986cfdfdcf8cb2384e96d5b44892aec2a5af9f43644';
  v_postimage CONSTANT TEXT := 'd1d16e3259d9c8955ccf14521be930b33e63281bd0c0ddca6ca06e249aa90da9';
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
    RAISE EXCEPTION 'Arena review587 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena review587 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_updated:=pg_catalog.replace(v_updated,$old587_0_0$  v_round_id TEXT;$old587_0_0$,$new587_0_0$  v_review_guard_present BOOLEAN;
  v_round_id TEXT;$new587_0_0$);
  v_updated:=pg_catalog.replace(v_updated,$old587_0_1$  SELECT submission.round_id INTO v_round_id$old587_0_1$,$new587_0_1$  -- Hold the singleton through reservation commit. A canonical guard update
  -- must wait for every earlier reservation transaction to finish.
  SELECT guard_commitment <> '' INTO STRICT v_review_guard_present
  FROM public.lab_arena_restart_claim_control WHERE singleton FOR SHARE;

  SELECT submission.round_id INTO v_round_id$new587_0_1$);
  v_updated:=pg_catalog.replace(v_updated,$old587_0_2$  IF v_submission.code_review_status = 'reviewing' THEN$old587_0_2$,$new587_0_2$  -- Completed decisions and exact replay above/below remain read-only.
  IF v_review_guard_present AND NOT (
      v_submission.code_review_status = 'reviewing'
      AND v_submission.code_review_claim IS NOT DISTINCT FROM p_claim_token_hash
  ) THEN
    RETURN pg_catalog.jsonb_build_object('status', 'paused');
  END IF;

  IF v_submission.code_review_status = 'reviewing' THEN$new587_0_2$);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena review587 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena review587 readback differs' USING ERRCODE='55000';
  END IF;
END;
$review587_0$;

DO $review587_1$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_code_review_schema_v1()';
  v_preimage CONSTANT TEXT := 'c2bcebe5ab7187a810871418fa15bc4d410108625e5b4d01b60ece7361c0deba';
  v_postimage CONSTANT TEXT := '1461e5cd607007e9a59f659f6935b089e4ff80d872da82d5376b8e882b2992d2';
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
    RAISE EXCEPTION 'Arena review587 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena review587 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_updated:=pg_catalog.replace(v_updated,$old587_1_0$'version', 207,$old587_1_0$,$new587_1_0$'version', 207,
    'atomic_restart_guard', TRUE,$new587_1_0$);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena review587 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena review587 readback differs' USING ERRCODE='55000';
  END IF;
END;
$review587_1$;

NOTIFY pgrst, 'reload schema';
COMMIT;
