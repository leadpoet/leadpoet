-- An eligible judge continuation must not wait for a busy or abandoned
-- alternative runner. Retain legacy confirmations, authenticated host-fault
-- exclusions, quarantine, backoff, lease fencing, caches, and billing guards.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $judge_continuation_handoff577$
DECLARE
  signature CONSTANT TEXT := 'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)';
  preimage CONSTANT TEXT := '11f8669d16966fbe41e5c2530c3ed273fea2abc1aacfc4264e2854f017f5b6f3';
  postimage CONSTANT TEXT := '5c7c9194d06d1d1920e24b8e85426d403cbd4b6f398afbcd6fe5f86f0dc5dacd';
  d TEXT; metadata JSONB;
BEGIN
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
  FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
  WHERE p.oid=signature::regprocedure
    AND pg_get_userbyid(p.proowner)='lab_arena_owner'
    AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
    AND p.prosecdef AND p.provolatile='v'
    AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
    AND l.lanname='plpgsql';
  IF encode(extensions.digest(d,'sha256'),'hex')=postimage THEN RETURN; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>preimage THEN
    RAISE EXCEPTION 'judge continuation handoff preimage differs' USING ERRCODE='55000';
  END IF;
  IF (length(d)-length(replace(d,$old$    AND (runs.previous_runner_hotkey IS NULL OR runs.previous_runner_hotkey <> p_runner_hotkey$old$,'')))<>length($old$    AND (runs.previous_runner_hotkey IS NULL OR runs.previous_runner_hotkey <> p_runner_hotkey$old$) THEN
    RAISE EXCEPTION 'judge continuation handoff anchor differs' USING ERRCODE='55000';
  END IF;
  d:=replace(d,$old$    AND (runs.previous_runner_hotkey IS NULL OR runs.previous_runner_hotkey <> p_runner_hotkey$old$,$new$    -- Preserve legacy confirmations. A validated judge continuation can use
    -- this available runner; another outstanding lease does not prove that
    -- its runner can accept the retry. Earlier host-fault guards still apply.
    AND (runs.previous_runner_hotkey IS NULL OR runs.previous_runner_hotkey <> p_runner_hotkey
         OR public.lab_arena__judge_continuation_v1(v_round,runs)$new$);
  IF encode(extensions.digest(d,'sha256'),'hex')<>postimage THEN
    RAISE EXCEPTION 'judge continuation handoff postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE d;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=signature::regprocedure)
      IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(signature::regprocedure),'sha256'),'hex')<>postimage THEN
    RAISE EXCEPTION 'judge continuation handoff readback differs' USING ERRCODE='55000';
  END IF;
END;
$judge_continuation_handoff577$;
COMMIT;
