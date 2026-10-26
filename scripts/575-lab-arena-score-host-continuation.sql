-- A stopped zero-call score keeps continuation authority across stage changes.
-- Preserve all receipt, no-provider-work, locking and execute-generation checks.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $score_host_continuation575$
DECLARE
  signature CONSTANT TEXT := 'public.lab_arena_expire_leases(text)';
  preimage CONSTANT TEXT := 'cbe667688a31c3e0fbc39e968721b92ff206780c6818df8afde4eeaedd98c187';
  postimage CONSTANT TEXT := 'f8b2d491b97404f6dbea55cedeac2fb277897b9bdd7d67447b4fd43e56340a83';
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
    RAISE EXCEPTION 'score host continuation preimage differs' USING ERRCODE='55000';
  END IF;
  IF (length(d)-length(replace(d,$old0$      AND v_early_recovery_allowed
      AND candidate.kind IN ('score', 'execute')
      AND candidate.stage_generation = v_round.stage_generation$old0$,'')))<>length($old0$      AND v_early_recovery_allowed
      AND candidate.kind IN ('score', 'execute')
      AND candidate.stage_generation = v_round.stage_generation$old0$) THEN
    RAISE EXCEPTION 'score host continuation anchor differs' USING ERRCODE='55000';
  END IF;
  d:=replace(d,$old0$      AND v_early_recovery_allowed
      AND candidate.kind IN ('score', 'execute')
      AND candidate.stage_generation = v_round.stage_generation$old0$,$new0$      AND v_early_recovery_allowed
      AND candidate.kind IN ('score', 'execute')
      AND (candidate.stage_generation = v_round.stage_generation
        OR (candidate.kind = 'score'
          AND public.lab_arena__judge_continuation_v1(v_round,candidate)))$new0$);
  IF (length(d)-length(replace(d,$old1$    -- candidate scan's MVCC snapshot for absence of billable/provider work.
    IF v_run.status = 'leased' AND v_run.kind IN ('score', 'execute')
       AND v_run.stage_generation = v_round.stage_generation$old1$,'')))<>length($old1$    -- candidate scan's MVCC snapshot for absence of billable/provider work.
    IF v_run.status = 'leased' AND v_run.kind IN ('score', 'execute')
       AND v_run.stage_generation = v_round.stage_generation$old1$) THEN
    RAISE EXCEPTION 'score host continuation anchor differs' USING ERRCODE='55000';
  END IF;
  d:=replace(d,$old1$    -- candidate scan's MVCC snapshot for absence of billable/provider work.
    IF v_run.status = 'leased' AND v_run.kind IN ('score', 'execute')
       AND v_run.stage_generation = v_round.stage_generation$old1$,$new1$    -- candidate scan's MVCC snapshot for absence of billable/provider work.
    IF v_run.status = 'leased' AND v_run.kind IN ('score', 'execute')
       AND (v_run.stage_generation = v_round.stage_generation
         OR (v_run.kind = 'score'
           AND public.lab_arena__judge_continuation_v1(v_round,v_run)))$new1$);
  IF encode(extensions.digest(d,'sha256'),'hex')<>postimage THEN
    RAISE EXCEPTION 'score host continuation postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE d;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=signature::regprocedure)
      IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(signature::regprocedure),'sha256'),'hex')<>postimage THEN
    RAISE EXCEPTION 'score host continuation readback differs' USING ERRCODE='55000';
  END IF;
END;
$score_host_continuation575$;
COMMIT;
