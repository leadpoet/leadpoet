-- Current-policy judge calls use settled spend and no monetary holds.
-- Retain historical score serialization and every other claim guard.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $confirmed_score_parallel455$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)';
  v_preimage CONSTANT TEXT := '990baa411ac35d075ded6a56f14c5bd23c72416f85706eab4a802e3f08f51b3e';
  v_postimage CONSTANT TEXT := 'cd889cc33098309a32effa193f6eb389b26d6ac1beaa0f448b6a01b7edf2b16e';
  v_definition TEXT;
  v_identity JSONB;
  v_reserve TEXT;
  v_old TEXT := $old$      runs.kind <> 'score'
      OR NOT EXISTS (
        SELECT 1
        FROM public.lab_arena_runs AS active_score$old$;
  v_new TEXT := $new$      runs.kind <> 'score'
      -- lab_arena_confirmed_score_parallel_v1: current-policy admission and
      -- settlement share the submission lock; pending calls hold no money.
      OR v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
           = 'successful_calls_per_icp_v1'
      OR NOT EXISTS (
        SELECT 1
        FROM public.lab_arena_runs AS active_score$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,
      p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM
    pg_catalog.jsonb_build_array('lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
      TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena confirmed-score parallel security shape differs';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::REGPROCEDURE)
  INTO v_reserve;
  IF pg_catalog.strpos(v_reserve,'lab_arena_confirmed_score_admission')=0
     OR pg_catalog.strpos(v_reserve,'-- Current-policy calls retain lifecycle rows')=0
     OR pg_catalog.strpos(v_reserve,'WHERE submission_id = v_run.submission_id FOR NO KEY UPDATE;')=0 THEN
    RAISE EXCEPTION 'Arena confirmed-score admission prerequisite differs';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN RETURN; END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena confirmed-score parallel preimage differs';
  END IF;
  v_definition:=pg_catalog.replace(v_definition,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena confirmed-score parallel postimage differs';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_signature::REGPROCEDURE),'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena confirmed-score parallel readback differs';
  END IF;
END;
$confirmed_score_parallel455$;
COMMIT;
