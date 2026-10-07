-- Close a future Arena execution stage even if some assignments are unfinished.
-- Existing accepted outputs and cost records remain authoritative. The old
-- terminal_doc.infrastructure_incomplete marks the latest failed run of each
-- unfinished assignment; it is not a model zero or a scored result.
-- Migration 420's bounded active-lease drain runs before either close body.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $partial_execution_close_422$
DECLARE
  v_patch RECORD;
  v_definition TEXT;
  v_identity JSONB;
  v_hash TEXT;
  v_old_count INTEGER := 0;
  v_new_count INTEGER := 0;
BEGIN
  -- Validate both functions before changing either one.
  FOR v_patch IN SELECT * FROM (VALUES
    ('public.lab_arena_close_parallel_execution_v1(text)',
     'ae4bdb09e65108983e131d9bb4fa1461852a3f600ed979680084fd2cfc213030',
     'ffadfbf44dc0583879495f90eeda37d653f01357f69e924de30f5d7fd1ac6c9e'),
    ('public.lab_arena_close_stage(text,smallint)',
     '75f7b992f8a71091fceef9f6cb72c209eca5a54381e3fc50cc8bb0462aeaad10',
     '545b91b93455e5d438f1cefe0587090591d94301a8b2973ce1c7a3e3b48a36f0')
  ) AS patches(signature,preimage,postimage)
  LOOP
    SELECT pg_catalog.pg_get_functiondef(p.oid),
      pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,
        p.prosecdef,p.provolatile,p.proconfig)
      INTO v_definition,v_identity FROM pg_catalog.pg_proc p
      JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
      WHERE p.oid=pg_catalog.to_regprocedure(v_patch.signature);
    IF v_definition IS NULL OR v_identity IS DISTINCT FROM
      pg_catalog.jsonb_build_array('lab_arena_owner',
        '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
        TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
      RAISE EXCEPTION 'Arena partial close security shape differs';
    END IF;
    v_hash:=pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex');
    IF v_hash=v_patch.preimage THEN v_old_count:=v_old_count+1;
    ELSIF v_hash=v_patch.postimage THEN v_new_count:=v_new_count+1;
    ELSE RAISE EXCEPTION 'Arena partial close preimage differs';
    END IF;
  END LOOP;
  IF v_new_count=2 THEN RETURN; END IF;
  IF v_old_count<>2 THEN RAISE EXCEPTION 'Arena partial close partial state differs'; END IF;
  FOR v_patch IN SELECT * FROM (VALUES
    ('public.lab_arena_close_parallel_execution_v1(text)',
     'ffadfbf44dc0583879495f90eeda37d653f01357f69e924de30f5d7fd1ac6c9e',
     $old0$  IF v_incomplete > 0 THEN
    v_next := 'cancelled';
    UPDATE public.lab_arena_rounds
    SET status = 'cancelled', status_generation = status_generation + 1,
        stage_generation = v_generation,
        cancel_reason = 'execution_incomplete:stage1:' || v_incomplete::TEXT
    WHERE round_id = p_round_id;
  ELSE
    v_next := 'stage1_closed';
    UPDATE public.lab_arena_rounds
    SET status = v_next, status_generation = status_generation + 1,
        stage_generation = v_generation
    WHERE round_id = p_round_id;
  END IF;$old0$,
     $new0$  -- Mark only the assignments counted as infrastructure-incomplete.
  -- A failure after claim cutoff may have no retry run and retain its cause.
  UPDATE public.lab_arena_runs AS incomplete
  SET terminal_doc = COALESCE(incomplete.terminal_doc, '{}'::JSONB) ||
    '{"infrastructure_incomplete":true}'::JSONB
  WHERE incomplete.run_id = ANY(v_incomplete_run_ids)
    AND incomplete.status = 'failed'
    AND incomplete.per_icp_score IS NULL;
  GET DIAGNOSTICS v_marked = ROW_COUNT;
  IF v_marked <> v_incomplete THEN
    RAISE EXCEPTION 'lab_arena_incomplete_marker_mismatch' USING ERRCODE = '23514';
  END IF;
  v_next := 'stage1_closed';
  UPDATE public.lab_arena_rounds
  SET status = v_next, status_generation = status_generation + 1,
      stage_generation = v_generation
  WHERE round_id = p_round_id;$new0$),
    ('public.lab_arena_close_stage(text,smallint)',
     '545b91b93455e5d438f1cefe0587090591d94301a8b2973ce1c7a3e3b48a36f0',
     $old1$  IF v_incomplete > 0 THEN
    v_next := 'cancelled';
    UPDATE public.lab_arena_rounds
    SET status = 'cancelled',
        status_generation = status_generation + 1,
        stage_generation = v_generation,
        cancel_reason =
          'execution_incomplete:stage' || p_stage::TEXT || ':' || v_incomplete::TEXT
    WHERE round_id = p_round_id;
  ELSE
    v_next := 'stage' || p_stage::TEXT || '_closed';
    UPDATE public.lab_arena_rounds
    SET status = v_next,
        status_generation = status_generation + 1,
        stage_generation = v_generation
    WHERE round_id = p_round_id;
  END IF;$old1$,
     $new1$  -- Keep all complete results and mark only the assignments counted
  -- as infrastructure-incomplete, with their original failure cause.
  UPDATE public.lab_arena_runs AS incomplete
  SET terminal_doc = COALESCE(incomplete.terminal_doc, '{}'::JSONB) ||
    '{"infrastructure_incomplete":true}'::JSONB
  WHERE incomplete.run_id = ANY(v_incomplete_run_ids)
    AND incomplete.status = 'failed'
    AND incomplete.per_icp_score IS NULL;
  GET DIAGNOSTICS v_marked = ROW_COUNT;
  IF v_marked <> v_incomplete THEN
    RAISE EXCEPTION 'lab_arena_incomplete_marker_mismatch' USING ERRCODE = '23514';
  END IF;
  v_next := 'stage' || p_stage::TEXT || '_closed';
  UPDATE public.lab_arena_rounds
  SET status = v_next,
      status_generation = status_generation + 1,
      stage_generation = v_generation
  WHERE round_id = p_round_id;$new1$)
  ) AS patches(signature,postimage,old_fragment,new_fragment)
  LOOP
    v_definition:=pg_catalog.pg_get_functiondef(v_patch.signature::REGPROCEDURE);
    v_definition:=pg_catalog.replace(v_definition,
      $old_decl$  v_incomplete INTEGER;$old_decl$,
      $new_decl$  v_incomplete INTEGER;
  v_incomplete_run_ids TEXT[];
  v_marked INTEGER;$new_decl$);
    v_definition:=pg_catalog.replace(v_definition,
      $old_count$  SELECT COUNT(*) INTO v_incomplete
  FROM ($old_count$,
      $new_count$  SELECT COUNT(*), pg_catalog.array_agg(outcomes.latest_run_id)
    INTO v_incomplete, v_incomplete_run_ids
  FROM ($new_count$);
    v_definition:=pg_catalog.replace(v_definition,
      $old_latest$      runs.assignment_id,
      pg_catalog.bool_or(runs.status = 'accepted') AS has_accepted,$old_latest$,
      $new_latest$      runs.assignment_id,
      (pg_catalog.array_agg(runs.run_id ORDER BY runs.attempt DESC))[1]
        AS latest_run_id,
      pg_catalog.bool_or(runs.status = 'accepted') AS has_accepted,$new_latest$);
    IF pg_catalog.length(v_definition)-pg_catalog.length(
      pg_catalog.replace(v_definition,v_patch.old_fragment,''))
       <> pg_catalog.length(v_patch.old_fragment) THEN
      RAISE EXCEPTION 'Arena partial close fragment differs';
    END IF;
    v_definition:=pg_catalog.replace(v_definition,v_patch.old_fragment,v_patch.new_fragment);
    IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_patch.postimage THEN
      RAISE EXCEPTION 'Arena partial close postimage differs';
    END IF;
    EXECUTE v_definition;
    IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
      v_patch.signature::REGPROCEDURE),'sha256'),'hex')<>v_patch.postimage THEN
      RAISE EXCEPTION 'Arena partial close readback differs';
    END IF;
  END LOOP;
END;
$partial_execution_close_422$;
COMMIT;
