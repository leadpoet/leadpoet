-- Align authenticated restart drain with the existing baseline recovery contract.
-- No admission, deadline, capture, host proof, accounting, or result changes.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $baseline_drain487$
DECLARE
  v_signatures TEXT[] := ARRAY['public.lab_arena_expire_leases(text)','public.lab_arena__restart_drain_state_v1()'];
  v_preimages TEXT[] := ARRAY['c8e1723044ab74469c9ed8f808cfa50f608149cc1bc86d9c19612474f0393d72','db4e1a3b9d51299e00d2b2a350f47700da1a8e2fbf13795bfc9fb54f1bab65f1'];
  v_postimages TEXT[] := ARRAY['35cbc97d9d8390f2d208e36617e7684cfcad4e1d1a8296ca4fbb461cc625f594','f00bdd46ba168268a5541fa3f47baa5edbb9aa0db0e41b03e6420259b0e2c469'];
  v_helper_signature CONSTANT TEXT := 'public.lab_arena__baseline_drain_phase_v1(public.lab_arena_rounds,public.lab_arena_runs)';
  v_helper_image CONSTANT TEXT := 'a806f446494828a61696202a353a43464b831dda65645263c922e838edcd6031';
  v_before TEXT[]; v_after TEXT[]; v_definition TEXT; v_identity JSONB;
  v_i INTEGER; v_patch RECORD;
BEGIN
  -- Verify the exact authorities this helper consumes before any replacement.
  FOR v_patch IN SELECT * FROM (VALUES
    ('public.lab_arena__baseline_recovery_phase_v1(public.lab_arena_rounds,public.lab_arena_runs)','4acbf6eba8b497455270d1f7b1ea4e407e3dbf175d136005e38fe81af90a9428'),
    ('public.lab_arena__baseline_recovery_run_v1(public.lab_arena_runs)','2f4352b22ac198b10f70dd6b90e9320db890b395ca53e3a6a56375cce04dcab6'),
    ('public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)','89cfed833d6bdeb3c15b0677b155679c33cf524d0055cd8ccab56986138e3c0d')
  ) AS dependencies(signature,image) LOOP
    SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.jsonb_build_array(
      owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_definition,v_identity FROM pg_catalog.pg_proc p
    JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
    WHERE p.oid=pg_catalog.to_regprocedure(v_patch.signature);
    IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')
         IS DISTINCT FROM v_patch.image
       OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array('lab_arena_owner',
         '{lab_arena_owner=X/lab_arena_owner}',TRUE,'s',
         ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
      RAISE EXCEPTION 'baseline drain authority differs: %',v_patch.signature USING ERRCODE='55000';
    END IF;
  END LOOP;
  FOR v_i IN 1..2 LOOP
    SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.jsonb_build_array(
      owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_definition,v_identity FROM pg_catalog.pg_proc p
    JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
    WHERE p.oid=pg_catalog.to_regprocedure(v_signatures[v_i]);
    IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
      'lab_arena_owner',CASE v_i WHEN 1 THEN
        '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
        ELSE '{lab_arena_owner=X/lab_arena_owner}' END,
      TRUE,CASE v_i WHEN 1 THEN 'v' ELSE 's' END,
      ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
      RAISE EXCEPTION 'baseline drain security shape differs: %',v_signatures[v_i] USING ERRCODE='55000';
    END IF;
    v_before:=array_append(v_before,v_definition);
  END LOOP;
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.jsonb_build_array(
    owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_helper_signature);
  IF v_definition IS NOT NULL THEN
    IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_helper_image
      OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array('lab_arena_owner',
        '{lab_arena_owner=X/lab_arena_owner}',TRUE,'s',
        ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
      RAISE EXCEPTION 'baseline drain helper differs' USING ERRCODE='55000';
    END IF;
    IF ARRAY(SELECT pg_catalog.encode(extensions.digest(x,'sha256'),'hex')
      FROM unnest(v_before) x)=v_postimages THEN RETURN; END IF;
  END IF;
  IF ARRAY(SELECT pg_catalog.encode(extensions.digest(x,'sha256'),'hex')
    FROM unnest(v_before) x)<>v_preimages THEN
    RAISE EXCEPTION 'baseline drain preimage differs' USING ERRCODE='55000';
  END IF;
  -- Install one owner-only phase contract; historical witnesses stay receipt-only.
  EXECUTE $helper487$CREATE OR REPLACE FUNCTION public.lab_arena__baseline_drain_phase_v1(p_round lab_arena_rounds, p_run lab_arena_runs)
 RETURNS boolean
 LANGUAGE sql
 STABLE SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public'
AS $function$
  -- Recovery admission already proves the frozen baseline execution lineage.
  -- Later judges retain that source authority and their existing judge chain.
  SELECT COALESCE(p_run.round_id=p_round.round_id AND
    (public.lab_arena__baseline_recovery_phase_v1(p_round,p_run)
      OR (p_run.kind='score' AND p_round.status='stage2_scoring'
        AND public.lab_arena__judge_continuation_v1(p_round,p_run)
        AND EXISTS (SELECT 1 FROM public.lab_arena_runs source
          WHERE source.run_id=p_run.scored_run_id AND source.kind='execute'
            AND source.status='accepted' AND source.terminal_cause='accepted'
            AND source.output_ref IS NOT NULL
            AND public.lab_arena__baseline_recovery_run_v1(source)))),FALSE);
$function$
$helper487$;
  ALTER FUNCTION public.lab_arena__baseline_drain_phase_v1(public.lab_arena_rounds,public.lab_arena_runs) OWNER TO lab_arena_owner;
  REVOKE ALL ON FUNCTION public.lab_arena__baseline_drain_phase_v1(public.lab_arena_rounds,public.lab_arena_runs) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.jsonb_build_array(
    owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_helper_signature::REGPROCEDURE;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_helper_image
    OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array('lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner}',TRUE,'s',
      ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'baseline drain helper readback differs' USING ERRCODE='55000';
  END IF;
  v_after:=v_before;
  FOR v_patch IN SELECT * FROM (VALUES
    (1,3,$old487_0$public.lab_arena__baseline_recovery_phase_v1(v_round,candidate)$old487_0$,$new487_0$public.lab_arena__baseline_drain_phase_v1(v_round,candidate)$new487_0$),
    (1,2,$old487_1$public.lab_arena__baseline_recovery_phase_v1(v_round,v_run)$old487_1$,$new487_1$public.lab_arena__baseline_drain_phase_v1(v_round,v_run)$new487_1$),
    (2,1,$old487_2$OR public.lab_arena__baseline_recovery_phase_v1(recovery_round,runs))$old487_2$,$new487_2$OR public.lab_arena__baseline_drain_phase_v1(recovery_round,runs)
         -- Only a durable expired receipt can use a later eligible witness.
         -- Live mutation checks above require this lease's own authority.
         OR (runs.kind='execute' AND runs.status='failed'
           AND runs.terminal_cause='lease_expired'
           AND runs.run_id=runs.assignment_id||':'||runs.attempt::TEXT
           AND EXISTS (SELECT 1 FROM public.lab_arena_runs successor
             WHERE successor.assignment_id=runs.assignment_id
               AND successor.attempt>runs.attempt
               AND successor.stage_generation=runs.stage_generation
               AND (successor.round_id,successor.submission_id,successor.miner_hotkey,
                 successor.stage,successor.icp_position,successor.kind)
                 IS NOT DISTINCT FROM
                 (runs.round_id,runs.submission_id,runs.miner_hotkey,
                  runs.stage,runs.icp_position,runs.kind)
               AND public.lab_arena__baseline_recovery_phase_v1(recovery_round,successor))))$new487_2$)
  ) AS patches(function_index,expected_count,old_text,new_text) LOOP
    v_definition:=v_after[v_patch.function_index];
    IF (pg_catalog.length(v_definition)-pg_catalog.length(
      pg_catalog.replace(v_definition,v_patch.old_text,'')))
      /pg_catalog.length(v_patch.old_text)<>v_patch.expected_count THEN
      RAISE EXCEPTION 'baseline drain anchor differs' USING ERRCODE='55000';
    END IF;
    v_after[v_patch.function_index]:=pg_catalog.replace(
      v_definition,v_patch.old_text,v_patch.new_text);
  END LOOP;
  FOR v_i IN 1..2 LOOP
    IF pg_catalog.encode(extensions.digest(v_after[v_i],'sha256'),'hex')<>v_postimages[v_i] THEN
      RAISE EXCEPTION 'baseline drain postimage differs' USING ERRCODE='55000';
    END IF;
    EXECUTE v_after[v_i];
    IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
      v_signatures[v_i]::REGPROCEDURE),'sha256'),'hex')<>v_postimages[v_i] THEN
      RAISE EXCEPTION 'baseline drain readback differs' USING ERRCODE='55000';
    END IF;
  END LOOP;
END;
$baseline_drain487$;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst,'reload schema';
COMMIT;
