-- Recognize already-authorized baseline continuation during guarded drain.
-- Preserve capture, generation, stopped-host, accounting and expiry authority.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $baseline_restart_phase486$
DECLARE
  v_signatures TEXT[] := ARRAY['public.lab_arena_expire_leases(text)','public.lab_arena_restart_quiescence_v1(text,text,bigint)','public.lab_arena__restart_drain_state_v1()'];
  v_preimages TEXT[] := ARRAY['f2f39310a400a5e3432ff120306594195b4d83eb5df8a9262b7c13eaf8a9cd33','01184c53733ee5159f642c8071864a7a897106f3d70521f0232dc44e44a2d483','65384dfe99fbbac1f484139cfcf2a213079601b8d3af458a7153ec0ef7b2438a'];
  v_postimages TEXT[] := ARRAY['c8e1723044ab74469c9ed8f808cfa50f608149cc1bc86d9c19612474f0393d72','052c937b72b622d3e627ffcb02a7364c98d479bbbc58c151129d0085658dce7b','db4e1a3b9d51299e00d2b2a350f47700da1a8e2fbf13795bfc9fb54f1bab65f1'];
  v_before TEXT[]; v_after TEXT[]; v_definition TEXT; v_identity JSONB;
  v_old TEXT[]; v_new TEXT[]; v_counts INTEGER[]; v_i INTEGER; v_j INTEGER;
BEGIN
  FOR v_i IN 1..3 LOOP
    SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.jsonb_build_array(
      owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_definition,v_identity FROM pg_catalog.pg_proc p
    JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
    WHERE p.oid=pg_catalog.to_regprocedure(v_signatures[v_i]);
    IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
      'lab_arena_owner',CASE v_i
        WHEN 3 THEN '{lab_arena_owner=X/lab_arena_owner}'
        WHEN 2 THEN '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner,service_role=X/lab_arena_owner}'
        ELSE '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}' END,
      TRUE,CASE v_i WHEN 3 THEN 's' ELSE 'v' END,
      ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
      RAISE EXCEPTION 'baseline restart phase security shape differs: %',v_signatures[v_i] USING ERRCODE='55000';
    END IF;
    v_before:=pg_catalog.array_append(v_before,v_definition);
  END LOOP;
  IF ARRAY(SELECT pg_catalog.encode(extensions.digest(x,'sha256'),'hex')
    FROM pg_catalog.unnest(v_before) x)=v_postimages THEN RETURN; END IF;
  IF ARRAY(SELECT pg_catalog.encode(extensions.digest(x,'sha256'),'hex')
    FROM pg_catalog.unnest(v_before) x)<>v_preimages THEN
    RAISE EXCEPTION 'baseline restart phase preimage differs' USING ERRCODE='55000';
  END IF;
  FOR v_i IN 1..3 LOOP
    CASE v_i
      WHEN 1 THEN
        v_old:=ARRAY[$old486_1_0$      AND v_round.status = 'stage' || candidate.stage::TEXT
          || CASE candidate.kind WHEN 'score' THEN '_scoring' ELSE '' END$old486_1_0$,$old486_1_1$       AND v_round.status = ('stage' || v_run.stage::TEXT
           || CASE v_run.kind WHEN 'score' THEN '_scoring' ELSE '' END)$old486_1_1$,$old486_1_2$          OR v_round.status<>('stage'||candidate.stage::text
            ||CASE candidate.kind WHEN 'score' THEN '_scoring' ELSE '' END)$old486_1_2$];
        v_new:=ARRAY[$new486_1_0$      AND (v_round.status = 'stage' || candidate.stage::TEXT
          || CASE candidate.kind WHEN 'score' THEN '_scoring' ELSE '' END
        OR public.lab_arena__baseline_recovery_phase_v1(v_round,candidate))$new486_1_0$,$new486_1_1$       AND (v_round.status = ('stage' || v_run.stage::TEXT
           || CASE v_run.kind WHEN 'score' THEN '_scoring' ELSE '' END)
         OR public.lab_arena__baseline_recovery_phase_v1(v_round,v_run))$new486_1_1$,$new486_1_2$          OR NOT (v_round.status=('stage'||candidate.stage::text
            ||CASE candidate.kind WHEN 'score' THEN '_scoring' ELSE '' END)
            OR public.lab_arena__baseline_recovery_phase_v1(v_round,candidate))$new486_1_2$];
        v_counts:=ARRAY[2,2,1];
      WHEN 2 THEN
        v_old:=ARRAY[$old486_2_0$                OR (round.status='stage2' AND overdue.stage=2)))
          )$old486_2_0$];
        v_new:=ARRAY[$new486_2_0$                OR (round.status='stage2' AND overdue.stage=2)))
            OR public.lab_arena__baseline_recovery_phase_v1(round,overdue)
          )$new486_2_0$];
        v_counts:=ARRAY[1];
      WHEN 3 THEN
        v_old:=ARRAY[$old486_3_0$       AND recovery_round.status = ('stage' || runs.stage::TEXT
           || CASE runs.kind WHEN 'score' THEN '_scoring' ELSE '' END)$old486_3_0$];
        v_new:=ARRAY[$new486_3_0$       AND (recovery_round.status = ('stage' || runs.stage::TEXT
           || CASE runs.kind WHEN 'score' THEN '_scoring' ELSE '' END)
         OR public.lab_arena__baseline_recovery_phase_v1(recovery_round,runs))$new486_3_0$];
        v_counts:=ARRAY[1];
    END CASE;
    v_definition:=v_before[v_i];
    FOR v_j IN 1..pg_catalog.cardinality(v_old) LOOP
      IF (pg_catalog.length(v_definition)-pg_catalog.length(
          pg_catalog.replace(v_definition,v_old[v_j],'')))/pg_catalog.length(v_old[v_j])<>v_counts[v_j] THEN
        RAISE EXCEPTION 'baseline restart phase anchor differs: %',v_signatures[v_i] USING ERRCODE='55000';
      END IF;
      v_definition:=pg_catalog.replace(v_definition,v_old[v_j],v_new[v_j]);
    END LOOP;
    IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimages[v_i] THEN
      RAISE EXCEPTION 'baseline restart phase postimage differs: %',v_signatures[v_i] USING ERRCODE='55000';
    END IF;
    v_after:=pg_catalog.array_append(v_after,v_definition);
  END LOOP;
  FOR v_i IN 1..3 LOOP
    EXECUTE v_after[v_i];
    IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
      v_signatures[v_i]::REGPROCEDURE),'sha256'),'hex')<>v_postimages[v_i] THEN
      RAISE EXCEPTION 'baseline restart phase readback differs: %',v_signatures[v_i] USING ERRCODE='55000';
    END IF;
  END LOOP;
END;
$baseline_restart_phase486$;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst,'reload schema';
COMMIT;
