-- Extend owner-authorized stopped-host drain to fully settled execute leases.
-- Reuse migration442's ownership/cleanup/accounting proof and bounded retries.
-- Completed-unpriced recovery remains score-only; no round or ledger data changes.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $stopped_execute_drain461$
DECLARE
  v_signatures TEXT[] := ARRAY['public.lab_arena_expire_leases(text)','public.lab_arena_restart_quiescence_v1(text,text,bigint)','public.lab_arena__restart_drain_state_v1()'];
  v_preimages TEXT[] := ARRAY['dadb3e139bb972ca8d3ef4894e0c2e3ba087f55c57973adaa21ed1a8d46d6c0a','9e7334a5d2d4d05e673351627061a84558fc865585e479bd2b35231b405e4abc','8d4147a0ade1685f14fbdda3af24106384bb6173f375cac417dc099ab627b943'];
  v_postimages TEXT[] := ARRAY['6fb3d102cf53f32ec2cdc653d3a43af398897019a16db83885136defa96b17bb','01184c53733ee5159f642c8071864a7a897106f3d70521f0232dc44e44a2d483','8d5b985c4186104bb35f4716fe785810d9bb2c580b2f3530cda5acdee3e77bc4'];
  v_before TEXT[];
  v_after TEXT[];
  v_definition TEXT;
  v_identity JSONB;
  v_old TEXT[];
  v_new TEXT[];
  v_i INTEGER;
  v_j INTEGER;
BEGIN
  FOR v_i IN 1..3 LOOP
    SELECT pg_get_functiondef(p.oid),jsonb_build_array(owner.rolname,p.proacl::text,
      p.prosecdef,p.provolatile,p.proconfig)
    INTO v_definition,v_identity FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner
    WHERE p.oid=v_signatures[v_i]::regprocedure;
    IF v_definition IS NULL OR v_identity->>0 IS DISTINCT FROM 'lab_arena_owner'
      OR v_identity->>1 IS DISTINCT FROM (CASE WHEN v_i=3
        THEN '{lab_arena_owner=X/lab_arena_owner}'
        WHEN v_i=2 THEN '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner,service_role=X/lab_arena_owner}'
        ELSE '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}' END)
      OR v_identity->2 IS DISTINCT FROM 'true'::jsonb
      OR v_identity->>3 IS DISTINCT FROM (CASE WHEN v_i=3 THEN 's' ELSE 'v' END)
      OR v_identity->4 IS DISTINCT FROM '["search_path=pg_catalog, public"]'::jsonb THEN
      RAISE EXCEPTION 'Arena stopped execute drain security shape differs' USING ERRCODE='55000';
    END IF;
    v_before:=array_append(v_before,v_definition);
  END LOOP;
  IF ARRAY(SELECT encode(extensions.digest(x,'sha256'),'hex') FROM unnest(v_before) x)=v_postimages THEN RETURN; END IF;
  IF ARRAY(SELECT encode(extensions.digest(x,'sha256'),'hex') FROM unnest(v_before) x)<>v_preimages THEN
    RAISE EXCEPTION 'Arena stopped execute drain preimage differs' USING ERRCODE='55000';
  END IF;
  FOR v_i IN 1..3 LOOP
    CASE v_i
      WHEN 1 THEN
        v_old:=ARRAY[$old1_0$        AND candidate.kind='score' AND EXISTS ($old1_0$,$old1_1$        AND (candidate.kind<>'score' OR candidate.stage_generation<>v_round.stage_generation
          OR v_round.status<>'stage'||candidate.stage::text||'_scoring'$old1_1$];
        v_new:=ARRAY[$new1_0$        AND candidate.kind IN ('score','execute') AND EXISTS ($new1_0$,$new1_1$        AND (candidate.kind NOT IN ('score','execute') OR candidate.stage_generation<>v_round.stage_generation
          OR v_round.status<>('stage'||candidate.stage::text
            ||CASE candidate.kind WHEN 'score' THEN '_scoring' ELSE '' END)$new1_1$];
      WHEN 2 THEN
        v_old:=ARRAY[$old2_0$          AND runs.status='leased' AND runs.kind='score'$old2_0$];
        v_new:=ARRAY[$new2_0$          AND runs.status='leased' AND runs.kind IN ('score','execute')$new2_0$];
      WHEN 3 THEN
        v_old:=ARRAY[$old3_0$          AND runs.kind='score' AND runs.result_doc IS NULL AND runs.output_ref IS NULL$old3_0$,$old3_1$          AND runs.terminal_doc->>'recovery_reason' IN (
            'authenticated_settled_score_runtime_host_error',
            'authenticated_completed_unpriced_score_runtime_host_error')$old3_1$];
        v_new:=ARRAY[$new3_0$          AND runs.kind IN ('score','execute') AND runs.result_doc IS NULL AND runs.output_ref IS NULL$new3_0$,$new3_1$          AND ((runs.kind='score' AND runs.terminal_doc->>'recovery_reason' IN (
            'authenticated_settled_score_runtime_host_error',
            'authenticated_completed_unpriced_score_runtime_host_error'))
            OR (runs.kind='execute' AND runs.terminal_doc->>'recovery_reason'=
              'authenticated_settled_execute_runtime_host_error'))$new3_1$];
    END CASE;
    v_definition:=v_before[v_i];
    FOR v_j IN 1..cardinality(v_old) LOOP
      IF (length(v_definition)-length(replace(v_definition,v_old[v_j],'')))
          /length(v_old[v_j])<>1 THEN
        RAISE EXCEPTION 'Arena stopped execute drain anchor differs' USING ERRCODE='55000';
      END IF;
      v_definition:=replace(v_definition,v_old[v_j],v_new[v_j]);
    END LOOP;
    IF encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimages[v_i] THEN
      RAISE EXCEPTION 'Arena stopped execute drain postimage differs' USING ERRCODE='55000';
    END IF;
    v_after:=array_append(v_after,v_definition);
  END LOOP;
  FOR v_i IN 1..3 LOOP
    EXECUTE v_after[v_i];
    IF encode(extensions.digest(pg_get_functiondef(v_signatures[v_i]::regprocedure),'sha256'),'hex')<>v_postimages[v_i] THEN
      RAISE EXCEPTION 'Arena stopped execute drain readback differs' USING ERRCODE='55000';
    END IF;
  END LOOP;
END;
$stopped_execute_drain461$;
NOTIFY pgrst,'reload schema';
COMMIT;
