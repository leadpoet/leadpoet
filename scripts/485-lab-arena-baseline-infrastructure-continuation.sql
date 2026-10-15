-- General continuation of proved infrastructure-failed baseline work.
-- Frozen inputs, accepted work, raw costs, budgets and provider call caps remain.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $continuation485_0$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena__baseline_recovery_phase_v1(public.lab_arena_rounds,public.lab_arena_runs)';
  v_preimage CONSTANT TEXT := '2bcb79d71b4b7c8e15bb9ae7e18ff4c9e1a4d971b1d92768008731c0a59201e4';
  v_postimage CONSTANT TEXT := '4acbf6eba8b497455270d1f7b1ea4e407e3dbf175d136005e38fe81af90a9428';
  v_definition TEXT; v_identity JSONB;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.jsonb_build_array(
    owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner}',TRUE,'s',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'baseline continuation security shape differs: %',v_signature USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN RETURN; END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'baseline continuation preimage differs: %',v_signature USING ERRCODE='55000';
  END IF;
  v_definition:=pg_catalog.replace(v_definition,$old485$(p_run.kind='execute' AND p_round.status='stage2')$old485$,$new485$(p_run.kind='execute' AND p_round.status IN ('stage2','stage2_closed','stage2_scoring'))$new485$);
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'baseline continuation postimage differs: %',v_signature USING ERRCODE='55000';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
      v_signature::REGPROCEDURE),'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'baseline continuation readback differs: %',v_signature USING ERRCODE='55000';
  END IF;
END;
$continuation485_0$;

DO $continuation485_1$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena__baseline_recovery_run_v1(public.lab_arena_runs)';
  v_preimage CONSTANT TEXT := '8d419093cc5a2bbd0a8fbe7b64cb768d2c782d8e8fa74676488c632b3491bb6e';
  v_postimage CONSTANT TEXT := '2f4352b22ac198b10f70dd6b90e9320db890b395ca53e3a6a56375cce04dcab6';
  v_definition TEXT; v_identity JSONB;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.jsonb_build_array(
    owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner}',TRUE,'s',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'baseline continuation security shape differs: %',v_signature USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN RETURN; END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'baseline continuation preimage differs: %',v_signature USING ERRCODE='55000';
  END IF;
  v_definition:=pg_catalog.replace(v_definition,$old485$    OR v_round.configuration_doc->>'max_attempts_per_assignment' IS DISTINCT FROM '2'$old485$,$new485$    OR COALESCE(v_round.configuration_doc->>'max_attempts_per_assignment','') !~ '^[1-9][0-9]*$'$new485$);
  v_definition:=pg_catalog.replace(v_definition,$old485$IF p_run.attempt IN (1,2) AND v_source.attempt=3 THEN$old485$,$new485$IF p_run.attempt IN (1,2) AND v_source.attempt>(v_round.configuration_doc->>'max_attempts_per_assignment')::INTEGER THEN$new485$);
  v_definition:=pg_catalog.replace(v_definition,$old485$  RETURN p_run.attempt=3 AND
    (SELECT count(*)=2$old485$,$new485$  RETURN ((p_run.kind='execute' AND p_run.attempt>(v_round.configuration_doc->>'max_attempts_per_assignment')::INTEGER)
      OR (p_run.kind='score' AND p_run.attempt=3)) AND
    (SELECT count(*)=p_run.attempt-1$new485$);
  v_definition:=pg_catalog.replace(v_definition,$old485$parent.attempt IN (1,2))$old485$,$new485$parent.attempt BETWEEN 1 AND p_run.attempt-1)$new485$);
  v_definition:=pg_catalog.replace(v_definition,$old485$other.attempt>3$old485$,$new485$other.attempt>p_run.attempt$new485$);
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'baseline continuation postimage differs: %',v_signature USING ERRCODE='55000';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
      v_signature::REGPROCEDURE),'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'baseline continuation readback differs: %',v_signature USING ERRCODE='55000';
  END IF;
END;
$continuation485_1$;

DO $continuation485_2$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)';
  v_preimage CONSTANT TEXT := '18a32813090fbe6ff0f323faa6361670e4bfe9317ecc5e56a01e9b01ca1d2674';
  v_postimage CONSTANT TEXT := '89cfed833d6bdeb3c15b0677b155679c33cf524d0055cd8ccab56986138e3c0d';
  v_definition TEXT; v_identity JSONB;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.jsonb_build_array(
    owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner}',TRUE,'s',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'baseline continuation security shape differs: %',v_signature USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN RETURN; END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'baseline continuation preimage differs: %',v_signature USING ERRCODE='55000';
  END IF;
  v_definition:=pg_catalog.replace(v_definition,$old485$recovered_source.kind='execute' AND recovered_source.attempt=3$old485$,$new485$recovered_source.kind='execute' AND recovered_source.attempt>
          (p_round.configuration_doc->>'max_attempts_per_assignment')::INTEGER$new485$);
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'baseline continuation postimage differs: %',v_signature USING ERRCODE='55000';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
      v_signature::REGPROCEDURE),'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'baseline continuation readback differs: %',v_signature USING ERRCODE='55000';
  END IF;
END;
$continuation485_2$;

DO $continuation485_4$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)';
  v_preimage CONSTANT TEXT := '1c5189dea108a913e420a7c41737f245f21168a17e4930a391981f805d440b80';
  v_postimage CONSTANT TEXT := 'b9da18be2a587210faae63a5f340e956dc437fd3e2d0b706e9621757aad75913';
  v_definition TEXT; v_identity JSONB;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.jsonb_build_array(
    owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'baseline continuation security shape differs: %',v_signature USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN RETURN; END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'baseline continuation preimage differs: %',v_signature USING ERRCODE='55000';
  END IF;
  v_definition:=pg_catalog.replace(v_definition,$old485$      runs.kind <> 'execute'
      OR v_round.configuration_doc ->> 'execution_sequence_policy'$old485$,$new485$      runs.kind <> 'execute'
      OR (public.lab_arena__baseline_recovery_phase_v1(v_round,runs)
        AND pg_catalog.clock_timestamp() <
          (v_round.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ)
      OR v_round.configuration_doc ->> 'execution_sequence_policy'$new485$);
  v_definition:=pg_catalog.replace(v_definition,$old485$    AND (runs.kind <> 'score' OR runs.attempt=1 OR$old485$,$new485$    -- Infrastructure successors share bounded queue backoff: 30s,60s,...,900s.
    AND (runs.kind<>'execute' OR NOT public.lab_arena__baseline_recovery_run_v1(runs) OR
      pg_catalog.clock_timestamp() >= runs.created_at + pg_catalog.make_interval(
        secs => LEAST(900::NUMERIC,30*power(2::NUMERIC,LEAST(runs.attempt-
          (v_round.configuration_doc->>'max_attempts_per_assignment')::INTEGER-1,5)))::INTEGER))
    AND (runs.kind <> 'score' OR runs.attempt=1 OR$new485$);
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'baseline continuation postimage differs: %',v_signature USING ERRCODE='55000';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
      v_signature::REGPROCEDURE),'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'baseline continuation readback differs: %',v_signature USING ERRCODE='55000';
  END IF;
END;
$continuation485_4$;

DO $continuation485_5$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_close_scoring(text,smallint)';
  v_preimage CONSTANT TEXT := '9efa74e255735c4aa4ca77e694cd9bf09920bf53bd7c3be5eb118c051d279953';
  v_postimage CONSTANT TEXT := 'c1e8f7c9646cc16c32f451456c020859da55c48c389bd27768b42ade794cc090';
  v_definition TEXT; v_identity JSONB;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.jsonb_build_array(
    owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'baseline continuation security shape differs: %',v_signature USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN RETURN; END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'baseline continuation preimage differs: %',v_signature USING ERRCODE='55000';
  END IF;
  v_definition:=pg_catalog.replace(v_definition,$old485$  v_generation := v_round.stage_generation + 1;$old485$,$new485$  IF p_stage=2 AND public.lab_arena_judge_continuation_enabled_v1(p_round_id)
    AND EXISTS (SELECT 1 FROM public.lab_arena_runs recovery
    WHERE recovery.round_id=p_round_id AND recovery.kind='execute'
      AND ((public.lab_arena__baseline_recovery_run_v1(recovery)
      AND (recovery.status IN ('leased','submitted')
        OR (recovery.status='pending' AND clock_timestamp()<
          (v_round.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ)
        OR (recovery.status='accepted' AND NOT EXISTS (SELECT 1 FROM public.lab_arena_runs judge
          WHERE judge.scored_run_id=recovery.run_id AND judge.kind='score'))))
        OR (recovery.status='failed' AND recovery.attempt<2147483647
          AND clock_timestamp()<(v_round.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ
          AND NOT EXISTS (SELECT 1 FROM public.lab_arena_runs later
            WHERE later.assignment_id=recovery.assignment_id AND later.attempt>recovery.attempt)
          AND public.lab_arena__baseline_recovery_run_v1(jsonb_populate_record(recovery,
            jsonb_build_object('attempt',recovery.attempt+1,
              'run_id',recovery.assignment_id||':'||(recovery.attempt+1)::TEXT)))))) THEN
    RETURN jsonb_build_object('status','draining','round_status',v_round.status,
      'stage_generation',v_round.stage_generation);
  END IF;
  v_generation := v_round.stage_generation + 1;$new485$);
  v_definition:=pg_catalog.replace(v_definition,$old485$))) AND kind = 'score' AND status$old485$,$new485$)) ) AND (kind = 'score' OR (p_stage=2 AND kind='execute'
        AND public.lab_arena__baseline_recovery_run_v1(lab_arena_runs))) AND status$new485$);
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'baseline continuation postimage differs: %',v_signature USING ERRCODE='55000';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
      v_signature::REGPROCEDURE),'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'baseline continuation readback differs: %',v_signature USING ERRCODE='55000';
  END IF;
END;
$continuation485_5$;

DO $continuation485_6$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_close_stage(text,smallint)';
  v_preimage CONSTANT TEXT := '635c881472b7d066ac1e22b0baab79659058710aa74a0310ddd381a6ab2d3aa4';
  v_postimage CONSTANT TEXT := '676aa701fd9a81031a4f5f36e4029c9960331dd43065824a714587d50c7444ea';
  v_definition TEXT; v_identity JSONB;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.jsonb_build_array(
    owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'baseline continuation security shape differs: %',v_signature USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN RETURN; END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'baseline continuation preimage differs: %',v_signature USING ERRCODE='55000';
  END IF;
  v_definition:=pg_catalog.replace(v_definition,$old485$      AND status IN ('leased', 'pending', 'submitted')$old485$,$new485$      AND status IN ('leased', 'pending', 'submitted')
      AND NOT (p_stage=2 AND kind='execute' AND status='pending'
        AND public.lab_arena__baseline_recovery_run_v1(lab_arena_runs)
        AND clock_timestamp()<(v_round.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ)$new485$);
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'baseline continuation postimage differs: %',v_signature USING ERRCODE='55000';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
      v_signature::REGPROCEDURE),'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'baseline continuation readback differs: %',v_signature USING ERRCODE='55000';
  END IF;
END;
$continuation485_6$;

DO $continuation485_7$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_recover_baseline_v1(text,jsonb)';
  v_preimage CONSTANT TEXT := 'fc708bc7e5727bf5fd049ba8381d3a23be07c595a4c36de8ed9dfc0099fef147';
  v_postimage CONSTANT TEXT := '09b077c40a98ad818ca9bb994c8d852af5ebe3006c439877332f2158a3570336';
  v_definition TEXT; v_identity JSONB;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.jsonb_build_array(
    owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'baseline continuation security shape differs: %',v_signature USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN RETURN; END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'baseline continuation preimage differs: %',v_signature USING ERRCODE='55000';
  END IF;
  v_definition:=pg_catalog.replace(v_definition,$old485$    OR clock_timestamp()>=(v_round.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ$old485$,$new485$$new485$);
  v_definition:=pg_catalog.replace(v_definition,$old485$WHERE round_id=p_round_id AND stage=1 AND attempt=2 AND status='failed'
      AND kind IN ('execute','score')$old485$,$new485$WHERE round_id=p_round_id AND stage=1 AND status='failed'
      AND attempt<2147483647
      AND ((kind='execute' AND attempt>=(v_round.configuration_doc->>'max_attempts_per_assignment')::INTEGER)
        OR (kind='score' AND attempt=2))
      AND NOT EXISTS (SELECT 1 FROM public.lab_arena_runs later
        WHERE later.assignment_id=lab_arena_runs.assignment_id AND later.attempt>lab_arena_runs.attempt)$new485$);
  v_definition:=pg_catalog.replace(v_definition,$old485$'run_id',v_parent.assignment_id||':3','attempt',3$old485$,$new485$'run_id',v_parent.assignment_id||':'||(v_parent.attempt+1)::TEXT,'attempt',v_parent.attempt+1$new485$);
  v_definition:=pg_catalog.replace(v_definition,$old485$      AND (v_parent.kind<>'execute' OR (v_round.status='stage2'
        AND clock_timestamp()<(v_round.configuration_doc#>>'{schedule,stage_2_close}')::TIMESTAMPTZ))$old485$,$new485$      AND clock_timestamp()<(v_round.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ$new485$);
  v_definition:=pg_catalog.replace(v_definition,$old485$(other.attempt>=3 OR other.status$old485$,$new485$(other.attempt>v_parent.attempt OR other.status$new485$);
  v_definition:=pg_catalog.replace(v_definition,$old485$v_new.icp_position,3,'pending'$old485$,$new485$v_new.icp_position,v_new.attempt,'pending'$new485$);
  v_definition:=pg_catalog.replace(v_definition,$old485$r.status='pending' AND r.kind='score'$old485$,$new485$r.status='pending' AND r.kind IN ('execute','score')$new485$);
  v_definition:=pg_catalog.replace(v_definition,$old485$-- Pending recovery judges cross existing generation transitions, never
  -- leased work. Claims remain blocked until the ordinary stage2 score phase.$old485$,$new485$-- Pending baseline work crosses generation transitions; live execution
  -- drains before stage close, and existing judge continuation owns retries.$new485$);
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'baseline continuation postimage differs: %',v_signature USING ERRCODE='55000';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
      v_signature::REGPROCEDURE),'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'baseline continuation readback differs: %',v_signature USING ERRCODE='55000';
  END IF;
END;
$continuation485_7$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst,'reload schema';
COMMIT;
