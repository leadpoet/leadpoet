-- A recovered baseline execution is deliberately absent from the immutable
-- original scoring plan. Its exact 470 witness authorizes the same 472 judge
-- chain; all source, frozen-input, participant and rollout guards remain.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $recovered_source$
DECLARE definition TEXT; owner_name TEXT; secure BOOLEAN; configuration TEXT[];
  access_list TEXT; volatility "char";
  old_text TEXT := $old$OR NOT EXISTS (SELECT 1 FROM pg_catalog.jsonb_array_elements(
      COALESCE(CASE p_run.stage WHEN 1 THEN p_round.stage1_scoring_plan_doc$old$;
  new_text TEXT := $new$OR (NOT EXISTS (SELECT 1 FROM public.lab_arena_runs recovered_source
      WHERE recovered_source.run_id=p_run.scored_run_id
        AND recovered_source.kind='execute' AND recovered_source.attempt=3
        AND public.lab_arena__baseline_recovery_run_v1(recovered_source))
      AND NOT EXISTS (SELECT 1 FROM pg_catalog.jsonb_array_elements(
      COALESCE(CASE p_run.stage WHEN 1 THEN p_round.stage1_scoring_plan_doc$new$;
  end_old TEXT := $old$FROM public.lab_arena_runs source WHERE source.run_id=p_run.scored_run_id))
    THEN RETURN FALSE;$old$;
  end_new TEXT := $new$FROM public.lab_arena_runs source WHERE source.run_id=p_run.scored_run_id)))
    THEN RETURN FALSE;$new$;
BEGIN
  SELECT pg_get_functiondef(proc.oid),pg_get_userbyid(proc.proowner),
    proc.prosecdef,proc.proconfig,proc.proacl::TEXT,proc.provolatile
    INTO definition,owner_name,secure,configuration,access_list,volatility
    FROM pg_proc proc WHERE proc.oid=to_regprocedure(
      'public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)');
  IF definition IS NULL OR owner_name<>'lab_arena_owner'
    OR secure IS DISTINCT FROM TRUE
    OR access_list IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
    OR volatility IS DISTINCT FROM 's'
    OR configuration IS DISTINCT FROM ARRAY['search_path=pg_catalog, public'] THEN
    RAISE EXCEPTION 'recovered execution judge authority differs';
  END IF;
  IF strpos(definition,new_text)=0 THEN
    IF (length(definition)-length(replace(definition,old_text,'')))/length(old_text)<>1
      OR (length(definition)-length(replace(definition,end_old,'')))/length(end_old)<>1 THEN
      RAISE EXCEPTION 'recovered execution judge anchor differs';
    END IF;
    EXECUTE replace(replace(definition,old_text,new_text),end_old,end_new);
  ELSIF strpos(definition,end_new)=0 THEN
    RAISE EXCEPTION 'recovered execution judge patch incomplete';
  END IF;
END;
$recovered_source$;

-- The existing stage2_judged baseline assembly has the same recovered source
-- authority. Accepted later judges retain all canonical membership checks in
-- this RPC and must use the same frozen scorer and input/cache identity.
DO $accepted_recovered_judge$
DECLARE definition TEXT; identity JSONB;
  old_text TEXT := $old$AND (v_run.per_icp_score IS NOT NULL OR public.lab_arena__baseline_recovery_run_v1(judged)))) THEN$old$;
  new_text TEXT := $new$AND (v_run.per_icp_score IS NOT NULL OR public.lab_arena__baseline_recovery_run_v1(judged)
            OR (public.lab_arena_judge_continuation_enabled_v1(v_round.round_id)
              AND (public.lab_arena__baseline_recovery_run_v1(v_run)
                OR EXISTS (SELECT 1 FROM pg_catalog.jsonb_array_elements(
                  COALESCE(v_round.stage1_scoring_plan_doc->'work_items','[]'::JSONB)) item
                  WHERE item->>'scored_run_id'=v_run.run_id
                    AND item->>'submission_id'=v_run.submission_id
                    AND item->>'icp_position'=v_run.icp_position::TEXT
                    AND item->>'output_ref'=v_run.output_ref))
              AND judged.judgment_scope_doc->>'round_id'=v_round.round_id
              AND judged.judgment_scope_doc->>'scorer_image_digest'=
                v_round.configuration_doc->>'scorer_image_digest'
              AND judged.judgment_scope_doc->>'scorer_image_reference'=
                v_round.configuration_doc->>'scorer_image_reference'
              AND judged.judgment_scope_doc->>'scoring_input_hash'=judged.judgment_input_hash
              AND judged.judgment_scope_doc->>'cache_key'=judged.judgment_cache_key)))) THEN$new$;
BEGIN
  SELECT pg_get_functiondef(proc.oid),jsonb_build_array(
    pg_get_userbyid(proc.proowner),proc.proacl::TEXT,proc.prosecdef,
    proc.provolatile,proc.proconfig) INTO definition,identity
    FROM pg_proc proc WHERE proc.oid=to_regprocedure(
      'public.lab_arena_record_run_scores(text,smallint,jsonb)');
  IF definition IS NULL OR identity IS DISTINCT FROM jsonb_build_array(
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'accepted recovered judge authority differs';
  END IF;
  IF strpos(definition,new_text)=0 THEN
    IF (length(definition)-length(replace(definition,old_text,'')))/length(old_text)<>1 THEN
      RAISE EXCEPTION 'accepted recovered judge anchor differs';
    END IF;
    EXECUTE replace(definition,old_text,new_text);
  END IF;
END;
$accepted_recovered_judge$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst,'reload schema';
COMMIT;
