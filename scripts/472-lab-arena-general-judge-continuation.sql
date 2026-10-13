-- Continue the existing score assignment chain until an accepted judgment exists.
-- Execution attempts, frozen scoring inputs, accepted rows, and published rounds
-- retain their existing authority. Pending retries use created_at for backoff.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

ALTER TABLE public.lab_arena_runs DROP CONSTRAINT lab_arena_runs_attempt_check;
ALTER TABLE public.lab_arena_runs ALTER COLUMN attempt TYPE INTEGER;
ALTER TABLE public.lab_arena_runs ADD CONSTRAINT lab_arena_runs_attempt_check CHECK (attempt >= 1);

CREATE OR REPLACE FUNCTION public.lab_arena_judge_continuation_enabled_v1(p_round_id TEXT)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $enabled$
  -- Versioned rollout boundary: rounds already opened for the next evaluation
  -- day enroll, while prior frozen rounds retain their exact state.
  SELECT COALESCE((SELECT round_row.evaluation_date::DATE >= DATE '2026-10-14'
      OR (round_row.configuration_doc->>'mode'='shadow'
        AND round_row.rewards_enabled IS FALSE
        AND round_row.created_at >= TIMESTAMPTZ '2026-10-13T16:30:00Z')
    FROM public.lab_arena_rounds round_row
    WHERE round_row.round_id=p_round_id),FALSE);
$enabled$;
ALTER FUNCTION public.lab_arena_judge_continuation_enabled_v1(TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_judge_continuation_enabled_v1(TEXT)
  FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- The older bounded baseline lane must obey the same rollout fence. This
-- prevents any prior-day baseline lease or queued recovery from reactivating.
DO $baseline_fence$
DECLARE p RECORD; definition TEXT; owner_name TEXT; secure BOOLEAN;
BEGIN
  FOR p IN SELECT * FROM (VALUES
    ('public.lab_arena__baseline_recovery_phase_v1(public.lab_arena_rounds,public.lab_arena_runs)',
      $old$public.lab_arena__baseline_recovery_run_v1(p_run) AND$old$,
      $new$public.lab_arena_judge_continuation_enabled_v1(p_run.round_id)
        AND public.lab_arena__baseline_recovery_run_v1(p_run) AND$new$),
    ('public.lab_arena_recover_baseline_v1(text,jsonb)',
      $old$  IF v_round.status NOT IN ('stage2','stage2_closed','stage2_scoring')$old$,
      $new$  IF NOT public.lab_arena_judge_continuation_enabled_v1(p_round_id)
    OR v_round.status NOT IN ('stage2','stage2_closed','stage2_scoring')$new$)
  ) AS patches(signature,old_text,new_text) LOOP
    SELECT pg_get_functiondef(proc.oid),pg_get_userbyid(proc.proowner),proc.prosecdef
      INTO definition,owner_name,secure FROM pg_proc proc
      WHERE proc.oid=to_regprocedure(p.signature);
    IF definition IS NULL OR owner_name<>'lab_arena_owner' OR secure IS DISTINCT FROM TRUE THEN
      RAISE EXCEPTION 'baseline recovery fence authority differs: %',p.signature;
    END IF;
    IF strpos(definition,p.new_text)>0 THEN CONTINUE; END IF;
    IF (length(definition)-length(replace(definition,p.old_text,'')))/length(p.old_text)<>1 THEN
      RAISE EXCEPTION 'baseline recovery fence anchor differs: %',p.signature;
    END IF;
    EXECUTE replace(definition,p.old_text,p.new_text);
  END LOOP;
END;
$baseline_fence$;

CREATE OR REPLACE FUNCTION public.lab_arena__judge_calls_clear_v1(p_assignment_id TEXT)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $clear$
  SELECT NOT EXISTS (
    SELECT 1 FROM (
      SELECT DISTINCT ON (l.run_id,l.call_identity)
        l.entry_kind,l.entry_doc,l.terminal_response
      FROM public.lab_arena_ledger l
      JOIN public.lab_arena_runs r ON r.run_id=l.run_id
      WHERE r.assignment_id=p_assignment_id AND l.call_identity IS NOT NULL
      ORDER BY l.run_id,l.call_identity,l.entry_id DESC
    ) heads WHERE heads.entry_kind IN ('reservation','dispatch')
      OR (heads.entry_kind='uncertain' AND
          heads.entry_doc#>'{call,call_succeeded}' IS DISTINCT FROM 'false'::JSONB)
      OR (heads.entry_kind='settlement' AND
          jsonb_typeof(heads.terminal_response->'call_succeeded') IS DISTINCT FROM 'boolean')
  );
$clear$;
ALTER FUNCTION public.lab_arena__judge_calls_clear_v1(TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__judge_calls_clear_v1(TEXT)
  FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena__judge_continuation_v1(
  p_round public.lab_arena_rounds,p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $continuation$
DECLARE v_source public.lab_arena_runs; v_parent public.lab_arena_runs;
BEGIN
  IF p_run.kind<>'score' OR p_run.round_id IS DISTINCT FROM p_round.round_id
    OR NOT public.lab_arena_judge_continuation_enabled_v1(p_round.round_id)
    OR p_run.stage NOT IN (1,2)
    OR NOT ((p_run.stage=1 AND p_round.status IN
      ('stage1_scoring','stage1_judged','stage1_scored','stage2','stage2_closed',
       'stage2_scoring','stage2_judged','scored'))
      OR (p_run.stage=2 AND p_round.status IN
      ('stage2_scoring','stage2_judged','scored')))
    OR p_run.assignment_id IS DISTINCT FROM
      p_run.round_id||':'||p_run.submission_id||':'||p_run.stage::TEXT||':'||
      p_run.icp_position::TEXT||':score'
    OR p_run.run_id IS DISTINCT FROM p_run.assignment_id||':'||p_run.attempt::TEXT
    OR p_run.judgment_scope_doc->>'scorer_image_digest' IS DISTINCT FROM
      p_round.configuration_doc->>'scorer_image_digest'
    OR p_run.judgment_scope_doc->>'scorer_image_reference' IS DISTINCT FROM
      p_round.configuration_doc->>'scorer_image_reference'
    OR p_run.judgment_scope_doc->>'round_id' IS DISTINCT FROM p_round.round_id
    OR p_run.judgment_scope_doc->>'scoring_input_hash' IS DISTINCT FROM p_run.judgment_input_hash
    OR p_run.judgment_scope_doc->>'cache_key' IS DISTINCT FROM p_run.judgment_cache_key
    OR p_run.scored_run_id IS NULL
    OR NOT EXISTS (SELECT 1 FROM pg_catalog.jsonb_array_elements(
      COALESCE(CASE p_run.stage WHEN 1 THEN p_round.stage1_scoring_plan_doc
        ELSE p_round.stage2_scoring_plan_doc END->'work_items','[]'::JSONB)) item
      WHERE item->>'scored_run_id'=p_run.scored_run_id
        AND item->>'submission_id'=p_run.submission_id
        AND item->>'icp_position'=p_run.icp_position::TEXT
        AND item->>'output_ref'=(SELECT source.output_ref
          FROM public.lab_arena_runs source WHERE source.run_id=p_run.scored_run_id))
    THEN RETURN FALSE; END IF;
  SELECT * INTO v_source FROM public.lab_arena_runs WHERE run_id=p_run.scored_run_id;
  IF NOT FOUND OR v_source.kind<>'execute' OR v_source.status<>'accepted'
    OR v_source.terminal_cause IS DISTINCT FROM 'accepted' OR v_source.output_ref IS NULL
    OR (v_source.round_id,v_source.submission_id,v_source.miner_hotkey,
        v_source.stage,v_source.icp_position) IS DISTINCT FROM
       (p_run.round_id,p_run.submission_id,p_run.miner_hotkey,
        p_run.stage,p_run.icp_position)
    OR NOT EXISTS (SELECT 1 FROM jsonb_array_elements(p_round.participants) participant
      WHERE participant->>'submission_id'=p_run.submission_id
        AND participant->>'miner_hotkey'=p_run.miner_hotkey)
    OR EXISTS (SELECT 1 FROM public.lab_arena_runs accepted
      WHERE accepted.assignment_id=p_run.assignment_id AND accepted.kind='score'
        AND accepted.status='accepted') THEN RETURN FALSE; END IF;
  IF p_run.attempt=1 THEN RETURN TRUE; END IF;
  SELECT * INTO v_parent FROM public.lab_arena_runs
    WHERE assignment_id=p_run.assignment_id AND attempt=p_run.attempt-1;
  RETURN FOUND AND v_parent.kind='score' AND v_parent.status='failed'
    AND v_parent.terminal_cause IN
      ('judge_error','judge_timeout','lease_expired','stage_closed','credential_error')
    AND v_parent.scored_run_id=p_run.scored_run_id
    AND v_parent.miner_hotkey=p_run.miner_hotkey
    AND v_parent.judgment_cache_key IS NOT DISTINCT FROM p_run.judgment_cache_key
    AND v_parent.judgment_input_hash IS NOT DISTINCT FROM p_run.judgment_input_hash
    AND v_parent.judgment_scope_doc IS NOT DISTINCT FROM p_run.judgment_scope_doc;
END;
$continuation$;
ALTER FUNCTION public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)
  FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- Patch only exact existing RPC source shapes, retaining their grants/owners.
DO $patch$
DECLARE p RECORD; definition TEXT; count_matches INTEGER; owner_name TEXT;
  secure BOOLEAN; current_signature TEXT;
BEGIN
  FOR p IN SELECT * FROM (VALUES
    ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)',
     $old$IF v_round.status NOT IN ('stage1', 'stage1_scoring', 'stage2', 'stage2_scoring') THEN$old$,
     $new$IF v_round.status NOT IN ('stage1', 'stage1_scoring', 'stage1_judged', 'stage1_scored',
      'stage2', 'stage2_closed', 'stage2_scoring', 'stage2_judged', 'scored') THEN$new$),
    ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)',
     $old$AND (v_round.status NOT IN ('stage1', 'stage2', 'stage1_scoring', 'stage2_scoring')
      OR NOT EXISTS ($old$,
     $new$AND (v_round.status NOT IN ('stage1', 'stage2', 'stage1_scoring', 'stage1_judged',
      'stage1_scored', 'stage2_closed', 'stage2_scoring', 'stage2_judged', 'scored')
      OR NOT EXISTS ($new$),
    ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)',
     $old$OR public.lab_arena__baseline_recovery_phase_v1(v_round,runs)$old$,
     $new$OR public.lab_arena__baseline_recovery_phase_v1(v_round,runs)
      OR public.lab_arena__judge_continuation_v1(v_round,runs)$new$),
    ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)',
     $old$runs.kind <> 'score'
      OR pg_catalog.clock_timestamp() <$old$,
     $new$runs.kind <> 'score'
      OR public.lab_arena__judge_continuation_v1(v_round,runs)
      OR pg_catalog.clock_timestamp() <$new$),
    ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)',
     $old$AND stage_generation = v_round.stage_generation;$old$,
     $new$AND (stage_generation = v_round.stage_generation OR
      public.lab_arena__judge_continuation_v1(v_round,lab_arena_runs));$new$),
    ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)',
     $old$AND runs.stage_generation = v_round.stage_generation
    AND (
      runs.kind <> 'execute'$old$,
     $new$AND (runs.stage_generation = v_round.stage_generation OR
      public.lab_arena__judge_continuation_v1(v_round,runs))
    AND (runs.kind <> 'score' OR runs.attempt=1 OR
      pg_catalog.clock_timestamp() >= runs.created_at + pg_catalog.make_interval(
        secs => LEAST(900::NUMERIC,30*power(2::NUMERIC,LEAST(runs.attempt-2,5)))::INTEGER))
    AND (
      runs.kind <> 'execute'$new$),
    ('public.lab_arena__lock_current_lease(text,text)',
     $old$OR v_run.stage_generation <> v_round.stage_generation$old$,
     $new$OR (v_run.stage_generation <> v_round.stage_generation
       AND NOT public.lab_arena__judge_continuation_v1(v_round,v_run))$new$),
    ('public.lab_arena__lock_current_lease(text,text)',
     $old$OR public.lab_arena__baseline_recovery_phase_v1(v_round,v_run)$old$,
     $new$OR public.lab_arena__baseline_recovery_phase_v1(v_round,v_run)
        OR public.lab_arena__judge_continuation_v1(v_round,v_run)$new$),
    ('public.lab_arena_expire_leases(text)',
     $old$IF v_round.status NOT IN ('stage1', 'stage1_scoring', 'stage2', 'stage2_scoring') THEN$old$,
     $new$IF v_round.status NOT IN ('stage1', 'stage1_scoring', 'stage1_judged', 'stage1_scored',
      'stage2', 'stage2_closed', 'stage2_scoring', 'stage2_judged', 'scored') THEN$new$),
    ('public.lab_arena_expire_leases(text)',
     $old$IF (
      v_run.attempt < LEAST(5, 2 +$old$,
     $new$IF (
      (v_run.kind='score' AND v_run.attempt<2147483647
        AND public.lab_arena__judge_continuation_v1(v_round,v_run)
        AND public.lab_arena__judge_calls_clear_v1(v_run.assignment_id)) OR
      v_run.attempt < LEAST(5, 2 +$new$),
    ('public.lab_arena_expire_leases(text)',
     $old$) AND NOT (v_run.attempt>=3 AND public.lab_arena__baseline_recovery_run_v1(v_run))
      AND v_run.stage_generation = v_round.stage_generation$old$,
     $new$) AND NOT (v_run.kind='execute' AND v_run.attempt>=3
        AND public.lab_arena__baseline_recovery_run_v1(v_run))
      AND (v_run.kind<>'score' OR (public.lab_arena__judge_continuation_v1(v_round,v_run)
        AND public.lab_arena__judge_calls_clear_v1(v_run.assignment_id)))
      AND (v_run.stage_generation = v_round.stage_generation OR
        public.lab_arena__judge_continuation_v1(v_round,v_run))$new$),
    ('public.lab_arena_expire_leases(text)',
     $old$  RETURN pg_catalog.jsonb_build_object('status', 'ok', 'expired', v_expired, 'retried', v_retried);$old$,
     $new$  -- Materialize successors for failures that predate this migration or
  -- crossed a scoring close. The round lock and one run per attempt keep this
  -- in the ordinary assignment queue.
  FOR v_run IN SELECT * FROM public.lab_arena_runs candidate
    WHERE candidate.round_id=p_round_id AND candidate.kind='score'
      AND candidate.status='failed' AND candidate.attempt<2147483647
      AND candidate.terminal_cause IN
        ('judge_error','judge_timeout','lease_expired','stage_closed','credential_error')
      AND NOT EXISTS (SELECT 1 FROM public.lab_arena_runs later
        WHERE later.assignment_id=candidate.assignment_id
          AND later.attempt>candidate.attempt)
    ORDER BY candidate.assignment_id FOR UPDATE
  LOOP
    IF public.lab_arena__judge_continuation_v1(v_round,v_run)
      AND public.lab_arena__judge_calls_clear_v1(v_run.assignment_id) THEN
      INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,
        stage,icp_position,attempt,status,lease_generation,stage_generation,kind,scored_run_id,
        previous_runner_hotkey,judgment_cache_key,judgment_input_hash,judgment_scope_doc,
        judgment_group_leader,judgment_group_miner_hotkeys,company_judgment_refs)
      VALUES(v_run.assignment_id||':'||(v_run.attempt+1)::TEXT,v_run.assignment_id,v_run.round_id,
        v_run.submission_id,v_run.miner_hotkey,v_run.stage,v_run.icp_position,v_run.attempt+1,
        'pending',v_run.lease_generation,v_round.stage_generation,'score',v_run.scored_run_id,
        v_run.runner_hotkey,v_run.judgment_cache_key,v_run.judgment_input_hash,
        v_run.judgment_scope_doc,v_run.judgment_group_leader,
        v_run.judgment_group_miner_hotkeys,v_run.company_judgment_refs)
      ON CONFLICT (run_id) DO NOTHING;
      v_retried:=v_retried+1;
    END IF;
  END LOOP;
  RETURN pg_catalog.jsonb_build_object('status', 'ok', 'expired', v_expired, 'retried', v_retried);$new$),
    ('public.lab_arena_complete_attempt(text,text,jsonb,text,text)',
     $old$IF v_status = 'failed'
     AND NOT (v_run.attempt>=3 AND public.lab_arena__baseline_recovery_run_v1(v_run))
     AND p_terminal_cause NOT IN ('budget_exhausted', 'credential_error')
     AND ($old$,
     $new$IF v_status = 'failed'
     AND ((v_run.kind='score' AND v_run.attempt<2147483647
       AND p_terminal_cause IN ('judge_error','judge_timeout','credential_error')
       AND public.lab_arena__judge_calls_clear_v1(v_run.assignment_id))
       OR (NOT (v_run.attempt>=3 AND public.lab_arena__baseline_recovery_run_v1(v_run))
         AND p_terminal_cause NOT IN ('budget_exhausted', 'credential_error')
         AND ($new$),
    ('public.lab_arena_complete_attempt(text,text,jsonb,text,text)',
     $old$     ) THEN
    SELECT * INTO v_round$old$,
     $new$     ))) AND (v_run.kind<>'score' OR
        public.lab_arena__judge_calls_clear_v1(v_run.assignment_id)) THEN
    SELECT * INTO v_round$new$),
    ('public.lab_arena_complete_attempt(text,text,jsonb,text,text)',
     $old$IF v_run.stage_generation = v_round.stage_generation
       AND ($old$,
     $new$IF (v_run.stage_generation = v_round.stage_generation OR
         public.lab_arena__judge_continuation_v1(v_round,v_run))
       AND (v_run.kind<>'score' OR public.lab_arena__judge_continuation_v1(v_round,v_run))
       AND ($new$)
  ) AS patches(signature,old_text,new_text) LOOP
    IF current_signature IS DISTINCT FROM p.signature THEN
      IF current_signature IS NOT NULL THEN EXECUTE definition; END IF;
      SELECT pg_get_functiondef(proc.oid),pg_get_userbyid(proc.proowner),proc.prosecdef
        INTO definition,owner_name,secure FROM pg_proc proc
        WHERE proc.oid=to_regprocedure(p.signature);
      IF definition IS NULL OR owner_name<>'lab_arena_owner' OR secure IS DISTINCT FROM TRUE THEN
        RAISE EXCEPTION 'judge continuation function authority differs: %',p.signature;
      END IF;
      current_signature:=p.signature;
    END IF;
    IF strpos(definition,p.new_text)>0 THEN CONTINUE; END IF;
    count_matches := (length(definition)-length(replace(definition,p.old_text,'')))/length(p.old_text);
    IF count_matches<>1 THEN RAISE EXCEPTION 'judge continuation anchor differs: %, %',
      p.signature,left(p.old_text,80); END IF;
    definition:=replace(definition,p.old_text,p.new_text);
  END LOOP;
  IF current_signature IS NOT NULL THEN EXECUTE definition; END IF;
END;
$patch$;

DO $score_patch$
DECLARE p RECORD; definition TEXT; owner_name TEXT; secure BOOLEAN;
BEGIN
  FOR p IN SELECT * FROM (VALUES
    ('public.lab_arena_record_run_scores(text,smallint,jsonb)',
     $old$  v_late_baseline BOOLEAN;$old$,
     $new$  v_late_baseline BOOLEAN;
  v_late_judge BOOLEAN;$new$),
    ('public.lab_arena_record_run_scores(text,smallint,jsonb)',
     $old$  IF NOT COALESCE(v_late_baseline,FALSE) AND v_round.status NOT IN ($old$,
     $new$  v_late_judge := v_round.status='scored' AND p_stage IN (1,2)
    AND public.lab_arena_judge_continuation_enabled_v1(p_round_id);
  IF NOT COALESCE(v_late_baseline,FALSE) AND NOT v_late_judge
    AND v_round.status NOT IN ($new$),
    ('public.lab_arena_record_run_scores(text,smallint,jsonb)',
     $old$    IF v_late_baseline AND (v_run.status<>'accepted'$old$,
     $new$    IF v_late_judge AND (v_run.status<>'accepted'
      OR v_run.terminal_cause<>'accepted' OR v_run.output_ref IS NULL
      OR v_run.assignment_id IS DISTINCT FROM
        v_run.round_id||':'||v_run.submission_id||':'||v_run.stage::TEXT||':'||v_run.icp_position::TEXT
      OR NOT EXISTS (SELECT 1 FROM public.lab_arena_runs judged
        WHERE judged.assignment_id=v_run.assignment_id||':score'
          AND judged.scored_run_id=v_run.run_id AND judged.kind='score'
          AND judged.status='accepted' AND judged.terminal_cause='accepted'
          AND judged.output_ref IS NOT NULL
          AND (judged.round_id,judged.submission_id,judged.miner_hotkey,
               judged.stage,judged.icp_position) IS NOT DISTINCT FROM
              (v_run.round_id,v_run.submission_id,v_run.miner_hotkey,
               v_run.stage,v_run.icp_position)
          AND judged.judgment_scope_doc->>'scorer_image_digest'=
            v_round.configuration_doc->>'scorer_image_digest'
          AND judged.judgment_scope_doc->>'scorer_image_reference'=
            v_round.configuration_doc->>'scorer_image_reference')) THEN
      RAISE EXCEPTION 'lab_arena_late_judge_score_invalid' USING ERRCODE='42501';
    END IF;
    IF v_late_baseline AND (v_run.status<>'accepted'$new$)
  ) AS patches(signature,old_text,new_text) LOOP
    SELECT pg_get_functiondef(proc.oid),pg_get_userbyid(proc.proowner),proc.prosecdef
      INTO definition,owner_name,secure FROM pg_proc proc
      WHERE proc.oid=to_regprocedure(p.signature);
    IF definition IS NULL OR owner_name<>'lab_arena_owner' THEN
      RAISE EXCEPTION 'judge score function authority differs: %',p.signature;
    END IF;
    IF p.signature LIKE '%record_run_scores%' AND secure IS DISTINCT FROM TRUE THEN
      RAISE EXCEPTION 'judge score function security differs';
    END IF;
    IF strpos(definition,p.new_text)>0 THEN CONTINUE; END IF;
    IF (length(definition)-length(replace(definition,p.old_text,'')))/length(p.old_text)<>1 THEN
      RAISE EXCEPTION 'judge score anchor differs: %, %',p.signature,left(p.old_text,80);
    END IF;
    EXECUTE replace(definition,p.old_text,p.new_text);
  END LOOP;
END;
$score_patch$;

CREATE OR REPLACE FUNCTION public.lab_arena__all_judgments_ready_v1(p_round_id TEXT)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $ready$
  SELECT NOT EXISTS (SELECT 1 FROM public.lab_arena_runs source
    WHERE source.round_id=p_round_id AND source.kind='execute'
      AND source.stage IN (1,2) AND source.status='accepted'
      AND (source.terminal_cause<>'accepted' OR source.output_ref IS NULL
        OR source.per_icp_score IS NULL
        OR NOT EXISTS (SELECT 1 FROM public.lab_arena_runs judge
          WHERE judge.assignment_id=source.assignment_id||':score'
            AND judge.scored_run_id=source.run_id AND judge.kind='score'
            AND judge.status='accepted' AND judge.terminal_cause='accepted'
            AND judge.output_ref IS NOT NULL
            AND (judge.round_id,judge.submission_id,judge.miner_hotkey,
                 judge.stage,judge.icp_position) IS NOT DISTINCT FROM
                (source.round_id,source.submission_id,source.miner_hotkey,
                 source.stage,source.icp_position)
            AND EXISTS (SELECT 1 FROM public.lab_arena_rounds round_row
              WHERE round_row.round_id=p_round_id
                AND judge.judgment_scope_doc->>'scorer_image_digest'=
                    round_row.configuration_doc->>'scorer_image_digest'
                AND judge.judgment_scope_doc->>'scorer_image_reference'=
                    round_row.configuration_doc->>'scorer_image_reference'))));
$ready$;
ALTER FUNCTION public.lab_arena__all_judgments_ready_v1(TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__all_judgments_ready_v1(TEXT)
  FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

DO $publication_patch$
DECLARE definition TEXT; owner_name TEXT; secure BOOLEAN;
  old_text TEXT:=$old$IF NOT public.lab_arena__baseline_final_ready_v1(OLD.round_id) THEN$old$;
  new_text TEXT:=$new$IF NOT public.lab_arena__baseline_final_ready_v1(OLD.round_id)
      OR (public.lab_arena_judge_continuation_enabled_v1(OLD.round_id)
        AND NOT public.lab_arena__all_judgments_ready_v1(OLD.round_id)) THEN$new$;
BEGIN
  SELECT pg_get_functiondef(proc.oid),pg_get_userbyid(proc.proowner),proc.prosecdef
    INTO definition,owner_name,secure FROM pg_proc proc
    WHERE proc.oid=to_regprocedure('public.lab_arena_complete_baseline_final_guard_v1()');
  IF definition IS NULL OR owner_name<>'lab_arena_owner' OR secure IS DISTINCT FROM TRUE THEN
    RAISE EXCEPTION 'judge final authority differs'; END IF;
  IF strpos(definition,new_text)=0 THEN
    IF (length(definition)-length(replace(definition,old_text,'')))/length(old_text)<>1 THEN
      RAISE EXCEPTION 'judge final authority anchor differs'; END IF;
    EXECUTE replace(definition,old_text,new_text);
  END IF;
END;
$publication_patch$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst,'reload schema';
COMMIT;
