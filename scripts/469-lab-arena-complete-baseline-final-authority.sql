-- A failed execution's deterministic zero is not a measured baseline.
-- Guard only new final authority. Preserve all existing publications,
-- promotions, signed reward bases, accepted scores, and provider accounting.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena__baseline_final_ready_v1(p_round_id TEXT)
RETURNS BOOLEAN
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $baseline_final_ready$
DECLARE
  v_round public.lab_arena_rounds;
  v_baseline JSONB;
  v_baseline_count BIGINT;
  v_count INTEGER;
  v_first INTEGER;
  v_position INTEGER;
  v_stage INTEGER;
  v_execution public.lab_arena_runs;
  v_executions BIGINT;
  v_cost JSONB;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id;
  IF NOT FOUND THEN RETURN FALSE; END IF;
  SELECT count(*), jsonb_agg(participant)->0 INTO v_baseline_count, v_baseline
  FROM jsonb_array_elements(v_round.participants) participant
  WHERE participant->'is_king' = 'true'::JSONB;
  IF v_baseline_count <> 1 OR COALESCE(v_baseline->>'submission_id','') = ''
     OR COALESCE(v_baseline->>'miner_hotkey','') = '' THEN RETURN FALSE; END IF;
  v_first := (v_round.configuration_doc->>'stage_1_icp_count')::INTEGER;
  v_count := v_first + (v_round.configuration_doc->>'stage_2_icp_count')::INTEGER;
  -- Match migration 353's frozen stage-position bounds and the shared
  -- benchmark_icp_count contract, including a nonempty second stage.
  IF v_count IS NULL OR v_count NOT BETWEEN 2 AND 100 OR v_first < 1
     OR v_count <= v_first THEN RETURN FALSE; END IF;
  FOR v_position IN 0..v_count - 1 LOOP
    v_stage := CASE WHEN v_round.configuration_doc->>'execution_sequence_policy'
                             = 'baseline_scored_first_v1' OR v_position < v_first
                    THEN 1 ELSE 2 END;
    SELECT count(*) INTO v_executions FROM public.lab_arena_runs execution
    WHERE execution.round_id = p_round_id
      AND execution.submission_id = v_baseline->>'submission_id'
      AND execution.miner_hotkey = v_baseline->>'miner_hotkey'
      AND execution.kind = 'execute' AND execution.stage = v_stage
      AND execution.icp_position = v_position AND execution.status = 'accepted'
      AND execution.terminal_cause = 'accepted' AND execution.per_icp_score IS NOT NULL
      AND execution.output_ref IS NOT NULL AND execution.result_doc IS NOT NULL;
    IF v_executions <> 1 THEN RETURN FALSE; END IF;
    SELECT * INTO STRICT v_execution FROM public.lab_arena_runs execution
    WHERE execution.round_id = p_round_id
      AND execution.submission_id = v_baseline->>'submission_id'
      AND execution.miner_hotkey = v_baseline->>'miner_hotkey'
      AND execution.kind = 'execute' AND execution.stage = v_stage
      AND execution.icp_position = v_position AND execution.status = 'accepted'
      AND execution.terminal_cause = 'accepted' AND execution.per_icp_score IS NOT NULL
      AND execution.output_ref IS NOT NULL AND execution.result_doc IS NOT NULL;
    -- Accepted cache aliases retain their own exact execution binding. They
    -- need no new provider call or original-judge restriction at this gate.
    IF NOT EXISTS (SELECT 1 FROM public.lab_arena_runs judge
      WHERE judge.round_id = p_round_id AND judge.submission_id = v_execution.submission_id
        AND judge.miner_hotkey = v_execution.miner_hotkey
        AND judge.stage = v_execution.stage AND judge.icp_position = v_position
        AND judge.kind = 'score' AND judge.scored_run_id = v_execution.run_id
        AND judge.status = 'accepted' AND judge.terminal_cause = 'accepted'
        AND judge.output_ref IS NOT NULL AND judge.result_doc IS NOT NULL
    ) THEN RETURN FALSE; END IF;
  END LOOP;
  v_cost := public.lab_arena__successful_call_cost_state(v_baseline->>'submission_id', 'execute', NULL);
  IF COALESCE((v_cost->>'inflight_calls')::BIGINT,-1) <> 0 THEN RETURN FALSE; END IF;
  IF v_round.configuration_doc->>'sourcing_cost_eligibility_policy'
       IN ('successful_calls_v1','successful_calls_per_icp_v1') THEN
    IF COALESCE((v_cost->>'success_unresolved_calls')::BIGINT,-1) <> 0 THEN RETURN FALSE; END IF;
  ELSE
    IF COALESCE((v_cost->>'uncertain_calls')::BIGINT,-1) <> 0 THEN RETURN FALSE; END IF;
  END IF;
  v_cost := public.lab_arena__successful_call_cost_state(v_baseline->>'submission_id', 'score', NULL);
  RETURN COALESCE((v_cost->>'inflight_calls')::BIGINT,-1) = 0;
END;
$baseline_final_ready$;
ALTER FUNCTION public.lab_arena__baseline_final_ready_v1(TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__baseline_final_ready_v1(TEXT)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_complete_baseline_final_guard_v1()
RETURNS TRIGGER
LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, public
AS $complete_baseline_final_guard$
BEGIN
  IF (NEW.status = 'published' AND OLD.status <> 'published')
     OR (NEW.promotion_doc IS NOT NULL AND OLD.promotion_doc IS NULL)
     OR (NEW.baseline_promoted_at IS NOT NULL AND OLD.baseline_promoted_at IS NULL)
     OR (NEW.reward_activated_at IS NOT NULL AND OLD.reward_activated_at IS NULL) THEN
    IF NOT public.lab_arena__baseline_final_ready_v1(OLD.round_id) THEN
      RAISE EXCEPTION 'lab_arena_final_baseline_incomplete' USING ERRCODE = '55000';
    END IF;
  END IF;
  RETURN NEW;
END;
$complete_baseline_final_guard$;
ALTER FUNCTION public.lab_arena_complete_baseline_final_guard_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_complete_baseline_final_guard_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_complete_baseline_final_guard ON public.lab_arena_rounds;
CREATE TRIGGER lab_arena_complete_baseline_final_guard BEFORE UPDATE ON public.lab_arena_rounds
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_complete_baseline_final_guard_v1();

-- A historical incomplete publication has no new authority and must not
-- obstruct a later complete publication in the same evaluation day.
DO $baseline_final_queue_continuity$
DECLARE
  v_signature TEXT;
  v_definition TEXT;
  v_anchor TEXT;
  v_replacement TEXT;
  v_owner NAME;
  v_config TEXT[];
BEGIN
  FOR v_signature, v_anchor, v_replacement IN SELECT * FROM (VALUES
    ('public.lab_arena_activate_reward(text,jsonb,jsonb)',
     'AND older.reward_activated_at IS NULL',
     'AND older.reward_activated_at IS NULL' || E'\n      AND public.lab_arena__baseline_final_ready_v1(older.round_id)'),
    ('public.lab_arena_prepare_promotion(text,jsonb)',
     'AND older.baseline_promoted_at IS NULL',
     'AND older.baseline_promoted_at IS NULL' || E'\n      AND public.lab_arena__baseline_final_ready_v1(older.round_id)'),
    ('public.lab_arena_reward_requires_promotion_v1()',
     'AND pending.baseline_promoted_at IS NULL',
     'AND pending.baseline_promoted_at IS NULL' || E'\n         AND public.lab_arena__baseline_final_ready_v1(pending.round_id)')
  ) AS fragments(signature, anchor, replacement) LOOP
    SELECT pg_get_userbyid(proowner), proconfig INTO v_owner,v_config
    FROM pg_proc WHERE oid = v_signature::REGPROCEDURE;
    IF v_owner IS DISTINCT FROM 'lab_arena_owner'
       OR NOT COALESCE('search_path=pg_catalog, public'=ANY(v_config),FALSE) THEN
      RAISE EXCEPTION 'Arena baseline final queue security shape differs: %',v_signature;
    END IF;
    v_definition := pg_get_functiondef(v_signature::REGPROCEDURE);
    IF strpos(v_definition, v_replacement) > 0 THEN CONTINUE; END IF;
    IF (length(v_definition)-length(replace(v_definition,v_anchor,''))) <> length(v_anchor) THEN
      RAISE EXCEPTION 'Arena baseline final queue anchor differs: %',v_signature;
    END IF;
    EXECUTE replace(v_definition,v_anchor,v_replacement);
  END LOOP;
END;
$baseline_final_queue_continuity$;
-- This existing trigger now calls the owner-only read helper. Pin its search
-- path and run only its existing scoped pending-promotion predicate as owner.
ALTER FUNCTION public.lab_arena_reward_requires_promotion_v1() SECURITY DEFINER;
ALTER FUNCTION public.lab_arena_reward_requires_promotion_v1() SET search_path = pg_catalog, public;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;
