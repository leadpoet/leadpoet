-- Make scoring close select the effective outcome for each planned execution.
-- A recovery may append a new score assignment namespace for the same
-- scored_run_id. Accepted evidence wins over an older failure, matching the
-- service selector, while all historical rows remain append-only.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';
LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
CREATE OR REPLACE FUNCTION public.lab_arena_close_scoring(p_round_id TEXT, p_stage SMALLINT)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_close_scoring$
DECLARE
  v_round public.lab_arena_rounds;
  v_run public.lab_arena_runs;
  v_generation BIGINT;
  v_incomplete INTEGER;
  v_next TEXT;
BEGIN
  IF p_stage NOT IN (1, 2) THEN
    RAISE EXCEPTION 'lab_arena_stage_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> ('stage' || p_stage::TEXT || '_scoring') THEN
    IF (p_stage = 1 AND v_round.status IN ('stage1_judged', 'stage1_scored',
                                           'stage2', 'stage2_closed', 'stage2_scoring', 'stage2_judged', 'scored', 'published'))
       OR (p_stage = 2 AND v_round.status IN ('stage2_judged', 'scored', 'published')) THEN
      RETURN pg_catalog.jsonb_build_object('status', 'existing', 'round_status', v_round.status,
        'stage_generation', v_round.stage_generation);
    END IF;
    RETURN pg_catalog.jsonb_build_object('status', 'stale', 'round_status', v_round.status,
      'stage_generation', v_round.stage_generation);
  END IF;
  v_generation := v_round.stage_generation + 1;
  FOR v_run IN
    SELECT * FROM public.lab_arena_runs
    WHERE round_id = p_round_id AND stage = p_stage AND kind = 'score' AND status IN ('leased', 'pending', 'submitted')
    ORDER BY assignment_id, attempt
    FOR UPDATE
  LOOP
    IF v_run.status = 'leased' THEN
      PERFORM public.lab_arena__terminate_open_calls(v_run.run_id, 'stage_closed');
    END IF;
    UPDATE public.lab_arena_runs
    SET status = 'failed', terminal_cause = 'stage_closed',
        terminal_doc = pg_catalog.jsonb_build_object('closed_at', pg_catalog.clock_timestamp(),
          'previous_status', v_run.status)
    WHERE run_id = v_run.run_id;
  END LOOP;
  SELECT COUNT(*) INTO v_incomplete FROM (
    SELECT DISTINCT ON (COALESCE(runs.scored_run_id, runs.assignment_id))
      COALESCE(runs.scored_run_id, runs.assignment_id) AS scored_run_id,
      runs.status, runs.terminal_cause
    FROM public.lab_arena_runs AS runs
    WHERE runs.round_id = p_round_id AND runs.stage = p_stage AND runs.kind = 'score'
    ORDER BY COALESCE(runs.scored_run_id, runs.assignment_id),
      (runs.status = 'accepted') DESC, runs.stage_generation DESC,
      runs.attempt DESC, runs.run_id DESC
  ) AS latest
  WHERE latest.status <> 'accepted';
  -- Closing a review window is not a round-wide qualification decision.
  -- The service verifies accepted artifacts and records only failed ICPs as
  -- zero, preserving every other participant and ICP result.
  -- service still fails closed for missing or malformed accepted evidence
  -- before recording any stage scores.
  v_next := 'stage' || p_stage::TEXT || '_judged';
  UPDATE public.lab_arena_rounds
  SET status = v_next, status_generation = status_generation + 1, stage_generation = v_generation
  WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'closed',
    'round_status', v_next, 'incomplete_assignments', v_incomplete, 'stage_generation', v_generation);
END;
$lab_arena_close_scoring$;
ALTER FUNCTION public.lab_arena_close_scoring(TEXT, SMALLINT) OWNER TO lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
