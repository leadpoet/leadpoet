-- 181-lab-arena-daily-icp-source.sql
-- Give the Arena one read-only function for the exact daily inputs frozen by
-- the public-baseline run. The service role cannot read either source table.

BEGIN;

DO $roles$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'lab_arena_owner'
  ) OR NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'lab_arena_service'
  ) THEN
    RAISE EXCEPTION 'scripts/179-lab-arena-v1.sql must be applied first';
  END IF;
END
$roles$;

REVOKE SELECT ON TABLE public.qualification_private_icp_sets FROM lab_arena_owner;

DROP POLICY IF EXISTS lab_arena_owner_current_daily_icp_set
  ON public.qualification_private_icp_sets;

GRANT SELECT ON TABLE public.research_lab_daily_rebenchmarks TO lab_arena_owner;
DROP POLICY IF EXISTS lab_arena_owner_current_daily_baseline
  ON public.research_lab_daily_rebenchmarks;
CREATE POLICY lab_arena_owner_current_daily_baseline
  ON public.research_lab_daily_rebenchmarks
  FOR SELECT
  TO lab_arena_owner
  USING (
    benchmark_date = pg_catalog.timezone(
      'UTC', pg_catalog.statement_timestamp()
    )::DATE
    AND baseline_id = 'leadpoet/pydantic-harness'
    AND status IN ('running', 'completed')
  );

-- Migration 179 froze the original 10+20 position arrays in this function.
-- Replace stage 2 with the remaining ten positions in the daily set.
CREATE OR REPLACE FUNCTION public.lab_arena_open_stage(
  p_round_id TEXT,
  p_stage SMALLINT,
  p_participants JSONB,
  p_icp_positions INTEGER[]
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_open_stage$
DECLARE
  v_round public.lab_arena_rounds;
  v_expected TEXT;
  v_next TEXT;
  v_generation BIGINT;
  v_participant JSONB;
  v_submission public.lab_arena_submissions;
  v_index INTEGER;
  v_position INTEGER;
  v_assignment TEXT;
  v_created INTEGER := 0;
  v_participant_count INTEGER;
  v_distinct_count INTEGER;
  v_expected_count INTEGER;
BEGIN
  IF p_stage NOT IN (1, 2)
     OR pg_catalog.jsonb_typeof(p_participants) IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(p_participants) < 1
     OR p_icp_positions IS NULL
     OR pg_catalog.cardinality(p_icp_positions) < 1 THEN
    RAISE EXCEPTION 'lab_arena_stage_input_invalid' USING ERRCODE = '22023';
  END IF;
  v_expected := CASE p_stage WHEN 1 THEN 'committed' ELSE 'stage1_scored' END;
  v_next := 'stage' || p_stage::TEXT;
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> v_expected THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale',
      'round_status', v_round.status,
      'stage_generation', v_round.stage_generation
    );
  END IF;
  IF (p_stage = 1 AND p_icp_positions IS DISTINCT FROM ARRAY[0,1,2,3,4,5,6,7,8,9])
     OR (p_stage = 2 AND p_icp_positions IS DISTINCT FROM ARRAY[10,11,12,13,14,15,16,17,18,19]) THEN
    RAISE EXCEPTION 'lab_arena_stage_position_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT COUNT(*), COUNT(DISTINCT participant ->> 'submission_id')
  INTO v_participant_count, v_distinct_count
  FROM pg_catalog.jsonb_array_elements(p_participants) AS participant;
  IF v_participant_count <> v_distinct_count THEN
    RAISE EXCEPTION 'lab_arena_stage_participants_duplicate' USING ERRCODE = '22023';
  END IF;
  IF p_stage = 1 THEN
    v_expected_count := pg_catalog.jsonb_array_length(v_round.participants);
  ELSE
    SELECT pg_catalog.jsonb_array_length(v_round.finalists) + COUNT(*)
    INTO v_expected_count
    FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
    WHERE COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE);
  END IF;
  IF v_participant_count <> v_expected_count THEN
    RAISE EXCEPTION 'lab_arena_stage_participants_invalid' USING ERRCODE = '22023';
  END IF;
  v_generation := v_round.stage_generation + 1;
  FOR v_participant IN
    SELECT value FROM pg_catalog.jsonb_array_elements(p_participants)
  LOOP
    SELECT * INTO v_submission
    FROM public.lab_arena_submissions
    WHERE submission_id = (v_participant ->> 'submission_id')
      AND round_id = p_round_id
      AND status = 'frozen';
    IF NOT FOUND OR v_submission.miner_hotkey <> (v_participant ->> 'miner_hotkey') THEN
      RAISE EXCEPTION 'lab_arena_participant_not_frozen' USING ERRCODE = '23503';
    END IF;
    IF NOT EXISTS (
      SELECT 1
      FROM pg_catalog.jsonb_array_elements(v_round.participants) AS original
      WHERE original ->> 'submission_id' = v_submission.submission_id
        AND original ->> 'miner_hotkey' = v_submission.miner_hotkey
        AND (
          p_stage = 1
          OR COALESCE((original ->> 'is_king')::BOOLEAN, FALSE)
          OR v_round.finalists ? v_submission.submission_id
        )
    ) THEN
      RAISE EXCEPTION 'lab_arena_stage_participants_invalid' USING ERRCODE = '22023';
    END IF;
    FOR v_index IN 1 .. pg_catalog.array_length(p_icp_positions, 1) LOOP
      v_position := p_icp_positions[v_index];
      v_assignment := p_round_id || ':' || v_submission.submission_id || ':'
        || p_stage::TEXT || ':' || v_position::TEXT;
      INSERT INTO public.lab_arena_runs (
        run_id, assignment_id, round_id, submission_id, miner_hotkey, stage,
        icp_position, attempt, status, stage_generation
      ) VALUES (
        v_assignment || ':1', v_assignment, p_round_id,
        v_submission.submission_id, v_submission.miner_hotkey, p_stage,
        v_position, 1, 'pending', v_generation
      );
      v_created := v_created + 1;
    END LOOP;
  END LOOP;
  UPDATE public.lab_arena_rounds
  SET status = v_next,
      status_generation = status_generation + 1,
      stage_generation = v_generation
  WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'ok',
    'round_status', v_next,
    'stage_generation', v_generation,
    'assignments', v_created
  );
END;
$lab_arena_open_stage$;
ALTER FUNCTION public.lab_arena_open_stage(TEXT, SMALLINT, JSONB, INTEGER[])
  OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_current_daily_icp_set(
  p_set_id BIGINT
)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog
AS $function$
DECLARE
  v_today BIGINT;
  v_result JSONB;
BEGIN
  v_today := pg_catalog.to_char(
    pg_catalog.timezone('UTC', pg_catalog.statement_timestamp()),
    'YYYYMMDD'
  )::BIGINT;
  IF p_set_id IS NULL OR p_set_id <> v_today THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'unavailable',
      'set_id', p_set_id
    );
  END IF;

  SELECT pg_catalog.jsonb_build_object(
           'status', 'ready',
           'set_id', (source.benchmark_input_doc ->> 'set_id')::BIGINT,
           'icps', source.benchmark_input_doc -> 'icps'
         )
    INTO v_result
    FROM public.research_lab_daily_rebenchmarks AS source
   WHERE source.benchmark_date = pg_catalog.timezone(
           'UTC', pg_catalog.statement_timestamp()
         )::DATE
     AND source.baseline_id = 'leadpoet/pydantic-harness'
     AND source.status IN ('running', 'completed')
     AND (source.benchmark_input_doc ->> 'set_id')::BIGINT = p_set_id
     AND pg_catalog.jsonb_typeof(source.benchmark_input_doc -> 'icps') = 'array'
     AND pg_catalog.jsonb_array_length(source.benchmark_input_doc -> 'icps') = 20
   LIMIT 1;

  RETURN COALESCE(
    v_result,
    pg_catalog.jsonb_build_object(
      'status', 'unavailable',
      'set_id', p_set_id
    )
  );
END
$function$;

ALTER FUNCTION public.lab_arena_current_daily_icp_set(BIGINT)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_current_daily_icp_set(BIGINT)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_current_daily_icp_set(BIGINT)
  TO lab_arena_service;

NOTIFY pgrst, 'reload schema';

COMMIT;
