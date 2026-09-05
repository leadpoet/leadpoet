-- 180-lab-arena-daily-competition.sql
-- Use the active daily qualification set directly in the Arena. The public
-- baseline and all miner bundles therefore run on one frozen set through one
-- execution and scoring path. The Arena service can call the narrow function
-- below, but it cannot read the private qualification table directly.

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

-- Required only for ownership transfers by the hosted migration role.
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

GRANT SELECT ON TABLE public.qualification_private_icp_sets TO lab_arena_owner;
DROP POLICY IF EXISTS lab_arena_owner_current_daily_icp_set
  ON public.qualification_private_icp_sets;
CREATE POLICY lab_arena_owner_current_daily_icp_set
  ON public.qualification_private_icp_sets
  FOR SELECT
  TO lab_arena_owner
  USING (
    is_active
    AND set_id = pg_catalog.to_char(
      pg_catalog.timezone('UTC', pg_catalog.statement_timestamp()),
      'YYYYMMDD'
    )::BIGINT
    AND (
      active_from IS NULL
      OR active_from <= pg_catalog.statement_timestamp()
    )
    AND (
      active_until IS NULL
      OR active_until > pg_catalog.statement_timestamp()
    )
  );

-- The service creates the initial public baseline, and later carries the
-- winner, after the miner submission cutoff. Only rows explicitly marked as
-- the service-owned baseline can use that exception. Miner requests cannot
-- supply this field through the public submission schema.
CREATE OR REPLACE FUNCTION public.lab_arena_register_submission(
  p_round_id TEXT,
  p_submission_id TEXT,
  p_miner_hotkey TEXT,
  p_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_register_submission$
DECLARE
  v_round public.lab_arena_rounds;
  v_existing public.lab_arena_submissions;
  v_is_baseline BOOLEAN;
BEGIN
  IF COALESCE(p_submission_id, '') !~ '^[A-Za-z0-9._:-]{1,64}$'
     OR pg_catalog.jsonb_typeof(p_doc) IS DISTINCT FROM 'object'
     OR pg_catalog.char_length(COALESCE(p_doc ->> 'submitted_reference', '')) NOT BETWEEN 1 AND 512 THEN
    RAISE EXCEPTION 'lab_arena_submission_input_invalid' USING ERRCODE = '22023';
  END IF;
  v_is_baseline := COALESCE((p_doc ->> 'is_king')::BOOLEAN, FALSE);
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR SHARE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> 'open'
     OR (
       NOT v_is_baseline
       AND (
         COALESCE(pg_catalog.clock_timestamp() < (v_round.configuration_doc #>> '{schedule,submission_open}')::TIMESTAMPTZ, TRUE)
         OR COALESCE(pg_catalog.clock_timestamp() >= (v_round.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ, TRUE)
       )
     ) THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'window_closed',
      'round_status', v_round.status
    );
  END IF;
  SELECT * INTO v_existing
  FROM public.lab_arena_submissions
  WHERE submission_id = p_submission_id;
  IF FOUND THEN
    IF v_existing.round_id = p_round_id
       AND v_existing.miner_hotkey = p_miner_hotkey
       AND v_existing.submitted_reference = (p_doc ->> 'submitted_reference') THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'existing',
        'submission_status', v_existing.status
      );
    END IF;
    RAISE EXCEPTION 'lab_arena_submission_conflict' USING ERRCODE = '23505';
  END IF;
  INSERT INTO public.lab_arena_submissions (
    submission_id, round_id, miner_hotkey, status, is_king,
    submitted_reference, consent, submission_doc
  ) VALUES (
    p_submission_id, p_round_id, p_miner_hotkey, 'uploaded', v_is_baseline,
    p_doc ->> 'submitted_reference', p_doc -> 'consent', p_doc
  );
  RETURN pg_catalog.jsonb_build_object(
    'status', 'registered',
    'submission_status', 'uploaded'
  );
END;
$lab_arena_register_submission$;
ALTER FUNCTION public.lab_arena_register_submission(TEXT, TEXT, TEXT, JSONB)
  OWNER TO lab_arena_owner;

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
           'set_id', source.set_id,
           'icps', source.icps
         )
    INTO v_result
    FROM public.qualification_private_icp_sets AS source
   WHERE source.set_id = p_set_id
     AND source.is_active
     AND (
       source.active_from IS NULL
       OR source.active_from <= pg_catalog.statement_timestamp()
     )
     AND (
       source.active_until IS NULL
       OR source.active_until > pg_catalog.statement_timestamp()
     )
     AND pg_catalog.jsonb_typeof(source.icps) = 'array'
     AND pg_catalog.jsonb_array_length(source.icps) = 20
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

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;
