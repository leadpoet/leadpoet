-- 200-lab-arena-next-day-icp-disclosure.sql
-- Persist the bank date used by a round and permit the Arena service to read
-- that exact historical daily bank after the next-day submission cutoff.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $lab_arena_200_requires_197$
BEGIN
  IF pg_catalog.to_regprocedure('public.lab_arena_schema_version_v1()') IS NULL
     OR (pg_catalog.pg_get_functiondef(
          pg_catalog.to_regprocedure('public.lab_arena_schema_version_v1()')
        ) NOT LIKE '%''version'', 197%'
     AND pg_catalog.pg_get_functiondef(
          pg_catalog.to_regprocedure('public.lab_arena_schema_version_v1()')
        ) NOT LIKE '%''version'', 200%') THEN
    RAISE EXCEPTION 'apply 197-lab-arena-reward-chain-scope.sql first';
  END IF;
  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'lab_arena_submissions'
      AND column_name = 'accepted_at'
      AND data_type = 'timestamp with time zone'
  ) THEN
    RAISE EXCEPTION 'apply 199-lab-arena-source-disclosure-time.sql first';
  END IF;
END;
$lab_arena_200_requires_197$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

ALTER TABLE public.lab_arena_rounds
  ADD COLUMN IF NOT EXISTS icp_set_date DATE;

CREATE OR REPLACE FUNCTION public.lab_arena_commit_round_v2(
  p_round_id TEXT,
  p_participants JSONB,
  p_benchmark_ref TEXT,
  p_evaluation_date TEXT,
  p_icp_set_date DATE,
  p_scorer_image_digest TEXT,
  p_scorer_image_reference TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_commit_round_v2$
DECLARE
  v_round public.lab_arena_rounds;
  v_expected_bank_date DATE;
  v_expected_evaluation_date DATE;
  v_pending_admissions INTEGER;
  v_participant_count INTEGER;
  v_frozen_count INTEGER;
  v_invalid_participants INTEGER;
  v_baseline_count INTEGER;
BEGIN
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> 'open' THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale',
      'round_status', v_round.status,
      'status_generation', v_round.status_generation
    );
  END IF;
  SELECT COUNT(*) INTO v_pending_admissions
  FROM public.lab_arena_submissions
  WHERE round_id = p_round_id AND status = 'accepted';
  IF v_pending_admissions <> 0 THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'retry',
      'round_status', v_round.status,
      'remaining_admissions', v_pending_admissions
    );
  END IF;
  BEGIN
    v_expected_bank_date := pg_catalog.timezone(
      'UTC',
      (v_round.configuration_doc #>> '{schedule,submission_open}')::TIMESTAMPTZ
    )::DATE;
    v_expected_evaluation_date := pg_catalog.timezone(
      'UTC',
      (v_round.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ
    )::DATE;
  EXCEPTION WHEN OTHERS THEN
    RAISE EXCEPTION 'lab_arena_round_commit_invalid' USING ERRCODE = '22023';
  END;
  IF pg_catalog.jsonb_typeof(p_participants) IS DISTINCT FROM 'array'
     OR pg_catalog.char_length(COALESCE(p_benchmark_ref, '')) NOT BETWEEN 1 AND 1024
     OR COALESCE(p_evaluation_date, '') !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
     OR p_evaluation_date::DATE <> v_expected_evaluation_date
     OR p_icp_set_date IS NULL
     OR p_icp_set_date <> v_expected_bank_date
     OR v_expected_evaluation_date <> v_expected_bank_date + 1 THEN
    RAISE EXCEPTION 'lab_arena_round_commit_invalid' USING ERRCODE = '22023';
  END IF;
  v_participant_count := pg_catalog.jsonb_array_length(p_participants);

  SELECT COUNT(*)
  INTO v_frozen_count
  FROM public.lab_arena_submissions
  WHERE round_id = p_round_id AND status = 'frozen';
  SELECT COUNT(*) INTO v_invalid_participants
  FROM pg_catalog.jsonb_array_elements(p_participants) AS participant
  WHERE pg_catalog.jsonb_typeof(participant) IS DISTINCT FROM 'object'
     OR NOT EXISTS (
       SELECT 1
       FROM public.lab_arena_submissions AS submission
       WHERE submission.round_id = p_round_id
         AND submission.status = 'frozen'
         AND submission.submission_id = participant ->> 'submission_id'
         AND submission.miner_hotkey = participant ->> 'miner_hotkey'
         AND submission.is_king = COALESCE(
           (participant ->> 'is_king')::BOOLEAN, FALSE
         )
     );
  SELECT COUNT(*) INTO v_baseline_count
  FROM pg_catalog.jsonb_array_elements(p_participants) AS participant
  WHERE COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE);
  IF v_participant_count <> v_frozen_count
     OR v_invalid_participants <> 0
     OR (
       SELECT COUNT(DISTINCT participant ->> 'submission_id')
       FROM pg_catalog.jsonb_array_elements(p_participants) AS participant
     ) <> v_participant_count
     OR v_baseline_count <> 1
     OR v_participant_count >
       COALESCE((v_round.configuration_doc ->> 'max_challengers')::INTEGER, 100) + 1 THEN
    RAISE EXCEPTION 'lab_arena_round_participants_invalid' USING ERRCODE = '22023';
  END IF;
  IF COALESCE(p_scorer_image_digest, '') !~ '^sha256:[0-9a-f]{64}$'
     OR pg_catalog.char_length(COALESCE(p_scorer_image_reference, '')) NOT BETWEEN 1 AND 512
     OR pg_catalog.right(
       p_scorer_image_reference,
       pg_catalog.char_length(p_scorer_image_digest) + 1
     ) <> '@' || p_scorer_image_digest THEN
    RAISE EXCEPTION 'lab_arena_scorer_image_invalid' USING ERRCODE = '22023';
  END IF;
  UPDATE public.lab_arena_rounds
  SET status = 'committed',
      status_generation = status_generation + 1,
      participants = p_participants,
      benchmark_ref = p_benchmark_ref,
      evaluation_date = p_evaluation_date,
      icp_set_date = p_icp_set_date,
      configuration_doc = v_round.configuration_doc || pg_catalog.jsonb_build_object(
        'scorer_image_digest', p_scorer_image_digest,
        'scorer_image_reference', p_scorer_image_reference
      )
  WHERE round_id = p_round_id;
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'ok',
    'round_status', v_round.status,
    'status_generation', v_round.status_generation
  );
END;
$lab_arena_commit_round_v2$;
ALTER FUNCTION public.lab_arena_commit_round_v2(
  TEXT, JSONB, TEXT, TEXT, DATE, TEXT, TEXT
) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_commit_round_v2(
  TEXT, JSONB, TEXT, TEXT, DATE, TEXT, TEXT
) FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_commit_round_v2(
  TEXT, JSONB, TEXT, TEXT, DATE, TEXT, TEXT
) TO lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_icp_set_date_write_once_v1()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog
AS $lab_arena_icp_set_date_write_once$
DECLARE
  v_expected DATE;
BEGIN
  IF OLD.icp_set_date IS NOT NULL
     AND NEW.icp_set_date IS DISTINCT FROM OLD.icp_set_date THEN
    RAISE EXCEPTION 'lab_arena_icp_set_date_write_once' USING ERRCODE = '42501';
  END IF;
  IF OLD.icp_set_date IS NULL AND NEW.icp_set_date IS NOT NULL THEN
    BEGIN
      v_expected := pg_catalog.timezone(
        'UTC',
        (OLD.configuration_doc #>> '{schedule,submission_open}')::TIMESTAMPTZ
      )::DATE;
    EXCEPTION WHEN OTHERS THEN
      RAISE EXCEPTION 'lab_arena_icp_set_date_invalid' USING ERRCODE = '22023';
    END;
    IF OLD.status <> 'open'
       OR NEW.status <> 'committed'
       OR NEW.icp_set_date <> v_expected THEN
      RAISE EXCEPTION 'lab_arena_icp_set_date_transition_invalid' USING ERRCODE = '42501';
    END IF;
  END IF;
  RETURN NEW;
END;
$lab_arena_icp_set_date_write_once$;
ALTER FUNCTION public.lab_arena_icp_set_date_write_once_v1()
  OWNER TO lab_arena_owner;

DROP TRIGGER IF EXISTS lab_arena_icp_set_date_write_once
  ON public.lab_arena_rounds;
CREATE TRIGGER lab_arena_icp_set_date_write_once
  BEFORE UPDATE ON public.lab_arena_rounds
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_icp_set_date_write_once_v1();

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
  v_all_participants BOOLEAN;
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
  v_all_participants := p_stage = 1 OR v_round.icp_set_date IS NOT NULL;
  IF v_all_participants THEN
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
          v_all_participants
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

DROP POLICY IF EXISTS lab_arena_owner_current_daily_icp_set
  ON public.qualification_private_icp_sets;
CREATE POLICY lab_arena_owner_current_daily_icp_set
  ON public.qualification_private_icp_sets
  FOR SELECT
  TO lab_arena_owner
  USING (
    pg_catalog.char_length(set_id::TEXT) = 8
    AND pg_catalog.to_char(
      pg_catalog.to_date(set_id::TEXT, 'YYYYMMDD'), 'YYYYMMDD'
    ) = set_id::TEXT
    AND pg_catalog.to_date(set_id::TEXT, 'YYYYMMDD') <=
        pg_catalog.timezone('UTC', pg_catalog.statement_timestamp())::DATE
    AND (
      (
        pg_catalog.to_date(set_id::TEXT, 'YYYYMMDD') =
          pg_catalog.timezone('UTC', pg_catalog.statement_timestamp())::DATE
        AND is_active
        AND (active_from IS NULL OR active_from <= pg_catalog.statement_timestamp())
        AND (active_until IS NULL OR active_until > pg_catalog.statement_timestamp())
      )
      OR
      (
        pg_catalog.to_date(set_id::TEXT, 'YYYYMMDD') <
          pg_catalog.timezone('UTC', pg_catalog.statement_timestamp())::DATE
        AND (
          active_from IS NULL
          OR active_from < (
            pg_catalog.to_date(set_id::TEXT, 'YYYYMMDD') + 1
          )::TIMESTAMP AT TIME ZONE 'UTC'
        )
        AND (
          active_until IS NULL
          OR active_until >
            pg_catalog.to_date(set_id::TEXT, 'YYYYMMDD')::TIMESTAMP AT TIME ZONE 'UTC'
        )
      )
    )
  );

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
  v_today DATE;
  v_requested DATE;
  v_result JSONB;
BEGIN
  v_today := pg_catalog.timezone(
    'UTC', pg_catalog.statement_timestamp()
  )::DATE;
  IF p_set_id IS NULL
     OR pg_catalog.char_length(p_set_id::TEXT) <> 8
     OR pg_catalog.to_char(
          pg_catalog.to_date(p_set_id::TEXT, 'YYYYMMDD'), 'YYYYMMDD'
        ) <> p_set_id::TEXT THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'unavailable', 'set_id', p_set_id
    );
  END IF;
  v_requested := pg_catalog.to_date(p_set_id::TEXT, 'YYYYMMDD');
  IF v_requested > v_today THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'unavailable', 'set_id', p_set_id
    );
  END IF;

  SELECT pg_catalog.jsonb_build_object(
           'status', 'ready', 'set_id', source.set_id, 'icps', source.icps
         )
    INTO v_result
    FROM public.qualification_private_icp_sets AS source
   WHERE source.set_id = p_set_id
     AND (
       (
         v_requested = v_today
         AND source.is_active
         AND (source.active_from IS NULL OR source.active_from <= pg_catalog.statement_timestamp())
         AND (source.active_until IS NULL OR source.active_until > pg_catalog.statement_timestamp())
       )
       OR
       (
         v_requested < v_today
         AND (source.active_from IS NULL OR source.active_from < (v_requested + 1)::TIMESTAMP AT TIME ZONE 'UTC')
         AND (source.active_until IS NULL OR source.active_until > v_requested::TIMESTAMP AT TIME ZONE 'UTC')
       )
     )
     AND pg_catalog.jsonb_typeof(source.icps) = 'array'
     AND pg_catalog.jsonb_array_length(source.icps) = 20
   LIMIT 1;

  RETURN COALESCE(
    v_result,
    pg_catalog.jsonb_build_object(
      'status', 'unavailable', 'set_id', p_set_id
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
