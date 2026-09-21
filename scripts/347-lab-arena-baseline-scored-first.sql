-- Future opted-in rounds score the baseline on all twenty ICPs before any
-- miner execution can be created. Historical scheduling remains unchanged.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

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
  v_baseline_first BOOLEAN;
BEGIN
  IF p_stage NOT IN (1, 2)
     OR pg_catalog.jsonb_typeof(p_participants) IS DISTINCT FROM 'array'
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
  v_baseline_first := COALESCE(
    v_round.configuration_doc ->> 'execution_sequence_policy'
      = 'baseline_scored_first_v1',
    FALSE
  );
  IF pg_catalog.jsonb_array_length(p_participants) < 1
     AND NOT (v_baseline_first AND p_stage = 2) THEN
    RAISE EXCEPTION 'lab_arena_stage_input_invalid' USING ERRCODE = '22023';
  END IF;
  IF v_baseline_first
     AND COALESCE(
       (v_round.configuration_doc ->> 'parallel_twenty_icp_execution')::BOOLEAN,
       FALSE
     ) THEN
    RAISE EXCEPTION 'lab_arena_execution_sequence_invalid' USING ERRCODE = '22023';
  END IF;
  IF v_baseline_first THEN
    IF p_icp_positions IS DISTINCT FROM
       ARRAY[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] THEN
      RAISE EXCEPTION 'lab_arena_stage_position_invalid' USING ERRCODE = '22023';
    END IF;
  ELSIF (p_stage = 1 AND p_icp_positions IS DISTINCT FROM ARRAY[0,1,2,3,4,5,6,7,8,9])
     OR (p_stage = 2 AND p_icp_positions IS DISTINCT FROM ARRAY[10,11,12,13,14,15,16,17,18,19]) THEN
    RAISE EXCEPTION 'lab_arena_stage_position_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT COUNT(*), COUNT(DISTINCT participant ->> 'submission_id')
  INTO v_participant_count, v_distinct_count
  FROM pg_catalog.jsonb_array_elements(p_participants) AS participant;
  IF v_participant_count <> v_distinct_count THEN
    RAISE EXCEPTION 'lab_arena_stage_participants_duplicate' USING ERRCODE = '22023';
  END IF;
  v_all_participants := NOT v_baseline_first
    AND (p_stage = 1 OR v_round.icp_set_date IS NOT NULL);
  IF v_baseline_first THEN
    SELECT COUNT(*) INTO v_expected_count
    FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
    WHERE COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE) = (p_stage = 1);
  ELSIF v_all_participants THEN
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
          (v_baseline_first AND
            COALESCE((original ->> 'is_king')::BOOLEAN, FALSE) = (p_stage = 1))
          OR v_all_participants
          OR (NOT v_baseline_first AND (
            COALESCE((original ->> 'is_king')::BOOLEAN, FALSE)
            OR v_round.finalists ? v_submission.submission_id
          ))
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

-- Keep every later integrity/contact receipt check in the installed trigger.
-- Replace only its fixed stage-position predicate and fail closed if a later
-- migration has changed that predicate.
DO $extend_run_position_guard$
DECLARE
  v_definition TEXT;
  v_old CONSTANT TEXT := $old$IF (NEW.stage = 1 AND NEW.icp_position NOT BETWEEN 0 AND 9)
     OR (NEW.stage = 2 AND NEW.icp_position NOT BETWEEN 10 AND
           CASE WHEN v_round.configuration_doc ->> 'integrity_policy'
                       = 'arena_integrity_v1' THEN 19 ELSE 29 END) THEN$old$;
  v_new CONSTANT TEXT := $new$IF (
       v_round.configuration_doc ->> 'execution_sequence_policy'
         = 'baseline_scored_first_v1'
       AND (
         NEW.stage NOT IN (1, 2)
         OR NEW.icp_position NOT BETWEEN 0 AND 19
       )
     ) OR (
       v_round.configuration_doc ->> 'execution_sequence_policy'
         IS DISTINCT FROM 'baseline_scored_first_v1'
       AND (
         (NEW.stage = 1 AND NEW.icp_position NOT BETWEEN 0 AND 9)
         OR (NEW.stage = 2 AND NEW.icp_position NOT BETWEEN 10 AND
               CASE WHEN v_round.configuration_doc ->> 'integrity_policy'
                           = 'arena_integrity_v1' THEN 19 ELSE 29 END)
       )
     ) THEN$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_integrity_run_guard_v1()'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, v_old) > 0
     AND pg_catalog.strpos(v_definition, v_new) = 0 THEN
    IF (pg_catalog.char_length(v_definition) - pg_catalog.char_length(
          pg_catalog.replace(v_definition, v_old, '')
        )) / pg_catalog.char_length(v_old) <> 1 THEN
      RAISE EXCEPTION 'lab_arena_run_position_guard_shape_changed';
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  ELSIF pg_catalog.strpos(v_definition, v_old) = 0
        AND pg_catalog.strpos(v_definition, v_new) > 0 THEN
    NULL;
  ELSE
    RAISE EXCEPTION 'lab_arena_run_position_guard_shape_changed';
  END IF;
END;
$extend_run_position_guard$;

DO $extend_scoring_position_guard$
DECLARE
  v_definition TEXT;
  v_old CONSTANT TEXT := $old$(v_item ->> 'icp_position')::INTEGER NOT BETWEEN (CASE p_stage WHEN 1 THEN 0 ELSE 10 END) AND (CASE p_stage WHEN 1 THEN 9 ELSE 19 END)$old$;
  v_new CONSTANT TEXT := $new$(v_item ->> 'icp_position')::INTEGER NOT BETWEEN
         (CASE WHEN v_round.configuration_doc ->> 'execution_sequence_policy'
                     = 'baseline_scored_first_v1'
               THEN 0 ELSE CASE p_stage WHEN 1 THEN 0 ELSE 10 END END)
         AND
         (CASE WHEN v_round.configuration_doc ->> 'execution_sequence_policy'
                     = 'baseline_scored_first_v1'
               THEN 19 ELSE CASE p_stage WHEN 1 THEN 9 ELSE 19 END END)$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, v_old) > 0
     AND pg_catalog.strpos(v_definition, v_new) = 0 THEN
    IF (pg_catalog.char_length(v_definition) - pg_catalog.char_length(
          pg_catalog.replace(v_definition, v_old, '')
        )) / pg_catalog.char_length(v_old) <> 1 THEN
      RAISE EXCEPTION 'lab_arena_scoring_position_guard_shape_changed';
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  ELSIF pg_catalog.strpos(v_definition, v_old) = 0
        AND pg_catalog.strpos(v_definition, v_new) > 0 THEN
    NULL;
  ELSE
    RAISE EXCEPTION 'lab_arena_scoring_position_guard_shape_changed';
  END IF;
END;
$extend_scoring_position_guard$;

DO $extend_stage1_score_guard$
DECLARE
  v_definition TEXT;
  v_old TEXT;
  v_new TEXT;
  v_changed BOOLEAN := FALSE;
  v_old_count INTEGER;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_transition_round(text,text,text,jsonb)'::pg_catalog.regprocedure
  ) INTO v_definition;

  v_old := $old$OR pg_catalog.jsonb_array_length(v_patch -> 'finalists') > 10$old$;
  v_new := $new$OR pg_catalog.jsonb_array_length(v_patch -> 'finalists') >
          (CASE WHEN v_round.configuration_doc ->> 'execution_sequence_policy'
                      = 'baseline_scored_first_v1'
               THEN (v_round.configuration_doc ->> 'max_challengers')::INTEGER
               ELSE 10 END)$new$;
  v_old_count := (pg_catalog.char_length(v_definition) - pg_catalog.char_length(
    pg_catalog.replace(v_definition, v_old, ''))) / pg_catalog.char_length(v_old);
  IF v_old_count = 1 AND pg_catalog.strpos(v_definition, v_new) = 0 THEN
    v_definition := pg_catalog.replace(v_definition, v_old, v_new);
    v_changed := TRUE;
  ELSIF v_old_count <> 0 OR pg_catalog.strpos(v_definition, v_new) = 0 THEN
    RAISE EXCEPTION 'lab_arena_finalist_max_guard_shape_changed';
  END IF;

  v_old := $old$) <> 10
      AND ($old$;
  v_new := $new$) <> (CASE WHEN v_round.configuration_doc ->> 'execution_sequence_policy'
                         = 'baseline_scored_first_v1' THEN 20 ELSE 10 END)
      AND ($new$;
  v_old_count := (pg_catalog.char_length(v_definition) - pg_catalog.char_length(
    pg_catalog.replace(v_definition, v_old, ''))) / pg_catalog.char_length(v_old);
  IF v_old_count = 1 AND pg_catalog.strpos(v_definition, v_new) = 0 THEN
    v_definition := pg_catalog.replace(v_definition, v_old, v_new);
    v_changed := TRUE;
  ELSIF v_old_count <> 0 OR pg_catalog.strpos(v_definition, v_new) = 0 THEN
    RAISE EXCEPTION 'lab_arena_baseline_count_guard_shape_changed';
  END IF;

  v_old := $old$) = 10
      );
    SELECT COUNT(*) INTO v_invalid_finalists$old$;
  v_new := $new$) = (CASE WHEN v_round.configuration_doc ->> 'execution_sequence_policy'
                         = 'baseline_scored_first_v1' THEN 0 ELSE 10 END)
      );
    SELECT COUNT(*) INTO v_invalid_finalists$new$;
  v_old_count := (pg_catalog.char_length(v_definition) - pg_catalog.char_length(
    pg_catalog.replace(v_definition, v_old, ''))) / pg_catalog.char_length(v_old);
  IF v_old_count = 1 AND pg_catalog.strpos(v_definition, v_new) = 0 THEN
    v_definition := pg_catalog.replace(v_definition, v_old, v_new);
    v_changed := TRUE;
  ELSIF v_old_count <> 0 OR pg_catalog.strpos(v_definition, v_new) = 0 THEN
    RAISE EXCEPTION 'lab_arena_challenger_count_guard_shape_changed';
  END IF;

  v_old := $old$) = 10
           )
       );
    IF v_invalid_finalists <> 0$old$;
  v_new := $new$) = (CASE WHEN v_round.configuration_doc ->> 'execution_sequence_policy'
                         = 'baseline_scored_first_v1' THEN 0 ELSE 10 END)
           )
       );
    IF v_invalid_finalists <> 0$new$;
  v_old_count := (pg_catalog.char_length(v_definition) - pg_catalog.char_length(
    pg_catalog.replace(v_definition, v_old, ''))) / pg_catalog.char_length(v_old);
  IF v_old_count = 1 AND pg_catalog.strpos(v_definition, v_new) = 0 THEN
    v_definition := pg_catalog.replace(v_definition, v_old, v_new);
    v_changed := TRUE;
  ELSIF v_old_count <> 0 OR pg_catalog.strpos(v_definition, v_new) = 0 THEN
    RAISE EXCEPTION 'lab_arena_finalist_membership_guard_shape_changed';
  END IF;

  v_old := $old$<> LEAST(10, v_challenger_count)$old$;
  v_new := $new$<> (CASE WHEN v_round.configuration_doc ->> 'execution_sequence_policy'
                           = 'baseline_scored_first_v1'
                    THEN v_challenger_count
                    ELSE LEAST(10, v_challenger_count) END)$new$;
  v_old_count := (pg_catalog.char_length(v_definition) - pg_catalog.char_length(
    pg_catalog.replace(v_definition, v_old, ''))) / pg_catalog.char_length(v_old);
  IF v_old_count = 1 AND pg_catalog.strpos(v_definition, v_new) = 0 THEN
    v_definition := pg_catalog.replace(v_definition, v_old, v_new);
    v_changed := TRUE;
  ELSIF v_old_count <> 0 OR pg_catalog.strpos(v_definition, v_new) = 0 THEN
    RAISE EXCEPTION 'lab_arena_finalist_count_guard_shape_changed';
  END IF;

  IF v_changed THEN
    EXECUTE v_definition;
  END IF;
END;
$extend_stage1_score_guard$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
