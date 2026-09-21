-- Use frozen stage counts and promotion margin for future rounds. Existing
-- configuration documents and signed hashes remain unchanged.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

ALTER TABLE public.lab_arena_runs
  DROP CONSTRAINT lab_arena_runs_icp_position_check;
ALTER TABLE public.lab_arena_runs
  ADD CONSTRAINT lab_arena_runs_icp_position_check
  CHECK (icp_position BETWEEN 0 AND 99);

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
  v_stage_1_count INTEGER;
  v_total_count INTEGER;
  v_expected_positions INTEGER[];
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
  v_stage_1_count := (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER;
  v_total_count := v_stage_1_count
    + (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER;
  IF v_stage_1_count < 1 OR v_total_count NOT BETWEEN 2 AND 100 THEN
    RAISE EXCEPTION 'lab_arena_stage_position_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT ARRAY(SELECT pg_catalog.generate_series(
    CASE WHEN v_baseline_first OR p_stage = 1 THEN 0 ELSE v_stage_1_count END,
    CASE WHEN v_baseline_first OR p_stage = 2 THEN v_total_count - 1
         ELSE v_stage_1_count - 1 END
  )) INTO v_expected_positions;
  IF p_icp_positions IS DISTINCT FROM v_expected_positions THEN
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

-- Patch only the installed predicates that carry a historical position count.
-- A missing or changed anchor aborts the migration before any change commits.
CREATE OR REPLACE FUNCTION pg_temp.arena_353_replace(
  p_function REGPROCEDURE, p_old TEXT, p_new TEXT,
  p_expected INTEGER DEFAULT 1
) RETURNS VOID LANGUAGE plpgsql AS $replace$
DECLARE
  v_definition TEXT;
  v_count INTEGER;
  v_new_count INTEGER;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p_function) INTO v_definition;
  IF v_definition IS NULL THEN
    RAISE EXCEPTION 'arena 353 function missing: %', p_function;
  END IF;
  v_count := (pg_catalog.length(v_definition)
      - pg_catalog.length(pg_catalog.replace(v_definition, p_old, '')))
      / pg_catalog.length(p_old);
  v_new_count := (pg_catalog.length(v_definition)
      - pg_catalog.length(pg_catalog.replace(v_definition, p_new, '')))
      / pg_catalog.length(p_new);
  IF v_new_count = p_expected THEN
    RETURN;
  END IF;
  IF v_count <> p_expected OR v_new_count <> 0 THEN
    RAISE EXCEPTION 'arena 353 function shape changed: %', p_function;
  END IF;
  EXECUTE pg_catalog.replace(v_definition, p_old, p_new);
END;
$replace$;

SELECT pg_temp.arena_353_replace(
  'public.lab_arena_integrity_run_guard_v1()'::REGPROCEDURE,
  $old$NEW.icp_position NOT BETWEEN 0 AND 19$old$,
  $new$NEW.icp_position NOT BETWEEN 0 AND
           (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
           + (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER - 1$new$
);
SELECT pg_temp.arena_353_replace(
  'public.lab_arena_integrity_run_guard_v1()'::REGPROCEDURE,
  $old$(NEW.stage = 1 AND NEW.icp_position NOT BETWEEN 0 AND 9)
         OR (NEW.stage = 2 AND NEW.icp_position NOT BETWEEN 10 AND
               CASE WHEN v_round.configuration_doc ->> 'integrity_policy'
                           = 'arena_integrity_v1' THEN 19 ELSE 29 END)$old$,
  $new$(NEW.stage = 1 AND NEW.icp_position NOT BETWEEN 0 AND
               (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER - 1)
         OR (NEW.stage = 2 AND NEW.icp_position NOT BETWEEN
               (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER AND
               (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
               + (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER - 1)$new$
);
ALTER FUNCTION public.lab_arena_integrity_run_guard_v1() OWNER TO lab_arena_owner;

SELECT pg_temp.arena_353_replace(
  'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::REGPROCEDURE,
  $old$(v_item ->> 'icp_position')::INTEGER NOT BETWEEN
         (CASE WHEN v_round.configuration_doc ->> 'execution_sequence_policy'
                     = 'baseline_scored_first_v1'
               THEN 0 ELSE CASE p_stage WHEN 1 THEN 0 ELSE 10 END END)
         AND
         (CASE WHEN v_round.configuration_doc ->> 'execution_sequence_policy'
                     = 'baseline_scored_first_v1'
               THEN 19 ELSE CASE p_stage WHEN 1 THEN 9 ELSE 19 END END)$old$,
  $new$(v_item ->> 'icp_position')::INTEGER NOT BETWEEN
         (CASE WHEN v_round.configuration_doc ->> 'execution_sequence_policy'
                     = 'baseline_scored_first_v1' OR p_stage = 1
               THEN 0 ELSE (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER END)
         AND
         (CASE WHEN v_round.configuration_doc ->> 'execution_sequence_policy'
                     = 'baseline_scored_first_v1' OR p_stage = 2
               THEN (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
                    + (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER - 1
               ELSE (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER - 1 END)$new$
);
ALTER FUNCTION public.lab_arena_open_scoring_v2(TEXT, SMALLINT, JSONB) OWNER TO lab_arena_owner;

SELECT pg_temp.arena_353_replace(
  'public.lab_arena_open_scoring_v3(text,smallint,jsonb)'::REGPROCEDURE,
  $old$(v_item ->> 'icp_position')::INTEGER NOT BETWEEN 0 AND 29$old$,
  $new$(v_item ->> 'icp_position')::INTEGER NOT BETWEEN 0 AND
           (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
           + (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER - 1$new$
);
ALTER FUNCTION public.lab_arena_open_scoring_v3(TEXT,SMALLINT,JSONB) OWNER TO lab_arena_owner;

SELECT pg_temp.arena_353_replace(
  'public.lab_arena_open_scoring(text,smallint,jsonb)'::REGPROCEDURE,
  $old$(v_item ->> 'icp_position')::INTEGER NOT BETWEEN 0 AND 29$old$,
  $new$(v_item ->> 'icp_position')::INTEGER NOT BETWEEN 0 AND
           (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
           + (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER - 1$new$
);
ALTER FUNCTION public.lab_arena_open_scoring(TEXT,SMALLINT,JSONB) OWNER TO lab_arena_owner;

SELECT pg_temp.arena_353_replace(
  'public.lab_arena_transition_round(text,text,text,jsonb)'::REGPROCEDURE,
  $old$THEN 20 ELSE 10 END)$old$,
  $new$THEN (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
                    + (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER
               ELSE (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER END)$new$
);
SELECT pg_temp.arena_353_replace(
  'public.lab_arena_transition_round(text,text,text,jsonb)'::REGPROCEDURE,
  $old$THEN 0 ELSE 10 END)$old$,
  $new$THEN 0 ELSE (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER END)$new$,
  2
);
ALTER FUNCTION public.lab_arena_transition_round(TEXT,TEXT,TEXT,JSONB) OWNER TO lab_arena_owner;

SELECT pg_temp.arena_353_replace(
  'public.lab_arena_current_daily_icp_set(bigint)'::REGPROCEDURE,
  $old$pg_catalog.jsonb_array_length(source.icps) = 20$old$,
  $new$pg_catalog.jsonb_array_length(source.icps) BETWEEN 2 AND 100$new$
);
ALTER FUNCTION public.lab_arena_current_daily_icp_set(BIGINT) OWNER TO lab_arena_owner;

SELECT pg_temp.arena_353_replace(
  'public.lab_arena_open_parallel_execution_v1(text,jsonb)'::REGPROCEDURE,
  $old$OR (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER <> 10
     OR (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER <> 10
     OR (v_round.configuration_doc ->> 'runner_slot_ceiling')::INTEGER
          NOT BETWEEN 1 AND 20$old$,
  $new$OR (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER < 1
     OR (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER < 1
     OR (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
          + (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER > 100
     OR (v_round.configuration_doc ->> 'runner_slot_ceiling')::INTEGER
          NOT BETWEEN 1 AND 20$new$
);
SELECT pg_temp.arena_353_replace(
  'public.lab_arena_open_parallel_execution_v1(text,jsonb)'::REGPROCEDURE,
  $old$FOR v_position IN 0 .. 19 LOOP
      v_stage := CASE WHEN v_position < 10 THEN 1 ELSE 2 END;$old$,
  $new$FOR v_position IN 0 ..
        (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
        + (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER - 1 LOOP
      v_stage := CASE WHEN v_position <
        (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
        THEN 1 ELSE 2 END;$new$
);
ALTER FUNCTION public.lab_arena_open_parallel_execution_v1(TEXT,JSONB) OWNER TO lab_arena_owner;

SELECT pg_temp.arena_353_replace(
  'public.lab_arena_close_parallel_execution_v1(text)'::REGPROCEDURE,
  $old$v_expected := pg_catalog.jsonb_array_length(v_round.participants) * 20;$old$,
  $new$v_expected := pg_catalog.jsonb_array_length(v_round.participants) *
    ((v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
     + (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER);$new$
);
ALTER FUNCTION public.lab_arena_close_parallel_execution_v1(TEXT) OWNER TO lab_arena_owner;

SELECT pg_temp.arena_353_replace(
  'public.lab_arena_activate_preexecuted_stage2_v1(text)'::REGPROCEDURE,
  $old$v_expected := pg_catalog.jsonb_array_length(v_round.participants) * 10;$old$,
  $new$v_expected := pg_catalog.jsonb_array_length(v_round.participants) *
    (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER;$new$
);
ALTER FUNCTION public.lab_arena_activate_preexecuted_stage2_v1(TEXT) OWNER TO lab_arena_owner;

SELECT pg_temp.arena_353_replace(
  'public.lab_arena_integrity_publication_guard_v1()'::REGPROCEDURE,
  $old$  v_allowed_failure BOOLEAN;$old$,
  $new$  v_allowed_failure BOOLEAN;
  v_benchmark_count INTEGER;
  v_promotion_margin NUMERIC;$new$
);
SELECT pg_temp.arena_353_replace(
  'public.lab_arena_integrity_publication_guard_v1()'::REGPROCEDURE,
  $old$  IF pg_catalog.jsonb_array_length(NEW.publication_doc -> 'participants')$old$,
  $new$  v_benchmark_count :=
    (NEW.configuration_doc ->> 'stage_1_icp_count')::INTEGER
    + (NEW.configuration_doc ->> 'stage_2_icp_count')::INTEGER;
  v_promotion_margin := COALESCE(
    (NEW.configuration_doc ->> 'promotion_margin')::NUMERIC, 1
  );
  IF v_benchmark_count NOT BETWEEN 2 AND 100
     OR v_promotion_margin NOT BETWEEN 0 AND 100 THEN
    RAISE EXCEPTION 'lab_arena_publication_configuration_invalid'
      USING ERRCODE = '22023';
  END IF;
  IF pg_catalog.jsonb_array_length(NEW.publication_doc -> 'participants')$new$
);
SELECT pg_temp.arena_353_replace(
  'public.lab_arena_integrity_publication_guard_v1()'::REGPROCEDURE,
  $old$AND runs.icp_position BETWEEN 0 AND 19$old$,
  $new$AND runs.icp_position BETWEEN 0 AND v_benchmark_count - 1$new$
);
SELECT pg_temp.arena_353_replace(
  'public.lab_arena_integrity_publication_guard_v1()'::REGPROCEDURE,
  $old$IF v_persisted_count = 20 THEN$old$,
  $new$IF v_persisted_count = v_benchmark_count THEN$new$
);
SELECT pg_temp.arena_353_replace(
  'public.lab_arena_integrity_publication_guard_v1()'::REGPROCEDURE,
  $old$pg_catalog.generate_series(0, 19)$old$,
  $new$pg_catalog.generate_series(0, v_benchmark_count - 1)$new$,
  2
);
SELECT pg_temp.arena_353_replace(
  'public.lab_arena_integrity_publication_guard_v1()'::REGPROCEDURE,
  $old$(v_cost ->> 'qualified_company_count')::NUMERIC > 100$old$,
  $new$(v_cost ->> 'qualified_company_count')::NUMERIC > 5 * v_benchmark_count$new$
);
SELECT pg_temp.arena_353_replace(
  'public.lab_arena_integrity_publication_guard_v1()'::REGPROCEDURE,
  $old$(ranking ->> 'final_score')::NUMERIC >= v_baseline_score + 1$old$,
  $new$(ranking ->> 'final_score')::NUMERIC >= v_baseline_score + v_promotion_margin$new$
);
ALTER FUNCTION public.lab_arena_integrity_publication_guard_v1() OWNER TO lab_arena_owner;

SELECT pg_temp.arena_353_replace(
  'public.lab_arena__integrity_submission_summary(text,text,integer[])'::REGPROCEDURE,
  $old$  v_expected INTEGER;$old$,
  $new$  v_expected INTEGER;
  v_benchmark_count INTEGER;$new$
);
SELECT pg_temp.arena_353_replace(
  'public.lab_arena__integrity_submission_summary(text,text,integer[])'::REGPROCEDURE,
  $old$  v_expected := pg_catalog.cardinality(p_positions);$old$,
  $new$  SELECT (configuration_doc ->> 'stage_1_icp_count')::INTEGER
       + (configuration_doc ->> 'stage_2_icp_count')::INTEGER
  INTO v_benchmark_count FROM public.lab_arena_rounds WHERE round_id = p_round_id;
  v_expected := pg_catalog.cardinality(p_positions);$new$
);
SELECT pg_temp.arena_353_replace(
  'public.lab_arena__integrity_submission_summary(text,text,integer[])'::REGPROCEDURE,
  $old$IF v_expected IS NULL OR v_expected < 1$old$,
  $new$IF v_benchmark_count IS NULL OR v_benchmark_count NOT BETWEEN 2 AND 100
     OR v_expected IS NULL OR v_expected < 1$new$
);
SELECT pg_temp.arena_353_replace(
  'public.lab_arena__integrity_submission_summary(text,text,integer[])'::REGPROCEDURE,
  $old$WHERE value NOT BETWEEN 0 AND 24$old$,
  $new$WHERE value NOT BETWEEN 0 AND v_benchmark_count - 1$new$
);
ALTER FUNCTION public.lab_arena__integrity_submission_summary(TEXT,TEXT,INTEGER[])
  OWNER TO lab_arena_owner;

SELECT pg_temp.arena_353_replace(
  'public.lab_arena_publication_baseline_guard_v1()'::REGPROCEDURE,
  $old$(v_summary ->> 'returned_company_count')::NUMERIC NOT BETWEEN 0 AND 100$old$,
  $new$(v_summary ->> 'returned_company_count')::NUMERIC NOT BETWEEN 0 AND
           5 * ((NEW.configuration_doc ->> 'stage_1_icp_count')::INTEGER
                + (NEW.configuration_doc ->> 'stage_2_icp_count')::INTEGER)$new$
);
SELECT pg_temp.arena_353_replace(
  'public.lab_arena_publication_baseline_guard_v1()'::REGPROCEDURE,
  $old$v_baseline_score + 1$old$,
  $new$v_baseline_score + COALESCE(
      (NEW.configuration_doc ->> 'promotion_margin')::NUMERIC, 1
    )$new$,
  2
);
ALTER FUNCTION public.lab_arena_publication_baseline_guard_v1() OWNER TO lab_arena_owner;

SELECT pg_temp.arena_353_replace(
  'public.lab_arena_icp_cost_eligibility(text,text,integer,integer)'::REGPROCEDURE,
  $old$p_icp_position NOT BETWEEN 0 AND 19$old$,
  $new$p_icp_position NOT BETWEEN 0 AND
       (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
       + (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER - 1$new$
);
ALTER FUNCTION public.lab_arena_icp_cost_eligibility(TEXT,TEXT,INTEGER,INTEGER) OWNER TO lab_arena_owner;

SELECT pg_temp.arena_353_replace(
  'public.lab_arena__per_icp_publication_valid(text,jsonb)'::REGPROCEDURE,
  $old$  v_is_baseline BOOLEAN;$old$,
  $new$  v_is_baseline BOOLEAN;
  v_benchmark_count INTEGER;$new$
);
SELECT pg_temp.arena_353_replace(
  'public.lab_arena__per_icp_publication_valid(text,jsonb)'::REGPROCEDURE,
  $old$BEGIN
  IF v_cost ->> 'sourcing_cost_eligibility_policy'$old$,
  $new$BEGIN
  SELECT (configuration_doc ->> 'stage_1_icp_count')::INTEGER
       + (configuration_doc ->> 'stage_2_icp_count')::INTEGER
  INTO v_benchmark_count FROM public.lab_arena_rounds WHERE round_id = p_round_id;
  IF v_benchmark_count NOT BETWEEN 2 AND 100 THEN
    RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch' USING ERRCODE = '22023';
  END IF;
  IF v_cost ->> 'sourcing_cost_eligibility_policy'$new$
);
SELECT pg_temp.arena_353_replace(
  'public.lab_arena__per_icp_publication_valid(text,jsonb)'::REGPROCEDURE,
  $old$pg_catalog.jsonb_array_length(v_cost -> 'per_icp') <> 20$old$,
  $new$pg_catalog.jsonb_array_length(v_cost -> 'per_icp') <> v_benchmark_count$new$
);
SELECT pg_temp.arena_353_replace(
  'public.lab_arena__per_icp_publication_valid(text,jsonb)'::REGPROCEDURE,
  $old$FOR v_position IN 0..19 LOOP$old$,
  $new$FOR v_position IN 0..v_benchmark_count - 1 LOOP$new$
);
SELECT pg_temp.arena_353_replace(
  'public.lab_arena__per_icp_publication_valid(text,jsonb)'::REGPROCEDURE,
  $old$IF v_score_count <> 20$old$,
  $new$IF v_score_count <> v_benchmark_count$new$
);
ALTER FUNCTION public.lab_arena__per_icp_publication_valid(TEXT,JSONB) OWNER TO lab_arena_owner;

SELECT pg_temp.arena_353_replace(
  'public.lab_arena_create_round(text,jsonb)'::REGPROCEDURE,
  $old$  PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('lab_arena.rounds', 0));$old$,
  $new$  IF pg_catalog.jsonb_typeof(p_configuration_doc -> 'stage_1_icp_count') IS DISTINCT FROM 'number'
     OR pg_catalog.jsonb_typeof(p_configuration_doc -> 'stage_2_icp_count') IS DISTINCT FROM 'number'
     OR (p_configuration_doc ->> 'stage_1_icp_count')::INTEGER < 1
     OR (p_configuration_doc ->> 'stage_2_icp_count')::INTEGER < 1
     OR (p_configuration_doc ->> 'stage_1_icp_count')::INTEGER
        + (p_configuration_doc ->> 'stage_2_icp_count')::INTEGER > 100
     OR (p_configuration_doc ? 'promotion_margin' AND (
       pg_catalog.jsonb_typeof(p_configuration_doc -> 'promotion_margin') IS DISTINCT FROM 'number'
       OR (p_configuration_doc ->> 'promotion_margin')::NUMERIC NOT BETWEEN 0 AND 100
     )) THEN
    RAISE EXCEPTION 'lab_arena_round_configuration_invalid' USING ERRCODE = '22023';
  END IF;
  PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('lab_arena.rounds', 0));$new$
);
ALTER FUNCTION public.lab_arena_create_round(TEXT,JSONB) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_dynamic_benchmark_schema_v1()
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public AS $schema$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.dynamic_benchmark_schema.v1',
    'version', 353,
    'max_benchmark_icps', 100,
    'default_benchmark_icps', 10,
    'default_promotion_margin', 0.5
  );
$schema$;
ALTER FUNCTION public.lab_arena_dynamic_benchmark_schema_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_dynamic_benchmark_schema_v1()
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_dynamic_benchmark_schema_v1()
  TO lab_arena_service;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
