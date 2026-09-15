-- Let newly flagged daily rounds execute all twenty ICPs before scoring.
-- Historical rounds keep the original two execution windows because their
-- immutable configuration does not contain the opt-in flag.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

DO $requires_current_arena$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_baseline_cost_eligibility_schema_v1()'
     ) IS NULL
     OR pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure(
       'public.lab_arena_baseline_cost_eligibility_schema_v1()'
     )) NOT LIKE '%''version'', 254%' THEN
    RAISE EXCEPTION 'apply migration 254 before migration 255';
  END IF;
END;
$requires_current_arena$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_open_parallel_execution_v1(
  p_round_id TEXT,
  p_participants JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $open_parallel_execution$
DECLARE
  v_round public.lab_arena_rounds;
  v_generation BIGINT;
  v_participant JSONB;
  v_submission public.lab_arena_submissions;
  v_position INTEGER;
  v_stage SMALLINT;
  v_assignment TEXT;
  v_created INTEGER := 0;
  v_participant_count INTEGER;
  v_distinct_count INTEGER;
BEGIN
  IF pg_catalog.jsonb_typeof(p_participants) IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(p_participants) < 1 THEN
    RAISE EXCEPTION 'lab_arena_parallel_execution_input_invalid'
      USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> 'committed' THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale', 'round_status', v_round.status,
      'stage_generation', v_round.stage_generation
    );
  END IF;
  IF v_round.configuration_doc ->> 'parallel_twenty_icp_execution'
       IS DISTINCT FROM 'true'
     OR pg_catalog.jsonb_typeof(
          v_round.configuration_doc -> 'parallel_twenty_icp_execution'
        ) IS DISTINCT FROM 'boolean'
     OR v_round.icp_set_date IS NULL
     OR (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER <> 10
     OR (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER <> 10
     OR (v_round.configuration_doc ->> 'runner_slot_ceiling')::INTEGER
          NOT BETWEEN 1 AND 20 THEN
    RAISE EXCEPTION 'lab_arena_parallel_execution_configuration_invalid'
      USING ERRCODE = '22023';
  END IF;
  SELECT COUNT(*), COUNT(DISTINCT participant ->> 'submission_id')
  INTO v_participant_count, v_distinct_count
  FROM pg_catalog.jsonb_array_elements(p_participants) AS participant;
  IF v_participant_count <> v_distinct_count
     OR v_participant_count <> pg_catalog.jsonb_array_length(v_round.participants)
  THEN
    RAISE EXCEPTION 'lab_arena_parallel_execution_participants_invalid'
      USING ERRCODE = '22023';
  END IF;
  v_generation := v_round.stage_generation + 1;
  FOR v_participant IN
    SELECT value FROM pg_catalog.jsonb_array_elements(p_participants)
  LOOP
    SELECT * INTO v_submission
    FROM public.lab_arena_submissions
    WHERE submission_id = v_participant ->> 'submission_id'
      AND round_id = p_round_id
      AND status = 'frozen';
    IF NOT FOUND
       OR v_submission.miner_hotkey IS DISTINCT FROM
            v_participant ->> 'miner_hotkey'
       OR NOT EXISTS (
         SELECT 1
         FROM pg_catalog.jsonb_array_elements(v_round.participants) AS original
         WHERE original ->> 'submission_id' = v_submission.submission_id
           AND original ->> 'miner_hotkey' = v_submission.miner_hotkey
       ) THEN
      RAISE EXCEPTION 'lab_arena_parallel_execution_participants_invalid'
        USING ERRCODE = '23503';
    END IF;
    FOR v_position IN 0 .. 19 LOOP
      v_stage := CASE WHEN v_position < 10 THEN 1 ELSE 2 END;
      v_assignment := p_round_id || ':' || v_submission.submission_id || ':'
        || v_stage::TEXT || ':' || v_position::TEXT;
      INSERT INTO public.lab_arena_runs (
        run_id, assignment_id, round_id, submission_id, miner_hotkey, stage,
        icp_position, attempt, status, stage_generation
      ) VALUES (
        v_assignment || ':1', v_assignment, p_round_id,
        v_submission.submission_id, v_submission.miner_hotkey, v_stage,
        v_position, 1, 'pending', v_generation
      );
      v_created := v_created + 1;
    END LOOP;
  END LOOP;
  UPDATE public.lab_arena_rounds
  SET status = 'stage1', status_generation = status_generation + 1,
      stage_generation = v_generation
  WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'ok', 'round_status', 'stage1',
    'stage_generation', v_generation, 'assignments', v_created
  );
END;
$open_parallel_execution$;
ALTER FUNCTION public.lab_arena_open_parallel_execution_v1(TEXT, JSONB)
  OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_close_parallel_execution_v1(
  p_round_id TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $close_parallel_execution$
DECLARE
  v_round public.lab_arena_rounds;
  v_run public.lab_arena_runs;
  v_generation BIGINT;
  v_assignments INTEGER;
  v_expected INTEGER;
  v_incomplete INTEGER;
  v_next TEXT;
BEGIN
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.configuration_doc ->> 'parallel_twenty_icp_execution'
       IS DISTINCT FROM 'true' THEN
    RAISE EXCEPTION 'lab_arena_parallel_execution_configuration_invalid'
      USING ERRCODE = '22023';
  END IF;
  IF v_round.status <> 'stage1' THEN
    IF v_round.status IN (
      'stage1_closed', 'stage1_scoring', 'stage1_judged', 'stage1_scored',
      'stage2', 'stage2_closed', 'stage2_scoring', 'stage2_judged',
      'scored', 'published'
    ) THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'existing', 'round_status', v_round.status,
        'stage_generation', v_round.stage_generation
      );
    END IF;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale', 'round_status', v_round.status,
      'stage_generation', v_round.stage_generation
    );
  END IF;
  SELECT COUNT(DISTINCT assignment_id) INTO v_assignments
  FROM public.lab_arena_runs
  WHERE round_id = p_round_id AND kind = 'execute' AND stage IN (1, 2);
  v_expected := pg_catalog.jsonb_array_length(v_round.participants) * 20;
  IF v_assignments <> v_expected THEN
    RAISE EXCEPTION 'lab_arena_parallel_execution_assignments_invalid'
      USING ERRCODE = '23514';
  END IF;
  v_generation := v_round.stage_generation + 1;
  FOR v_run IN
    SELECT * FROM public.lab_arena_runs
    WHERE round_id = p_round_id AND stage IN (1, 2) AND kind = 'execute'
      AND status IN ('leased', 'pending', 'submitted')
    ORDER BY assignment_id, attempt
    FOR UPDATE
  LOOP
    IF v_run.status = 'leased' THEN
      PERFORM public.lab_arena__terminate_open_calls(v_run.run_id, 'stage_closed');
    END IF;
    UPDATE public.lab_arena_runs
    SET status = 'failed', terminal_cause = 'stage_closed',
        terminal_doc = pg_catalog.jsonb_build_object(
          'closed_at', pg_catalog.clock_timestamp(),
          'previous_status', v_run.status
        )
    WHERE run_id = v_run.run_id;
  END LOOP;
  SELECT COUNT(*) INTO v_incomplete FROM (
    SELECT runs.assignment_id
    FROM public.lab_arena_runs AS runs
    WHERE runs.round_id = p_round_id
      AND runs.stage IN (1, 2)
      AND runs.kind = 'execute'
    GROUP BY runs.assignment_id
    HAVING bool_and(runs.status <> 'accepted')
       AND NOT bool_or(COALESCE(runs.terminal_cause, '') IN (
         'model_timeout', 'invalid_output', 'budget_exhausted',
         'credential_error', 'model_error'
       ))
  ) AS incomplete;
  IF v_incomplete > 0 THEN
    v_next := 'cancelled';
    UPDATE public.lab_arena_rounds
    SET status = 'cancelled', status_generation = status_generation + 1,
        stage_generation = v_generation,
        cancel_reason = 'execution_incomplete:stage1:' || v_incomplete::TEXT
    WHERE round_id = p_round_id;
  ELSE
    v_next := 'stage1_closed';
    UPDATE public.lab_arena_rounds
    SET status = v_next, status_generation = status_generation + 1,
        stage_generation = v_generation
    WHERE round_id = p_round_id;
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status', CASE WHEN v_next = 'cancelled' THEN 'cancelled' ELSE 'closed' END,
    'round_status', v_next, 'incomplete_assignments', v_incomplete,
    'stage_generation', v_generation
  );
END;
$close_parallel_execution$;
ALTER FUNCTION public.lab_arena_close_parallel_execution_v1(TEXT)
  OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_activate_preexecuted_stage2_v1(
  p_round_id TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $activate_stage2$
DECLARE
  v_round public.lab_arena_rounds;
  v_assignments INTEGER;
  v_open INTEGER;
  v_expected INTEGER;
  v_generation BIGINT;
BEGIN
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.configuration_doc ->> 'parallel_twenty_icp_execution'
       IS DISTINCT FROM 'true' THEN
    RAISE EXCEPTION 'lab_arena_parallel_execution_configuration_invalid'
      USING ERRCODE = '22023';
  END IF;
  IF v_round.status <> 'stage1_scored' THEN
    IF v_round.status IN (
      'stage2', 'stage2_closed', 'stage2_scoring', 'stage2_judged',
      'scored', 'published'
    ) THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'existing', 'round_status', v_round.status,
        'stage_generation', v_round.stage_generation
      );
    END IF;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale', 'round_status', v_round.status,
      'stage_generation', v_round.stage_generation
    );
  END IF;
  SELECT COUNT(DISTINCT assignment_id),
         COUNT(*) FILTER (WHERE status IN ('pending', 'leased', 'submitted'))
  INTO v_assignments, v_open
  FROM public.lab_arena_runs
  WHERE round_id = p_round_id AND stage = 2 AND kind = 'execute';
  v_expected := pg_catalog.jsonb_array_length(v_round.participants) * 10;
  IF v_assignments <> v_expected OR v_open <> 0 THEN
    RAISE EXCEPTION 'lab_arena_preexecuted_stage2_invalid'
      USING ERRCODE = '23514';
  END IF;
  v_generation := v_round.stage_generation + 1;
  UPDATE public.lab_arena_rounds
  SET status = 'stage2', status_generation = status_generation + 1,
      stage_generation = v_generation
  WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'ok', 'round_status', 'stage2',
    'stage_generation', v_generation, 'assignments', v_expected
  );
END;
$activate_stage2$;
ALTER FUNCTION public.lab_arena_activate_preexecuted_stage2_v1(TEXT)
  OWNER TO lab_arena_owner;

-- Provider calls and completion use this shared lease guard. Permit a
-- precreated stage-two execute lease only during the flagged stage-one
-- execution window; score leases and all historical rounds keep the exact
-- phase match.
DO $parallel_twenty_lease_guard$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$
     OR v_round.status <> ('stage' || v_run.stage::TEXT || CASE v_run.kind WHEN 'score' THEN '_scoring' ELSE '' END) THEN
$old$;
  v_new TEXT := $new$
     OR NOT (
       v_round.status = (
         'stage' || v_run.stage::TEXT
         || CASE v_run.kind WHEN 'score' THEN '_scoring' ELSE '' END
       )
       OR (
         -- lab_arena_parallel_twenty_lease_guard
         v_round.status = 'stage1'
         AND v_round.configuration_doc ->>
               'parallel_twenty_icp_execution' = 'true'
         AND v_run.kind = 'execute'
         AND v_run.stage IN (1, 2)
       )
     ) THEN
$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena__lock_current_lease(text,text)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_parallel_twenty_lease_guard'
     ) > 0 THEN
    RETURN;
  END IF;
  IF (pg_catalog.length(v_definition)
       - pg_catalog.length(pg_catalog.replace(v_definition, v_old, '')))
       / pg_catalog.length(v_old) <> 1 THEN
    RAISE EXCEPTION 'lab_arena_current_lease parallel shape unexpected';
  END IF;
  EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
END;
$parallel_twenty_lease_guard$;

-- Patch the composed claim function instead of replacing it. This retains the
-- restart drain, score serialization, judgment cache, source, credential,
-- champion funding, and provider reconciliation protections installed before
-- this migration.
DO $parallel_twenty_claim$
DECLARE
  v_definition TEXT;
  v_declaration_old TEXT := $old$
  v_limit INTEGER;
  v_active INTEGER;
$old$;
  v_declaration_new TEXT := $new$
  v_limit INTEGER;
  v_batch_size INTEGER;
  v_active INTEGER;
$new$;
  v_limit_old TEXT := $old$
  v_limit := LEAST(p_declared_parallelism, p_slot_ceiling);
  SELECT COUNT(*) INTO v_active FROM public.lab_arena_runs
$old$;
  v_limit_new TEXT := $new$
  -- lab_arena_parallel_twenty_icp_execution: the smaller of the signed local
  -- capacity and immutable round cap owns the lease limit and per-model
  -- ICP barrier. A request cannot exceed the frozen round cap.
  IF v_round.configuration_doc ->> 'parallel_twenty_icp_execution' = 'true' THEN
    IF pg_catalog.jsonb_typeof(
         v_round.configuration_doc -> 'runner_slot_ceiling'
       ) IS DISTINCT FROM 'number'
       OR (v_round.configuration_doc ->> 'runner_slot_ceiling')::INTEGER
            NOT BETWEEN 1 AND 20
       OR p_slot_ceiling IS DISTINCT FROM
            (v_round.configuration_doc ->> 'runner_slot_ceiling')::INTEGER THEN
      RAISE EXCEPTION 'lab_arena_parallel_execution_capacity_invalid'
        USING ERRCODE = '22023';
    END IF;
    v_batch_size :=
      (v_round.configuration_doc ->> 'runner_slot_ceiling')::INTEGER;
  ELSE
    v_batch_size := p_slot_ceiling;
  END IF;
  v_limit := LEAST(p_declared_parallelism, v_batch_size);
  SELECT COUNT(*) INTO v_active FROM public.lab_arena_runs
$new$;
  v_where_old TEXT := $old$
  WHERE runs.round_id = p_round_id AND runs.stage = v_stage AND runs.status = 'pending'
    AND runs.stage_generation = v_round.stage_generation
$old$;
  v_where_new TEXT := $new$
  WHERE runs.round_id = p_round_id
    AND (
      runs.stage = v_stage
      OR (
        v_round.status = 'stage1'
        AND v_round.configuration_doc ->>
              'parallel_twenty_icp_execution' = 'true'
        AND runs.kind = 'execute'
        AND runs.stage = 2
      )
    )
    AND runs.status = 'pending'
    AND runs.stage_generation = v_round.stage_generation
    AND (
      v_round.status <> 'stage1'
      OR v_round.configuration_doc ->>
           'parallel_twenty_icp_execution' IS DISTINCT FROM 'true'
      OR runs.kind <> 'execute'
      OR runs.icp_position / v_limit = (
        SELECT MIN(open_run.icp_position / v_limit)
        FROM public.lab_arena_runs AS open_run
        WHERE open_run.round_id = runs.round_id
          AND open_run.submission_id = runs.submission_id
          AND open_run.kind = 'execute'
          AND open_run.stage IN (1, 2)
          AND open_run.stage_generation = v_round.stage_generation
          AND open_run.status IN ('pending', 'leased', 'submitted')
      )
    )
$new$;
  v_order_old TEXT := $old$
  ORDER BY runs.icp_position, runs.created_at, runs.assignment_id
$old$;
  v_order_new TEXT := $new$
  ORDER BY
    CASE WHEN v_round.status = 'stage1'
                  AND v_round.configuration_doc ->>
                        'parallel_twenty_icp_execution' = 'true'
                  AND runs.kind = 'execute'
      THEN (
        SELECT participant.ordinality
        FROM pg_catalog.jsonb_array_elements(v_round.participants)
             WITH ORDINALITY AS participant(value, ordinality)
        WHERE participant.value ->> 'submission_id' = runs.submission_id
      )
    END,
    CASE WHEN v_round.status = 'stage1'
                  AND v_round.configuration_doc ->>
                        'parallel_twenty_icp_execution' = 'true'
                  AND runs.kind = 'execute'
      THEN runs.icp_position
    END,
    runs.icp_position, runs.created_at, runs.assignment_id
$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_parallel_twenty_icp_execution'
     ) > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, 'lab_arena_score_submission_serialization') = 0
     OR pg_catalog.strpos(v_definition, 'lab_arena_closed_scoring_reservation_claim') = 0
     OR pg_catalog.strpos(v_definition, 'runner_authority_exclusions') = 0
     OR pg_catalog.strpos(v_definition, 'company_judgment_cache') = 0
     OR pg_catalog.strpos(v_definition, 'champion_funding_sources') = 0
     OR pg_catalog.strpos(v_definition, 'lab-arena-claim-control') = 0 THEN
    RAISE EXCEPTION 'apply the current Arena claim migrations before migration 255';
  END IF;
  IF (pg_catalog.length(v_definition)
       - pg_catalog.length(pg_catalog.replace(v_definition, v_declaration_old, '')))
       / pg_catalog.length(v_declaration_old) <> 1
     OR (pg_catalog.length(v_definition)
       - pg_catalog.length(pg_catalog.replace(v_definition, v_limit_old, '')))
       / pg_catalog.length(v_limit_old) <> 1
     OR (pg_catalog.length(v_definition)
       - pg_catalog.length(pg_catalog.replace(v_definition, v_where_old, '')))
       / pg_catalog.length(v_where_old) <> 1
     OR (pg_catalog.length(v_definition)
       - pg_catalog.length(pg_catalog.replace(v_definition, v_order_old, '')))
       / pg_catalog.length(v_order_old) <> 1 THEN
    RAISE EXCEPTION 'lab_arena_claim_assignment parallel shape unexpected';
  END IF;
  v_definition := pg_catalog.replace(
    v_definition, v_declaration_old, v_declaration_new
  );
  v_definition := pg_catalog.replace(v_definition, v_limit_old, v_limit_new);
  v_definition := pg_catalog.replace(v_definition, v_where_old, v_where_new);
  EXECUTE pg_catalog.replace(v_definition, v_order_old, v_order_new);
END;
$parallel_twenty_claim$;

DO $verify_parallel_twenty$
DECLARE
  v_definition TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF v_definition IS NULL
     OR pg_catalog.strpos(
       v_definition, 'lab_arena_parallel_twenty_icp_execution'
     ) = 0
     OR pg_catalog.strpos(
       v_definition, 'lab_arena_score_submission_serialization'
     ) = 0
     OR pg_catalog.strpos(v_definition, 'company_judgment_cache') = 0
     OR pg_catalog.strpos(v_definition, 'champion_funding_sources') = 0 THEN
    RAISE EXCEPTION 'lab_arena_parallel_execution_claim_invalid';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena__lock_current_lease(text,text)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF v_definition IS NULL
     OR pg_catalog.strpos(
       v_definition, 'lab_arena_parallel_twenty_lease_guard'
     ) = 0 THEN
    RAISE EXCEPTION 'lab_arena_parallel_execution_lease_guard_invalid';
  END IF;
END;
$verify_parallel_twenty$;

CREATE OR REPLACE FUNCTION public.lab_arena_parallel_execution_schema_v1()
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $parallel_execution_schema$
DECLARE
  v_claim_definition TEXT;
  v_lease_definition TEXT;
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_open_parallel_execution_v1(text,jsonb)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_close_parallel_execution_v1(text)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_activate_preexecuted_stage2_v1(text)'
     ) IS NULL THEN
    RAISE EXCEPTION 'lab_arena_parallel_execution_rpc_missing';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::pg_catalog.regprocedure
  ) INTO v_claim_definition;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena__lock_current_lease(text,text)'::pg_catalog.regprocedure
  ) INTO v_lease_definition;
  IF pg_catalog.strpos(
       v_claim_definition, 'lab_arena_parallel_twenty_icp_execution'
     ) = 0
     OR pg_catalog.strpos(
       v_lease_definition, 'lab_arena_parallel_twenty_lease_guard'
     ) = 0 THEN
    RAISE EXCEPTION 'lab_arena_parallel_execution_guard_missing';
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.parallel_execution_schema.v1',
    'version', 255,
    'max_parallel_icps', 20
  );
END;
$parallel_execution_schema$;
ALTER FUNCTION public.lab_arena_parallel_execution_schema_v1()
  OWNER TO lab_arena_owner;

REVOKE ALL ON FUNCTION public.lab_arena_open_parallel_execution_v1(TEXT, JSONB)
  FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.lab_arena_close_parallel_execution_v1(TEXT)
  FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.lab_arena_activate_preexecuted_stage2_v1(TEXT)
  FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.lab_arena_parallel_execution_schema_v1()
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_open_parallel_execution_v1(TEXT, JSONB),
  public.lab_arena_close_parallel_execution_v1(TEXT),
  public.lab_arena_activate_preexecuted_stage2_v1(TEXT),
  public.lab_arena_parallel_execution_schema_v1()
  TO lab_arena_service;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
