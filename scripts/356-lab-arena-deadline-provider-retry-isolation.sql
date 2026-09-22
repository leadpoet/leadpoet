-- A dispatched final retry that the stage deadline closes may fall back to an
-- earlier authoritative provider error for that assignment and score zero.
-- Unstarted retries, unknown work, and trust failures still stop the round.
-- This migration changes only future stage closure; it does not rewrite
-- historical rounds, run results, or ledger entries.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $deadline_provider_retry_prerequisites$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_close_stage(text,smallint)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_close_parallel_execution_v1(text)'
     ) IS NULL
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'lab_arena_owner'
     )
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'lab_arena_service'
     ) THEN
    RAISE EXCEPTION 'apply migrations 326 and 353 before migration 356';
  END IF;
END;
$deadline_provider_retry_prerequisites$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_close_stage(
  p_round_id TEXT,
  p_stage SMALLINT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_close_stage$
DECLARE
  v_round public.lab_arena_rounds;
  v_run public.lab_arena_runs;
  v_generation BIGINT;
  v_incomplete INTEGER;
  v_deadline_provider_retry_exhausted BOOLEAN;
  v_next TEXT;
BEGIN
  IF p_stage NOT IN (1, 2) THEN
    RAISE EXCEPTION 'lab_arena_stage_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round
  FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> ('stage' || p_stage::TEXT) THEN
    IF (p_stage = 1 AND v_round.status IN (
          'stage1_closed', 'stage1_scoring', 'stage1_judged', 'stage1_scored',
          'stage2', 'stage2_closed', 'stage2_scoring', 'stage2_judged',
          'scored', 'published'
        ))
       OR (p_stage = 2 AND v_round.status IN (
          'stage2_closed', 'stage2_scoring', 'stage2_judged',
          'scored', 'published'
        )) THEN
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

  v_generation := v_round.stage_generation + 1;
  FOR v_run IN
    SELECT * FROM public.lab_arena_runs
    WHERE round_id = p_round_id AND stage = p_stage AND kind = 'execute'
      AND status IN ('leased', 'pending', 'submitted')
    ORDER BY assignment_id, attempt
    FOR UPDATE
  LOOP
    v_deadline_provider_retry_exhausted := FALSE;
    IF v_run.status = 'leased' THEN
      v_deadline_provider_retry_exhausted :=
        v_run.attempt >= 2
        AND EXISTS (
          SELECT 1
          FROM public.lab_arena_runs AS prior
          WHERE prior.assignment_id = v_run.assignment_id
            AND prior.attempt < v_run.attempt
            AND prior.status = 'failed'
            AND prior.terminal_cause = 'provider_error'
        )
        AND EXISTS (
          SELECT 1
          FROM public.lab_arena_ledger AS evidence
          WHERE evidence.run_id = v_run.run_id
            AND evidence.entry_kind = 'dispatch'
        );
      PERFORM public.lab_arena__terminate_open_calls(v_run.run_id, 'stage_closed');
    END IF;
    UPDATE public.lab_arena_runs
    SET status = 'failed', terminal_cause = 'stage_closed',
        terminal_doc = pg_catalog.jsonb_build_object(
          'closed_at', pg_catalog.clock_timestamp(),
          'previous_status', v_run.status
        ) || CASE WHEN v_deadline_provider_retry_exhausted
          THEN pg_catalog.jsonb_build_object(
            'deadline_provider_retry_exhausted', TRUE
          )
          ELSE '{}'::JSONB
        END
    WHERE run_id = v_run.run_id;
  END LOOP;

  SELECT COUNT(*) INTO v_incomplete
  FROM (
    SELECT
      runs.assignment_id,
      pg_catalog.bool_or(runs.status = 'accepted') AS has_accepted,
      pg_catalog.bool_or(COALESCE(runs.terminal_cause, '') IN (
        'model_timeout', 'invalid_output', 'budget_exhausted',
        'credential_error', 'model_error'
      )) AS has_model_zero,
      (pg_catalog.array_agg(runs.attempt ORDER BY runs.attempt DESC))[1]
        AS latest_attempt,
      (pg_catalog.array_agg(runs.status ORDER BY runs.attempt DESC))[1]
        AS latest_status,
      (pg_catalog.array_agg(
        COALESCE(runs.terminal_cause, '') ORDER BY runs.attempt DESC
      ))[1] AS latest_cause,
      (pg_catalog.array_agg(
        runs.terminal_doc ORDER BY runs.attempt DESC
      ))[1] AS latest_terminal_doc
    FROM public.lab_arena_runs AS runs
    WHERE runs.round_id = p_round_id
      AND runs.stage = p_stage
      AND runs.kind = 'execute'
    GROUP BY runs.assignment_id
  ) AS outcomes
  WHERE NOT outcomes.has_accepted
    AND NOT outcomes.has_model_zero
    AND NOT (
      outcomes.latest_attempt >= 2
      AND outcomes.latest_status = 'failed'
      AND outcomes.latest_cause = 'provider_error'
    )
    AND NOT (
      outcomes.latest_attempt >= 2
      AND outcomes.latest_status = 'failed'
      AND outcomes.latest_cause = 'stage_closed'
      AND COALESCE(
        outcomes.latest_terminal_doc @>
          '{"deadline_provider_retry_exhausted":true}'::JSONB,
        FALSE
      )
    );

  IF v_incomplete > 0 THEN
    v_next := 'cancelled';
    UPDATE public.lab_arena_rounds
    SET status = 'cancelled',
        status_generation = status_generation + 1,
        stage_generation = v_generation,
        cancel_reason =
          'execution_incomplete:stage' || p_stage::TEXT || ':' || v_incomplete::TEXT
    WHERE round_id = p_round_id;
  ELSE
    v_next := 'stage' || p_stage::TEXT || '_closed';
    UPDATE public.lab_arena_rounds
    SET status = v_next,
        status_generation = status_generation + 1,
        stage_generation = v_generation
    WHERE round_id = p_round_id;
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status', CASE WHEN v_next = 'cancelled' THEN 'cancelled' ELSE 'closed' END,
    'round_status', v_next,
    'incomplete_assignments', v_incomplete,
    'stage_generation', v_generation
  );
END;
$lab_arena_close_stage$;
ALTER FUNCTION public.lab_arena_close_stage(TEXT, SMALLINT)
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
  v_deadline_provider_retry_exhausted BOOLEAN;
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
  v_expected := pg_catalog.jsonb_array_length(v_round.participants) *
    ((v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
     + (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER);
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
    v_deadline_provider_retry_exhausted := FALSE;
    IF v_run.status = 'leased' THEN
      v_deadline_provider_retry_exhausted :=
        v_run.attempt >= 2
        AND EXISTS (
          SELECT 1
          FROM public.lab_arena_runs AS prior
          WHERE prior.assignment_id = v_run.assignment_id
            AND prior.attempt < v_run.attempt
            AND prior.status = 'failed'
            AND prior.terminal_cause = 'provider_error'
        )
        AND EXISTS (
          SELECT 1
          FROM public.lab_arena_ledger AS evidence
          WHERE evidence.run_id = v_run.run_id
            AND evidence.entry_kind = 'dispatch'
        );
      PERFORM public.lab_arena__terminate_open_calls(v_run.run_id, 'stage_closed');
    END IF;
    UPDATE public.lab_arena_runs
    SET status = 'failed', terminal_cause = 'stage_closed',
        terminal_doc = pg_catalog.jsonb_build_object(
          'closed_at', pg_catalog.clock_timestamp(),
          'previous_status', v_run.status
        ) || CASE WHEN v_deadline_provider_retry_exhausted
          THEN pg_catalog.jsonb_build_object(
            'deadline_provider_retry_exhausted', TRUE
          )
          ELSE '{}'::JSONB
        END
    WHERE run_id = v_run.run_id;
  END LOOP;

  SELECT COUNT(*) INTO v_incomplete
  FROM (
    SELECT
      runs.assignment_id,
      pg_catalog.bool_or(runs.status = 'accepted') AS has_accepted,
      pg_catalog.bool_or(COALESCE(runs.terminal_cause, '') IN (
        'model_timeout', 'invalid_output', 'budget_exhausted',
        'credential_error', 'model_error'
      )) AS has_model_zero,
      (pg_catalog.array_agg(runs.attempt ORDER BY runs.attempt DESC))[1]
        AS latest_attempt,
      (pg_catalog.array_agg(runs.status ORDER BY runs.attempt DESC))[1]
        AS latest_status,
      (pg_catalog.array_agg(
        COALESCE(runs.terminal_cause, '') ORDER BY runs.attempt DESC
      ))[1] AS latest_cause,
      (pg_catalog.array_agg(
        runs.terminal_doc ORDER BY runs.attempt DESC
      ))[1] AS latest_terminal_doc
    FROM public.lab_arena_runs AS runs
    WHERE runs.round_id = p_round_id
      AND runs.stage IN (1, 2)
      AND runs.kind = 'execute'
    GROUP BY runs.assignment_id
  ) AS outcomes
  WHERE NOT outcomes.has_accepted
    AND NOT outcomes.has_model_zero
    AND NOT (
      outcomes.latest_attempt >= 2
      AND outcomes.latest_status = 'failed'
      AND outcomes.latest_cause = 'provider_error'
    )
    AND NOT (
      outcomes.latest_attempt >= 2
      AND outcomes.latest_status = 'failed'
      AND outcomes.latest_cause = 'stage_closed'
      AND COALESCE(
        outcomes.latest_terminal_doc @>
          '{"deadline_provider_retry_exhausted":true}'::JSONB,
        FALSE
      )
    );

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

REVOKE ALL ON FUNCTION public.lab_arena_close_stage(TEXT, SMALLINT)
  FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.lab_arena_close_parallel_execution_v1(TEXT)
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_close_stage(TEXT, SMALLINT),
  public.lab_arena_close_parallel_execution_v1(TEXT)
  TO lab_arena_service;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
