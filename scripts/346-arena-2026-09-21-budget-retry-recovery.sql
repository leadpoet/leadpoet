-- Add one normal second attempt for the ten Sep21 baseline assignments that
-- exhausted the accidentally frozen 200-call OpenRouter quota.  Keep all
-- first attempts, provider charges, active later ICPs, sources, and round state.
-- Independent read-only object verification before release found archive SHA-256
-- 82a444e0282bac6820a61a4dcb7f8aba33423318f64ebd3aaf94c0b35415adc3 and
-- source commit 8e467397527cba8cde839a3f86302e35eb5a2edd.  Those values are not
-- persisted in the round tables, so this repair binds the exact stored ref and
-- size in all three locations where that source identity is recorded.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_ledger IN SHARE ROW EXCLUSIVE MODE;

DO $recover_sep21_budget_exhausted$
DECLARE
  v_round public.lab_arena_rounds;
  v_baseline public.lab_arena_submissions;
  v_positions CONSTANT INTEGER[] := ARRAY[0,1,2,3,4,5,6,7,8,9];
  v_source_ref CONSTANT TEXT :=
    'arena/arena-2026-09-21/sources/baseline-2026-09-21.tar.gz';
  v_source_size CONSTANT BIGINT := 858906;
  v_retry_count BIGINT;
  v_count BIGINT;
  v_round_before JSONB;
  v_submissions_before JSONB;
  v_runs_before JSONB;
  v_ledger_count BIGINT;
  v_ledger_max BIGINT;
  v_ledger_sum NUMERIC;
BEGIN
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-21'
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Sep21 retry recovery round is missing'
      USING ERRCODE = 'P0002';
  END IF;

  SELECT * INTO v_baseline
  FROM public.lab_arena_submissions
  WHERE submission_id = 'baseline-2026-09-21'
    AND round_id = v_round.round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Sep21 retry recovery baseline is missing'
      USING ERRCODE = 'P0002';
  END IF;

  -- These values identify the one frozen round and persisted source record.
  IF v_round.configuration_doc ->> 'schema_version'
       IS DISTINCT FROM 'leadpoet.lab_arena.round_configuration.v1'
     OR v_round.configuration_doc ->> 'round_id'
       IS DISTINCT FROM 'arena-2026-09-21'
     OR v_round.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
     OR v_round.configuration_doc #>> '{schedule,submission_cutoff}'
       IS DISTINCT FROM '2026-09-21T00:00:00Z'
     OR v_round.configuration_doc -> 'call_quotas'
       IS DISTINCT FROM
       '{"deepline":200,"openrouter":2000,"scrapingdog":200}'::JSONB
     OR (v_round.configuration_doc ->> 'icp_wall_clock_seconds')::INTEGER
       IS DISTINCT FROM 2700
     OR v_round.configuration_doc ->> 'checkpoint_deadline_policy'
       IS DISTINCT FROM 'atomic_checkpoint_45m_v1'
     OR (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
       IS DISTINCT FROM 4000000
     OR (v_round.configuration_doc ->> 'cost_per_company_microusd')::BIGINT
       IS DISTINCT FROM 800000
     OR v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
       IS DISTINCT FROM 'successful_calls_per_icp_v1'
     OR v_round.configuration_doc ->> 'parallel_twenty_icp_execution'
       IS DISTINCT FROM 'true'
     OR (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
       IS DISTINCT FROM 10
     OR (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER
       IS DISTINCT FROM 10
     OR (v_round.configuration_doc ->> 'max_attempts_per_assignment')::INTEGER
       IS DISTINCT FROM 2
     OR v_round.benchmark_ref IS DISTINCT FROM
       'arena/arena-2026-09-21/benchmark.json'
     OR v_round.evaluation_date IS DISTINCT FROM '2026-09-21'
     OR v_round.icp_set_date IS DISTINCT FROM '2026-09-20'
     OR pg_catalog.jsonb_array_length(v_round.participants) IS DISTINCT FROM 6
     OR (SELECT count(*)
         FROM pg_catalog.jsonb_array_elements(v_round.participants) AS item
         WHERE item ->> 'submission_id' = v_baseline.submission_id
           AND item ->> 'miner_hotkey' = v_baseline.miner_hotkey
           AND item ->> 'source_ref' = v_source_ref
           AND (item ->> 'source_size_bytes')::BIGINT = v_source_size
           AND COALESCE((item ->> 'is_king')::BOOLEAN,FALSE)) <> 1
     OR NOT v_baseline.is_king
     OR v_baseline.status IS DISTINCT FROM 'frozen'
     OR v_baseline.source_ref IS DISTINCT FROM v_source_ref
     OR v_baseline.source_size_bytes IS DISTINCT FROM v_source_size
     OR v_baseline.submission_doc ->> 'source_ref'
       IS DISTINCT FROM v_source_ref
     OR (v_baseline.submission_doc ->> 'source_size_bytes')::BIGINT
       IS DISTINCT FROM v_source_size
     OR v_baseline.submission_doc #>> '{consent,public_rerun}'
       IS DISTINCT FROM 'true'
     OR v_baseline.submission_doc ->> 'is_king'
       IS DISTINCT FROM 'true' THEN
    RAISE EXCEPTION 'Sep21 retry recovery source or profile differs'
      USING ERRCODE = '55000';
  END IF;

  SELECT count(*) INTO v_retry_count
  FROM public.lab_arena_runs AS retry
  WHERE retry.round_id = v_round.round_id
    AND retry.submission_id = v_baseline.submission_id
    AND retry.kind = 'execute'
    AND retry.icp_position = ANY(v_positions)
    AND retry.attempt = 2;

  IF v_retry_count = 10 THEN
    -- Replays may see pending, leased, or terminal retries.  Validate only the
    -- immutable identity copied by this repair and never reset progress.
    IF EXISTS (
      SELECT 1
      FROM public.lab_arena_runs AS retry
      LEFT JOIN public.lab_arena_runs AS prior
        ON prior.assignment_id = retry.assignment_id AND prior.attempt = 1
      WHERE retry.round_id = v_round.round_id
        AND retry.submission_id = v_baseline.submission_id
        AND retry.icp_position = ANY(v_positions)
        AND retry.attempt = 2
        AND (
          prior.run_id IS NULL
          OR retry.run_id IS DISTINCT FROM retry.assignment_id || ':2'
          OR retry.round_id IS DISTINCT FROM prior.round_id
          OR retry.submission_id IS DISTINCT FROM prior.submission_id
          OR retry.miner_hotkey IS DISTINCT FROM prior.miner_hotkey
          OR retry.stage IS DISTINCT FROM prior.stage
          OR retry.icp_position IS DISTINCT FROM prior.icp_position
          OR retry.kind IS DISTINCT FROM 'execute'
          OR retry.scored_run_id IS NOT NULL
          OR retry.stage_generation IS DISTINCT FROM prior.stage_generation
          OR retry.previous_runner_hotkey IS DISTINCT FROM prior.runner_hotkey
        )
    ) OR (SELECT count(DISTINCT icp_position)
          FROM public.lab_arena_runs
          WHERE round_id = v_round.round_id
            AND submission_id = v_baseline.submission_id
            AND kind = 'execute' AND attempt = 2
            AND icp_position = ANY(v_positions)) <> 10 THEN
      RAISE EXCEPTION 'Sep21 retry recovery replay rows differ'
        USING ERRCODE = '55000';
    END IF;
    RETURN;
  ELSIF v_retry_count <> 0 THEN
    RAISE EXCEPTION 'Sep21 retry recovery has conflicting second attempts'
      USING ERRCODE = '55000';
  END IF;

  IF v_round.status IS DISTINCT FROM 'stage1'
     OR v_round.cancel_reason IS NOT NULL
     OR v_round.stage1_scoring_plan_doc IS NOT NULL
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR v_round.stage3_scoring_plan_doc IS NOT NULL
     OR v_round.publication_doc IS NOT NULL
     OR v_round.published_at IS NOT NULL
     OR v_round.finalists IS NOT NULL THEN
    RAISE EXCEPTION 'Sep21 retry recovery requires active unscored stage1'
      USING ERRCODE = '55000';
  END IF;

  -- The baseline must still have exactly its twenty original assignments.
  IF (SELECT count(*) FROM public.lab_arena_runs
      WHERE round_id = v_round.round_id
        AND submission_id = v_baseline.submission_id
        AND kind = 'execute') <> 20
     OR (SELECT count(DISTINCT assignment_id) FROM public.lab_arena_runs
         WHERE round_id = v_round.round_id
           AND submission_id = v_baseline.submission_id
           AND kind = 'execute') <> 20
     OR (SELECT count(DISTINCT icp_position) FROM public.lab_arena_runs
         WHERE round_id = v_round.round_id
           AND submission_id = v_baseline.submission_id
           AND kind = 'execute') <> 20
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round.round_id
         AND submission_id = v_baseline.submission_id
         AND kind = 'execute'
         AND (
           attempt <> 1
           OR stage <> CASE WHEN icp_position < 10 THEN 1 ELSE 2 END
           OR assignment_id IS DISTINCT FROM
             v_round.round_id || ':' || v_baseline.submission_id || ':' ||
             (CASE WHEN icp_position < 10 THEN 1 ELSE 2 END)::TEXT || ':' ||
             icp_position::TEXT
           OR run_id IS DISTINCT FROM assignment_id || ':1'
           OR stage_generation IS DISTINCT FROM v_round.stage_generation
         )
     ) THEN
    RAISE EXCEPTION 'Sep21 retry recovery baseline assignments differ'
      USING ERRCODE = '55000';
  END IF;

  -- Only the ten exact budget-stopped first attempts are eligible.
  IF (SELECT count(*) FROM public.lab_arena_runs
      WHERE round_id = v_round.round_id
        AND submission_id = v_baseline.submission_id
        AND kind = 'execute' AND attempt = 1 AND stage = 1
        AND icp_position = ANY(v_positions)
        AND status = 'failed' AND terminal_cause = 'budget_exhausted'
        AND result_doc ->> 'terminal_status' = 'budget_exhausted'
        AND output_ref IS NULL AND runner_hotkey IS NOT NULL) <> 10 THEN
    RAISE EXCEPTION 'Sep21 retry recovery target failures differ'
      USING ERRCODE = '55000';
  END IF;

  IF EXISTS (
    WITH target_runs AS (
      SELECT run_id FROM public.lab_arena_runs
      WHERE round_id = v_round.round_id
        AND submission_id = v_baseline.submission_id
        AND kind = 'execute' AND attempt = 1
        AND icp_position = ANY(v_positions)
    ), heads AS (
      SELECT DISTINCT ON (ledger.run_id,ledger.call_identity)
        ledger.run_id,ledger.provider,ledger.entry_kind,ledger.entry_doc,
        ledger.amount_microusd
      FROM public.lab_arena_ledger AS ledger
      JOIN target_runs ON target_runs.run_id = ledger.run_id
      WHERE ledger.call_identity IS NOT NULL
      ORDER BY ledger.run_id,ledger.call_identity,ledger.entry_id DESC
    ), per_run AS (
      SELECT target_runs.run_id,
        count(*) FILTER (WHERE heads.provider = 'openrouter'
                           AND heads.entry_kind = 'settlement') AS settled_or,
        count(*) FILTER (WHERE heads.provider = 'openrouter'
                           AND heads.entry_kind = 'refusal'
                           AND heads.entry_doc ->> 'reason' = 'per_icp_quota')
          AS quota_refusals,
        count(*) FILTER (WHERE heads.entry_kind IN ('reservation','dispatch'))
          AS inflight,
        COALESCE(sum(heads.amount_microusd)
          FILTER (WHERE heads.entry_kind = 'settlement'),0) AS settled_cost
      FROM target_runs LEFT JOIN heads ON heads.run_id = target_runs.run_id
      GROUP BY target_runs.run_id
    )
    SELECT 1 FROM per_run
    WHERE settled_or <> 200 OR quota_refusals < 1 OR inflight <> 0
      OR settled_cost >= 4000000
  ) THEN
    RAISE EXCEPTION 'Sep21 retry recovery target ledger differs or is inflight'
      USING ERRCODE = '55000';
  END IF;

  SELECT pg_catalog.to_jsonb(v_round) INTO v_round_before;
  SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(row_data) ORDER BY submission_id)
    INTO v_submissions_before
  FROM public.lab_arena_submissions AS row_data
  WHERE round_id = v_round.round_id;
  SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(row_data) ORDER BY run_id)
    INTO v_runs_before
  FROM public.lab_arena_runs AS row_data
  WHERE round_id = v_round.round_id;
  SELECT count(*),max(entry_id),COALESCE(sum(amount_microusd),0)
    INTO v_ledger_count,v_ledger_max,v_ledger_sum
  FROM public.lab_arena_ledger WHERE round_id = v_round.round_id;

  INSERT INTO public.lab_arena_runs (
    run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
    icp_position,attempt,status,lease_generation,stage_generation,kind,
    previous_runner_hotkey
  )
  SELECT assignment_id || ':2',assignment_id,round_id,submission_id,
    miner_hotkey,stage,icp_position,2,'pending',lease_generation,
    stage_generation,'execute',runner_hotkey
  FROM public.lab_arena_runs
  WHERE round_id = v_round.round_id
    AND submission_id = v_baseline.submission_id
    AND kind = 'execute' AND attempt = 1
    AND icp_position = ANY(v_positions)
  ORDER BY icp_position;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 10 THEN
    RAISE EXCEPTION 'Sep21 retry recovery insert count differs'
      USING ERRCODE = '55000';
  END IF;

  IF (SELECT pg_catalog.to_jsonb(row_data)
      FROM public.lab_arena_rounds AS row_data
      WHERE round_id = v_round.round_id) IS DISTINCT FROM v_round_before
     OR (SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(row_data)
                                    ORDER BY submission_id)
         FROM public.lab_arena_submissions AS row_data
         WHERE round_id = v_round.round_id) IS DISTINCT FROM v_submissions_before
     OR (SELECT pg_catalog.jsonb_agg(pg_catalog.to_jsonb(row_data) ORDER BY run_id)
         FROM public.lab_arena_runs AS row_data
         WHERE round_id = v_round.round_id
           AND NOT (
             submission_id = v_baseline.submission_id
             AND kind = 'execute' AND attempt = 2
             AND icp_position = ANY(v_positions)
           ))
       IS DISTINCT FROM v_runs_before
     OR (SELECT ROW(count(*),max(entry_id),COALESCE(sum(amount_microusd),0))
         FROM public.lab_arena_ledger WHERE round_id = v_round.round_id)
       IS DISTINCT FROM ROW(v_ledger_count,v_ledger_max,v_ledger_sum) THEN
    RAISE EXCEPTION 'Sep21 retry recovery changed protected state'
      USING ERRCODE = '55000';
  END IF;
END;
$recover_sep21_budget_exhausted$;

COMMIT;
