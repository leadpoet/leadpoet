-- Resume only the three logical stage-1 scores left unfinished when the
-- September 14 Arena recovery was cancelled. Existing rows remain immutable.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE MODE;
LOCK TABLE public.lab_arena_submission_credentials IN SHARE MODE;
LOCK TABLE public.lab_arena_ledger IN SHARE MODE;
LOCK TABLE public.lab_arena_judgment_cache IN SHARE MODE;
LOCK TABLE public.lab_arena_company_judgment_reservations IN SHARE MODE;

DO $lab_arena_249_recover_20260914_remaining_scoring$
DECLARE
  v_round_id CONSTANT TEXT := 'arena-2026-09-14';
  v_target_submission CONSTANT TEXT :=
    'sub-dd76cafd44732cb230a476ab031b24ad';
  v_recovery_suffix CONSTANT TEXT := ':recovery249';
  v_prior_suffix CONSTANT TEXT := ':recovery244';
  v_target_assignments CONSTANT TEXT[] := ARRAY[
    'arena-2026-09-14:sub-dd76cafd44732cb230a476ab031b24ad:1:5:score:recovery249',
    'arena-2026-09-14:sub-dd76cafd44732cb230a476ab031b24ad:1:6:score:recovery249',
    'arena-2026-09-14:sub-dd76cafd44732cb230a476ab031b24ad:1:8:score:recovery249'
  ]::TEXT[];
  v_participant_ids CONSTANT TEXT[] := ARRAY[
    'baseline-2026-09-14',
    'sub-09f530ce94b6f63221ce4d69962e3988',
    'sub-0bee4ffefcb6a792ca70d16d722fa1ac',
    'sub-287f56f3e3718776a6aa10e4c130e154',
    'sub-2968d4fba9582f9d8881bfd998290cbe',
    'sub-4c5b60cfe00763f2d022031ecbd40039',
    'sub-4e87858d429552dd34271afa3be03a9b',
    'sub-804245a7d75aa38aa4c74a59e07b51d2',
    'sub-8ec81a864da4e10204d803a3fb7c3b58',
    'sub-946ff9115aec053aeb36a02476175b82',
    'sub-a7a50fba0fe795bc6f87a433fb6682f3',
    'sub-dae52a6a6cca8db5914a259116bc1b56',
    'sub-dd76cafd44732cb230a476ab031b24ad'
  ]::TEXT[];
  v_scorer_digest CONSTANT TEXT :=
    'sha256:412beed7799ecfcceb9c07c44f36644f8214c33513a117d603dede98c0b918e9';
  v_scorer_reference CONSTANT TEXT :=
    '493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@'
    || v_scorer_digest;
  v_expected_schedule CONSTANT JSONB := pg_catalog.jsonb_build_object(
    'submission_open', '2026-09-13T00:00:00Z',
    'submission_cutoff', '2026-09-14T00:00:00Z',
    'benchmark_deadline', '2026-09-14T00:30:00Z',
    'stage_1_start', '2026-09-14T00:30:01Z',
    'stage_1_close', '2026-09-14T04:30:01Z',
    'stage_1_scoring_close', '2026-09-14T11:00:01Z',
    'stage_2_start', '2026-09-14T11:00:02Z',
    'stage_2_close', '2026-09-14T14:00:02Z',
    'final_scoring_close', '2026-09-14T20:30:02Z',
    'stage_3_start', '2026-09-14T20:30:03Z',
    'stage_3_close', '2026-09-14T21:30:03Z',
    'stage_3_scoring_close', '2026-09-14T23:20:03Z',
    'publication_deadline', '2026-09-14T23:20:04Z'
  );

  v_round public.lab_arena_rounds%ROWTYPE;
  v_runs_hash TEXT;
  v_ledger_hash TEXT;
  v_submission_hash TEXT;
  v_credential_hash TEXT;
  v_cache_hash TEXT;
  v_reservation_hash TEXT;
  v_unrelated_hash TEXT;
  v_round_stable JSONB;
  v_configuration_stable JSONB;
  v_stage1_scoring_close TIMESTAMPTZ;
  v_schema JSONB;
  v_definition TEXT;
  v_missing_positions INTEGER[];
  v_missing_submissions TEXT[];
  v_count INTEGER;
BEGIN
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = v_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RETURN;
  END IF;

  -- A replay is a no-op only when the complete three-assignment namespace is
  -- already present with the exact frozen logical-item bindings. Later retry
  -- attempts and terminal progress do not invalidate that marker.
  SELECT pg_catalog.count(*) INTO v_count
  FROM public.lab_arena_runs
  WHERE round_id = v_round_id
    AND assignment_id LIKE '%' || v_recovery_suffix;
  IF v_count > 0 THEN
    IF (SELECT pg_catalog.count(*)
        FROM public.lab_arena_runs
        WHERE round_id = v_round_id
          AND assignment_id LIKE '%' || v_recovery_suffix
          AND attempt = 1) <> 3
       OR (SELECT pg_catalog.array_agg(DISTINCT assignment_id ORDER BY assignment_id)
           FROM public.lab_arena_runs
           WHERE round_id = v_round_id
             AND assignment_id LIKE '%' || v_recovery_suffix)
          IS DISTINCT FROM v_target_assignments
       OR EXISTS (
         SELECT 1
         FROM public.lab_arena_runs AS recovered
         WHERE recovered.round_id = v_round_id
           AND recovered.assignment_id LIKE '%' || v_recovery_suffix
           AND (
             recovered.submission_id <> v_target_submission
             OR recovered.stage <> 1
             OR recovered.kind <> 'score'
             OR recovered.icp_position NOT IN (5, 6, 8)
             OR recovered.assignment_id <> v_round_id || ':'
                || recovered.submission_id || ':1:'
                || recovered.icp_position::TEXT || ':score' || v_recovery_suffix
             OR recovered.scored_run_id <> v_round_id || ':'
                || recovered.submission_id || ':1:'
                || recovered.icp_position::TEXT || ':1'
             OR recovered.stage_generation <> 8
           )
       )
       OR EXISTS (
         SELECT 1
         FROM public.lab_arena_runs AS recovered
         LEFT JOIN LATERAL (
           SELECT prior.*
           FROM public.lab_arena_runs AS prior
           WHERE prior.round_id = v_round_id
             AND prior.submission_id = recovered.submission_id
             AND prior.icp_position = recovered.icp_position
             AND prior.scored_run_id = recovered.scored_run_id
             AND prior.assignment_id LIKE '%' || v_prior_suffix
           ORDER BY prior.stage_generation DESC, prior.attempt DESC,
             prior.created_at DESC, prior.run_id DESC
           LIMIT 1
         ) AS prior ON TRUE
         WHERE recovered.round_id = v_round_id
           AND recovered.assignment_id LIKE '%' || v_recovery_suffix
           AND recovered.attempt = 1
           AND (
             prior.run_id IS NULL
             OR recovered.judgment_cache_key
                IS DISTINCT FROM prior.judgment_cache_key
             OR recovered.judgment_input_hash
                IS DISTINCT FROM prior.judgment_input_hash
             OR recovered.judgment_scope_doc
                IS DISTINCT FROM prior.judgment_scope_doc
             OR recovered.company_judgment_refs
                IS DISTINCT FROM prior.company_judgment_refs
           )
       ) THEN
      RAISE EXCEPTION 'arena 2026-09-14 recovery249 replay state differs';
    END IF;
    RETURN;
  END IF;

  -- Never reopen the round unless the closed-scoring reservation fix is
  -- installed. The schema version and both modified admission paths are
  -- checked because the recovery depends on all three pieces together.
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_deepline_cost_reconciliation_schema_v1()'
     ) IS NULL THEN
    RAISE EXCEPTION 'arena 2026-09-14 recovery249 prerequisite missing';
  END IF;
  SELECT public.lab_arena_deepline_cost_reconciliation_schema_v1()
  INTO v_schema;
  IF (v_schema ->> 'version')::INTEGER < 248 THEN
    RAISE EXCEPTION 'arena 2026-09-14 recovery249 prerequisite version differs';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'
      ::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_closed_scoring_reservation_admission'
     ) = 0 THEN
    RAISE EXCEPTION 'arena 2026-09-14 recovery249 reserve prerequisite missing';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'
      ::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_closed_scoring_reservation_claim'
     ) = 0 THEN
    RAISE EXCEPTION 'arena 2026-09-14 recovery249 claim prerequisite missing';
  END IF;

  IF v_round.status <> 'cancelled'
     OR v_round.cancel_reason <> 'scoring_incomplete'
     OR v_round.status_generation <> 8
     OR v_round.stage_generation <> 7
     OR v_round.evaluation_date <> '2026-09-14'
     OR v_round.icp_set_date <> '2026-09-13'
     OR v_round.published_at IS NOT NULL
     OR v_round.publication_doc IS NOT NULL
     OR v_round.finalists IS NOT NULL
     OR v_round.king_outcome IS NOT NULL
     OR v_round.reward_basis_hash IS NOT NULL
     OR v_round.reward_basis_doc IS NOT NULL
     OR v_round.signing_key_doc IS NOT NULL
     OR v_round.reward_activated_at IS NOT NULL
     OR pg_catalog.jsonb_typeof(v_round.participants) IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(v_round.participants) <> 13
     OR (
       SELECT pg_catalog.array_agg(
         item ->> 'submission_id' ORDER BY item ->> 'submission_id'
       )
       FROM pg_catalog.jsonb_array_elements(v_round.participants)
         AS participant(item)
     ) IS DISTINCT FROM v_participant_ids
     OR pg_catalog.jsonb_typeof(v_round.stage1_scoring_plan_doc)
        IS DISTINCT FROM 'object'
     OR v_round.stage1_scoring_plan_doc ->> 'round_id' IS DISTINCT FROM v_round_id
     OR (v_round.stage1_scoring_plan_doc ->> 'stage')::INTEGER IS DISTINCT FROM 1
     OR pg_catalog.jsonb_array_length(
          v_round.stage1_scoring_plan_doc -> 'work_items'
        ) <> 130
     OR pg_catalog.jsonb_array_length(
          v_round.stage1_scoring_plan_doc -> 'zero_rows'
        ) <> 0
     OR v_round.configuration_doc -> 'schedule' IS DISTINCT FROM v_expected_schedule
     OR v_round.configuration_doc ->> 'round_id' IS DISTINCT FROM v_round_id
     OR v_round.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
     OR v_round.configuration_doc ->> 'rewards_enabled' IS DISTINCT FROM 'true'
     OR v_round.configuration_doc ->> 'stage_1_icp_count' IS DISTINCT FROM '10'
     OR v_round.configuration_doc ->> 'stage_2_icp_count' IS DISTINCT FROM '10'
     OR v_round.configuration_doc ->> 'execution_cap_microusd'
        IS DISTINCT FROM '80000000'
     OR v_round.configuration_doc ->> 'scoring_cap_microusd'
        IS DISTINCT FROM '50000000'
     OR v_round.configuration_doc ->> 'cost_per_company_microusd'
        IS DISTINCT FROM '800000'
     OR v_round.configuration_doc ->> 'integrity_policy'
        IS DISTINCT FROM 'arena_integrity_v1'
     OR v_round.configuration_doc ->> 'max_attempts_per_assignment'
        IS DISTINCT FROM '2'
     OR v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
        IS DISTINCT FROM 'successful_calls_v1'
     OR v_round.configuration_doc ->> 'scorer_image_digest'
        IS DISTINCT FROM v_scorer_digest
     OR v_round.configuration_doc ->> 'scorer_image_reference'
        IS DISTINCT FROM v_scorer_reference
     OR (SELECT pg_catalog.count(*)
         FROM public.lab_arena_submissions
         WHERE round_id = v_round_id AND status = 'frozen') <> 13
     OR (SELECT pg_catalog.count(*)
         FROM public.lab_arena_submission_credentials
         WHERE submission_id = ANY(v_participant_ids)) <> 28
     OR (SELECT pg_catalog.count(*)
         FROM public.lab_arena_submission_credentials
         WHERE submission_id = ANY(v_participant_ids)
           AND provider = 'openrouter') <> 12
     OR (SELECT pg_catalog.count(*)
         FROM public.lab_arena_submission_credentials
         WHERE submission_id = ANY(v_participant_ids)
           AND provider = 'deepline') <> 12
     OR (SELECT pg_catalog.count(*)
         FROM public.lab_arena_submission_credentials
         WHERE submission_id = ANY(v_participant_ids)
           AND provider = 'scrapingdog') <> 4 THEN
    RAISE EXCEPTION 'arena 2026-09-14 frozen round state differs for recovery249';
  END IF;

  IF pg_catalog.strpos(
       pg_catalog.pg_get_functiondef(
         'public.lab_arena_close_scoring(text,smallint)'::pg_catalog.regprocedure
       ),
       'lab_arena_scoring_logical_item_completion'
     ) = 0 THEN
    RAISE EXCEPTION 'arena 2026-09-14 logical scoring completion guard missing';
  END IF;

  -- Bind the exact observed post-recovery244 state. There are 130 accepted
  -- executions, 127 accepted logical scores, and only three logical scores
  -- without an accepted result.
  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id = v_round_id) <> 358
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'execute'
           AND status = 'accepted' AND terminal_cause = 'accepted') <> 130
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score') <> 228
     OR (SELECT pg_catalog.count(DISTINCT assignment_id)
         FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score') <> 223
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND attempt = 2) <> 5
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND status = 'accepted' AND terminal_cause = 'accepted') <> 127
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND status = 'failed' AND terminal_cause = 'credential_error') <> 2
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND status = 'failed' AND terminal_cause = 'judge_error') <> 6
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'score'
           AND status = 'failed' AND terminal_cause = 'stage_closed') <> 93
     OR (SELECT pg_catalog.count(DISTINCT assignment_id)
         FROM public.lab_arena_runs
         WHERE round_id = v_round_id
           AND assignment_id LIKE '%' || v_prior_suffix) <> 93
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id
           AND assignment_id LIKE '%' || v_prior_suffix) <> 96
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round_id
         AND status IN ('pending', 'leased', 'submitted')
     )
     OR EXISTS (
       SELECT 1
       FROM public.lab_arena_company_judgment_reservations AS reservation
       JOIN public.lab_arena_runs AS run ON run.run_id = reservation.run_id
       WHERE run.round_id = v_round_id
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-14 run state differs for recovery249';
  END IF;

  IF EXISTS (
       SELECT 1
       FROM pg_catalog.jsonb_array_elements(
         v_round.stage1_scoring_plan_doc -> 'work_items'
       ) AS plan(item)
       WHERE NOT EXISTS (
         SELECT 1
         FROM public.lab_arena_runs AS execution
         WHERE execution.round_id = v_round_id
           AND execution.kind = 'execute'
           AND execution.status = 'accepted'
           AND execution.run_id = plan.item ->> 'scored_run_id'
           AND execution.submission_id = plan.item ->> 'submission_id'
           AND execution.icp_position =
             (plan.item ->> 'icp_position')::SMALLINT
           AND execution.output_ref = plan.item ->> 'output_ref'
       )
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-14 execution plan binding differs for recovery249';
  END IF;

  WITH missing AS (
    SELECT plan.item
    FROM pg_catalog.jsonb_array_elements(
      v_round.stage1_scoring_plan_doc -> 'work_items'
    ) AS plan(item)
    WHERE NOT EXISTS (
      SELECT 1
      FROM public.lab_arena_runs AS accepted
      WHERE accepted.round_id = v_round_id
        AND accepted.kind = 'score'
        AND accepted.status = 'accepted'
        AND accepted.submission_id = plan.item ->> 'submission_id'
        AND accepted.icp_position =
          (plan.item ->> 'icp_position')::SMALLINT
        AND accepted.scored_run_id = plan.item ->> 'scored_run_id'
    )
  )
  SELECT pg_catalog.count(*),
    pg_catalog.array_agg(
      (item ->> 'icp_position')::INTEGER
      ORDER BY (item ->> 'icp_position')::INTEGER
    ),
    pg_catalog.array_agg(
      item ->> 'submission_id'
      ORDER BY (item ->> 'icp_position')::INTEGER
    )
  INTO v_count, v_missing_positions, v_missing_submissions
  FROM missing;
  IF v_count <> 3
     OR v_missing_positions IS DISTINCT FROM ARRAY[5, 6, 8]::INTEGER[]
     OR v_missing_submissions IS DISTINCT FROM ARRAY[
       v_target_submission, v_target_submission, v_target_submission
     ]::TEXT[] THEN
    RAISE EXCEPTION
      'arena 2026-09-14 unfinished logical set differs for recovery249: count %, positions %, submissions %',
      v_count, v_missing_positions, v_missing_submissions;
  END IF;

  -- Each missing item must end in a failed recovery244 assignment. Its frozen
  -- input, cache, company references, and scored execution are copied below.
  IF (WITH latest_missing AS (
        SELECT DISTINCT ON (
          run.submission_id, run.icp_position, run.scored_run_id
        ) run.*
        FROM public.lab_arena_runs AS run
        WHERE run.round_id = v_round_id
          AND run.stage = 1
          AND run.kind = 'score'
          AND run.submission_id = v_target_submission
          AND run.icp_position IN (5, 6, 8)
        ORDER BY run.submission_id, run.icp_position, run.scored_run_id,
          run.stage_generation DESC, run.attempt DESC,
          run.created_at DESC, run.run_id DESC
      )
      SELECT pg_catalog.count(*) FROM latest_missing
      WHERE status = 'failed'
        AND terminal_cause = 'stage_closed'
        AND assignment_id LIKE '%' || v_prior_suffix
        AND stage_generation = 5) <> 3 THEN
    RAISE EXCEPTION 'arena 2026-09-14 prior recovery binding differs for recovery249';
  END IF;

  -- Snapshot all state that this migration must not change.
  SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(run)::TEXT), '|' ORDER BY run_id
         )) INTO v_runs_hash
  FROM public.lab_arena_runs AS run
  WHERE round_id = v_round_id;
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(ledger)::TEXT), '|' ORDER BY entry_id
         ), '')) INTO v_ledger_hash
  FROM public.lab_arena_ledger AS ledger
  WHERE round_id = v_round_id;
  SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(submission)::TEXT),
           '|' ORDER BY submission_id
         )) INTO v_submission_hash
  FROM public.lab_arena_submissions AS submission
  WHERE round_id = v_round_id;
  SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(credential)::TEXT),
           '|' ORDER BY submission_id, provider
         )) INTO v_credential_hash
  FROM public.lab_arena_submission_credentials AS credential
  WHERE submission_id = ANY(v_participant_ids);
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(cache_row)::TEXT),
           '|' ORDER BY cache_key
         ), '')) INTO v_cache_hash
  FROM public.lab_arena_judgment_cache AS cache_row
  WHERE scope_doc ->> 'round_id' = v_round_id;
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(reservation)::TEXT),
           '|' ORDER BY reservation.cache_key, reservation.authority_slot
         ), '')) INTO v_reservation_hash
  FROM public.lab_arena_company_judgment_reservations AS reservation
  JOIN public.lab_arena_runs AS run ON run.run_id = reservation.run_id
  WHERE run.round_id = v_round_id;
  SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(other_round)::TEXT),
           '|' ORDER BY round_id
         ), '')) INTO v_unrelated_hash
  FROM public.lab_arena_rounds AS other_round
  WHERE round_id <> v_round_id;
  v_round_stable := pg_catalog.to_jsonb(v_round)
    - 'status' - 'status_generation' - 'stage_generation'
    - 'cancel_reason' - 'configuration_doc' - 'updated_at';
  v_configuration_stable := v_round.configuration_doc - 'schedule';

  WITH latest_missing AS (
    SELECT DISTINCT ON (
      run.submission_id, run.icp_position, run.scored_run_id
    ) run.*
    FROM public.lab_arena_runs AS run
    WHERE run.round_id = v_round_id
      AND run.stage = 1
      AND run.kind = 'score'
      AND run.submission_id = v_target_submission
      AND run.icp_position IN (5, 6, 8)
    ORDER BY run.submission_id, run.icp_position, run.scored_run_id,
      run.stage_generation DESC, run.attempt DESC,
      run.created_at DESC, run.run_id DESC
  )
  INSERT INTO public.lab_arena_runs (
    run_id, assignment_id, round_id, submission_id, miner_hotkey, stage,
    icp_position, attempt, status, lease_generation, stage_generation, kind,
    scored_run_id, previous_runner_hotkey, judgment_cache_key,
    judgment_input_hash, judgment_scope_doc, judgment_group_leader,
    judgment_group_miner_hotkeys, company_judgment_refs
  )
  SELECT new_assignment || ':1', new_assignment, round_id, submission_id,
    miner_hotkey, stage, icp_position, 1, 'pending', 0, 8, kind,
    scored_run_id, NULL, judgment_cache_key, judgment_input_hash,
    judgment_scope_doc, FALSE, judgment_group_miner_hotkeys,
    company_judgment_refs
  FROM latest_missing
  CROSS JOIN LATERAL (
    SELECT v_round_id || ':' || latest_missing.submission_id || ':1:'
      || latest_missing.icp_position::TEXT || ':score'
      || v_recovery_suffix AS new_assignment
  ) AS recovered;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 3 THEN
    RAISE EXCEPTION 'arena 2026-09-14 recovery249 insert count differs';
  END IF;

  WITH groups AS (
    SELECT judgment_cache_key, pg_catalog.min(run_id) AS leader_run_id,
      pg_catalog.array_agg(DISTINCT miner_hotkey ORDER BY miner_hotkey)
        AS miner_hotkeys
    FROM public.lab_arena_runs
    WHERE round_id = v_round_id
      AND assignment_id LIKE '%' || v_recovery_suffix
    GROUP BY judgment_cache_key
  )
  UPDATE public.lab_arena_runs AS run
  SET judgment_group_leader = run.run_id = groups.leader_run_id,
      judgment_group_miner_hotkeys = groups.miner_hotkeys
  FROM groups
  WHERE run.round_id = v_round_id
    AND run.assignment_id LIKE '%' || v_recovery_suffix
    AND run.judgment_cache_key = groups.judgment_cache_key;

  v_stage1_scoring_close := pg_catalog.transaction_timestamp()
    + INTERVAL '6 hours 30 minutes';
  ALTER TABLE public.lab_arena_rounds
    DISABLE TRIGGER lab_arena_rounds_write_once;
  UPDATE public.lab_arena_rounds
  SET status = 'stage1_scoring',
      status_generation = 9,
      stage_generation = 8,
      cancel_reason = NULL,
      configuration_doc = pg_catalog.jsonb_set(
        configuration_doc,
        '{schedule}',
        configuration_doc -> 'schedule' || pg_catalog.jsonb_build_object(
          'stage_1_scoring_close', pg_catalog.to_char(
            v_stage1_scoring_close AT TIME ZONE 'UTC',
            'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
          ),
          'stage_2_start', pg_catalog.to_char(
            (v_stage1_scoring_close + INTERVAL '1 second') AT TIME ZONE 'UTC',
            'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
          ),
          'stage_2_close', pg_catalog.to_char(
            (v_stage1_scoring_close + INTERVAL '3 hours 1 second')
              AT TIME ZONE 'UTC',
            'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
          ),
          'final_scoring_close', pg_catalog.to_char(
            (v_stage1_scoring_close + INTERVAL '9 hours 30 minutes 1 second')
              AT TIME ZONE 'UTC',
            'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
          ),
          'stage_3_start', pg_catalog.to_char(
            (v_stage1_scoring_close + INTERVAL '9 hours 30 minutes 2 seconds')
              AT TIME ZONE 'UTC',
            'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
          ),
          'stage_3_close', pg_catalog.to_char(
            (v_stage1_scoring_close + INTERVAL '10 hours 30 minutes 2 seconds')
              AT TIME ZONE 'UTC',
            'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
          ),
          'stage_3_scoring_close', pg_catalog.to_char(
            (v_stage1_scoring_close + INTERVAL '12 hours 20 minutes 2 seconds')
              AT TIME ZONE 'UTC',
            'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
          ),
          'publication_deadline', pg_catalog.to_char(
            (v_stage1_scoring_close + INTERVAL '12 hours 20 minutes 3 seconds')
              AT TIME ZONE 'UTC',
            'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
          )
        ),
        FALSE
      )
  WHERE round_id = v_round_id
    AND status = 'cancelled'
    AND status_generation = 8
    AND stage_generation = 7;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  ALTER TABLE public.lab_arena_rounds
    ENABLE TRIGGER lab_arena_rounds_write_once;
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'arena 2026-09-14 recovery249 transition failed';
  END IF;

  -- Verify that the transaction changed only the round state, future schedule,
  -- and three new pending rows with the copied frozen bindings.
  IF (SELECT pg_catalog.array_agg(assignment_id ORDER BY assignment_id)
      FROM public.lab_arena_runs
      WHERE round_id = v_round_id
        AND assignment_id LIKE '%' || v_recovery_suffix
        AND attempt = 1 AND status = 'pending'
        AND run_id = assignment_id || ':1'
        AND stage_generation = 8) IS DISTINCT FROM v_target_assignments
     OR EXISTS (
       SELECT 1
       FROM public.lab_arena_runs AS recovered
       JOIN LATERAL (
         SELECT prior.*
         FROM public.lab_arena_runs AS prior
         WHERE prior.round_id = v_round_id
           AND prior.submission_id = recovered.submission_id
           AND prior.icp_position = recovered.icp_position
           AND prior.scored_run_id = recovered.scored_run_id
           AND prior.assignment_id LIKE '%' || v_prior_suffix
         ORDER BY prior.stage_generation DESC, prior.attempt DESC,
           prior.created_at DESC, prior.run_id DESC
         LIMIT 1
       ) AS prior ON TRUE
       WHERE recovered.round_id = v_round_id
         AND recovered.assignment_id LIKE '%' || v_recovery_suffix
         AND (
           recovered.judgment_cache_key IS DISTINCT FROM prior.judgment_cache_key
           OR recovered.judgment_input_hash IS DISTINCT FROM prior.judgment_input_hash
           OR recovered.judgment_scope_doc IS DISTINCT FROM prior.judgment_scope_doc
           OR recovered.company_judgment_refs IS DISTINCT FROM prior.company_judgment_refs
         )
     )
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(run)::TEXT), '|' ORDER BY run_id
         )) FROM public.lab_arena_runs AS run
         WHERE round_id = v_round_id
           AND assignment_id NOT LIKE '%' || v_recovery_suffix)
          IS DISTINCT FROM v_runs_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(ledger)::TEXT), '|' ORDER BY entry_id
         ), '')) FROM public.lab_arena_ledger AS ledger
         WHERE round_id = v_round_id) IS DISTINCT FROM v_ledger_hash
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(submission)::TEXT),
           '|' ORDER BY submission_id
         )) FROM public.lab_arena_submissions AS submission
         WHERE round_id = v_round_id) IS DISTINCT FROM v_submission_hash
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(credential)::TEXT),
           '|' ORDER BY submission_id, provider
         )) FROM public.lab_arena_submission_credentials AS credential
         WHERE submission_id = ANY(v_participant_ids))
          IS DISTINCT FROM v_credential_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(cache_row)::TEXT),
           '|' ORDER BY cache_key
         ), '')) FROM public.lab_arena_judgment_cache AS cache_row
         WHERE scope_doc ->> 'round_id' = v_round_id)
          IS DISTINCT FROM v_cache_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(reservation)::TEXT),
           '|' ORDER BY reservation.cache_key, reservation.authority_slot
         ), ''))
         FROM public.lab_arena_company_judgment_reservations AS reservation
         JOIN public.lab_arena_runs AS run ON run.run_id = reservation.run_id
         WHERE run.round_id = v_round_id) IS DISTINCT FROM v_reservation_hash
     OR (SELECT pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.md5(pg_catalog.to_jsonb(other_round)::TEXT),
           '|' ORDER BY round_id
         ), '')) FROM public.lab_arena_rounds AS other_round
         WHERE round_id <> v_round_id) IS DISTINCT FROM v_unrelated_hash
     OR (SELECT pg_catalog.to_jsonb(current_round)
           - 'status' - 'status_generation' - 'stage_generation'
           - 'cancel_reason' - 'configuration_doc' - 'updated_at'
         FROM public.lab_arena_rounds AS current_round
         WHERE round_id = v_round_id) IS DISTINCT FROM v_round_stable
     OR (SELECT configuration_doc - 'schedule'
         FROM public.lab_arena_rounds
         WHERE round_id = v_round_id) IS DISTINCT FROM v_configuration_stable THEN
    RAISE EXCEPTION 'arena 2026-09-14 recovery249 verification failed';
  END IF;
END;
$lab_arena_249_recover_20260914_remaining_scoring$;

COMMIT;
