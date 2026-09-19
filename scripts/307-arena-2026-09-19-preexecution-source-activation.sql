-- 307: Activate the reviewed Sep19 baseline archive before execution.
-- This source-only transition is valid only while every Sep19 execution run
-- remains pending and the ledger contains only the completed code reviews.
-- It preserves both round configurations, the frozen bank, all miner sources,
-- all 100 run rows, all 12 ledger rows, and all Sep20 state.
--
-- External object preconditions, verified by the protected read/apply helper:
--   new source ref: arena/arena-2026-09-19/sources/baseline-2026-09-19-preexecution307.tar.gz
--   new source sha256: b3624778756db5460e456fd054c3aae28d5b8e51f7b9213a2f60bf19a25a4196
--   new source size: 678209 bytes
--   new source commit: f3fd3acc10bc58a95c0728c61fe0fc044cf57303
--   old source sha256: 9ad94f534ffcecd6c43a554d79ec6f30c6f40d8d3892277dcc04a415f9bb5f38
--   old source size: 673106 bytes
--   old source commit: 66459d982ad43106a9186339cd42558594abb99a
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_ledger IN SHARE ROW EXCLUSIVE MODE;

DO $activate_sep19_preexecution$
DECLARE
  v_round19 public.lab_arena_rounds;
  v_round20 public.lab_arena_rounds;
  v_baseline public.lab_arena_submissions;
  v_baseline_after public.lab_arena_submissions;
  v_round19_after public.lab_arena_rounds;
  v_round20_after public.lab_arena_rounds;
  v_new_participants JSONB;
  v_runs_before JSONB;
  v_runs_after JSONB;
  v_ledger_before JSONB;
  v_ledger_after JSONB;
  v_miners19_before JSONB;
  v_miners19_after JSONB;
  v_submissions20_before JSONB;
  v_submissions20_after JSONB;
  v_expected_submission_doc JSONB;
  v_old_state BOOLEAN;
  v_new_state BOOLEAN;
  v_count BIGINT;
  v_call_quotas CONSTANT JSONB :=
    '{"deepline":200,"openrouter":200,"scrapingdog":200}'::JSONB;
  v_scoring_quotas CONSTANT JSONB :=
    '{"deepline":40,"openrouter":120,"scrapingdog":150}'::JSONB;
  v_old_source_ref CONSTANT TEXT :=
    'arena/arena-2026-09-19/sources/baseline-2026-09-19-preexecution306.tar.gz';
  v_old_source_size CONSTANT BIGINT := 673106;
  v_old_source_sha CONSTANT TEXT :=
    '9ad94f534ffcecd6c43a554d79ec6f30c6f40d8d3892277dcc04a415f9bb5f38';
  v_old_source_commit CONSTANT TEXT :=
    '66459d982ad43106a9186339cd42558594abb99a';
  v_old_override CONSTANT JSONB := pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.preexecution_source_override.v1',
    'migration', '306-arena-2026-09-19-preexecution-source-quota-activation',
    'previous_source_ref',
      'arena/arena-2026-09-19/sources/baseline-2026-09-19.tar.gz',
    'previous_source_size_bytes', 673162,
    'previous_source_sha256',
      '780d959564bd075fd4ead854c2878f9f4581cebdf73d1a83f9d4d7a3b4252ef0',
    'previous_source_commit', 'e8572c97f5b69b0bd39ebad8edc47f91df2bfe59'
  );
  v_new_source_ref CONSTANT TEXT :=
    'arena/arena-2026-09-19/sources/baseline-2026-09-19-preexecution307.tar.gz';
  v_new_source_size CONSTANT BIGINT := 678209;
  v_new_source_sha CONSTANT TEXT :=
    'b3624778756db5460e456fd054c3aae28d5b8e51f7b9213a2f60bf19a25a4196';
  v_new_source_commit CONSTANT TEXT :=
    'f3fd3acc10bc58a95c0728c61fe0fc044cf57303';
  v_new_override CONSTANT JSONB := pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.preexecution_source_override.v1',
    'migration', '307-arena-2026-09-19-preexecution-source-activation',
    'previous_source_ref', v_old_source_ref,
    'previous_source_size_bytes', v_old_source_size,
    'previous_source_sha256', v_old_source_sha,
    'previous_source_commit', v_old_source_commit
  );
BEGIN
  IF NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
         AND tgname = 'lab_arena_rounds_write_once'
         AND NOT tgisinternal AND tgenabled = 'O'
     ) OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_submissions'::pg_catalog.regclass
         AND tgname = 'lab_arena_submissions_frozen'
         AND NOT tgisinternal AND tgenabled = 'O'
     ) THEN
    RAISE EXCEPTION 'Arena immutability trigger state differs'
      USING ERRCODE = '55000';
  END IF;

  IF v_new_source_size NOT BETWEEN 1 AND 10485760
     OR v_new_source_sha !~ '^[0-9a-f]{64}$'
     OR v_new_source_commit !~ '^[0-9a-f]{40}$'
     OR v_new_source_sha = v_old_source_sha
     OR v_new_source_commit = v_old_source_commit THEN
    RAISE EXCEPTION 'Sep19 replacement source facts are invalid'
      USING ERRCODE = '22023';
  END IF;

  SELECT * INTO v_round19 FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-19' FOR UPDATE;
  SELECT * INTO v_round20 FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-20' FOR UPDATE;
  SELECT * INTO v_baseline FROM public.lab_arena_submissions
  WHERE round_id = 'arena-2026-09-19'
    AND submission_id = 'baseline-2026-09-19' FOR UPDATE;
  IF v_round19.round_id IS NULL OR v_round20.round_id IS NULL
     OR v_baseline.submission_id IS NULL THEN
    RAISE EXCEPTION 'Sep19/Sep20 activation row is missing'
      USING ERRCODE = 'P0002';
  END IF;

  IF v_round19.status IS DISTINCT FROM 'stage1'
     OR v_round19.configuration_doc ->> 'schema_version'
          IS DISTINCT FROM 'leadpoet.lab_arena.round_configuration.v1'
     OR v_round19.configuration_doc ->> 'round_id'
          IS DISTINCT FROM 'arena-2026-09-19'
     OR v_round19.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
     OR v_round19.configuration_doc -> 'call_quotas' IS DISTINCT FROM v_call_quotas
     OR v_round19.configuration_doc -> 'scoring_call_quotas'
          IS DISTINCT FROM v_scoring_quotas
     OR v_round19.configuration_doc ->> 'sourcing_cost_eligibility_policy'
          IS DISTINCT FROM 'successful_calls_per_icp_v1'
     OR (v_round19.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
          IS DISTINCT FROM 4000000
     OR (v_round19.configuration_doc ->> 'cost_per_company_microusd')::BIGINT
          IS DISTINCT FROM 800000
     OR v_round19.benchmark_ref
          IS DISTINCT FROM 'arena/arena-2026-09-19/benchmark.json'
     OR v_round19.evaluation_date IS DISTINCT FROM '2026-09-19'
     OR v_round19.icp_set_date IS DISTINCT FROM DATE '2026-09-18'
     OR v_round19.stage1_scoring_plan_doc IS NOT NULL
     OR v_round19.stage2_scoring_plan_doc IS NOT NULL
     OR v_round19.stage3_scoring_plan_doc IS NOT NULL
     OR v_round19.finalists IS NOT NULL
     OR v_round19.publication_doc IS NOT NULL
     OR v_round19.published_at IS NOT NULL
     OR v_round19.cancel_reason IS NOT NULL
     OR pg_catalog.jsonb_typeof(v_round19.participants) IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(v_round19.participants) IS DISTINCT FROM 5
     OR v_baseline.status IS DISTINCT FROM 'frozen'
     OR v_baseline.is_king IS NOT TRUE THEN
    RAISE EXCEPTION 'Sep19 preexecution round state differs'
      USING ERRCODE = '55000';
  END IF;

  v_old_state :=
    v_baseline.source_ref IS NOT DISTINCT FROM v_old_source_ref
    AND v_baseline.source_size_bytes IS NOT DISTINCT FROM v_old_source_size
    AND v_baseline.submission_doc ->> 'source_ref' IS NOT DISTINCT FROM v_old_source_ref
    AND (v_baseline.submission_doc ->> 'source_size_bytes')::BIGINT
      IS NOT DISTINCT FROM v_old_source_size
    AND v_baseline.submission_doc ->> 'source_sha256'
      IS NOT DISTINCT FROM v_old_source_sha
    AND v_baseline.submission_doc ->> 'source_commit'
      IS NOT DISTINCT FROM v_old_source_commit
    AND v_baseline.submission_doc -> 'preexecution_source_override'
      IS NOT DISTINCT FROM v_old_override;
  v_new_state :=
    v_baseline.source_ref IS NOT DISTINCT FROM v_new_source_ref
    AND v_baseline.source_size_bytes IS NOT DISTINCT FROM v_new_source_size
    AND v_baseline.submission_doc ->> 'source_ref' IS NOT DISTINCT FROM v_new_source_ref
    AND (v_baseline.submission_doc ->> 'source_size_bytes')::BIGINT
      IS NOT DISTINCT FROM v_new_source_size
    AND v_baseline.submission_doc ->> 'source_sha256'
      IS NOT DISTINCT FROM v_new_source_sha
    AND v_baseline.submission_doc ->> 'source_commit'
      IS NOT DISTINCT FROM v_new_source_commit
    AND v_baseline.submission_doc -> 'preexecution_source_override'
      IS NOT DISTINCT FROM v_new_override;
  IF NOT COALESCE(v_old_state OR v_new_state, FALSE) THEN
    RAISE EXCEPTION 'Sep19 baseline source state differs'
      USING ERRCODE = '55000';
  END IF;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
      WHERE round_id = 'arena-2026-09-19' AND status = 'frozen') <> 5
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
         WHERE round_id = 'arena-2026-09-19') <> 5
     OR EXISTS (
       SELECT 1
       FROM pg_catalog.jsonb_array_elements(v_round19.participants) item
       LEFT JOIN public.lab_arena_submissions submission
         ON submission.round_id = 'arena-2026-09-19'
        AND submission.submission_id = item ->> 'submission_id'
       WHERE submission.submission_id IS NULL
          OR submission.miner_hotkey IS DISTINCT FROM item ->> 'miner_hotkey'
          OR submission.source_ref IS DISTINCT FROM item ->> 'source_ref'
          OR submission.source_size_bytes IS DISTINCT FROM
             (item ->> 'source_size_bytes')::BIGINT
     ) THEN
    RAISE EXCEPTION 'Sep19 frozen participant state differs'
      USING ERRCODE = '55000';
  END IF;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id = 'arena-2026-09-19'
        AND submission_id = 'baseline-2026-09-19') <> 20
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = 'arena-2026-09-19'
         AND submission_id = 'baseline-2026-09-19'
         AND (status IS DISTINCT FROM 'pending'
           OR lease_generation IS DISTINCT FROM 0
           OR runner_hotkey IS NOT NULL OR lease_token_hash IS NOT NULL
           OR lease_expires_at IS NOT NULL OR claim_request_id IS NOT NULL
           OR claim_request_hash IS NOT NULL OR claim_response IS NOT NULL
           OR result_doc IS NOT NULL OR output_ref IS NOT NULL
           OR terminal_cause IS NOT NULL OR terminal_doc IS NOT NULL
           OR per_icp_score IS NOT NULL OR qualification_doc IS NOT NULL
           OR participation_accepted_at IS NOT NULL
           OR scored_run_id IS NOT NULL OR previous_runner_hotkey IS NOT NULL
           OR judgment_cache_key IS NOT NULL OR judgment_input_hash IS NOT NULL
           OR judgment_scope_doc IS NOT NULL OR judgment_group_leader IS NOT NULL
           OR judgment_group_miner_hotkeys IS NOT NULL
           OR judgment_cache_source_run_id IS NOT NULL
           OR company_judgment_refs IS NOT NULL
           OR champion_funding_sources IS NOT NULL
           OR champion_restart_required IS NOT FALSE)
     ) THEN
    RAISE EXCEPTION 'Sep19 baseline execution was claimed'
      USING ERRCODE = '55000';
  END IF;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id = 'arena-2026-09-19') <> 100
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-19' AND stage = 1) <> 50
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-19' AND stage = 2) <> 50
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-19' AND status = 'accepted') <> 6
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-19' AND status = 'pending') <> 94
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = 'arena-2026-09-19'
         AND (kind IS DISTINCT FROM 'execute'
           OR status NOT IN ('pending', 'accepted')
           OR attempt IS DISTINCT FROM 1
           OR icp_position NOT BETWEEN 0 AND 19
           OR stage IS DISTINCT FROM CASE WHEN icp_position < 10 THEN 1 ELSE 2 END
           OR stage_generation IS DISTINCT FROM v_round19.stage_generation)
     ) OR EXISTS (
       SELECT 1 FROM (
         SELECT submission_id, pg_catalog.count(*) AS n,
                pg_catalog.count(DISTINCT icp_position) AS positions
         FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-19'
         GROUP BY submission_id
       ) counts WHERE n <> 20 OR positions <> 20
     ) THEN
    RAISE EXCEPTION 'Sep19 execution state or run plan differs'
      USING ERRCODE = '55000';
  END IF;

  IF EXISTS (
       SELECT 1
       FROM public.lab_arena_ledger entry
       LEFT JOIN public.lab_arena_runs run ON run.run_id = entry.run_id
       WHERE entry.round_id = 'arena-2026-09-19'
         AND (entry.submission_id = 'baseline-2026-09-19'
           OR run.submission_id = 'baseline-2026-09-19'
           OR run.kind = 'score'
           OR entry.operation_id IS DISTINCT FROM 'openrouter.code_review')
     ) THEN
    RAISE EXCEPTION 'Sep19 baseline or scoring ledger differs'
      USING ERRCODE = '55000';
  END IF;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
      WHERE round_id = 'arena-2026-09-19') <> 12
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_ledger
       WHERE round_id = 'arena-2026-09-19'
         AND run_id IS NOT NULL
     ) OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
           WHERE round_id = 'arena-2026-09-19' AND entry_kind = 'reservation') <> 4
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE round_id = 'arena-2026-09-19' AND entry_kind = 'dispatch') <> 4
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE round_id = 'arena-2026-09-19' AND entry_kind = 'settlement') <> 4 THEN
    RAISE EXCEPTION 'Sep19 code-review ledger or execution ledger differs'
      USING ERRCODE = '55000';
  END IF;

  IF v_round20.status IS DISTINCT FROM 'open'
     OR v_round20.configuration_doc ->> 'schema_version'
          IS DISTINCT FROM 'leadpoet.lab_arena.round_configuration.v1'
     OR v_round20.configuration_doc ->> 'round_id'
          IS DISTINCT FROM 'arena-2026-09-20'
     OR v_round20.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
     OR v_round20.configuration_doc -> 'call_quotas' IS DISTINCT FROM v_call_quotas
     OR v_round20.configuration_doc -> 'scoring_call_quotas'
          IS DISTINCT FROM v_scoring_quotas
     OR v_round20.configuration_doc ->> 'sourcing_cost_eligibility_policy'
          IS DISTINCT FROM 'successful_calls_per_icp_v1'
     OR (v_round20.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
          IS DISTINCT FROM 4000000
     OR (v_round20.configuration_doc ->> 'cost_per_company_microusd')::BIGINT
          IS DISTINCT FROM 800000
     OR v_round20.participants IS NOT NULL
     OR v_round20.benchmark_ref IS NOT NULL
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
                WHERE round_id = 'arena-2026-09-20') THEN
    RAISE EXCEPTION 'Sep20 open state differs'
      USING ERRCODE = '55000';
  END IF;

  -- A replay is harmless only while every preexecution guard still holds.
  IF v_new_state THEN
    RETURN;
  END IF;

  SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(run) ORDER BY run_id),'[]'::JSONB)
    INTO v_runs_before FROM public.lab_arena_runs run
    WHERE round_id = 'arena-2026-09-19';
  SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(entry) ORDER BY entry_id),'[]'::JSONB)
    INTO v_ledger_before FROM public.lab_arena_ledger entry
    WHERE round_id = 'arena-2026-09-19';
  SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(submission) ORDER BY submission_id),'[]'::JSONB)
    INTO v_miners19_before FROM public.lab_arena_submissions submission
    WHERE round_id = 'arena-2026-09-19'
      AND submission_id <> 'baseline-2026-09-19';
  SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(submission) ORDER BY submission_id),'[]'::JSONB)
    INTO v_submissions20_before FROM public.lab_arena_submissions submission
    WHERE round_id = 'arena-2026-09-20';

  SELECT pg_catalog.jsonb_agg(
      CASE WHEN item ->> 'submission_id' = 'baseline-2026-09-19'
        THEN item || pg_catalog.jsonb_build_object(
          'source_ref', v_new_source_ref,
          'source_size_bytes', v_new_source_size
        ) ELSE item END ORDER BY ordinal
    ) INTO v_new_participants
  FROM pg_catalog.jsonb_array_elements(v_round19.participants)
       WITH ORDINALITY participant(item, ordinal);
  IF (SELECT pg_catalog.count(*) FROM pg_catalog.jsonb_array_elements(v_new_participants) item
      WHERE item ->> 'submission_id' = 'baseline-2026-09-19'
        AND item ->> 'source_ref' = v_new_source_ref
        AND (item ->> 'source_size_bytes')::BIGINT = v_new_source_size) <> 1 THEN
    RAISE EXCEPTION 'Sep19 baseline participant projection differs'
      USING ERRCODE = '55000';
  END IF;

  v_expected_submission_doc := v_baseline.submission_doc ||
    pg_catalog.jsonb_build_object(
      'source_ref', v_new_source_ref,
      'source_size_bytes', v_new_source_size,
      'source_sha256', v_new_source_sha,
      'source_commit', v_new_source_commit,
      'preexecution_source_override', v_new_override
    );

  ALTER TABLE public.lab_arena_rounds
    DISABLE TRIGGER lab_arena_rounds_write_once;
  ALTER TABLE public.lab_arena_submissions
    DISABLE TRIGGER lab_arena_submissions_frozen;

  UPDATE public.lab_arena_submissions
  SET source_ref = v_new_source_ref,
      source_size_bytes = v_new_source_size,
      submission_doc = v_expected_submission_doc,
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = 'arena-2026-09-19'
    AND submission_id = 'baseline-2026-09-19';
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'Sep19 baseline update count differs';
  END IF;

  UPDATE public.lab_arena_rounds
  SET participants = v_new_participants,
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = 'arena-2026-09-19';
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'Sep19 round update count differs';
  END IF;

  ALTER TABLE public.lab_arena_submissions
    ENABLE TRIGGER lab_arena_submissions_frozen;
  ALTER TABLE public.lab_arena_rounds
    ENABLE TRIGGER lab_arena_rounds_write_once;

  SELECT * INTO v_round19_after FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-19';
  SELECT * INTO v_round20_after FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-20';
  SELECT * INTO v_baseline_after FROM public.lab_arena_submissions
  WHERE round_id = 'arena-2026-09-19'
    AND submission_id = 'baseline-2026-09-19';
  SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(run) ORDER BY run_id),'[]'::JSONB)
    INTO v_runs_after FROM public.lab_arena_runs run
    WHERE round_id = 'arena-2026-09-19';
  SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(entry) ORDER BY entry_id),'[]'::JSONB)
    INTO v_ledger_after FROM public.lab_arena_ledger entry
    WHERE round_id = 'arena-2026-09-19';
  SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(submission) ORDER BY submission_id),'[]'::JSONB)
    INTO v_miners19_after FROM public.lab_arena_submissions submission
    WHERE round_id = 'arena-2026-09-19'
      AND submission_id <> 'baseline-2026-09-19';
  SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(submission) ORDER BY submission_id),'[]'::JSONB)
    INTO v_submissions20_after FROM public.lab_arena_submissions submission
    WHERE round_id = 'arena-2026-09-20';

  IF (pg_catalog.to_jsonb(v_round19_after) - 'participants' - 'updated_at')
       IS DISTINCT FROM (pg_catalog.to_jsonb(v_round19) - 'participants' - 'updated_at')
     OR v_round19_after.participants IS DISTINCT FROM v_new_participants
     OR pg_catalog.to_jsonb(v_round20_after) IS DISTINCT FROM pg_catalog.to_jsonb(v_round20)
     OR (pg_catalog.to_jsonb(v_baseline_after)
         - 'source_ref' - 'source_size_bytes' - 'submission_doc' - 'updated_at')
       IS DISTINCT FROM
       (pg_catalog.to_jsonb(v_baseline)
         - 'source_ref' - 'source_size_bytes' - 'submission_doc' - 'updated_at')
     OR v_baseline_after.source_ref IS DISTINCT FROM v_new_source_ref
     OR v_baseline_after.source_size_bytes IS DISTINCT FROM v_new_source_size
     OR v_baseline_after.submission_doc IS DISTINCT FROM v_expected_submission_doc
     OR v_runs_after IS DISTINCT FROM v_runs_before
     OR v_ledger_after IS DISTINCT FROM v_ledger_before
     OR v_miners19_after IS DISTINCT FROM v_miners19_before
     OR v_submissions20_after IS DISTINCT FROM v_submissions20_before THEN
    RAISE EXCEPTION 'Sep19/Sep20 activation changed an unapproved field'
      USING ERRCODE = '55000';
  END IF;

  IF NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
         AND tgname = 'lab_arena_rounds_write_once'
         AND NOT tgisinternal AND tgenabled = 'O'
     ) OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_submissions'::pg_catalog.regclass
         AND tgname = 'lab_arena_submissions_frozen'
         AND NOT tgisinternal AND tgenabled = 'O'
     ) THEN
    RAISE EXCEPTION 'Arena immutability trigger restore differs'
      USING ERRCODE = '55000';
  END IF;
END;
$activate_sep19_preexecution$;

COMMIT;
