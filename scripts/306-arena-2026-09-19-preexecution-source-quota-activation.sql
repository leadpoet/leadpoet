-- 306: Activate the reviewed Sep19 baseline archive and 200-call sourcing quotas.
-- This transition is valid only before any Sep19 execution was claimed. It
-- preserves the frozen benchmark, all miner submissions, all 100 pending runs,
-- and the completed code-review ledger. Sep20 receives only the quota update.
--
-- External object preconditions, verified by the protected read/apply helper:
--   new source ref: arena/arena-2026-09-19/sources/baseline-2026-09-19-preexecution306.tar.gz
--   new source sha256: 9ad94f534ffcecd6c43a554d79ec6f30c6f40d8d3892277dcc04a415f9bb5f38
--   new source size: 673106 bytes
--   new source commit: 66459d982ad43106a9186339cd42558594abb99a
--   old source sha256: 780d959564bd075fd4ead854c2878f9f4581cebdf73d1a83f9d4d7a3b4252ef0
--   old source commit: e8572c97f5b69b0bd39ebad8edc47f91df2bfe59
--   benchmark sha256: ef951b631d1edb592587f5a19bd7e48d75548aceee4b1a22b3100d851abdabb5
--   benchmark size: 30119 bytes
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
  v_count BIGINT;
  v_old_quotas CONSTANT JSONB :=
    '{"deepline":30,"openrouter":200,"scrapingdog":30}'::JSONB;
  v_new_quotas CONSTANT JSONB :=
    '{"deepline":200,"openrouter":200,"scrapingdog":200}'::JSONB;
  v_scoring_quotas CONSTANT JSONB :=
    '{"deepline":40,"openrouter":120,"scrapingdog":150}'::JSONB;
  v_old_source_ref CONSTANT TEXT :=
    'arena/arena-2026-09-19/sources/baseline-2026-09-19.tar.gz';
  v_old_source_size CONSTANT BIGINT := 673162;
  v_old_source_sha CONSTANT TEXT :=
    '780d959564bd075fd4ead854c2878f9f4581cebdf73d1a83f9d4d7a3b4252ef0';
  v_old_source_commit CONSTANT TEXT :=
    'e8572c97f5b69b0bd39ebad8edc47f91df2bfe59';
  v_new_source_ref CONSTANT TEXT :=
    'arena/arena-2026-09-19/sources/baseline-2026-09-19-preexecution306.tar.gz';
  v_new_source_size CONSTANT BIGINT := 673106;
  v_new_source_sha CONSTANT TEXT := '9ad94f534ffcecd6c43a554d79ec6f30c6f40d8d3892277dcc04a415f9bb5f38';
  v_new_source_commit CONSTANT TEXT := '66459d982ad43106a9186339cd42558594abb99a';
  v_expected_submission_doc JSONB;
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

  -- Harmless replay after the complete atomic transition.
  IF v_round19.configuration_doc -> 'call_quotas' = v_new_quotas
     AND v_round20.configuration_doc -> 'call_quotas' = v_new_quotas
     AND v_baseline.source_ref = v_new_source_ref
     AND v_baseline.source_size_bytes = v_new_source_size
     AND v_baseline.submission_doc ->> 'source_sha256' = v_new_source_sha
     AND v_baseline.submission_doc ->> 'source_commit' = v_new_source_commit
     AND EXISTS (
       SELECT 1 FROM pg_catalog.jsonb_array_elements(v_round19.participants) item
       WHERE item ->> 'submission_id' = 'baseline-2026-09-19'
         AND item ->> 'source_ref' = v_new_source_ref
         AND (item ->> 'source_size_bytes')::BIGINT = v_new_source_size
     ) THEN
    RETURN;
  END IF;

  IF v_round19.status IS DISTINCT FROM 'stage1'
     OR v_round19.configuration_doc ->> 'schema_version'
          IS DISTINCT FROM 'leadpoet.lab_arena.round_configuration.v1'
     OR v_round19.configuration_doc ->> 'round_id' IS DISTINCT FROM 'arena-2026-09-19'
     OR v_round19.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
     OR v_round19.configuration_doc -> 'call_quotas' IS DISTINCT FROM v_old_quotas
     OR v_round19.configuration_doc -> 'scoring_call_quotas'
          IS DISTINCT FROM v_scoring_quotas
     OR v_round19.configuration_doc ->> 'sourcing_cost_eligibility_policy'
          IS DISTINCT FROM 'successful_calls_per_icp_v1'
     OR (v_round19.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
          IS DISTINCT FROM 4000000
     OR (v_round19.configuration_doc ->> 'cost_per_company_microusd')::BIGINT
          IS DISTINCT FROM 800000
     OR v_round19.benchmark_ref IS DISTINCT FROM 'arena/arena-2026-09-19/benchmark.json'
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
     OR v_baseline.status IS DISTINCT FROM 'frozen' OR v_baseline.is_king IS NOT TRUE
     OR v_baseline.source_ref IS DISTINCT FROM v_old_source_ref
     OR v_baseline.source_size_bytes IS DISTINCT FROM v_old_source_size
     OR v_baseline.submission_doc ->> 'source_ref' IS DISTINCT FROM v_old_source_ref
     OR (v_baseline.submission_doc ->> 'source_size_bytes')::BIGINT IS DISTINCT FROM v_old_source_size
     OR v_baseline.submission_doc ? 'source_sha256'
     OR v_baseline.submission_doc ? 'source_commit'
     OR v_baseline.submission_doc ? 'preexecution_source_override' THEN
    RAISE EXCEPTION 'Sep19 preexecution source state differs'
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
      WHERE round_id = 'arena-2026-09-19') <> 100
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-19' AND stage = 1) <> 50
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-19' AND stage = 2) <> 50
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = 'arena-2026-09-19'
         AND (kind IS DISTINCT FROM 'execute' OR status IS DISTINCT FROM 'pending'
           OR attempt IS DISTINCT FROM 1
           OR icp_position NOT BETWEEN 0 AND 19
           OR stage IS DISTINCT FROM CASE WHEN icp_position < 10 THEN 1 ELSE 2 END
           OR lease_generation IS DISTINCT FROM 0
           OR stage_generation IS DISTINCT FROM v_round19.stage_generation
           OR runner_hotkey IS NOT NULL OR lease_token_hash IS NOT NULL
           OR lease_expires_at IS NOT NULL OR claim_request_id IS NOT NULL
           OR claim_request_hash IS NOT NULL OR claim_response IS NOT NULL
           OR result_doc IS NOT NULL OR output_ref IS NOT NULL
           OR terminal_cause IS NOT NULL OR terminal_doc IS NOT NULL
           OR per_icp_score IS NOT NULL OR qualification_doc IS NOT NULL)
     ) OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs run
       LEFT JOIN public.lab_arena_submissions submission
         ON submission.round_id = run.round_id
        AND submission.submission_id = run.submission_id
       WHERE run.round_id = 'arena-2026-09-19'
         AND submission.submission_id IS NULL
     ) OR EXISTS (
       SELECT 1 FROM (
         SELECT submission_id, pg_catalog.count(*) AS n,
                pg_catalog.count(DISTINCT icp_position) AS positions
         FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-19'
         GROUP BY submission_id
       ) counts WHERE n <> 20 OR positions <> 20
     ) THEN
    RAISE EXCEPTION 'Sep19 execution was dispatched or run plan differs'
      USING ERRCODE = '55000';
  END IF;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
      WHERE round_id = 'arena-2026-09-19') <> 12
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_ledger
       WHERE round_id = 'arena-2026-09-19'
         AND (run_id IS NOT NULL OR operation_id IS DISTINCT FROM 'openrouter.code_review')
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
     OR v_round20.configuration_doc ->> 'round_id' IS DISTINCT FROM 'arena-2026-09-20'
     OR v_round20.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
     OR v_round20.configuration_doc -> 'call_quotas' IS DISTINCT FROM v_old_quotas
     OR v_round20.configuration_doc -> 'scoring_call_quotas'
          IS DISTINCT FROM v_scoring_quotas
     OR v_round20.configuration_doc ->> 'sourcing_cost_eligibility_policy'
          IS DISTINCT FROM 'successful_calls_per_icp_v1'
     OR (v_round20.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
          IS DISTINCT FROM 4000000
     OR (v_round20.configuration_doc ->> 'cost_per_company_microusd')::BIGINT
          IS DISTINCT FROM 800000
     OR v_round20.participants IS NOT NULL OR v_round20.benchmark_ref IS NOT NULL
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
                WHERE round_id = 'arena-2026-09-20') THEN
    RAISE EXCEPTION 'Sep20 open quota state differs'
      USING ERRCODE = '55000';
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
      'preexecution_source_override', pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.lab_arena.preexecution_source_override.v1',
        'migration', '306-arena-2026-09-19-preexecution-source-quota-activation',
        'previous_source_ref', v_old_source_ref,
        'previous_source_size_bytes', v_old_source_size,
        'previous_source_sha256', v_old_source_sha,
        'previous_source_commit', v_old_source_commit
      )
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
  SET configuration_doc = pg_catalog.jsonb_set(
        configuration_doc, '{call_quotas}', v_new_quotas, FALSE),
      participants = v_new_participants,
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = 'arena-2026-09-19';
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'Sep19 round update count differs';
  END IF;

  UPDATE public.lab_arena_rounds
  SET configuration_doc = pg_catalog.jsonb_set(
        configuration_doc, '{call_quotas}', v_new_quotas, FALSE),
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = 'arena-2026-09-20';
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'Sep20 round update count differs';
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

  IF (pg_catalog.to_jsonb(v_round19_after) - 'configuration_doc' - 'participants' - 'updated_at')
       IS DISTINCT FROM
       (pg_catalog.to_jsonb(v_round19) - 'configuration_doc' - 'participants' - 'updated_at')
     OR (v_round19_after.configuration_doc - 'call_quotas')
       IS DISTINCT FROM (v_round19.configuration_doc - 'call_quotas')
     OR v_round19_after.configuration_doc -> 'call_quotas' IS DISTINCT FROM v_new_quotas
     OR v_round19_after.participants IS DISTINCT FROM v_new_participants
     OR (pg_catalog.to_jsonb(v_round20_after) - 'configuration_doc' - 'updated_at')
       IS DISTINCT FROM (pg_catalog.to_jsonb(v_round20) - 'configuration_doc' - 'updated_at')
     OR (v_round20_after.configuration_doc - 'call_quotas')
       IS DISTINCT FROM (v_round20.configuration_doc - 'call_quotas')
     OR v_round20_after.configuration_doc -> 'call_quotas' IS DISTINCT FROM v_new_quotas
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
