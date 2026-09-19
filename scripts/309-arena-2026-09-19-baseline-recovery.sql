-- One-time Sep19 baseline recovery after terminal stage-1 execution failure.
--
-- Sealed recovery schedule. The function and invocation below carry this
-- same literal so a caller cannot substitute another window.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_prepare_sep19_recovery309_v1(
  p_forward_schedule JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public, extensions
AS $recover_sep19_baseline309$
DECLARE
  round_before public.lab_arena_rounds;
  baseline_before public.lab_arena_submissions;
  round_after public.lab_arena_rounds;
  baseline_after public.lab_arena_submissions;
  archive_round public.lab_arena_rounds;
  archive_baseline public.lab_arena_submissions;
  miner_runs_before JSONB;
  miner_runs_after JSONB;
  miner_ledger_before JSONB;
  miner_ledger_after JSONB;
  other_state_before JSONB;
  other_state_after JSONB;
  new_participants JSONB;
  archive_participants JSONB;
  archive_configuration JSONB;
  expected_submission_doc JSONB;
  baseline_cost_before JSONB;
  baseline_cost_after JSONB;
  archive_cost JSONB;
  recovery_schedule JSONB := p_forward_schedule;
  expected_forward_schedule CONSTANT JSONB := '{"benchmark_deadline":"2026-09-19T06:00:00Z","final_scoring_close":"2026-09-19T18:00:00Z","publication_deadline":"2026-09-19T18:00:01Z","stage_1_close":"2026-09-19T09:00:01Z","stage_1_scoring_close":"2026-09-19T12:00:00Z","stage_1_start":"2026-09-19T06:00:01Z","stage_2_close":"2026-09-19T15:00:01Z","stage_2_start":"2026-09-19T12:00:01Z","submission_cutoff":"2026-09-19T00:00:00Z","submission_open":"2026-09-18T00:00:00Z"}'::JSONB;
  moved_runs BIGINT;
  moved_ledger BIGINT;
  inserted_runs BIGINT;
  unsettled BIGINT;
  position INTEGER;
  stage SMALLINT;
  assignment TEXT;
  old_source_ref CONSTANT TEXT :=
    'arena/arena-2026-09-19/sources/baseline-2026-09-19-preexecution307.tar.gz';
  old_source_size CONSTANT BIGINT := 678209;
  old_source_sha CONSTANT TEXT :=
    'b3624778756db5460e456fd054c3aae28d5b8e51f7b9213a2f60bf19a25a4196';
  old_source_commit CONSTANT TEXT :=
    'f3fd3acc10bc58a95c0728c61fe0fc044cf57303';
  new_source_ref CONSTANT TEXT :=
    'arena/arena-2026-09-19/sources/baseline-2026-09-19-recovery309-latest.tar.gz';
  new_source_size CONSTANT BIGINT := 682790;
  new_source_sha CONSTANT TEXT :=
    '7bbb45bedca037da7d4dfaf75541841acd52a5afbb30fb93499b5c2c61b9a3b9';
  new_source_commit CONSTANT TEXT :=
    '6257c66ac72dd362b4b9e3206848831dd1f5e0e4';
BEGIN
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('arena-2026-09-19-recovery309', 0)
  );
  LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_ledger IN ACCESS EXCLUSIVE MODE;

  IF recovery_schedule IS DISTINCT FROM expected_forward_schedule
     OR pg_catalog.jsonb_typeof(recovery_schedule) IS DISTINCT FROM 'object'
     OR recovery_schedule ? 'submission_open' IS FALSE
     OR recovery_schedule ? 'submission_cutoff' IS FALSE
     OR recovery_schedule ? 'benchmark_deadline' IS FALSE
     OR recovery_schedule ? 'stage_1_start' IS FALSE
     OR recovery_schedule ? 'stage_1_close' IS FALSE
     OR recovery_schedule ? 'stage_1_scoring_close' IS FALSE
     OR recovery_schedule ? 'stage_2_start' IS FALSE
     OR recovery_schedule ? 'stage_2_close' IS FALSE
     OR recovery_schedule ? 'final_scoring_close' IS FALSE
     OR recovery_schedule ? 'publication_deadline' IS FALSE
     OR recovery_schedule ->> 'submission_open'
          IS DISTINCT FROM '2026-09-18T00:00:00Z'
     OR recovery_schedule ->> 'submission_cutoff'
          IS DISTINCT FROM '2026-09-19T00:00:00Z'
     OR (recovery_schedule ->> 'stage_1_start')::TIMESTAMPTZ
          IS DISTINCT FROM
          (recovery_schedule ->> 'benchmark_deadline')::TIMESTAMPTZ
            + INTERVAL '1 second'
     OR (recovery_schedule ->> 'stage_1_close')::TIMESTAMPTZ
          IS DISTINCT FROM
          (recovery_schedule ->> 'stage_1_start')::TIMESTAMPTZ
            + INTERVAL '3 hours'
     OR (recovery_schedule ->> 'stage_1_scoring_close')::TIMESTAMPTZ
          <= (recovery_schedule ->> 'stage_1_close')::TIMESTAMPTZ
     OR (recovery_schedule ->> 'stage_2_start')::TIMESTAMPTZ
          IS DISTINCT FROM
          (recovery_schedule ->> 'stage_1_scoring_close')::TIMESTAMPTZ
            + INTERVAL '1 second'
     OR (recovery_schedule ->> 'stage_2_close')::TIMESTAMPTZ
          <= (recovery_schedule ->> 'stage_2_start')::TIMESTAMPTZ
     OR (recovery_schedule ->> 'final_scoring_close')::TIMESTAMPTZ
          <= (recovery_schedule ->> 'stage_2_close')::TIMESTAMPTZ
     OR (recovery_schedule ->> 'publication_deadline')::TIMESTAMPTZ
          IS DISTINCT FROM
          (recovery_schedule ->> 'final_scoring_close')::TIMESTAMPTZ
            + INTERVAL '1 second'
     THEN
    RAISE EXCEPTION 'Sep19 recovery309 schedule differs'
      USING ERRCODE = '22023';
  END IF;

  SELECT * INTO round_before FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-19' FOR UPDATE;
  SELECT * INTO baseline_before FROM public.lab_arena_submissions
  WHERE round_id = 'arena-2026-09-19'
    AND submission_id = 'baseline-2026-09-19' FOR UPDATE;
  IF round_before.round_id IS NULL OR baseline_before.submission_id IS NULL THEN
    RAISE EXCEPTION 'Sep19 recovery309 rows are missing' USING ERRCODE = 'P0002';
  END IF;

  -- A replay is accepted only when both archive and active states remain exact.
  IF EXISTS (
    SELECT 1 FROM public.lab_arena_rounds
    WHERE round_id = 'arena-2026-09-19-r309archive'
  ) THEN
    SELECT * INTO archive_round FROM public.lab_arena_rounds
    WHERE round_id = 'arena-2026-09-19-r309archive';
    SELECT * INTO archive_baseline FROM public.lab_arena_submissions
    WHERE round_id = 'arena-2026-09-19-r309archive'
      AND submission_id = 'baseline-2026-09-19:r309archive';
    SELECT public.lab_arena__successful_call_cost_state(
      'baseline-2026-09-19', 'execute', NULL
    ) INTO baseline_cost_after;
    SELECT public.lab_arena__successful_call_cost_state(
      'baseline-2026-09-19:r309archive', 'execute', NULL
    ) INTO archive_cost;
    IF round_before.status NOT IN (
         'stage1', 'stage1_closed', 'stage1_scoring', 'stage1_judged',
         'stage1_scored', 'stage2', 'stage2_closed', 'stage2_scoring',
         'stage2_judged', 'scored', 'published', 'cancelled'
       )
       OR baseline_before.source_ref IS DISTINCT FROM new_source_ref
       OR baseline_before.source_size_bytes IS DISTINCT FROM new_source_size
       OR baseline_before.submission_doc ->> 'source_sha256'
            IS DISTINCT FROM new_source_sha
       OR baseline_before.submission_doc ->> 'source_commit'
            IS DISTINCT FROM new_source_commit
       OR baseline_before.submission_doc ? 'preexecution_source_override'
       OR round_before.configuration_doc -> 'schedule'
            IS DISTINCT FROM recovery_schedule
       OR round_before.configuration_doc ->> 'schema_version'
            IS DISTINCT FROM 'leadpoet.lab_arena.round_configuration.v1'
       OR round_before.configuration_doc ->> 'integrity_policy'
            IS DISTINCT FROM 'arena_integrity_v1'
       OR round_before.configuration_doc -> 'call_quotas'
            IS DISTINCT FROM
            '{"deepline":200,"openrouter":200,"scrapingdog":200}'::JSONB
       OR round_before.configuration_doc -> 'scoring_call_quotas'
            IS DISTINCT FROM
            '{"deepline":40,"openrouter":120,"scrapingdog":150}'::JSONB
       OR (SELECT pg_catalog.count(DISTINCT assignment_id)
           FROM public.lab_arena_runs
           WHERE round_id = 'arena-2026-09-19'
             AND submission_id = 'baseline-2026-09-19'
             AND kind = 'execute'
             AND assignment_id LIKE '%:recovery309') <> 20
       OR EXISTS (
         SELECT 1 FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-19'
           AND submission_id = 'baseline-2026-09-19'
           AND kind = 'execute'
           AND assignment_id NOT LIKE '%:recovery309'
       )
       OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
           WHERE round_id = 'arena-2026-09-19'
             AND submission_id <> 'baseline-2026-09-19'
             AND kind = 'execute') <> 80
       OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
           WHERE round_id = 'arena-2026-09-19'
             AND submission_id <> 'baseline-2026-09-19'
             AND kind = 'execute' AND status = 'accepted'
             AND terminal_cause = 'accepted') <> 80
       OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
           WHERE round_id = 'arena-2026-09-19-r309archive'
             AND submission_id = 'baseline-2026-09-19:r309archive'
             AND kind = 'execute') <> 30
       OR archive_round.benchmark_ref IS DISTINCT FROM round_before.benchmark_ref
       OR (
         round_before.configuration_doc - 'schedule'
       ) IS DISTINCT FROM (
         (
           archive_round.configuration_doc
             - 'schedule' - 'archived_reward_basis_hash'
             - 'archived_effective_reward_epoch'
             - 'archived_reward_activated_at' - 'archive_reason'
             - 'archived_baseline_attempt_count'
             - 'archived_baseline_uncertain_failed_calls'
         ) || pg_catalog.jsonb_build_object(
           'round_id', 'arena-2026-09-19',
           'mode', 'live',
           'rewards_enabled', TRUE
         )
       )
       OR archive_baseline.source_ref IS DISTINCT FROM old_source_ref
       OR archive_baseline.source_size_bytes IS DISTINCT FROM old_source_size
       OR archive_baseline.submission_doc ->> 'source_sha256'
            IS DISTINCT FROM old_source_sha
       OR archive_baseline.submission_doc ->> 'source_commit'
            IS DISTINCT FROM old_source_commit
       OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
           WHERE round_id = 'arena-2026-09-19-r309archive') <> 5
       OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
           WHERE round_id = 'arena-2026-09-19-r309archive'
             AND submission_id = 'baseline-2026-09-19:r309archive') <> 6102
       OR COALESCE((archive_cost ->> 'uncertain_calls')::BIGINT, -1) <> 215
       OR COALESCE((archive_cost ->> 'inflight_calls')::BIGINT, -1) <> 0
       OR COALESCE((archive_cost ->> 'success_unresolved_calls')::BIGINT, -1) <> 0
       OR COALESCE((archive_cost ->> 'settled_microusd')::BIGINT, -1) <> 13027527
       OR COALESCE(
            (archive_cost ->> 'reserved_or_uncertain_microusd')::BIGINT, -1
          ) <> 23393484
       OR COALESCE((archive_cost ->> 'refused_calls')::BIGINT, -1) <> 12
       OR COALESCE((archive_cost ->> 'successful_calls')::BIGINT, -1) <> 1804
       THEN
      RAISE EXCEPTION 'existing Sep19 recovery309 differs' USING ERRCODE = '55000';
    END IF;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'existing',
      'round_id', 'arena-2026-09-19',
      'fresh_baseline_execute_count', 20,
      'retained_miner_execute_count', 80,
      'archived_baseline_attempt_count', 30,
      'score_namespace', 'score'
    );
  END IF;

  IF round_before.status IS DISTINCT FROM 'cancelled'
     OR round_before.cancel_reason
          IS DISTINCT FROM 'execution_incomplete:stage1:10'
     OR round_before.benchmark_ref
          IS DISTINCT FROM 'arena/arena-2026-09-19/benchmark.json'
     OR round_before.evaluation_date IS DISTINCT FROM '2026-09-19'
     OR round_before.icp_set_date IS DISTINCT FROM DATE '2026-09-18'
     OR round_before.stage1_scoring_plan_doc IS NOT NULL
     OR round_before.stage2_scoring_plan_doc IS NOT NULL
     OR round_before.stage3_scoring_plan_doc IS NOT NULL
     OR round_before.finalists IS NOT NULL
     OR round_before.publication_doc IS NOT NULL
     OR round_before.published_at IS NOT NULL
     OR pg_catalog.jsonb_typeof(round_before.participants)
          IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(round_before.participants) <> 5
     OR round_before.configuration_doc ->> 'schema_version'
          IS DISTINCT FROM 'leadpoet.lab_arena.round_configuration.v1'
     OR round_before.configuration_doc ->> 'round_id'
          IS DISTINCT FROM 'arena-2026-09-19'
     OR round_before.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
     OR round_before.configuration_doc ->> 'integrity_policy'
          IS DISTINCT FROM 'arena_integrity_v1'
     OR round_before.configuration_doc ->> 'rewards_enabled'
          IS DISTINCT FROM 'true'
     OR round_before.configuration_doc ->> 'scorer_image_digest'
          IS DISTINCT FROM
          'sha256:085e6b586ccac28ca7ec2c8996dd871f1a53050e843398b514f67d8c798aae5c'
     OR (round_before.configuration_doc ->> 'icp_wall_clock_seconds')::INTEGER
          IS DISTINCT FROM 2700
     OR (round_before.configuration_doc ->> 'parallel_twenty_icp_execution')::BOOLEAN
          IS DISTINCT FROM TRUE
     OR (round_before.configuration_doc ->> 'stage_1_icp_count')::INTEGER
          IS DISTINCT FROM 10
     OR (round_before.configuration_doc ->> 'stage_2_icp_count')::INTEGER
          IS DISTINCT FROM 10
     OR pg_catalog.jsonb_typeof(round_before.configuration_doc -> 'schedule')
          IS DISTINCT FROM 'object'
     OR round_before.configuration_doc -> 'call_quotas'
          IS DISTINCT FROM '{"deepline":200,"openrouter":200,"scrapingdog":200}'::JSONB
     OR round_before.configuration_doc -> 'scoring_call_quotas'
          IS DISTINCT FROM '{"deepline":40,"openrouter":120,"scrapingdog":150}'::JSONB
     OR round_before.configuration_doc ->> 'sourcing_cost_eligibility_policy'
          IS DISTINCT FROM 'successful_calls_per_icp_v1'
     OR (round_before.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
          IS DISTINCT FROM 4000000
     OR (round_before.configuration_doc ->> 'cost_per_company_microusd')::BIGINT
          IS DISTINCT FROM 800000
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
         WHERE round_id = 'arena-2026-09-19' AND status = 'frozen') <> 5
     OR baseline_before.status IS DISTINCT FROM 'frozen'
     OR baseline_before.is_king IS NOT TRUE
     OR baseline_before.source_ref IS DISTINCT FROM old_source_ref
     OR baseline_before.source_size_bytes IS DISTINCT FROM old_source_size
     OR baseline_before.submission_doc ->> 'source_sha256'
          IS DISTINCT FROM old_source_sha
     OR baseline_before.submission_doc ->> 'source_commit'
          IS DISTINCT FROM old_source_commit
     OR EXISTS (
       SELECT 1
       FROM pg_catalog.jsonb_array_elements(round_before.participants) item
       LEFT JOIN public.lab_arena_submissions submission
         ON submission.round_id = 'arena-2026-09-19'
        AND submission.submission_id = item ->> 'submission_id'
       WHERE submission.submission_id IS NULL
          OR submission.miner_hotkey IS DISTINCT FROM item ->> 'miner_hotkey'
          OR submission.source_ref IS DISTINCT FROM item ->> 'source_ref'
          OR submission.source_size_bytes IS DISTINCT FROM
             (item ->> 'source_size_bytes')::BIGINT
     ) THEN
    RAISE EXCEPTION 'Sep19 recovery309 terminal round preimage differs'
     USING ERRCODE = '55000';
  END IF;

  IF (recovery_schedule ->> 'benchmark_deadline')::TIMESTAMPTZ
       <= pg_catalog.clock_timestamp() THEN
    RAISE EXCEPTION 'Sep19 recovery309 admission window closed'
      USING ERRCODE = '22023';
  END IF;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id = 'arena-2026-09-19') <> 110
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-19'
           AND status = 'accepted' AND terminal_cause = 'accepted') <> 86
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-19' AND status = 'failed') <> 24
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = 'arena-2026-09-19'
         AND (kind IS DISTINCT FROM 'execute'
           OR status IN ('pending', 'leased', 'submitted'))
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = 'arena-2026-09-19'
         AND (per_icp_score IS NOT NULL OR qualification_doc IS NOT NULL)
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = 'arena-2026-09-19'
         AND (runner_hotkey IS NOT NULL AND status NOT IN ('accepted','failed'))
     )
     OR EXISTS (
       SELECT 1
       FROM public.lab_arena_company_judgments judgment
       JOIN public.lab_arena_runs score
         ON score.run_id = judgment.source_score_run_id
       WHERE score.round_id = 'arena-2026-09-19'
     )
     OR EXISTS (
       SELECT 1
       FROM public.lab_arena_company_judgment_reservations reservation
       JOIN public.lab_arena_runs run ON run.run_id = reservation.run_id
       WHERE run.round_id = 'arena-2026-09-19'
     ) THEN
    RAISE EXCEPTION 'Sep19 recovery309 terminal execution totals differ'
      USING ERRCODE = '55000';
  END IF;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id = 'arena-2026-09-19'
        AND submission_id = 'baseline-2026-09-19') <> 30
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-19'
           AND submission_id = 'baseline-2026-09-19'
           AND status = 'accepted' AND terminal_cause = 'accepted') <> 6
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-19'
           AND submission_id = 'baseline-2026-09-19'
           AND status = 'failed') <> 24
     OR (SELECT pg_catalog.count(DISTINCT assignment_id)
         FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-19'
           AND submission_id = 'baseline-2026-09-19') <> 20
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-19'
           AND submission_id = 'baseline-2026-09-19' AND attempt = 1) <> 20
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-19'
           AND submission_id = 'baseline-2026-09-19' AND attempt = 2) <> 10
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-19'
           AND submission_id <> 'baseline-2026-09-19'
           AND status = 'accepted' AND terminal_cause = 'accepted') <> 80
     OR EXISTS (
       SELECT 1 FROM (
         SELECT submission_id, pg_catalog.count(*) AS count_rows,
                pg_catalog.count(DISTINCT icp_position) AS positions
         FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-19'
           AND submission_id <> 'baseline-2026-09-19'
         GROUP BY submission_id
       ) miner WHERE count_rows <> 20 OR positions <> 20
     ) THEN
    RAISE EXCEPTION 'Sep19 recovery309 baseline/miner attempt shape differs'
      USING ERRCODE = '55000';
  END IF;

  SELECT public.lab_arena__successful_call_cost_state(
    'baseline-2026-09-19', 'execute', NULL
  ) INTO baseline_cost_before;
  IF EXISTS (
    SELECT 1 FROM pg_catalog.pg_trigger candidate
    WHERE candidate.tgrelid IN (
      'public.lab_arena_rounds'::pg_catalog.regclass,
      'public.lab_arena_submissions'::pg_catalog.regclass,
      'public.lab_arena_runs'::pg_catalog.regclass,
      'public.lab_arena_ledger'::pg_catalog.regclass
    )
      AND NOT candidate.tgisinternal
      AND candidate.tgenabled IS DISTINCT FROM 'O'
  ) THEN
    RAISE EXCEPTION 'Sep19 recovery309 preexisting trigger state differs'
      USING ERRCODE = '55000';
  END IF;
  SELECT pg_catalog.count(*) INTO unsettled
  FROM public.lab_arena_submissions submission
  CROSS JOIN (VALUES ('execute'::TEXT), ('score'::TEXT)) kind(value)
  CROSS JOIN LATERAL (
    SELECT public.lab_arena__successful_call_cost_state(
      submission.submission_id, kind.value, NULL
    ) state
  ) cost
  WHERE submission.round_id = 'arena-2026-09-19'
    AND (
      COALESCE((cost.state ->> 'inflight_calls')::BIGINT, -1) <> 0
      OR COALESCE((cost.state ->> 'success_unresolved_calls')::BIGINT, -1) <> 0
    );
  IF COALESCE((baseline_cost_before ->> 'uncertain_calls')::BIGINT, -1) <> 215
     OR COALESCE((baseline_cost_before ->> 'inflight_calls')::BIGINT, -1) <> 0
     OR COALESCE(
          (baseline_cost_before ->> 'success_unresolved_calls')::BIGINT, -1
        ) <> 0
     OR COALESCE((baseline_cost_before ->> 'settled_microusd')::BIGINT, -1)
          <> 13027527
     OR COALESCE(
          (baseline_cost_before ->> 'reserved_or_uncertain_microusd')::BIGINT, -1
        ) <> 23393484
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE round_id = 'arena-2026-09-19'
           AND submission_id = 'baseline-2026-09-19') <> 6102
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE round_id = 'arena-2026-09-19'
           AND submission_id <> 'baseline-2026-09-19') <> 4908
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE round_id = 'arena-2026-09-19') <> 11010
     OR unsettled <> 0
     OR EXISTS (
       SELECT 1 FROM (VALUES
         ('deepline','reservation',371), ('deepline','dispatch',371),
         ('deepline','settlement',371), ('openrouter','reservation',1649),
         ('openrouter','dispatch',1649), ('openrouter','settlement',1434),
         ('openrouter','uncertain',215), ('openrouter','refusal',12),
         ('scrapingdog','reservation',10), ('scrapingdog','dispatch',10),
         ('scrapingdog','settlement',10)
       ) expected(provider, entry_kind, count_rows)
       WHERE (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger ledger
              WHERE ledger.round_id = 'arena-2026-09-19'
                AND ledger.submission_id = 'baseline-2026-09-19'
                AND ledger.provider = expected.provider
                AND ledger.entry_kind = expected.entry_kind) <> expected.count_rows
     ) THEN
    RAISE EXCEPTION 'Sep19 recovery309 baseline liability state differs'
      USING ERRCODE = '55000';
  END IF;

  SELECT pg_catalog.jsonb_build_object(
    'count', pg_catalog.count(*),
    'sha256', 'sha256:' || pg_catalog.encode(extensions.digest(
      COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
        pg_catalog.to_jsonb(run)::TEXT, 'sha256'), 'hex'), '' ORDER BY run_id), ''),
      'sha256'), 'hex')
  ) INTO miner_runs_before
  FROM public.lab_arena_runs run
  WHERE round_id = 'arena-2026-09-19'
    AND submission_id <> 'baseline-2026-09-19';
  SELECT pg_catalog.jsonb_build_object(
    'count', pg_catalog.count(*),
    'sha256', 'sha256:' || pg_catalog.encode(extensions.digest(
      COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
        pg_catalog.to_jsonb(entry)::TEXT, 'sha256'), 'hex'), '' ORDER BY entry_id), ''),
      'sha256'), 'hex')
  ) INTO miner_ledger_before
  FROM public.lab_arena_ledger entry
  WHERE round_id = 'arena-2026-09-19'
    AND submission_id IS DISTINCT FROM 'baseline-2026-09-19';
  SELECT pg_catalog.jsonb_build_object(
    'rounds', (SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(row)::TEXT,'sha256'),'hex'),'' ORDER BY round_id),''),'sha256'),'hex')) FROM public.lab_arena_rounds row WHERE round_id <> 'arena-2026-09-19'),
    'submissions', (SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(row)::TEXT,'sha256'),'hex'),'' ORDER BY submission_id),''),'sha256'),'hex')) FROM public.lab_arena_submissions row WHERE round_id <> 'arena-2026-09-19'),
    'runs', (SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(row)::TEXT,'sha256'),'hex'),'' ORDER BY run_id),''),'sha256'),'hex')) FROM public.lab_arena_runs row WHERE round_id <> 'arena-2026-09-19'),
    'ledger', (SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(row)::TEXT,'sha256'),'hex'),'' ORDER BY entry_id),''),'sha256'),'hex')) FROM public.lab_arena_ledger row WHERE round_id <> 'arena-2026-09-19')
  ) INTO other_state_before;

  SELECT pg_catalog.jsonb_agg(
    CASE WHEN item ->> 'submission_id' = 'baseline-2026-09-19'
      THEN item || pg_catalog.jsonb_build_object(
        'source_ref', new_source_ref,
        'source_size_bytes', new_source_size
      ) ELSE item END ORDER BY ordinal
  ) INTO new_participants
  FROM pg_catalog.jsonb_array_elements(round_before.participants)
       WITH ORDINALITY participant(item, ordinal);
  IF (SELECT pg_catalog.count(*) FROM pg_catalog.jsonb_array_elements(
        new_participants) item
      WHERE item ->> 'submission_id' = 'baseline-2026-09-19'
        AND item ->> 'source_ref' = new_source_ref
        AND (item ->> 'source_size_bytes')::BIGINT = new_source_size) <> 1 THEN
    RAISE EXCEPTION 'Sep19 recovery309 participant projection differs';
  END IF;
  SELECT pg_catalog.jsonb_agg(
    item || pg_catalog.jsonb_build_object(
      'submission_id', item ->> 'submission_id' || ':r309archive'
    ) ORDER BY ordinal
  ) INTO archive_participants
  FROM pg_catalog.jsonb_array_elements(round_before.participants)
       WITH ORDINALITY participant(item, ordinal);

  archive_configuration := round_before.configuration_doc ||
    pg_catalog.jsonb_build_object(
      'round_id', 'arena-2026-09-19-r309archive',
      'mode', 'shadow',
      'rewards_enabled', FALSE,
      'archived_reward_basis_hash', round_before.reward_basis_hash,
      'archived_effective_reward_epoch', round_before.effective_reward_epoch,
      'archived_reward_activated_at', round_before.reward_activated_at,
      'archive_reason', 'authorized_sep19_baseline_recovery309',
      'archived_baseline_attempt_count', 30,
      'archived_baseline_uncertain_failed_calls', 215
    );
  expected_submission_doc :=
    (baseline_before.submission_doc - 'preexecution_source_override') ||
    pg_catalog.jsonb_build_object(
      'source_ref', new_source_ref,
      'source_size_bytes', new_source_size,
      'source_sha256', new_source_sha,
      'source_commit', new_source_commit,
      'recovery_archive_round_id', 'arena-2026-09-19-r309archive'
    );

  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_submissions DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER USER;

  INSERT INTO public.lab_arena_rounds(
    round_id,status,status_generation,stage_generation,configuration_doc,
    rewards_enabled,participants,benchmark_ref,evaluation_date,
    stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,publication_doc,
    king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,
    reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,
    cancel_reason,published_at,created_at,updated_at,promotion_required,
    promotion_doc,baseline_promoted_at,icp_set_date,confirmation_bank_ref,
    confirmation_bank_hash,confirmation_cohort,stage3_scoring_plan_doc,
    champion_funding_frozen,champion_submission_id,champion_hotkey,
    champion_fallback_providers
  )
  SELECT round_id,status,status_generation,stage_generation,configuration_doc,
    rewards_enabled,participants,benchmark_ref,evaluation_date,
    stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,publication_doc,
    king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,
    reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,
    cancel_reason,published_at,created_at,updated_at,promotion_required,
    promotion_doc,baseline_promoted_at,icp_set_date,confirmation_bank_ref,
    confirmation_bank_hash,confirmation_cohort,stage3_scoring_plan_doc,
    champion_funding_frozen,champion_submission_id,champion_hotkey,
    champion_fallback_providers
  FROM pg_catalog.jsonb_populate_record(
    NULL::public.lab_arena_rounds,
    pg_catalog.to_jsonb(round_before) || pg_catalog.jsonb_build_object(
      'round_id', 'arena-2026-09-19-r309archive',
      'status', 'cancelled',
      'configuration_doc', archive_configuration,
      'participants', archive_participants,
      'rewards_enabled', FALSE,
      'effective_reward_epoch', NULL,
      'reward_basis_hash', NULL,
      'reward_basis_doc', NULL,
      'signing_key_doc', NULL,
      'reward_activated_at', NULL,
      'cancel_reason', 'authorized_sep19_baseline_recovery309_archive'
    )
  );

  INSERT INTO public.lab_arena_submissions
  SELECT (pg_catalog.jsonb_populate_record(
    NULL::public.lab_arena_submissions,
    pg_catalog.to_jsonb(submission) || pg_catalog.jsonb_build_object(
      'round_id', 'arena-2026-09-19-r309archive',
      'submission_id', submission.submission_id || ':r309archive'
    )
  )).*
  FROM public.lab_arena_submissions submission
  WHERE submission.round_id = 'arena-2026-09-19';

  UPDATE public.lab_arena_runs
  SET round_id = 'arena-2026-09-19-r309archive',
      submission_id = 'baseline-2026-09-19:r309archive'
  WHERE round_id = 'arena-2026-09-19'
    AND submission_id = 'baseline-2026-09-19'
    AND kind = 'execute';
  GET DIAGNOSTICS moved_runs = ROW_COUNT;

  UPDATE public.lab_arena_ledger
  SET round_id = 'arena-2026-09-19-r309archive',
      submission_id = 'baseline-2026-09-19:r309archive'
  WHERE round_id = 'arena-2026-09-19'
    AND submission_id = 'baseline-2026-09-19';
  GET DIAGNOSTICS moved_ledger = ROW_COUNT;

  UPDATE public.lab_arena_submissions
  SET source_ref = new_source_ref,
      source_size_bytes = new_source_size,
      submission_doc = expected_submission_doc,
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = 'arena-2026-09-19'
    AND submission_id = 'baseline-2026-09-19';
  IF NOT FOUND THEN RAISE EXCEPTION 'Sep19 recovery309 baseline update missing'; END IF;

  UPDATE public.lab_arena_rounds
  SET status = 'stage1',
      status_generation = status_generation + 1,
      stage_generation = stage_generation + 1,
      configuration_doc = pg_catalog.jsonb_set(
        configuration_doc, '{schedule}', recovery_schedule, TRUE
      ),
      participants = new_participants,
      stage1_scoring_plan_doc = NULL,
      stage2_scoring_plan_doc = NULL,
      stage3_scoring_plan_doc = NULL,
      finalists = NULL,
      publication_doc = NULL,
      published_at = NULL,
      cancel_reason = NULL,
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = 'arena-2026-09-19';
  IF NOT FOUND THEN RAISE EXCEPTION 'Sep19 recovery309 round update missing'; END IF;

  inserted_runs := 0;
  FOR position IN 0..19 LOOP
    stage := CASE WHEN position < 10 THEN 1 ELSE 2 END;
    assignment := 'arena-2026-09-19:baseline-2026-09-19:' ||
      stage::TEXT || ':' || position::TEXT || ':recovery309';
    INSERT INTO public.lab_arena_runs(
      run_id, assignment_id, round_id, submission_id, miner_hotkey,
      stage, icp_position, attempt, kind, status, stage_generation
    ) VALUES (
      assignment || ':1', assignment, 'arena-2026-09-19',
      'baseline-2026-09-19', baseline_before.miner_hotkey,
      stage, position, 1, 'execute', 'pending', round_before.stage_generation + 1
    );
    inserted_runs := inserted_runs + 1;
  END LOOP;

  ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_submissions ENABLE TRIGGER USER;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER;

  SELECT * INTO round_after FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-19';
  SELECT * INTO baseline_after FROM public.lab_arena_submissions
  WHERE submission_id = 'baseline-2026-09-19';
  SELECT * INTO archive_round FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-19-r309archive';
  SELECT * INTO archive_baseline FROM public.lab_arena_submissions
  WHERE submission_id = 'baseline-2026-09-19:r309archive';
  SELECT pg_catalog.jsonb_build_object(
    'count', pg_catalog.count(*),
    'sha256', 'sha256:' || pg_catalog.encode(extensions.digest(
      COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
        pg_catalog.to_jsonb(run)::TEXT, 'sha256'), 'hex'), '' ORDER BY run_id), ''),
      'sha256'), 'hex')
  ) INTO miner_runs_after
  FROM public.lab_arena_runs run
  WHERE round_id = 'arena-2026-09-19'
    AND submission_id <> 'baseline-2026-09-19';
  SELECT pg_catalog.jsonb_build_object(
    'count', pg_catalog.count(*),
    'sha256', 'sha256:' || pg_catalog.encode(extensions.digest(
      COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(
        pg_catalog.to_jsonb(entry)::TEXT, 'sha256'), 'hex'), '' ORDER BY entry_id), ''),
      'sha256'), 'hex')
  ) INTO miner_ledger_after
  FROM public.lab_arena_ledger entry
  WHERE round_id = 'arena-2026-09-19'
    AND submission_id IS DISTINCT FROM 'baseline-2026-09-19';
  SELECT pg_catalog.jsonb_build_object(
    'rounds', (SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(row)::TEXT,'sha256'),'hex'),'' ORDER BY round_id),''),'sha256'),'hex')) FROM public.lab_arena_rounds row WHERE round_id NOT IN ('arena-2026-09-19','arena-2026-09-19-r309archive')),
    'submissions', (SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(row)::TEXT,'sha256'),'hex'),'' ORDER BY submission_id),''),'sha256'),'hex')) FROM public.lab_arena_submissions row WHERE round_id NOT IN ('arena-2026-09-19','arena-2026-09-19-r309archive')),
    'runs', (SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(row)::TEXT,'sha256'),'hex'),'' ORDER BY run_id),''),'sha256'),'hex')) FROM public.lab_arena_runs row WHERE round_id NOT IN ('arena-2026-09-19','arena-2026-09-19-r309archive')),
    'ledger', (SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(row)::TEXT,'sha256'),'hex'),'' ORDER BY entry_id),''),'sha256'),'hex')) FROM public.lab_arena_ledger row WHERE round_id NOT IN ('arena-2026-09-19','arena-2026-09-19-r309archive'))
  ) INTO other_state_after;
  SELECT public.lab_arena__successful_call_cost_state(
    'baseline-2026-09-19', 'execute', NULL
  ) INTO baseline_cost_after;
  SELECT public.lab_arena__successful_call_cost_state(
    'baseline-2026-09-19:r309archive', 'execute', NULL
  ) INTO archive_cost;

  IF moved_runs <> 30 OR inserted_runs <> 20
     OR miner_runs_after IS DISTINCT FROM miner_runs_before
     OR miner_ledger_after IS DISTINCT FROM miner_ledger_before
     OR other_state_after IS DISTINCT FROM other_state_before
     OR (pg_catalog.to_jsonb(round_after)
          - 'status' - 'status_generation' - 'stage_generation'
          - 'configuration_doc' - 'participants' - 'stage1_scoring_plan_doc'
          - 'stage2_scoring_plan_doc' - 'stage3_scoring_plan_doc'
          - 'finalists' - 'publication_doc' - 'published_at'
          - 'cancel_reason' - 'updated_at')
        IS DISTINCT FROM
        (pg_catalog.to_jsonb(round_before)
          - 'status' - 'status_generation' - 'stage_generation'
          - 'configuration_doc' - 'participants' - 'stage1_scoring_plan_doc'
          - 'stage2_scoring_plan_doc' - 'stage3_scoring_plan_doc'
          - 'finalists' - 'publication_doc' - 'published_at'
          - 'cancel_reason' - 'updated_at')
     OR (round_after.configuration_doc - 'schedule')
          IS DISTINCT FROM (round_before.configuration_doc - 'schedule')
     OR round_after.benchmark_ref IS DISTINCT FROM round_before.benchmark_ref
     OR round_after.king_outcome IS DISTINCT FROM round_before.king_outcome
     OR round_after.king_hotkey IS DISTINCT FROM round_before.king_hotkey
     OR round_after.king_start_epoch IS DISTINCT FROM round_before.king_start_epoch
     OR round_after.rewards_enabled IS DISTINCT FROM round_before.rewards_enabled
     OR round_after.reward_basis_hash IS DISTINCT FROM round_before.reward_basis_hash
     OR round_after.reward_basis_doc IS DISTINCT FROM round_before.reward_basis_doc
     OR round_after.signing_key_doc IS DISTINCT FROM round_before.signing_key_doc
     OR round_after.effective_reward_epoch
          IS DISTINCT FROM round_before.effective_reward_epoch
     OR round_after.reward_activated_at
          IS DISTINCT FROM round_before.reward_activated_at
     OR (pg_catalog.to_jsonb(baseline_after)
          - 'source_ref' - 'source_size_bytes' - 'submission_doc' - 'updated_at')
        IS DISTINCT FROM
        (pg_catalog.to_jsonb(baseline_before)
          - 'source_ref' - 'source_size_bytes' - 'submission_doc' - 'updated_at')
     OR baseline_after.submission_doc IS DISTINCT FROM expected_submission_doc
     OR (pg_catalog.to_jsonb(archive_baseline) - 'round_id' - 'submission_id')
          IS DISTINCT FROM
        (pg_catalog.to_jsonb(baseline_before) - 'round_id' - 'submission_id')
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
         WHERE round_id = 'arena-2026-09-19-r309archive') <> 5
     OR EXISTS (
       SELECT 1
       FROM public.lab_arena_submissions active_submission
       LEFT JOIN public.lab_arena_submissions archived_submission
         ON archived_submission.round_id = 'arena-2026-09-19-r309archive'
        AND archived_submission.submission_id =
          active_submission.submission_id || ':r309archive'
       WHERE active_submission.round_id = 'arena-2026-09-19'
         AND active_submission.submission_id <> 'baseline-2026-09-19'
         AND (
           archived_submission.submission_id IS NULL
           OR (pg_catalog.to_jsonb(archived_submission)
                - 'round_id' - 'submission_id') IS DISTINCT FROM
              (pg_catalog.to_jsonb(active_submission)
                - 'round_id' - 'submission_id')
         )
     )
     OR (pg_catalog.to_jsonb(archive_round)
          - 'round_id' - 'status' - 'configuration_doc' - 'participants'
          - 'rewards_enabled' - 'effective_reward_epoch' - 'reward_basis_hash'
          - 'reward_basis_doc' - 'signing_key_doc' - 'reward_activated_at'
          - 'cancel_reason') IS DISTINCT FROM
        (pg_catalog.to_jsonb(round_before)
          - 'round_id' - 'status' - 'configuration_doc' - 'participants'
          - 'rewards_enabled' - 'effective_reward_epoch' - 'reward_basis_hash'
          - 'reward_basis_doc' - 'signing_key_doc' - 'reward_activated_at'
          - 'cancel_reason')
     OR archive_round.configuration_doc IS DISTINCT FROM archive_configuration
     OR archive_round.participants IS DISTINCT FROM archive_participants
     OR archive_round.status IS DISTINCT FROM 'cancelled'
     OR archive_round.cancel_reason
          IS DISTINCT FROM 'authorized_sep19_baseline_recovery309_archive'
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = 'arena-2026-09-19-r309archive'
           AND submission_id = 'baseline-2026-09-19:r309archive') <> 30
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE round_id = 'arena-2026-09-19-r309archive'
           AND submission_id = 'baseline-2026-09-19:r309archive') <> moved_ledger
     OR COALESCE((baseline_cost_after ->> 'call_count')::BIGINT, -1) <> 0
     OR COALESCE((archive_cost ->> 'uncertain_calls')::BIGINT, -1) <> 215
     OR COALESCE((archive_cost ->> 'inflight_calls')::BIGINT, -1) <> 0
     OR COALESCE((archive_cost ->> 'success_unresolved_calls')::BIGINT, -1) <> 0
     OR COALESCE((archive_cost ->> 'settled_microusd')::BIGINT, -1) <> 13027527
     OR COALESCE(
          (archive_cost ->> 'reserved_or_uncertain_microusd')::BIGINT, -1
        ) <> 23393484
     OR COALESCE((archive_cost ->> 'refused_calls')::BIGINT, -1) <> 12
     OR COALESCE((archive_cost ->> 'successful_calls')::BIGINT, -1) <> 1804
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
                WHERE round_id = 'arena-2026-09-19' AND kind = 'score') THEN
    RAISE EXCEPTION 'Sep19 recovery309 atomic preservation differs'
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
     ) OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_runs'::pg_catalog.regclass
         AND tgname = 'lab_arena_runs_terminal'
         AND NOT tgisinternal AND tgenabled = 'O'
     ) OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_ledger'::pg_catalog.regclass
         AND tgname = 'lab_arena_ledger_append_only'
         AND NOT tgisinternal AND tgenabled = 'O'
     ) OR EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger candidate
       WHERE candidate.tgrelid IN (
         'public.lab_arena_rounds'::pg_catalog.regclass,
         'public.lab_arena_submissions'::pg_catalog.regclass,
         'public.lab_arena_runs'::pg_catalog.regclass,
         'public.lab_arena_ledger'::pg_catalog.regclass
       )
         AND NOT candidate.tgisinternal
         AND candidate.tgenabled IS DISTINCT FROM 'O'
     ) THEN
    RAISE EXCEPTION 'Sep19 recovery309 trigger restore differs';
  END IF;

  RETURN pg_catalog.jsonb_build_object(
    'status', 'prepared',
    'round_id', 'arena-2026-09-19',
    'fresh_baseline_execute_count', 20,
    'retained_miner_execute_count', 80,
    'archived_baseline_attempt_count', moved_runs,
    'archived_baseline_ledger_count', moved_ledger,
    'archived_failed_uncertain_calls', 215,
    'score_namespace', 'score'
  );
END;
$recover_sep19_baseline309$;

ALTER FUNCTION public.lab_arena_prepare_sep19_recovery309_v1(JSONB)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_prepare_sep19_recovery309_v1(JSONB)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

SELECT public.lab_arena_prepare_sep19_recovery309_v1(
  '{"benchmark_deadline":"2026-09-19T06:00:00Z","final_scoring_close":"2026-09-19T18:00:00Z","publication_deadline":"2026-09-19T18:00:01Z","stage_1_close":"2026-09-19T09:00:01Z","stage_1_scoring_close":"2026-09-19T12:00:00Z","stage_1_start":"2026-09-19T06:00:01Z","stage_2_close":"2026-09-19T15:00:01Z","stage_2_start":"2026-09-19T12:00:01Z","submission_cutoff":"2026-09-19T00:00:00Z","submission_open":"2026-09-18T00:00:00Z"}'::JSONB
);

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
