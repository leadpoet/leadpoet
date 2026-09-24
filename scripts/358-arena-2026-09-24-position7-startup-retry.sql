-- Append one execution attempt for Sep24 baseline ICP position 7.  Its first
-- model_error happened before sandbox startup and consumed the normal retry;
-- attempt 2 was the only model execution.  Keep the round, frozen source and
-- bank, completed outputs, old attempts, and all provider accounting unchanged.
--
-- Read-only object evidence: source commit
-- f5a95ff38865c0249750ec45f66c06794016abef, archive SHA-256
-- 423c6c6ac369ec96404083852887f5513bed5e891d0140232221da65bd229329.
-- Canonical application commitments: configuration
-- sha256:fe6bbdca83b4b2e95288b6ef4e1b8a7984b2e900b5af5cb1e304c0a7ca45cffd,
-- Sep23 ICP bank
-- sha256:d4dedd1e5092532870026f041a42e05ee8d9b592940ff64b61249578b28dffc1.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

SELECT pg_catalog.pg_advisory_xact_lock(
  pg_catalog.hashtextextended('lab-arena-claim-control', 0)
);
LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_ledger IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.qualification_private_icp_sets IN SHARE MODE;

DO $recover_sep24_position7$
DECLARE
  v_round public.lab_arena_rounds;
  v_baseline public.lab_arena_submissions;
  v_attempt1 public.lab_arena_runs;
  v_attempt2 public.lab_arena_runs;
  v_retry public.lab_arena_runs;
  v_control public.lab_arena_restart_claim_control;
  v_bank JSONB;
  v_round_before JSONB;
  v_bank_before JSONB;
  v_submissions_before JSONB;
  v_runs_before JSONB;
  v_ledger_before JSONB;
  v_active BIGINT;
  v_count BIGINT;
  v_stage_close TIMESTAMPTZ;
  v_assignment CONSTANT TEXT :=
    'arena-2026-09-24:baseline-2026-09-24:1:7';
  v_source_ref CONSTANT TEXT :=
    'arena/arena-2026-09-24/sources/baseline-2026-09-24.tar.gz';
BEGIN
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-24'
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Sep24 position7 recovery round is missing'
      USING ERRCODE = 'P0002';
  END IF;

  SELECT * INTO v_baseline
  FROM public.lab_arena_submissions
  WHERE round_id = v_round.round_id
    AND submission_id = 'baseline-2026-09-24'
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Sep24 position7 recovery baseline is missing'
      USING ERRCODE = 'P0002';
  END IF;

  SELECT pg_catalog.to_jsonb(source) INTO v_bank
  FROM public.qualification_private_icp_sets AS source
  WHERE source.set_id = 20260923;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Sep24 position7 recovery bank is missing'
      USING ERRCODE = 'P0002';
  END IF;

  -- Bind the exact frozen policy, source identity, participant commitment, and
  -- ten-ICP bank.  PostgreSQL-native hashes are included so any unlisted field
  -- drift also fails closed.
  IF 'sha256:' || pg_catalog.encode(extensions.digest(
       v_round.configuration_doc::TEXT, 'sha256'), 'hex')
       IS DISTINCT FROM
       'sha256:51db3a10993646f6aa07503bd76dea9244c4c97879f54f2a1b67b7eebcd3c8da'
     OR 'sha256:' || pg_catalog.encode(extensions.digest(
       v_round.participants::TEXT, 'sha256'), 'hex')
       IS DISTINCT FROM
       'sha256:d8f69ece7505dfd31e3570b2fc21f3589f80aaa49d6b46adb1f82a986232029a'
     OR v_round.configuration_doc ->> 'schema_version'
       IS DISTINCT FROM 'leadpoet.lab_arena.round_configuration.v1'
     OR v_round.configuration_doc ->> 'round_id'
       IS DISTINCT FROM v_round.round_id
     OR v_round.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
     OR v_round.configuration_doc ->> 'network_name' IS DISTINCT FROM 'finney'
     OR (v_round.configuration_doc ->> 'netuid')::INTEGER IS DISTINCT FROM 71
     OR v_round.configuration_doc ->> 'integrity_policy'
       IS DISTINCT FROM 'arena_integrity_v1'
     OR v_round.configuration_doc ->> 'intent_details_policy'
       IS DISTINCT FROM 'intent_details_v1'
     OR v_round.configuration_doc ->> 'execution_sequence_policy'
       IS DISTINCT FROM 'baseline_scored_first_v1'
     OR v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
       IS DISTINCT FROM 'successful_calls_per_icp_v1'
     OR v_round.configuration_doc #>> '{scorer_policy,scoring_adapter_version}'
       IS DISTINCT FROM 'qualification_integrity_v2'
     OR v_round.configuration_doc ? 'contact_policy'
     OR v_round.configuration_doc -> 'call_quotas'
       IS DISTINCT FROM
       '{"deepline":200,"openrouter":2000,"scrapingdog":200}'::JSONB
     OR v_round.configuration_doc -> 'scoring_call_quotas'
       IS DISTINCT FROM
       '{"deepline":2000,"openrouter":2000,"scrapingdog":2000}'::JSONB
     OR (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
       IS DISTINCT FROM 5
     OR (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER
       IS DISTINCT FROM 5
     OR (v_round.configuration_doc ->> 'max_attempts_per_assignment')::INTEGER
       IS DISTINCT FROM 2
     OR (v_round.configuration_doc ->> 'icp_wall_clock_seconds')::INTEGER
       IS DISTINCT FROM 3600
     OR (v_round.configuration_doc ->> 'runner_slot_ceiling')::INTEGER
       IS DISTINCT FROM 20
     OR (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
       IS DISTINCT FROM 4000000
     OR (v_round.configuration_doc ->> 'cost_per_company_microusd')::BIGINT
       IS DISTINCT FROM 800000
     OR v_round.benchmark_ref IS DISTINCT FROM
       'arena/arena-2026-09-24/benchmark.json'
     OR v_round.evaluation_date IS DISTINCT FROM '2026-09-24'
     OR v_round.icp_set_date IS DISTINCT FROM '2026-09-23'
     OR pg_catalog.jsonb_array_length(v_round.participants) IS DISTINCT FROM 9
     OR NOT v_baseline.is_king
     OR v_baseline.status IS DISTINCT FROM 'frozen'
     OR v_baseline.miner_hotkey IS DISTINCT FROM
       '5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9'
     OR v_baseline.source_ref IS DISTINCT FROM v_source_ref
     OR v_baseline.source_size_bytes IS DISTINCT FROM 817521
     OR v_baseline.submission_doc ->> 'source_ref' IS DISTINCT FROM v_source_ref
     OR (v_baseline.submission_doc ->> 'source_size_bytes')::BIGINT
       IS DISTINCT FROM 817521
     OR (SELECT pg_catalog.count(*)
         FROM pg_catalog.jsonb_array_elements(v_round.participants) AS item
         WHERE item ->> 'submission_id' = v_baseline.submission_id
           AND item ->> 'miner_hotkey' = v_baseline.miner_hotkey
           AND item ->> 'source_ref' = v_source_ref
           AND (item ->> 'source_size_bytes')::BIGINT = 817521
           AND COALESCE((item ->> 'is_king')::BOOLEAN, FALSE))
       IS DISTINCT FROM 1::BIGINT
     OR pg_catalog.jsonb_array_length(v_bank -> 'icps') IS DISTINCT FROM 10
     OR 'sha256:' || pg_catalog.encode(extensions.digest(
       (v_bank -> 'icps')::TEXT, 'sha256'), 'hex')
       IS DISTINCT FROM
       'sha256:48d6387354b9cead518aff20c8b681bb8236aec64efcbe01dc05bcab1f4d6972'
  THEN
    RAISE EXCEPTION 'Sep24 position7 recovery source, bank, or policy differs'
      USING ERRCODE = '55000';
  END IF;

  -- A replay validates the immutable identity and never resets a retry that
  -- has been leased, completed, scored, or consumed by later stage progress.
  SELECT * INTO v_retry
  FROM public.lab_arena_runs
  WHERE run_id = v_assignment || ':3';
  IF FOUND THEN
    IF v_retry.assignment_id IS DISTINCT FROM v_assignment
       OR v_retry.round_id IS DISTINCT FROM v_round.round_id
       OR v_retry.submission_id IS DISTINCT FROM v_baseline.submission_id
       OR v_retry.miner_hotkey IS DISTINCT FROM v_baseline.miner_hotkey
       OR v_retry.stage IS DISTINCT FROM 1
       OR v_retry.icp_position IS DISTINCT FROM 7
       OR v_retry.attempt IS DISTINCT FROM 3
       OR v_retry.kind IS DISTINCT FROM 'execute'
       OR v_retry.stage_generation IS DISTINCT FROM 1
       OR v_retry.previous_runner_hotkey IS DISTINCT FROM
         '5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9'
       OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
           WHERE assignment_id = v_assignment) IS DISTINCT FROM 3::BIGINT THEN
      RAISE EXCEPTION 'Sep24 position7 recovery replay row differs'
        USING ERRCODE = '55000';
    END IF;
    RETURN;
  END IF;

  SELECT * INTO v_control
  FROM public.lab_arena_restart_claim_control
  WHERE singleton;
  IF NOT FOUND
     OR v_control.operator_paused
     OR v_control.guard_commitment <> ''
     OR v_control.owner_commitment <> ''
     OR v_control.restart_scope <> ''
     OR v_control.restart_phase <> ''
     OR v_control.guard_expires_at IS NOT NULL
     OR v_control.captured_leases <> '[]'::JSONB THEN
    RAISE EXCEPTION 'Sep24 position7 recovery restart guard is active'
      USING ERRCODE = '55000';
  END IF;

  v_stage_close :=
    (v_round.configuration_doc #>> '{schedule,stage_1_close}')::TIMESTAMPTZ;
  IF v_round.status IS DISTINCT FROM 'stage1'
     OR v_round.status_generation IS DISTINCT FROM 2
     OR v_round.stage_generation IS DISTINCT FROM 1
     OR v_round.cancel_reason IS NOT NULL
     OR v_round.stage1_scoring_plan_doc IS NOT NULL
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR v_round.stage3_scoring_plan_doc IS NOT NULL
     OR v_round.finalists IS NOT NULL
     OR v_round.publication_doc IS NOT NULL
     OR v_round.published_at IS NOT NULL
     OR v_round.reward_basis_doc IS NOT NULL
     OR v_round.reward_basis_hash IS NOT NULL
     OR v_round.reward_activated_at IS NOT NULL
     OR v_round.signing_key_doc IS NOT NULL
     OR v_round.effective_reward_epoch IS NOT NULL
     OR v_stage_close IS DISTINCT FROM '2026-09-24T04:30:01Z'::TIMESTAMPTZ
     OR pg_catalog.clock_timestamp() + INTERVAL '60 minutes' > v_stage_close
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round.round_id
         AND (kind = 'score' OR per_icp_score IS NOT NULL
              OR scored_run_id IS NOT NULL OR qualification_doc IS NOT NULL)
     ) THEN
    RAISE EXCEPTION 'Sep24 position7 recovery requires active unscored stage1 with one hour remaining'
      USING ERRCODE = '55000';
  END IF;

  SELECT pg_catalog.count(*) INTO v_active
  FROM public.lab_arena_runs
  WHERE round_id = v_round.round_id
    AND status IN ('pending', 'leased', 'submitted');
  IF v_active >=
       (v_round.configuration_doc ->> 'runner_slot_ceiling')::INTEGER THEN
    RAISE EXCEPTION 'Sep24 position7 recovery has no execution capacity'
      USING ERRCODE = '55000';
  END IF;

  SELECT * INTO v_attempt1 FROM public.lab_arena_runs
  WHERE run_id = v_assignment || ':1';
  SELECT * INTO v_attempt2 FROM public.lab_arena_runs
  WHERE run_id = v_assignment || ':2';
  IF v_attempt1.run_id IS NULL OR v_attempt2.run_id IS NULL
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE assignment_id = v_assignment) IS DISTINCT FROM 2::BIGINT
     OR v_attempt1.assignment_id IS DISTINCT FROM v_assignment
     OR v_attempt2.assignment_id IS DISTINCT FROM v_assignment
     OR v_attempt1.round_id IS DISTINCT FROM v_round.round_id
     OR v_attempt2.round_id IS DISTINCT FROM v_round.round_id
     OR v_attempt1.submission_id IS DISTINCT FROM v_baseline.submission_id
     OR v_attempt2.submission_id IS DISTINCT FROM v_baseline.submission_id
     OR v_attempt1.miner_hotkey IS DISTINCT FROM v_baseline.miner_hotkey
     OR v_attempt2.miner_hotkey IS DISTINCT FROM v_baseline.miner_hotkey
     OR v_attempt1.stage IS DISTINCT FROM 1
     OR v_attempt2.stage IS DISTINCT FROM 1
     OR v_attempt1.icp_position IS DISTINCT FROM 7
     OR v_attempt2.icp_position IS DISTINCT FROM 7
     OR v_attempt1.attempt IS DISTINCT FROM 1
     OR v_attempt2.attempt IS DISTINCT FROM 2
     OR v_attempt1.kind IS DISTINCT FROM 'execute'
     OR v_attempt2.kind IS DISTINCT FROM 'execute'
     OR v_attempt1.stage_generation IS DISTINCT FROM 1
     OR v_attempt2.stage_generation IS DISTINCT FROM 1
     OR v_attempt1.status IS DISTINCT FROM 'failed'
     OR v_attempt2.status IS DISTINCT FROM 'failed'
     OR v_attempt1.terminal_cause IS DISTINCT FROM 'model_error'
     OR v_attempt2.terminal_cause IS DISTINCT FROM 'model_error'
     OR v_attempt1.output_ref IS NOT NULL OR v_attempt2.output_ref IS NOT NULL
     OR v_attempt1.per_icp_score IS NOT NULL OR v_attempt2.per_icp_score IS NOT NULL
     OR v_attempt1.qualification_doc IS NOT NULL
     OR v_attempt2.qualification_doc IS NOT NULL
     OR v_attempt1.terminal_doc IS NOT NULL OR v_attempt2.terminal_doc IS NOT NULL
     OR v_attempt1.runner_hotkey IS DISTINCT FROM
       '5Chnr6Y72gdfTFdoZnsCvkndKpMk8jAtt9JAYKaNG3LmU4BW'
     OR v_attempt2.runner_hotkey IS DISTINCT FROM
       '5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9'
     OR v_attempt2.previous_runner_hotkey IS DISTINCT FROM
       v_attempt1.runner_hotkey
     OR v_attempt1.result_doc IS DISTINCT FROM
       '{"finished_at":"2026-09-24T00:02:09Z","resource_summary":{"cpu_seconds":0.0,"max_rss_bytes":0,"provider_call_count":0,"stderr_bytes":0,"stdout_bytes":0,"wall_seconds":0.0},"schema_version":"leadpoet.lab_arena.run_result.v1","started_at":"2026-09-24T00:01:36Z","terminal_status":"model_error"}'::JSONB
     OR v_attempt2.result_doc IS DISTINCT FROM
       '{"finished_at":"2026-09-24T01:02:14Z","resource_summary":{"cpu_seconds":78.432882,"max_rss_bytes":334307328,"provider_call_count":119,"stderr_bytes":963,"stdout_bytes":0,"wall_seconds":3577.251937283203,"web_egress":{"active_limit_rejection_count":0,"byte_limit_rejection_count":0,"cleanup_block_rejection_count":0,"connection_count":20,"download_bytes":25329812,"exit_fingerprint":"b8ed2a8279876c4d","failure_count":0,"policy_version":"webshare_parallel_v1","total_limit_rejection_count":0,"upload_bytes":3579152,"worker_slot":8}},"schema_version":"leadpoet.lab_arena.run_result.v1","started_at":"2026-09-24T00:02:36Z","terminal_status":"model_error"}'::JSONB
  THEN
    RAISE EXCEPTION 'Sep24 position7 recovery attempts differ'
      USING ERRCODE = '55000';
  END IF;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
      WHERE run_id IN (v_attempt1.run_id, v_attempt2.run_id))
       IS DISTINCT FROM 360::BIGINT
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE run_id IN (v_attempt1.run_id, v_attempt2.run_id)
           AND entry_kind = 'reservation') IS DISTINCT FROM 120::BIGINT
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE run_id IN (v_attempt1.run_id, v_attempt2.run_id)
           AND entry_kind = 'dispatch') IS DISTINCT FROM 120::BIGINT
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE run_id IN (v_attempt1.run_id, v_attempt2.run_id)
           AND entry_kind = 'settlement') IS DISTINCT FROM 120::BIGINT
     OR (SELECT COALESCE(pg_catalog.sum(head.amount_microusd), 0)
         FROM (
           SELECT DISTINCT ON (call_identity)
             call_identity, entry_kind, amount_microusd
           FROM public.lab_arena_ledger
           WHERE run_id IN (v_attempt1.run_id, v_attempt2.run_id)
             AND call_identity IS NOT NULL
           ORDER BY call_identity, entry_id DESC
         ) AS head WHERE head.entry_kind = 'settlement')
       IS DISTINCT FROM 261868::NUMERIC
     OR EXISTS (
       SELECT 1 FROM (
         SELECT DISTINCT ON (call_identity) entry_kind
         FROM public.lab_arena_ledger
         WHERE run_id IN (v_attempt1.run_id, v_attempt2.run_id)
           AND call_identity IS NOT NULL
         ORDER BY call_identity, entry_id DESC
       ) AS head
       WHERE head.entry_kind IN ('reservation', 'dispatch', 'uncertain')
     ) THEN
    RAISE EXCEPTION 'Sep24 position7 recovery accounting differs or is open'
      USING ERRCODE = '55000';
  END IF;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id = v_round.round_id AND status = 'accepted'
        AND output_ref IS NOT NULL) < 6
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round.round_id AND status = 'accepted'
         AND output_ref IS NULL
     ) THEN
    RAISE EXCEPTION 'Sep24 position7 recovery completed outputs differ'
      USING ERRCODE = '55000';
  END IF;

  SELECT pg_catalog.to_jsonb(v_round) INTO v_round_before;
  SELECT pg_catalog.to_jsonb(source) INTO v_bank_before
  FROM public.qualification_private_icp_sets AS source
  WHERE source.set_id = 20260923;
  SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(row_data)
           ORDER BY submission_id), '[]'::JSONB)
    INTO v_submissions_before
  FROM public.lab_arena_submissions AS row_data
  WHERE round_id = v_round.round_id;
  SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(row_data)
           ORDER BY run_id), '[]'::JSONB)
    INTO v_runs_before
  FROM public.lab_arena_runs AS row_data
  WHERE round_id = v_round.round_id;
  SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(row_data)
           ORDER BY entry_id), '[]'::JSONB)
    INTO v_ledger_before
  FROM public.lab_arena_ledger AS row_data
  WHERE round_id = v_round.round_id;

  INSERT INTO public.lab_arena_runs (
    run_id, assignment_id, round_id, submission_id, miner_hotkey, stage,
    icp_position, attempt, status, lease_generation, stage_generation, kind,
    previous_runner_hotkey
  ) VALUES (
    v_assignment || ':3', v_assignment, v_attempt2.round_id,
    v_attempt2.submission_id, v_attempt2.miner_hotkey, v_attempt2.stage,
    v_attempt2.icp_position, 3, 'pending', v_attempt2.lease_generation,
    v_attempt2.stage_generation, v_attempt2.kind, v_attempt2.runner_hotkey
  );
  GET DIAGNOSTICS v_count = ROW_COUNT;

  IF v_count IS DISTINCT FROM 1::BIGINT
     OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE run_id = v_assignment || ':3'
         AND assignment_id = v_assignment
         AND round_id = v_attempt2.round_id
         AND submission_id = v_attempt2.submission_id
         AND miner_hotkey = v_attempt2.miner_hotkey
         AND stage = 1 AND icp_position = 7 AND attempt = 3
         AND status = 'pending'
         AND lease_generation = v_attempt2.lease_generation
         AND stage_generation = v_attempt2.stage_generation
         AND kind = 'execute'
         AND previous_runner_hotkey = v_attempt2.runner_hotkey
     )
     OR (SELECT pg_catalog.to_jsonb(row_data)
         FROM public.lab_arena_rounds AS row_data
         WHERE round_id = v_round.round_id) IS DISTINCT FROM v_round_before
     OR (SELECT pg_catalog.to_jsonb(source)
         FROM public.qualification_private_icp_sets AS source
         WHERE source.set_id = 20260923) IS DISTINCT FROM v_bank_before
     OR (SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(row_data)
                  ORDER BY submission_id), '[]'::JSONB)
         FROM public.lab_arena_submissions AS row_data
         WHERE round_id = v_round.round_id) IS DISTINCT FROM v_submissions_before
     OR (SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(row_data)
                  ORDER BY run_id), '[]'::JSONB)
         FROM public.lab_arena_runs AS row_data
         WHERE round_id = v_round.round_id
           AND run_id <> v_assignment || ':3') IS DISTINCT FROM v_runs_before
     OR (SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.to_jsonb(row_data)
                  ORDER BY entry_id), '[]'::JSONB)
         FROM public.lab_arena_ledger AS row_data
         WHERE round_id = v_round.round_id) IS DISTINCT FROM v_ledger_before THEN
    RAISE EXCEPTION 'Sep24 position7 recovery changed protected state'
      USING ERRCODE = '55000';
  END IF;
END;
$recover_sep24_position7$;

COMMIT;
