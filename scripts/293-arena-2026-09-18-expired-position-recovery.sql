-- Resume only Sep18 baseline position 12 after its final recovered worker
-- exited with one dispatched OpenRouter call. Lease expiry preserved that call
-- as uncertain at its full reservation. Keep that accounting evidence and all
-- prior attempts immutable; add one fresh execution attempt only.
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

DO $recover_sep18_expired_position$
DECLARE
  v_round public.lab_arena_rounds;
  v_target public.lab_arena_runs;
  v_control public.lab_arena_restart_claim_control;
  v_existing public.lab_arena_runs;
  v_round_stable JSONB;
  v_runs_hash TEXT;
  v_submissions_hash TEXT;
  v_ledger_hash TEXT;
  v_count BIGINT;
  v_resume_generation BIGINT := 5;
  v_stage_close TIMESTAMPTZ;
  v_scoring_close TIMESTAMPTZ;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-18' FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'sep18 expired-position recovery round missing';
  END IF;
  SELECT * INTO v_control FROM public.lab_arena_restart_claim_control
  WHERE singleton;
  IF NOT FOUND
     OR v_control.operator_paused
     OR v_control.guard_commitment <> ''
     OR v_control.owner_commitment <> ''
     OR v_control.restart_scope <> ''
     OR v_control.restart_phase <> ''
     OR v_control.guard_expires_at IS NOT NULL
     OR v_control.captured_leases <> '[]'::JSONB THEN
    RAISE EXCEPTION 'sep18 expired-position recovery restart guard is active';
  END IF;

  -- A replay after the exact retry progressed must not reset it.
  SELECT * INTO v_existing FROM public.lab_arena_runs
  WHERE run_id = 'arena-2026-09-18:baseline-2026-09-18:2:12:4';
  IF FOUND THEN
    IF v_existing.assignment_id IS DISTINCT FROM
         'arena-2026-09-18:baseline-2026-09-18:2:12'
       OR v_existing.round_id IS DISTINCT FROM v_round.round_id
       OR v_existing.submission_id IS DISTINCT FROM 'baseline-2026-09-18'
       OR v_existing.stage IS DISTINCT FROM 2
       OR v_existing.icp_position IS DISTINCT FROM 12
       OR v_existing.attempt IS DISTINCT FROM 4
       OR v_existing.kind IS DISTINCT FROM 'execute'
       OR v_existing.stage_generation IS DISTINCT FROM 5
       OR v_round.status NOT IN (
         'stage1', 'stage1_closed', 'stage1_scoring', 'stage1_judged',
         'stage1_scored', 'stage2', 'stage2_closed', 'stage2_scoring',
         'stage2_judged', 'scored', 'published'
       ) THEN
      RAISE EXCEPTION 'sep18 expired-position recovery replay state differs';
    END IF;
    RETURN;
  END IF;

  v_stage_close := (
    v_round.configuration_doc #>> '{schedule,stage_1_close}'
  )::TIMESTAMPTZ;
  v_scoring_close := (
    v_round.configuration_doc #>> '{schedule,stage_1_scoring_close}'
  )::TIMESTAMPTZ;
  IF v_round.status IS DISTINCT FROM 'cancelled'
     OR v_round.cancel_reason IS DISTINCT FROM
          'execution_incomplete:stage1:1'
     OR v_round.status_generation IS DISTINCT FROM 5
     OR v_round.stage_generation IS DISTINCT FROM 4
     OR v_round.stage1_scoring_plan_doc IS NOT NULL
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR v_round.finalists IS NOT NULL
     OR v_round.publication_doc IS NOT NULL
     OR v_round.published_at IS NOT NULL
     OR v_round.reward_basis_doc IS NOT NULL
     OR v_round.reward_basis_hash IS NOT NULL
     OR v_round.reward_activated_at IS NOT NULL
     OR v_round.signing_key_doc IS NOT NULL
     OR v_round.effective_reward_epoch IS NOT NULL
     OR v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
          IS DISTINCT FROM 'successful_calls_per_icp_v1'
     OR (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
          IS DISTINCT FROM 4000000
     OR (v_round.configuration_doc ->> 'cost_per_company_microusd')::BIGINT
          IS DISTINCT FROM 800000
     OR v_round.configuration_doc ->> 'parallel_twenty_icp_execution'
          IS DISTINCT FROM 'true'
     OR (v_round.configuration_doc ->> 'icp_wall_clock_seconds')::INTEGER
          IS DISTINCT FROM 2700
     OR (v_round.configuration_doc ->> 'scoring_wall_clock_seconds')::INTEGER
          IS DISTINCT FROM 900
     OR (v_round.configuration_doc ->> 'runner_slot_ceiling')::INTEGER
          IS DISTINCT FROM 20
     OR v_stage_close IS DISTINCT FROM '2026-09-18T16:30:01Z'::TIMESTAMPTZ
     OR v_scoring_close IS DISTINCT FROM
          '2026-09-18T19:30:01Z'::TIMESTAMPTZ
     OR pg_catalog.clock_timestamp() + INTERVAL '45 minutes' > v_stage_close
     OR v_scoring_close - v_stage_close < INTERVAL '160 minutes'
     OR pg_catalog.jsonb_array_length(v_round.participants) IS DISTINCT FROM 5
     OR v_round.benchmark_ref IS DISTINCT FROM
          'arena/arena-2026-09-18/benchmark.json' THEN
    RAISE EXCEPTION 'sep18 expired-position recovery round state differs';
  END IF;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
      WHERE round_id = v_round.round_id) IS DISTINCT FROM 5::BIGINT
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
         WHERE round_id = v_round.round_id AND status = 'frozen'
           AND source_ref IS NOT NULL AND source_size_bytes IS NOT NULL)
          IS DISTINCT FROM 5::BIGINT
     OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_submissions
       WHERE submission_id = 'baseline-2026-09-18'
         AND round_id = v_round.round_id AND status = 'frozen' AND is_king
         AND source_ref =
           'arena/arena-2026-09-18/sources/baseline-2026-09-18.tar.gz'
         AND source_size_bytes = 604847
     ) THEN
    RAISE EXCEPTION 'sep18 expired-position recovery source state differs';
  END IF;

  SELECT * INTO v_target FROM public.lab_arena_runs
  WHERE run_id = 'arena-2026-09-18:baseline-2026-09-18:2:12:3';
  IF NOT FOUND
     OR v_target.assignment_id IS DISTINCT FROM
          'arena-2026-09-18:baseline-2026-09-18:2:12'
     OR v_target.round_id IS DISTINCT FROM v_round.round_id
     OR v_target.submission_id IS DISTINCT FROM 'baseline-2026-09-18'
     OR v_target.stage IS DISTINCT FROM 2
     OR v_target.icp_position IS DISTINCT FROM 12
     OR v_target.attempt IS DISTINCT FROM 3
     OR v_target.kind IS DISTINCT FROM 'execute'
     OR v_target.status IS DISTINCT FROM 'failed'
     OR v_target.terminal_cause IS DISTINCT FROM 'lease_expired'
     OR v_target.result_doc IS NOT NULL
     OR v_target.output_ref IS NOT NULL
     OR v_target.lease_expires_at IS NULL
     OR v_target.runner_hotkey IS NULL
     OR v_target.lease_token_hash IS NULL
     OR v_target.previous_runner_hotkey IS NULL
     OR pg_catalog.jsonb_typeof(v_target.terminal_doc) IS DISTINCT FROM 'object'
     OR v_target.terminal_doc - 'expired_at' <> '{}'::JSONB
     OR COALESCE(v_target.terminal_doc ->> 'expired_at', '') = ''
     OR (v_target.terminal_doc ->> 'expired_at')::TIMESTAMPTZ
          < v_target.lease_expires_at THEN
    RAISE EXCEPTION 'sep18 expired-position recovery target state differs';
  END IF;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id = v_round.round_id) IS DISTINCT FROM 116::BIGINT
     OR (SELECT pg_catalog.count(DISTINCT assignment_id)
         FROM public.lab_arena_runs WHERE round_id = v_round.round_id)
          IS DISTINCT FROM 100::BIGINT
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round.round_id AND status = 'accepted')
          IS DISTINCT FROM 93::BIGINT
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round.round_id AND status = 'accepted'
           AND output_ref IS NOT NULL) IS DISTINCT FROM 93::BIGINT
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round.round_id AND status = 'failed')
          IS DISTINCT FROM 23::BIGINT
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round.round_id
         AND status IN ('pending', 'leased', 'submitted')
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round.round_id AND attempt >= 4
     )
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round.round_id
           AND submission_id = 'baseline-2026-09-18'
           AND kind = 'execute' AND stage = 2 AND attempt = 3)
          IS DISTINCT FROM 6::BIGINT
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round.round_id
           AND submission_id = 'baseline-2026-09-18'
           AND kind = 'execute' AND stage = 2 AND attempt = 3
           AND icp_position IN (10, 11, 13, 14, 19))
          IS DISTINCT FROM 5::BIGINT
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round.round_id
         AND submission_id = 'baseline-2026-09-18'
         AND kind = 'execute' AND stage = 2 AND attempt = 3
         AND icp_position IN (10, 11, 13, 14, 19)
         AND (status <> 'failed' OR terminal_cause <> 'budget_exhausted'
              OR result_doc ->> 'terminal_status' <> 'budget_exhausted'
              OR output_ref IS NOT NULL)
     ) THEN
    RAISE EXCEPTION 'sep18 expired-position recovery execution state differs';
  END IF;

  SELECT pg_catalog.count(*) INTO v_count
  FROM public.lab_arena_ledger
  WHERE run_id = v_target.run_id;
  IF v_count IS DISTINCT FROM 30::BIGINT
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE run_id = v_target.run_id AND entry_kind = 'reservation')
          IS DISTINCT FROM 10::BIGINT
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE run_id = v_target.run_id AND entry_kind = 'dispatch')
          IS DISTINCT FROM 10::BIGINT
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE run_id = v_target.run_id AND entry_kind = 'settlement')
          IS DISTINCT FROM 9::BIGINT
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE run_id = v_target.run_id AND entry_kind = 'uncertain'
           AND provider = 'openrouter'
           AND operation_id = 'openrouter.responses'
           AND amount_microusd = 90616
           AND entry_doc = '{"reason":"lease_expired","call":{"call_succeeded":false}}'::JSONB
           AND terminal_response IS NULL) IS DISTINCT FROM 1::BIGINT
     OR EXISTS (
       SELECT 1 FROM (
         SELECT DISTINCT ON (call_identity) entry_kind
         FROM public.lab_arena_ledger
         WHERE round_id = v_round.round_id AND call_identity IS NOT NULL
         ORDER BY call_identity, entry_id DESC
       ) AS heads WHERE entry_kind IN ('reservation', 'dispatch')
     ) THEN
    RAISE EXCEPTION 'sep18 expired-position recovery accounting state differs';
  END IF;

  v_round_stable := to_jsonb(v_round) - 'status' - 'status_generation'
    - 'stage_generation' - 'cancel_reason' - 'updated_at';
  SELECT pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
      COALESCE(pg_catalog.jsonb_agg(to_jsonb(r) ORDER BY run_id), '[]'::JSONB)::TEXT,
      'UTF8'), 'sha256'), 'hex')
    INTO v_runs_hash
  FROM public.lab_arena_runs r WHERE round_id = v_round.round_id;
  SELECT pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
      COALESCE(pg_catalog.jsonb_agg(to_jsonb(s) ORDER BY submission_id), '[]'::JSONB)::TEXT,
      'UTF8'), 'sha256'), 'hex')
    INTO v_submissions_hash
  FROM public.lab_arena_submissions s WHERE round_id = v_round.round_id;
  SELECT pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
      COALESCE(pg_catalog.jsonb_agg(to_jsonb(l) ORDER BY entry_id), '[]'::JSONB)::TEXT,
      'UTF8'), 'sha256'), 'hex')
    INTO v_ledger_hash
  FROM public.lab_arena_ledger l WHERE round_id = v_round.round_id;

  INSERT INTO public.lab_arena_runs (
    run_id, assignment_id, round_id, submission_id, miner_hotkey, stage,
    icp_position, attempt, status, lease_generation, stage_generation, kind,
    previous_runner_hotkey
  ) VALUES (
    v_target.assignment_id || ':4', v_target.assignment_id, v_target.round_id,
    v_target.submission_id, v_target.miner_hotkey, v_target.stage,
    v_target.icp_position, 4, 'pending', v_target.lease_generation,
    v_resume_generation, v_target.kind, v_target.runner_hotkey
  );

  ALTER TABLE public.lab_arena_rounds
    DISABLE TRIGGER lab_arena_rounds_write_once;
  UPDATE public.lab_arena_rounds
  SET status = 'stage1', status_generation = 6,
      stage_generation = v_resume_generation, cancel_reason = NULL
  WHERE round_id = v_round.round_id;
  ALTER TABLE public.lab_arena_rounds
    ENABLE TRIGGER lab_arena_rounds_write_once;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id = v_round.round_id) IS DISTINCT FROM 117::BIGINT
     OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE run_id = v_target.assignment_id || ':4'
         AND assignment_id = v_target.assignment_id
         AND round_id = v_target.round_id
         AND submission_id = v_target.submission_id
         AND miner_hotkey = v_target.miner_hotkey
         AND stage = 2 AND icp_position = 12 AND attempt = 4
         AND status = 'pending' AND lease_generation = v_target.lease_generation
         AND stage_generation = v_resume_generation AND kind = 'execute'
         AND previous_runner_hotkey = v_target.runner_hotkey
     )
     OR (SELECT pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
           COALESCE(pg_catalog.jsonb_agg(to_jsonb(r) ORDER BY run_id), '[]'::JSONB)::TEXT,
           'UTF8'), 'sha256'), 'hex')
         FROM public.lab_arena_runs r
         WHERE round_id = v_round.round_id
           AND run_id <> v_target.assignment_id || ':4')
          IS DISTINCT FROM v_runs_hash
     OR (SELECT pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
           COALESCE(pg_catalog.jsonb_agg(to_jsonb(s) ORDER BY submission_id), '[]'::JSONB)::TEXT,
           'UTF8'), 'sha256'), 'hex')
         FROM public.lab_arena_submissions s WHERE round_id = v_round.round_id)
          IS DISTINCT FROM v_submissions_hash
     OR (SELECT pg_catalog.encode(extensions.digest(pg_catalog.convert_to(
           COALESCE(pg_catalog.jsonb_agg(to_jsonb(l) ORDER BY entry_id), '[]'::JSONB)::TEXT,
           'UTF8'), 'sha256'), 'hex')
         FROM public.lab_arena_ledger l WHERE round_id = v_round.round_id)
          IS DISTINCT FROM v_ledger_hash
     OR (SELECT to_jsonb(r) - 'status' - 'status_generation'
                - 'stage_generation' - 'cancel_reason' - 'updated_at'
         FROM public.lab_arena_rounds r WHERE round_id = v_round.round_id)
          IS DISTINCT FROM v_round_stable THEN
    RAISE EXCEPTION 'sep18 expired-position recovery changed preserved state';
  END IF;
END;
$recover_sep18_expired_position$;
COMMIT;
