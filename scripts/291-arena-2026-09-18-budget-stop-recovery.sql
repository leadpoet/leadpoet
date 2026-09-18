-- Resume only the six unfinished Sep18 assignments after deploying the
-- per-ICP budget-stop classification fix. Keep every prior attempt and charge.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN SHARE ROW EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_ledger IN SHARE ROW EXCLUSIVE MODE;

DO $recover_sep18_unfinished$
DECLARE
  v_round public.lab_arena_rounds;
  v_prior public.lab_arena_runs;
  v_positions INTEGER[] := ARRAY[10,11,12,13,14,19];
  v_position INTEGER;
  v_retry_count INTEGER;
  v_count INTEGER;
  v_stable_round JSONB;
  v_runs JSONB;
  v_submissions JSONB;
  v_ledger_count BIGINT;
  v_ledger_max BIGINT;
  v_ledger_sum NUMERIC;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-18' FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'sep18 recovery round missing';
  END IF;
  IF v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
       IS DISTINCT FROM 'successful_calls_per_icp_v1'
     OR (v_round.configuration_doc ->> 'execution_icp_cap_microusd')::BIGINT
       IS DISTINCT FROM 4000000
     OR (v_round.configuration_doc ->> 'cost_per_company_microusd')::BIGINT
       IS DISTINCT FROM 800000
     OR v_round.configuration_doc ->> 'parallel_twenty_icp_execution'
       IS DISTINCT FROM 'true'
     OR pg_catalog.jsonb_array_length(v_round.participants) IS DISTINCT FROM 5
     OR v_round.benchmark_ref IS DISTINCT FROM
       'arena/arena-2026-09-18/benchmark.json'
     OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_submissions
       WHERE submission_id = 'baseline-2026-09-18'
         AND round_id = v_round.round_id AND is_king
         AND source_ref =
           'arena/arena-2026-09-18/sources/baseline-2026-09-18.tar.gz'
         AND source_size_bytes = 604847
     ) THEN
    RAISE EXCEPTION 'sep18 recovery source or policy differs';
  END IF;

  SELECT count(*) INTO v_retry_count FROM public.lab_arena_runs
  WHERE round_id = v_round.round_id AND submission_id = 'baseline-2026-09-18'
    AND kind = 'execute' AND stage = 2 AND icp_position = ANY(v_positions)
    AND attempt = 3
    AND assignment_id = 'arena-2026-09-18:baseline-2026-09-18:2:'
      || icp_position::TEXT
    AND run_id = assignment_id || ':3';
  IF v_retry_count = 6 THEN
    -- An applied recovery must never reset work that has since progressed.
    IF v_round.status NOT IN (
      'stage1','stage1_closed','stage1_scoring','stage1_judged','stage1_scored',
      'stage2','stage2_closed','stage2_scoring','stage2_judged','scored','published'
    ) THEN
      RAISE EXCEPTION 'sep18 recovery replay state differs';
    END IF;
    RETURN;
  END IF;
  IF v_retry_count <> 0 OR v_round.status IS DISTINCT FROM 'cancelled'
     OR v_round.cancel_reason IS DISTINCT FROM 'execution_incomplete:stage1:6'
     OR v_round.status_generation IS DISTINCT FROM 3
     OR v_round.stage_generation IS DISTINCT FROM 2
     OR v_round.stage1_scoring_plan_doc IS NOT NULL
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR v_round.finalists IS NOT NULL
     OR v_round.publication_doc IS NOT NULL
     OR v_round.published_at IS NOT NULL
     OR v_round.reward_basis_doc IS NOT NULL
     OR v_round.reward_basis_hash IS NOT NULL
     OR v_round.reward_activated_at IS NOT NULL
     OR v_round.signing_key_doc IS NOT NULL
     OR v_round.effective_reward_epoch IS NOT NULL THEN
    RAISE EXCEPTION 'sep18 recovery terminal state differs';
  END IF;
  IF (SELECT count(*) FROM public.lab_arena_submissions
      WHERE round_id = v_round.round_id AND status = 'frozen') <> 5
     OR EXISTS (
       SELECT 1 FROM pg_catalog.jsonb_array_elements(v_round.participants) p
       WHERE NOT EXISTS (
         SELECT 1 FROM public.lab_arena_submissions s
         WHERE s.round_id = v_round.round_id
           AND s.submission_id = p ->> 'submission_id'
           AND s.miner_hotkey = p ->> 'miner_hotkey'
           AND s.is_king = COALESCE((p ->> 'is_king')::BOOLEAN,FALSE)
       )
     )
     OR (SELECT count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round.round_id) <> 110
     OR (SELECT count(DISTINCT (submission_id,icp_position))
         FROM public.lab_arena_runs WHERE round_id = v_round.round_id) <> 100
     OR (SELECT count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round.round_id AND status = 'accepted') <> 93
     OR (SELECT count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round.round_id AND status = 'accepted'
           AND submission_id <> 'baseline-2026-09-18') <> 80
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs WHERE round_id = v_round.round_id
         AND (kind <> 'execute' OR status NOT IN ('accepted','failed')
              OR per_icp_score IS NOT NULL OR qualification_doc IS NOT NULL)
     )
     OR EXISTS (
       SELECT 1 FROM (
         SELECT DISTINCT ON (run_id,call_identity) entry_kind
         FROM public.lab_arena_ledger
         WHERE round_id = v_round.round_id AND call_identity IS NOT NULL
         ORDER BY run_id,call_identity,entry_id DESC
       ) heads WHERE entry_kind IN ('reservation','dispatch')
     ) THEN
    RAISE EXCEPTION 'sep18 recovery execution snapshot differs';
  END IF;
  FOREACH v_position IN ARRAY v_positions LOOP
    SELECT * INTO v_prior FROM public.lab_arena_runs
    WHERE run_id = 'arena-2026-09-18:baseline-2026-09-18:2:'
      || v_position::TEXT || ':2';
    IF NOT FOUND OR v_prior.round_id <> v_round.round_id
       OR v_prior.submission_id <> 'baseline-2026-09-18'
       OR v_prior.kind <> 'execute' OR v_prior.stage <> 2
       OR v_prior.icp_position <> v_position OR v_prior.attempt <> 2
       OR v_prior.status <> 'failed' OR v_prior.terminal_cause <> 'provider_error'
       OR v_prior.result_doc ->> 'terminal_status' IS DISTINCT FROM 'provider_error'
       OR v_prior.output_ref IS NOT NULL
       OR v_prior.champion_restart_required
       OR v_prior.assignment_id IS DISTINCT FROM
          'arena-2026-09-18:baseline-2026-09-18:2:' || v_position::TEXT
       OR (SELECT count(*) FROM public.lab_arena_runs
           WHERE assignment_id = v_prior.assignment_id) <> 2 THEN
      RAISE EXCEPTION 'sep18 recovery unfinished attempt differs';
    END IF;
    IF v_position <> 12 AND NOT EXISTS (
      SELECT 1 FROM public.lab_arena_ledger
      WHERE run_id = v_prior.run_id AND entry_kind = 'refusal'
        AND entry_doc ->> 'reason' = 'money_cap'
    ) THEN
      RAISE EXCEPTION 'sep18 recovery budget-stop evidence missing';
    END IF;
  END LOOP;
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_trigger
    WHERE tgrelid = 'public.lab_arena_rounds'::regclass
      AND tgname = 'lab_arena_rounds_write_once' AND tgenabled = 'O'
  ) THEN
    RAISE EXCEPTION 'sep18 recovery round guard is not enabled';
  END IF;

  SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id) INTO v_runs
  FROM public.lab_arena_runs r WHERE round_id = v_round.round_id;
  SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id) INTO v_submissions
  FROM public.lab_arena_submissions s WHERE round_id = v_round.round_id;
  SELECT count(*),max(entry_id),sum(amount_microusd)
    INTO v_ledger_count,v_ledger_max,v_ledger_sum
  FROM public.lab_arena_ledger WHERE round_id = v_round.round_id;
  v_stable_round := to_jsonb(v_round) - 'status' - 'status_generation'
    - 'stage_generation' - 'cancel_reason' - 'updated_at';

  INSERT INTO public.lab_arena_runs (
    run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
    icp_position,attempt,status,lease_generation,stage_generation,kind,
    previous_runner_hotkey
  )
  SELECT assignment_id || ':3',assignment_id,round_id,submission_id,
    miner_hotkey,stage,icp_position,3,'pending',lease_generation,3,'execute',
    runner_hotkey
  FROM public.lab_arena_runs
  WHERE round_id = v_round.round_id AND submission_id = 'baseline-2026-09-18'
    AND kind = 'execute' AND stage = 2 AND icp_position = ANY(v_positions)
    AND attempt = 2;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  IF v_count <> 6 THEN RAISE EXCEPTION 'sep18 recovery retry count differs'; END IF;

  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once;
  UPDATE public.lab_arena_rounds
  SET status = 'stage1',status_generation = 4,stage_generation = 3,
      cancel_reason = NULL
  WHERE round_id = v_round.round_id;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once;

  IF (SELECT jsonb_agg(to_jsonb(r) ORDER BY run_id)
      FROM public.lab_arena_runs r
      WHERE round_id = v_round.round_id AND attempt < 3) IS DISTINCT FROM v_runs
     OR (SELECT jsonb_agg(to_jsonb(s) ORDER BY submission_id)
         FROM public.lab_arena_submissions s
         WHERE round_id = v_round.round_id) IS DISTINCT FROM v_submissions
     OR (SELECT ROW(count(*),max(entry_id),sum(amount_microusd))
         FROM public.lab_arena_ledger WHERE round_id = v_round.round_id)
          IS DISTINCT FROM ROW(v_ledger_count,v_ledger_max,v_ledger_sum)
     OR (SELECT to_jsonb(r) - 'status' - 'status_generation'
                - 'stage_generation' - 'cancel_reason' - 'updated_at'
         FROM public.lab_arena_rounds r WHERE round_id = v_round.round_id)
          IS DISTINCT FROM v_stable_round THEN
    RAISE EXCEPTION 'sep18 recovery changed preserved state';
  END IF;
END;
$recover_sep18_unfinished$;
COMMIT;
