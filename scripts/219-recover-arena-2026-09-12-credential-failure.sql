-- Recover the cancelled 2026-09-12 Arena round after migration 218 fixes the
-- cross-provider credential-refusal classification. This migration changes
-- only ten failed execution attempts that have same-run ledger evidence of
-- the proven miner-key 402 or a later budget refusal caused by that 402. It
-- preserves accepted outputs and all accounting, then uses the canonical
-- stage close function. Fresh databases and an already-progressed recovered
-- round are safe no-ops.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_submissions IN SHARE MODE;
LOCK TABLE public.lab_arena_ledger IN SHARE MODE;

DO $lab_arena_219_recover_20260912$
DECLARE
  v_round_id CONSTANT TEXT := 'arena-2026-09-12';
  v_failed_submission CONSTANT TEXT :=
    'sub-d39d12f74f5f6fb56837d22890c7dd94';
  v_preserved_submission CONSTANT TEXT :=
    'sub-ddd33c5c142f317e0279265f179b5bc1';
  v_expected_participant_ids CONSTANT TEXT[] := ARRAY[
    'baseline-2026-09-12',
    'sub-1460c4e33f19f923b969ef2fd941072f',
    'sub-64750217950f089a3bec21ee6fe72d6e',
    'sub-67fdb5d43bc8f9ea4cb6e74df4797037',
    'sub-c2ab2d55f187807c2793433701485825',
    'sub-d39d12f74f5f6fb56837d22890c7dd94',
    'sub-ddd33c5c142f317e0279265f179b5bc1',
    'sub-eb8a764bc501efb7d47f9c6c74d49756'
  ]::TEXT[];
  v_credential_failure_at CONSTANT TIMESTAMPTZ :=
    '2026-09-12T00:22:34.933602Z'::TIMESTAMPTZ;
  v_repair_run_ids CONSTANT TEXT[] := ARRAY[
    'arena-2026-09-12:sub-d39d12f74f5f6fb56837d22890c7dd94:1:4:2',
    'arena-2026-09-12:sub-d39d12f74f5f6fb56837d22890c7dd94:1:5:2',
    'arena-2026-09-12:sub-d39d12f74f5f6fb56837d22890c7dd94:1:6:1',
    'arena-2026-09-12:sub-d39d12f74f5f6fb56837d22890c7dd94:1:6:2',
    'arena-2026-09-12:sub-d39d12f74f5f6fb56837d22890c7dd94:1:7:1',
    'arena-2026-09-12:sub-d39d12f74f5f6fb56837d22890c7dd94:1:7:2',
    'arena-2026-09-12:sub-d39d12f74f5f6fb56837d22890c7dd94:1:8:1',
    'arena-2026-09-12:sub-d39d12f74f5f6fb56837d22890c7dd94:1:8:2',
    'arena-2026-09-12:sub-d39d12f74f5f6fb56837d22890c7dd94:1:9:1',
    'arena-2026-09-12:sub-d39d12f74f5f6fb56837d22890c7dd94:1:9:2'
  ]::TEXT[];
  v_round public.lab_arena_rounds%ROWTYPE;
  v_close JSONB;
  v_participant_ids TEXT[];
  v_run_id TEXT;
  v_old_count INTEGER;
  v_new_count INTEGER;
  v_count INTEGER;
  v_ledger_count BIGINT;
  v_ledger_amount NUMERIC;
  v_ledger_hash TEXT;
  v_accepted_hash TEXT;
  v_round_stable JSONB;
  v_reserve_definition TEXT;
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_close_stage(text,smallint)'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply the current Arena schema before migration 219';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(procedure.oid)
  INTO v_reserve_definition
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena_reserve_call'
    AND procedure.pronargs = 9;
  IF v_reserve_definition IS NULL
     OR pg_catalog.strpos(
       v_reserve_definition, 'v_prior_miner_credential_refusal'
     ) = 0
     OR pg_catalog.strpos(
       v_reserve_definition, 'AND ledger.provider = p_provider'
     ) > 0 THEN
    RAISE EXCEPTION
      'apply 218-lab-arena-cross-provider-credential-refusal.sql first';
  END IF;

  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = v_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RETURN;
  END IF;

  IF NOT EXISTS (
       SELECT 1
       FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_runs'::pg_catalog.regclass
         AND tgname = 'lab_arena_runs_terminal'
         AND tgenabled = 'O'
         AND NOT tgisinternal
     )
     OR NOT EXISTS (
       SELECT 1
       FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
         AND tgname = 'lab_arena_rounds_write_once'
         AND tgenabled = 'O'
         AND NOT tgisinternal
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-12 recovery guard state differs';
  END IF;

  IF pg_catalog.jsonb_typeof(v_round.participants) IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(v_round.participants) <> 8 THEN
    RAISE EXCEPTION 'arena 2026-09-12 participant count differs';
  END IF;
  SELECT pg_catalog.array_agg(item ->> 'submission_id' ORDER BY item ->> 'submission_id')
  INTO v_participant_ids
  FROM pg_catalog.jsonb_array_elements(v_round.participants) AS item;
  IF v_participant_ids IS DISTINCT FROM v_expected_participant_ids THEN
    RAISE EXCEPTION 'arena 2026-09-12 participant identities differ';
  END IF;
  IF (
    SELECT pg_catalog.array_agg(submission_id ORDER BY submission_id)
    FROM public.lab_arena_submissions
    WHERE round_id = v_round_id AND status = 'frozen'
  ) IS DISTINCT FROM v_participant_ids THEN
    RAISE EXCEPTION 'arena 2026-09-12 frozen participants differ';
  END IF;
  IF EXISTS (
    SELECT 1
    FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
    LEFT JOIN public.lab_arena_submissions AS submission
      ON submission.submission_id = participant ->> 'submission_id'
     AND submission.round_id = v_round_id
    WHERE submission.submission_id IS NULL
       OR submission.status <> 'frozen'
       OR submission.miner_hotkey IS DISTINCT FROM
          participant ->> 'miner_hotkey'
       OR submission.is_king IS DISTINCT FROM COALESCE(
          (participant ->> 'is_king')::BOOLEAN, FALSE
       )
  ) OR (
    SELECT pg_catalog.count(*)
    FROM public.lab_arena_submissions
    WHERE round_id = v_round_id AND is_king
  ) <> 1 OR NOT EXISTS (
    SELECT 1 FROM public.lab_arena_submissions
    WHERE submission_id = 'baseline-2026-09-12'
      AND round_id = v_round_id AND status = 'frozen' AND is_king
  ) THEN
    RAISE EXCEPTION 'arena 2026-09-12 participant freeze state differs';
  END IF;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id = v_round_id AND stage = 1 AND kind = 'execute') <> 87
     OR (SELECT pg_catalog.count(DISTINCT assignment_id)
         FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'execute') <> 80
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round_id AND stage = 1 AND kind = 'execute'
         AND (attempt NOT IN (1, 2)
              OR submission_id <> ALL(v_participant_ids))
     )
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'execute'
           AND attempt = 2) <> 7
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'execute'
           AND status = 'accepted'
           AND terminal_cause = 'accepted'
           AND result_doc ->> 'terminal_status' = 'accepted') <> 73
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'execute'
           AND submission_id = v_preserved_submission
           AND status = 'accepted' AND terminal_cause = 'accepted'
           AND result_doc ->> 'terminal_status' = 'accepted') <> 10
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'execute'
           AND terminal_cause = 'model_error'
           AND result_doc ->> 'terminal_status' = 'model_error') <> 3
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs AS run
       JOIN pg_catalog.jsonb_array_elements(v_round.participants) AS participant
         ON participant ->> 'submission_id' = run.submission_id
       WHERE run.round_id = v_round_id
         AND run.miner_hotkey IS DISTINCT FROM participant ->> 'miner_hotkey'
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-12 execution evidence differs';
  END IF;

  SELECT pg_catalog.count(*) FILTER (
           WHERE terminal_cause = 'provider_error'
             AND result_doc ->> 'terminal_status' = 'provider_error'
         ),
         pg_catalog.count(*) FILTER (
           WHERE terminal_cause = 'credential_error'
             AND result_doc ->> 'terminal_status' = 'credential_error'
         )
  INTO v_old_count, v_new_count
  FROM public.lab_arena_runs
  WHERE run_id = ANY(v_repair_run_ids);
  IF v_old_count + v_new_count <> 10
     OR (v_old_count <> 10 AND v_new_count <> 10) THEN
    RAISE EXCEPTION 'arena 2026-09-12 correction state differs';
  END IF;
  IF EXISTS (
    SELECT 1 FROM public.lab_arena_runs
    WHERE run_id = ANY(v_repair_run_ids)
      AND (round_id <> v_round_id OR submission_id <> v_failed_submission
           OR stage <> 1 OR kind <> 'execute' OR status <> 'failed'
           OR run_id <> assignment_id || ':' || attempt::TEXT
           OR updated_at < v_credential_failure_at)
  ) THEN
    RAISE EXCEPTION 'arena 2026-09-12 correction row identity differs';
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM public.lab_arena_runs
    WHERE run_id =
      'arena-2026-09-12:sub-d39d12f74f5f6fb56837d22890c7dd94:1:2:2'
      AND status = 'failed' AND kind = 'execute' AND stage = 1
      AND icp_position = 2 AND attempt = 2
      AND terminal_cause = 'provider_error'
      AND result_doc ->> 'terminal_status' = 'provider_error'
      AND created_at < v_credential_failure_at
  ) THEN
    RAISE EXCEPTION 'arena 2026-09-12 pre-credential failure changed';
  END IF;

  WITH heads AS (
    SELECT DISTINCT ON (ledger.call_identity) ledger.*
    FROM public.lab_arena_ledger AS ledger
    JOIN public.lab_arena_runs AS run ON run.run_id = ledger.run_id
    WHERE ledger.round_id = v_round_id
      AND ledger.submission_id = v_failed_submission
      AND run.kind = 'execute'
      AND ledger.call_identity IS NOT NULL
    ORDER BY ledger.call_identity, ledger.entry_id DESC
  )
  SELECT pg_catalog.count(*) INTO v_count
  FROM heads
  WHERE entry_kind = 'uncertain'
    AND provider = 'deepline'
    AND funding_source = 'miner_key'
    AND amount_microusd = 46218434
    AND created_at = v_credential_failure_at
    AND entry_doc #>> '{call,reason}' = 'missing_provider_cost'
    AND entry_doc #>> '{call,provider_status}' = '402';
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'arena 2026-09-12 credential failure proof differs';
  END IF;

  FOREACH v_run_id IN ARRAY v_repair_run_ids LOOP
    IF NOT EXISTS (
      WITH heads AS (
        SELECT DISTINCT ON (ledger.call_identity) ledger.*
        FROM public.lab_arena_ledger AS ledger
        WHERE ledger.run_id = v_run_id
          AND ledger.call_identity IS NOT NULL
        ORDER BY ledger.call_identity, ledger.entry_id DESC
      )
      SELECT 1 FROM heads
      WHERE funding_source = 'miner_key'
        AND (
          (entry_kind = 'uncertain' AND provider = 'deepline'
           AND amount_microusd = 46218434
           AND created_at = v_credential_failure_at
           AND entry_doc #>> '{call,reason}' = 'missing_provider_cost'
           AND entry_doc #>> '{call,provider_status}' = '402')
          OR
          (entry_kind = 'refusal' AND created_at >= v_credential_failure_at
           AND entry_doc ->> 'reason' IN (
             'money_cap', 'provider_cost_uncertain'
           ))
        )
    ) THEN
      RAISE EXCEPTION
        'arena 2026-09-12 run lacks credential failure proof: %', v_run_id;
    END IF;
  END LOOP;

  SELECT pg_catalog.count(*), COALESCE(pg_catalog.sum(amount_microusd), 0),
         pg_catalog.md5(COALESCE(pg_catalog.string_agg(
           pg_catalog.to_jsonb(ledger)::TEXT, '|' ORDER BY entry_id
         ), ''))
  INTO v_ledger_count, v_ledger_amount, v_ledger_hash
  FROM public.lab_arena_ledger AS ledger
  WHERE round_id = v_round_id;
  SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.to_jsonb(run)::TEXT, '|' ORDER BY run_id
         ))
  INTO v_accepted_hash
  FROM public.lab_arena_runs AS run
  WHERE round_id = v_round_id AND status = 'accepted';
  v_round_stable := pg_catalog.to_jsonb(v_round)
    - 'status' - 'status_generation' - 'stage_generation'
    - 'cancel_reason' - 'updated_at';

  IF v_new_count = 10 THEN
    IF v_round.status = 'cancelled' THEN
      RAISE EXCEPTION 'arena 2026-09-12 recovered rows have stale cancellation';
    END IF;
    IF v_round.status NOT IN (
         'stage1_closed', 'stage1_scoring', 'stage1_judged', 'stage1_scored',
         'stage2', 'stage2_closed', 'stage2_scoring', 'stage2_judged',
         'scored', 'published'
       ) THEN
      RAISE EXCEPTION 'arena 2026-09-12 recovered status differs';
    END IF;
    RETURN;
  END IF;

  IF v_round.status <> 'cancelled'
     OR v_round.cancel_reason <> 'execution_incomplete:stage1:4'
     OR v_round.stage1_scoring_plan_doc IS NOT NULL
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR v_round.stage3_scoring_plan_doc IS NOT NULL
     OR v_round.finalists IS NOT NULL
     OR v_round.confirmation_cohort IS NOT NULL
     OR v_round.publication_doc IS NOT NULL
     OR v_round.king_outcome IS NOT NULL
     OR v_round.published_at IS NOT NULL
     OR v_round.reward_basis_hash IS NOT NULL
     OR v_round.reward_basis_doc IS NOT NULL
     OR v_round.signing_key_doc IS NOT NULL
     OR v_round.reward_activated_at IS NOT NULL
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round_id
         AND (kind = 'score' OR status IN ('pending', 'leased', 'submitted'))
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = v_round_id
         AND (stage <> 1 OR kind <> 'execute')
     )
     OR EXISTS (
       SELECT 1 FROM (
         SELECT DISTINCT ON (call_identity) entry_kind
         FROM public.lab_arena_ledger
         WHERE round_id = v_round_id AND call_identity IS NOT NULL
         ORDER BY call_identity, entry_id DESC
       ) AS head
       WHERE entry_kind IN ('reservation', 'dispatch')
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-12 recovery preflight differs';
  END IF;

  ALTER TABLE public.lab_arena_runs
    DISABLE TRIGGER lab_arena_runs_terminal;
  UPDATE public.lab_arena_runs
  SET terminal_cause = 'credential_error',
      result_doc = pg_catalog.jsonb_set(
        result_doc, '{terminal_status}',
        pg_catalog.to_jsonb('credential_error'::TEXT), FALSE
      )
  WHERE run_id = ANY(v_repair_run_ids)
    AND status = 'failed'
    AND kind = 'execute'
    AND stage = 1
    AND submission_id = v_failed_submission
    AND terminal_cause = 'provider_error'
    AND result_doc ->> 'terminal_status' = 'provider_error';
  GET DIAGNOSTICS v_count = ROW_COUNT;
  ALTER TABLE public.lab_arena_runs
    ENABLE TRIGGER lab_arena_runs_terminal;
  IF v_count <> 10 THEN
    RAISE EXCEPTION 'arena 2026-09-12 corrected attempt count differs';
  END IF;

  ALTER TABLE public.lab_arena_rounds
    DISABLE TRIGGER lab_arena_rounds_write_once;
  UPDATE public.lab_arena_rounds
  SET status = 'stage1', cancel_reason = NULL
  WHERE round_id = v_round_id
    AND status = 'cancelled'
    AND cancel_reason = 'execution_incomplete:stage1:4';
  GET DIAGNOSTICS v_count = ROW_COUNT;
  ALTER TABLE public.lab_arena_rounds
    ENABLE TRIGGER lab_arena_rounds_write_once;
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'arena 2026-09-12 restart state differs';
  END IF;

  v_close := public.lab_arena_close_stage(v_round_id, 1::SMALLINT);
  IF v_close ->> 'status' <> 'closed'
     OR v_close ->> 'round_status' <> 'stage1_closed'
     OR (v_close ->> 'incomplete_assignments')::INTEGER <> 0 THEN
    RAISE EXCEPTION 'arena 2026-09-12 canonical close failed: %', v_close;
  END IF;

  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
      WHERE round_id = v_round_id AND stage = 1 AND kind = 'execute') <> 87
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND stage = 1 AND kind = 'execute'
           AND status = 'accepted') <> 73
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs
         WHERE round_id = v_round_id AND run_id = ANY(v_repair_run_ids)
           AND status = 'failed' AND terminal_cause = 'credential_error'
           AND result_doc ->> 'terminal_status' = 'credential_error') <> 10
     OR (SELECT ROW(pg_catalog.count(*),
                       COALESCE(pg_catalog.sum(amount_microusd), 0),
                pg_catalog.md5(COALESCE(pg_catalog.string_agg(
                  pg_catalog.to_jsonb(ledger)::TEXT, '|' ORDER BY entry_id
                ), '')))
         FROM public.lab_arena_ledger AS ledger
         WHERE round_id = v_round_id)
        IS DISTINCT FROM ROW(v_ledger_count, v_ledger_amount, v_ledger_hash)
     OR (SELECT pg_catalog.md5(pg_catalog.string_agg(
           pg_catalog.to_jsonb(run)::TEXT, '|' ORDER BY run_id
         )) FROM public.lab_arena_runs AS run
         WHERE round_id = v_round_id AND status = 'accepted')
        IS DISTINCT FROM v_accepted_hash
     OR (SELECT pg_catalog.to_jsonb(round_row)
           - 'status' - 'status_generation' - 'stage_generation'
           - 'cancel_reason' - 'updated_at'
         FROM public.lab_arena_rounds AS round_row
         WHERE round_id = v_round_id) IS DISTINCT FROM v_round_stable
     OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_rounds
       WHERE round_id = v_round_id AND status = 'stage1_closed'
         AND cancel_reason IS NULL
     )
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_runs'::pg_catalog.regclass
         AND tgname = 'lab_arena_runs_terminal' AND tgenabled = 'O'
     )
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_trigger
       WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
         AND tgname = 'lab_arena_rounds_write_once' AND tgenabled = 'O'
     ) THEN
    RAISE EXCEPTION 'arena 2026-09-12 recovery postcondition differs';
  END IF;
END;
$lab_arena_219_recover_20260912$;

COMMIT;
