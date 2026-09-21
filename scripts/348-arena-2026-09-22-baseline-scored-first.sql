-- Opt an already-created, still-unstarted Sep22 round into baseline-first
-- execution. If the row does not exist yet, the service default creates it
-- with the same policy.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.lab_arena_runs IN SHARE ROW EXCLUSIVE MODE;

DO $sep22_baseline_first$
DECLARE
  v_round public.lab_arena_rounds;
  v_after public.lab_arena_rounds;
  v_expected JSONB;
  v_count INTEGER;
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_trigger
    WHERE tgrelid = 'public.lab_arena_rounds'::REGCLASS
      AND tgname = 'lab_arena_rounds_write_once'
      AND NOT tgisinternal AND tgenabled = 'O'
  ) THEN
    RAISE EXCEPTION 'Arena round write-once trigger is not enabled';
  END IF;

  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-22'
  FOR UPDATE;
  IF NOT FOUND THEN
    RETURN;
  END IF;
  IF v_round.configuration_doc ->> 'round_id'
       IS DISTINCT FROM 'arena-2026-09-22'
     OR v_round.configuration_doc ->> 'mode' IS DISTINCT FROM 'live'
     OR v_round.configuration_doc #>> '{schedule,submission_cutoff}'
       IS DISTINCT FROM '2026-09-22T00:00:00Z' THEN
    RAISE EXCEPTION 'Sep22 baseline-first transition has unexpected configuration';
  END IF;

  IF v_round.configuration_doc ->> 'execution_sequence_policy'
       = 'baseline_scored_first_v1'
     AND NOT (v_round.configuration_doc ? 'parallel_twenty_icp_execution') THEN
    RETURN;
  END IF;
  IF v_round.status IS DISTINCT FROM 'open'
     OR v_round.configuration_doc ->> 'execution_sequence_policy' IS NOT NULL
     OR (v_round.configuration_doc ->> 'parallel_twenty_icp_execution')::BOOLEAN
          IS DISTINCT FROM TRUE
     OR v_round.benchmark_ref IS NOT NULL
     OR v_round.participants IS NOT NULL
     OR v_round.stage1_scoring_plan_doc IS NOT NULL
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR v_round.stage3_scoring_plan_doc IS NOT NULL
     OR v_round.finalists IS NOT NULL
     OR v_round.publication_doc IS NOT NULL
     OR v_round.published_at IS NOT NULL
     OR v_round.cancel_reason IS NOT NULL
     OR v_round.king_outcome IS NOT NULL
     OR v_round.king_hotkey IS NOT NULL
     OR v_round.king_start_epoch IS NOT NULL
     OR v_round.effective_reward_epoch IS NOT NULL
     OR v_round.reward_basis_hash IS NOT NULL
     OR v_round.reward_basis_doc IS NOT NULL
     OR v_round.signing_key_doc IS NOT NULL
     OR v_round.reward_activated_at IS NOT NULL
     OR v_round.promotion_doc IS NOT NULL
     OR v_round.baseline_promoted_at IS NOT NULL
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_runs
       WHERE round_id = 'arena-2026-09-22'
     ) THEN
    RAISE EXCEPTION 'Sep22 round is not parallel, open, and unstarted';
  END IF;

  v_expected := (
    v_round.configuration_doc - 'parallel_twenty_icp_execution'
  ) || pg_catalog.jsonb_build_object(
    'execution_sequence_policy', 'baseline_scored_first_v1'
  );
  ALTER TABLE public.lab_arena_rounds
    DISABLE TRIGGER lab_arena_rounds_write_once;
  UPDATE public.lab_arena_rounds
  SET configuration_doc = v_expected,
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = 'arena-2026-09-22';
  GET DIAGNOSTICS v_count = ROW_COUNT;
  ALTER TABLE public.lab_arena_rounds
    ENABLE TRIGGER lab_arena_rounds_write_once;
  IF v_count <> 1 THEN
    RAISE EXCEPTION 'Sep22 baseline-first update count differs';
  END IF;

  SELECT * INTO v_after
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-22';
  IF v_after.configuration_doc IS DISTINCT FROM v_expected
     OR (pg_catalog.to_jsonb(v_after) - 'configuration_doc' - 'updated_at')
          IS DISTINCT FROM
        (pg_catalog.to_jsonb(v_round) - 'configuration_doc' - 'updated_at') THEN
    RAISE EXCEPTION 'Sep22 baseline-first transition changed protected state';
  END IF;
END;
$sep22_baseline_first$;

COMMIT;
