-- Apply the authorized eight-point promotion policy to uncommitted live rounds.
-- New rounds receive the same default from contracts.DEFAULT_PROMOTION_MARGIN.
-- Submitted sources, grading, published outcomes and reward state are unchanged.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

LOCK TABLE public.lab_arena_rounds IN SHARE ROW EXCLUSIVE MODE;

DO $promotion_margin_cutover$
DECLARE
  v_round public.lab_arena_rounds;
  v_after public.lab_arena_rounds;
  v_configuration JSONB;
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_trigger
    WHERE tgrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
      AND tgname = 'lab_arena_rounds_write_once'
      AND NOT tgisinternal AND tgenabled = 'O'
  ) THEN
    RAISE EXCEPTION 'Arena round write-once trigger must be enabled';
  END IF;

  ALTER TABLE public.lab_arena_rounds
    DISABLE TRIGGER lab_arena_rounds_write_once;

  FOR v_round IN
    SELECT * FROM public.lab_arena_rounds
    WHERE status = 'open'
      AND configuration_doc ->> 'mode' = 'live'
      AND COALESCE(configuration_doc ->> 'network_name', 'finney') = 'finney'
      AND COALESCE((configuration_doc ->> 'netuid')::INTEGER, 71) = 71
      AND configuration_doc -> 'promotion_margin' = '0.5'::JSONB
      AND (configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ
          > pg_catalog.clock_timestamp()
    FOR UPDATE
  LOOP
    -- Lock the round before checking child work. Registration and cutoff use
    -- the same row lock, so no work can start during this configuration change.
    IF v_round.benchmark_ref IS NOT NULL
       OR (v_round.participants IS NOT NULL
           AND v_round.participants <> '[]'::JSONB)
       OR v_round.publication_doc IS NOT NULL
       OR v_round.published_at IS NOT NULL
       OR v_round.promotion_doc IS NOT NULL
       OR v_round.baseline_promoted_at IS NOT NULL
       OR v_round.reward_basis_doc IS NOT NULL
       OR v_round.reward_basis_hash IS NOT NULL
       OR v_round.reward_activated_at IS NOT NULL
       OR v_round.effective_reward_epoch IS NOT NULL
       OR v_round.signing_key_doc IS NOT NULL
       OR v_round.king_start_epoch IS NOT NULL
       OR v_round.king_outcome IS NOT NULL
       OR v_round.king_hotkey IS NOT NULL
       OR EXISTS (SELECT 1 FROM public.lab_arena_runs
                  WHERE round_id = v_round.round_id)
       OR EXISTS (SELECT 1 FROM public.lab_arena_submissions
                  WHERE round_id = v_round.round_id
                    AND (status = 'frozen' OR is_king)) THEN
      CONTINUE;
    END IF;

    v_configuration := pg_catalog.jsonb_set(
      v_round.configuration_doc, '{promotion_margin}', '8'::JSONB, FALSE
    );
    UPDATE public.lab_arena_rounds
      SET configuration_doc = v_configuration,
          updated_at = pg_catalog.clock_timestamp()
      WHERE round_id = v_round.round_id;
    SELECT * INTO STRICT v_after FROM public.lab_arena_rounds
      WHERE round_id = v_round.round_id;
    IF v_after.configuration_doc IS DISTINCT FROM v_configuration
       OR (pg_catalog.to_jsonb(v_after) - 'configuration_doc' - 'updated_at')
          IS DISTINCT FROM
          (pg_catalog.to_jsonb(v_round) - 'configuration_doc' - 'updated_at') THEN
      RAISE EXCEPTION 'promotion policy cutover changed protected round state';
    END IF;
  END LOOP;
  ALTER TABLE public.lab_arena_rounds
    ENABLE TRIGGER lab_arena_rounds_write_once;
END;
$promotion_margin_cutover$;

COMMIT;
