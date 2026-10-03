-- An older unpromoted crown cannot block funding after a newer live day is
-- published in the same chain scope. Keep its promotion history unchanged.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

CREATE OR REPLACE FUNCTION public.lab_arena_freeze_champion_funding(p_round_id TEXT)
RETURNS JSONB LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public AS $fn$
DECLARE
  v_round public.lab_arena_rounds;
  v_submission public.lab_arena_submissions;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002'; END IF;
  IF v_round.champion_funding_frozen OR v_round.status <> 'open' THEN
    -- An already-running pre-migration round keeps its original funding.
    RETURN jsonb_build_object('status','existing');
  END IF;
  IF EXISTS (
    SELECT 1 FROM public.lab_arena_rounds AS prior
    WHERE prior.arena_network_name = v_round.arena_network_name
      AND prior.arena_netuid = v_round.arena_netuid
      AND prior.configuration_doc ->> 'mode' = v_round.configuration_doc ->> 'mode'
      AND (v_round.configuration_doc ->> 'mode' <> 'live' OR prior.rewards_enabled)
      AND prior.status = 'published' AND prior.promotion_required
      AND prior.publication_doc #>> '{king_decision,outcome}' = 'crowned'
      AND prior.baseline_promoted_at IS NULL
      AND (
        prior.configuration_doc ->> 'mode' <> 'live'
        OR NOT EXISTS (
          SELECT 1 FROM public.lab_arena_rounds AS newer
          WHERE newer.status = 'published'
            AND newer.configuration_doc ->> 'mode' = 'live'
            AND newer.arena_network_name = prior.arena_network_name
            AND newer.arena_netuid = prior.arena_netuid
            AND newer.evaluation_date > prior.evaluation_date
        )
      )
  ) THEN RETURN jsonb_build_object('status','promotion_pending'); END IF;
  -- Repository content/commit is deliberately not an ownership input.
  SELECT submission.* INTO v_submission
  FROM public.lab_arena_rounds AS prior
  JOIN public.lab_arena_submissions AS submission
    ON submission.submission_id = prior.publication_doc #>> '{king_decision,winner_submission_id}'
    AND submission.round_id = prior.round_id
    AND submission.miner_hotkey = prior.publication_doc #>> '{king_decision,king_hotkey}'
    AND submission.status = 'frozen' AND NOT submission.is_king
  WHERE prior.arena_network_name = v_round.arena_network_name
    AND prior.arena_netuid = v_round.arena_netuid
    AND prior.configuration_doc ->> 'mode' = v_round.configuration_doc ->> 'mode'
    AND (v_round.configuration_doc ->> 'mode' <> 'live' OR prior.rewards_enabled)
    AND prior.status = 'published' AND prior.baseline_promoted_at IS NOT NULL
    AND prior.publication_doc #>> '{king_decision,outcome}' = 'crowned'
    AND prior.round_id <> p_round_id
  ORDER BY prior.baseline_promoted_at DESC, prior.round_id DESC LIMIT 1;
  UPDATE public.lab_arena_rounds SET champion_funding_frozen = TRUE,
    champion_submission_id = v_submission.submission_id,
    champion_hotkey = v_submission.miner_hotkey
  WHERE round_id = p_round_id;
  RETURN jsonb_build_object('status','frozen');
END;
$fn$;
ALTER FUNCTION public.lab_arena_freeze_champion_funding(TEXT) OWNER TO lab_arena_owner;

COMMIT;
