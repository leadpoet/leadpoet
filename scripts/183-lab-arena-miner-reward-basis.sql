-- 183-lab-arena-miner-reward-basis.sql
-- Keep the organizer baseline as a daily score threshold, never a reward payee.

BEGIN;

-- Required only for ownership transfers by the hosted migration role.
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- New publications have exactly two outcomes: a miner strictly beat the
-- baseline, or no miner did. The stored round participants keep the internal
-- is_king flag for migration compatibility, but public documents call it
-- is_baseline.
CREATE OR REPLACE FUNCTION public.lab_arena_publication_baseline_guard_v1()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $lab_arena_publication_baseline_guard$
DECLARE
  v_decision JSONB;
  v_baseline_id TEXT;
  v_baseline_count BIGINT;
  v_public_baseline_count BIGINT;
  v_baseline_ranking JSONB;
  v_winner_ranking JSONB;
  v_winner_participant JSONB;
  v_baseline_score NUMERIC;
  v_winner_score NUMERIC;
  v_winner_id TEXT;
BEGIN
  IF NEW.status <> 'published' OR OLD.status = 'published' THEN
    RETURN NEW;
  END IF;

  SELECT pg_catalog.count(*), pg_catalog.min(participant ->> 'submission_id')
  INTO v_baseline_count, v_baseline_id
  FROM pg_catalog.jsonb_array_elements(COALESCE(NEW.participants, '[]'::JSONB)) AS participant
  WHERE COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE);
  IF v_baseline_count <> 1 OR COALESCE(v_baseline_id, '') = '' THEN
    RAISE EXCEPTION 'lab_arena_publication_baseline_invalid' USING ERRCODE = '22023';
  END IF;

  IF pg_catalog.jsonb_typeof(NEW.publication_doc -> 'participants') IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_typeof(NEW.publication_doc -> 'final_ranking') IS DISTINCT FROM 'array'
     OR EXISTS (
       SELECT 1
       FROM pg_catalog.jsonb_array_elements(NEW.publication_doc -> 'participants') AS participant
       WHERE participant ? 'is_king' OR NOT participant ? 'is_baseline'
     )
     OR EXISTS (
       SELECT 1
       FROM pg_catalog.jsonb_array_elements(NEW.publication_doc -> 'final_ranking') AS ranking
       WHERE ranking ? 'is_king' OR NOT ranking ? 'is_baseline'
     ) THEN
    RAISE EXCEPTION 'lab_arena_publication_baseline_fields_invalid' USING ERRCODE = '22023';
  END IF;

  SELECT pg_catalog.count(*)
  INTO v_public_baseline_count
  FROM pg_catalog.jsonb_array_elements(NEW.publication_doc -> 'participants') AS participant
  WHERE COALESCE((participant ->> 'is_baseline')::BOOLEAN, FALSE)
    AND participant ->> 'submission_id' = v_baseline_id;
  IF v_public_baseline_count <> 1 OR EXISTS (
    SELECT 1
    FROM pg_catalog.jsonb_array_elements(NEW.publication_doc -> 'participants') AS participant
    WHERE COALESCE((participant ->> 'is_baseline')::BOOLEAN, FALSE)
      AND participant ->> 'submission_id' <> v_baseline_id
  ) THEN
    RAISE EXCEPTION 'lab_arena_publication_baseline_invalid' USING ERRCODE = '22023';
  END IF;

  SELECT ranking INTO v_baseline_ranking
  FROM pg_catalog.jsonb_array_elements(NEW.publication_doc -> 'final_ranking') AS ranking
  WHERE ranking ->> 'submission_id' = v_baseline_id
    AND COALESCE((ranking ->> 'is_baseline')::BOOLEAN, FALSE)
  LIMIT 1;
  IF v_baseline_ranking IS NULL THEN
    RAISE EXCEPTION 'lab_arena_publication_baseline_invalid' USING ERRCODE = '22023';
  END IF;
  IF pg_catalog.jsonb_typeof(v_baseline_ranking -> 'final_score') = 'number' THEN
    v_baseline_score := (v_baseline_ranking ->> 'final_score')::NUMERIC;
  END IF;

  v_decision := NEW.publication_doc -> 'king_decision';
  IF pg_catalog.jsonb_typeof(v_decision) IS DISTINCT FROM 'object'
     OR v_decision ->> 'outcome' NOT IN ('crowned', 'no_king') THEN
    RAISE EXCEPTION 'lab_arena_publication_decision_invalid' USING ERRCODE = '22023';
  END IF;

  IF v_decision ->> 'outcome' = 'no_king' THEN
    IF COALESCE(v_decision ->> 'king_hotkey', '') <> ''
       OR COALESCE(v_decision ->> 'king_submission_id', '') <> ''
       OR COALESCE(v_decision ->> 'winner_submission_id', '') <> '' THEN
      RAISE EXCEPTION 'lab_arena_publication_decision_invalid' USING ERRCODE = '22023';
    END IF;
    IF v_baseline_score IS NOT NULL AND EXISTS (
      SELECT 1
      FROM pg_catalog.jsonb_array_elements(NEW.publication_doc -> 'final_ranking') AS ranking
      WHERE NOT COALESCE((ranking ->> 'is_baseline')::BOOLEAN, FALSE)
        AND pg_catalog.jsonb_typeof(ranking -> 'final_score') = 'number'
        AND (ranking ->> 'final_score')::NUMERIC > v_baseline_score
    ) THEN
      RAISE EXCEPTION 'lab_arena_publication_winner_missing' USING ERRCODE = '22023';
    END IF;
    RETURN NEW;
  END IF;

  v_winner_id := v_decision ->> 'winner_submission_id';
  IF COALESCE(v_winner_id, '') = ''
     OR v_decision ->> 'king_submission_id' IS DISTINCT FROM v_winner_id
     OR v_decision ->> 'king_hotkey' IS NOT DISTINCT FROM NEW.configuration_doc ->> 'baseline_hotkey' THEN
    RAISE EXCEPTION 'lab_arena_publication_winner_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT participant INTO v_winner_participant
  FROM pg_catalog.jsonb_array_elements(COALESCE(NEW.participants, '[]'::JSONB)) AS participant
  WHERE participant ->> 'submission_id' = v_winner_id
  LIMIT 1;
  SELECT ranking INTO v_winner_ranking
  FROM pg_catalog.jsonb_array_elements(NEW.publication_doc -> 'final_ranking') AS ranking
  WHERE ranking ->> 'submission_id' = v_winner_id
  LIMIT 1;
  IF v_winner_participant IS NULL
     OR COALESCE((v_winner_participant ->> 'is_king')::BOOLEAN, FALSE)
     OR v_winner_participant ->> 'miner_hotkey' IS DISTINCT FROM v_decision ->> 'king_hotkey'
     OR v_winner_ranking IS NULL
     OR COALESCE((v_winner_ranking ->> 'is_baseline')::BOOLEAN, FALSE)
     OR pg_catalog.jsonb_typeof(v_winner_ranking -> 'final_score') IS DISTINCT FROM 'number'
     OR v_baseline_score IS NULL THEN
    RAISE EXCEPTION 'lab_arena_publication_winner_invalid' USING ERRCODE = '22023';
  END IF;
  v_winner_score := (v_winner_ranking ->> 'final_score')::NUMERIC;
  IF v_winner_score <= v_baseline_score OR EXISTS (
    SELECT 1
    FROM pg_catalog.jsonb_array_elements(NEW.publication_doc -> 'final_ranking') AS ranking
    WHERE NOT COALESCE((ranking ->> 'is_baseline')::BOOLEAN, FALSE)
      AND pg_catalog.jsonb_typeof(ranking -> 'final_score') = 'number'
      AND (
        (ranking ->> 'final_score')::NUMERIC > v_winner_score
        OR (
          (ranking ->> 'final_score')::NUMERIC = v_winner_score
          AND ranking ->> 'submission_id' < v_winner_id
        )
      )
  ) THEN
    RAISE EXCEPTION 'lab_arena_publication_winner_invalid' USING ERRCODE = '22023';
  END IF;
  RETURN NEW;
END;
$lab_arena_publication_baseline_guard$;
ALTER FUNCTION public.lab_arena_publication_baseline_guard_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_publication_baseline_guard_v1() FROM PUBLIC;

DROP TRIGGER IF EXISTS lab_arena_publication_baseline_guard ON public.lab_arena_rounds;
CREATE TRIGGER lab_arena_publication_baseline_guard
  BEFORE UPDATE ON public.lab_arena_rounds
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_publication_baseline_guard_v1();

-- Activation derives the paying miner from the daily winner plus the latest
-- activated miner basis. A no-winner day carries that miner as defended. A
-- new miner starts a new schedule. Organizer baseline bases are ignored.
CREATE OR REPLACE FUNCTION public.lab_arena_activate_reward(
  p_round_id TEXT,
  p_reward_basis JSONB,
  p_signing_key_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_activate_reward$
DECLARE
  v_round public.lab_arena_rounds;
  v_maximum BIGINT;
  v_effective BIGINT;
  v_hash TEXT;
  v_daily_hotkey TEXT := '';
  v_previous_basis JSONB;
  v_expected_hotkey TEXT := '';
  v_expected_outcome TEXT := 'no_king';
  v_expected_start BIGINT := 0;
BEGIN
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab_arena.reward_epoch', 0)
  );
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.reward_activated_at IS NOT NULL THEN
    IF v_round.reward_basis_doc IS NOT DISTINCT FROM p_reward_basis
       AND v_round.signing_key_doc IS NOT DISTINCT FROM p_signing_key_doc THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'existing',
        'effective_reward_epoch', v_round.effective_reward_epoch
      );
    END IF;
    RAISE EXCEPTION 'lab_arena_reward_activation_mismatch' USING ERRCODE = '22023';
  END IF;
  IF v_round.status <> 'published' THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale',
      'round_status', v_round.status
    );
  END IF;
  IF NOT v_round.rewards_enabled
     OR v_round.configuration_doc ->> 'mode' <> 'live' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'disabled');
  END IF;
  IF EXISTS (
    SELECT 1 FROM public.lab_arena_rounds AS older
    WHERE older.status = 'published'
      AND older.rewards_enabled
      AND older.configuration_doc ->> 'mode' = 'live'
      AND older.reward_activated_at IS NULL
      AND (older.created_at, older.round_id) < (v_round.created_at, v_round.round_id)
  ) THEN
    RETURN pg_catalog.jsonb_build_object('status', 'waiting_for_older_round');
  END IF;

  SELECT candidate.reward_basis_doc INTO v_previous_basis
  FROM (
    SELECT reward_basis_doc, configuration_doc, effective_reward_epoch
    FROM public.lab_arena_rounds
    WHERE reward_activated_at IS NOT NULL
      AND reward_basis_doc IS NOT NULL
      AND configuration_doc ->> 'mode' = 'live'
    ORDER BY effective_reward_epoch DESC
    LIMIT 200
  ) AS candidate
  WHERE candidate.reward_basis_doc ->> 'king_outcome' IN ('crowned', 'defended')
    AND COALESCE(candidate.reward_basis_doc ->> 'king_hotkey', '') <> ''
    AND candidate.reward_basis_doc ->> 'king_hotkey'
        IS DISTINCT FROM candidate.configuration_doc ->> 'baseline_hotkey'
    AND candidate.reward_basis_doc ->> 'king_hotkey'
        IS DISTINCT FROM v_round.configuration_doc ->> 'baseline_hotkey'
  ORDER BY candidate.effective_reward_epoch DESC
  LIMIT 1;

  IF v_round.publication_doc #>> '{king_decision,outcome}' = 'crowned'
     AND COALESCE(v_round.publication_doc #>> '{king_decision,king_hotkey}', '') <> ''
     AND v_round.publication_doc #>> '{king_decision,king_hotkey}'
         IS DISTINCT FROM v_round.configuration_doc ->> 'baseline_hotkey' THEN
    v_daily_hotkey := v_round.publication_doc #>> '{king_decision,king_hotkey}';
  END IF;

  IF v_daily_hotkey <> '' THEN
    v_expected_hotkey := v_daily_hotkey;
    IF v_previous_basis IS NOT NULL
       AND v_previous_basis ->> 'king_hotkey' = v_daily_hotkey THEN
      v_expected_outcome := 'defended';
      v_expected_start := (v_previous_basis ->> 'king_start_epoch')::BIGINT;
    ELSE
      v_expected_outcome := 'crowned';
    END IF;
  ELSIF v_previous_basis IS NOT NULL THEN
    v_expected_hotkey := v_previous_basis ->> 'king_hotkey';
    v_expected_outcome := 'defended';
    v_expected_start := (v_previous_basis ->> 'king_start_epoch')::BIGINT;
  END IF;

  v_effective := (p_reward_basis ->> 'effective_reward_epoch')::BIGINT;
  IF v_expected_outcome = 'crowned' THEN
    v_expected_start := v_effective;
  END IF;
  v_hash := p_reward_basis ->> 'reward_basis_hash';
  IF pg_catalog.jsonb_typeof(p_reward_basis) IS DISTINCT FROM 'object'
     OR p_reward_basis ->> 'schema_version' <> 'leadpoet.lab_arena.reward_basis.v1'
     OR p_reward_basis ->> 'round_id' <> p_round_id
     OR v_effective IS NULL OR v_effective < 0
     OR COALESCE(v_hash, '') !~ '^sha256:[0-9a-f]{64}$'
     OR p_reward_basis ->> 'king_outcome' IS DISTINCT FROM v_expected_outcome
     OR COALESCE(p_reward_basis ->> 'king_hotkey', '') IS DISTINCT FROM v_expected_hotkey
     OR (p_reward_basis ->> 'published_at')::TIMESTAMPTZ IS DISTINCT FROM v_round.published_at
     OR (p_reward_basis ->> 'king_start_epoch')::BIGINT IS DISTINCT FROM v_expected_start
     OR p_reward_basis -> 'reward_constants' IS DISTINCT FROM v_round.configuration_doc -> 'reward_constants'
     OR pg_catalog.jsonb_typeof(p_reward_basis -> 'signature') IS DISTINCT FROM 'object'
     OR pg_catalog.jsonb_typeof(p_signing_key_doc) IS DISTINCT FROM 'object'
     OR COALESCE(p_signing_key_doc ->> 'public_key_hash', '') !~ '^sha256:[0-9a-f]{64}$'
     OR p_signing_key_doc ->> 'public_key_hash' IS DISTINCT FROM p_reward_basis -> 'signature' ->> 'public_key_hash' THEN
    RAISE EXCEPTION 'lab_arena_reward_activation_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT pg_catalog.max(effective_reward_epoch)
  INTO v_maximum
  FROM public.lab_arena_rounds
  WHERE reward_activated_at IS NOT NULL;
  IF v_maximum IS NOT NULL AND v_effective <= v_maximum THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'epoch_conflict',
      'minimum_effective_reward_epoch', v_maximum + 1
    );
  END IF;
  UPDATE public.lab_arena_rounds
  SET effective_reward_epoch = v_effective,
      king_start_epoch = (p_reward_basis ->> 'king_start_epoch')::BIGINT,
      reward_basis_hash = v_hash,
      reward_basis_doc = p_reward_basis,
      signing_key_doc = p_signing_key_doc,
      reward_activated_at = pg_catalog.clock_timestamp()
  WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'activated',
    'effective_reward_epoch', v_effective
  );
END;
$lab_arena_activate_reward$;
ALTER FUNCTION public.lab_arena_activate_reward(TEXT, JSONB, JSONB) OWNER TO lab_arena_owner;

-- Old organizer-baseline activations must stop governing immediately. Valid
-- miner bases and explicit no-winner bases remain available to the weight path.
CREATE OR REPLACE VIEW public.lab_arena_reward_basis_v1 AS
  SELECT round_id, effective_reward_epoch, reward_basis_hash, reward_basis_doc, signing_key_doc,
         reward_basis_doc ->> 'king_outcome' AS king_outcome,
         NULLIF(reward_basis_doc ->> 'king_hotkey', '') AS king_hotkey,
         (reward_basis_doc ->> 'king_start_epoch')::BIGINT AS king_start_epoch,
         published_at
  FROM public.lab_arena_rounds
  WHERE status = 'published'
    AND configuration_doc ->> 'mode' = 'live'
    AND rewards_enabled
    AND reward_activated_at IS NOT NULL
    AND reward_basis_doc IS NOT NULL
    AND signing_key_doc IS NOT NULL
    AND (
      reward_basis_doc ->> 'king_outcome' = 'no_king'
      OR (
        reward_basis_doc ->> 'king_outcome' IN ('crowned', 'defended')
        AND reward_basis_doc ->> 'king_hotkey'
            IS DISTINCT FROM configuration_doc ->> 'baseline_hotkey'
      )
    );
ALTER VIEW public.lab_arena_reward_basis_v1 OWNER TO lab_arena_owner;
REVOKE ALL ON public.lab_arena_reward_basis_v1 FROM PUBLIC;
GRANT SELECT ON public.lab_arena_reward_basis_v1 TO lab_arena_service;
DO $lab_arena_reward_basis_acl$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'service_role') THEN
    EXECUTE 'GRANT SELECT ON public.lab_arena_reward_basis_v1 TO service_role';
  END IF;
END;
$lab_arena_reward_basis_acl$;

NOTIFY pgrst, 'reload schema';

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;
