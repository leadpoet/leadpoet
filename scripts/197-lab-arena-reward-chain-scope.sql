-- 197-lab-arena-reward-chain-scope.sql
-- Keep reward epochs, predecessor kings, activation ordering, and public reward
-- reads inside the round's immutable Arena network/netuid pair.

BEGIN;

DO $lab_arena_197_requires_194$
BEGIN
  IF pg_catalog.to_regprocedure('public.lab_arena_schema_version_v1()') IS NULL
     OR (pg_catalog.pg_get_functiondef(
          pg_catalog.to_regprocedure('public.lab_arena_schema_version_v1()')
        ) NOT LIKE '%''version'', 194%'
     AND pg_catalog.pg_get_functiondef(
          pg_catalog.to_regprocedure('public.lab_arena_schema_version_v1()')
        ) NOT LIKE '%''version'', 197%') THEN
    RAISE EXCEPTION 'apply 194-lab-arena-open-scorer-refresh.sql first';
  END IF;
END;
$lab_arena_197_requires_194$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- Effective epochs are chain-local. Preserve every existing value while
-- allowing two chain scopes to activate the same epoch independently.
DROP INDEX IF EXISTS public.lab_arena_rounds_effective_reward_epoch_uq;
CREATE UNIQUE INDEX IF NOT EXISTS lab_arena_rounds_reward_chain_epoch_uq
  ON public.lab_arena_rounds (
    arena_network_name, arena_netuid, effective_reward_epoch
  )
  WHERE effective_reward_epoch IS NOT NULL;

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
      AND older.arena_network_name = v_round.arena_network_name
      AND older.arena_netuid = v_round.arena_netuid
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
      AND arena_network_name = v_round.arena_network_name
      AND arena_netuid = v_round.arena_netuid
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
  WHERE reward_activated_at IS NOT NULL
    AND arena_network_name = v_round.arena_network_name
    AND arena_netuid = v_round.arena_netuid;
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

-- A pending promotion on another chain cannot block this chain's rewards.
-- Baseline Git publication itself keeps its existing serialized ordering.
CREATE OR REPLACE FUNCTION public.lab_arena_reward_requires_promotion_v1()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $lab_arena_reward_requires_promotion$
BEGIN
  IF OLD.reward_activated_at IS NULL
     AND NEW.reward_activated_at IS NOT NULL
     AND EXISTS (
       SELECT 1
       FROM public.lab_arena_rounds AS pending
       WHERE pending.status = 'published'
         AND pending.configuration_doc ->> 'mode' = 'live'
         -- Stored generated columns are not computed yet in a BEFORE trigger.
         -- The chain is immutable, so use the existing row's scope.
         AND pending.arena_network_name = OLD.arena_network_name
         AND pending.arena_netuid = OLD.arena_netuid
         AND pending.promotion_required
         AND pending.publication_doc #>> '{king_decision,outcome}' = 'crowned'
         AND pending.baseline_promoted_at IS NULL
     ) THEN
    RAISE EXCEPTION 'lab_arena_reward_waiting_for_promotion' USING ERRCODE = '55000';
  END IF;
  RETURN NEW;
END;
$lab_arena_reward_requires_promotion$;
ALTER FUNCTION public.lab_arena_reward_requires_promotion_v1() OWNER TO lab_arena_owner;

-- Append the immutable chain pair so existing consumers retain their column
-- order and every reader can select the governing row for its own scope.
CREATE OR REPLACE VIEW public.lab_arena_reward_basis_v1 AS
  SELECT round_id, effective_reward_epoch, reward_basis_hash, reward_basis_doc,
         signing_key_doc,
         reward_basis_doc ->> 'king_outcome' AS king_outcome,
         NULLIF(reward_basis_doc ->> 'king_hotkey', '') AS king_hotkey,
         (reward_basis_doc ->> 'king_start_epoch')::BIGINT AS king_start_epoch,
         published_at, arena_network_name, arena_netuid
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

CREATE OR REPLACE FUNCTION public.lab_arena_schema_version_v1()
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_schema_version$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.schema_version.v1', 'version', 197
  );
$lab_arena_schema_version$;
ALTER FUNCTION public.lab_arena_schema_version_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_schema_version_v1() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.lab_arena_schema_version_v1() TO lab_arena_service;

NOTIFY pgrst, 'reload schema';
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;
