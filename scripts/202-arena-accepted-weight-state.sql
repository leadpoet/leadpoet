-- Append-only accepted Arena weight state and non-authoritative chain outcomes.
-- These tables contain no receipt ancestry or release identity.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $requires_197$
BEGIN
  IF pg_catalog.to_regprocedure('public.lab_arena_schema_version_v1()') IS NULL
     OR pg_catalog.pg_get_functiondef(
          pg_catalog.to_regprocedure('public.lab_arena_schema_version_v1()')
        ) NOT LIKE '%''version'', 197%' THEN
    RAISE EXCEPTION 'apply 197-lab-arena-reward-chain-scope.sql first';
  END IF;
END;
$requires_197$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $input_acl$
DECLARE relation_name TEXT;
BEGIN
  FOREACH relation_name IN ARRAY ARRAY[
    'research_lab_emission_allocation_snapshots',
    'fulfillment_score_consensus', 'banned_hotkeys'
  ] LOOP
    IF to_regclass('public.' || relation_name) IS NOT NULL THEN
      EXECUTE format('GRANT SELECT ON public.%I TO lab_arena_owner', relation_name);
      EXECUTE format('DROP POLICY IF EXISTS lab_arena_weight_input_read ON public.%I', relation_name);
      EXECUTE format('CREATE POLICY lab_arena_weight_input_read ON public.%I FOR SELECT TO lab_arena_owner USING (true)', relation_name);
    END IF;
  END LOOP;
END;
$input_acl$;

CREATE TABLE IF NOT EXISTS public.lab_arena_accepted_weight_states (
  network TEXT NOT NULL CHECK (network ~ '^[a-z][a-z0-9_-]{0,31}$'),
  netuid INTEGER NOT NULL CHECK (netuid > 0),
  epoch BIGINT NOT NULL CHECK (epoch >= 0),
  state_hash TEXT NOT NULL CHECK (state_hash ~ '^sha256:[0-9a-f]{64}$'),
  state_doc JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
  PRIMARY KEY (network, netuid, epoch),
  UNIQUE (state_hash),
  CHECK (state_doc ->> 'schema_version' = 'leadpoet.arena.accepted_weight_state.v1'),
  CHECK (state_doc ->> 'network' = network),
  CHECK ((state_doc ->> 'netuid')::INTEGER = netuid),
  CHECK ((state_doc ->> 'epoch')::BIGINT = epoch),
  CHECK (state_doc ->> 'state_hash' = state_hash)
  ,CHECK (state_doc ?& ARRAY['schema_version','network','genesis_hash','netuid','epoch','valid_from_block','valid_until_block','reward_basis','fixed_allocations','fulfillment_demands','burn_hotkey','issued_at','state_hash','signature'])
);
ALTER TABLE public.lab_arena_accepted_weight_states OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_accepted_weight_states ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.lab_arena_accepted_weight_states FROM PUBLIC;
GRANT SELECT ON public.lab_arena_accepted_weight_states TO lab_arena_service;
DROP POLICY IF EXISTS lab_arena_service_read ON public.lab_arena_accepted_weight_states;
CREATE POLICY lab_arena_service_read ON public.lab_arena_accepted_weight_states
  FOR SELECT TO lab_arena_service USING (true);

CREATE TABLE IF NOT EXISTS public.lab_arena_chain_outcomes (
  network TEXT NOT NULL CHECK (network ~ '^[a-z][a-z0-9_-]{0,31}$'),
  netuid INTEGER NOT NULL CHECK (netuid > 0),
  epoch BIGINT NOT NULL CHECK (epoch >= 0),
  validator_hotkey TEXT NOT NULL CHECK (length(validator_hotkey) BETWEEN 1 AND 64),
  request_id TEXT NOT NULL CHECK (request_id ~ '^sha256:[0-9a-f]{64}$'),
  extrinsic_hash TEXT NOT NULL CHECK (extrinsic_hash ~ '^0x[0-9a-f]{64}$'),
  outcome_doc JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
  PRIMARY KEY (network, netuid, epoch, validator_hotkey, request_id),
  UNIQUE (network, netuid, epoch, validator_hotkey, extrinsic_hash),
  CHECK (outcome_doc ->> 'network' = network),
  CHECK ((outcome_doc ->> 'netuid')::INTEGER = netuid),
  CHECK ((outcome_doc ->> 'epoch')::BIGINT = epoch),
  CHECK (outcome_doc ->> 'validator_hotkey' = validator_hotkey),
  CHECK (outcome_doc ->> 'request_id' = request_id),
  CHECK (outcome_doc ->> 'extrinsic_hash' = extrinsic_hash)
);
ALTER TABLE public.lab_arena_chain_outcomes OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_chain_outcomes ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.lab_arena_chain_outcomes FROM PUBLIC;
GRANT SELECT ON public.lab_arena_chain_outcomes TO lab_arena_service;
DROP POLICY IF EXISTS lab_arena_service_read ON public.lab_arena_chain_outcomes;
CREATE POLICY lab_arena_service_read ON public.lab_arena_chain_outcomes
  FOR SELECT TO lab_arena_service USING (true);

CREATE OR REPLACE FUNCTION public.lab_arena_publish_weight_state_v1(
  p_network TEXT, p_netuid INTEGER, p_epoch BIGINT,
  p_state_hash TEXT, p_state_doc JSONB
) RETURNS JSONB
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $publish$
DECLARE existing public.lab_arena_accepted_weight_states;
BEGIN
  INSERT INTO public.lab_arena_accepted_weight_states(
    network, netuid, epoch, state_hash, state_doc
  ) VALUES (p_network, p_netuid, p_epoch, p_state_hash, p_state_doc)
  ON CONFLICT (network, netuid, epoch) DO NOTHING;
  SELECT * INTO existing FROM public.lab_arena_accepted_weight_states
    WHERE network = p_network AND netuid = p_netuid AND epoch = p_epoch;
  -- ECDSA signatures are non-deterministic. The content-addressed body hash
  -- is the immutable identity; an equivalent racing signature is harmless.
  IF existing.state_hash IS DISTINCT FROM p_state_hash THEN
    RAISE EXCEPTION 'lab_arena_weight_state_conflict' USING ERRCODE = '23505';
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'accepted', 'state_hash', existing.state_hash,
    'state', existing.state_doc
  );
END;
$publish$;
ALTER FUNCTION public.lab_arena_publish_weight_state_v1(TEXT, INTEGER, BIGINT, TEXT, JSONB) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_publish_weight_state_v1(TEXT, INTEGER, BIGINT, TEXT, JSONB) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.lab_arena_publish_weight_state_v1(TEXT, INTEGER, BIGINT, TEXT, JSONB) TO lab_arena_service;

-- One plain snapshot of already accepted economic inputs. This is a migration
-- bridge only: no receipt or historical ancestry crosses the function.
CREATE OR REPLACE FUNCTION public.lab_arena_weight_inputs_v1(
  p_epoch BIGINT, p_netuid INTEGER, p_burn_hotkey TEXT,
  p_fulfillment_enabled BOOLEAN, p_leaderboard_enabled BOOLEAN
) RETURNS JSONB
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $inputs$
DECLARE
  allocation JSONB;
  allocation_count INTEGER;
  lab_cap NUMERIC;
  allocated NUMERIC;
  unallocated NUMERIC;
  invalid_allocation_rows INTEGER;
  fixed JSONB;
  demands JSONB;
BEGIN
  IF COALESCE(p_burn_hotkey, '') = '' THEN
    RAISE EXCEPTION 'lab_arena_burn_hotkey_missing' USING ERRCODE = '22023';
  END IF;
  SELECT count(*) INTO allocation_count
  FROM public.research_lab_emission_allocation_snapshots
  WHERE epoch = p_epoch AND netuid = p_netuid AND snapshot_status = 'active';
  IF allocation_count <> 1 THEN
    RAISE EXCEPTION 'lab_arena_accepted_allocation_missing' USING ERRCODE = 'P0002';
  END IF;
  SELECT allocation_doc, lab_cap_alpha_percent INTO allocation, lab_cap
  FROM public.research_lab_emission_allocation_snapshots
  WHERE epoch = p_epoch AND netuid = p_netuid AND snapshot_status = 'active';
  IF allocation IS NULL OR lab_cap IS NULL
     OR jsonb_typeof(allocation -> 'reimbursement_allocations') IS DISTINCT FROM 'array'
     OR jsonb_typeof(allocation -> 'champion_allocations') IS DISTINCT FROM 'array'
     OR jsonb_typeof(allocation -> 'queued_champion_allocations') IS DISTINCT FROM 'array'
     OR jsonb_typeof(allocation -> 'lab_cap_percent') IS DISTINCT FROM 'number'
     OR jsonb_typeof(allocation -> 'unallocated_percent') IS DISTINCT FROM 'number'
     OR (allocation ->> 'lab_cap_percent')::NUMERIC IS DISTINCT FROM lab_cap
     OR lab_cap < 0 OR lab_cap > 100 THEN
    RAISE EXCEPTION 'lab_arena_accepted_allocation_invalid' USING ERRCODE = '22023';
  END IF;
  unallocated := (allocation ->> 'unallocated_percent')::NUMERIC / 100;
  SELECT count(*) INTO invalid_allocation_rows
  FROM jsonb_array_elements(
      (allocation -> 'reimbursement_allocations')
      || (allocation -> 'champion_allocations')
      || (allocation -> 'queued_champion_allocations')
  ) item
  WHERE jsonb_typeof(item) IS DISTINCT FROM 'object'
     OR jsonb_typeof(item -> 'miner_hotkey') IS DISTINCT FROM 'string'
     OR COALESCE(item ->> 'miner_hotkey', '') = ''
     OR jsonb_typeof(item -> 'paid_alpha_percent') IS DISTINCT FROM 'number'
     OR (item ->> 'paid_alpha_percent')::NUMERIC < 0;
  IF unallocated < 0 OR unallocated > lab_cap / 100 OR invalid_allocation_rows <> 0 THEN
    RAISE EXCEPTION 'lab_arena_accepted_allocation_invalid' USING ERRCODE = '22023';
  END IF;

  WITH allocation_rows AS (
    SELECT item ->> 'miner_hotkey' AS hotkey,
           (item ->> 'paid_alpha_percent')::NUMERIC / 100 AS share
    FROM jsonb_array_elements(
      COALESCE(allocation -> 'reimbursement_allocations', '[]'::JSONB)
      || COALESCE(allocation -> 'champion_allocations', '[]'::JSONB)
      || COALESCE(allocation -> 'queued_champion_allocations', '[]'::JSONB)
    ) item
  ), leaderboard_source AS (
    SELECT c.miner_hotkey AS hotkey, count(*) AS wins,
           sum(COALESCE(c.reward_pct, 0)) AS reward_total
    FROM public.fulfillment_score_consensus c
    WHERE c.is_winner IS TRUE
      AND c.computed_at >= pg_catalog.clock_timestamp() - interval '7 days'
      AND c.computed_at <= pg_catalog.clock_timestamp()
      AND NOT EXISTS (
        SELECT 1 FROM public.banned_hotkeys b WHERE b.hotkey = c.miner_hotkey
      )
    GROUP BY c.miner_hotkey
  ), leaderboard AS (
    SELECT hotkey, CASE row_number() OVER (ORDER BY wins DESC, reward_total DESC, hotkey)
      WHEN 1 THEN 0.05::NUMERIC WHEN 2 THEN 0.03::NUMERIC
      WHEN 3 THEN 0.015::NUMERIC END AS share
    FROM leaderboard_source ORDER BY wins DESC, reward_total DESC, hotkey LIMIT 3
  ), named AS (
    SELECT hotkey, share FROM allocation_rows WHERE hotkey IS NOT NULL AND share > 0
    UNION ALL SELECT hotkey, share FROM leaderboard
      WHERE share > 0 AND p_leaderboard_enabled
  ), totals AS (
    SELECT COALESCE(sum(share), 0) AS allocation_total FROM allocation_rows
  ), with_burn AS (
    SELECT hotkey, share FROM named
    UNION ALL
    SELECT p_burn_hotkey,
      GREATEST(0::NUMERIC, lab_cap / 100 - allocation_total)
      + (0.095::NUMERIC - COALESCE((SELECT sum(share) FROM leaderboard WHERE p_leaderboard_enabled), 0))
    FROM totals
  ), combined AS (
    SELECT hotkey, sum(share) AS share FROM with_burn
    WHERE share > 0 GROUP BY hotkey
  )
  SELECT COALESCE(jsonb_agg(jsonb_build_object(
      'hotkey', hotkey, 'share_ppb', round(share * 1000000000)::BIGINT
    ) ORDER BY hotkey), '[]'::JSONB)
    INTO fixed FROM combined;

  SELECT COALESCE(sum((item ->> 'paid_alpha_percent')::NUMERIC / 100), 0)
    INTO allocated
  FROM jsonb_array_elements(
    (allocation -> 'reimbursement_allocations')
    || (allocation -> 'champion_allocations')
    || (allocation -> 'queued_champion_allocations')
  ) item;
  IF allocated + unallocated > lab_cap / 100 + 0.000000001::NUMERIC THEN
    RAISE EXCEPTION 'lab_arena_accepted_allocation_over_cap' USING ERRCODE = '22023';
  END IF;

  SELECT COALESCE(jsonb_agg(jsonb_build_object(
      'hotkey', miner_hotkey,
      'share_ppb', round(total_reward * 1000000000)::BIGINT
    ) ORDER BY miner_hotkey), '[]'::JSONB)
    INTO demands
  FROM (
    SELECT miner_hotkey, sum(reward_pct)::NUMERIC AS total_reward
    FROM public.fulfillment_score_consensus
    WHERE p_fulfillment_enabled AND reward_pct IS NOT NULL
      AND reward_expires_epoch > p_epoch
    GROUP BY miner_hotkey HAVING sum(reward_pct) > 0
  ) active;

  RETURN jsonb_build_object(
    'fixed_allocations', fixed,
    'fulfillment_demands', demands,
    'burn_hotkey', p_burn_hotkey
  );
END;
$inputs$;
ALTER FUNCTION public.lab_arena_weight_inputs_v1(BIGINT, INTEGER, TEXT, BOOLEAN, BOOLEAN) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_weight_inputs_v1(BIGINT, INTEGER, TEXT, BOOLEAN, BOOLEAN) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.lab_arena_weight_inputs_v1(BIGINT, INTEGER, TEXT, BOOLEAN, BOOLEAN) TO lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_record_chain_outcome_v1(
  p_network TEXT, p_netuid INTEGER, p_epoch BIGINT,
  p_validator_hotkey TEXT, p_request_id TEXT, p_extrinsic_hash TEXT,
  p_outcome_doc JSONB
) RETURNS JSONB
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $outcome$
DECLARE existing public.lab_arena_chain_outcomes;
BEGIN
  INSERT INTO public.lab_arena_chain_outcomes(
    network, netuid, epoch, validator_hotkey, request_id,
    extrinsic_hash, outcome_doc
  ) VALUES (
    p_network, p_netuid, p_epoch, p_validator_hotkey, p_request_id,
    p_extrinsic_hash, p_outcome_doc
  ) ON CONFLICT (network, netuid, epoch, validator_hotkey, request_id) DO NOTHING;
  SELECT * INTO existing FROM public.lab_arena_chain_outcomes
    WHERE network = p_network AND netuid = p_netuid AND epoch = p_epoch
      AND validator_hotkey = p_validator_hotkey AND request_id = p_request_id;
  IF existing.outcome_doc IS DISTINCT FROM p_outcome_doc THEN
    RAISE EXCEPTION 'lab_arena_chain_outcome_conflict' USING ERRCODE = '23505';
  END IF;
  RETURN pg_catalog.jsonb_build_object('status', 'recorded');
END;
$outcome$;
ALTER FUNCTION public.lab_arena_record_chain_outcome_v1(TEXT, INTEGER, BIGINT, TEXT, TEXT, TEXT, JSONB) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_record_chain_outcome_v1(TEXT, INTEGER, BIGINT, TEXT, TEXT, TEXT, JSONB) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.lab_arena_record_chain_outcome_v1(TEXT, INTEGER, BIGINT, TEXT, TEXT, TEXT, JSONB) TO lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_weight_state_schema_v1()
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $schema$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.weight_state_schema.v1', 'version', 202
  );
$schema$;
ALTER FUNCTION public.lab_arena_weight_state_schema_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_weight_state_schema_v1() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.lab_arena_weight_state_schema_v1() TO lab_arena_service;

NOTIFY pgrst, 'reload schema';
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;
