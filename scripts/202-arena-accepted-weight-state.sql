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
  ,CONSTRAINT lab_arena_accepted_weight_state_fields_ck CHECK (
    state_doc ?& ARRAY['schema_version','network','genesis_hash','netuid','epoch','valid_from_block','valid_until_block','reward_basis','burn_hotkey','issued_at','state_hash','signature']
  )
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
