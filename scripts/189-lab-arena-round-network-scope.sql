-- 189-lab-arena-round-network-scope.sql
-- Give every Arena round one queryable chain scope. Historical configurations
-- did not carry the pair and remain permanently assigned to Finney/netuid 71.

BEGIN;

ALTER TABLE public.lab_arena_rounds
  ADD COLUMN IF NOT EXISTS arena_network_name TEXT
  GENERATED ALWAYS AS (
    COALESCE(configuration_doc ->> 'network_name', 'finney')
  ) STORED;

ALTER TABLE public.lab_arena_rounds
  ADD COLUMN IF NOT EXISTS arena_netuid BIGINT
  GENERATED ALWAYS AS (
    COALESCE((configuration_doc ->> 'netuid')::BIGINT, 71)
  ) STORED;

ALTER TABLE public.lab_arena_rounds
  DROP CONSTRAINT IF EXISTS lab_arena_round_network_pair;
ALTER TABLE public.lab_arena_rounds
  ADD CONSTRAINT lab_arena_round_network_pair CHECK (
    (configuration_doc ? 'network_name') = (configuration_doc ? 'netuid')
    AND arena_network_name ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$'
    AND arena_netuid > 0
  );

CREATE INDEX IF NOT EXISTS lab_arena_rounds_chain_status_created_idx
  ON public.lab_arena_rounds (
    arena_network_name, arena_netuid, status, created_at DESC
  );

-- Stored generated values are computed after BEFORE triggers. Reinstall the
-- existing write-once function so its two whole-row promotion comparisons
-- ignore only these derived columns; every durable source column keeps the
-- migration-188 rule unchanged.
CREATE OR REPLACE FUNCTION public.lab_arena_rounds_write_once_v1()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog
AS $lab_arena_rounds_write_once$
BEGIN
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'lab_arena_rounds rows are never deleted' USING ERRCODE = '42501';
  END IF;
  IF OLD.status = 'published' THEN
    IF OLD.promotion_doc IS NULL
       AND NEW.promotion_doc IS NOT NULL
       AND NEW.baseline_promoted_at IS NOT DISTINCT FROM OLD.baseline_promoted_at
       AND (pg_catalog.to_jsonb(NEW) - 'promotion_doc' - 'updated_at'
            - 'arena_network_name' - 'arena_netuid') =
           (pg_catalog.to_jsonb(OLD) - 'promotion_doc' - 'updated_at'
            - 'arena_network_name' - 'arena_netuid') THEN
      NEW.updated_at := pg_catalog.clock_timestamp();
      RETURN NEW;
    END IF;
    IF OLD.promotion_doc IS NOT NULL
       AND NEW.promotion_doc = OLD.promotion_doc
       AND OLD.baseline_promoted_at IS NULL
       AND NEW.baseline_promoted_at IS NOT NULL
       AND (pg_catalog.to_jsonb(NEW) - 'baseline_promoted_at' - 'updated_at'
            - 'arena_network_name' - 'arena_netuid') =
           (pg_catalog.to_jsonb(OLD) - 'baseline_promoted_at' - 'updated_at'
            - 'arena_network_name' - 'arena_netuid') THEN
      NEW.updated_at := pg_catalog.clock_timestamp();
      RETURN NEW;
    END IF;
    IF OLD.reward_activated_at IS NULL
       AND NEW.status = OLD.status
       AND NEW.status_generation = OLD.status_generation
       AND NEW.stage_generation = OLD.stage_generation
       AND NEW.configuration_doc = OLD.configuration_doc
       AND NEW.rewards_enabled = OLD.rewards_enabled
       AND NEW.participants IS NOT DISTINCT FROM OLD.participants
       AND NEW.benchmark_ref IS NOT DISTINCT FROM OLD.benchmark_ref
       AND NEW.evaluation_date IS NOT DISTINCT FROM OLD.evaluation_date
       AND NEW.stage1_scoring_plan_doc IS NOT DISTINCT FROM OLD.stage1_scoring_plan_doc
       AND NEW.stage2_scoring_plan_doc IS NOT DISTINCT FROM OLD.stage2_scoring_plan_doc
       AND NEW.finalists IS NOT DISTINCT FROM OLD.finalists
       AND NEW.publication_doc IS NOT DISTINCT FROM OLD.publication_doc
       AND NEW.king_outcome IS NOT DISTINCT FROM OLD.king_outcome
       AND NEW.king_hotkey IS NOT DISTINCT FROM OLD.king_hotkey
       AND NEW.cancel_reason IS NOT DISTINCT FROM OLD.cancel_reason
       AND NEW.published_at IS NOT DISTINCT FROM OLD.published_at
       AND NEW.promotion_required = OLD.promotion_required
       AND NEW.promotion_doc IS NOT DISTINCT FROM OLD.promotion_doc
       AND NEW.baseline_promoted_at IS NOT DISTINCT FROM OLD.baseline_promoted_at
       AND NEW.reward_activated_at IS NOT NULL
       AND NEW.effective_reward_epoch IS NOT NULL
       AND NEW.reward_basis_hash IS NOT NULL
       AND NEW.reward_basis_doc IS NOT NULL
       AND NEW.signing_key_doc IS NOT NULL
       AND NEW.king_start_epoch IS NOT NULL THEN
      NEW.updated_at := pg_catalog.clock_timestamp();
      RETURN NEW;
    END IF;
    RAISE EXCEPTION 'published round is immutable outside promotion or reward activation' USING ERRCODE = '42501';
  END IF;
  IF OLD.status = 'cancelled' AND NEW.status <> 'cancelled' THEN
    RAISE EXCEPTION 'cancelled round cannot be reopened' USING ERRCODE = '42501';
  END IF;
  IF NEW.configuration_doc <> OLD.configuration_doc THEN
    RAISE EXCEPTION 'round configuration is write-once' USING ERRCODE = '42501';
  END IF;
  IF (OLD.stage1_scoring_plan_doc IS NOT NULL AND NEW.stage1_scoring_plan_doc IS DISTINCT FROM OLD.stage1_scoring_plan_doc)
     OR (OLD.stage2_scoring_plan_doc IS NOT NULL AND NEW.stage2_scoring_plan_doc IS DISTINCT FROM OLD.stage2_scoring_plan_doc)
     OR (OLD.finalists IS NOT NULL AND NEW.finalists IS DISTINCT FROM OLD.finalists)
     OR (OLD.publication_doc IS NOT NULL AND NEW.publication_doc IS DISTINCT FROM OLD.publication_doc)
     OR (OLD.reward_basis_hash IS NOT NULL AND NEW.reward_basis_hash IS DISTINCT FROM OLD.reward_basis_hash)
     OR (OLD.reward_basis_doc IS NOT NULL AND NEW.reward_basis_doc IS DISTINCT FROM OLD.reward_basis_doc)
     OR (OLD.king_outcome IS NOT NULL AND NEW.king_outcome IS DISTINCT FROM OLD.king_outcome)
     OR (OLD.effective_reward_epoch IS NOT NULL AND NEW.effective_reward_epoch IS DISTINCT FROM OLD.effective_reward_epoch)
     OR (OLD.reward_activated_at IS NOT NULL AND NEW.reward_activated_at IS DISTINCT FROM OLD.reward_activated_at)
     OR (OLD.promotion_doc IS NOT NULL AND NEW.promotion_doc IS DISTINCT FROM OLD.promotion_doc)
     OR (OLD.baseline_promoted_at IS NOT NULL AND NEW.baseline_promoted_at IS DISTINCT FROM OLD.baseline_promoted_at)
     OR NEW.promotion_required IS DISTINCT FROM OLD.promotion_required THEN
    RAISE EXCEPTION 'round publication and commitment columns are write-once' USING ERRCODE = '42501';
  END IF;
  IF NEW.status = 'published' AND (
       NEW.publication_doc IS NULL OR NEW.king_outcome IS NULL
       OR NEW.published_at IS NULL) THEN
    RAISE EXCEPTION 'publication requires the compact competition result' USING ERRCODE = '23514';
  END IF;
  NEW.updated_at := pg_catalog.clock_timestamp();
  RETURN NEW;
END;
$lab_arena_rounds_write_once$;
ALTER FUNCTION public.lab_arena_rounds_write_once_v1() OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_schema_version_v1()
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$ SELECT pg_catalog.jsonb_build_object(
  'schema_version', 'leadpoet.lab_arena.schema_version.v1',
  'version', 189
) $$;
ALTER FUNCTION public.lab_arena_schema_version_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_schema_version_v1() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.lab_arena_schema_version_v1() TO lab_arena_service;

NOTIFY pgrst, 'reload schema';

COMMIT;
