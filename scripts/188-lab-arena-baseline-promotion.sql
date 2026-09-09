-- 188-lab-arena-baseline-promotion.sql
-- Persist the recoverable Git plan for each live crowned baseline promotion.

BEGIN;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

ALTER TABLE public.lab_arena_rounds
  ADD COLUMN IF NOT EXISTS promotion_required BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE public.lab_arena_rounds
  ADD COLUMN IF NOT EXISTS promotion_doc JSONB;
ALTER TABLE public.lab_arena_rounds
  ADD COLUMN IF NOT EXISTS baseline_promoted_at TIMESTAMPTZ;

-- In-flight rounds must use the new promotion boundary when they eventually
-- publish. Terminal historical rows retain FALSE.
UPDATE public.lab_arena_rounds
SET promotion_required = TRUE
WHERE status IN (
  'open', 'committed',
  'stage1', 'stage1_closed', 'stage1_scoring', 'stage1_judged', 'stage1_scored',
  'stage2', 'stage2_closed', 'stage2_scoring', 'stage2_judged',
  'scored'
);

-- New rounds opt in at creation.
ALTER TABLE public.lab_arena_rounds
  ALTER COLUMN promotion_required SET DEFAULT TRUE;

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
    -- The service role has no table-write grant. These two narrow shapes are
    -- therefore reachable through the promotion RPCs, while all publication
    -- and plan fields remain write-once.
    IF OLD.promotion_doc IS NULL
       AND NEW.promotion_doc IS NOT NULL
       AND NEW.baseline_promoted_at IS NOT DISTINCT FROM OLD.baseline_promoted_at
       AND (pg_catalog.to_jsonb(NEW) - 'promotion_doc' - 'updated_at') =
           (pg_catalog.to_jsonb(OLD) - 'promotion_doc' - 'updated_at') THEN
      NEW.updated_at := pg_catalog.clock_timestamp();
      RETURN NEW;
    END IF;
    IF OLD.promotion_doc IS NOT NULL
       AND NEW.promotion_doc = OLD.promotion_doc
       AND OLD.baseline_promoted_at IS NULL
       AND NEW.baseline_promoted_at IS NOT NULL
       AND (pg_catalog.to_jsonb(NEW) - 'baseline_promoted_at' - 'updated_at') =
           (pg_catalog.to_jsonb(OLD) - 'baseline_promoted_at' - 'updated_at') THEN
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

CREATE OR REPLACE FUNCTION public.lab_arena_prepare_promotion(
  p_round_id TEXT,
  p_plan JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_prepare_promotion$
DECLARE
  v_round public.lab_arena_rounds;
  v_winner_id TEXT;
  v_winner_hotkey TEXT;
BEGIN
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab_arena.baseline_promotion', 0)
  );
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> 'published'
     OR v_round.configuration_doc ->> 'mode' <> 'live'
     OR NOT v_round.promotion_required
     OR v_round.publication_doc #>> '{king_decision,outcome}' <> 'crowned' THEN
    RAISE EXCEPTION 'lab_arena_promotion_not_required' USING ERRCODE = '55000';
  END IF;

  v_winner_id := v_round.publication_doc #>> '{king_decision,winner_submission_id}';
  v_winner_hotkey := v_round.publication_doc #>> '{king_decision,king_hotkey}';
  IF COALESCE(v_winner_id, '') = ''
     OR v_winner_id IS DISTINCT FROM
        v_round.publication_doc #>> '{king_decision,king_submission_id}'
     OR COALESCE(v_winner_hotkey, '') = ''
     OR NOT EXISTS (
       SELECT 1
       FROM public.lab_arena_submissions AS submission
       WHERE submission.round_id = p_round_id
         AND submission.submission_id = v_winner_id
         AND submission.miner_hotkey = v_winner_hotkey
         AND submission.status = 'frozen'
         AND NOT submission.is_king
     ) THEN
    RAISE EXCEPTION 'lab_arena_promotion_winner_invalid' USING ERRCODE = '22023';
  END IF;

  IF pg_catalog.jsonb_typeof(p_plan) IS DISTINCT FROM 'object'
     OR (SELECT pg_catalog.count(*) FROM pg_catalog.jsonb_object_keys(p_plan)) <> 4
     OR COALESCE(p_plan ->> 'commit', '') !~ '^[0-9a-f]{40}$'
     OR COALESCE(p_plan ->> 'main_before', '') !~ '^[0-9a-f]{40}$'
     OR COALESCE(p_plan ->> 'lab_before', '') !~ '^[0-9a-f]{40}$'
     OR COALESCE(p_plan ->> 'timestamp', '') !~
        '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]+)?(Z|[+]00:00)$' THEN
    RAISE EXCEPTION 'lab_arena_promotion_plan_invalid' USING ERRCODE = '22023';
  END IF;
  PERFORM (p_plan ->> 'timestamp')::TIMESTAMPTZ;

  IF v_round.promotion_doc IS NOT NULL THEN
    IF v_round.promotion_doc = p_plan THEN
      RETURN pg_catalog.jsonb_build_object('status', 'existing', 'plan', v_round.promotion_doc);
    END IF;
    RAISE EXCEPTION 'lab_arena_promotion_plan_mismatch' USING ERRCODE = '22023';
  END IF;
  IF v_round.baseline_promoted_at IS NOT NULL THEN
    RAISE EXCEPTION 'lab_arena_promotion_state_invalid' USING ERRCODE = '55000';
  END IF;
  IF EXISTS (
    SELECT 1
    FROM public.lab_arena_rounds AS older
    WHERE older.status = 'published'
      AND older.configuration_doc ->> 'mode' = 'live'
      AND older.promotion_required
      AND older.publication_doc #>> '{king_decision,outcome}' = 'crowned'
      AND older.baseline_promoted_at IS NULL
      AND (older.created_at, older.round_id) < (v_round.created_at, v_round.round_id)
  ) THEN
    RETURN pg_catalog.jsonb_build_object('status', 'waiting_for_older_promotion');
  END IF;

  UPDATE public.lab_arena_rounds
  SET promotion_doc = p_plan
  WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object('status', 'prepared', 'plan', p_plan);
END;
$lab_arena_prepare_promotion$;

CREATE OR REPLACE FUNCTION public.lab_arena_complete_promotion(
  p_round_id TEXT,
  p_plan JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_complete_promotion$
DECLARE
  v_round public.lab_arena_rounds;
BEGIN
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab_arena.baseline_promotion', 0)
  );
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.promotion_doc IS NULL
     OR v_round.promotion_doc IS DISTINCT FROM p_plan THEN
    RAISE EXCEPTION 'lab_arena_promotion_plan_mismatch' USING ERRCODE = '22023';
  END IF;
  IF v_round.baseline_promoted_at IS NOT NULL THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'existing', 'baseline_promoted_at', v_round.baseline_promoted_at
    );
  END IF;
  UPDATE public.lab_arena_rounds
  SET baseline_promoted_at = pg_catalog.clock_timestamp()
  WHERE round_id = p_round_id;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'promoted', 'baseline_promoted_at', v_round.baseline_promoted_at
  );
END;
$lab_arena_complete_promotion$;

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
         AND pending.promotion_required
         AND pending.publication_doc #>> '{king_decision,outcome}' = 'crowned'
         AND pending.baseline_promoted_at IS NULL
     ) THEN
    RAISE EXCEPTION 'lab_arena_reward_waiting_for_promotion' USING ERRCODE = '55000';
  END IF;
  RETURN NEW;
END;
$lab_arena_reward_requires_promotion$;

DROP TRIGGER IF EXISTS lab_arena_reward_requires_promotion ON public.lab_arena_rounds;
CREATE TRIGGER lab_arena_reward_requires_promotion
  BEFORE UPDATE OF reward_activated_at ON public.lab_arena_rounds
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_reward_requires_promotion_v1();

ALTER FUNCTION public.lab_arena_prepare_promotion(TEXT, JSONB) OWNER TO lab_arena_owner;
ALTER FUNCTION public.lab_arena_complete_promotion(TEXT, JSONB) OWNER TO lab_arena_owner;
ALTER FUNCTION public.lab_arena_reward_requires_promotion_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_prepare_promotion(TEXT, JSONB) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.lab_arena_complete_promotion(TEXT, JSONB) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.lab_arena_reward_requires_promotion_v1() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.lab_arena_prepare_promotion(TEXT, JSONB) TO lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_complete_promotion(TEXT, JSONB) TO lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_schema_version_v1()
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$ SELECT pg_catalog.jsonb_build_object(
  'schema_version', 'leadpoet.lab_arena.schema_version.v1',
  'version', 188
) $$;
ALTER FUNCTION public.lab_arena_schema_version_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_schema_version_v1() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.lab_arena_schema_version_v1() TO lab_arena_service;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;
