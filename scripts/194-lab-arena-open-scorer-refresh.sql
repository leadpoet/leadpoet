-- 194-lab-arena-open-scorer-refresh.sql
-- Refresh the trusted scorer pin when an open round atomically commits its benchmark.

BEGIN;

DO $lab_arena_194_requires_193$
BEGIN
  IF pg_catalog.to_regprocedure('public.lab_arena_schema_version_v1()') IS NULL
     OR (pg_catalog.pg_get_functiondef(
          pg_catalog.to_regprocedure('public.lab_arena_schema_version_v1()')
        ) NOT LIKE '%''version'', 193%'
     AND pg_catalog.pg_get_functiondef(
          pg_catalog.to_regprocedure('public.lab_arena_schema_version_v1()')
        ) NOT LIKE '%''version'', 194%') THEN
    RAISE EXCEPTION 'apply 193-lab-arena-upload-recovery.sql first';
  END IF;
END;
$lab_arena_194_requires_193$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_transition_round(
  p_round_id TEXT,
  p_expected_status TEXT,
  p_next_status TEXT,
  p_patch JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_transition_round$
DECLARE
  v_round public.lab_arena_rounds;
  v_patch JSONB := COALESCE(p_patch, '{}'::JSONB);
  v_keys TEXT[];
  v_allowed TEXT[];
  v_baseline_count INTEGER;
  v_challenger_count INTEGER;
  v_missing_scores INTEGER;
  v_invalid_finalists INTEGER;
  v_publication JSONB;
  v_decision JSONB;
BEGIN
  IF pg_catalog.jsonb_typeof(v_patch) <> 'object' THEN
    RAISE EXCEPTION 'lab_arena_patch_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> p_expected_status THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale', 'round_status', v_round.status, 'status_generation', v_round.status_generation);
  END IF;
  SELECT COALESCE(pg_catalog.array_agg(key ORDER BY key), ARRAY[]::TEXT[]) INTO v_keys
  FROM pg_catalog.jsonb_object_keys(v_patch) AS key;

  IF p_expected_status = 'open' AND p_next_status = 'committed' THEN
    v_allowed := ARRAY['participants', 'benchmark_ref', 'evaluation_date'];
    IF NOT (
      (v_keys @> v_allowed AND v_allowed @> v_keys)
      OR (
        v_keys @> (v_allowed || ARRAY['scorer_image_digest', 'scorer_image_reference'])
        AND (v_allowed || ARRAY['scorer_image_digest', 'scorer_image_reference']) @> v_keys
      )
    ) THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    IF pg_catalog.jsonb_typeof(v_patch -> 'participants') <> 'array'
       OR pg_catalog.char_length(COALESCE(v_patch ->> 'benchmark_ref', '')) NOT BETWEEN 1 AND 1024
       OR COALESCE(v_patch ->> 'evaluation_date', '') !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' THEN
      RAISE EXCEPTION 'lab_arena_round_commit_invalid' USING ERRCODE = '22023';
    END IF;
    IF (v_patch ? 'scorer_image_digest') <> (v_patch ? 'scorer_image_reference')
       OR (
         v_patch ? 'scorer_image_digest'
         AND (
           COALESCE(v_patch ->> 'scorer_image_digest', '') !~ '^sha256:[0-9a-f]{64}$'
           OR pg_catalog.char_length(COALESCE(v_patch ->> 'scorer_image_reference', '')) NOT BETWEEN 1 AND 512
           OR pg_catalog.right(
             v_patch ->> 'scorer_image_reference',
             pg_catalog.char_length(v_patch ->> 'scorer_image_digest') + 1
           ) <> '@' || (v_patch ->> 'scorer_image_digest')
         )
       ) THEN
      RAISE EXCEPTION 'lab_arena_scorer_image_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.lab_arena_rounds
    SET status = 'committed', status_generation = status_generation + 1,
        participants = v_patch -> 'participants',
        benchmark_ref = v_patch ->> 'benchmark_ref',
        evaluation_date = v_patch ->> 'evaluation_date',
        configuration_doc = CASE
          WHEN v_patch ? 'scorer_image_digest' THEN
            v_round.configuration_doc || pg_catalog.jsonb_build_object(
              'scorer_image_digest', v_patch ->> 'scorer_image_digest',
              'scorer_image_reference', v_patch ->> 'scorer_image_reference'
            )
          ELSE v_round.configuration_doc
        END
    WHERE round_id = p_round_id;
  ELSIF p_expected_status = 'stage1_closed' AND p_next_status = 'stage1_closed' THEN
    v_allowed := ARRAY['stage1_scoring_plan_doc'];
    IF NOT (v_keys @> v_allowed AND v_allowed @> v_keys)
       OR pg_catalog.jsonb_typeof(v_patch -> 'stage1_scoring_plan_doc') IS DISTINCT FROM 'object' THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    IF v_round.stage1_scoring_plan_doc IS NOT NULL THEN
      IF v_round.stage1_scoring_plan_doc = (v_patch -> 'stage1_scoring_plan_doc') THEN
        RETURN pg_catalog.jsonb_build_object('status', 'existing', 'round_status', v_round.status, 'status_generation', v_round.status_generation);
      END IF;
      RAISE EXCEPTION 'lab_arena_scoring_plan_write_once' USING ERRCODE = '42501';
    END IF;
    UPDATE public.lab_arena_rounds
    SET stage1_scoring_plan_doc = v_patch -> 'stage1_scoring_plan_doc'
    WHERE round_id = p_round_id;
  ELSIF p_expected_status = 'stage2_closed' AND p_next_status = 'stage2_closed' THEN
    v_allowed := ARRAY['stage2_scoring_plan_doc'];
    IF NOT (v_keys @> v_allowed AND v_allowed @> v_keys)
       OR pg_catalog.jsonb_typeof(v_patch -> 'stage2_scoring_plan_doc') IS DISTINCT FROM 'object' THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    IF v_round.stage2_scoring_plan_doc IS NOT NULL THEN
      IF v_round.stage2_scoring_plan_doc = (v_patch -> 'stage2_scoring_plan_doc') THEN
        RETURN pg_catalog.jsonb_build_object('status', 'existing', 'round_status', v_round.status, 'status_generation', v_round.status_generation);
      END IF;
      RAISE EXCEPTION 'lab_arena_scoring_plan_write_once' USING ERRCODE = '42501';
    END IF;
    UPDATE public.lab_arena_rounds
    SET stage2_scoring_plan_doc = v_patch -> 'stage2_scoring_plan_doc'
    WHERE round_id = p_round_id;
  ELSIF p_expected_status = 'stage1_judged' AND p_next_status = 'stage1_scored' THEN
    v_allowed := ARRAY['finalists'];
    IF NOT (v_keys @> v_allowed AND v_allowed @> v_keys)
       OR pg_catalog.jsonb_typeof(v_patch -> 'finalists') IS DISTINCT FROM 'array'
       OR pg_catalog.jsonb_array_length(v_patch -> 'finalists') > 10
       OR v_round.stage1_scoring_plan_doc IS NULL THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    SELECT COUNT(*) INTO v_baseline_count
    FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
    WHERE COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE);
    SELECT COUNT(*) INTO v_missing_scores
    FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
    WHERE (
      SELECT COUNT(DISTINCT runs.icp_position)
      FROM public.lab_arena_runs AS runs
      WHERE runs.round_id = p_round_id
        AND runs.stage = 1
        AND runs.kind = 'execute'
        AND runs.submission_id = participant ->> 'submission_id'
        AND runs.per_icp_score IS NOT NULL
    ) <> 10
      AND (
        v_baseline_count <> 1
        OR COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE)
      );
    IF v_missing_scores <> 0 THEN
      RAISE EXCEPTION 'lab_arena_stage1_baseline_scores_incomplete' USING ERRCODE = '22023';
    END IF;
    SELECT COUNT(*) INTO v_challenger_count
    FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
    WHERE NOT COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE)
      AND (
        v_baseline_count <> 1
        OR (
          SELECT COUNT(DISTINCT runs.icp_position)
          FROM public.lab_arena_runs AS runs
          WHERE runs.round_id = p_round_id
            AND runs.stage = 1
            AND runs.kind = 'execute'
            AND runs.submission_id = participant ->> 'submission_id'
            AND runs.per_icp_score IS NOT NULL
        ) = 10
      );
    SELECT COUNT(*) INTO v_invalid_finalists
    FROM pg_catalog.jsonb_array_elements(v_patch -> 'finalists') AS finalist
    WHERE pg_catalog.jsonb_typeof(finalist) <> 'string'
       OR NOT EXISTS (
         SELECT 1
         FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
         WHERE participant ->> 'submission_id' = finalist #>> '{}'
           AND NOT COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE)
           AND (
             v_baseline_count <> 1
             OR (
               SELECT COUNT(DISTINCT runs.icp_position)
               FROM public.lab_arena_runs AS runs
               WHERE runs.round_id = p_round_id
                 AND runs.stage = 1
                 AND runs.kind = 'execute'
                 AND runs.submission_id = participant ->> 'submission_id'
                 AND runs.per_icp_score IS NOT NULL
             ) = 10
           )
       );
    IF v_invalid_finalists <> 0
       OR pg_catalog.jsonb_array_length(v_patch -> 'finalists') <> LEAST(10, v_challenger_count)
       OR (
         SELECT COUNT(DISTINCT finalist #>> '{}')
         FROM pg_catalog.jsonb_array_elements(v_patch -> 'finalists') AS finalist
       ) <> pg_catalog.jsonb_array_length(v_patch -> 'finalists') THEN
      RAISE EXCEPTION 'lab_arena_finalists_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.lab_arena_rounds
    SET status = 'stage1_scored', status_generation = status_generation + 1,
        finalists = v_patch -> 'finalists'
    WHERE round_id = p_round_id;
  ELSIF p_expected_status = 'stage2_judged' AND p_next_status = 'scored' THEN
    v_allowed := ARRAY[]::TEXT[];
    IF v_keys <> ARRAY[]::TEXT[]
       OR v_round.stage2_scoring_plan_doc IS NULL THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.lab_arena_rounds
    SET status = 'scored', status_generation = status_generation + 1
    WHERE round_id = p_round_id;
  ELSIF p_expected_status = 'scored' AND p_next_status = 'published' THEN
    v_allowed := ARRAY['publication_doc', 'published_at'];
    v_publication := v_patch -> 'publication_doc';
    v_decision := v_publication -> 'king_decision';
    IF NOT (v_keys @> v_allowed AND v_allowed @> v_keys)
       OR pg_catalog.jsonb_typeof(v_patch -> 'published_at') <> 'string'
       OR COALESCE(v_patch ->> 'published_at', '') = ''
       OR pg_catalog.jsonb_typeof(v_publication) IS DISTINCT FROM 'object'
       OR (SELECT pg_catalog.count(*) FROM pg_catalog.jsonb_object_keys(v_publication)) <> 8
       OR v_publication ->> 'schema_version' <> 'leadpoet.lab_arena.publication.v1'
       OR v_publication ->> 'round_id' <> p_round_id
       OR v_publication ->> 'published_at' <> v_patch ->> 'published_at'
       OR pg_catalog.jsonb_typeof(v_publication -> 'participants') IS DISTINCT FROM 'array'
       OR pg_catalog.jsonb_typeof(v_publication -> 'stage1_ranking') IS DISTINCT FROM 'array'
       OR pg_catalog.jsonb_typeof(v_publication -> 'finalists') IS DISTINCT FROM 'array'
       OR v_publication -> 'finalists' IS DISTINCT FROM v_round.finalists
       OR pg_catalog.jsonb_typeof(v_publication -> 'final_ranking') IS DISTINCT FROM 'array'
       OR pg_catalog.jsonb_typeof(v_decision) IS DISTINCT FROM 'object'
       OR v_decision ->> 'outcome' NOT IN ('crowned', 'defended', 'retained_ineligible', 'no_king')
       OR (v_decision ->> 'outcome' = 'no_king' AND COALESCE(v_decision ->> 'king_hotkey', '') <> '')
       OR (v_decision ->> 'outcome' <> 'no_king' AND COALESCE(v_decision ->> 'king_hotkey', '') !~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$') THEN
      RAISE EXCEPTION 'lab_arena_publication_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.lab_arena_rounds
    SET status = 'published', status_generation = status_generation + 1,
        publication_doc = v_patch -> 'publication_doc',
        king_outcome = v_decision ->> 'outcome',
        king_hotkey = NULLIF(v_decision ->> 'king_hotkey', ''),
        published_at = (v_patch ->> 'published_at')::TIMESTAMPTZ
    WHERE round_id = p_round_id;
  ELSE
    RAISE EXCEPTION 'lab_arena_transition_invalid' USING ERRCODE = '22023';
  END IF;

  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'ok', 'round_status', v_round.status, 'status_generation', v_round.status_generation);
END;
$lab_arena_transition_round$;
ALTER FUNCTION public.lab_arena_transition_round(TEXT, TEXT, TEXT, JSONB) OWNER TO lab_arena_owner;

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
  IF NEW.configuration_doc IS DISTINCT FROM OLD.configuration_doc THEN
    IF NOT (
      OLD.status = 'open'
      AND NEW.status = 'committed'
      AND (NEW.configuration_doc - 'scorer_image_digest' - 'scorer_image_reference') =
          (OLD.configuration_doc - 'scorer_image_digest' - 'scorer_image_reference')
      AND COALESCE(NEW.configuration_doc ->> 'scorer_image_digest', '') ~ '^sha256:[0-9a-f]{64}$'
      AND pg_catalog.char_length(
        COALESCE(NEW.configuration_doc ->> 'scorer_image_reference', '')
      ) BETWEEN 1 AND 512
      AND pg_catalog.right(
        NEW.configuration_doc ->> 'scorer_image_reference',
        pg_catalog.char_length(NEW.configuration_doc ->> 'scorer_image_digest') + 1
      ) = '@' || (NEW.configuration_doc ->> 'scorer_image_digest')
    ) THEN
      RAISE EXCEPTION 'round configuration is write-once' USING ERRCODE = '42501';
    END IF;
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
AS $lab_arena_schema_version$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.schema_version.v1', 'version', 194
  );
$lab_arena_schema_version$;
ALTER FUNCTION public.lab_arena_schema_version_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_schema_version_v1() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.lab_arena_schema_version_v1() TO lab_arena_service;
NOTIFY pgrst, 'reload schema';
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;
