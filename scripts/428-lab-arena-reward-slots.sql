-- Reward slots allocate already published and successfully promoted winners.
-- Scoring, crown selection, promotion and immutable historical bases are unchanged.
-- Tier constants are supplied by the signed source policy, never duplicated here.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

CREATE OR REPLACE FUNCTION public.lab_arena_reward_slot_snapshot(
  p_round_id TEXT, p_slot_policy JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $reward_slot_snapshot$
DECLARE
  v_round public.lab_arena_rounds;
  v_tier JSONB;
  v_minimum NUMERIC;
  v_previous NUMERIC;
  v_share_total NUMERIC := 0;
  v_slots JSONB;
  v_current_verified BOOLEAN;
BEGIN
  IF pg_catalog.jsonb_typeof(p_slot_policy) IS DISTINCT FROM 'object' THEN
    RAISE EXCEPTION 'lab_arena_slot_policy_invalid' USING ERRCODE = '22023';
  END IF;
  IF (SELECT pg_catalog.count(*) FROM pg_catalog.jsonb_object_keys(p_slot_policy)) <> 2
     OR COALESCE(p_slot_policy ->> 'assignment_mode', '') NOT IN ('highest_only', 'all_qualifying')
     OR pg_catalog.jsonb_typeof(p_slot_policy -> 'tiers') IS DISTINCT FROM 'array' THEN
    RAISE EXCEPTION 'lab_arena_slot_policy_invalid' USING ERRCODE = '22023';
  END IF;
  IF pg_catalog.jsonb_array_length(p_slot_policy -> 'tiers') <> 3 THEN
    RAISE EXCEPTION 'lab_arena_slot_policy_invalid' USING ERRCODE = '22023';
  END IF;
  FOR v_tier IN SELECT value FROM pg_catalog.jsonb_array_elements(p_slot_policy -> 'tiers') LOOP
    IF pg_catalog.jsonb_typeof(v_tier) IS DISTINCT FROM 'object' THEN
      RAISE EXCEPTION 'lab_arena_slot_policy_invalid' USING ERRCODE = '22023';
    END IF;
    IF (SELECT pg_catalog.count(*) FROM pg_catalog.jsonb_object_keys(v_tier)) <> 2
       OR pg_catalog.jsonb_typeof(v_tier -> 'minimum_improvement') IS DISTINCT FROM 'number'
       OR COALESCE(v_tier ->> 'minimum_improvement', '') !~ '^[1-9][0-9]*$'
       OR pg_catalog.jsonb_typeof(v_tier -> 'allocation_percent') IS DISTINCT FROM 'number'
       OR COALESCE(v_tier ->> 'allocation_percent', '') !~ '^[1-9][0-9]*$' THEN
      RAISE EXCEPTION 'lab_arena_slot_policy_invalid' USING ERRCODE = '22023';
    END IF;
    v_minimum := (v_tier ->> 'minimum_improvement')::NUMERIC;
    IF v_minimum > 100 OR (v_previous IS NOT NULL AND v_minimum >= v_previous) THEN
      RAISE EXCEPTION 'lab_arena_slot_policy_invalid' USING ERRCODE = '22023';
    END IF;
    v_previous := v_minimum;
    v_share_total := v_share_total + (v_tier ->> 'allocation_percent')::NUMERIC;
  END LOOP;
  IF v_share_total <> 100 THEN
    RAISE EXCEPTION 'lab_arena_slot_policy_invalid' USING ERRCODE = '22023';
  END IF;

  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  WITH tiers AS (
    SELECT value, ordinality,
      (value ->> 'minimum_improvement')::NUMERIC AS minimum,
      pg_catalog.lag((value ->> 'minimum_improvement')::NUMERIC)
        OVER (ORDER BY ordinality) AS upper_minimum
    FROM pg_catalog.jsonb_array_elements(p_slot_policy -> 'tiers') WITH ORDINALITY
  ), candidates AS (
    SELECT r.round_id, r.evaluation_date, r.baseline_promoted_at,
      r.publication_doc #>> '{king_decision,winner_submission_id}' AS winner_id,
      r.publication_doc #>> '{king_decision,king_hotkey}' AS winner_hotkey,
      r.configuration_doc ->> 'baseline_hotkey' AS baseline_hotkey,
      r.publication_doc
    FROM public.lab_arena_rounds AS r
    WHERE v_round.status = 'published' AND v_round.rewards_enabled
      AND v_round.configuration_doc ->> 'mode' = 'live'
      AND r.status = 'published' AND r.rewards_enabled
      AND r.configuration_doc ->> 'mode' = 'live'
      AND r.arena_network_name = v_round.arena_network_name
      AND r.arena_netuid = v_round.arena_netuid
      AND r.evaluation_date <= v_round.evaluation_date
      AND r.baseline_promoted_at IS NOT NULL
      AND (r.round_id = p_round_id OR (
        r.reward_activated_at IS NOT NULL AND r.reward_basis_doc IS NOT NULL
        AND r.signing_key_doc IS NOT NULL
      ))
      AND r.publication_doc #>> '{king_decision,outcome}' = 'crowned'
      AND r.publication_doc ->> 'schema_version' = 'leadpoet.lab_arena.publication.v1'
      AND r.publication_doc ->> 'round_id' = r.round_id
      AND COALESCE(r.publication_doc #>> '{king_decision,winner_submission_id}', '') <> ''
      AND r.publication_doc #>> '{king_decision,winner_submission_id}' =
        r.publication_doc #>> '{king_decision,king_submission_id}'
      AND COALESCE(r.publication_doc #>> '{king_decision,king_hotkey}', '') <> ''
      AND COALESCE(r.configuration_doc ->> 'baseline_hotkey', '') <> ''
      AND r.publication_doc #>> '{king_decision,king_hotkey}' <>
        r.configuration_doc ->> 'baseline_hotkey'
      AND r.publication_doc #>> '{king_decision,king_hotkey}' IS DISTINCT FROM
        v_round.configuration_doc ->> 'baseline_hotkey'
  ), verified AS (
    SELECT c.*, winner.entry ->> 'submission_id' AS submission_id,
      baseline.entry ->> 'submission_id' AS baseline_submission_id,
      CASE WHEN pg_catalog.jsonb_typeof(winner.entry -> 'final_score') = 'number'
        THEN (winner.entry ->> 'final_score')::NUMERIC END AS winner_score,
      CASE WHEN pg_catalog.jsonb_typeof(baseline.entry -> 'final_score') = 'number'
        THEN (baseline.entry ->> 'final_score')::NUMERIC END AS baseline_score
    FROM candidates AS c
    CROSS JOIN LATERAL (
      SELECT pg_catalog.jsonb_agg(ranked.entry) -> 0 AS entry, pg_catalog.count(*) AS count
      FROM pg_catalog.jsonb_array_elements(CASE
        WHEN pg_catalog.jsonb_typeof(c.publication_doc -> 'final_ranking') = 'array'
          THEN c.publication_doc -> 'final_ranking' ELSE '[]'::JSONB END) AS ranked(entry)
      WHERE ranked.entry ->> 'submission_id' = c.winner_id
    ) AS winner
    CROSS JOIN LATERAL (
      SELECT pg_catalog.jsonb_agg(ranked.entry) -> 0 AS entry, pg_catalog.count(*) AS count
      FROM pg_catalog.jsonb_array_elements(CASE
        WHEN pg_catalog.jsonb_typeof(c.publication_doc -> 'final_ranking') = 'array'
          THEN c.publication_doc -> 'final_ranking' ELSE '[]'::JSONB END) AS ranked(entry)
      WHERE ranked.entry -> 'is_baseline' = 'true'::JSONB
    ) AS baseline
    WHERE winner.count = 1 AND baseline.count = 1
      AND winner.entry -> 'eligible' = 'true'::JSONB
      AND winner.entry -> 'is_baseline' = 'false'::JSONB
      AND baseline.entry -> 'eligible' = 'true'::JSONB
      AND COALESCE(baseline.entry ->> 'submission_id', '') <> ''
      AND baseline.entry ->> 'submission_id' <> c.winner_id
      AND (SELECT pg_catalog.count(*) FROM pg_catalog.jsonb_array_elements(CASE
        WHEN pg_catalog.jsonb_typeof(c.publication_doc -> 'final_ranking') = 'array'
          THEN c.publication_doc -> 'final_ranking' ELSE '[]'::JSONB END) AS ranked(entry)
        WHERE ranked.entry ->> 'submission_id' = baseline.entry ->> 'submission_id') = 1
      AND (SELECT pg_catalog.count(*) FROM pg_catalog.jsonb_array_elements(CASE
        WHEN pg_catalog.jsonb_typeof(c.publication_doc -> 'participants') = 'array'
          THEN c.publication_doc -> 'participants' ELSE '[]'::JSONB END) AS participant(p)
        WHERE p ->> 'submission_id' = c.winner_id) = 1
      AND (SELECT pg_catalog.count(*) FROM pg_catalog.jsonb_array_elements(CASE
        WHEN pg_catalog.jsonb_typeof(c.publication_doc -> 'participants') = 'array'
          THEN c.publication_doc -> 'participants' ELSE '[]'::JSONB END) AS participant(p)
        WHERE p ->> 'submission_id' = baseline.entry ->> 'submission_id') = 1
      AND (SELECT pg_catalog.count(*) FROM pg_catalog.jsonb_array_elements(CASE
        WHEN pg_catalog.jsonb_typeof(c.publication_doc -> 'participants') = 'array'
          THEN c.publication_doc -> 'participants' ELSE '[]'::JSONB END) AS participant(p)
        WHERE p -> 'is_baseline' = 'true'::JSONB) = 1
      AND EXISTS (SELECT 1 FROM pg_catalog.jsonb_array_elements(CASE
        WHEN pg_catalog.jsonb_typeof(c.publication_doc -> 'participants') = 'array'
          THEN c.publication_doc -> 'participants' ELSE '[]'::JSONB END) AS participant(p)
        WHERE p ->> 'submission_id' = c.winner_id
          AND p ->> 'miner_hotkey' = c.winner_hotkey
          AND p -> 'is_baseline' = 'false'::JSONB)
      AND EXISTS (SELECT 1 FROM pg_catalog.jsonb_array_elements(CASE
        WHEN pg_catalog.jsonb_typeof(c.publication_doc -> 'participants') = 'array'
          THEN c.publication_doc -> 'participants' ELSE '[]'::JSONB END) AS participant(p)
        WHERE p ->> 'submission_id' = baseline.entry ->> 'submission_id'
          AND p ->> 'miner_hotkey' = c.baseline_hotkey
          AND p -> 'is_baseline' = 'true'::JSONB)
  )
  SELECT pg_catalog.jsonb_agg(slot.entry ORDER BY tiers.ordinality),
    EXISTS (SELECT 1 FROM verified AS v WHERE v.round_id = p_round_id
      AND v.baseline_score BETWEEN 0 AND 100 AND v.winner_score BETWEEN 0 AND 100)
  INTO v_slots, v_current_verified
  FROM tiers
  LEFT JOIN LATERAL (
    SELECT pg_catalog.jsonb_build_object(
      'round_id', v.round_id, 'submission_id', v.submission_id,
      'miner_hotkey', v.winner_hotkey,
      'baseline_submission_id', v.baseline_submission_id,
      'baseline_score', v.baseline_score, 'winner_score', v.winner_score
    ) AS entry
    FROM verified AS v
    WHERE v.baseline_score BETWEEN 0 AND 100 AND v.winner_score BETWEEN 0 AND 100
      AND v.winner_score - v.baseline_score >= tiers.minimum
      AND (p_slot_policy ->> 'assignment_mode' = 'all_qualifying'
        OR tiers.upper_minimum IS NULL
        OR v.winner_score - v.baseline_score < tiers.upper_minimum)
    ORDER BY v.evaluation_date DESC, v.baseline_promoted_at DESC, v.round_id DESC
    LIMIT 1
  ) AS slot ON TRUE;
  IF v_round.status = 'published' AND v_round.rewards_enabled
     AND v_round.configuration_doc ->> 'mode' = 'live'
     AND v_round.baseline_promoted_at IS NOT NULL
     AND v_round.publication_doc #>> '{king_decision,outcome}' = 'crowned'
     AND NOT v_current_verified THEN
    RAISE EXCEPTION 'lab_arena_reward_current_achievement_invalid' USING ERRCODE = '22023';
  END IF;
  RETURN pg_catalog.jsonb_build_object('reward_slots', v_slots);
END;
$reward_slot_snapshot$;
ALTER FUNCTION public.lab_arena_reward_slot_snapshot(TEXT, JSONB) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_reward_slot_snapshot(TEXT, JSONB) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.lab_arena_reward_slot_snapshot(TEXT, JSONB) TO lab_arena_service;
DO $reward_slots_acl$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'service_role') THEN
    GRANT EXECUTE ON FUNCTION public.lab_arena_reward_slot_snapshot(TEXT, JSONB) TO service_role;
  END IF;
END;
$reward_slots_acl$;

-- Preserve every existing activation guard. Only admit v2 and verify its slots
-- under the same transaction advisory lock before writing its signed basis.
DO $reward_slots_activation$
DECLARE
  v_definition TEXT := pg_catalog.pg_get_functiondef(
    'public.lab_arena_activate_reward(text,jsonb,jsonb)'::REGPROCEDURE);
  v_hash TEXT;
  v_old_schema TEXT := $old_schema$     OR p_reward_basis ->> 'schema_version' <> 'leadpoet.lab_arena.reward_basis.v1'$old_schema$;
  v_new_schema TEXT := $new_schema$     OR COALESCE(p_reward_basis ->> 'schema_version', '') NOT IN (
       'leadpoet.lab_arena.reward_basis.v1', 'leadpoet.lab_arena.reward_basis.v2'
     )$new_schema$;
  v_old_insert TEXT := $old_insert$  SELECT pg_catalog.max(effective_reward_epoch)$old_insert$;
  v_new_insert TEXT := $new_insert$  -- 428 signed reward slots: allocation only; existing crown authority remains.
  IF p_reward_basis ->> 'schema_version' = 'leadpoet.lab_arena.reward_basis.v2'
     AND v_round.publication_doc #>> '{king_decision,outcome}' = 'crowned'
     AND v_round.baseline_promoted_at IS NULL THEN
    RETURN pg_catalog.jsonb_build_object('status', 'waiting_for_promotion');
  END IF;
  IF p_reward_basis ->> 'schema_version' = 'leadpoet.lab_arena.reward_basis.v1'
     AND EXISTS (
       SELECT 1 FROM public.lab_arena_rounds AS prior
       WHERE prior.reward_activated_at IS NOT NULL
         AND prior.arena_network_name = v_round.arena_network_name
         AND prior.arena_netuid = v_round.arena_netuid
         AND prior.reward_basis_doc ->> 'schema_version' = 'leadpoet.lab_arena.reward_basis.v2'
     ) THEN
    RAISE EXCEPTION 'lab_arena_reward_basis_downgrade' USING ERRCODE = '22023';
  END IF;
  IF p_reward_basis ->> 'schema_version' = 'leadpoet.lab_arena.reward_basis.v2'
     AND p_reward_basis -> 'reward_slots' IS DISTINCT FROM
       public.lab_arena_reward_slot_snapshot(
         p_round_id, p_reward_basis -> 'slot_policy'
       ) -> 'reward_slots' THEN
    RAISE EXCEPTION 'lab_arena_reward_slots_mismatch' USING ERRCODE = '22023';
  END IF;
  SELECT pg_catalog.max(effective_reward_epoch)$new_insert$;
BEGIN
  v_hash := pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex');
  IF v_hash = 'd9df099d6c480f3ee0e930ee535555d67c103ed115181b123d64826ac6c11642' THEN
    RETURN;
  END IF;
  IF v_hash <> '4cbf2feb43a90cd5b0f806301412fa3715906b7f9256a3b3141510efdbb06965' THEN
    RAISE EXCEPTION '428 reward activation definition differs' USING ERRCODE = '55000';
  END IF;
  v_definition := pg_catalog.replace(v_definition, v_old_schema, v_new_schema);
  v_definition := pg_catalog.replace(v_definition, v_old_insert, v_new_insert);
  IF pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex') <>
     'd9df099d6c480f3ee0e930ee535555d67c103ed115181b123d64826ac6c11642' THEN
    RAISE EXCEPTION '428 reward activation seam differs' USING ERRCODE = '55000';
  END IF;
  EXECUTE v_definition;
END;
$reward_slots_activation$;

NOTIFY pgrst, 'reload schema';
COMMIT;
