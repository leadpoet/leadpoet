-- Prospective total-emissions slots. Preserve frozen bases, ordinal incumbents,
-- existing publication proofs, exceptions591/592 and function permissions.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
DO $total_emissions681$
DECLARE
  v_signature REGPROCEDURE := 'public.lab_arena_reward_slot_snapshot(text,jsonb)'::REGPROCEDURE;
  v_definition TEXT;
  v_updated TEXT;
  v_identity JSONB;
  v_schema_acl ACLITEM[];
  v_hash TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner), p.proacl::TEXT,
      p.prosecdef, p.provolatile, p.proconfig, l.lanname)
  INTO v_definition, v_identity FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_language AS l ON l.oid = p.prolang WHERE p.oid = v_signature;
  SELECT nspacl INTO v_schema_acl FROM pg_catalog.pg_namespace WHERE nspname = 'public';
  IF v_identity IS DISTINCT FROM
    '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION '681 reward slot security differs' USING ERRCODE = '55000';
  END IF;
  v_hash := pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex');
  IF v_hash = '953fce9f8c5b5fff1cfc01e296113bd9844d957c8981ba216c14d619e91d6ad1' THEN RETURN; END IF;
  IF v_hash <> '0870c5ab466388cdc89b8396e7565e547f4278b2870ea18c3e8bd2bb4dfca5c6' THEN
    RAISE EXCEPTION '681 reward slot preimage differs' USING ERRCODE = '55000';
  END IF;
  v_updated := v_definition;
  v_updated := pg_catalog.replace(v_updated, $old0$  v_current_verified BOOLEAN;$old0$,
    $new0$  v_current_verified BOOLEAN;
  v_total_emissions BOOLEAN;
  v_prior public.lab_arena_rounds;
  v_carried JSONB := '[null,null,null]'::JSONB;
  v_entry JSONB;
  v_carried_verified BOOLEAN;
  v_source_policy JSONB;$new0$);
  v_updated := pg_catalog.replace(v_updated, $old1$p_slot_policy - 'pool_percent' - 'decay'$old1$,
    $new1$p_slot_policy - 'pool_percent' - 'decay' - 'allocation_basis'$new1$);
  v_updated := pg_catalog.replace(v_updated, $old2$  IF p_slot_policy ? 'decay' THEN$old2$,
    $new2$  IF p_slot_policy ? 'allocation_basis' AND (
       p_slot_policy -> 'allocation_basis' IS DISTINCT FROM '"total_emissions"'::JSONB
       OR NOT (p_slot_policy ? 'pool_percent')
     ) THEN
    RAISE EXCEPTION 'lab_arena_slot_policy_invalid' USING ERRCODE = '22023';
  END IF;
  v_total_emissions := p_slot_policy ->> 'allocation_basis' = 'total_emissions';
  IF p_slot_policy ? 'decay' THEN$new2$);
  v_updated := pg_catalog.replace(v_updated, $old3$  IF v_share_total <> 100 THEN$old3$,
    $new3$  IF v_share_total <> (CASE WHEN v_total_emissions THEN (p_slot_policy ->> 'pool_percent')::NUMERIC ELSE 100 END) THEN$new3$);
  v_updated := pg_catalog.replace(v_updated, $old4$  WITH tiers AS ($old4$,
    $new4$  -- 681 prospective allocation: carry signed ordinal holders, never search
  -- old winners under the new thresholds. Only this round may replace them.
  IF v_total_emissions THEN
    SELECT * INTO v_prior FROM public.lab_arena_rounds AS prior
    WHERE prior.round_id <> p_round_id
      AND prior.status = 'published' AND prior.rewards_enabled
      AND prior.configuration_doc ->> 'mode' = 'live'
      AND prior.arena_network_name = v_round.arena_network_name
      AND prior.arena_netuid = v_round.arena_netuid
      AND prior.evaluation_date <= v_round.evaluation_date
      AND prior.reward_activated_at IS NOT NULL
      AND prior.signing_key_doc IS NOT NULL
      AND prior.reward_basis_doc ->> 'schema_version' = 'leadpoet.lab_arena.reward_basis.v2'
      AND (v_round.effective_reward_epoch IS NULL
        OR prior.effective_reward_epoch < v_round.effective_reward_epoch)
    ORDER BY prior.effective_reward_epoch DESC, prior.evaluation_date DESC, prior.round_id DESC
    LIMIT 1;
    IF FOUND THEN
      v_carried := v_prior.reward_basis_doc -> 'reward_slots';
      IF pg_catalog.jsonb_typeof(v_carried) IS DISTINCT FROM 'array' THEN
        RAISE EXCEPTION 'lab_arena_reward_carried_slots_invalid' USING ERRCODE = '22023';
      END IF;
      IF pg_catalog.jsonb_array_length(v_carried) <> 3
         OR v_prior.effective_reward_epoch IS NULL OR v_prior.effective_reward_epoch < 0
         OR v_prior.reward_basis_doc ->> 'round_id' IS DISTINCT FROM v_prior.round_id
         OR v_prior.reward_basis_doc -> 'effective_reward_epoch' IS DISTINCT FROM
           pg_catalog.to_jsonb(v_prior.effective_reward_epoch) THEN
        RAISE EXCEPTION 'lab_arena_reward_carried_slots_invalid' USING ERRCODE = '22023';
      END IF;
      FOR v_entry IN SELECT value FROM pg_catalog.jsonb_array_elements(v_carried) LOOP
        IF v_entry = 'null'::JSONB THEN CONTINUE; END IF;
        IF pg_catalog.jsonb_typeof(v_entry) IS DISTINCT FROM 'object' THEN
          RAISE EXCEPTION 'lab_arena_reward_carried_slots_invalid' USING ERRCODE = '22023';
        END IF;
        IF (SELECT pg_catalog.count(*) FROM pg_catalog.jsonb_object_keys(v_entry - 'start_epoch')) <> 6
           OR (((v_prior.reward_basis_doc -> 'slot_policy') ? 'decay')
             IS DISTINCT FROM (v_entry ? 'start_epoch'))
           OR NOT (v_entry ?& ARRAY['round_id','submission_id','miner_hotkey',
             'baseline_submission_id','baseline_score','winner_score'])
           OR EXISTS (SELECT 1 FROM pg_catalog.unnest(ARRAY['round_id','submission_id',
             'miner_hotkey','baseline_submission_id']) AS names(name)
             WHERE pg_catalog.jsonb_typeof(v_entry -> name) IS DISTINCT FROM 'string'
               OR v_entry ->> name = '')
           OR pg_catalog.jsonb_typeof(v_entry -> 'baseline_score') IS DISTINCT FROM 'number'
           OR pg_catalog.jsonb_typeof(v_entry -> 'winner_score') IS DISTINCT FROM 'number'
           OR (v_entry ? 'start_epoch' AND NOT (
             (v_entry -> 'start_epoch' = 'null'::JSONB AND v_entry ->> 'round_id' = v_prior.round_id)
             OR (pg_catalog.jsonb_typeof(v_entry -> 'start_epoch') = 'number'
               AND (v_entry ->> 'start_epoch') ~ '^(0|[1-9][0-9]*)$')
           )) THEN
          RAISE EXCEPTION 'lab_arena_reward_carried_slots_invalid' USING ERRCODE = '22023';
        END IF;
        IF v_entry ->> 'start_epoch' IS NOT NULL AND
           (v_entry ->> 'start_epoch')::NUMERIC > v_prior.effective_reward_epoch THEN
          RAISE EXCEPTION 'lab_arena_reward_carried_slots_invalid' USING ERRCODE = '22023';
        END IF;
      END LOOP;
      v_source_policy := v_prior.reward_basis_doc -> 'slot_policy';
      IF COALESCE(v_source_policy ->> 'allocation_basis', '') <> 'total_emissions'
         AND pg_catalog.jsonb_build_array(
           v_source_policy #> '{tiers,0,minimum_improvement}',
           v_source_policy #> '{tiers,1,minimum_improvement}',
           v_source_policy #> '{tiers,2,minimum_improvement}') IS DISTINCT FROM '[20,15,8]'::JSONB THEN
        IF pg_catalog.jsonb_build_array(
             v_source_policy #> '{tiers,0,minimum_improvement}',
             v_source_policy #> '{tiers,1,minimum_improvement}',
             v_source_policy #> '{tiers,2,minimum_improvement}') IS DISTINCT FROM '[10,5,1]'::JSONB
           OR v_source_policy ->> 'assignment_mode' IS DISTINCT FROM 'all_qualifying' THEN
          RAISE EXCEPTION 'lab_arena_reward_carried_policy_invalid' USING ERRCODE = '22023';
        END IF;
        -- Match the existing October23 projection using only represented holders.
        SELECT pg_catalog.jsonb_agg(selected.entry ORDER BY thresholds.ordinality)
          INTO v_carried
        FROM pg_catalog.jsonb_array_elements('[20,15,8]'::JSONB) WITH ORDINALITY AS thresholds(minimum,ordinality)
        LEFT JOIN LATERAL (
          SELECT old.entry FROM pg_catalog.jsonb_array_elements(v_carried) WITH ORDINALITY AS old(entry,ordinality)
          WHERE old.entry <> 'null'::JSONB
            AND (old.entry ->> 'winner_score')::NUMERIC - (old.entry ->> 'baseline_score')::NUMERIC
              >= (thresholds.minimum::TEXT)::NUMERIC
          ORDER BY old.ordinality DESC LIMIT 1
        ) AS selected ON TRUE;
      END IF;
      IF EXISTS (SELECT 1 FROM pg_catalog.jsonb_array_elements(v_carried) WITH ORDINALITY AS old(entry,ordinality)
        WHERE old.entry <> 'null'::JSONB
          AND (old.entry ->> 'winner_score')::NUMERIC - (old.entry ->> 'baseline_score')::NUMERIC
            < (p_slot_policy #>> ARRAY['tiers',(old.ordinality - 1)::TEXT,'minimum_improvement'])::NUMERIC) THEN
        RAISE EXCEPTION 'lab_arena_reward_carried_slots_invalid' USING ERRCODE = '22023';
      END IF;
    END IF;
  END IF;
  WITH tiers AS ($new4$);
  v_updated := pg_catalog.replace(v_updated, $old5$  SELECT pg_catalog.jsonb_agg(slot.entry ORDER BY tiers.ordinality),$old5$,
    $new5$  SELECT pg_catalog.jsonb_agg(CASE WHEN v_total_emissions THEN
      COALESCE(slot.entry, carried.entry) ELSE slot.entry END ORDER BY tiers.ordinality),$new5$);
  v_updated := pg_catalog.replace(v_updated, $old6$      AND v.baseline_score BETWEEN 0 AND 100 AND v.winner_score BETWEEN 0 AND 100)
  INTO v_slots, v_current_verified$old6$,
    $new6$      AND v.baseline_score BETWEEN 0 AND 100 AND v.winner_score BETWEEN 0 AND 100),
    NOT EXISTS (
      SELECT 1 FROM pg_catalog.jsonb_array_elements(v_carried) AS old(entry)
      WHERE old.entry <> 'null'::JSONB AND NOT EXISTS (
        SELECT 1 FROM verified AS v
        WHERE v.round_id <> p_round_id
          AND v.evaluation_date <= v_prior.evaluation_date
          AND v.effective_reward_epoch <= v_prior.effective_reward_epoch
          AND v.baseline_score BETWEEN 0 AND 100 AND v.winner_score BETWEEN 0 AND 100
          AND old.entry - 'start_epoch' = pg_catalog.jsonb_build_object(
            'round_id', v.round_id, 'submission_id', v.submission_id,
            'miner_hotkey', v.winner_hotkey, 'baseline_submission_id', v.baseline_submission_id,
            'baseline_score', v.baseline_score, 'winner_score', v.winner_score)
      )
    )
  INTO v_slots, v_current_verified, v_carried_verified$new6$);
  v_updated := pg_catalog.replace(v_updated, $old7$  FROM tiers
  LEFT JOIN LATERAL ($old7$,
    $new7$  FROM tiers
  LEFT JOIN LATERAL (
    SELECT CASE WHEN v_carried -> (tiers.ordinality::INT - 1) = 'null'::JSONB THEN NULL
      ELSE (v_carried -> (tiers.ordinality::INT - 1)) - 'start_epoch'
        || CASE WHEN p_slot_policy ? 'decay' THEN pg_catalog.jsonb_build_object(
          'start_epoch', COALESCE(
            v_carried -> (tiers.ordinality::INT - 1) ->> 'start_epoch',
            CASE WHEN v_carried -> (tiers.ordinality::INT - 1) ->> 'round_id' = v_prior.round_id
              THEN v_prior.effective_reward_epoch::TEXT ELSE source.effective_reward_epoch::TEXT END
          )::BIGINT) ELSE '{}'::JSONB END END AS entry
    FROM (SELECT 1) AS singleton
    LEFT JOIN public.lab_arena_rounds AS source
      ON source.round_id = v_carried -> (tiers.ordinality::INT - 1) ->> 'round_id'
  ) AS carried ON TRUE
  LEFT JOIN LATERAL ($new7$);
  v_updated := pg_catalog.replace(v_updated, $old8$    WHERE v.baseline_score BETWEEN 0 AND 100 AND v.winner_score BETWEEN 0 AND 100
      AND v.winner_score$old8$,
    $new8$    WHERE v.baseline_score BETWEEN 0 AND 100 AND v.winner_score BETWEEN 0 AND 100
      AND (NOT COALESCE(v_total_emissions, FALSE) OR v.round_id = p_round_id)
      AND v.winner_score$new8$);
  v_updated := pg_catalog.replace(v_updated, $old9$  IF v_round.status = 'published' AND v_round.rewards_enabled$old9$,
    $new9$  IF v_total_emissions AND NOT v_carried_verified THEN
    RAISE EXCEPTION 'lab_arena_reward_carried_slots_invalid' USING ERRCODE = '22023';
  END IF;
  IF v_round.status = 'published' AND v_round.rewards_enabled$new9$);
  IF pg_catalog.encode(extensions.digest(v_updated, 'sha256'), 'hex') <> '953fce9f8c5b5fff1cfc01e296113bd9844d957c8981ba216c14d619e91d6ad1' THEN
    RAISE EXCEPTION '681 reward slot postimage differs' USING ERRCODE = '55000';
  END IF;
  EXECUTE v_updated;
  IF (SELECT pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner), p.proacl::TEXT,
       p.prosecdef, p.provolatile, p.proconfig, l.lanname)
      FROM pg_catalog.pg_proc AS p JOIN pg_catalog.pg_language AS l ON l.oid = p.prolang
      WHERE p.oid = v_signature) IS DISTINCT FROM v_identity
     OR pg_catalog.pg_get_functiondef(v_signature) IS DISTINCT FROM v_updated
     OR (SELECT nspacl FROM pg_catalog.pg_namespace WHERE nspname = 'public') IS DISTINCT FROM v_schema_acl THEN
    RAISE EXCEPTION '681 reward slot readback differs' USING ERRCODE = '55000';
  END IF;
END;
$total_emissions681$;
NOTIFY pgrst, 'reload schema';
COMMIT;
