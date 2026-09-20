-- Select the newest usable signed reward basis before deciding whether a
-- champion can continue. A later nonpaying basis is an explicit continuity
-- barrier; it must not allow an older paying champion to return.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

DO $lab_arena_342_prerequisites$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_activate_reward(text,jsonb,jsonb)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena__champion_reward_factor(text,text)'
     ) IS NULL
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'lab_arena_owner'
     ) THEN
    RAISE EXCEPTION 'apply migrations 197 and 227 before migration 342'
      USING ERRCODE = '55000';
  END IF;
END;
$lab_arena_342_prerequisites$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $lab_arena_342_patch$
DECLARE
  v_definition TEXT;
  v_definition_hash TEXT;
  v_original_hash CONSTANT TEXT :=
    'd6a8b1ec1502db3b5546b5434b4d1ed26a306ea01da0e8b1f6e623d5360038e5';
  v_patched_hash CONSTANT TEXT :=
    '7481e80080726d65844f3b101db19745b232304e8f95dfe0c33cd91d71faf090';
  v_old TEXT := $old$
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
$old$;
  v_new TEXT := $new$
  SELECT candidate.reward_basis_doc INTO v_previous_basis
  FROM (
    SELECT reward_basis_doc, signing_key_doc, configuration_doc,
           effective_reward_epoch
    FROM public.lab_arena_rounds
    WHERE reward_activated_at IS NOT NULL
      AND reward_basis_doc IS NOT NULL
      AND signing_key_doc IS NOT NULL
      AND rewards_enabled
      AND configuration_doc ->> 'mode' = 'live'
      AND arena_network_name = v_round.arena_network_name
      AND arena_netuid = v_round.arena_netuid
    ORDER BY effective_reward_epoch DESC
    LIMIT 200
  ) AS candidate
  WHERE candidate.reward_basis_doc ->> 'king_outcome' = 'no_king'
     OR (
       candidate.reward_basis_doc ->> 'king_outcome' IN (
         'crowned', 'defended', 'retained_ineligible'
       )
       AND COALESCE(
         candidate.configuration_doc ->> 'baseline_hotkey', ''
       ) <> ''
       AND COALESCE(
         candidate.reward_basis_doc ->> 'king_hotkey', ''
       ) <> ''
       AND candidate.reward_basis_doc ->> 'king_hotkey'
           IS DISTINCT FROM candidate.configuration_doc ->> 'baseline_hotkey'
     )
  ORDER BY candidate.effective_reward_epoch DESC
  LIMIT 1;

  IF v_previous_basis IS NOT NULL
     AND (
       v_previous_basis ->> 'king_outcome' NOT IN ('crowned', 'defended')
       OR v_previous_basis ->> 'king_hotkey'
          IS NOT DISTINCT FROM v_round.configuration_doc ->> 'baseline_hotkey'
     ) THEN
    v_previous_basis := NULL;
  END IF;
$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_activate_reward(text,jsonb,jsonb)'::pg_catalog.regprocedure
  ) INTO v_definition;
  v_definition_hash := pg_catalog.encode(
    extensions.digest(v_definition, 'sha256'), 'hex'
  );
  IF v_definition_hash = v_original_hash THEN
    IF (
      pg_catalog.length(v_definition)
      - pg_catalog.length(pg_catalog.replace(v_definition, v_old, ''))
    ) <> pg_catalog.length(v_old) THEN
      RAISE EXCEPTION 'reward predecessor barrier seam differs'
        USING ERRCODE = '55000';
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  ELSIF v_definition_hash <> v_patched_hash THEN
    RAISE EXCEPTION 'reward activation definition differs: %',
      v_definition_hash USING ERRCODE = '55000';
  END IF;

  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_activate_reward(text,jsonb,jsonb)'::pg_catalog.regprocedure
  ) INTO v_definition;
  v_definition_hash := pg_catalog.encode(
    extensions.digest(v_definition, 'sha256'), 'hex'
  );
  IF v_definition_hash <> v_patched_hash THEN
    RAISE EXCEPTION 'reward predecessor barrier patch differs: %',
      v_definition_hash USING ERRCODE = '55000';
  END IF;
END;
$lab_arena_342_patch$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
