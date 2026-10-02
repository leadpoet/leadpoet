-- A late historical publication remains a competition result. A strictly
-- newer published day in the same chain scope owns Git and reward authority.
-- Bind these narrow edits to the exact deployed function definitions.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '120s';

DO $guard_promotion$
DECLARE
  v_definition TEXT := pg_catalog.pg_get_functiondef(
    'public.lab_arena_prepare_promotion(text,jsonb)'::REGPROCEDURE);
  v_old TEXT;
  v_new TEXT;
BEGIN
  IF pg_catalog.strpos(v_definition,'377 monotonic day authority') > 0 THEN
    IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex') <>
       'ff0c1f790c327125276c6805a50f7f85e1cd020c30e36f730c775df05d5a6415' THEN
      RAISE EXCEPTION '377 promotion replay differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex') <>
     '9e7f967eb62058d20a0f0b0806b38a1d218613264cb0f54e2ea27c98ea9b3e1e' THEN
    RAISE EXCEPTION '377 promotion function differs' USING ERRCODE='55000';
  END IF;
  v_old := $old$  v_winner_hotkey TEXT;$old$;
  v_new := $new$  v_winner_hotkey TEXT;
  v_newer_round_id TEXT;$new$;
  IF (pg_catalog.length(v_definition)-pg_catalog.length(pg_catalog.replace(v_definition,v_old,''))) <> pg_catalog.length(v_old) THEN
    RAISE EXCEPTION '377 promotion declaration differs' USING ERRCODE='55000';
  END IF;
  v_definition := pg_catalog.replace(v_definition,v_old,v_new);
  v_old := $old$  v_winner_id := v_round.publication_doc #>> '{king_decision,winner_submission_id}';$old$;
  v_new := $new$  -- 377 monotonic day authority
  SELECT newer.round_id INTO v_newer_round_id
  FROM public.lab_arena_rounds AS newer
  WHERE newer.status = 'published'
    AND newer.configuration_doc ->> 'mode' = 'live'
    AND newer.arena_network_name = v_round.arena_network_name
    AND newer.arena_netuid = v_round.arena_netuid
    AND newer.evaluation_date > v_round.evaluation_date
  ORDER BY newer.evaluation_date DESC, newer.created_at DESC, newer.round_id DESC
  LIMIT 1;
  IF v_newer_round_id IS NOT NULL THEN
    RETURN pg_catalog.jsonb_build_object('status','superseded',
      'newer_round_id',v_newer_round_id);
  END IF;

  v_winner_id := v_round.publication_doc #>> '{king_decision,winner_submission_id}';$new$;
  IF (pg_catalog.length(v_definition)-pg_catalog.length(pg_catalog.replace(v_definition,v_old,''))) <> pg_catalog.length(v_old) THEN
    RAISE EXCEPTION '377 promotion insertion differs' USING ERRCODE='55000';
  END IF;
  v_definition := pg_catalog.replace(v_definition,v_old,v_new);
  v_old := $old$      AND older.baseline_promoted_at IS NULL$old$;
  v_new := $new$      AND older.baseline_promoted_at IS NULL
      AND NOT EXISTS (
        SELECT 1 FROM public.lab_arena_rounds AS newer
        WHERE newer.status = 'published'
          AND newer.configuration_doc ->> 'mode' = 'live'
          AND newer.arena_network_name = older.arena_network_name
          AND newer.arena_netuid = older.arena_netuid
          AND newer.evaluation_date > older.evaluation_date
      )$new$;
  IF (pg_catalog.length(v_definition)-pg_catalog.length(pg_catalog.replace(v_definition,v_old,''))) <> pg_catalog.length(v_old) THEN
    RAISE EXCEPTION '377 promotion predecessor differs' USING ERRCODE='55000';
  END IF;
  EXECUTE pg_catalog.replace(v_definition,v_old,v_new);
END;
$guard_promotion$;

DO $guard_reward$
DECLARE
  v_definition TEXT := pg_catalog.pg_get_functiondef(
    'public.lab_arena_activate_reward(text,jsonb,jsonb)'::REGPROCEDURE);
  v_old TEXT;
  v_new TEXT;
BEGIN
  IF pg_catalog.strpos(v_definition,'377 monotonic day authority') > 0 THEN
    IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex') <>
       '4cbf2feb43a90cd5b0f806301412fa3715906b7f9256a3b3141510efdbb06965' THEN
      RAISE EXCEPTION '377 reward replay differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex') <>
     '7481e80080726d65844f3b101db19745b232304e8f95dfe0c33cd91d71faf090' THEN
    RAISE EXCEPTION '377 reward function differs' USING ERRCODE='55000';
  END IF;
  v_old := $old$  v_expected_start BIGINT := 0;$old$;
  v_new := $new$  v_expected_start BIGINT := 0;
  v_newer_round_id TEXT;$new$;
  IF (pg_catalog.length(v_definition)-pg_catalog.length(pg_catalog.replace(v_definition,v_old,''))) <> pg_catalog.length(v_old) THEN
    RAISE EXCEPTION '377 reward declaration differs' USING ERRCODE='55000';
  END IF;
  v_definition := pg_catalog.replace(v_definition,v_old,v_new);
  v_old := $old$  IF EXISTS (
    SELECT 1 FROM public.lab_arena_rounds AS older
    WHERE older.status = 'published'$old$;
  v_new := $new$  -- 377 monotonic day authority
  SELECT newer.round_id INTO v_newer_round_id
  FROM public.lab_arena_rounds AS newer
  WHERE newer.status = 'published'
    AND newer.configuration_doc ->> 'mode' = 'live'
    AND newer.arena_network_name = v_round.arena_network_name
    AND newer.arena_netuid = v_round.arena_netuid
    AND newer.evaluation_date > v_round.evaluation_date
  ORDER BY newer.evaluation_date DESC, newer.created_at DESC, newer.round_id DESC
  LIMIT 1;
  IF v_newer_round_id IS NOT NULL THEN
    RETURN pg_catalog.jsonb_build_object('status','superseded',
      'newer_round_id',v_newer_round_id);
  END IF;
  IF EXISTS (
    SELECT 1 FROM public.lab_arena_rounds AS older
    WHERE older.status = 'published'$new$;
  IF (pg_catalog.length(v_definition)-pg_catalog.length(pg_catalog.replace(v_definition,v_old,''))) <> pg_catalog.length(v_old) THEN
    RAISE EXCEPTION '377 reward insertion differs' USING ERRCODE='55000';
  END IF;
  v_definition := pg_catalog.replace(v_definition,v_old,v_new);
  v_old := $old$      AND older.reward_activated_at IS NULL$old$;
  v_new := $new$      AND older.reward_activated_at IS NULL
      AND NOT EXISTS (
        SELECT 1 FROM public.lab_arena_rounds AS newer
        WHERE newer.status = 'published'
          AND newer.configuration_doc ->> 'mode' = 'live'
          AND newer.arena_network_name = older.arena_network_name
          AND newer.arena_netuid = older.arena_netuid
          AND newer.evaluation_date > older.evaluation_date
      )$new$;
  IF (pg_catalog.length(v_definition)-pg_catalog.length(pg_catalog.replace(v_definition,v_old,''))) <> pg_catalog.length(v_old) THEN
    RAISE EXCEPTION '377 reward predecessor differs' USING ERRCODE='55000';
  END IF;
  EXECUTE pg_catalog.replace(v_definition,v_old,v_new);
END;
$guard_reward$;

DO $guard_reward_promotion$
DECLARE
  v_definition TEXT := pg_catalog.pg_get_functiondef(
    'public.lab_arena_reward_requires_promotion_v1()'::REGPROCEDURE);
  v_old TEXT;
  v_new TEXT;
BEGIN
  IF pg_catalog.strpos(v_definition,'377 monotonic day authority') > 0 THEN
    IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex') <>
       '6cca8a0dfaa0b1940f649eda23a443f018048d8ceb9c2ae07f456876891ac7b3' THEN
      RAISE EXCEPTION '377 promotion trigger replay differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex') <>
     'b2788f1861e21f439dbb3c03ea3a8dec51edba594849a203c8b1071d1d51e024' THEN
    RAISE EXCEPTION '377 promotion trigger differs' USING ERRCODE='55000';
  END IF;
  v_old := $old$         AND pending.baseline_promoted_at IS NULL$old$;
  v_new := $new$         AND pending.baseline_promoted_at IS NULL
         -- 377 monotonic day authority
         AND NOT EXISTS (
           SELECT 1 FROM public.lab_arena_rounds AS newer
           WHERE newer.status = 'published'
             AND newer.configuration_doc ->> 'mode' = 'live'
             AND newer.arena_network_name = pending.arena_network_name
             AND newer.arena_netuid = pending.arena_netuid
             AND newer.evaluation_date > pending.evaluation_date
         )$new$;
  IF (pg_catalog.length(v_definition)-pg_catalog.length(pg_catalog.replace(v_definition,v_old,''))) <> pg_catalog.length(v_old) THEN
    RAISE EXCEPTION '377 promotion trigger predecessor differs' USING ERRCODE='55000';
  END IF;
  EXECUTE pg_catalog.replace(v_definition,v_old,v_new);
END;
$guard_reward_promotion$;

COMMIT;
