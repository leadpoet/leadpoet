-- Preserve empty legacy capture contracts as proof of a champion source freeze.
-- Change only the existing compact reader. No historical rows are modified.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $defender601$
DECLARE
  target REGPROCEDURE :=
    'public.lab_arena_public_configuration_v1(public.lab_arena_rounds)'::REGPROCEDURE;
  definition TEXT;
  updated TEXT;
  before_hash TEXT;
  identity JSONB;
  anchor CONSTANT TEXT :=
    $old$SELECT COALESCE(pg_catalog.jsonb_object_agg(key, value), '{}'::JSONB)$old$;
  replacement CONSTANT TEXT :=
    $new$SELECT COALESCE(pg_catalog.jsonb_object_agg(key, value), '{}'::JSONB)
    || pg_catalog.jsonb_build_object('baseline_champion_source_frozen',
      p_round.champion_funding_frozen IS TRUE
      AND pg_catalog.jsonb_typeof(p_round.configuration_doc -> 'baseline_reasoning_configuration')
        IS NOT DISTINCT FROM 'object')$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid), pg_catalog.jsonb_build_array(
    pg_catalog.pg_get_userbyid(p.proowner), p.proacl::TEXT, p.prosecdef,
    p.provolatile, p.proconfig, language.lanname)
  INTO definition, identity
  FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_language language ON language.oid = p.prolang
  WHERE p.oid = target;
  IF identity IS DISTINCT FROM
    '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",false,"i",["search_path=pg_catalog, public"],"sql"]'::JSONB THEN
    RAISE EXCEPTION 'defending champion projection security differs' USING ERRCODE = '42501';
  END IF;
  before_hash := encode(extensions.digest(definition, 'sha256'), 'hex');
  IF before_hash = '27c919a93360fa64d5b7dc33a931f3254edf3b8596b9da15d565ccca4fb7101b' THEN
    RETURN;
  END IF;
  IF before_hash <> 'd679d7b678ca6fc0d1e23cfc5cac0455981c7f09727ae98c3b9d2325acd32017'
    OR length(definition) - length(replace(definition, anchor, '')) <> length(anchor) THEN
    RAISE EXCEPTION 'defending champion projection preimage differs' USING ERRCODE = '55000';
  END IF;
  updated := replace(definition, anchor, replacement);
  IF encode(extensions.digest(updated, 'sha256'), 'hex') <>
    '27c919a93360fa64d5b7dc33a931f3254edf3b8596b9da15d565ccca4fb7101b' THEN
    RAISE EXCEPTION 'defending champion projection postimage differs' USING ERRCODE = '55000';
  END IF;
  EXECUTE updated;
  IF encode(extensions.digest(pg_catalog.pg_get_functiondef(target), 'sha256'), 'hex') <>
    '27c919a93360fa64d5b7dc33a931f3254edf3b8596b9da15d565ccca4fb7101b'
    OR (SELECT pg_catalog.jsonb_build_array(
      pg_catalog.pg_get_userbyid(p.proowner), p.proacl::TEXT, p.prosecdef,
      p.provolatile, p.proconfig, language.lanname)
      FROM pg_catalog.pg_proc p
      JOIN pg_catalog.pg_language language ON language.oid = p.prolang
      WHERE p.oid = target) IS DISTINCT FROM identity THEN
    RAISE EXCEPTION 'defending champion projection readback differs' USING ERRCODE = '55000';
  END IF;
END;
$defender601$;

NOTIFY pgrst, 'reload schema';
COMMIT;
