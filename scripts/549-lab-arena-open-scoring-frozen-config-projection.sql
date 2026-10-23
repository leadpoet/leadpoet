-- Materialize the unchanged frozen configuration once before scoring work items.
-- No round, run, or ledger row is changed by this migration.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $patch549$
DECLARE
  target OID := 'public.lab_arena_open_scoring_v3(text,smallint,jsonb)'::regprocedure;
  definition TEXT;
  updated TEXT;
  before_metadata JSONB;
  identity JSONB;
  old_fragment TEXT := $old549$  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;$old549$;
  new_fragment TEXT := $new549$  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  -- Materialize the unchanged frozen JSON once for repeated policy/ref checks.
  v_round.configuration_doc := CASE
    WHEN pg_catalog.jsonb_typeof(v_round.configuration_doc) = 'object'
      THEN v_round.configuration_doc || '{}'::jsonb
    ELSE v_round.configuration_doc
  END;$new549$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
         pg_catalog.to_jsonb(p)-'prosrc',
         pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,
           p.prosecdef,p.provolatile,p.proconfig,l.lanname)
    INTO definition,before_metadata,identity
  FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  JOIN pg_catalog.pg_language l ON l.oid=p.prolang
  WHERE p.oid=target;
  IF identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner',
    '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'v',ARRAY['search_path=pg_catalog, public','statement_timeout=60s']::TEXT[],
    'plpgsql') THEN
    RAISE EXCEPTION 'Arena open scoring security differs' USING ERRCODE='42501';
  END IF;
  IF pg_catalog.encode(extensions.digest(definition,'sha256'),'hex') =
    '0d2c9cd4aadea7d8de4f455dfc9d3d352527940b3833a9ad9af5ca967fce18e3' THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(definition,'sha256'),'hex') <>
    '51743119194087238e0b30e1f7803b1b5699950573d3215064b40c1844c9c42b'
    OR pg_catalog.length(definition)-pg_catalog.length(
      pg_catalog.replace(definition,old_fragment,'')) <> pg_catalog.length(old_fragment) THEN
    RAISE EXCEPTION 'Arena open scoring preimage differs' USING ERRCODE='55000';
  END IF;
  updated := pg_catalog.replace(definition,old_fragment,new_fragment);
  EXECUTE updated;
  IF (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=target)
      IS DISTINCT FROM before_metadata
    OR (SELECT pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(p.oid),'sha256'),'hex')
        FROM pg_catalog.pg_proc p WHERE p.oid=target) <>
      '0d2c9cd4aadea7d8de4f455dfc9d3d352527940b3833a9ad9af5ca967fce18e3' THEN
    RAISE EXCEPTION 'Arena open scoring postimage differs' USING ERRCODE='55000';
  END IF;
END $patch549$;
COMMIT;
