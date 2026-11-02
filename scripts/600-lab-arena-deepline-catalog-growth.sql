-- Allow bounded Deepline catalog growth in both active commit RPCs.
-- 16384 matches MAX_CATALOG_TOOLS in lab_arena/deepline_catalog.py.
-- Preserve all byte, policy, budget, authority and frozen-round guards.
-- Change no stored rows or historical snapshots.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $deepline_catalog_growth600$
DECLARE
  item RECORD;
  target OID;
  definition TEXT;
  updated TEXT;
  identity_before JSONB;
  identity_after JSONB;
  old_guard TEXT := $old_guard$pg_catalog.jsonb_array_length(p_deepline_catalog -> 'tools') NOT BETWEEN 1 AND 4096$old_guard$;
  new_guard TEXT := $new_guard$pg_catalog.jsonb_array_length(p_deepline_catalog -> 'tools') NOT BETWEEN 1 AND 16384$new_guard$;
BEGIN
  FOR item IN SELECT * FROM (VALUES
    ('public.lab_arena_commit_round_v3(text,jsonb,text,text,date,text,text,jsonb)',
     '0ea283fc9c04988a6b1cfcb61868095564f3a03f55822337fab32d609d19f400',
     'dae9b449382b9c755b2ca2e484abb38e5e142a9b815bba5e8f9f8c4c771fcdc6'),
    ('public.lab_arena_commit_round_v4(text,jsonb,text,text,date,text,text,jsonb,text)',
     '6ef6077441a3e13d8cdaf08462e7787e8f550f1d22d00f888a9df57841694d5f',
     '4d9ae77e29a65fb4e3c548b3e9d469f33a75e3bc7f9b8511edfbe160acd207f9')
  ) AS expected(signature, preimage, postimage) LOOP
    target := pg_catalog.to_regprocedure(item.signature);
    SELECT pg_catalog.pg_get_functiondef(p.oid),
           pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),
             p.proacl::TEXT, p.prosecdef, p.provolatile, p.proconfig, l.lanname)
      INTO definition, identity_before
      FROM pg_catalog.pg_proc p
      JOIN pg_catalog.pg_language l ON l.oid = p.prolang
      WHERE p.oid = target;
    IF definition IS NULL OR identity_before IS DISTINCT FROM
       pg_catalog.jsonb_build_array('lab_arena_owner',
         '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
         TRUE, 'v', ARRAY['search_path=pg_catalog, public']::TEXT[], 'plpgsql') THEN
      RAISE EXCEPTION 'Deepline catalog growth authority differs: %', item.signature
        USING ERRCODE = '55000';
    END IF;
    IF pg_catalog.encode(extensions.digest(definition, 'sha256'), 'hex') = item.postimage THEN
      CONTINUE;
    END IF;
    IF pg_catalog.encode(extensions.digest(definition, 'sha256'), 'hex') <> item.preimage
       OR (pg_catalog.length(definition) - pg_catalog.length(pg_catalog.replace(definition, old_guard, '')))
          <> pg_catalog.length(old_guard) THEN
      RAISE EXCEPTION 'Deepline catalog growth preimage differs: %', item.signature
        USING ERRCODE = '55000';
    END IF;
    updated := pg_catalog.replace(definition, old_guard, new_guard);
    IF pg_catalog.encode(extensions.digest(updated, 'sha256'), 'hex') <> item.postimage THEN
      RAISE EXCEPTION 'Deepline catalog growth target differs: %', item.signature
        USING ERRCODE = '55000';
    END IF;
    EXECUTE updated;
    SELECT pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),
             p.proacl::TEXT, p.prosecdef, p.provolatile, p.proconfig, l.lanname)
      INTO identity_after
      FROM pg_catalog.pg_proc p
      JOIN pg_catalog.pg_language l ON l.oid = p.prolang
      WHERE p.oid = target;
    IF identity_after IS DISTINCT FROM identity_before
       OR pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(target), 'sha256'), 'hex')
          <> item.postimage THEN
      RAISE EXCEPTION 'Deepline catalog growth readback differs: %', item.signature
        USING ERRCODE = '55000';
    END IF;
  END LOOP;
END;
$deepline_catalog_growth600$;
NOTIFY pgrst, 'reload schema';
COMMIT;
