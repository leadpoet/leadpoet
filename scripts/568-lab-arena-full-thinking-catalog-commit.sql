-- Let future full-thinking rounds commit the frozen $6 Deepline catalog.
-- Existing per-ICP rounds keep the exact $4 guard. No rows are changed.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $full_thinking_catalog_commit562$
DECLARE
  item RECORD;
  target OID;
  definition TEXT;
  updated TEXT;
  identity_before JSONB;
  identity_after JSONB;
  old_guard TEXT := $old_guard$v_round.configuration_doc ->> 'execution_icp_cap_microusd' IS DISTINCT FROM '4000000'$old_guard$;
  new_guard TEXT := $new_guard$v_round.configuration_doc ->> 'execution_icp_cap_microusd' IS DISTINCT FROM
        (CASE WHEN v_round.configuration_doc ->> 'reasoning_capture_policy'
           = 'full_generated_thinking_v1' THEN '6000000' ELSE '4000000' END)$new_guard$;
BEGIN
  FOR item IN SELECT * FROM (VALUES
    ('public.lab_arena_commit_round_v3(text,jsonb,text,text,date,text,text,jsonb)',
     '3434a1c212c79fb698146a1598be2cb86a182b8ac8edefe2a38a28f0c303f5fa',
     '0ea283fc9c04988a6b1cfcb61868095564f3a03f55822337fab32d609d19f400'),
    ('public.lab_arena_commit_round_v4(text,jsonb,text,text,date,text,text,jsonb,text)',
     '774920cd5438f1cf1977a1c8bb78c398cfc2eb49c383acd8579a4e52e57086c8',
     '6ef6077441a3e13d8cdaf08462e7787e8f550f1d22d00f888a9df57841694d5f')
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
      RAISE EXCEPTION 'full thinking catalog commit authority differs: %', item.signature
        USING ERRCODE = '55000';
    END IF;
    IF pg_catalog.encode(extensions.digest(definition, 'sha256'), 'hex') = item.postimage THEN
      CONTINUE;
    END IF;
    IF pg_catalog.encode(extensions.digest(definition, 'sha256'), 'hex') <> item.preimage
       OR (pg_catalog.length(definition) - pg_catalog.length(pg_catalog.replace(definition, old_guard, '')))
          <> pg_catalog.length(old_guard) THEN
      RAISE EXCEPTION 'full thinking catalog commit preimage differs: %', item.signature
        USING ERRCODE = '55000';
    END IF;
    updated := pg_catalog.replace(definition, old_guard, new_guard);
    IF pg_catalog.encode(extensions.digest(updated, 'sha256'), 'hex') <> item.postimage THEN
      RAISE EXCEPTION 'full thinking catalog commit target differs: %', item.signature
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
      RAISE EXCEPTION 'full thinking catalog commit readback differs: %', item.signature
        USING ERRCODE = '55000';
    END IF;
  END LOOP;
END;
$full_thinking_catalog_commit562$;
NOTIFY pgrst, 'reload schema';
COMMIT;
