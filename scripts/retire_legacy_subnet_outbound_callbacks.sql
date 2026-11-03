-- One-time AWS retirement, not a migration-history entry or data migration.
-- Exact 2026-10-29 inventory: legacy-outbound-removal-manifest.json
-- Manifest SHA256: a2a9d34a6e0893cc57409eb502f72f044630e7bd8b45848c91a20c76208164f2
-- Only the coordinator may apply these reviewed bytes to the dedicated target.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $retire_legacy_subnet_outbound_callbacks$
DECLARE
  item RECORD;
  target OID;
  actual_hash TEXT;
  actual_identity JSONB;
  present_count INTEGER;
  names CONSTANT TEXT[] := ARRAY[
    'process_consensus_async', 'process_rep_score_async',
    'cron_process_rep_scores', 'process_pending_rep_scores',
    'process_pending_rep_scores_continuous'
  ];
BEGIN
  SELECT count(*) INTO present_count
    FROM pg_catalog.pg_proc p
    JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
    WHERE n.nspname = 'public' AND p.proname = ANY(names);
  IF present_count = 0 THEN
    RETURN;
  END IF;
  IF present_count <> 5 THEN
    RAISE EXCEPTION 'legacy outbound callback set differs'
      USING ERRCODE = '55000';
  END IF;

  -- Check the entire set before any DROP. Do not print function bodies.
  FOR item IN SELECT * FROM (VALUES
    ('public.process_consensus_async(uuid)', 'p_prospect_id uuid',
     'cecf1696058702e956c6ec1b31c49235d49232fa68eea12c3771b43098729ddf'),
    ('public.process_rep_score_async(uuid)', 'p_rep_score_pending_id uuid',
     '8af2094e7ac4fe8895067f4c6dd3b219d231364a684d072aebbbf0a02bbc508c'),
    ('public.cron_process_rep_scores()', '',
     'eaaaeac7c5d4d25056c2d5fb1f9c9d3a87de1eaf4b8efc0c135bb1c0101c0ca5'),
    ('public.process_pending_rep_scores()', '',
     '1671b69bf59c070b9ddcc2887d013bb84ca5dbbbb86292565205b2ed3938cb79'),
    ('public.process_pending_rep_scores_continuous()', '',
     'c7d8f01c38355266f3a489b3fe750fab4c066a1476fb5af9c9c5b1daae31226f')
  ) AS expected(signature, arguments, sha256) LOOP
    target := pg_catalog.to_regprocedure(item.signature);
    SELECT pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(
             pg_catalog.pg_get_functiondef(p.oid), 'UTF8')), 'hex'),
           pg_catalog.jsonb_build_array(
             pg_catalog.pg_get_userbyid(p.proowner), p.proacl::TEXT,
             p.prosecdef, p.provolatile, p.proconfig, l.lanname,
             p.prokind, pg_catalog.pg_get_function_identity_arguments(p.oid),
             pg_catalog.pg_get_function_result(p.oid))
      INTO actual_hash, actual_identity
      FROM pg_catalog.pg_proc p
      JOIN pg_catalog.pg_language l ON l.oid = p.prolang
      WHERE p.oid = target;
    IF actual_hash IS DISTINCT FROM item.sha256
       OR actual_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
         'postgres',
         '{=X/postgres,postgres=X/postgres,anon=X/postgres,authenticated=X/postgres,service_role=X/postgres}',
         TRUE, 'v', NULL::TEXT[], 'plpgsql', 'f', item.arguments, 'void') THEN
      RAISE EXCEPTION 'legacy outbound callback identity differs: %', item.signature
        USING ERRCODE = '55000';
    END IF;
    -- RESTRICT alone can remove automatic dependents. Require the audited
    -- empty incoming dependency set as well, before removing anything.
    IF EXISTS (
      SELECT 1 FROM pg_catalog.pg_depend d
      WHERE d.refclassid = 'pg_catalog.pg_proc'::pg_catalog.regclass
        AND d.refobjid = target
    ) THEN
      RAISE EXCEPTION 'legacy outbound callback has a dependent: %', item.signature
        USING ERRCODE = '55000';
    END IF;
  END LOOP;

  -- PL/pgSQL calls need not have pg_depend entries. Fail closed on a new
  -- stored-routine text reference outside the exact retired set as well.
  -- Conservative name matching also covers quoted and mixed-case calls.
  IF EXISTS (
    SELECT 1 FROM pg_catalog.pg_proc p
    JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
    WHERE pg_catalog.left(n.nspname, 3) <> 'pg_'
      AND n.nspname <> 'information_schema'
      AND NOT (n.nspname = 'public' AND p.proname = ANY(names))
      AND EXISTS (
        SELECT 1 FROM pg_catalog.unnest(names) AS retired(name)
        WHERE pg_catalog.strpos(pg_catalog.lower(p.prosrc), retired.name) > 0
      )
  ) THEN
    RAISE EXCEPTION 'legacy outbound callback has a stored routine caller'
      USING ERRCODE = '55000';
  END IF;

  DROP FUNCTION public.process_consensus_async(uuid),
    public.process_rep_score_async(uuid), public.cron_process_rep_scores(),
    public.process_pending_rep_scores(),
    public.process_pending_rep_scores_continuous() RESTRICT;

  IF EXISTS (
    SELECT 1 FROM pg_catalog.pg_proc p
    JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
    WHERE n.nspname = 'public' AND p.proname = ANY(names)
  ) THEN
    RAISE EXCEPTION 'legacy outbound callback retirement readback differs'
      USING ERRCODE = '55000';
  END IF;
END;
$retire_legacy_subnet_outbound_callbacks$;

NOTIFY pgrst, 'reload schema';
COMMIT;
