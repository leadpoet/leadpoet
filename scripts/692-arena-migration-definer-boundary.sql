-- Remove default public access to the audited legacy owner functions before
-- enabling the dedicated Arena migration login. Explicit application grants,
-- function bodies, owners, table permissions and Arena results stay unchanged.
BEGIN;
SET LOCAL lock_timeout = '2s';
SET LOCAL statement_timeout = '30s';
SET LOCAL search_path = pg_catalog;

DO $migration$
DECLARE
  signature text;
  target oid;
  application_role text;
  signatures constant text[] := ARRAY[
    'public.check_consensus_with_timeout(uuid)',
    'public.cleanup_ghost_pulls()',
    'public.create_filter_options_table()',
    'public.create_queries_table()',
    'public.debug_validator_check(text)',
    'public.enforce_rate_limits()',
    'public.increment_rejection_count(text,uuid)',
    'public.is_token_valid()',
    'public.is_valid_validator(text)',
    'public.prevent_duplicate_leads()',
    'public.process_timed_out_prospects()',
    'public.pull_prospects_for_validator()',
    'public.refresh_consensus_results()',
    'public.reset_expired_prospects()',
    'public.seed_filter_options()',
    'public.setup_rls()',
    'public.trigger_cooldown_enforcement(text)'
  ];
BEGIN
  -- Preflight the full set before changing any ACL. Unknown deployments fail.
  FOREACH signature IN ARRAY signatures LOOP
    target := to_regprocedure(signature);
    IF target IS NULL OR NOT EXISTS (
      SELECT 1 FROM pg_proc
      WHERE oid = target AND prosecdef AND proowner = 'postgres'::regrole
    ) THEN
      RAISE EXCEPTION 'Legacy function identity differs: %', signature;
    END IF;
    FOREACH application_role IN ARRAY ARRAY['anon','authenticated','service_role'] LOOP
      IF NOT EXISTS (
        SELECT 1 FROM pg_proc p,
          LATERAL aclexplode(coalesce(p.proacl, acldefault('f',p.proowner))) acl
        WHERE p.oid = target AND acl.grantee = application_role::regrole
          AND acl.privilege_type = 'EXECUTE'
      ) THEN
        RAISE EXCEPTION 'Explicit application grant missing: % / %',
          signature, application_role;
      END IF;
    END LOOP;
  END LOOP;

  FOREACH signature IN ARRAY signatures LOOP
    EXECUTE format('REVOKE EXECUTE ON FUNCTION %s FROM PUBLIC',
                   to_regprocedure(signature));
    IF has_function_privilege('lab_arena_owner', to_regprocedure(signature), 'EXECUTE') THEN
      RAISE EXCEPTION 'Unexpected inherited owner-function access: %', signature;
    END IF;
  END LOOP;
END
$migration$;
COMMIT;
