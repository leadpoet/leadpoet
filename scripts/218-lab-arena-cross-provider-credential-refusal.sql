-- Preserve a proven miner credential refusal when its uncertain charge blocks
-- a later call to another provider under the shared submission budget. The
-- locked uncertain amount and every provider-specific credential remain
-- unchanged; this only corrects the terminal-cause evidence returned to the
-- runner.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $lab_arena_218_cross_provider_credential_refusal$
DECLARE
  v_definition TEXT;
  v_provider_scoped TEXT := $old$
          WHERE ledger.submission_id = v_run.submission_id
            AND ledger.provider = p_provider
            AND ledger.funding_source = 'miner_key'
$old$;
  v_submission_scoped TEXT := $new$
          WHERE ledger.submission_id = v_run.submission_id
            AND ledger.funding_source = 'miner_key'
$new$;
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply 214-lab-arena-prior-credential-refusal.sql first';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(procedure.oid) INTO v_definition
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena_reserve_call'
    AND procedure.pronargs = 9;
  IF pg_catalog.strpos(v_definition, v_provider_scoped) = 0 THEN
    IF pg_catalog.strpos(v_definition, v_submission_scoped) > 0 THEN
      RETURN;
    END IF;
    RAISE EXCEPTION 'lab_arena_reserve_call_credential_scope_unexpected';
  END IF;
  EXECUTE pg_catalog.replace(
    v_definition, v_provider_scoped, v_submission_scoped
  );
END;
$lab_arena_218_cross_provider_credential_refusal$;

NOTIFY pgrst, 'reload schema';
COMMIT;
