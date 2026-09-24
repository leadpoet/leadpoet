-- A proven pre-sandbox infrastructure failure must not consume one of the two
-- model executions.  Only an execute attempt whose first run has the fixed
-- zero-resource provider diagnostic can receive this one extra model retry.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $setup_failure_retry_prerequisites$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_complete_attempt(text,text,jsonb,text,text)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_champion_funding_schema_v1()'
     ) IS NULL
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'lab_arena_owner'
     )
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'lab_arena_service'
     ) THEN
    RAISE EXCEPTION 'apply migrations 217 and 227 before migration 359';
  END IF;
END;
$setup_failure_retry_prerequisites$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $setup_failure_retry_patch$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$     AND v_run.attempt < LEAST(5, 2 +
       (SELECT count(*) FROM public.lab_arena_runs AS failed_run
        WHERE failed_run.assignment_id = v_run.assignment_id AND failed_run.champion_restart_required)) THEN$old$;
  v_new TEXT := $new$     AND (
       v_run.attempt < LEAST(5, 2 +
         (SELECT count(*) FROM public.lab_arena_runs AS failed_run
          WHERE failed_run.assignment_id = v_run.assignment_id AND failed_run.champion_restart_required))
       OR (
         -- lab_arena_setup_failure_model_retry_v1
         v_run.kind = 'execute'
         AND v_run.attempt = 2
         AND p_terminal_cause IN ('model_error', 'model_timeout', 'invalid_output')
         AND CASE
           WHEN pg_catalog.jsonb_typeof(
                  p_result #> '{resource_summary,wall_seconds}'
                ) = 'number'
           THEN (p_result #>> '{resource_summary,wall_seconds}')::NUMERIC > 0
           ELSE FALSE
         END
         AND EXISTS (
           SELECT 1
           FROM public.lab_arena_runs AS setup_run
           WHERE setup_run.assignment_id = v_run.assignment_id
             AND setup_run.attempt = 1
             AND setup_run.kind = 'execute'
             AND setup_run.status = 'failed'
             AND setup_run.terminal_cause = 'provider_error'
             AND setup_run.output_ref IS NULL
             AND setup_run.result_doc ->> 'terminal_status' = 'provider_error'
             AND setup_run.result_doc -> 'resource_summary' =
               '{"wall_seconds":0,"cpu_seconds":0,"max_rss_bytes":0,"stdout_bytes":0,"stderr_bytes":0,"provider_call_count":0}'::JSONB
             AND setup_run.result_doc -> 'failure_diagnostic' =
               '{"stage":"provider_call","error_class":"provider_unavailable","reason":"provider_error"}'::JSONB
             AND NOT EXISTS (
               SELECT 1 FROM public.lab_arena_ledger AS setup_ledger
               WHERE setup_ledger.run_id = setup_run.run_id
             )
         )
       )
     ) THEN$new$;
  v_owner NAME;
  v_acl ACLITEM[];
  v_security_definer BOOLEAN;
  v_volatility "char";
  v_config TEXT[];
  v_after_owner NAME;
  v_after_acl ACLITEM[];
  v_after_security_definer BOOLEAN;
  v_after_volatility "char";
  v_after_config TEXT[];
BEGIN
  SELECT pg_catalog.pg_get_functiondef(procedure.oid), owner.rolname,
         procedure.proacl, procedure.prosecdef, procedure.provolatile,
         procedure.proconfig
  INTO v_definition, v_owner, v_acl, v_security_definer, v_volatility, v_config
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = procedure.proowner
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena_complete_attempt'
    AND procedure.pronargs = 5;

  IF v_owner IS DISTINCT FROM 'lab_arena_owner'
     OR NOT COALESCE(v_security_definer, FALSE)
     OR v_volatility IS DISTINCT FROM 'v'
     OR v_config IS NULL
     OR NOT ('search_path=pg_catalog, public' = ANY(v_config)) THEN
    RAISE EXCEPTION 'lab_arena_complete_attempt_security_shape_unexpected';
  END IF;

  IF pg_catalog.strpos(
       v_definition, 'lab_arena_setup_failure_model_retry_v1'
     ) = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0
       OR pg_catalog.strpos(
         pg_catalog.substr(
           v_definition,
           pg_catalog.strpos(v_definition, v_old) + pg_catalog.length(v_old)
         ),
         v_old
       ) > 0 THEN
      RAISE EXCEPTION 'lab_arena_complete_attempt_retry_shape_unexpected';
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  END IF;

  SELECT owner.rolname, procedure.proacl, procedure.prosecdef,
         procedure.provolatile, procedure.proconfig
  INTO v_after_owner, v_after_acl, v_after_security_definer,
       v_after_volatility, v_after_config
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = procedure.proowner
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena_complete_attempt'
    AND procedure.pronargs = 5;

  IF v_after_owner IS DISTINCT FROM v_owner
     OR v_after_acl IS DISTINCT FROM v_acl
     OR v_after_security_definer IS DISTINCT FROM v_security_definer
     OR v_after_volatility IS DISTINCT FROM v_volatility
     OR v_after_config IS DISTINCT FROM v_config THEN
    RAISE EXCEPTION 'lab_arena_complete_attempt_permissions_changed';
  END IF;
END;
$setup_failure_retry_patch$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
