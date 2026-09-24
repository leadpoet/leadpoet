-- Do not let one runner consume both bounded attempts after it reports a
-- proven pre-sandbox failure.  A different eligible runner can still execute
-- the unchanged frozen source, and ordinary model failures keep the existing
-- lone-runner fallback.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $zero_setup_runner_handoff_prerequisites$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_parallel_execution_schema_v1()'
     ) IS NULL
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'lab_arena_owner'
     ) THEN
    RAISE EXCEPTION 'apply current Arena claim migrations before migration 360';
  END IF;
END;
$zero_setup_runner_handoff_prerequisites$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $zero_setup_runner_handoff_patch$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$    AND (runs.previous_runner_hotkey IS NULL OR runs.previous_runner_hotkey <> p_runner_hotkey
         OR NOT EXISTS ($old$;
  v_new TEXT := $new$    AND (
      runs.previous_runner_hotkey IS NULL
      OR runs.previous_runner_hotkey <> p_runner_hotkey
      OR NOT EXISTS (
        -- lab_arena_zero_setup_runner_handoff_v1: do not use the normal
        -- lone-runner fallback when this same runner produced the immediately
        -- preceding, exact pre-sandbox failure fingerprint.
          SELECT 1
          FROM public.lab_arena_runs AS setup_run
          WHERE setup_run.assignment_id = runs.assignment_id
            AND setup_run.attempt = runs.attempt - 1
            AND setup_run.kind = 'execute'
            AND setup_run.status = 'failed'
            AND setup_run.runner_hotkey = p_runner_hotkey
            AND setup_run.output_ref IS NULL
            AND setup_run.terminal_cause IN ('model_error', 'provider_error')
            AND setup_run.result_doc ->> 'terminal_status' =
                setup_run.terminal_cause
            AND setup_run.result_doc -> 'resource_summary' =
              '{"wall_seconds":0,"cpu_seconds":0,"max_rss_bytes":0,"stdout_bytes":0,"stderr_bytes":0,"provider_call_count":0}'::JSONB
            AND (
              setup_run.terminal_cause = 'model_error'
              OR setup_run.result_doc -> 'failure_diagnostic' =
                '{"stage":"provider_call","error_class":"provider_unavailable","reason":"provider_error"}'::JSONB
            )
            AND NOT EXISTS (
              SELECT 1
              FROM public.lab_arena_ledger AS setup_ledger
              WHERE setup_ledger.run_id = setup_run.run_id
            )
      )
    )
    AND (runs.previous_runner_hotkey IS NULL OR runs.previous_runner_hotkey <> p_runner_hotkey
         OR NOT EXISTS ($new$;
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
    AND procedure.proname = 'lab_arena_claim_assignment'
    AND procedure.pronargs = 9;

  IF v_owner IS DISTINCT FROM 'lab_arena_owner'
     OR NOT COALESCE(v_security_definer, FALSE)
     OR v_volatility IS DISTINCT FROM 'v'
     OR v_config IS NULL
     OR NOT ('search_path=pg_catalog, public' = ANY(v_config)) THEN
    RAISE EXCEPTION 'lab_arena_claim_assignment_security_shape_unexpected';
  END IF;

  IF pg_catalog.strpos(
       v_definition, 'lab_arena_zero_setup_runner_handoff_v1'
     ) = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0
       OR pg_catalog.strpos(
         pg_catalog.substr(
           v_definition,
           pg_catalog.strpos(v_definition, v_old) + pg_catalog.length(v_old)
         ),
         v_old
       ) > 0 THEN
      RAISE EXCEPTION 'lab_arena_claim_assignment_handoff_shape_unexpected';
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
    AND procedure.proname = 'lab_arena_claim_assignment'
    AND procedure.pronargs = 9;

  IF v_after_owner IS DISTINCT FROM v_owner
     OR v_after_acl IS DISTINCT FROM v_acl
     OR v_after_security_definer IS DISTINCT FROM v_security_definer
     OR v_after_volatility IS DISTINCT FROM v_volatility
     OR v_after_config IS DISTINCT FROM v_config THEN
    RAISE EXCEPTION 'lab_arena_claim_assignment_permissions_changed';
  END IF;
END;
$zero_setup_runner_handoff_patch$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
