-- Scale each validator's active model submissions with verified local slots.
-- Keep the frozen round ceiling, lease serialization, retry and host guards.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $model_capacity$
DECLARE
  v_claim TEXT;
  v_open TEXT;
  v_schema TEXT;
  v_owner NAME;
  v_acl ACLITEM[];
  v_security_definer BOOLEAN;
  v_volatility "char";
  v_config TEXT[];
  v_old_declaration CONSTANT TEXT := $old$  v_active INTEGER;
  v_expires TIMESTAMPTZ;$old$;
  v_new_declaration CONSTANT TEXT := $new$  v_active INTEGER;
  v_benchmark_count INTEGER;
  v_model_limit INTEGER;
  v_active_submissions TEXT[];
  v_expires TIMESTAMPTZ;$new$;
  v_old_limit CONSTANT TEXT := $old$  v_limit := LEAST(p_declared_parallelism, v_batch_size);
  SELECT COUNT(*) INTO v_active FROM public.lab_arena_runs$old$;
  v_new_limit CONSTANT TEXT := $new$  v_limit := LEAST(p_declared_parallelism, v_batch_size);
  -- lab_arena_proxy_model_capacity_v1: the round's frozen ICP bank determines
  -- how many distinct model submissions may hold live leases on this runner.
  v_benchmark_count :=
    (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
    + (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER;
  IF v_benchmark_count NOT BETWEEN 2 AND 100 THEN
    RAISE EXCEPTION 'lab_arena_model_capacity_configuration_invalid'
      USING ERRCODE = '22023';
  END IF;
  v_model_limit := GREATEST(1, v_limit / v_benchmark_count);
  SELECT ARRAY_AGG(DISTINCT active_run.submission_id)
  INTO v_active_submissions
  FROM public.lab_arena_runs AS active_run
  WHERE active_run.round_id = p_round_id
    AND active_run.runner_hotkey = p_runner_hotkey
    AND active_run.stage_generation = v_round.stage_generation
    AND active_run.status = 'leased'
    AND active_run.lease_expires_at > pg_catalog.clock_timestamp()
    AND active_run.kind = CASE
      WHEN v_round.status IN ('stage1', 'stage2') THEN 'execute'
      ELSE 'score' END;
  SELECT COUNT(*) INTO v_active FROM public.lab_arena_runs$new$;
  v_old_pending CONSTANT TEXT := $old$    AND runs.status = 'pending'
    AND runs.stage_generation = v_round.stage_generation$old$;
  v_new_pending CONSTANT TEXT := $new$    AND runs.status = 'pending'
    AND runs.stage_generation = v_round.stage_generation
    AND (
      runs.submission_id = ANY(COALESCE(v_active_submissions, ARRAY[]::TEXT[]))
      OR COALESCE(pg_catalog.cardinality(v_active_submissions), 0) < v_model_limit
    )$new$;
  v_old_bound CONSTANT TEXT := 'NOT BETWEEN 1 AND 20';
  v_new_bound CONSTANT TEXT := 'NOT BETWEEN 1 AND 251';
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid), owner.rolname, p.proacl,
         p.prosecdef, p.provolatile, p.proconfig
  INTO v_claim, v_owner, v_acl, v_security_definer, v_volatility, v_config
  FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_namespace AS n ON n.oid = p.pronamespace
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = p.proowner
  WHERE n.nspname = 'public' AND p.proname = 'lab_arena_claim_assignment'
    AND p.pronargs = 9;
  IF v_claim IS NULL
     OR v_owner IS DISTINCT FROM 'lab_arena_owner'
     OR v_acl IS DISTINCT FROM
       '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'::ACLITEM[]
     OR v_security_definer IS DISTINCT FROM TRUE
     OR v_volatility IS DISTINCT FROM 'v'
     OR v_config IS NULL
     OR NOT ('search_path=pg_catalog, public' = ANY(v_config))
     OR pg_catalog.strpos(v_claim, 'lab_arena_untransitioned_host_fault_guard_v1') = 0
     OR pg_catalog.strpos(v_claim, 'lab_arena_parallel_twenty_icp_execution') = 0 THEN
    RAISE EXCEPTION 'Arena model capacity claim security shape differs' USING ERRCODE = '55000';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_open_parallel_execution_v1(text,jsonb)'::REGPROCEDURE
  ) INTO v_open;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_parallel_execution_schema_v1()'::REGPROCEDURE
  ) INTO v_schema;
  IF pg_catalog.strpos(v_claim, 'lab_arena_proxy_model_capacity_v1') > 0 THEN
    IF pg_catalog.strpos(v_claim, v_new_bound) = 0
       OR pg_catalog.strpos(v_open, v_new_bound) = 0
       OR pg_catalog.strpos(v_schema, '''max_parallel_icps'', 251') = 0
       OR pg_catalog.strpos(v_claim, v_old_declaration) > 0
       OR pg_catalog.strpos(v_claim, v_new_pending) = 0 THEN
      RAISE EXCEPTION 'Arena model capacity migration replay differs' USING ERRCODE = '55000';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_claim, 'sha256'), 'hex')
       IS DISTINCT FROM '3693942877ca01a3dbdb12f01338bba1093c38d6ef8e46a1c6e6e8be9f1bafa5'
     OR pg_catalog.encode(extensions.digest(v_open, 'sha256'), 'hex')
       IS DISTINCT FROM 'f877aa5f69ff908191c2c8c539a016c7f5ebe2e19bc94e87e8b194bffe76cf8d'
     OR pg_catalog.encode(extensions.digest(v_schema, 'sha256'), 'hex')
       IS DISTINCT FROM '7fe3ae026c4a3df847e1c24760028fef73a0f0ef8401c03c100368ec33c39278'
     OR (pg_catalog.length(v_claim)-pg_catalog.length(pg_catalog.replace(v_claim,v_old_declaration,'')))
          <> pg_catalog.length(v_old_declaration)
     OR (pg_catalog.length(v_claim)-pg_catalog.length(pg_catalog.replace(v_claim,v_old_limit,'')))
          <> pg_catalog.length(v_old_limit)
     OR (pg_catalog.length(v_claim)-pg_catalog.length(pg_catalog.replace(v_claim,v_old_pending,'')))
          <> pg_catalog.length(v_old_pending)
     OR (pg_catalog.length(v_claim)-pg_catalog.length(pg_catalog.replace(v_claim,v_old_bound,'')))
          <> pg_catalog.length(v_old_bound)
     OR (pg_catalog.length(v_open)-pg_catalog.length(pg_catalog.replace(v_open,v_old_bound,'')))
          <> pg_catalog.length(v_old_bound)
     OR (pg_catalog.length(v_schema)-pg_catalog.length(pg_catalog.replace(v_schema,'''max_parallel_icps'', 20','')))
          <> pg_catalog.length('''max_parallel_icps'', 20') THEN
    RAISE EXCEPTION 'Arena model capacity function preimage differs' USING ERRCODE = '55000';
  END IF;
  v_claim := pg_catalog.replace(v_claim, v_old_declaration, v_new_declaration);
  v_claim := pg_catalog.replace(v_claim, v_old_limit, v_new_limit);
  v_claim := pg_catalog.replace(v_claim, v_old_pending, v_new_pending);
  EXECUTE pg_catalog.replace(v_claim, v_old_bound, v_new_bound);
  EXECUTE pg_catalog.replace(v_open, v_old_bound, v_new_bound);
  EXECUTE pg_catalog.replace(v_schema, '''max_parallel_icps'', 20', '''max_parallel_icps'', 251');
END;
$model_capacity$;

COMMIT;
