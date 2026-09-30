-- Reuse the per-position eligibility already proven by the publication guard.
-- The first loop still validates every reported row against authoritative cost
-- and integrity state. The final binary64 aggregate must not repeat those same
-- expensive functions for the same positions.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $publication_eligibility_reuse$
DECLARE
  v_definition TEXT;
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
  v_old_declaration TEXT := $old$  v_benchmark_count INTEGER;
BEGIN$old$;
  v_new_declaration TEXT := $new$  v_benchmark_count INTEGER;
  v_verified_eligibility BOOLEAN[];
BEGIN$new$;
  v_old_initialization TEXT := $old$  END IF;
  IF v_cost ->> 'sourcing_cost_eligibility_policy'$old$;
  v_new_initialization TEXT := $new$  END IF;
  -- lab_arena_verified_publication_eligibility_v1
  v_verified_eligibility := pg_catalog.array_fill(
    FALSE, ARRAY[v_benchmark_count]
  );
  IF v_cost ->> 'sourcing_cost_eligibility_policy'$new$;
  v_old_capture TEXT := $old$    END IF;
    v_total_spend := v_total_spend
      + (v_expected ->> 'competition_sourcing_microusd')::BIGINT;$old$;
  v_new_capture TEXT := $new$    END IF;
    v_verified_eligibility[v_position + 1] :=
      (v_expected ->> 'eligible')::BOOLEAN;
    v_total_spend := v_total_spend
      + (v_expected ->> 'competition_sourcing_microusd')::BIGINT;$new$;
  v_old_average TEXT := $old$(public.lab_arena_icp_cost_eligibility(
        p_round_id, v_submission_id, runs.icp_position,
        (public.lab_arena__integrity_submission_summary(
          p_round_id, v_submission_id, ARRAY[runs.icp_position]
        ) ->> 'qualified_company_count')::INTEGER
      ) ->> 'eligible')::BOOLEAN$old$;
  v_new_average TEXT := $new$v_verified_eligibility[runs.icp_position + 1]$new$;
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena__per_icp_publication_valid(text,jsonb)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_dynamic_benchmark_schema_v1()'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply migrations 353 and 362 before migration 367';
  END IF;
  IF (public.lab_arena_dynamic_benchmark_schema_v1() ->> 'version')::INTEGER
       IS DISTINCT FROM 353 THEN
    RAISE EXCEPTION 'migration 353 schema differs';
  END IF;

  SELECT pg_catalog.pg_get_functiondef(procedure.oid), owner.rolname,
         procedure.proacl, procedure.prosecdef, procedure.provolatile,
         procedure.proconfig
  INTO v_definition, v_owner, v_acl, v_security_definer, v_volatility, v_config
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = procedure.proowner
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena__per_icp_publication_valid'
    AND procedure.pronargs = 2;

  IF v_owner IS DISTINCT FROM 'lab_arena_owner'
     OR NOT COALESCE(v_security_definer, FALSE)
     OR v_volatility IS DISTINCT FROM 's'
     OR v_config IS NULL
     OR NOT ('search_path=pg_catalog, public' = ANY(v_config)) THEN
    RAISE EXCEPTION 'Arena publication guard security shape differs';
  END IF;

  -- Reapplying the exact migration is a no-op. A partial or unfamiliar
  -- function shape fails closed instead of weakening publication checks.
  IF pg_catalog.strpos(v_definition, v_new_declaration) > 0
     AND pg_catalog.strpos(v_definition, v_new_initialization) > 0
     AND pg_catalog.strpos(v_definition, v_new_capture) > 0
     AND pg_catalog.strpos(v_definition, v_new_average) > 0
     AND pg_catalog.strpos(v_definition, v_old_declaration) = 0
     AND pg_catalog.strpos(v_definition, v_old_initialization) = 0
     AND pg_catalog.strpos(v_definition, v_old_capture) = 0
     AND pg_catalog.strpos(v_definition, v_old_average) = 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, v_old_declaration) = 0
     OR pg_catalog.strpos(v_definition, v_old_initialization) = 0
     OR pg_catalog.strpos(v_definition, v_old_capture) = 0
     OR pg_catalog.strpos(v_definition, v_old_average) = 0
     OR pg_catalog.strpos(v_definition, v_new_declaration) > 0
     OR pg_catalog.strpos(v_definition, v_new_initialization) > 0
     OR pg_catalog.strpos(v_definition, v_new_capture) > 0
     OR pg_catalog.strpos(v_definition, v_new_average) > 0 THEN
    RAISE EXCEPTION 'Arena publication eligibility function shape differs';
  END IF;

  v_definition := pg_catalog.replace(
    v_definition, v_old_declaration, v_new_declaration
  );
  v_definition := pg_catalog.replace(
    v_definition, v_old_initialization, v_new_initialization
  );
  v_definition := pg_catalog.replace(
    v_definition, v_old_capture, v_new_capture
  );
  v_definition := pg_catalog.replace(
    v_definition, v_old_average, v_new_average
  );
  EXECUTE v_definition;

  SELECT owner.rolname, procedure.proacl, procedure.prosecdef,
         procedure.provolatile, procedure.proconfig
  INTO v_after_owner, v_after_acl, v_after_security_definer,
       v_after_volatility, v_after_config
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = procedure.proowner
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena__per_icp_publication_valid'
    AND procedure.pronargs = 2;

  IF v_after_owner IS DISTINCT FROM v_owner
     OR v_after_acl IS DISTINCT FROM v_acl
     OR v_after_security_definer IS DISTINCT FROM v_security_definer
     OR v_after_volatility IS DISTINCT FROM v_volatility
     OR v_after_config IS DISTINCT FROM v_config THEN
    RAISE EXCEPTION 'Arena publication guard permissions changed';
  END IF;
END;
$publication_eligibility_reuse$;

DO $verify_publication_eligibility_reuse$
DECLARE
  v_definition TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena__per_icp_publication_valid(text,jsonb)'::REGPROCEDURE
  ) INTO v_definition;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_verified_publication_eligibility_v1'
     ) = 0
     OR pg_catalog.strpos(
       v_definition,
       'v_verified_eligibility[runs.icp_position + 1]'
     ) = 0
     OR pg_catalog.strpos(
       v_definition,
       'THEN runs.per_icp_score ELSE 0 END)::DOUBLE PRECISION'
     ) = 0
     OR (
       pg_catalog.length(v_definition)
       - pg_catalog.length(pg_catalog.replace(
           v_definition, 'public.lab_arena_icp_cost_eligibility(', ''
         ))
     ) / pg_catalog.length('public.lab_arena_icp_cost_eligibility(') <> 1
     OR (
       pg_catalog.length(v_definition)
       - pg_catalog.length(pg_catalog.replace(
           v_definition, 'public.lab_arena__integrity_submission_summary(', ''
         ))
     ) / pg_catalog.length(
       'public.lab_arena__integrity_submission_summary('
     ) <> 1 THEN
    RAISE EXCEPTION 'Arena publication eligibility reuse patch failed';
  END IF;
END;
$verify_publication_eligibility_reuse$;

NOTIFY pgrst, 'reload schema';
COMMIT;
