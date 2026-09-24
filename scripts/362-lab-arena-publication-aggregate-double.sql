-- Compare the published aggregate in the same binary64 domain as the service.
-- Per-ICP NUMERIC(12,6) values remain authoritative and unchanged; PostgreSQL
-- averages those exact values, then rounds the aggregate once to double.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $publication_aggregate_double$
DECLARE
  v_schema JSONB;
  v_definition TEXT;
  v_old_type TEXT := $old$  v_expected_score NUMERIC;$old$;
  v_new_type TEXT := $new$  v_expected_score DOUBLE PRECISION;$new$;
  v_old_average TEXT := $old$    THEN runs.per_icp_score ELSE 0 END)
  INTO v_score_count, v_accepted_count, v_expected_score$old$;
  v_new_average TEXT := $new$    THEN runs.per_icp_score ELSE 0 END)::DOUBLE PRECISION
  INTO v_score_count, v_accepted_count, v_expected_score$new$;
  v_old_compare TEXT := $old$       OR (p_ranking ->> 'final_score')::NUMERIC IS DISTINCT FROM v_expected_score$old$;
  v_new_compare TEXT := $new$       OR (p_ranking ->> 'final_score')::DOUBLE PRECISION IS DISTINCT FROM v_expected_score$new$;
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena__per_icp_publication_valid(text,jsonb)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_dynamic_benchmark_schema_v1()'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply migration 353 before migration 362';
  END IF;
  SELECT public.lab_arena_dynamic_benchmark_schema_v1() INTO v_schema;
  IF (v_schema ->> 'version')::INTEGER IS DISTINCT FROM 353 THEN
    RAISE EXCEPTION 'migration 353 schema differs';
  END IF;

  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena__per_icp_publication_valid(text,jsonb)'::REGPROCEDURE
  ) INTO v_definition;

  -- Reapplying the exact migration is a no-op. Any partial or unfamiliar
  -- function shape fails instead of weakening a publication guard.
  IF pg_catalog.strpos(v_definition, v_new_type) > 0
     AND pg_catalog.strpos(v_definition, v_new_average) > 0
     AND pg_catalog.strpos(v_definition, v_new_compare) > 0
     AND pg_catalog.strpos(v_definition, v_old_type) = 0
     AND pg_catalog.strpos(v_definition, v_old_average) = 0
     AND pg_catalog.strpos(v_definition, v_old_compare) = 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, v_old_type) = 0
     OR pg_catalog.strpos(v_definition, v_old_average) = 0
     OR pg_catalog.strpos(v_definition, v_old_compare) = 0
     OR pg_catalog.strpos(v_definition, v_new_type) > 0
     OR pg_catalog.strpos(v_definition, v_new_average) > 0
     OR pg_catalog.strpos(v_definition, v_new_compare) > 0 THEN
    RAISE EXCEPTION 'Arena publication aggregate function shape differs';
  END IF;

  v_definition := pg_catalog.replace(v_definition, v_old_type, v_new_type);
  v_definition := pg_catalog.replace(
    v_definition, v_old_average, v_new_average
  );
  v_definition := pg_catalog.replace(
    v_definition, v_old_compare, v_new_compare
  );
  EXECUTE v_definition;
END;
$publication_aggregate_double$;

ALTER FUNCTION public.lab_arena__per_icp_publication_valid(TEXT,JSONB)
  OWNER TO lab_arena_owner;

DO $verify_publication_aggregate_double$
DECLARE
  v_definition TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena__per_icp_publication_valid(text,jsonb)'::REGPROCEDURE
  ) INTO v_definition;
  IF pg_catalog.strpos(
       v_definition, 'v_expected_score DOUBLE PRECISION'
     ) = 0
     OR pg_catalog.strpos(
       v_definition,
       'THEN runs.per_icp_score ELSE 0 END)::DOUBLE PRECISION'
     ) = 0
     OR pg_catalog.strpos(
       v_definition,
       '(p_ranking ->> ''final_score'')::DOUBLE PRECISION IS DISTINCT FROM v_expected_score'
     ) = 0
     OR pg_catalog.strpos(v_definition, 'v_expected_score NUMERIC') > 0 THEN
    RAISE EXCEPTION 'Arena publication aggregate double patch failed';
  END IF;
END;
$verify_publication_aggregate_double$;

NOTIFY pgrst, 'reload schema';
COMMIT;
