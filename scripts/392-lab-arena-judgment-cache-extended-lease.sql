-- Match the judgment-cache wrapper to the lease lengths already accepted by
-- reserve_call and settle_call after migrations 329 and 354. Migration 391
-- remains immutable because it has been applied in production.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $judgment_cache_extended_lease$
DECLARE
  v_definition TEXT;
  v_old CONSTANT TEXT :=
    'OR COALESCE(p_lease_ttl_seconds, 0) NOT BETWEEN 60 AND 3600 THEN';
  v_new CONSTANT TEXT := $guard$OR (COALESCE(p_lease_ttl_seconds, 0) NOT BETWEEN 60 AND 3600
         AND COALESCE(p_lease_ttl_seconds, 0) <> 4500
         AND COALESCE(p_lease_ttl_seconds, 0) <> 6300) THEN$guard$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_reserve_judgment_call(text,text,text,text,text,text,bigint,jsonb,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition,
       'COALESCE(p_lease_ttl_seconds, 0) <> 4500') > 0 THEN
    IF pg_catalog.strpos(v_definition,
         'COALESCE(p_lease_ttl_seconds, 0) <> 6300') = 0 THEN
      RAISE EXCEPTION 'judgment cache lease guard is incomplete';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, 'judgment_cache_scope') = 0
     OR pg_catalog.strpos(v_definition, 'lab_arena_reserve_call') = 0
     OR (pg_catalog.length(v_definition)
         - pg_catalog.length(pg_catalog.replace(v_definition, v_old, '')))
         / pg_catalog.length(v_old) <> 1 THEN
    RAISE EXCEPTION 'judgment cache lease guard shape differs';
  END IF;
  EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
END;
$judgment_cache_extended_lease$;

COMMIT;
