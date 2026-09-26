-- Keep bounded diagnostics large enough for a legal score run (up to 6,000
-- provider calls, two events each) and both 64-KiB encoded runtime streams.
-- The per-run ceilings already bound writes. Minute ceilings dropped valid
-- compact bursts without providing a stronger per-run storage bound.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $trajectory_capacity$
DECLARE
  v_definition TEXT;
  v_before TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_append_trajectory_events_v1(text,text,jsonb)'::regprocedure
  ) INTO v_definition;
  v_before := v_definition;
  IF pg_catalog.strpos(v_definition,
      'v_runtime_count >= 512 OR v_runtime_recent >= 256') > 0
     AND pg_catalog.strpos(v_definition,
      'v_provider_count >= 9488 OR v_provider_recent >= 600') > 0
     AND pg_catalog.strpos(v_definition,
      'v_runtime_count + v_provider_count >= 10000') > 0 THEN
    v_definition := pg_catalog.replace(v_definition,
      'v_runtime_count >= 512 OR v_runtime_recent >= 256',
      'v_runtime_count >= 512');
    v_definition := pg_catalog.replace(v_definition,
      'v_provider_count >= 9488 OR v_provider_recent >= 600',
      'v_provider_count >= 15872');
    v_definition := pg_catalog.replace(v_definition,
      'v_runtime_count + v_provider_count >= 10000',
      'v_runtime_count + v_provider_count >= 16384');
  ELSIF pg_catalog.strpos(v_definition, 'v_runtime_count >= 512 THEN') = 0
     OR pg_catalog.strpos(v_definition, 'v_provider_count >= 15872 THEN') = 0
     OR pg_catalog.strpos(v_definition,
      'v_runtime_count + v_provider_count >= 16384') = 0 THEN
    RAISE EXCEPTION 'unexpected Arena trajectory capacity function';
  END IF;
  IF v_definition IS DISTINCT FROM v_before THEN
    EXECUTE v_definition;
  END IF;
END;
$trajectory_capacity$;

NOTIFY pgrst, 'reload schema';
COMMIT;
