-- Locate a bounded page before opening large frozen configuration documents.
-- Existing chain/status indexes cannot serve the unfiltered competition
-- history order. The mode expression also avoids decoding every shadow and
-- archived round just to decide whether it belongs to a public live read.
-- No rows, privileges, policies, publication documents or API contracts change.
BEGIN;
SET LOCAL lock_timeout = '500ms';
SET LOCAL statement_timeout = '5s';

CREATE INDEX IF NOT EXISTS lab_arena_rounds_chain_mode_created_idx
  ON public.lab_arena_rounds (
    arena_network_name, arena_netuid, (configuration_doc ->> 'mode'), created_at DESC
  );

-- One extraction pass instead of separately detoasting the frozen catalog
-- for each public configuration key. Preserve absent keys and explicit null.
CREATE TEMP TABLE lab_arena_464_schema_acl ON COMMIT DROP AS
SELECT nspacl AS acl,
       pg_catalog.has_schema_privilege('lab_arena_owner', 'public', 'CREATE') AS had_create
FROM pg_catalog.pg_namespace WHERE nspname = 'public';
DO $temporary_create$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.lab_arena_464_schema_acl) THEN
    GRANT CREATE ON SCHEMA public TO lab_arena_owner;
  END IF;
END;
$temporary_create$;

CREATE OR REPLACE FUNCTION public.lab_arena_public_configuration_v1(
  p_round public.lab_arena_rounds
)
RETURNS JSONB
LANGUAGE sql
IMMUTABLE
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $public_configuration$
  SELECT COALESCE(pg_catalog.jsonb_object_agg(key, value), '{}'::JSONB)
  FROM pg_catalog.jsonb_each(CASE
    WHEN pg_catalog.jsonb_typeof(p_round.configuration_doc) = 'object'
      THEN p_round.configuration_doc ELSE '{}'::JSONB END)
  WHERE key = ANY(ARRAY[
    'mode', 'network_name', 'netuid', 'schedule',
    'benchmark_disclosure_policy', 'stage_1_icp_count', 'stage_2_icp_count',
    'promotion_margin', 'execution_sequence_policy', 'sourcing_cost_eligibility_policy',
    'recovery_source_round_id', 'integrity_policy', 'contact_policy',
    'company_quality_policy', 'intent_details_policy', 'baseline_hotkey'
  ]::TEXT[]);
$public_configuration$;

ALTER FUNCTION public.lab_arena_public_configuration_v1(public.lab_arena_rounds)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_public_configuration_v1(public.lab_arena_rounds)
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_public_configuration_v1(public.lab_arena_rounds)
  TO lab_arena_service;

DO $restore_create$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.lab_arena_464_schema_acl) THEN
    REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  END IF;
  IF (SELECT nspacl FROM pg_catalog.pg_namespace WHERE nspname = 'public')
       IS DISTINCT FROM (SELECT acl FROM pg_temp.lab_arena_464_schema_acl) THEN
    RAISE EXCEPTION 'lab_arena_public_configuration_schema_acl_changed';
  END IF;
END;
$restore_create$;

-- Collect expression statistics immediately instead of waiting for autovacuum.
ANALYZE public.lab_arena_rounds;
NOTIFY pgrst, 'reload schema';
COMMIT;
