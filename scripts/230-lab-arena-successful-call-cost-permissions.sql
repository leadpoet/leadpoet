-- Supabase can grant named API roles EXECUTE on newly created functions.
-- Revoking PUBLIC alone does not remove those explicit default grants.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

REVOKE ALL ON FUNCTION
  public.lab_arena__successful_call_cost_state(TEXT, TEXT, TEXT),
  public.lab_arena__successful_call_eligibility(TEXT, TEXT, BIGINT),
  public.lab_arena__successful_call_publication_valid(TEXT, JSONB, BIGINT),
  public.lab_arena__successful_call_publication_eligibility_valid(TEXT, JSONB, BIGINT)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

REVOKE ALL ON FUNCTION
  public.lab_arena_submission_costs(TEXT),
  public.lab_arena_successful_call_cost_schema_v1()
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION
  public.lab_arena_submission_costs(TEXT),
  public.lab_arena_successful_call_cost_schema_v1()
  TO lab_arena_service;

-- Bind startup to the permission correction as well as the cost rules.
CREATE OR REPLACE FUNCTION public.lab_arena_successful_call_cost_schema_v1()
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog
AS $lab_arena_successful_call_cost_schema$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.successful_call_cost_schema.v1',
    'version', 230,
    'policy', 'successful_calls_v1'
  );
$lab_arena_successful_call_cost_schema$;

NOTIFY pgrst, 'reload schema';
COMMIT;
