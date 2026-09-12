-- Migration 218: exclude fully reused company judgments from participation.
-- Apply after 217. Existing participation timestamps remain unchanged.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_runs_participation_v1()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog
AS $participation$
DECLARE
  v_company_cache JSONB;
  v_original_work BOOLEAN := TRUE;
BEGIN
  IF TG_OP = 'INSERT' THEN
    IF NEW.participation_accepted_at IS NOT NULL THEN
      RAISE EXCEPTION 'participation timestamp is database-owned'
        USING ERRCODE = '42501';
    END IF;
    RETURN NEW;
  END IF;
  IF NEW.participation_accepted_at IS DISTINCT FROM
      OLD.participation_accepted_at THEN
    RAISE EXCEPTION 'participation timestamp is immutable'
      USING ERRCODE = '42501';
  END IF;
  IF OLD.participation_accepted_at IS NOT NULL AND (
      NEW.runner_hotkey IS DISTINCT FROM OLD.runner_hotkey
      OR NEW.round_id IS DISTINCT FROM OLD.round_id) THEN
    RAISE EXCEPTION 'participation identity is immutable'
      USING ERRCODE = '42501';
  END IF;

  -- A score with company refs uses the database-generated claim cache as its
  -- proof of original work. At least one miss means this lease produced a new
  -- judgment. An all-hit or malformed cache is reuse and earns no timestamp.
  -- Scores from before the company-judgment protocol have NULL refs and keep
  -- the original participation behavior. Execute runs are also unchanged.
  IF OLD.kind = 'score' AND OLD.company_judgment_refs IS NOT NULL THEN
    v_original_work := FALSE;
    v_company_cache := OLD.claim_response -> 'company_judgment_cache';
    IF pg_catalog.jsonb_typeof(OLD.company_judgment_refs) = 'array'
       AND pg_catalog.jsonb_typeof(v_company_cache) = 'object'
       AND v_company_cache ->> 'schema_version' =
           'leadpoet.lab_arena.company_judgment_lease.v1'
       AND pg_catalog.jsonb_typeof(v_company_cache -> 'hits') = 'array'
       AND pg_catalog.jsonb_typeof(v_company_cache -> 'misses') = 'array' THEN
      v_original_work :=
        pg_catalog.jsonb_array_length(v_company_cache -> 'misses') > 0;
    END IF;
  END IF;

  -- The completion RPC remains authoritative for the accepted transition,
  -- lease owner, generation, output, and accounting checks. This trigger only
  -- decides whether that accepted transition contains original work.
  IF v_original_work
      AND OLD.status = 'leased' AND NEW.status = 'accepted'
      AND NEW.terminal_cause = 'accepted'
      AND OLD.runner_hotkey IS NOT NULL
      AND NEW.runner_hotkey IS NOT DISTINCT FROM OLD.runner_hotkey
      AND NEW.round_id IS NOT DISTINCT FROM OLD.round_id
      AND OLD.lease_token_hash IS NOT NULL
      AND OLD.claim_request_id IS NOT NULL
      AND (NEW.result_doc ->> 'schema_version') IS DISTINCT FROM
          'leadpoet.lab_arena.cached_run_result.v1' THEN
    NEW.participation_accepted_at := pg_catalog.clock_timestamp();
  END IF;
  RETURN NEW;
END;
$participation$;
ALTER FUNCTION public.lab_arena_runs_participation_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_runs_participation_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

COMMENT ON FUNCTION public.lab_arena_runs_participation_v1() IS
  'Records immutable accepted-work time for original execute and score work; fully reused or malformed company-judgment claims do not earn credit.';

CREATE OR REPLACE FUNCTION public.lab_arena_participation_schema_v1()
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog
AS $lab_arena_participation_schema$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.participation_schema.v1',
    'version', 218
  );
$lab_arena_participation_schema$;
ALTER FUNCTION public.lab_arena_participation_schema_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_participation_schema_v1()
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_participation_schema_v1()
  TO lab_arena_service;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
