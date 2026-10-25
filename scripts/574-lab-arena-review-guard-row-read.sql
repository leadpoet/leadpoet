-- Code-review admission needs only retained restart-guard presence. The
-- canonical state RPC also serializes claims and classifies captured receipts;
-- keep that RPC unchanged for restart controllers. This read holds only the
-- singleton row FOR SHARE, so it waits for an actual guard update but not for
-- a lease sweep holding the global claim-control advisory lock.
-- Retained expired guards still block review. Operator pause semantics and
-- the existing callback-to-review-reservation interval do not change.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
CREATE TEMP TABLE arena574_acl ON COMMIT DROP AS
SELECT nspacl AS acl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_review_guard_state_v1()
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $review_guard_state$
DECLARE
  v_present BOOLEAN;
BEGIN
  SELECT guard_commitment <> '' INTO STRICT v_present
  FROM public.lab_arena_restart_claim_control
  WHERE singleton FOR SHARE;
  RETURN pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.review_guard_state.v1',
    'guard_present', v_present
  );
END;
$review_guard_state$;
ALTER FUNCTION public.lab_arena_review_guard_state_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_review_guard_state_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_review_guard_state_v1()
  TO lab_arena_service;
DO $restore574$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.arena574_acl) THEN
    REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  END IF;
  IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM
      (SELECT acl FROM pg_temp.arena574_acl) THEN
    RAISE EXCEPTION 'Arena review guard schema ACL changed';
  END IF;
END;
$restore574$;
NOTIFY pgrst, 'reload schema';
COMMIT;
