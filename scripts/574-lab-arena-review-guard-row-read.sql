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
NOTIFY pgrst, 'reload schema';
COMMIT;
