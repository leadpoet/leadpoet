-- An abandoned restart guard must stop withholding paid review work once its
-- lease expires. The retained commitment still records that a restart was
-- begun, so keep reporting it; add the lease verdict the canonical restart
-- state RPC already reports, so review can read presence and activity from the
-- same row read migration 574 introduced. The row is still held only FOR SHARE,
-- so this read never waits on the global claim-control advisory lock. Operator
-- pause semantics, deadline-driven round progression and every restart
-- controller path are unchanged.
BEGIN;
SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='30s';
CREATE TEMP TABLE arena705_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create FROM pg_namespace WHERE nspname='public';
DO $prerequisites705$
BEGIN
  IF pg_catalog.to_regproc('public.lab_arena_review_guard_state_v1') IS NULL THEN
    RAISE EXCEPTION 'apply the Arena review guard row read before migration 705';
  END IF;
END;
$prerequisites705$;
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
  v_active BOOLEAN;
BEGIN
  SELECT guard_commitment <> '',
         guard_commitment <> ''
           AND guard_expires_at IS NOT NULL
           AND guard_expires_at > pg_catalog.clock_timestamp()
  INTO STRICT v_present, v_active
  FROM public.lab_arena_restart_claim_control
  WHERE singleton FOR SHARE;
  RETURN pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.review_guard_state.v1',
    'guard_present', v_present,
    'guard_active', v_active
  );
END;
$review_guard_state$;
ALTER FUNCTION public.lab_arena_review_guard_state_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_review_guard_state_v1()
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_review_guard_state_v1()
  TO lab_arena_service;
DO $restore705$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.arena705_acl) THEN
    REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  END IF;
  IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM
      (SELECT nspacl FROM pg_temp.arena705_acl) THEN
    RAISE EXCEPTION 'Arena review guard schema ACL changed';
  END IF;
END;
$restore705$;
NOTIFY pgrst, 'reload schema';
COMMIT;
