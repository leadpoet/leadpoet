-- Honour the restart guard's own lease when admitting paid code review.
--
-- The guard is leased: `lab_arena_acquire_restart_guard_v1` stamps
-- `guard_expires_at`, and the control CHECK makes that column NOT NULL for as
-- long as a commitment is held. Only an owner-driven abort or release clears
-- the commitment; nothing sweeps a lapsed one. A restart controller that dies
-- between acquiring and releasing therefore leaves a retained, expired guard
-- that withholds paid review work for good. That has happened on five days in
-- a row.
--
-- Review admission is gated in two places and both read presence only, so
-- either one alone keeps the work stopped:
--   * `lab_arena_review_guard_state_v1` (migration 574), which the service
--     reads before reserving a review; and
--   * `lab_arena_begin_submission_review` (migration 587), which returns
--     `paused` for the same reason inside the reservation itself.
-- Both are corrected here, in one transaction, because applying either half
-- alone would let the service admit cycles that the database then refuses --
-- quieter telemetry, same stopped work.
--
-- Restart semantics do not change. The canonical
-- `lab_arena_restart_guard_state_v1` keeps reporting `guard_present` and
-- `guard_active` exactly as before, restart controllers keep seeing a retained
-- expired guard, operator pause is untouched, and the review read still holds
-- only the singleton row FOR SHARE so it never queues behind a lease sweep
-- holding the global claim-control advisory lock.
--
-- The 587 patch below verifies its preimage digest and then asserts that the
-- readback is byte-identical to the text this migration constructed, which is
-- the same guarantee a hardcoded postimage digest gives. It is written that
-- way because the postimage could not be computed without a live database.
--
-- INSTALL NOTE: unlike 587, this change only relaxes a gate. An old function
-- body still in flight at apply time is the stricter behaviour, so no review
-- drain is required.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

CREATE TEMP TABLE arena595_acl ON COMMIT DROP AS
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
  v_active BOOLEAN;
BEGIN
  -- A NULL expiry cannot occur while a commitment is held (the control CHECK
  -- forbids it); treating it as never-expiring keeps an unexpected row shape
  -- failing closed rather than releasing review work.
  SELECT guard_commitment <> '',
    guard_commitment <> ''
      AND COALESCE(guard_expires_at, 'infinity'::TIMESTAMPTZ)
        > pg_catalog.clock_timestamp()
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

DO $review595_0$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)';
  v_preimage CONSTANT TEXT := '38ed2f3616f92a0c17a3ef1c8f7617f229fed949f560f1bb3bb1f91c03c6534a';
  v_oid OID;
  v_definition TEXT;
  v_updated TEXT;
  v_metadata JSONB;
  v_identity JSONB;
  v_old TEXT;
  v_new TEXT;
BEGIN
  v_oid:=pg_catalog.to_regprocedure(v_signature);
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.to_jsonb(p)-'prosrc',
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_metadata,v_identity
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_oid;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner',
    '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena review595 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.position('v_review_guard_active' IN v_definition) > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena review595 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;

  v_old:=$old595_0_0$  v_review_guard_present BOOLEAN;$old595_0_0$;
  v_new:=$new595_0_0$  v_review_guard_active BOOLEAN;$new595_0_0$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena review595 anchor 0 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);

  v_old:=$old595_0_1$  SELECT guard_commitment <> '' INTO STRICT v_review_guard_present
  FROM public.lab_arena_restart_claim_control WHERE singleton FOR SHARE;$old595_0_1$;
  v_new:=$new595_0_1$  -- The guard carries a lease. An owner that never reaches its release step
  -- leaves the commitment behind, so holding on presence alone stops review
  -- work for good. A NULL expiry cannot occur while a commitment is held, and
  -- is read as never-expiring so an unexpected row shape still fails closed.
  SELECT guard_commitment <> ''
    AND COALESCE(guard_expires_at, 'infinity'::TIMESTAMPTZ)
      > pg_catalog.clock_timestamp()
    INTO STRICT v_review_guard_active
  FROM public.lab_arena_restart_claim_control WHERE singleton FOR SHARE;$new595_0_1$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena review595 anchor 1 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);

  v_old:=$old595_0_2$  IF v_review_guard_present AND NOT ($old595_0_2$;
  v_new:=$new595_0_2$  IF v_review_guard_active AND NOT ($new595_0_2$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena review595 anchor 2 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);

  IF pg_catalog.position('v_review_guard_present' IN v_updated) > 0 THEN
    RAISE EXCEPTION 'Arena review595 left a presence-only guard behind' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.pg_get_functiondef(v_oid) IS DISTINCT FROM v_updated
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena review595 readback differs' USING ERRCODE='55000';
  END IF;
END;
$review595_0$;

DO $restore595$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.arena595_acl) THEN
    REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  END IF;
  IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM
      (SELECT acl FROM pg_temp.arena595_acl) THEN
    RAISE EXCEPTION 'Arena review guard schema ACL changed';
  END IF;
END;
$restore595$;
NOTIFY pgrst, 'reload schema';
COMMIT;
