-- A runtime error on one lease must not lock healthy leases during recovery.
-- Both recovery paths already require a bound runtime.error after locking the
-- candidate. Filter impossible candidates before FOR UPDATE; keep every fresh
-- authentication, stage, accounting and natural-expiry check unchanged.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $expiry_recovery_candidate_locks552$
DECLARE
  d TEXT;
  metadata JSONB;
  target OID := 'public.lab_arena_expire_leases(text)'::regprocedure;
  old CONSTANT TEXT := $old$    WHERE candidate.round_id = p_round_id AND candidate.status = 'leased'
$old$;
  replacement CONSTANT TEXT := $new$    WHERE candidate.round_id = p_round_id AND candidate.status = 'leased'
      AND EXISTS (SELECT 1 FROM public.lab_arena_trajectory_events candidate_error
        WHERE candidate_error.run_id=candidate.run_id
          AND candidate_error.event_kind='runtime.error')
$new$;
BEGIN
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
    FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target
      AND pg_get_userbyid(p.proowner)='lab_arena_owner'
      AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
      AND p.prosecdef AND p.provolatile='v'
      AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
      AND l.lanname='plpgsql';
  IF encode(extensions.digest(d,'sha256'),'hex')=
      '5025e77ac2a602941e7166d0a201ba49efb03d0da8af1a0fda1e1889056de40d' THEN RETURN; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>
      'd1ffdf531e598254d3fd4241e44ca2a1d5721580e559d1c613ad8ccfd3b3f0ad'
      OR (length(d)-length(replace(d,old,'')))<>2*length(old) THEN
    RAISE EXCEPTION 'expiry recovery candidate locks preimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE replace(d,old,replacement);
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
      OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>
        '5025e77ac2a602941e7166d0a201ba49efb03d0da8af1a0fda1e1889056de40d' THEN
    RAISE EXCEPTION 'expiry recovery candidate locks readback differs' USING ERRCODE='55000';
  END IF;
END;
$expiry_recovery_candidate_locks552$;
COMMIT;
