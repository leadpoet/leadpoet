-- Skip ordinary expiry scans for healthy captured work during restart drains.
-- Existing expiry remains the sole authority for recovery and accounting.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $quiescence_recovery_gate$
DECLARE d TEXT; metadata JSONB;
  target OID:='public.lab_arena_restart_quiescence_v1(text,text,bigint)'::regprocedure;
  old CONSTANT TEXT:=$old$          AND runs.status='leased' AND runs.kind IN ('score','execute')
$old$;
  replacement CONSTANT TEXT:=$new$          AND runs.status='leased' AND runs.kind IN ('score','execute')
          -- A cheap signal only: ordinary expiry retains all owner, identity,
          -- generation and accounting proofs. Healthy captured work needs no
          -- recovery scan or lease row locks on every drain poll.
          AND (runs.lease_expires_at<=clock_timestamp() OR EXISTS (
            SELECT 1 FROM public.lab_arena_trajectory_events recovery
            WHERE recovery.run_id=runs.run_id
              AND recovery.event_kind='runtime.error'))
$new$;
BEGIN
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
    FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target
      AND pg_get_userbyid(p.proowner)='lab_arena_owner'
      AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner,service_role=X/lab_arena_owner}'
      AND p.prosecdef AND p.provolatile='v'
      AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
      AND l.lanname='plpgsql';
  IF encode(extensions.digest(d,'sha256'),'hex')='155bdb39497e08055e51ff57c2c87e4b1efac8888f2202fcbc75e2989fa8a501' THEN RETURN; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>'f11ba1674937d27e4a5378c2a4fe074f01611c17d361326393851d4414dfb349'
    OR (length(d)-length(replace(d,old,'')))<>length(old) THEN
    RAISE EXCEPTION 'quiescence recovery gate source differs' USING ERRCODE='55000';
  END IF;
  EXECUTE replace(d,old,replacement);
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'155bdb39497e08055e51ff57c2c87e4b1efac8888f2202fcbc75e2989fa8a501' THEN
    RAISE EXCEPTION 'quiescence recovery gate readback differs' USING ERRCODE='55000';
  END IF;
END;
$quiescence_recovery_gate$;
COMMIT;
