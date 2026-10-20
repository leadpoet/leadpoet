-- Reject provably absent inherited credit proof before decoding company evidence.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $credit_negative_fastpath$
DECLARE d TEXT; metadata JSONB;
  target OID:='public.lab_arena_miner_credit_failure_v1(text)'::regprocedure;
  old CONSTANT TEXT:=$old$  IF NOT FOUND OR r.kind<>'score'$old$;
  replacement CONSTANT TEXT:=$new$  -- A pending successor cannot inherit credit failure without a failed
  -- credential predecessor. Reject that negative case before decoding the
  -- round policy and company evidence; all positive proof stays below.
  IF FOUND AND (r.status='pending' OR (r.status='failed'
      AND r.terminal_cause='stage_closed'
      AND r.terminal_doc->>'previous_status'='pending'))
    AND NOT EXISTS (SELECT 1 FROM public.lab_arena_runs predecessor
      WHERE predecessor.assignment_id=r.assignment_id
        AND predecessor.attempt<r.attempt AND predecessor.status='failed'
        AND predecessor.terminal_cause='credential_error') THEN
    RETURN FALSE;
  END IF;
  IF NOT FOUND OR r.kind<>'score'$new$;
BEGIN
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
    FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target
      AND pg_get_userbyid(p.proowner)='lab_arena_owner'
      AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
      AND p.prosecdef AND p.provolatile='s'
      AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
      AND l.lanname='plpgsql';
  IF encode(extensions.digest(d,'sha256'),'hex')='7ae6b0fd1e77c2dee07f677bb50e0d014a16d1b62040dc5791f2df7b65f1d20c' THEN RETURN; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>'f81fc633f2a8d25f6633b96e1157e6c1165bdc34833bc6e72ac6734868c5e724'
    OR (length(d)-length(replace(d,old,'')))<>length(old) THEN
    RAISE EXCEPTION 'credit negative fastpath source differs' USING ERRCODE='55000';
  END IF;
  EXECUTE replace(d,old,replacement);
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'7ae6b0fd1e77c2dee07f677bb50e0d014a16d1b62040dc5791f2df7b65f1d20c' THEN
    RAISE EXCEPTION 'credit negative fastpath readback differs' USING ERRCODE='55000';
  END IF;
END;
$credit_negative_fastpath$;
COMMIT;
