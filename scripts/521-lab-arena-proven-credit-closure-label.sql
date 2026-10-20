-- Attribute only future closures already proved by native miner credit evidence.
-- Existing terminal rows and their original documents are never changed.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $proven_credit_closure_label$
DECLARE d TEXT; metadata JSONB;
  target OID:='public.lab_arena_expire_leases(text)'::regprocedure;
  old CONSTANT TEXT:=$old$terminal_doc=jsonb_build_object('closed_at',clock_timestamp(),'previous_status','pending')$old$;
  replacement CONSTANT TEXT:=$new$terminal_doc=jsonb_build_object('closed_at',clock_timestamp(),'previous_status','pending',
      'failure_reason','provider_credit_exhausted')$new$;
BEGIN
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
    FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target
      AND pg_get_userbyid(p.proowner)='lab_arena_owner'
      AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
      AND p.prosecdef AND p.provolatile='v'
      AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
      AND l.lanname='plpgsql';
  IF encode(extensions.digest(d,'sha256'),'hex')='86d4aa32b7549d87ac7a1c21de6a70751276bfcbfd10b6700ef4ce27cad58217' THEN RETURN; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>'7a37e5379434e4c111a3136feaa0ad37c739546e326110fda01011a1c5e3ef5e'
    OR (length(d)-length(replace(d,old,'')))<>length(old) THEN
    RAISE EXCEPTION 'proven credit closure label source differs' USING ERRCODE='55000';
  END IF;
  EXECUTE replace(d,old,replacement);
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'86d4aa32b7549d87ac7a1c21de6a70751276bfcbfd10b6700ef4ce27cad58217' THEN
    RAISE EXCEPTION 'proven credit closure label readback differs' USING ERRCODE='55000';
  END IF;
END;
$proven_credit_closure_label$;
COMMIT;
