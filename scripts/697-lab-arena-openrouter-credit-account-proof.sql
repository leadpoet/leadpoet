-- A bound native OpenRouter credit refusal is an account diagnostic even when
-- the provider's charge is unknown. Retain all owner, request, billing and
-- funded-recovery checks; do not grant zero-charge proof or alter stored rows.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $credit697$
DECLARE d TEXT; metadata JSONB;
 target OID:='public.lab_arena_miner_credit_failure_v1(text)'::regprocedure;
 old_clause TEXT:=$old697$IF h.provider='deepline' AND evidence->>'reason'='out_of_credit'$old697$;
 new_clause TEXT:=$new697$IF h.provider IN ('deepline','openrouter') AND evidence->>'reason'='out_of_credit'$new697$;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target
 AND pg_get_userbyid(p.proowner)='lab_arena_owner'
 AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
 AND p.prosecdef AND p.provolatile='s'
 AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[] AND l.lanname='plpgsql';
 IF encode(extensions.digest(d,'sha256'),'hex')='9800912044b2aba92b6e66f203e348a412f4c53b3753612b43573f1989370aec' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'23fbb6e5d40450916b75110d16d9b3c9c369264014dea51e17db7f3f978786c6'
 OR (length(d)-length(replace(d,old_clause,'')))<>length(old_clause) THEN
  RAISE EXCEPTION 'OpenRouter credit evidence preimage differs' USING ERRCODE='55000';
 END IF;
 d:=replace(d,old_clause,new_clause);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'9800912044b2aba92b6e66f203e348a412f4c53b3753612b43573f1989370aec' THEN
  RAISE EXCEPTION 'OpenRouter credit evidence postimage differs' USING ERRCODE='55000';
 END IF;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
  RAISE EXCEPTION 'OpenRouter credit evidence function metadata differs' USING ERRCODE='55000';
 END IF;
END;
$credit697$;
COMMIT;
