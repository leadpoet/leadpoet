-- Bind the interrupted-response proof to the existing call-identity indexes.
-- The full identity, payer, request and reservation/dispatch order checks remain.
-- This function-only patch changes no retry, billing, score or persisted data.
BEGIN;
SET LOCAL lock_timeout='5s';SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
DO $patch690$
DECLARE p RECORD;d TEXT;definition TEXT;metadata JSONB;
 old_text TEXT:=$old690$   WHERE r.entry_kind='reservation' AND d.entry_kind='dispatch'
    AND r.entry_id<d.entry_id AND d.entry_id<p_entry.entry_id$old690$;
 new_text TEXT:=$new690$   WHERE r.entry_kind='reservation' AND d.entry_kind='dispatch'
    AND r.call_identity=p_entry.call_identity
    AND r.entry_id<d.entry_id AND d.entry_id<p_entry.entry_id$new690$;
BEGIN
 SELECT * INTO STRICT p FROM pg_proc WHERE oid='public.lab_arena__retained_provider_transient_v1(public.lab_arena_ledger)'::regprocedure;
 metadata:=to_jsonb(p)-'prosrc';d:=p.prosrc;
 IF pg_get_userbyid(p.proowner)<>'lab_arena_owner'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname='plpgsql')
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
  OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u' THEN
  RAISE EXCEPTION 'retained proof690 security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='4d359779d99e492d325f8d2e55cd71c00e9156dee6cfa566ea1db34431e4b57f' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'b971cc230ccc60162c21325040e291663feceb6ef5a18ac19f8cbac974dbabc0'
  OR (length(d)-length(replace(d,old_text,'')))/length(old_text)<>1 THEN
  RAISE EXCEPTION 'retained proof690 preimage differs'; END IF;
 d:=replace(d,old_text,new_text);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'4d359779d99e492d325f8d2e55cd71c00e9156dee6cfa566ea1db34431e4b57f' THEN
  RAISE EXCEPTION 'retained proof690 postimage differs'; END IF;
 definition:=pg_get_functiondef(p.oid);
 IF (length(definition)-length(replace(definition,p.prosrc,'')))/length(p.prosrc)<>1 THEN
  RAISE EXCEPTION 'retained proof690 definition differs'; END IF;
 EXECUTE replace(definition,p.prosrc,d);
 IF (SELECT to_jsonb(q)-'prosrc' FROM pg_proc q WHERE q.oid=p.oid) IS DISTINCT FROM metadata
  OR (SELECT encode(extensions.digest(q.prosrc,'sha256'),'hex') FROM pg_proc q WHERE q.oid=p.oid)<>'4d359779d99e492d325f8d2e55cd71c00e9156dee6cfa566ea1db34431e4b57f' THEN
  RAISE EXCEPTION 'retained proof690 readback differs'; END IF;
END;
$patch690$;
NOTIFY pgrst,'reload schema';COMMIT;
