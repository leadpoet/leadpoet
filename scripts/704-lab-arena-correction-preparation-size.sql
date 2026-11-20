-- Keep the complete frozen policy and exact state checks in correction receipts.
-- The existing 1 MiB bound cannot contain normal multi-megabyte round policies.
BEGIN;
SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena704_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $patch704$
DECLARE p RECORD;d TEXT;metadata JSONB;defaults TEXT;
 target TEXT:='public.lab_arena_prepare_corrected_execution_judgment_v1(jsonb,text,text,boolean)';
 preimage TEXT:='b7df2faf423f8752fac855b5d3e196bdbb4c80da20b9018cfbe8958a8631ea68';
 postimage TEXT:='e3ec97d24d3bed1c9dd114ce781e400c1fbfecaaa736b6a458f778b98aac9676';
 old_text TEXT:='octet_length(p_preparation::TEXT)>1048576';
 new_text TEXT:='octet_length(p_preparation::TEXT)>8388608';
BEGIN
 SELECT * INTO STRICT p FROM pg_proc WHERE oid=target::regprocedure;
 metadata:=to_jsonb(p)-'prosrc'-'proargdefaults';defaults:=pg_get_expr(p.proargdefaults,0);d:=p.prosrc;
 IF pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.provolatile<>'v' OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname='plpgsql')
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
  OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u' THEN RAISE EXCEPTION 'preparation704 security differs';END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')=postimage THEN RETURN;END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>preimage THEN RAISE EXCEPTION 'preparation704 preimage differs';END IF;
 IF (length(d)-length(replace(d,old_text,'')))/length(old_text)<>1 THEN RAISE EXCEPTION 'preparation704 anchor differs';END IF;
 d:=replace(d,old_text,new_text);
 IF encode(extensions.digest(d,'sha256'),'hex')<>postimage THEN RAISE EXCEPTION 'preparation704 postimage differs';END IF;
 EXECUTE replace(pg_get_functiondef(p.oid),'$function$'||p.prosrc||'$function$',quote_literal(d));
 IF (SELECT to_jsonb(q)-'prosrc'-'proargdefaults' FROM pg_proc q WHERE q.oid=p.oid)IS DISTINCT FROM metadata
  OR (SELECT pg_get_expr(q.proargdefaults,0) FROM pg_proc q WHERE q.oid=p.oid)IS DISTINCT FROM defaults
  OR (SELECT encode(extensions.digest(q.prosrc,'sha256'),'hex') FROM pg_proc q WHERE q.oid=p.oid)<>postimage THEN
  RAISE EXCEPTION 'preparation704 metadata changed';END IF;
END;
$patch704$;
DO $restore704$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.arena704_acl)THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;END IF;
 IF(SELECT nspacl FROM pg_namespace WHERE nspname='public')IS DISTINCT FROM(SELECT nspacl FROM pg_temp.arena704_acl)THEN
  RAISE EXCEPTION 'preparation704 schema ACL differs';END IF;
END;
$restore704$;
NOTIFY pgrst,'reload schema';COMMIT;
