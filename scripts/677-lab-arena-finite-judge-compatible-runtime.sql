-- A finite owner-approved invocation uses the selected compatible runtime.
-- Preserve frozen grading, exact parent/checkpoints, costs and all other kinds.
BEGIN;
SET LOCAL lock_timeout='5s';SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena677_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE')AS had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $patch677$
DECLARE p RECORD;d TEXT;before_doc JSONB;patch RECORD;
 target TEXT:='public.lab_arena__grant_failed_judge_recovery_v1(text,text,text,text,text,text,boolean,text,bigint)';
 preimage TEXT:='bd28185d724de879d2ba86a176163e8cac106fa1cf9bc02c9599b809318d16db';
 postimage TEXT:='ea1287fd5ca2a7d1fae4da2bf623dfeb3a33b80d1b0a5f9634fd681e58d7b770';
BEGIN
 SELECT * INTO STRICT p FROM pg_proc WHERE oid=target::regprocedure;
 before_doc:=to_jsonb(p)-'prosrc';d:=p.prosrc;
 IF pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.provolatile<>'v' OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname='plpgsql')
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
  OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u' THEN RAISE EXCEPTION 'finite judge677 function security differs';END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')=postimage THEN RETURN;END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>preimage THEN RAISE EXCEPTION 'finite judge677 function preimage differs';END IF;
 FOR patch IN SELECT * FROM (VALUES
 ($old677$OR parent.result_doc#>>'{failure_diagnostic,stage}' IS DISTINCT FROM 'scorer'
  OR parent.result_doc#>>'{failure_diagnostic,error_class}' IS DISTINCT FROM 'judge_error'$old677$,
 $new677$OR NOT (
   (parent.result_doc#>>'{failure_diagnostic,stage}' IS NOT DISTINCT FROM 'scorer'
    AND parent.result_doc#>>'{failure_diagnostic,error_class}' IS NOT DISTINCT FROM 'judge_error')
   OR (p_recovery_kind='operator_judge_budget_exception_v1'
    AND parent.result_doc#>>'{failure_diagnostic,stage}' IS NOT DISTINCT FROM 'sandbox'
    AND parent.result_doc#>>'{failure_diagnostic,error_class}' IS NOT DISTINCT FROM 'judge_timeout'))$new677$),
 ($old677$OR (p_recovery_kind IN ('operator_provider_fault_recovery_v1','operator_judge_budget_exception_v1') AND mapping.runtime_image_digest IS DISTINCT FROM$old677$,
 $new677$OR (p_recovery_kind='operator_provider_fault_recovery_v1' AND mapping.runtime_image_digest IS DISTINCT FROM$new677$)
 )changes(old_text,new_text)LOOP
  IF (length(d)-length(replace(d,patch.old_text,'')))/length(patch.old_text)<>1 THEN RAISE EXCEPTION 'finite judge677 anchor differs';END IF;
  d:=replace(d,patch.old_text,patch.new_text);
 END LOOP;
 IF encode(extensions.digest(d,'sha256'),'hex')<>postimage THEN RAISE EXCEPTION 'finite judge677 postimage differs';END IF;
 EXECUTE replace(pg_get_functiondef(p.oid),'$function$'||p.prosrc||'$function$',quote_literal(d));
 IF (SELECT to_jsonb(q)-'prosrc' FROM pg_proc q WHERE q.oid=p.oid)IS DISTINCT FROM before_doc
  OR (SELECT encode(extensions.digest(q.prosrc,'sha256'),'hex')FROM pg_proc q WHERE q.oid=p.oid)<>postimage THEN RAISE EXCEPTION 'finite judge677 function metadata changed';END IF;
END;
$patch677$;
DO $restore677$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.arena677_acl)THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public')IS DISTINCT FROM(SELECT nspacl FROM pg_temp.arena677_acl)THEN RAISE EXCEPTION 'finite judge677 schema ACL differs';END IF;
END;
$restore677$;
NOTIFY pgrst,'reload schema';COMMIT;
