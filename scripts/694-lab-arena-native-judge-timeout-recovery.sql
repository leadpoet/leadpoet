-- Match the runner's native sandbox timeout terminal and its legacy envelope.
-- Reuse only existing owner-approved budget / reviewed scorer-correction paths.
-- Preserve parent/checkpoint CAS, clear calls, runtime rules, finite grants,
-- frozen source/deadlines, payer identity, costs and all function metadata.
BEGIN;
SET LOCAL lock_timeout='5s';SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
DO $patch694$
DECLARE p RECORD;d TEXT;definition TEXT;metadata JSONB;patch RECORD;
 preimage TEXT:='1f2b05d010a26698a2aa9b7c402868582884376efc6533857178d813e8b3fc85';
 postimage TEXT:='2c064da4290e3095821fc3cbbc3f79a4c3cadc0da1b2fe496d5f0907bd09b743';
BEGIN
 SELECT * INTO STRICT p FROM pg_proc WHERE oid='public.lab_arena__grant_failed_judge_recovery_v1(text,text,text,text,text,text,boolean,text,bigint)'::regprocedure;
 metadata:=to_jsonb(p)-'prosrc';d:=p.prosrc;
 IF pg_get_userbyid(p.proowner)<>'lab_arena_owner'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.provolatile<>'v'
  OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname='plpgsql')
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
  OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u' THEN
  RAISE EXCEPTION 'native timeout recovery694 security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')=postimage THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>preimage THEN
  RAISE EXCEPTION 'native timeout recovery694 preimage differs'; END IF;
 FOR patch IN SELECT * FROM (VALUES
 ($old694$   OR (p_recovery_kind='operator_judge_budget_exception_v1'
    AND parent.result_doc#>>'{failure_diagnostic,stage}' IS NOT DISTINCT FROM 'sandbox'$old694$,
 $new694$   OR (p_recovery_kind IN ('operator_judge_budget_exception_v1','operator_scorer_bug_recovery_v1')
    AND parent.result_doc#>>'{failure_diagnostic,stage}' IS NOT DISTINCT FROM 'sandbox'$new694$),
 ($old694$   OR (p_recovery_kind='operator_judge_budget_exception_v1'
    AND parent.terminal_cause='lease_expired'$old694$,
 $new694$   OR (p_recovery_kind IN ('operator_judge_budget_exception_v1','operator_scorer_bug_recovery_v1')
    AND parent.terminal_cause='judge_timeout' AND parent.result_doc->>'terminal_status'='judge_timeout'
    AND parent.result_doc#>>'{failure_diagnostic,stage}'='sandbox'
    AND parent.result_doc#>>'{failure_diagnostic,error_class}'='judge_timeout')
   OR (p_recovery_kind='operator_judge_budget_exception_v1'
    AND parent.terminal_cause='lease_expired'$new694$)
 )changes(old_text,new_text) LOOP
  IF (length(d)-length(replace(d,patch.old_text,'')))/length(patch.old_text)<>1 THEN
   RAISE EXCEPTION 'native timeout recovery694 anchor differs'; END IF;
  d:=replace(d,patch.old_text,patch.new_text);
 END LOOP;
 IF encode(extensions.digest(d,'sha256'),'hex')<>postimage THEN
  RAISE EXCEPTION 'native timeout recovery694 postimage differs'; END IF;
 definition:=pg_get_functiondef(p.oid);
 IF (length(definition)-length(replace(definition,p.prosrc,'')))/length(p.prosrc)<>1 THEN
  RAISE EXCEPTION 'native timeout recovery694 definition differs'; END IF;
 EXECUTE replace(definition,p.prosrc,d);
 IF (SELECT to_jsonb(q)-'prosrc' FROM pg_proc q WHERE q.oid=p.oid) IS DISTINCT FROM metadata
  OR (SELECT encode(extensions.digest(q.prosrc,'sha256'),'hex') FROM pg_proc q WHERE q.oid=p.oid)<>postimage THEN
  RAISE EXCEPTION 'native timeout recovery694 readback differs'; END IF;
END;
$patch694$;
NOTIFY pgrst,'reload schema';COMMIT;
