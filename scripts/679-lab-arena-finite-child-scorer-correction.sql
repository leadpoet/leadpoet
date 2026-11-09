-- One owner-reviewed scorer correction may follow an exact finite budget child.
-- The correction reference is explicit reviewed code-defect proof, not automatic
-- entitlement from malformed output. Keep the assignment-wide prior-bug veto,
-- one budget exception, frozen work, checkpoints, deadlines and payer lineage.
-- Patch only existing grant/payer definitions; add no retry or recovery service.
BEGIN;
SET LOCAL lock_timeout='5s';SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena679_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE')AS had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $patch679$
DECLARE p RECORD; target RECORD; d TEXT; metadata JSONB; digest TEXT;patch RECORD;
BEGIN
 FOR target IN SELECT * FROM (VALUES
 ('public.lab_arena__grant_failed_judge_recovery_v1(text,text,text,text,text,text,boolean,text,bigint)','6ed2e46a2f323cf3c9ec21f8a4266f542a4cb302b097d66b27084504651f6c04','5ef271ba7abf1141209895ac036aa17702402d2e5bcc35c881334196c5e2acea','v','plpgsql','CREATE OR REPLACE FUNCTION public.lab_arena__grant_failed_judge_recovery_v1(p_parent_run_id TEXT,p_parent_sha256 TEXT,p_runtime_image_digest TEXT,p_request_id TEXT,p_actor_ref TEXT,p_correction_ref TEXT,p_dry_run BOOLEAN,p_recovery_kind TEXT,p_provider_entry_id BIGINT) RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS '),
 ('public.lab_arena__operator_scorer_bug_host_v1(public.lab_arena_runs)','70ddcfeeb7238014bf33eaf6684a61f448a3dc6465f23cb38301a8d5fd9f1cb6','67bfc7396785c998502021f287298817a1ad135fb744d4102e29e9b24877f63e','s','sql','CREATE OR REPLACE FUNCTION public.lab_arena__operator_scorer_bug_host_v1(p_run public.lab_arena_runs) RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS ')
 ) functions(signature,preimage,postimage,volatility,language,definition) LOOP
  SELECT * INTO STRICT p FROM pg_proc WHERE oid=target.signature::regprocedure;
  metadata:=to_jsonb(p)-'prosrc';d:=p.prosrc;digest:=encode(extensions.digest(d,'sha256'),'hex');
  IF pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
   OR NOT p.prosecdef OR p.provolatile::TEXT<>target.volatility OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname=target.language)
   OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
   OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u' THEN RAISE EXCEPTION 'finite child correction679 function security differs: %',target.signature; END IF;
  IF digest=target.postimage THEN CONTINUE; END IF;
  IF digest<>target.preimage THEN RAISE EXCEPTION 'finite child correction679 function preimage differs: %',target.signature; END IF;
  FOR patch IN SELECT * FROM (VALUES
 ('plpgsql',$old679$AND h.recovery_doc->>'recovery_kind'='operator_provider_fault_recovery_v1'$old679$,$new679$AND h.recovery_doc->>'recovery_kind' IN ('operator_provider_fault_recovery_v1','operator_judge_budget_exception_v1')
     AND (h.recovery_doc->>'recovery_kind'<>'operator_judge_budget_exception_v1'
      OR NOT EXISTS(SELECT 1 FROM public.lab_arena_runs live WHERE live.assignment_id=parent.assignment_id AND live.status IN('pending','leased','submitted')))$new679$),
 ('plpgsql',$old679$to_jsonb(public.lab_arena__nov06_organizer_openrouter665_v1(original))$old679$,$new679$to_jsonb(public.lab_arena__nov06_organizer_openrouter665_v1(original) OR (h.recovery_doc->>'recovery_kind'='operator_judge_budget_exception_v1' AND public.lab_arena__operator_scorer_bug_host_v1(original)))$new679$),
 ('plpgsql',$old679$AND NOT public.lab_arena__operator_scorer_bug_child_v1(original)$old679$,$new679$AND (h.recovery_doc->>'recovery_kind'='operator_judge_budget_exception_v1' OR NOT public.lab_arena__operator_scorer_bug_child_v1(original))$new679$),
 ('sql',$old679$AND provider_h.recovery_doc->>'recovery_kind'='operator_provider_fault_recovery_v1'$old679$,$new679$AND provider_h.recovery_doc->>'recovery_kind' IN ('operator_provider_fault_recovery_v1','operator_judge_budget_exception_v1')$new679$),
 ('sql',$old679$AND public.lab_arena__nov06_organizer_openrouter665_v1(original)$old679$,$new679$AND (public.lab_arena__nov06_organizer_openrouter665_v1(original) OR (provider_h.recovery_doc->>'recovery_kind'='operator_judge_budget_exception_v1' AND public.lab_arena__operator_scorer_bug_host_v1(original)))$new679$)
  )changes(kind,old_text,new_text)LOOP
   IF patch.kind<>target.language THEN CONTINUE;END IF;
   IF (length(d)-length(replace(d,patch.old_text,'')))/length(patch.old_text)<>1 THEN RAISE EXCEPTION 'finite child correction679 function anchor differs: %',target.signature;END IF;
   d:=replace(d,patch.old_text,patch.new_text);
  END LOOP;
  IF encode(extensions.digest(d,'sha256'),'hex')<>target.postimage THEN RAISE EXCEPTION 'finite child correction679 function postimage differs: %',target.signature; END IF;
  EXECUTE target.definition||quote_literal(d);
  IF (SELECT to_jsonb(q)-'prosrc' FROM pg_proc q WHERE q.oid=p.oid)IS DISTINCT FROM metadata
   OR (SELECT encode(extensions.digest(q.prosrc,'sha256'),'hex')FROM pg_proc q WHERE q.oid=p.oid)<>target.postimage THEN RAISE EXCEPTION 'finite child correction679 function readback differs: %',target.signature; END IF;
 END LOOP;
END;
$patch679$;
DO $restore679$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.arena679_acl)THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public')IS DISTINCT FROM(SELECT nspacl FROM pg_temp.arena679_acl)THEN RAISE EXCEPTION 'finite child correction679 schema ACL differs'; END IF;
END;
$restore679$;
NOTIFY pgrst,'reload schema';COMMIT;
