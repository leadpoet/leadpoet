-- Extend only the existing one-time owner budget exception's parent diagnostic.
-- Reuse680's bound final retained-failure proof. Preserve all authority, ancestry,
-- checkpoints, runtime, deadlines, cumulative costs, quotas and payer lineage.
BEGIN;
SET LOCAL lock_timeout='5s';SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena685_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE')AS had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $patch685$
DECLARE p RECORD;d TEXT;metadata JSONB;digest TEXT;
 target TEXT:='public.lab_arena__grant_failed_judge_recovery_v1(text,text,text,text,text,text,boolean,text,bigint)';
 preimage TEXT:='5ef271ba7abf1141209895ac036aa17702402d2e5bcc35c881334196c5e2acea';
 postimage TEXT:='71c6916636c04c49b333840d6a9ab76215532a051b84ff4e6f2dca00e3b930f7';
 old_text TEXT:=$old685$OR NOT (
   (parent.result_doc#>>'{failure_diagnostic,stage}' IS NOT DISTINCT FROM 'scorer'
    AND parent.result_doc#>>'{failure_diagnostic,error_class}' IS NOT DISTINCT FROM 'judge_error')
   OR (p_recovery_kind='operator_judge_budget_exception_v1'
    AND parent.result_doc#>>'{failure_diagnostic,stage}' IS NOT DISTINCT FROM 'sandbox'
    AND parent.result_doc#>>'{failure_diagnostic,error_class}' IS NOT DISTINCT FROM 'judge_timeout'))$old685$;
 new_text TEXT:=$new685$OR NOT (
   (parent.result_doc#>>'{failure_diagnostic,stage}' IS NOT DISTINCT FROM 'scorer'
    AND parent.result_doc#>>'{failure_diagnostic,error_class}' IS NOT DISTINCT FROM 'judge_error')
   OR (p_recovery_kind='operator_judge_budget_exception_v1'
    AND parent.result_doc#>>'{failure_diagnostic,stage}' IS NOT DISTINCT FROM 'sandbox'
    AND parent.result_doc#>>'{failure_diagnostic,error_class}' IS NOT DISTINCT FROM 'judge_timeout')
   OR ((p_recovery_kind='operator_judge_budget_exception_v1' AND
         parent.terminal_cause='judge_error' AND parent.result_doc->>'terminal_status'='judge_error'
         AND parent.result_doc#>>'{failure_diagnostic,stage}'='provider_call'
         AND parent.result_doc#>>'{failure_diagnostic,error_class}'='provider_unavailable'
         AND parent.result_doc#>>'{failure_diagnostic,reason}'='provider_error'
         AND jsonb_typeof(parent.result_doc#>'{resource_summary,provider_unresolved_failure_count}')='number'
         AND (parent.result_doc#>>'{resource_summary,provider_unresolved_failure_count}')::NUMERIC>0
         AND EXISTS (
           SELECT 1 FROM public.lab_arena_ledger failed
           WHERE failed.run_id=parent.run_id
             AND (failed.round_id,failed.submission_id,failed.miner_hotkey,failed.stage)
               IS NOT DISTINCT FROM (parent.round_id,parent.submission_id,parent.miner_hotkey,parent.stage)
             AND ((failed.entry_kind='settlement' AND failed.terminal_response->'call_succeeded'='false'::JSONB)
               OR (failed.entry_kind='uncertain' AND failed.entry_doc#>'{call,call_succeeded}'='false'::JSONB))
             AND public.lab_arena__retained_provider_transient_v1(failed)
             -- No later attempted broker call may have healed the failure.
             -- Closed cost reconciliation of this same failed call is allowed.
             AND NOT EXISTS (SELECT 1 FROM public.lab_arena_ledger later
               WHERE later.run_id=parent.run_id AND later.entry_id>failed.entry_id
                 AND (later.call_identity IS DISTINCT FROM failed.call_identity
                   OR later.entry_kind IN ('reservation','dispatch','recovery')
                   OR later.terminal_response->'call_succeeded'='true'::JSONB
                   OR later.entry_doc#>'{call,call_succeeded}'='true'::JSONB))
         )) IS TRUE))$new685$;
BEGIN
 SELECT * INTO STRICT p FROM pg_proc WHERE oid=target::regprocedure;
 metadata:=to_jsonb(p)-'prosrc';d:=p.prosrc;digest:=encode(extensions.digest(d,'sha256'),'hex');
 IF pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.provolatile<>'v' OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname='plpgsql')
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
  OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u' THEN RAISE EXCEPTION 'provider budget685 function security differs';END IF;
 IF NOT EXISTS(SELECT 1 FROM pg_proc proof
   WHERE proof.oid=to_regprocedure('public.lab_arena__retained_provider_transient_v1(public.lab_arena_ledger)')
     AND encode(extensions.digest(proof.prosrc,'sha256'),'hex')='e8aab74f0d9a3ddd37d01cd3e4dc8df375030ab3fa6f4e532bdd829e3326f235'
     AND pg_get_userbyid(proof.proowner)='lab_arena_owner' AND proof.prosecdef
     AND proof.provolatile='s' AND NOT proof.proisstrict AND NOT proof.proleakproof
     AND proof.proparallel='u' AND proof.prolang=(SELECT oid FROM pg_language WHERE lanname='plpgsql')
     AND proof.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner}'
     AND proof.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
   RAISE EXCEPTION 'provider budget685 retained provider proof differs'; END IF;
 IF digest=postimage THEN RETURN;END IF;
 IF digest<>preimage THEN RAISE EXCEPTION 'provider budget685 function preimage differs';END IF;
 IF (length(d)-length(replace(d,old_text,'')))/length(old_text)<>1 THEN RAISE EXCEPTION 'provider budget685 anchor differs';END IF;
 d:=replace(d,old_text,new_text);
 IF encode(extensions.digest(d,'sha256'),'hex')<>postimage THEN RAISE EXCEPTION 'provider budget685 postimage differs: %',encode(extensions.digest(d,'sha256'),'hex');END IF;
 EXECUTE 'CREATE OR REPLACE FUNCTION public.lab_arena__grant_failed_judge_recovery_v1(p_parent_run_id TEXT,p_parent_sha256 TEXT,p_runtime_image_digest TEXT,p_request_id TEXT,p_actor_ref TEXT,p_correction_ref TEXT,p_dry_run BOOLEAN,p_recovery_kind TEXT,p_provider_entry_id BIGINT) RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS '||quote_literal(d);
 IF (SELECT to_jsonb(q)-'prosrc' FROM pg_proc q WHERE q.oid=p.oid)IS DISTINCT FROM metadata
  OR (SELECT encode(extensions.digest(q.prosrc,'sha256'),'hex')FROM pg_proc q WHERE q.oid=p.oid)<>postimage THEN RAISE EXCEPTION 'provider budget685 function readback differs';END IF;
END;
$patch685$;
DO $restore685$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.arena685_acl)THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public')IS DISTINCT FROM(SELECT nspacl FROM pg_temp.arena685_acl)THEN RAISE EXCEPTION 'provider budget685 schema ACL differs';END IF;
END;
$restore685$;
NOTIFY pgrst,'reload schema';COMMIT;
