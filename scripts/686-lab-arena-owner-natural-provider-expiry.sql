-- Permit one explicit owner budget grant after an exact finite provider child
-- naturally expires. Retain the native claim, immutable journal and expiry proof.
-- Preserve all other parent shapes, guards, costs, checkpoints and payer lineage.
BEGIN;
SET LOCAL lock_timeout='5s';SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena686_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE')AS had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $patch686$
DECLARE p RECORD;d TEXT;metadata JSONB;digest TEXT;
 target TEXT:='public.lab_arena__grant_failed_judge_recovery_v1(text,text,text,text,text,text,boolean,text,bigint)';
 preimage TEXT:='71c6916636c04c49b333840d6a9ab76215532a051b84ff4e6f2dca00e3b930f7';
 postimage TEXT:='b8774cb231b95ebea25aef8377769187cb20df1efd1dd7a989909dd34314a4e6';
 old_text TEXT:=$old686$OR parent.terminal_cause<>'judge_error' OR parent.result_doc->>'terminal_status' IS DISTINCT FROM 'judge_error'
  OR NOT (
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
         )) IS TRUE))$old686$;
 new_text TEXT:=$new686$OR NOT (((parent.terminal_cause='judge_error' AND parent.result_doc->>'terminal_status'='judge_error'
   AND (
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
         )) IS TRUE)))
   OR (p_recovery_kind='operator_judge_budget_exception_v1'
    AND parent.terminal_cause='lease_expired' AND parent.result_doc IS NULL AND parent.output_ref IS NULL
    AND NOT parent.champion_restart_required
    AND parent.lease_generation>0 AND parent.runner_hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'
    AND parent.claim_response->'reviewer_runtime_doc'=parent.reviewer_runtime_doc
    AND parent.claim_response->'runner_authority_exclusions' @> jsonb_build_array(parent.runner_hotkey)
    AND parent.lease_token_hash ~ '^sha256:[0-9a-f]{64}$'
    AND parent.claim_request_id ~ '^[0-9a-f]{32}$'
    AND parent.claim_request_hash ~ '^sha256:[0-9a-f]{64}$'
    AND parent.claim_response @> jsonb_build_object('status','leased','request_id',parent.claim_request_id,
      'run_id',parent.run_id,'assignment_id',parent.assignment_id,'submission_id',parent.submission_id,
      'miner_hotkey',parent.miner_hotkey,'stage',parent.stage,'icp_position',parent.icp_position,
      'attempt',parent.attempt,'kind',parent.kind,'scored_run_id',parent.scored_run_id,
      'lease_generation',parent.lease_generation,'stage_generation',parent.stage_generation)
    AND parent.terminal_doc=jsonb_build_object('expired_at',parent.terminal_doc->'expired_at')
    AND jsonb_typeof(parent.terminal_doc->'expired_at')='string'
    AND jsonb_typeof(parent.claim_response->'lease_expires_at')='string'
    AND isfinite(parent.lease_expires_at)
    AND isfinite((parent.claim_response->>'lease_expires_at')::TIMESTAMPTZ)
    AND parent.lease_expires_at>=(parent.claim_response->>'lease_expires_at')::TIMESTAMPTZ
    AND (parent.terminal_doc->>'expired_at')::TIMESTAMPTZ>=parent.lease_expires_at
    AND (parent.terminal_doc->>'expired_at')::TIMESTAMPTZ<=clock_timestamp()
    AND public.lab_arena__operator_scorer_bug_child_v1(parent)
    AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
    JOIN public.lab_arena_runs original ON original.run_id=h.recovery_doc->>'parent_run_id'
    WHERE h.round_id=parent.round_id
     AND h.request_hash='sha256:'||encode(extensions.digest(h.recovery_doc::TEXT,'sha256'),'hex')
     AND h.recovery_doc->>'operation'='round_operational_update_v1'
     AND h.recovery_doc->>'recovery_kind'='operator_provider_fault_recovery_v1'
     AND h.recovery_doc->'child_identity'=jsonb_build_object('run_id',parent.run_id,'assignment_id',parent.assignment_id,'round_id',parent.round_id,'submission_id',parent.submission_id,'miner_hotkey',parent.miner_hotkey,'stage',parent.stage,'icp_position',parent.icp_position,'attempt',parent.attempt,'scored_run_id',parent.scored_run_id,'stage_generation',parent.stage_generation,'scoring_generation_floor',parent.scoring_generation_floor,'judgment_cache_key',parent.judgment_cache_key,'judgment_input_hash',parent.judgment_input_hash,'judgment_scope_doc',parent.judgment_scope_doc,'company_judgment_refs',parent.company_judgment_refs)
     AND h.recovery_doc->'organizer_openrouter_authorized'=to_jsonb(public.lab_arena__nov06_organizer_openrouter665_v1(original))
     AND h.recovery_doc->>'runtime_image_digest'=parent.reviewer_runtime_doc->>'runtime_image_digest'
     AND h.recovery_doc->>'runtime_image_reference'=parent.reviewer_runtime_doc->>'runtime_image_reference'
     AND h.recovery_doc->'scoring_generation_floor'=to_jsonb(parent.scoring_generation_floor)
     AND h.recovery_doc->>'parent_sha256'='sha256:'||encode(extensions.digest(to_jsonb(original)::TEXT,'sha256'),'hex')
     AND (original.assignment_id,original.round_id,original.scored_run_id,original.scoring_generation_floor)
      IS NOT DISTINCT FROM (parent.assignment_id,parent.round_id,parent.scored_run_id,parent.scoring_generation_floor)
     AND original.attempt+1=parent.attempt
     AND NOT public.lab_arena__operator_scorer_bug_child_v1(original)))) IS TRUE)$new686$;
BEGIN
 SELECT * INTO STRICT p FROM pg_proc WHERE oid=target::regprocedure;
 metadata:=to_jsonb(p)-'prosrc';d:=p.prosrc;digest:=encode(extensions.digest(d,'sha256'),'hex');
 IF pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.provolatile<>'v' OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname='plpgsql')
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
  OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u' THEN RAISE EXCEPTION 'provider expiry686 function security differs';END IF;
 IF NOT EXISTS(SELECT 1 FROM pg_proc proof
   WHERE proof.oid=to_regprocedure('public.lab_arena__retained_provider_transient_v1(public.lab_arena_ledger)')
     AND encode(extensions.digest(proof.prosrc,'sha256'),'hex')='e8aab74f0d9a3ddd37d01cd3e4dc8df375030ab3fa6f4e532bdd829e3326f235'
     AND pg_get_userbyid(proof.proowner)='lab_arena_owner' AND proof.prosecdef
     AND proof.provolatile='s' AND NOT proof.proisstrict AND NOT proof.proleakproof
     AND proof.proparallel='u' AND proof.prolang=(SELECT oid FROM pg_language WHERE lanname='plpgsql')
     AND proof.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner}'
     AND proof.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
   RAISE EXCEPTION 'provider expiry686 retained provider proof differs'; END IF;
 IF NOT EXISTS(SELECT 1 FROM pg_proc proof
   WHERE proof.oid=to_regprocedure('public.lab_arena__operator_scorer_bug_child_v1(public.lab_arena_runs)')
     AND encode(extensions.digest(proof.prosrc,'sha256'),'hex')='7b77d2d082958a2122a6fb172b424f308c5a1bb508fe990874c82135f3000938'
     AND pg_get_userbyid(proof.proowner)='lab_arena_owner' AND proof.prosecdef
     AND proof.provolatile='s' AND NOT proof.proisstrict AND NOT proof.proleakproof
     AND proof.proparallel='u' AND proof.prolang=(SELECT oid FROM pg_language WHERE lanname='sql')
     AND proof.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner}'
     AND proof.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
   RAISE EXCEPTION 'provider expiry686 finite child proof differs'; END IF;
 IF digest=postimage THEN RETURN;END IF;
 IF digest<>preimage THEN RAISE EXCEPTION 'provider expiry686 function preimage differs';END IF;
 IF (length(d)-length(replace(d,old_text,'')))/length(old_text)<>1 THEN RAISE EXCEPTION 'provider expiry686 anchor differs';END IF;
 d:=replace(d,old_text,new_text);
 IF encode(extensions.digest(d,'sha256'),'hex')<>postimage THEN RAISE EXCEPTION 'provider expiry686 postimage differs: %',encode(extensions.digest(d,'sha256'),'hex');END IF;
 EXECUTE 'CREATE OR REPLACE FUNCTION public.lab_arena__grant_failed_judge_recovery_v1(p_parent_run_id TEXT,p_parent_sha256 TEXT,p_runtime_image_digest TEXT,p_request_id TEXT,p_actor_ref TEXT,p_correction_ref TEXT,p_dry_run BOOLEAN,p_recovery_kind TEXT,p_provider_entry_id BIGINT) RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS '||quote_literal(d);
 IF (SELECT to_jsonb(q)-'prosrc' FROM pg_proc q WHERE q.oid=p.oid)IS DISTINCT FROM metadata
  OR (SELECT encode(extensions.digest(q.prosrc,'sha256'),'hex')FROM pg_proc q WHERE q.oid=p.oid)<>postimage THEN RAISE EXCEPTION 'provider expiry686 function readback differs';END IF;
END;
$patch686$;
DO $restore686$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.arena686_acl)THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public')IS DISTINCT FROM(SELECT nspacl FROM pg_temp.arena686_acl)THEN RAISE EXCEPTION 'provider expiry686 schema ACL differs';END IF;
END;
$restore686$;
NOTIFY pgrst,'reload schema';COMMIT;
