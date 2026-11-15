-- Recognize broker-proved interrupted replies in the existing judge recovery.
-- Keep uncertainty, cumulative charges, checkpoints, frozen rules and deadlines.
-- An owner may replace one interrupted approved exception; no automatic chain.
BEGIN;
SET LOCAL lock_timeout='5s';SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
DO $patch689_0$
DECLARE p RECORD;d TEXT;definition TEXT;metadata JSONB;
 old_text TEXT:=$old689_0$ receipt:=call_doc->'upstream_response_receipt';request_hash:=receipt->>'request_hash';$old689_0$;
 new_text TEXT:=$new689_0$ -- A broker-observed response can stop before a complete body exists. This
 -- proves an interrupted read, not a usable answer or a settled charge. Only
 -- the broker writes these fields; validators cannot call ledger RPCs directly.
 IF (p_entry.provider='openrouter' AND p_entry.operation_id='openrouter.chat'
  AND p_entry.call_identity ~ '^sha256:[0-9a-f]{64}$'
  AND call_doc->'call_succeeded'='false'::JSONB
  AND call_doc->'transport_failure'='true'::JSONB
  AND call_doc->>'reason'='missing_provider_cost'
  AND call_doc->'provider_status'='0'::JSONB
  AND call_doc->'observed_provider_status'='200'::JSONB
  AND call_doc->>'transport_error_class' IN ('ReadTimeout','ReadError','RemoteProtocolError')
  AND call_doc->>'openrouter_generation_id' ~ '^gen-[A-Za-z0-9_-]{1,240}$'
  AND call_doc->>'credential_fingerprint' ~ '^sha256:[0-9a-f]{64}$'
  AND EXISTS(SELECT 1 FROM public.lab_arena_ledger r
   JOIN public.lab_arena_ledger d ON d.call_identity=r.call_identity
   WHERE r.entry_kind='reservation' AND d.entry_kind='dispatch'
    AND r.entry_id<d.entry_id AND d.entry_id<p_entry.entry_id
    AND r.entry_doc->>'request_hash' ~ '^sha256:[0-9a-f]{64}$'
    AND (r.run_id,r.round_id,r.submission_id,r.miner_hotkey,r.stage,r.call_identity,
      r.provider,r.operation_id,r.funding_source) IS NOT DISTINCT FROM
     (p_entry.run_id,p_entry.round_id,p_entry.submission_id,p_entry.miner_hotkey,p_entry.stage,
      p_entry.call_identity,p_entry.provider,p_entry.operation_id,p_entry.funding_source)
    AND (d.run_id,d.round_id,d.submission_id,d.miner_hotkey,d.stage,
      d.provider,d.operation_id,d.funding_source) IS NOT DISTINCT FROM
     (r.run_id,r.round_id,r.submission_id,r.miner_hotkey,r.stage,
      r.provider,r.operation_id,r.funding_source))) IS TRUE THEN RETURN TRUE; END IF;
 receipt:=call_doc->'upstream_response_receipt';request_hash:=receipt->>'request_hash';$new689_0$;
BEGIN
 SELECT * INTO STRICT p FROM pg_proc WHERE oid='public.lab_arena__retained_provider_transient_v1(public.lab_arena_ledger)'::regprocedure;
 metadata:=to_jsonb(p)-'prosrc';d:=p.prosrc;
 IF pg_get_userbyid(p.proowner)<>'lab_arena_owner'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname='plpgsql')
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
  OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u' THEN
  RAISE EXCEPTION 'interrupted response689 security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='b971cc230ccc60162c21325040e291663feceb6ef5a18ac19f8cbac974dbabc0' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'e8aab74f0d9a3ddd37d01cd3e4dc8df375030ab3fa6f4e532bdd829e3326f235'
  OR (length(d)-length(replace(d,old_text,'')))/length(old_text)<>1 THEN
  RAISE EXCEPTION 'interrupted response689 preimage differs'; END IF;
 d:=replace(d,old_text,new_text);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'b971cc230ccc60162c21325040e291663feceb6ef5a18ac19f8cbac974dbabc0' THEN
  RAISE EXCEPTION 'interrupted response689 postimage differs'; END IF;
 definition:=pg_get_functiondef(p.oid);
 IF (length(definition)-length(replace(definition,p.prosrc,'')))/length(p.prosrc)<>1 THEN
  RAISE EXCEPTION 'interrupted response689 definition differs'; END IF;
 EXECUTE replace(definition,p.prosrc,d);
 IF (SELECT to_jsonb(q)-'prosrc' FROM pg_proc q WHERE q.oid=p.oid) IS DISTINCT FROM metadata
  OR (SELECT encode(extensions.digest(q.prosrc,'sha256'),'hex') FROM pg_proc q WHERE q.oid=p.oid)<>'b971cc230ccc60162c21325040e291663feceb6ef5a18ac19f8cbac974dbabc0' THEN
  RAISE EXCEPTION 'interrupted response689 readback differs'; END IF;
END;
$patch689_0$;
DO $patch689_1$
DECLARE p RECORD;d TEXT;definition TEXT;metadata JSONB;
 old_text TEXT:=$old689_1$  OR (p_recovery_kind='operator_judge_budget_exception_v1' AND (
   EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
    WHERE h.round_id=parent.round_id AND h.recovery_doc->>'recovery_kind'='operator_judge_budget_exception_v1'
     AND h.recovery_doc#>>'{child_identity,assignment_id}'=parent.assignment_id)$old689_1$;
 new_text TEXT:=$new689_1$  OR (p_recovery_kind='operator_judge_budget_exception_v1' AND (
   (EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
    WHERE h.round_id=parent.round_id AND h.recovery_doc->>'recovery_kind'='operator_judge_budget_exception_v1'
     AND h.recovery_doc#>>'{child_identity,assignment_id}'=parent.assignment_id)
    -- One explicitly approved replacement after a proved interrupted response.
    -- No automatic descendants and no third budget grant.
    AND NOT ((parent.terminal_cause='judge_error'
     AND parent.result_doc->>'terminal_status'='judge_error'
     AND parent.result_doc#>>'{failure_diagnostic,stage}'='provider_call'
     AND parent.result_doc#>>'{failure_diagnostic,error_class}'='provider_unavailable'
     AND parent.result_doc#>>'{failure_diagnostic,reason}'='provider_error'
     AND public.lab_arena__operator_scorer_bug_child_v1(parent)
     AND (SELECT count(*) FROM public.lab_arena_scoring_supersessions h
      WHERE h.round_id=parent.round_id
       AND h.recovery_doc->>'recovery_kind'='operator_judge_budget_exception_v1'
       AND h.recovery_doc#>>'{child_identity,assignment_id}'=parent.assignment_id)=1
     AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      JOIN public.lab_arena_runs original ON original.run_id=h.recovery_doc->>'parent_run_id'
      WHERE h.round_id=parent.round_id
       AND h.recovery_doc->>'recovery_kind'='operator_judge_budget_exception_v1'
       AND h.request_hash='sha256:'||encode(extensions.digest(h.recovery_doc::TEXT,'sha256'),'hex')
       AND h.recovery_doc#>>'{child_identity,run_id}'=parent.run_id
       AND h.recovery_doc->>'runtime_image_digest'=parent.reviewer_runtime_doc->>'runtime_image_digest'
       AND h.recovery_doc->>'parent_sha256'='sha256:'||encode(extensions.digest(to_jsonb(original)::TEXT,'sha256'),'hex')
       AND original.assignment_id=parent.assignment_id AND original.attempt+1=parent.attempt
       AND h.recovery_doc->>'approval_ref' IS DISTINCT FROM p_correction_ref)
     AND EXISTS(SELECT 1 FROM public.lab_arena_ledger failed
      WHERE failed.run_id=parent.run_id
       AND (failed.round_id,failed.submission_id,failed.miner_hotkey,failed.stage)
        IS NOT DISTINCT FROM (parent.round_id,parent.submission_id,parent.miner_hotkey,parent.stage)
       AND (CASE failed.entry_kind WHEN 'uncertain' THEN failed.entry_doc->'call'
            ELSE failed.terminal_response END)->'transport_failure'='true'::JSONB
       AND public.lab_arena__retained_provider_transient_v1(failed)
       AND NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger later
        WHERE later.run_id=parent.run_id AND later.entry_id>failed.entry_id
         AND (later.call_identity IS DISTINCT FROM failed.call_identity
          OR later.entry_kind IN ('reservation','dispatch','recovery')
          OR later.terminal_response->'call_succeeded'='true'::JSONB
          OR later.entry_doc#>'{call,call_succeeded}'='true'::JSONB)))
     ) IS TRUE))$new689_1$;
BEGIN
 SELECT * INTO STRICT p FROM pg_proc WHERE oid='public.lab_arena__grant_failed_judge_recovery_v1(text,text,text,text,text,text,boolean,text,bigint)'::regprocedure;
 metadata:=to_jsonb(p)-'prosrc';d:=p.prosrc;
 IF pg_get_userbyid(p.proowner)<>'lab_arena_owner'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.provolatile<>'v'
  OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname='plpgsql')
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
  OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u' THEN
  RAISE EXCEPTION 'interrupted response689 security differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='5befa705e76c54757dea017657c7a60a8cd184b7f27419153c5157671740c77e' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'b8774cb231b95ebea25aef8377769187cb20df1efd1dd7a989909dd34314a4e6'
  OR (length(d)-length(replace(d,old_text,'')))/length(old_text)<>1 THEN
  RAISE EXCEPTION 'interrupted response689 preimage differs'; END IF;
 d:=replace(d,old_text,new_text);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'5befa705e76c54757dea017657c7a60a8cd184b7f27419153c5157671740c77e' THEN
  RAISE EXCEPTION 'interrupted response689 postimage differs'; END IF;
 definition:=pg_get_functiondef(p.oid);
 IF (length(definition)-length(replace(definition,p.prosrc,'')))/length(p.prosrc)<>1 THEN
  RAISE EXCEPTION 'interrupted response689 definition differs'; END IF;
 EXECUTE replace(definition,p.prosrc,d);
 IF (SELECT to_jsonb(q)-'prosrc' FROM pg_proc q WHERE q.oid=p.oid) IS DISTINCT FROM metadata
  OR (SELECT encode(extensions.digest(q.prosrc,'sha256'),'hex') FROM pg_proc q WHERE q.oid=p.oid)<>'5befa705e76c54757dea017657c7a60a8cd184b7f27419153c5157671740c77e' THEN
  RAISE EXCEPTION 'interrupted response689 readback differs'; END IF;
END;
$patch689_1$;
NOTIFY pgrst,'reload schema';COMMIT;
