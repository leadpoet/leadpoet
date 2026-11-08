-- One owner-attested scorer correction may follow an exact provider-fault child.
-- Existing claims, expiry, checkpoints, budgets and both payer paths stay bound.
-- No new retry loop, provider recursion, or correction of a scorer-bug child.
BEGIN;
SET LOCAL lock_timeout='5s';SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena675_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE')AS had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $patch675$
DECLARE p RECORD; target RECORD; d TEXT; metadata JSONB; digest TEXT; old_text TEXT; new_text TEXT;
BEGIN
 FOR target IN SELECT * FROM (VALUES
 ('public.lab_arena__grant_failed_judge_recovery_v1(text,text,text,text,text,text,boolean,text,bigint)','d223ae52b42ec0db4d6af3cf2a026c5a3d829456c9f20965b0820866067c00bb','b4cd43774b7b59f83cd176683802a89eb09839fc2af469ceb08b90679692ccb8','v','plpgsql','CREATE OR REPLACE FUNCTION public.lab_arena__grant_failed_judge_recovery_v1(p_parent_run_id TEXT,p_parent_sha256 TEXT,p_runtime_image_digest TEXT,p_request_id TEXT,p_actor_ref TEXT,p_correction_ref TEXT,p_dry_run BOOLEAN,p_recovery_kind TEXT,p_provider_entry_id BIGINT) RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS '),
 ('public.lab_arena__operator_scorer_bug_host_v1(public.lab_arena_runs)','80ba3049f11453970f60c226414db94db48af57a05edf81c625170916b28a02d','1cd44c3ac95f7b8155eb540fbc26f83362f34985d6171704d7d6cedf4fb3d383','s','sql','CREATE OR REPLACE FUNCTION public.lab_arena__operator_scorer_bug_host_v1(p_run public.lab_arena_runs) RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS ')
 ) functions(signature,preimage,postimage,volatility,language,definition) LOOP
  SELECT * INTO STRICT p FROM pg_proc WHERE oid=target.signature::regprocedure;
  metadata:=to_jsonb(p)-'prosrc';d:=p.prosrc;digest:=encode(extensions.digest(d,'sha256'),'hex');
  IF pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
   OR NOT p.prosecdef OR p.provolatile::TEXT<>target.volatility OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname=target.language)
   OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
   OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u' THEN RAISE EXCEPTION 'provider child correction675 function security differs: %',target.signature; END IF;
  IF digest=target.postimage THEN CONTINUE; END IF;
  IF digest<>target.preimage THEN RAISE EXCEPTION 'provider child correction675 function preimage differs: %',target.signature; END IF;
  IF target.language='plpgsql' THEN
   old_text:=$old675$OR public.lab_arena__operator_scorer_bug_child_v1(parent)$old675$;
   new_text:=$new675$OR ((public.lab_arena__operator_scorer_bug_child_v1(parent) OR EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions recorded
    LEFT JOIN public.lab_arena_runs recorded_parent ON recorded_parent.run_id=recorded.recovery_doc->>'parent_run_id'
    WHERE recorded.round_id=parent.round_id AND (recorded_parent.assignment_id=parent.assignment_id
     OR recorded.recovery_doc#>>'{child_identity,run_id}'=parent.run_id
     OR recorded.recovery_doc#>>'{child_identity,assignment_id}'=parent.assignment_id))) AND (
   p_recovery_kind<>'operator_scorer_bug_recovery_v1'
   OR NOT EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
    JOIN public.lab_arena_runs original ON original.run_id=h.recovery_doc->>'parent_run_id'
    WHERE h.round_id=parent.round_id
     AND h.request_hash='sha256:'||encode(extensions.digest(h.recovery_doc::TEXT,'sha256'),'hex')
     AND h.recovery_doc->>'operation'='round_operational_update_v1'
     AND h.recovery_doc->>'recovery_kind'='operator_provider_fault_recovery_v1'
     AND h.recovery_doc->'child_identity'=jsonb_build_object('run_id',parent.run_id,'assignment_id',parent.assignment_id,'round_id',parent.round_id,'submission_id',parent.submission_id,'miner_hotkey',parent.miner_hotkey,'stage',parent.stage,'icp_position',parent.icp_position,'attempt',parent.attempt,'scored_run_id',parent.scored_run_id,'stage_generation',parent.stage_generation,'scoring_generation_floor',parent.scoring_generation_floor,'judgment_cache_key',parent.judgment_cache_key,'judgment_input_hash',parent.judgment_input_hash,'judgment_scope_doc',parent.judgment_scope_doc,'company_judgment_refs',parent.company_judgment_refs)
     AND h.recovery_doc->'organizer_openrouter_authorized'=to_jsonb(public.lab_arena__nov06_organizer_openrouter665_v1(original))
     AND h.recovery_doc->>'runtime_image_digest'=parent.reviewer_runtime_doc->>'runtime_image_digest'
     AND h.recovery_doc->>'parent_sha256'='sha256:'||encode(extensions.digest(to_jsonb(original)::TEXT,'sha256'),'hex')
     AND (original.assignment_id,original.round_id,original.scored_run_id,original.scoring_generation_floor)
      IS NOT DISTINCT FROM (parent.assignment_id,parent.round_id,parent.scored_run_id,parent.scoring_generation_floor)
     AND original.attempt+1=parent.attempt
     AND NOT public.lab_arena__operator_scorer_bug_child_v1(original))
   OR EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions corrected
    WHERE corrected.round_id=parent.round_id
     AND corrected.recovery_doc->>'operation'='round_operational_update_v1'
     AND corrected.recovery_doc->>'recovery_kind'='operator_scorer_bug_recovery_v1'
     AND (corrected.recovery_doc#>>'{child_identity,assignment_id}'=parent.assignment_id OR EXISTS(SELECT 1 FROM public.lab_arena_runs prior_parent WHERE prior_parent.run_id=corrected.recovery_doc->>'parent_run_id' AND prior_parent.assignment_id=parent.assignment_id)))))$new675$;
  ELSE
   old_text:=$hostold675$AND public.lab_arena__nov06_organizer_openrouter665_v1(parent)$hostold675$;
   new_text:=$hostnew675$AND (public.lab_arena__nov06_organizer_openrouter665_v1(parent)
    OR (h.recovery_doc->>'recovery_kind'='operator_scorer_bug_recovery_v1'
     AND h.request_hash='sha256:'||encode(extensions.digest(h.recovery_doc::TEXT,'sha256'),'hex')
     AND h.recovery_doc->>'parent_sha256'='sha256:'||encode(extensions.digest(to_jsonb(parent)::TEXT,'sha256'),'hex')
     AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions provider_h
      JOIN public.lab_arena_runs original ON original.run_id=provider_h.recovery_doc->>'parent_run_id'
      WHERE provider_h.round_id=parent.round_id
       AND provider_h.request_hash='sha256:'||encode(extensions.digest(provider_h.recovery_doc::TEXT,'sha256'),'hex')
       AND provider_h.recovery_doc->>'operation'='round_operational_update_v1'
       AND provider_h.recovery_doc->>'recovery_kind'='operator_provider_fault_recovery_v1'
       AND provider_h.recovery_doc->'child_identity'=jsonb_build_object('run_id',parent.run_id,'assignment_id',parent.assignment_id,'round_id',parent.round_id,'submission_id',parent.submission_id,'miner_hotkey',parent.miner_hotkey,'stage',parent.stage,'icp_position',parent.icp_position,'attempt',parent.attempt,'scored_run_id',parent.scored_run_id,'stage_generation',parent.stage_generation,'scoring_generation_floor',parent.scoring_generation_floor,'judgment_cache_key',parent.judgment_cache_key,'judgment_input_hash',parent.judgment_input_hash,'judgment_scope_doc',parent.judgment_scope_doc,'company_judgment_refs',parent.company_judgment_refs)
       AND provider_h.recovery_doc->'organizer_openrouter_authorized'='true'::JSONB
       AND provider_h.recovery_doc->>'runtime_image_digest'=parent.reviewer_runtime_doc->>'runtime_image_digest'
       AND h.recovery_doc->>'runtime_image_digest'<>provider_h.recovery_doc->>'runtime_image_digest'
       AND provider_h.recovery_doc->>'parent_sha256'='sha256:'||encode(extensions.digest(to_jsonb(original)::TEXT,'sha256'),'hex')
       AND (original.assignment_id,original.round_id,original.scored_run_id,original.scoring_generation_floor)
      IS NOT DISTINCT FROM (parent.assignment_id,parent.round_id,parent.scored_run_id,parent.scoring_generation_floor)
     AND original.attempt+1=parent.attempt
       AND public.lab_arena__nov06_organizer_openrouter665_v1(original))))$hostnew675$;
  END IF;
  IF (length(d)-length(replace(d,old_text,'')))/length(old_text)<>1 THEN RAISE EXCEPTION 'provider child correction675 function anchor differs: %',target.signature; END IF;
  d:=replace(d,old_text,new_text);
  IF target.language='plpgsql' THEN
   IF (length(d)-length(replace(d,$orgold675$organizer:=public.lab_arena__nov06_organizer_openrouter665_v1(parent);$orgold675$,'')))/length($orgold675$organizer:=public.lab_arena__nov06_organizer_openrouter665_v1(parent);$orgold675$)<>1 THEN RAISE EXCEPTION 'provider child correction675 payer anchor differs'; END IF;
   d:=replace(d,$orgold675$organizer:=public.lab_arena__nov06_organizer_openrouter665_v1(parent);$orgold675$,$orgnew675$organizer:=public.lab_arena__nov06_organizer_openrouter665_v1(parent) OR (p_recovery_kind='operator_scorer_bug_recovery_v1' AND public.lab_arena__operator_scorer_bug_host_v1(parent));$orgnew675$);
  END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>target.postimage THEN RAISE EXCEPTION 'provider child correction675 function postimage differs: %',target.signature; END IF;
  EXECUTE target.definition||quote_literal(d);
  IF (SELECT to_jsonb(q)-'prosrc' FROM pg_proc q WHERE q.oid=p.oid)IS DISTINCT FROM metadata
   OR (SELECT encode(extensions.digest(q.prosrc,'sha256'),'hex')FROM pg_proc q WHERE q.oid=p.oid)<>target.postimage THEN RAISE EXCEPTION 'provider child correction675 function readback differs: %',target.signature; END IF;
 END LOOP;
END;
$patch675$;
DO $restore675$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.arena675_acl)THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public')IS DISTINCT FROM(SELECT nspacl FROM pg_temp.arena675_acl)THEN RAISE EXCEPTION 'provider child correction675 schema ACL differs'; END IF;
END;
$restore675$;
NOTIFY pgrst,'reload schema';COMMIT;
