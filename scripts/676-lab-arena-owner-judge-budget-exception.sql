-- One explicit owner-approved same-policy invocation per score assignment.
-- Keep ordinary retry limits, grading, checkpoints, cumulative costs and payers.
-- This is a finite budget exception, not a scorer or provider bug correction.
BEGIN;
SET LOCAL lock_timeout='5s';SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena676_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE')AS had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $patch676$
DECLARE target RECORD;p RECORD;d TEXT;before_doc JSONB;old_text TEXT;new_text TEXT;fragment TEXT;start_at INTEGER;end_at INTEGER;
BEGIN
 FOR target IN SELECT * FROM (VALUES
 ('public.lab_arena__grant_failed_judge_recovery_v1(text,text,text,text,text,text,boolean,text,bigint)','b4cd43774b7b59f83cd176683802a89eb09839fc2af469ceb08b90679692ccb8','bd28185d724de879d2ba86a176163e8cac106fa1cf9bc02c9599b809318d16db','plpgsql'),
 ('public.lab_arena__operator_scorer_bug_child_v1(public.lab_arena_runs)','1bd5b92b7331bf82ad5696f25a9d941b1fbe6d414416d5b1d18c4e76d7607511','7b77d2d082958a2122a6fb172b424f308c5a1bb508fe990874c82135f3000938','sql'),
 ('public.lab_arena__operator_scorer_bug_host_v1(public.lab_arena_runs)','1cd44c3ac95f7b8155eb540fbc26f83362f34985d6171704d7d6cedf4fb3d383','70ddcfeeb7238014bf33eaf6684a61f448a3dc6465f23cb38301a8d5fd9f1cb6','sql'),
 ('public.lab_arena__operator_scorer_bug_claim_v1(public.lab_arena_rounds,public.lab_arena_runs)','0fdb1b5e6b182401ba3ff95a18a3c14ab857078e682f282ee8cd131de92e315c','f329afb04a5f29afb0253d16fb63da00bc852782b22675f08bd7849c29213c3b','sql')
 )f(signature,preimage,postimage,language)LOOP
 SELECT * INTO STRICT p FROM pg_proc WHERE oid=target.signature::regprocedure;
 before_doc:=to_jsonb(p)-'prosrc';d:=p.prosrc;
 IF pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname=target.language)
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
  OR p.provolatile::TEXT<>(CASE WHEN target.signature LIKE '%_child_%' OR target.signature LIKE '%_host_%' THEN 's' ELSE 'v' END)
  OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u' THEN RAISE EXCEPTION 'judge exception676 function security differs: %',target.signature;END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')=target.postimage THEN CONTINUE;END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>target.preimage THEN RAISE EXCEPTION 'judge exception676 function preimage differs: %',target.signature;END IF;
 old_text:=$kinds676$('operator_scorer_bug_recovery_v1','operator_provider_fault_recovery_v1')$kinds676$;
 new_text:=$newkinds676$('operator_scorer_bug_recovery_v1','operator_provider_fault_recovery_v1','operator_judge_budget_exception_v1')$newkinds676$;
 IF (length(d)-length(replace(d,old_text,'')))/length(old_text)<>1 THEN RAISE EXCEPTION 'judge exception676 kind anchor differs';END IF;
 d:=replace(d,old_text,new_text);
 IF target.language='plpgsql' THEN
  -- Preserve the old recovery ancestry rules verbatim for the old two kinds.
  start_at:=strpos(d,'OR ((public.lab_arena__operator_scorer_bug_child_v1(parent)');
  end_at:=strpos(d,'OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.assignment_id=parent.assignment_id');
  IF start_at=0 OR end_at<=start_at THEN RAISE EXCEPTION 'judge exception676 parent anchors differ';END IF;
  fragment:=substr(d,start_at,end_at-start_at);
  d:=overlay(d PLACING 'OR (p_recovery_kind<>''operator_judge_budget_exception_v1'' AND ('||substr(fragment,4)||'))
  OR (p_recovery_kind=''operator_judge_budget_exception_v1'' AND (
   EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
    WHERE h.round_id=parent.round_id AND h.recovery_doc->>''recovery_kind''=''operator_judge_budget_exception_v1''
     AND h.recovery_doc#>>''{child_identity,assignment_id}''=parent.assignment_id)
   OR EXISTS(SELECT 1 FROM public.lab_arena_runs live WHERE live.assignment_id=parent.assignment_id AND live.status IN(''pending'',''leased'',''submitted''))))
  ' FROM start_at FOR end_at-start_at);
  old_text:=$runtime676$OR (p_recovery_kind='operator_provider_fault_recovery_v1' AND mapping.runtime_image_digest IS DISTINCT FROM$runtime676$;
  new_text:=$newruntime676$OR (p_recovery_kind IN ('operator_provider_fault_recovery_v1','operator_judge_budget_exception_v1') AND mapping.runtime_image_digest IS DISTINCT FROM$newruntime676$;
  IF strpos(d,old_text)=0 THEN RAISE EXCEPTION 'judge exception676 runtime anchor differs';END IF;d:=replace(d,old_text,new_text);
  old_text:=$ref676$prior.recovery_doc->>'correction_ref' IS DISTINCT FROM p_correction_ref$ref676$;
  new_text:=$newref676$(CASE WHEN p_recovery_kind='operator_judge_budget_exception_v1' THEN prior.recovery_doc->>'approval_ref' ELSE prior.recovery_doc->>'correction_ref' END) IS DISTINCT FROM p_correction_ref$newref676$;
  IF strpos(d,old_text)=0 THEN RAISE EXCEPTION 'judge exception676 approval anchor differs';END IF;d:=replace(d,old_text,new_text);
  old_text:=$payer676$(p_recovery_kind='operator_scorer_bug_recovery_v1' AND public.lab_arena__operator_scorer_bug_host_v1(parent))$payer676$;
  new_text:=$newpayer676$(p_recovery_kind IN ('operator_scorer_bug_recovery_v1','operator_judge_budget_exception_v1') AND public.lab_arena__operator_scorer_bug_host_v1(parent))$newpayer676$;
  IF strpos(d,old_text)=0 THEN RAISE EXCEPTION 'judge exception676 payer anchor differs';END IF;d:=replace(d,old_text,new_text);
  old_text:=$proof676$response:=jsonb_build_object('status',CASE WHEN p_dry_run$proof676$;
  new_text:=$newproof676$IF p_recovery_kind='operator_judge_budget_exception_v1' THEN
  proof:=(proof-'correction_ref')||jsonb_build_object('approval_ref',p_correction_ref,
   'reason','Explicit owner-approved one additional same-policy judge invocation; no automatic successor or cap reset.');
 END IF;
 response:=jsonb_build_object('status',CASE WHEN p_dry_run$newproof676$;
  IF strpos(d,old_text)=0 THEN RAISE EXCEPTION 'judge exception676 proof anchor differs';END IF;d:=replace(d,old_text,new_text);
 ELSIF target.signature LIKE '%_host_%' THEN
  -- A finite child inherits only its exact parent's proven payer lineage.
  -- The decreasing attempt and immutable parent/hash prevent cycles.
  old_text:=$host676$AND (public.lab_arena__nov06_organizer_openrouter665_v1(parent)$host676$;
  new_text:=$newhost676$AND ((h.recovery_doc->>'recovery_kind'='operator_judge_budget_exception_v1'
    AND h.request_hash='sha256:'||encode(extensions.digest(h.recovery_doc::TEXT,'sha256'),'hex')
    AND h.recovery_doc->>'parent_sha256'='sha256:'||encode(extensions.digest(to_jsonb(parent)::TEXT,'sha256'),'hex')
    AND parent.attempt+1=p_run.attempt
    AND (parent.assignment_id,parent.round_id,parent.scored_run_id,parent.scoring_generation_floor)
      IS NOT DISTINCT FROM (p_run.assignment_id,p_run.round_id,p_run.scored_run_id,p_run.scoring_generation_floor)
    AND (public.lab_arena__operator_scorer_bug_host_v1(parent) OR public.lab_arena__nov06_organizer_openrouter665_v1(parent)))
   OR (h.recovery_doc->>'recovery_kind'<>'operator_judge_budget_exception_v1' AND public.lab_arena__nov06_organizer_openrouter665_v1(parent))$newhost676$;
  IF strpos(d,old_text)=0 THEN RAISE EXCEPTION 'judge exception676 host anchor differs';END IF;d:=replace(d,old_text,new_text);
 END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>target.postimage THEN RAISE EXCEPTION 'judge exception676 postimage differs: %',target.signature;END IF;
 EXECUTE replace(pg_get_functiondef(p.oid),'$function$'||p.prosrc||'$function$',quote_literal(d));
 IF (SELECT to_jsonb(q)-'prosrc' FROM pg_proc q WHERE q.oid=p.oid)IS DISTINCT FROM before_doc
  OR (SELECT encode(extensions.digest(q.prosrc,'sha256'),'hex')FROM pg_proc q WHERE q.oid=p.oid)<>target.postimage THEN RAISE EXCEPTION 'judge exception676 metadata changed';END IF;
 END LOOP;
END;
$patch676$;

DO $wrapper676$
DECLARE p RECORD;
BEGIN
 SELECT * INTO p FROM pg_proc WHERE oid=to_regprocedure('public.lab_arena_grant_judge_budget_exception_v1(text,text,text,text,text,text,jsonb,jsonb,boolean)');
 IF FOUND AND (pg_get_userbyid(p.proowner)<>'lab_arena_owner'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.provolatile<>'v' OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname='plpgsql')
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
  OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u'
  OR encode(extensions.digest(p.prosrc,'sha256'),'hex')<>'20bdecff3195d884fa1a551792e4ab9cdc236e57ddf06c7d2027ec869f84cf4c')THEN
  RAISE EXCEPTION 'judge exception676 installed wrapper differs';END IF;
END;
$wrapper676$;

-- Approved checkpoints are an explicit CAS input, not inferred from a later run.
CREATE OR REPLACE FUNCTION public.lab_arena_grant_judge_budget_exception_v1(
 p_parent_run_id TEXT,p_parent_sha256 TEXT,p_runtime_image_digest TEXT,p_request_id TEXT,p_actor_ref TEXT,
 p_approval_ref TEXT,p_checkpoints JSONB,p_signal_checkpoints JSONB,p_dry_run BOOLEAN DEFAULT TRUE)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $grant676$
DECLARE preview JSONB;result JSONB;
BEGIN
 IF p_approval_ref IS NULL OR length(btrim(p_approval_ref)) NOT BETWEEN 1 AND 500
  OR jsonb_typeof(p_checkpoints)IS DISTINCT FROM 'object'
  OR jsonb_typeof(p_signal_checkpoints)IS DISTINCT FROM 'object'
  OR p_dry_run IS NULL THEN RAISE EXCEPTION 'judge budget exception approval invalid';END IF;
 -- Core acquires the ordinary claim/round/parent locks and pins the mapping.
 -- Its dry run has no durable writes; locks remain held through the append.
 preview:=public.lab_arena__grant_failed_judge_recovery_v1(p_parent_run_id,p_parent_sha256,p_runtime_image_digest,
  p_request_id,p_actor_ref,p_approval_ref,TRUE,'operator_judge_budget_exception_v1',NULL);
 IF preview->'checkpoints' IS DISTINCT FROM p_checkpoints
  OR preview->'signal_checkpoints' IS DISTINCT FROM p_signal_checkpoints THEN
  RAISE EXCEPTION 'judge budget exception approved checkpoints differ';END IF;
 IF p_dry_run OR preview->'idempotent'='true'::JSONB THEN RETURN preview;END IF;
 result:=public.lab_arena__grant_failed_judge_recovery_v1(p_parent_run_id,p_parent_sha256,p_runtime_image_digest,
  p_request_id,p_actor_ref,p_approval_ref,FALSE,'operator_judge_budget_exception_v1',NULL);
 IF result->'checkpoints' IS DISTINCT FROM p_checkpoints
  OR result->'signal_checkpoints' IS DISTINCT FROM p_signal_checkpoints THEN
  RAISE EXCEPTION 'judge budget exception approved checkpoints changed';END IF;
 RETURN result;
END;
$grant676$;
ALTER FUNCTION public.lab_arena_grant_judge_budget_exception_v1(TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,JSONB,JSONB,BOOLEAN) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_grant_judge_budget_exception_v1(TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,JSONB,JSONB,BOOLEAN) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DO $restore676$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.arena676_acl)THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public')IS DISTINCT FROM(SELECT nspacl FROM pg_temp.arena676_acl)THEN RAISE EXCEPTION 'judge exception676 schema ACL differs';END IF;
END;
$restore676$;
NOTIFY pgrst,'reload schema';COMMIT;
