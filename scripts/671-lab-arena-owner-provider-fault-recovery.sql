-- One explicit, owner-attested successor after a retained shared-pool 429.
-- Reuse666 child identity, claim, completion, expiry, journals and accounting.
-- No global retry-cap, frozen policy, image, payer or accepted result changes.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena671_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $extend671$
DECLARE d TEXT; before_source TEXT; patch RECORD; target RECORD; oid_value OID; metadata JSONB;
BEGIN
 -- A migration replay must preserve the installed shared body and wrappers.
 IF to_regprocedure('public.lab_arena__grant_failed_judge_recovery_v1(text,text,text,text,text,text,boolean,text,bigint)') IS NOT NULL THEN
  FOR target IN SELECT * FROM (VALUES
 ('public.lab_arena__grant_failed_judge_recovery_v1(text,text,text,text,text,text,boolean,text,bigint)','6bfec20fb4155365534ccbe7369c5fb41657115a500896d685d080f70fd84429'),
 ('public.lab_arena__operator_scorer_bug_child_v1(public.lab_arena_runs)','1bd5b92b7331bf82ad5696f25a9d941b1fbe6d414416d5b1d18c4e76d7607511'),
 ('public.lab_arena__operator_scorer_bug_host_v1(public.lab_arena_runs)','80ba3049f11453970f60c226414db94db48af57a05edf81c625170916b28a02d'),
 ('public.lab_arena__operator_scorer_bug_claim_v1(public.lab_arena_rounds,public.lab_arena_runs)','0fdb1b5e6b182401ba3ff95a18a3c14ab857078e682f282ee8cd131de92e315c')
,
 ('public.lab_arena_grant_scorer_bug_recovery_v1(text,text,text,text,text,text,boolean)','2104461a8649b823f9d008f93fd71484e7be521c1b57eba6968320c94936827a'),
 ('public.lab_arena_grant_provider_fault_recovery_v1(text,text,text,text,text,text,bigint,boolean)','5ae2e0386fd76d8dfc98b13c2cdb358d28f6b8086140f3850b96cd9f4226d7d1')
  ) installed(signature,source_sha256) LOOP
   SELECT p.prosrc,to_jsonb(p)-'prosrc' INTO d,metadata FROM pg_proc p WHERE p.oid=to_regprocedure(target.signature);
   IF d IS NULL OR encode(extensions.digest(d,'sha256'),'hex')<>target.source_sha256
    OR metadata->>'proowner' IS DISTINCT FROM ('lab_arena_owner'::regrole::OID)::TEXT
    OR metadata->'prosecdef' IS DISTINCT FROM 'true'::JSONB
    OR metadata->'proconfig' IS DISTINCT FROM '["search_path=pg_catalog, public"]'::JSONB
    OR metadata->'proacl' IS DISTINCT FROM '["lab_arena_owner=X/lab_arena_owner"]'::JSONB THEN
    RAISE EXCEPTION 'provider recovery671 installed postimage differs'; END IF;
  END LOOP;
  RETURN;
 END IF;
 SELECT prosrc INTO d FROM pg_proc WHERE oid='public.lab_arena_grant_scorer_bug_recovery_v1(text,text,text,text,text,text,boolean)'::regprocedure;
 IF NOT EXISTS(SELECT 1 FROM pg_proc p WHERE p.oid='public.lab_arena_grant_scorer_bug_recovery_v1(text,text,text,text,text,text,boolean)'::regprocedure AND pg_get_userbyid(p.proowner)='lab_arena_owner' AND p.prosecdef AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner}' AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[])
  OR encode(extensions.digest(d,'sha256'),'hex')<>'967e7add12fe07a4222d677903a622fb6d0e67a282775faa655749b4253028c6' THEN
  RAISE EXCEPTION 'provider recovery671 scorer grant preimage differs'; END IF;
 FOR patch IN SELECT * FROM (VALUES
 ($old671$IF p_parent_run_id IS NULL$old671$,$new671$IF p_recovery_kind IS NULL OR p_recovery_kind NOT IN ('operator_scorer_bug_recovery_v1','operator_provider_fault_recovery_v1')
  OR (p_recovery_kind='operator_scorer_bug_recovery_v1' AND p_provider_entry_id IS NOT NULL)
  OR (p_recovery_kind='operator_provider_fault_recovery_v1' AND p_provider_entry_id IS NULL)
  OR p_parent_run_id IS NULL$new671$),
 ($old671$prior.recovery_doc->>'recovery_kind'<>'operator_scorer_bug_recovery_v1'$old671$,$new671$prior.recovery_doc->>'recovery_kind' IS DISTINCT FROM p_recovery_kind
   OR (p_recovery_kind='operator_provider_fault_recovery_v1' AND prior.recovery_doc->'provider_entry_id' IS DISTINCT FROM to_jsonb(p_provider_entry_id))$new671$),
 ($old671$OR mapping.runtime_image_digest=coalesce(parent.reviewer_runtime_doc->>'runtime_image_digest',
       r.configuration_doc->>'scorer_image_digest')$old671$,$new671$OR (p_recovery_kind='operator_scorer_bug_recovery_v1' AND mapping.runtime_image_digest=coalesce(parent.reviewer_runtime_doc->>'runtime_image_digest',
       r.configuration_doc->>'scorer_image_digest'))
  OR (p_recovery_kind='operator_provider_fault_recovery_v1' AND mapping.runtime_image_digest IS DISTINCT FROM coalesce(parent.reviewer_runtime_doc->>'runtime_image_digest',r.configuration_doc->>'scorer_image_digest'))$new671$),
 ($old671$child:=parent; child.run_id$old671$,$new671$IF p_recovery_kind='operator_provider_fault_recovery_v1' THEN
  IF parent.result_doc#>>'{failure_diagnostic,reason}' IS DISTINCT FROM 'provider_error'
   OR EXISTS(SELECT 1 FROM public.lab_arena_runs live WHERE live.assignment_id=parent.assignment_id AND live.status IN('pending','leased','submitted'))
   OR NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger l
    WHERE l.entry_id=p_provider_entry_id AND l.run_id=parent.run_id AND l.round_id=parent.round_id
     AND l.provider='openrouter' AND l.operation_id='openrouter.chat' AND l.entry_kind='uncertain'
     AND l.entry_doc#>'{call,call_succeeded}'='false'::JSONB
     AND l.entry_doc#>'{call,upstream_response_receipt,status}'='429'::JSONB
     AND l.entry_doc#>>'{call,upstream_response_receipt,call_identity}'=l.call_identity
     AND l.entry_doc#>>'{call,upstream_response_receipt,operation_id}'=l.operation_id
     AND l.entry_doc#>>'{call,upstream_response_receipt,body_sha256}'='sha256:'||encode(extensions.digest(decode(l.entry_doc#>>'{call,upstream_response_receipt,body_b64}','base64'),'sha256'),'hex')
     AND (convert_from(decode(l.entry_doc#>>'{call,upstream_response_receipt,body_b64}','base64'),'UTF8')::JSONB)#>'{error,code}'='429'::JSONB
     AND (convert_from(decode(l.entry_doc#>>'{call,upstream_response_receipt,body_b64}','base64'),'UTF8')::JSONB)#>>'{error,metadata,limit_source}'='upstream_provider_shared_pool'
     AND (convert_from(decode(l.entry_doc#>>'{call,upstream_response_receipt,body_b64}','base64'),'UTF8')::JSONB)#>'{error,metadata,is_byok}'='false'::JSONB
     AND EXISTS(SELECT 1 FROM public.lab_arena_ledger dispatch WHERE dispatch.run_id=l.run_id AND dispatch.call_identity=l.call_identity AND dispatch.entry_kind='dispatch' AND dispatch.entry_id<l.entry_id AND dispatch.provider=l.provider AND dispatch.operation_id=l.operation_id AND dispatch.funding_source=l.funding_source)
     AND EXISTS(SELECT 1 FROM public.lab_arena_ledger reservation WHERE reservation.run_id=l.run_id AND reservation.call_identity=l.call_identity AND reservation.entry_kind='reservation' AND reservation.entry_id<l.entry_id AND reservation.provider=l.provider AND reservation.operation_id=l.operation_id AND reservation.funding_source=l.funding_source AND reservation.entry_doc->>'request_hash'=l.entry_doc#>>'{call,upstream_response_receipt,request_hash}')) THEN
   RAISE EXCEPTION 'provider fault recovery retained shared-pool proof differs'; END IF;
 END IF;
 child:=parent; child.run_id$new671$),
 ($old671$'recovery_kind','operator_scorer_bug_recovery_v1'$old671$,$new671$'recovery_kind',p_recovery_kind$new671$),
 ($old671$response:=jsonb_build_object('status',CASE WHEN p_dry_run$old671$,$new671$IF p_recovery_kind='operator_provider_fault_recovery_v1' THEN proof:=proof||jsonb_build_object('provider_entry_id',p_provider_entry_id); END IF;
 response:=jsonb_build_object('status',CASE WHEN p_dry_run$new671$)
 ) changes(old_text,new_text) LOOP
  IF (length(d)-length(replace(d,patch.old_text,'')))/length(patch.old_text)<>1 THEN
   RAISE EXCEPTION 'provider recovery671 grant anchor differs'; END IF;
  d:=replace(d,patch.old_text,patch.new_text);
 END LOOP;
 EXECUTE 'CREATE FUNCTION public.lab_arena__grant_failed_judge_recovery_v1(p_parent_run_id TEXT,p_parent_sha256 TEXT,p_runtime_image_digest TEXT,p_request_id TEXT,p_actor_ref TEXT,p_correction_ref TEXT,p_dry_run BOOLEAN,p_recovery_kind TEXT,p_provider_entry_id BIGINT) RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS '||quote_literal(d);
 ALTER FUNCTION public.lab_arena__grant_failed_judge_recovery_v1(TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,BOOLEAN,TEXT,BIGINT) OWNER TO lab_arena_owner;
 REVOKE ALL ON FUNCTION public.lab_arena__grant_failed_judge_recovery_v1(TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,BOOLEAN,TEXT,BIGINT) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
 -- The existing child/claim/payer helpers recognize both explicit journal kinds.
 -- Completion and expiry already use this child identity and prevent recursion.
 FOR target IN SELECT * FROM (VALUES
 ('public.lab_arena__operator_scorer_bug_child_v1(public.lab_arena_runs)','52470b9f7bc12e4e7789c47513ab09e26bd4be49cbd684d0863f31172b7fc4de'),
 ('public.lab_arena__operator_scorer_bug_host_v1(public.lab_arena_runs)','7cdaf78e4d0bc2a88eacf32e0142f2d4859a6f9d7bc06db438cb041205ebb749'),
 ('public.lab_arena__operator_scorer_bug_claim_v1(public.lab_arena_rounds,public.lab_arena_runs)','9f8de76cd4f39c9ecb14d5b05fc9ad140c9b26881872d80b77163866f69f3235')
 ) functions(signature,source_sha256) LOOP
  oid_value:=to_regprocedure(target.signature);
  SELECT pg_get_functiondef(p.oid),p.prosrc,to_jsonb(p)-'prosrc' INTO d,before_source,metadata FROM pg_proc p WHERE p.oid=oid_value;
  IF encode(extensions.digest(before_source,'sha256'),'hex')<>target.source_sha256
   OR metadata->>'proowner' IS DISTINCT FROM ('lab_arena_owner'::regrole::OID)::TEXT
   OR metadata->'prosecdef' IS DISTINCT FROM 'true'::JSONB
   OR metadata->'proconfig' IS DISTINCT FROM '["search_path=pg_catalog, public"]'::JSONB
   OR metadata->'proacl' IS DISTINCT FROM '["lab_arena_owner=X/lab_arena_owner"]'::JSONB THEN
   RAISE EXCEPTION 'provider recovery671 child helper preimage differs'; END IF;
  IF (length(d)-length(replace(d,$old671$h.recovery_doc->>'recovery_kind'='operator_scorer_bug_recovery_v1'$old671$,'')))/length($old671$h.recovery_doc->>'recovery_kind'='operator_scorer_bug_recovery_v1'$old671$)<>1 THEN
   RAISE EXCEPTION 'provider recovery671 child helper anchor differs'; END IF;
  EXECUTE replace(d,$old671$h.recovery_doc->>'recovery_kind'='operator_scorer_bug_recovery_v1'$old671$,$new671$h.recovery_doc->>'recovery_kind' IN ('operator_scorer_bug_recovery_v1','operator_provider_fault_recovery_v1')$new671$);
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=oid_value) IS DISTINCT FROM metadata THEN
   RAISE EXCEPTION 'provider recovery671 child helper metadata changed'; END IF;
 END LOOP;
END;
$extend671$;
CREATE OR REPLACE FUNCTION public.lab_arena_grant_scorer_bug_recovery_v1(
 p_parent_run_id TEXT,p_parent_sha256 TEXT,p_runtime_image_digest TEXT,p_request_id TEXT,p_actor_ref TEXT,p_correction_ref TEXT,p_dry_run BOOLEAN DEFAULT TRUE)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $wrapper671$
BEGIN
 RETURN public.lab_arena__grant_failed_judge_recovery_v1(p_parent_run_id,p_parent_sha256,p_runtime_image_digest,p_request_id,p_actor_ref,p_correction_ref,p_dry_run,'operator_scorer_bug_recovery_v1',NULL);
END;
$wrapper671$;
CREATE OR REPLACE FUNCTION public.lab_arena_grant_provider_fault_recovery_v1(
 p_parent_run_id TEXT,p_parent_sha256 TEXT,p_runtime_image_digest TEXT,p_request_id TEXT,p_actor_ref TEXT,p_attestation_ref TEXT,p_provider_entry_id BIGINT,p_dry_run BOOLEAN DEFAULT TRUE)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $wrapper671$
BEGIN
 RETURN public.lab_arena__grant_failed_judge_recovery_v1(p_parent_run_id,p_parent_sha256,p_runtime_image_digest,p_request_id,p_actor_ref,p_attestation_ref,p_dry_run,'operator_provider_fault_recovery_v1',p_provider_entry_id);
END;
$wrapper671$;
ALTER FUNCTION public.lab_arena_grant_provider_fault_recovery_v1(TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,BIGINT,BOOLEAN) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_grant_provider_fault_recovery_v1(TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,BIGINT,BOOLEAN) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DO $restore671$
BEGIN
 IF NOT (SELECT had_create FROM pg_temp.arena671_schema_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.arena671_schema_acl) THEN
  RAISE EXCEPTION 'provider recovery671 schema ACL differs'; END IF;
END;
$restore671$;
NOTIFY pgrst,'reload schema';
COMMIT;
