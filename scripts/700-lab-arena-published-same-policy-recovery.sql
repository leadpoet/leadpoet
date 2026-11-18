-- Owner-approved published recovery can reuse the unchanged scorer after a
-- provider incident. Reuse the existing single budget exception; preserve
-- accepted work, frozen policy, checkpoints, costs and archived reward authority.
-- No automatic retries, new recovery service or historical reward changes.
BEGIN;
SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena700_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $patch700$
DECLARE p RECORD;d TEXT;metadata JSONB;patch RECORD;defaults TEXT;
 target TEXT:='public.lab_arena_correct_published_judgments_v1(text,text,text,timestamptz,timestamptz,jsonb,text,text,text,boolean)';
 preimage TEXT:='91f31d78aef2d11931d867f4789cf92fb7ffb0a7ca4131ebe06378fa251ba0dd';
 postimage TEXT:='8f2f9e1d339639d6a50e19863e12b4bd1ddc5fbf9733d6723811f7c54f4ed68d';
BEGIN
 SELECT * INTO STRICT p FROM pg_proc WHERE oid=target::regprocedure;
 metadata:=to_jsonb(p)-'prosrc'-'proargdefaults';defaults:=pg_get_expr(p.proargdefaults,0);d:=p.prosrc;
 IF pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR NOT p.prosecdef OR p.provolatile<>'v' OR p.prolang<>(SELECT oid FROM pg_language WHERE lanname='plpgsql')
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
  OR p.proisstrict OR p.proleakproof OR p.proparallel<>'u' THEN RAISE EXCEPTION 'published recovery700 security differs';END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')=postimage THEN RETURN;END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>preimage THEN RAISE EXCEPTION 'published recovery700 preimage differs';END IF;
 FOR patch IN SELECT * FROM (VALUES
($old700$revision BIGINT;digest TEXT;round_hash TEXT;$old700$,$new700$revision BIGINT;digest TEXT;round_hash TEXT;recovery_kind TEXT;$new700$),
($old700$FOR item IN SELECT value FROM jsonb_array_elements(p_requests)LOOP
   IF NOT EXISTS$old700$,$new700$FOR item IN SELECT value FROM jsonb_array_elements(p_requests)LOOP
   SELECT * INTO STRICT parent FROM public.lab_arena_runs WHERE run_id=item->>'parent_run_id';
   recovery_kind:=CASE WHEN p_runtime_image_digest=coalesce(parent.reviewer_runtime_doc->>'runtime_image_digest',r.configuration_doc->>'scorer_image_digest')
    THEN 'operator_judge_budget_exception_v1' ELSE 'operator_scorer_bug_recovery_v1' END;
   IF NOT EXISTS$new700$),
($old700$AND g.recovery_doc->>'recovery_kind'='operator_scorer_bug_recovery_v1'$old700$,$new700$AND g.recovery_doc->>'recovery_kind'=recovery_kind$new700$),
($old700$AND g.recovery_doc->>'runtime_image_digest'=p_runtime_image_digest AND g.recovery_doc->>'correction_ref'=p_approval_ref$old700$,$new700$AND g.recovery_doc->>'runtime_image_digest'=p_runtime_image_digest
     AND g.recovery_doc->>(CASE WHEN recovery_kind='operator_judge_budget_exception_v1' THEN 'approval_ref' ELSE 'correction_ref' END)=p_approval_ref$new700$),
($old700$SELECT * INTO STRICT parent FROM public.lab_arena_runs WHERE run_id=item->>'parent_run_id'FOR UPDATE;
  IF parent$old700$,$new700$SELECT * INTO STRICT parent FROM public.lab_arena_runs WHERE run_id=item->>'parent_run_id'FOR UPDATE;
  recovery_kind:=CASE WHEN p_runtime_image_digest=coalesce(parent.reviewer_runtime_doc->>'runtime_image_digest',r.configuration_doc->>'scorer_image_digest')
   THEN 'operator_judge_budget_exception_v1' ELSE 'operator_scorer_bug_recovery_v1' END;
  IF parent$new700$),
($old700$   OR p_runtime_image_digest=coalesce(parent.reviewer_runtime_doc->>'runtime_image_digest',r.configuration_doc->>'scorer_image_digest')
$old700$,$new700$$new700$),
($old700$h.recovery_doc->>'recovery_kind'='operator_scorer_bug_recovery_v1'
     AND h.recovery_doc#>>'{child_identity,assignment_id}'=parent.assignment_id$old700$,$new700$h.recovery_doc->>'recovery_kind'=recovery_kind
     AND h.recovery_doc#>>'{child_identity,assignment_id}'=parent.assignment_id$new700$),
($old700$  grants:=grants||jsonb_build_array(public.lab_arena_grant_scorer_bug_recovery_v1(item->>'parent_run_id',item->>'parent_sha256',
   p_runtime_image_digest,item->>'request_id',p_actor_ref,p_approval_ref,FALSE));$old700$,$new700$  SELECT * INTO STRICT parent FROM public.lab_arena_runs WHERE run_id=item->>'parent_run_id';
  IF p_runtime_image_digest=coalesce(parent.reviewer_runtime_doc->>'runtime_image_digest',r.configuration_doc->>'scorer_image_digest') THEN
   grants:=grants||jsonb_build_array(public.lab_arena_grant_judge_budget_exception_v1(item->>'parent_run_id',item->>'parent_sha256',
    p_runtime_image_digest,item->>'request_id',p_actor_ref,p_approval_ref,item->'checkpoints',item->'signal_checkpoints',FALSE));
  ELSE
   grants:=grants||jsonb_build_array(public.lab_arena_grant_scorer_bug_recovery_v1(item->>'parent_run_id',item->>'parent_sha256',
    p_runtime_image_digest,item->>'request_id',p_actor_ref,p_approval_ref,FALSE));
  END IF;$new700$)
 ) changes(old_text,new_text) LOOP
  IF (length(d)-length(replace(d,patch.old_text,'')))/length(patch.old_text)<>1 THEN RAISE EXCEPTION 'published recovery700 anchor differs';END IF;
  d:=replace(d,patch.old_text,patch.new_text);
 END LOOP;
 IF encode(extensions.digest(d,'sha256'),'hex')<>postimage THEN RAISE EXCEPTION 'published recovery700 postimage differs';END IF;
 EXECUTE replace(pg_get_functiondef(p.oid),'$function$'||p.prosrc||'$function$',quote_literal(d));
 IF (SELECT to_jsonb(q)-'prosrc'-'proargdefaults' FROM pg_proc q WHERE q.oid=p.oid)IS DISTINCT FROM metadata
  OR (SELECT pg_get_expr(q.proargdefaults,0) FROM pg_proc q WHERE q.oid=p.oid)IS DISTINCT FROM defaults
  OR (SELECT encode(extensions.digest(q.prosrc,'sha256'),'hex') FROM pg_proc q WHERE q.oid=p.oid)<>postimage THEN
  RAISE EXCEPTION 'published recovery700 metadata changed';END IF;
END;
$patch700$;
DO $restore700$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.arena700_acl)THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;END IF;
 IF(SELECT nspacl FROM pg_namespace WHERE nspname='public')IS DISTINCT FROM(SELECT nspacl FROM pg_temp.arena700_acl)THEN
  RAISE EXCEPTION 'published recovery700 schema ACL differs';END IF;
END;
$restore700$;
NOTIFY pgrst,'reload schema';COMMIT;
