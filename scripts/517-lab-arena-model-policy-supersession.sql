-- A guarded operator model-slot revision rejudges saved outputs uniformly.
-- Original score attempts, cache authorities, charges and trajectory stay put.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
DO $security517$
BEGIN
 IF (SELECT jsonb_build_array(pg_get_userbyid(c.relowner),(SELECT jsonb_agg(jsonb_build_array(pg_get_userbyid(a.grantee),a.privilege_type,a.is_grantable,pg_get_userbyid(a.grantor)) ORDER BY pg_get_userbyid(a.grantee),a.privilege_type) FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner)))a WHERE a.grantee<>c.relowner),c.relrowsecurity,c.relforcerowsecurity,c.reloptions)
   FROM pg_class c WHERE c.oid=to_regclass('public.lab_arena_published_results_v1'))
   IS DISTINCT FROM $relation517$["lab_arena_owner",[["lab_arena_service","SELECT",false,"lab_arena_owner"],["leadpoet_parity_reader","SELECT",false,"lab_arena_owner"]],false,false,["security_invoker=true"]]$relation517$::JSONB THEN
  RAISE EXCEPTION 'Supersession relation security differs: lab_arena_published_results_v1' USING ERRCODE='42501';
 END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(c.relowner),(SELECT jsonb_agg(jsonb_build_array(pg_get_userbyid(a.grantee),a.privilege_type,a.is_grantable,pg_get_userbyid(a.grantor)) ORDER BY pg_get_userbyid(a.grantee),a.privilege_type) FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner)))a WHERE a.grantee<>c.relowner),c.relrowsecurity,c.relforcerowsecurity,c.reloptions)
   FROM pg_class c WHERE c.oid=to_regclass('public.lab_arena_restart_claim_control'))
   IS DISTINCT FROM $relation517$["lab_arena_owner",[["leadpoet_parity_reader","SELECT",false,"lab_arena_owner"]],true,false,null]$relation517$::JSONB THEN
  RAISE EXCEPTION 'Supersession relation security differs: lab_arena_restart_claim_control' USING ERRCODE='42501';
 END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(c.relowner),(SELECT jsonb_agg(jsonb_build_array(pg_get_userbyid(a.grantee),a.privilege_type,a.is_grantable,pg_get_userbyid(a.grantor)) ORDER BY pg_get_userbyid(a.grantee),a.privilege_type) FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner)))a WHERE a.grantee<>c.relowner),c.relrowsecurity,c.relforcerowsecurity,c.reloptions)
   FROM pg_class c WHERE c.oid=to_regclass('public.lab_arena_rounds'))
   IS DISTINCT FROM $relation517$["lab_arena_owner",[["lab_arena_service","SELECT",false,"lab_arena_owner"],["leadpoet_parity_reader","SELECT",false,"lab_arena_owner"]],true,false,null]$relation517$::JSONB THEN
  RAISE EXCEPTION 'Supersession relation security differs: lab_arena_rounds' USING ERRCODE='42501';
 END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(c.relowner),(SELECT jsonb_agg(jsonb_build_array(pg_get_userbyid(a.grantee),a.privilege_type,a.is_grantable,pg_get_userbyid(a.grantor)) ORDER BY pg_get_userbyid(a.grantee),a.privilege_type) FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner)))a WHERE a.grantee<>c.relowner),c.relrowsecurity,c.relforcerowsecurity,c.reloptions)
   FROM pg_class c WHERE c.oid=to_regclass('public.lab_arena_runs'))
   IS DISTINCT FROM $relation517$["lab_arena_owner",[["lab_arena_service","SELECT",false,"lab_arena_owner"],["leadpoet_parity_reader","SELECT",false,"lab_arena_owner"]],true,false,null]$relation517$::JSONB THEN
  RAISE EXCEPTION 'Supersession relation security differs: lab_arena_runs' USING ERRCODE='42501';
 END IF;
END;
$security517$;
DO $installed517$
DECLARE t OID:=to_regclass('public.lab_arena_scoring_supersessions');
 f OID:=to_regprocedure('public.lab_arena_supersede_role_model_v1(text,jsonb,bigint,bigint,text,jsonb,text,text)');
 reader OID:=to_regprocedure('public.lab_arena_scoring_policy_history_v1(text)');
 hold_fn OID:=to_regprocedure('public.lab_arena_scoring_operator_hold_v1(text,jsonb,bigint,bigint,bigint,text,text,boolean)');
BEGIN
 IF t IS NOT NULL AND EXISTS(SELECT 1 FROM pg_class c WHERE c.oid=t AND (
   pg_get_userbyid(c.relowner)<>'lab_arena_owner' OR NOT c.relrowsecurity
   OR c.relforcerowsecurity OR c.reloptions IS NOT NULL
   OR EXISTS(SELECT 1 FROM pg_policy policy WHERE policy.polrelid=c.oid)
   OR NOT EXISTS(SELECT 1 FROM pg_trigger trigger WHERE trigger.tgrelid=c.oid AND NOT trigger.tgisinternal
     AND trigger.tgname='lab_arena_scoring_supersessions_append_only'
     AND trigger.tgfoid=to_regprocedure('public.lab_arena_append_only_v1()')
     AND trigger.tgenabled='O' AND trigger.tgtype=27)
   OR EXISTS(SELECT 1 FROM pg_trigger trigger WHERE trigger.tgrelid=c.oid AND NOT trigger.tgisinternal
     AND trigger.tgname<>'lab_arena_scoring_supersessions_append_only')
   OR EXISTS(SELECT 1 FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner))) a
     WHERE a.grantee<>c.relowner AND NOT (a.grantee=to_regrole('lab_arena_service')
       AND a.privilege_type='SELECT' AND NOT a.is_grantable)))) THEN
  RAISE EXCEPTION 'Supersession history security differs' USING ERRCODE='42501';
 END IF;
 IF f IS NOT NULL AND EXISTS(SELECT 1 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
   WHERE p.oid=f AND (pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef
    OR p.provolatile<>'v' OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']
    OR l.lanname<>'plpgsql'
    OR EXISTS(SELECT 1 FROM aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a
      WHERE a.grantee<>p.proowner))) THEN
  RAISE EXCEPTION 'Supersession operator security differs' USING ERRCODE='42501';
 END IF;
 IF reader IS NOT NULL AND (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,
   p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
   WHERE p.oid=reader) IS DISTINCT FROM
   '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"sql"]'::JSONB THEN
  RAISE EXCEPTION 'Supersession policy reader security differs' USING ERRCODE='42501';
 END IF;
 IF hold_fn IS NOT NULL AND (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,
   p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
   WHERE p.oid=hold_fn) IS DISTINCT FROM
   '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
  RAISE EXCEPTION 'Supersession operator hold security differs' USING ERRCODE='42501';
 END IF;
END;
$installed517$;
ALTER TABLE public.lab_arena_rounds ADD COLUMN IF NOT EXISTS scoring_generation_floor BIGINT NOT NULL DEFAULT 0
  CHECK(scoring_generation_floor>=0);
-- Existing rows retain revision zero. Omitted values for new rows are
-- authenticated from the round by the existing INSERT integrity guard.
ALTER TABLE public.lab_arena_runs ADD COLUMN IF NOT EXISTS scoring_generation_floor BIGINT NOT NULL DEFAULT 0
 CHECK(scoring_generation_floor>=0 AND (kind='score' OR scoring_generation_floor=0));
ALTER TABLE public.lab_arena_runs ALTER COLUMN scoring_generation_floor DROP DEFAULT;
DO $accepted_index517$
DECLARE d TEXT;
BEGIN
 IF NOT EXISTS(SELECT 1 FROM pg_class c JOIN pg_index i ON i.indexrelid=c.oid
   WHERE c.oid=to_regclass('public.lab_arena_runs_one_active_attempt_uq')
    AND pg_get_userbyid(c.relowner)='lab_arena_owner' AND i.indisunique AND i.indisvalid AND i.indisready
    AND pg_get_indexdef(c.oid)='CREATE UNIQUE INDEX lab_arena_runs_one_active_attempt_uq ON public.lab_arena_runs USING btree (assignment_id) WHERE (status = ANY (ARRAY[''pending''::text, ''leased''::text, ''submitted''::text]))') THEN
  RAISE EXCEPTION 'Supersession active index differs' USING ERRCODE='55000';
 END IF;
 SELECT pg_get_indexdef(c.oid) INTO d FROM pg_class c JOIN pg_index i ON i.indexrelid=c.oid
 WHERE c.oid=to_regclass('public.lab_arena_runs_one_accepted_uq')
  AND pg_get_userbyid(c.relowner)='lab_arena_owner' AND i.indisunique AND i.indisvalid AND i.indisready;
 IF d='CREATE UNIQUE INDEX lab_arena_runs_one_accepted_uq ON public.lab_arena_runs USING btree (assignment_id, scoring_generation_floor) WHERE (status = ''accepted''::text)' THEN RETURN; END IF;
 IF d IS DISTINCT FROM 'CREATE UNIQUE INDEX lab_arena_runs_one_accepted_uq ON public.lab_arena_runs USING btree (assignment_id) WHERE (status = ''accepted''::text)' THEN
  RAISE EXCEPTION 'Supersession accepted index differs' USING ERRCODE='55000';
 END IF;
 DROP INDEX public.lab_arena_runs_one_accepted_uq;
 CREATE UNIQUE INDEX lab_arena_runs_one_accepted_uq ON public.lab_arena_runs(assignment_id,scoring_generation_floor) WHERE status='accepted';
 IF pg_get_indexdef('public.lab_arena_runs_one_accepted_uq'::regclass) IS DISTINCT FROM
  'CREATE UNIQUE INDEX lab_arena_runs_one_accepted_uq ON public.lab_arena_runs USING btree (assignment_id, scoring_generation_floor) WHERE (status = ''accepted''::text)' THEN
  RAISE EXCEPTION 'Supersession accepted index readback differs' USING ERRCODE='55000';
 END IF;
END;
$accepted_index517$;
CREATE TABLE IF NOT EXISTS public.lab_arena_scoring_supersessions (
  round_id TEXT NOT NULL REFERENCES public.lab_arena_rounds(round_id),
  generation_floor BIGINT NOT NULL CHECK(generation_floor>0),
  request_id TEXT NOT NULL,
  request_hash TEXT NOT NULL CHECK(request_hash ~ '^sha256:[0-9a-f]{64}$'),
  actor_ref TEXT NOT NULL,
  transaction_id BIGINT NOT NULL,
  backend_pid INTEGER NOT NULL,
  old_round_doc JSONB NOT NULL,
  new_round_doc JSONB NOT NULL,
  old_score_projections JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY(round_id,generation_floor), UNIQUE(round_id,request_id)
);
ALTER TABLE public.lab_arena_scoring_supersessions OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_scoring_supersessions ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.lab_arena_scoring_supersessions FROM PUBLIC,anon,authenticated,service_role,lab_arena_service,leadpoet_parity_reader;
-- Ordinary invoker triggers may test the exception; RLS reveals no history to
-- the API role. Only the owner-created transaction can see its authorization.
GRANT SELECT ON public.lab_arena_scoring_supersessions TO lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_scoring_supersessions_append_only ON public.lab_arena_scoring_supersessions;
CREATE TRIGGER lab_arena_scoring_supersessions_append_only BEFORE UPDATE OR DELETE
 ON public.lab_arena_scoring_supersessions FOR EACH ROW EXECUTE FUNCTION public.lab_arena_append_only_v1();

CREATE OR REPLACE FUNCTION public.lab_arena_supersede_role_model_v1(
 p_round_id TEXT,p_expected_configuration JSONB,p_expected_status_generation BIGINT,
 p_expected_stage_generation BIGINT,p_new_model TEXT,p_work_items JSONB,
 p_request_id TEXT,p_actor_ref TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $supersede$
DECLARE
 r public.lab_arena_rounds; n public.lab_arena_rounds;
 h public.lab_arena_scoring_supersessions;
 c public.lab_arena_restart_claim_control;
 source public.lab_arena_runs; previous public.lab_arena_runs;
 item JSONB; projections JSONB; config JSONB; request_hash TEXT;
 next_attempt INTEGER; queued INTEGER:=0; accepted_count INTEGER;
BEGIN
 IF p_round_id IS NULL OR jsonb_typeof(p_expected_configuration) IS DISTINCT FROM 'object'
  OR p_expected_status_generation IS NULL OR p_expected_stage_generation IS NULL
  OR p_new_model IS NULL OR p_new_model !~ '^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._:-]{0,119}$'
  OR jsonb_typeof(p_work_items) IS DISTINCT FROM 'array'
  OR char_length(coalesce(p_request_id,'')) NOT BETWEEN 16 AND 128
  OR char_length(coalesce(p_actor_ref,'')) NOT BETWEEN 1 AND 128 THEN
  RAISE EXCEPTION 'lab_arena_supersession_input_invalid' USING ERRCODE='22023';
 END IF;
 request_hash:='sha256:'||encode(extensions.digest(jsonb_build_object(
   'configuration',p_expected_configuration,'status_generation',p_expected_status_generation,
   'stage_generation',p_expected_stage_generation,'new_model',p_new_model,
   'work_items',p_work_items,'actor_ref',p_actor_ref)::TEXT,'sha256'),'hex');
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO r FROM public.lab_arena_rounds WHERE round_id=p_round_id FOR UPDATE;
 IF NOT FOUND THEN RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE='P0002'; END IF;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
   WHERE round_id=p_round_id AND request_id=p_request_id;
 IF FOUND THEN
  IF h.request_hash IS DISTINCT FROM request_hash THEN
   RAISE EXCEPTION 'lab_arena_supersession_request_conflict' USING ERRCODE='22023';
  END IF;
  RETURN jsonb_build_object('status','existing','generation_floor',h.generation_floor);
 END IF;
 SELECT * INTO c FROM public.lab_arena_restart_claim_control WHERE singleton FOR SHARE;
 IF c.singleton IS NULL OR NOT c.operator_paused
  OR c.actor_ref IS DISTINCT FROM p_actor_ref
  OR coalesce(c.guard_commitment,'')<>'' OR coalesce(c.owner_commitment,'')<>''
  OR c.guard_expires_at IS NOT NULL OR coalesce(c.candidate_commit,'')<>''
  OR coalesce(c.restart_scope,'')<>'' OR coalesce(c.restart_phase,'')<>''
  OR coalesce(c.captured_leases,'[]'::JSONB)<>'[]'::JSONB
  OR r.status NOT IN ('stage2_scoring','stage2_judged','scored')
  OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL
  OR r.reward_activated_at IS NOT NULL OR r.reward_basis_hash IS NOT NULL
  OR r.configuration_doc IS DISTINCT FROM p_expected_configuration
  OR r.status_generation IS DISTINCT FROM p_expected_status_generation
  OR r.stage_generation IS DISTINCT FROM p_expected_stage_generation
  OR r.status_generation>=9223372036854775806
  OR r.stage_generation>=9223372036854775806
  OR r.configuration_doc->>'execution_sequence_policy' IS DISTINCT FROM 'baseline_scored_first_v1'
  OR r.configuration_doc->>'integrity_policy' IS DISTINCT FROM 'arena_integrity_v1'
  OR NOT public.lab_arena__company_judgment_policy_v1(r)
  OR NOT public.lab_arena_judge_continuation_enabled_v1(r.round_id)
  OR jsonb_typeof(r.configuration_doc#>'{scorer_policy,judge_models,role_batch_check}') IS DISTINCT FROM 'string'
  OR jsonb_typeof(r.configuration_doc#>'{scorer_policy,env_bindings}') IS DISTINCT FROM 'object'
  OR r.configuration_doc#>>'{scorer_policy,judge_models,role_batch_check}'=p_new_model THEN
  RAISE EXCEPTION 'lab_arena_supersession_precondition_failed' USING ERRCODE='55000';
 END IF;
 IF EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=p_round_id
   AND a.status IN ('pending','leased','submitted'))
  OR EXISTS(SELECT 1 FROM (SELECT DISTINCT assignment_id FROM public.lab_arena_runs
      WHERE round_id=p_round_id) a
    WHERE NOT public.lab_arena__judge_calls_clear_v1(a.assignment_id)) THEN
  RAISE EXCEPTION 'lab_arena_supersession_not_drained' USING ERRCODE='55000';
 END IF;
 config:=jsonb_set(jsonb_set(r.configuration_doc,
   '{scorer_policy,judge_models,role_batch_check}',to_jsonb(p_new_model)),
   '{scorer_policy,env_bindings,ARENA_ROLE_BATCH_MODEL}',to_jsonb(p_new_model));
 n:=r; n.configuration_doc:=config; n.stage_generation:=r.stage_generation+1;
 n.status_generation:=r.status_generation+1; n.scoring_generation_floor:=n.stage_generation;
 n.status:='stage2_scoring';
 SELECT count(*) INTO accepted_count FROM public.lab_arena_runs a
  WHERE a.round_id=p_round_id AND a.kind='execute' AND a.status='accepted';
 IF accepted_count=0 OR accepted_count<>jsonb_array_length(p_work_items)
  OR accepted_count<>(SELECT count(DISTINCT x->>'scored_run_id') FROM jsonb_array_elements(p_work_items)x)
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=p_round_id
    AND a.kind='execute' AND a.status='accepted'
    GROUP BY a.assignment_id HAVING count(*)<>1) THEN
  RAISE EXCEPTION 'lab_arena_supersession_work_incomplete' USING ERRCODE='22023';
 END IF;
 FOR item IN SELECT value FROM jsonb_array_elements(p_work_items) LOOP
  IF jsonb_typeof(item) IS DISTINCT FROM 'object'
   OR (SELECT array_agg(k ORDER BY k) FROM jsonb_object_keys(item)k) IS DISTINCT FROM
     ARRAY['company_judgment_refs','icp_position','output_ref','scored_run_id','submission_id']::TEXT[]
   OR jsonb_typeof(item->'icp_position') IS DISTINCT FROM 'number'
   OR NOT public.lab_arena__company_judgment_refs_v1(n,item->'company_judgment_refs') THEN
   RAISE EXCEPTION 'lab_arena_supersession_item_invalid' USING ERRCODE='22023';
  END IF;
  SELECT * INTO source FROM public.lab_arena_runs a WHERE a.run_id=item->>'scored_run_id'
    AND a.round_id=p_round_id AND a.kind='execute' AND a.status='accepted'
    AND a.terminal_cause='accepted' AND a.submission_id=item->>'submission_id'
    AND a.icp_position=(item->>'icp_position')::INTEGER
    AND a.output_ref=item->>'output_ref' FOR UPDATE;
  IF NOT FOUND OR source.assignment_id IS DISTINCT FROM
    source.round_id||':'||source.submission_id||':'||source.stage::TEXT||':'||source.icp_position::TEXT
    OR NOT EXISTS(SELECT 1 FROM jsonb_array_elements(r.participants)p
      WHERE p->>'submission_id'=source.submission_id AND p->>'miner_hotkey'=source.miner_hotkey)
    OR NOT EXISTS(SELECT 1 FROM jsonb_array_elements(CASE source.stage WHEN 1
      THEN r.stage1_scoring_plan_doc ELSE r.stage2_scoring_plan_doc END->'work_items')w
      WHERE w->>'scored_run_id'=source.run_id AND w->>'output_ref'=source.output_ref
       AND w->>'submission_id'=source.submission_id AND w->>'icp_position'=source.icp_position::TEXT)
    OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.assignment_id=source.assignment_id||':score'
      AND a.attempt=2147483647)
    OR EXISTS(SELECT 1 FROM public.lab_arena_runs a,
      jsonb_array_elements(a.company_judgment_refs) old_ref,
      jsonb_array_elements(item->'company_judgment_refs') new_ref
      WHERE a.scored_run_id=source.run_id AND a.kind='score'
       AND old_ref->>'cache_key'=new_ref->>'cache_key') THEN
    RAISE EXCEPTION 'lab_arena_supersession_source_invalid' USING ERRCODE='22023';
  END IF;
 END LOOP;
 SELECT coalesce(jsonb_object_agg(a.run_id,jsonb_build_object('per_icp_score',a.per_icp_score,
   'qualification_doc',a.qualification_doc)),'{}'::JSONB) INTO projections
 FROM public.lab_arena_runs a WHERE a.round_id=p_round_id AND a.kind='execute';
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,
   actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections)
 VALUES(p_round_id,n.scoring_generation_floor,p_request_id,request_hash,p_actor_ref,
   txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at',projections);
 UPDATE public.lab_arena_rounds SET configuration_doc=n.configuration_doc,status=n.status,
   status_generation=n.status_generation,stage_generation=n.stage_generation,
   scoring_generation_floor=n.scoring_generation_floor WHERE round_id=p_round_id;
 UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL
   WHERE round_id=p_round_id AND kind='execute'
     AND (per_icp_score IS NOT NULL OR qualification_doc IS NOT NULL);
 FOR item IN SELECT value FROM jsonb_array_elements(p_work_items) LOOP
  SELECT * INTO source FROM public.lab_arena_runs WHERE run_id=item->>'scored_run_id';
  SELECT * INTO previous FROM public.lab_arena_runs
    WHERE assignment_id=source.assignment_id||':score' ORDER BY attempt DESC LIMIT 1;
  next_attempt:=coalesce(previous.attempt,0)+1;
  INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,
    stage,icp_position,attempt,status,lease_generation,stage_generation,kind,scored_run_id,
    company_judgment_refs)
  VALUES(source.assignment_id||':score:'||next_attempt::TEXT,source.assignment_id||':score',
    p_round_id,source.submission_id,source.miner_hotkey,source.stage,source.icp_position,
    next_attempt,'pending',coalesce(previous.lease_generation,0),n.stage_generation,'score',source.run_id,
    item->'company_judgment_refs');
  queued:=queued+1;
 END LOOP;
 RETURN jsonb_build_object('status','superseded','generation_floor',n.scoring_generation_floor,
   'queued',queued,'request_hash',request_hash);
END;
$supersede$;
ALTER FUNCTION public.lab_arena_supersede_role_model_v1(TEXT,JSONB,BIGINT,BIGINT,TEXT,JSONB,TEXT,TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_supersede_role_model_v1(TEXT,JSONB,BIGINT,BIGINT,TEXT,JSONB,TEXT,TEXT)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
-- Owner-only operational hold. It cannot replace another operator or a
-- canonical restart guard, and never changes a lease or guard generation.
CREATE OR REPLACE FUNCTION public.lab_arena_scoring_operator_hold_v1(
 p_round_id TEXT,p_expected_configuration JSONB,p_expected_status_generation BIGINT,
 p_expected_stage_generation BIGINT,p_expected_guard_generation BIGINT,
 p_actor_ref TEXT,p_reason TEXT,p_hold BOOLEAN)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $hold$
DECLARE r public.lab_arena_rounds; c public.lab_arena_restart_claim_control;
BEGIN
 IF jsonb_typeof(p_expected_configuration) IS DISTINCT FROM 'object'
  OR p_expected_status_generation IS NULL OR p_expected_stage_generation IS NULL
  OR p_expected_guard_generation IS NULL OR p_hold IS NULL
  OR char_length(coalesce(p_actor_ref,'')) NOT BETWEEN 1 AND 128
  OR char_length(coalesce(p_reason,'')) NOT BETWEEN 1 AND 256 THEN
  RAISE EXCEPTION 'lab_arena_operator_hold_input_invalid' USING ERRCODE='22023';
 END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO c FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
 SELECT * INTO r FROM public.lab_arena_rounds WHERE round_id=p_round_id FOR SHARE;
 IF NOT FOUND OR r.status NOT IN ('stage2_scoring','stage2_judged','scored')
  OR r.configuration_doc IS DISTINCT FROM p_expected_configuration
  OR r.status_generation IS DISTINCT FROM p_expected_status_generation
  OR r.stage_generation IS DISTINCT FROM p_expected_stage_generation
  OR c.singleton IS NULL OR c.guard_generation IS DISTINCT FROM p_expected_guard_generation
  OR coalesce(c.guard_commitment,'')<>'' OR coalesce(c.owner_commitment,'')<>''
  OR c.guard_expires_at IS NOT NULL OR coalesce(c.candidate_commit,'')<>''
  OR coalesce(c.restart_scope,'')<>'' OR coalesce(c.restart_phase,'')<>''
  OR coalesce(c.captured_leases,'[]'::JSONB)<>'[]'::JSONB THEN
  RAISE EXCEPTION 'lab_arena_operator_hold_cas_failed' USING ERRCODE='55000';
 END IF;
 IF (c.operator_paused OR NOT p_hold)
  AND (c.actor_ref IS DISTINCT FROM p_actor_ref OR c.pause_reason IS DISTINCT FROM p_reason) THEN
  RAISE EXCEPTION 'lab_arena_operator_hold_owner_conflict' USING ERRCODE='42501';
 END IF;
 IF c.operator_paused IS NOT DISTINCT FROM p_hold THEN
  RETURN jsonb_build_object('status','existing','operator_paused',p_hold,'guard_generation',c.guard_generation);
 END IF;
 UPDATE public.lab_arena_restart_claim_control SET operator_paused=p_hold,
  actor_ref=p_actor_ref,pause_reason=p_reason,updated_at=clock_timestamp() WHERE singleton;
 RETURN jsonb_build_object('status',CASE WHEN p_hold THEN 'held' ELSE 'released' END,
   'operator_paused',p_hold,'guard_generation',c.guard_generation);
END;
$hold$;
ALTER FUNCTION public.lab_arena_scoring_operator_hold_v1(TEXT,JSONB,BIGINT,BIGINT,BIGINT,TEXT,TEXT,BOOLEAN) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_scoring_operator_hold_v1(TEXT,JSONB,BIGINT,BIGINT,BIGINT,TEXT,TEXT,BOOLEAN)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
CREATE OR REPLACE FUNCTION public.lab_arena_scoring_policy_history_v1(p_round_id TEXT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $history$
 SELECT jsonb_build_object('round_id',p_round_id,'revisions',coalesce(jsonb_agg(
   jsonb_build_object('previous_generation_floor',coalesce((h.old_round_doc->>'scoring_generation_floor')::BIGINT,0),
     'generation_floor',h.generation_floor,
     'old_scorer_policy',h.old_round_doc#>'{configuration_doc,scorer_policy}',
     'new_scorer_policy',h.new_round_doc#>'{configuration_doc,scorer_policy}')
   ORDER BY h.generation_floor),'[]'::JSONB))
 FROM public.lab_arena_scoring_supersessions h WHERE h.round_id=p_round_id;
$history$;
ALTER FUNCTION public.lab_arena_scoring_policy_history_v1(TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_scoring_policy_history_v1(TEXT)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_scoring_policy_history_v1(TEXT) TO lab_arena_service;
-- Trigger and exact existing-function amendments follow below.

DO $patch517_0$
DECLARE d TEXT; m JSONB; target OID:=to_regprocedure('public.lab_arena_rounds_write_once_v1()');
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF d IS NULL THEN RAISE EXCEPTION 'Supersession function missing: lab_arena_rounds_write_once_v1' USING ERRCODE='55000'; END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security517$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",false,"v",["search_path=pg_catalog"],"plpgsql"]$security517$::JSONB THEN RAISE EXCEPTION 'Supersession security differs: lab_arena_rounds_write_once_v1' USING ERRCODE='42501'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='f6e03c8ad4080e3e713e282719fae154ba5d3d98b9151fab282ac098e44a7c77' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'31e88aa868a9caecaffd4571a294e5fb91ce2e29b29748f2157eb98eca05b64b' THEN RAISE EXCEPTION 'Supersession preimage differs: lab_arena_rounds_write_once_v1' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,$needle517_0_0$BEGIN
  IF TG_OP = 'DELETE' THEN$needle517_0_0$,'')))<>length($needle517_0_0$BEGIN
  IF TG_OP = 'DELETE' THEN$needle517_0_0$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena_rounds_write_once_v1/0' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_0_0$BEGIN
  IF TG_OP = 'DELETE' THEN$needle517_0_0$,$replace517_0_0$BEGIN
  IF TG_OP='UPDATE' AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
    WHERE h.round_id=OLD.round_id AND h.transaction_id=txid_current()
      AND h.backend_pid=pg_backend_pid()
      AND h.old_round_doc=to_jsonb(OLD)-'updated_at'
      -- Stored generated columns are NULL in a BEFORE trigger. Their input
      -- configuration remains part of the exact comparison.
      AND (h.new_round_doc-ARRAY['arena_network_name','arena_netuid'])=
        (to_jsonb(NEW)-ARRAY['updated_at','arena_network_name','arena_netuid'])) THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
  IF NEW.scoring_generation_floor IS DISTINCT FROM OLD.scoring_generation_floor THEN
    RAISE EXCEPTION 'scoring generation floor is operator-owned' USING ERRCODE='42501';
  END IF;
  IF TG_OP = 'DELETE' THEN$replace517_0_0$);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'f6e03c8ad4080e3e713e282719fae154ba5d3d98b9151fab282ac098e44a7c77' THEN RAISE EXCEPTION 'Supersession postimage differs: lab_arena_rounds_write_once_v1' USING ERRCODE='55000'; END IF;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'f6e03c8ad4080e3e713e282719fae154ba5d3d98b9151fab282ac098e44a7c77' THEN RAISE EXCEPTION 'Supersession readback differs: lab_arena_rounds_write_once_v1' USING ERRCODE='55000'; END IF;
END;
$patch517_0$;

DO $patch517_1$
DECLARE d TEXT; m JSONB; target OID:=to_regprocedure('public.lab_arena_runs_terminal_v1()');
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF d IS NULL THEN RAISE EXCEPTION 'Supersession function missing: lab_arena_runs_terminal_v1' USING ERRCODE='55000'; END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security517$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",false,"v",["search_path=pg_catalog"],"plpgsql"]$security517$::JSONB THEN RAISE EXCEPTION 'Supersession security differs: lab_arena_runs_terminal_v1' USING ERRCODE='42501'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='e4e49916138628b720271f6f467656574f8ae62b08024f4891b3f74bfa7f5920' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'a36d6445666779f1da45e20ee5de8cba54831885a68e36be1b13a37a875cb245' THEN RAISE EXCEPTION 'Supersession preimage differs: lab_arena_runs_terminal_v1' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,$needle517_1_0$BEGIN
  IF TG_OP = 'DELETE' THEN$needle517_1_0$,'')))<>length($needle517_1_0$BEGIN
  IF TG_OP = 'DELETE' THEN$needle517_1_0$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena_runs_terminal_v1/0' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_1_0$BEGIN
  IF TG_OP = 'DELETE' THEN$needle517_1_0$,$replace517_1_0$BEGIN
  IF TG_OP='UPDATE' AND OLD.kind='execute'
    AND NEW.per_icp_score IS NULL AND NEW.qualification_doc IS NULL
    AND (to_jsonb(NEW)-ARRAY['per_icp_score','qualification_doc','updated_at'])=
        (to_jsonb(OLD)-ARRAY['per_icp_score','qualification_doc','updated_at'])
    AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      WHERE h.round_id=OLD.round_id AND h.transaction_id=txid_current()
       AND h.backend_pid=pg_backend_pid()
       AND h.old_score_projections->OLD.run_id=jsonb_build_object(
         'per_icp_score',OLD.per_icp_score,'qualification_doc',OLD.qualification_doc)) THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
  IF TG_OP = 'DELETE' THEN$replace517_1_0$);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'e4e49916138628b720271f6f467656574f8ae62b08024f4891b3f74bfa7f5920' THEN RAISE EXCEPTION 'Supersession postimage differs: lab_arena_runs_terminal_v1' USING ERRCODE='55000'; END IF;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'e4e49916138628b720271f6f467656574f8ae62b08024f4891b3f74bfa7f5920' THEN RAISE EXCEPTION 'Supersession readback differs: lab_arena_runs_terminal_v1' USING ERRCODE='55000'; END IF;
END;
$patch517_1$;

DO $patch517_2$
DECLARE d TEXT; m JSONB; target OID:=to_regprocedure('public.lab_arena_integrity_run_guard_v1()');
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF d IS NULL THEN RAISE EXCEPTION 'Supersession function missing: lab_arena_integrity_run_guard_v1' USING ERRCODE='55000'; END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security517$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,anon=X/lab_arena_owner,authenticated=X/lab_arena_owner,service_role=X/lab_arena_owner}",false,"v",["search_path=pg_catalog, public"],"plpgsql"]$security517$::JSONB THEN RAISE EXCEPTION 'Supersession security differs: lab_arena_integrity_run_guard_v1' USING ERRCODE='42501'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='dd7fcd3811ea5f85a5437df0a9c77b80bb688ff3a972527949d29816fcf4376d' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'a5c4e600c6edb30d4bec3ed0714e2dabd9c2f8cfd60f25446eae522907721de7' THEN RAISE EXCEPTION 'Supersession preimage differs: lab_arena_integrity_run_guard_v1' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,$needle517_2_0$BEGIN
  SELECT * INTO v_round$needle517_2_0$,'')))<>length($needle517_2_0$BEGIN
  SELECT * INTO v_round$needle517_2_0$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena_integrity_run_guard_v1/0' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_2_0$BEGIN
  SELECT * INTO v_round$needle517_2_0$,$replace517_2_0$BEGIN
  IF TG_OP='UPDATE' AND OLD.kind='execute'
    AND NEW.per_icp_score IS NULL AND NEW.qualification_doc IS NULL
    AND (to_jsonb(NEW)-ARRAY['per_icp_score','qualification_doc','updated_at'])=
        (to_jsonb(OLD)-ARRAY['per_icp_score','qualification_doc','updated_at'])
    AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      WHERE h.round_id=OLD.round_id AND h.transaction_id=txid_current()
       AND h.backend_pid=pg_backend_pid()
       AND h.old_score_projections->OLD.run_id=jsonb_build_object(
         'per_icp_score',OLD.per_icp_score,'qualification_doc',OLD.qualification_doc)) THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
  SELECT * INTO v_round$replace517_2_0$);
 IF (length(d)-length(replace(d,$needle517_2_1$BEGIN
$needle517_2_1$,'')))<>length($needle517_2_1$BEGIN
$needle517_2_1$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena_integrity_run_guard_v1/1' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_2_1$BEGIN
$needle517_2_1$,$replace517_2_1$BEGIN
  IF TG_OP='UPDATE' AND NEW.scoring_generation_floor IS DISTINCT FROM OLD.scoring_generation_floor THEN
    RAISE EXCEPTION 'lab_arena_score_revision_immutable' USING ERRCODE='42501';
  END IF;
$replace517_2_1$);
 IF (length(d)-length(replace(d,$needle517_2_2$  IF NOT FOUND THEN RETURN NEW; END IF;$needle517_2_2$,'')))<>length($needle517_2_2$  IF NOT FOUND THEN RETURN NEW; END IF;$needle517_2_2$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena_integrity_run_guard_v1/2' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_2_2$  IF NOT FOUND THEN RETURN NEW; END IF;$needle517_2_2$,$replace517_2_2$  IF NOT FOUND THEN RETURN NEW; END IF;
  IF TG_OP='INSERT' THEN
    IF NEW.scoring_generation_floor IS NOT NULL AND NEW.scoring_generation_floor IS DISTINCT FROM
      (CASE WHEN NEW.kind='score' THEN v_round.scoring_generation_floor ELSE 0 END) THEN
      RAISE EXCEPTION 'lab_arena_score_revision_invalid' USING ERRCODE='42501';
    END IF;
    NEW.scoring_generation_floor:=CASE WHEN NEW.kind='score' THEN v_round.scoring_generation_floor ELSE 0 END;
  END IF;$replace517_2_2$);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'dd7fcd3811ea5f85a5437df0a9c77b80bb688ff3a972527949d29816fcf4376d' THEN RAISE EXCEPTION 'Supersession postimage differs: lab_arena_integrity_run_guard_v1' USING ERRCODE='55000'; END IF;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'dd7fcd3811ea5f85a5437df0a9c77b80bb688ff3a972527949d29816fcf4376d' THEN RAISE EXCEPTION 'Supersession readback differs: lab_arena_integrity_run_guard_v1' USING ERRCODE='55000'; END IF;
END;
$patch517_2$;

DO $patch517_3$
DECLARE d TEXT; m JSONB; target OID:=to_regprocedure('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)');
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF d IS NULL THEN RAISE EXCEPTION 'Supersession function missing: lab_arena_claim_assignment' USING ERRCODE='55000'; END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security517$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]$security517$::JSONB THEN RAISE EXCEPTION 'Supersession security differs: lab_arena_claim_assignment' USING ERRCODE='42501'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='4c1c8aa24e6c0f63bdb0ed40fb4cd47266438668445134b017beb5c062845133' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'a1ee8afde0f2f04f6cdd5e9c0cda3d0710c5d9d2b1c72e05e6f604722e7dd345' THEN RAISE EXCEPTION 'Supersession preimage differs: lab_arena_claim_assignment' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,$needle517_3_0$  WHERE runs.round_id = p_round_id
$needle517_3_0$,'')))<>length($needle517_3_0$  WHERE runs.round_id = p_round_id
$needle517_3_0$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena_claim_assignment/0' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_3_0$  WHERE runs.round_id = p_round_id
$needle517_3_0$,$replace517_3_0$  WHERE runs.round_id = p_round_id
    AND (runs.kind<>'score' OR runs.scoring_generation_floor=v_round.scoring_generation_floor)
$replace517_3_0$);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'4c1c8aa24e6c0f63bdb0ed40fb4cd47266438668445134b017beb5c062845133' THEN RAISE EXCEPTION 'Supersession postimage differs: lab_arena_claim_assignment' USING ERRCODE='55000'; END IF;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'4c1c8aa24e6c0f63bdb0ed40fb4cd47266438668445134b017beb5c062845133' THEN RAISE EXCEPTION 'Supersession readback differs: lab_arena_claim_assignment' USING ERRCODE='55000'; END IF;
END;
$patch517_3$;

DO $patch517_4$
DECLARE d TEXT; m JSONB; target OID:=to_regprocedure('public.lab_arena__lock_current_lease(text,text)');
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF d IS NULL THEN RAISE EXCEPTION 'Supersession function missing: lab_arena__lock_current_lease' USING ERRCODE='55000'; END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security517$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]$security517$::JSONB THEN RAISE EXCEPTION 'Supersession security differs: lab_arena__lock_current_lease' USING ERRCODE='42501'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='d4dd945fe53f5aa47ffcc8adb7a682d725081974f767db2c562129ef024eb640' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'19af62e9629d2188642351e02019a67a94956af3bdc19adea8c584b91f47a408' THEN RAISE EXCEPTION 'Supersession preimage differs: lab_arena__lock_current_lease' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,$needle517_4_0$  IF v_run.status <> 'leased'$needle517_4_0$,'')))<>length($needle517_4_0$  IF v_run.status <> 'leased'$needle517_4_0$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena__lock_current_lease/0' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_4_0$  IF v_run.status <> 'leased'$needle517_4_0$,$replace517_4_0$  IF (v_run.kind='score' AND v_run.scoring_generation_floor<>v_round.scoring_generation_floor)
     OR v_run.status <> 'leased'$replace517_4_0$);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'d4dd945fe53f5aa47ffcc8adb7a682d725081974f767db2c562129ef024eb640' THEN RAISE EXCEPTION 'Supersession postimage differs: lab_arena__lock_current_lease' USING ERRCODE='55000'; END IF;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'d4dd945fe53f5aa47ffcc8adb7a682d725081974f767db2c562129ef024eb640' THEN RAISE EXCEPTION 'Supersession readback differs: lab_arena__lock_current_lease' USING ERRCODE='55000'; END IF;
END;
$patch517_4$;

DO $patch517_5$
DECLARE d TEXT; m JSONB; target OID:=to_regprocedure('public.lab_arena__judge_continuation_v1(lab_arena_rounds,lab_arena_runs)');
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF d IS NULL THEN RAISE EXCEPTION 'Supersession function missing: lab_arena__judge_continuation_v1' USING ERRCODE='55000'; END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security517$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]$security517$::JSONB THEN RAISE EXCEPTION 'Supersession security differs: lab_arena__judge_continuation_v1' USING ERRCODE='42501'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='3a0ff0eb9209d66f766daaee4eebab92a81f5f1d05e5d189130d7ec1dc69b0af' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'c514e2a070fb1f5bac1037136834988de75e6140fe78c1549d4e831e39bc901f' THEN RAISE EXCEPTION 'Supersession preimage differs: lab_arena__judge_continuation_v1' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,$needle517_5_0$  IF p_run.kind<>'score' OR$needle517_5_0$,'')))<>length($needle517_5_0$  IF p_run.kind<>'score' OR$needle517_5_0$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena__judge_continuation_v1/0' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_5_0$  IF p_run.kind<>'score' OR$needle517_5_0$,$replace517_5_0$  IF p_run.scoring_generation_floor<>p_round.scoring_generation_floor THEN RETURN FALSE; END IF;
  IF p_run.kind<>'score' OR$replace517_5_0$);
 IF (length(d)-length(replace(d,$needle517_5_1$AND accepted.status='accepted') THEN RETURN FALSE; END IF;$needle517_5_1$,'')))<>length($needle517_5_1$AND accepted.status='accepted') THEN RETURN FALSE; END IF;$needle517_5_1$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena__judge_continuation_v1/1' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_5_1$AND accepted.status='accepted') THEN RETURN FALSE; END IF;$needle517_5_1$,$replace517_5_1$AND accepted.status='accepted'
        AND accepted.scoring_generation_floor=p_round.scoring_generation_floor) THEN RETURN FALSE; END IF;$replace517_5_1$);
 IF (length(d)-length(replace(d,$needle517_5_2$  IF p_run.attempt=1 THEN RETURN TRUE; END IF;$needle517_5_2$,'')))<>length($needle517_5_2$  IF p_run.attempt=1 THEN RETURN TRUE; END IF;$needle517_5_2$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena__judge_continuation_v1/2' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_5_2$  IF p_run.attempt=1 THEN RETURN TRUE; END IF;$needle517_5_2$,$replace517_5_2$  -- The operator-created first attempt of this grading revision is a root;
  -- it need not reinterpret an accepted predecessor under a different model.
  IF p_round.scoring_generation_floor>0
    AND p_run.scoring_generation_floor=p_round.scoring_generation_floor
    AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      WHERE h.round_id=p_round.round_id AND h.generation_floor=p_round.scoring_generation_floor
       AND h.new_round_doc->'configuration_doc'=p_round.configuration_doc)
    AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs earlier
      WHERE earlier.assignment_id=p_run.assignment_id AND earlier.attempt<p_run.attempt
        AND earlier.scoring_generation_floor=p_round.scoring_generation_floor) THEN RETURN TRUE; END IF;
  IF p_run.attempt=1 THEN RETURN TRUE; END IF;$replace517_5_2$);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'3a0ff0eb9209d66f766daaee4eebab92a81f5f1d05e5d189130d7ec1dc69b0af' THEN RAISE EXCEPTION 'Supersession postimage differs: lab_arena__judge_continuation_v1' USING ERRCODE='55000'; END IF;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'3a0ff0eb9209d66f766daaee4eebab92a81f5f1d05e5d189130d7ec1dc69b0af' THEN RAISE EXCEPTION 'Supersession readback differs: lab_arena__judge_continuation_v1' USING ERRCODE='55000'; END IF;
END;
$patch517_5$;

DO $patch517_6$
DECLARE d TEXT; m JSONB; target OID:=to_regprocedure('public.lab_arena_expire_leases(text)');
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF d IS NULL THEN RAISE EXCEPTION 'Supersession function missing: lab_arena_expire_leases' USING ERRCODE='55000'; END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security517$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]$security517$::JSONB THEN RAISE EXCEPTION 'Supersession security differs: lab_arena_expire_leases' USING ERRCODE='42501'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='7a37e5379434e4c111a3136feaa0ad37c739546e326110fda01011a1c5e3ef5e' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'7c70db8f77d1552c315a1ef6ab4d0ad325f0a93850bcb3d302424858680ad002' THEN RAISE EXCEPTION 'Supersession preimage differs: lab_arena_expire_leases' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,$needle517_6_0$WHERE candidate.round_id=p_round_id AND candidate.kind='score'
      AND candidate.status='failed'$needle517_6_0$,'')))<>length($needle517_6_0$WHERE candidate.round_id=p_round_id AND candidate.kind='score'
      AND candidate.status='failed'$needle517_6_0$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena_expire_leases/0' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_6_0$WHERE candidate.round_id=p_round_id AND candidate.kind='score'
      AND candidate.status='failed'$needle517_6_0$,$replace517_6_0$WHERE candidate.round_id=p_round_id AND candidate.kind='score'
      AND candidate.scoring_generation_floor=v_round.scoring_generation_floor
      AND candidate.status='failed'$replace517_6_0$);
 IF (length(d)-length(replace(d,$needle517_6_1$    AND public.lab_arena_miner_credit_failure_v1(pending.run_id)$needle517_6_1$,'')))<>length($needle517_6_1$    AND public.lab_arena_miner_credit_failure_v1(pending.run_id)$needle517_6_1$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena_expire_leases/1' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_6_1$    AND public.lab_arena_miner_credit_failure_v1(pending.run_id)$needle517_6_1$,$replace517_6_1$     AND pending.scoring_generation_floor=v_round.scoring_generation_floor
    AND public.lab_arena_miner_credit_failure_v1(pending.run_id)$replace517_6_1$);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'7a37e5379434e4c111a3136feaa0ad37c739546e326110fda01011a1c5e3ef5e' THEN RAISE EXCEPTION 'Supersession postimage differs: lab_arena_expire_leases' USING ERRCODE='55000'; END IF;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'7a37e5379434e4c111a3136feaa0ad37c739546e326110fda01011a1c5e3ef5e' THEN RAISE EXCEPTION 'Supersession readback differs: lab_arena_expire_leases' USING ERRCODE='55000'; END IF;
END;
$patch517_6$;

DO $patch517_7$
DECLARE d TEXT; m JSONB; target OID:=to_regprocedure('public.lab_arena_close_scoring(text,smallint)');
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF d IS NULL THEN RAISE EXCEPTION 'Supersession function missing: lab_arena_close_scoring' USING ERRCODE='55000'; END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security517$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]$security517$::JSONB THEN RAISE EXCEPTION 'Supersession security differs: lab_arena_close_scoring' USING ERRCODE='42501'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='a07b809f89df827f3e0034e71f01c587b49bf1718715c7b60320e1521f322a6c' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'c1e8f7c9646cc16c32f451456c020859da55c48c389bd27768b42ade794cc090' THEN RAISE EXCEPTION 'Supersession preimage differs: lab_arena_close_scoring' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,$needle517_7_0$  IF v_round.status <> ('stage' || p_stage::TEXT || '_scoring') THEN$needle517_7_0$,'')))<>length($needle517_7_0$  IF v_round.status <> ('stage' || p_stage::TEXT || '_scoring') THEN$needle517_7_0$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena_close_scoring/0' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_7_0$  IF v_round.status <> ('stage' || p_stage::TEXT || '_scoring') THEN$needle517_7_0$,$replace517_7_0$  IF v_round.status <> ('stage' || p_stage::TEXT || '_scoring')
    AND NOT (p_stage=2 AND v_round.status='stage2_judged'
      AND EXISTS(SELECT 1 FROM public.lab_arena_restart_claim_control control
        WHERE control.singleton AND control.operator_paused)
      AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs active
        WHERE active.round_id=p_round_id AND active.status IN ('leased','submitted'))
      AND EXISTS(SELECT 1 FROM public.lab_arena_runs pending
        WHERE pending.round_id=p_round_id AND pending.stage=p_stage AND pending.kind='score'
          AND pending.status='pending'
          AND public.lab_arena__judge_continuation_v1(v_round,pending))) THEN$replace517_7_0$);
 IF (length(d)-length(replace(d,$needle517_7_1$AND status IN ('leased', 'pending', 'submitted')
$needle517_7_1$,'')))<>length($needle517_7_1$AND status IN ('leased', 'pending', 'submitted')
$needle517_7_1$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena_close_scoring/1' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_7_1$AND status IN ('leased', 'pending', 'submitted')
$needle517_7_1$,$replace517_7_1$AND status IN ('leased', 'pending', 'submitted')
      AND (kind<>'score' OR scoring_generation_floor=v_round.scoring_generation_floor)
$replace517_7_1$);
 IF (length(d)-length(replace(d,$needle517_7_2$WHERE runs.round_id = p_round_id AND runs.stage = p_stage AND runs.kind = 'score'$needle517_7_2$,'')))<>length($needle517_7_2$WHERE runs.round_id = p_round_id AND runs.stage = p_stage AND runs.kind = 'score'$needle517_7_2$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena_close_scoring/2' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_7_2$WHERE runs.round_id = p_round_id AND runs.stage = p_stage AND runs.kind = 'score'$needle517_7_2$,$replace517_7_2$WHERE runs.round_id = p_round_id AND runs.stage = p_stage AND runs.kind = 'score'
      AND runs.scoring_generation_floor=v_round.scoring_generation_floor$replace517_7_2$);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'a07b809f89df827f3e0034e71f01c587b49bf1718715c7b60320e1521f322a6c' THEN RAISE EXCEPTION 'Supersession postimage differs: lab_arena_close_scoring' USING ERRCODE='55000'; END IF;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'a07b809f89df827f3e0034e71f01c587b49bf1718715c7b60320e1521f322a6c' THEN RAISE EXCEPTION 'Supersession readback differs: lab_arena_close_scoring' USING ERRCODE='55000'; END IF;
END;
$patch517_7$;

DO $patch517_8$
DECLARE d TEXT; m JSONB; target OID:=to_regprocedure('public.lab_arena__miner_credit_judgment_terminal_v1(text)');
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF d IS NULL THEN RAISE EXCEPTION 'Supersession function missing: lab_arena__miner_credit_judgment_terminal_v1' USING ERRCODE='55000'; END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security517$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"sql"]$security517$::JSONB THEN RAISE EXCEPTION 'Supersession security differs: lab_arena__miner_credit_judgment_terminal_v1' USING ERRCODE='42501'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='fa255113c389c07157a8ff32f8cb8a40272eccf5e5a33b45350171e3ff4ed4f8' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'5baaa33595e7d229d3ff7f9de11a734e265696d96b254fe2ebf85ff3bd7d5af2' THEN RAISE EXCEPTION 'Supersession preimage differs: lab_arena__miner_credit_judgment_terminal_v1' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,$needle517_8_0$AND active.status IN ('pending','leased','submitted'))$needle517_8_0$,'')))<>length($needle517_8_0$AND active.status IN ('pending','leased','submitted'))$needle517_8_0$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena__miner_credit_judgment_terminal_v1/0' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_8_0$AND active.status IN ('pending','leased','submitted'))$needle517_8_0$,$replace517_8_0$AND active.status IN ('pending','leased','submitted')
          AND active.scoring_generation_floor=round_row.scoring_generation_floor)$replace517_8_0$);
 IF (length(d)-length(replace(d,$needle517_8_1$AND judge.stage=source.stage AND judge.icp_position=source.icp_position
$needle517_8_1$,'')))<>length($needle517_8_1$AND judge.stage=source.stage AND judge.icp_position=source.icp_position
$needle517_8_1$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena__miner_credit_judgment_terminal_v1/1' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_8_1$AND judge.stage=source.stage AND judge.icp_position=source.icp_position
$needle517_8_1$,$replace517_8_1$AND judge.stage=source.stage AND judge.icp_position=source.icp_position
        AND judge.scoring_generation_floor=round_row.scoring_generation_floor
$replace517_8_1$);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'fa255113c389c07157a8ff32f8cb8a40272eccf5e5a33b45350171e3ff4ed4f8' THEN RAISE EXCEPTION 'Supersession postimage differs: lab_arena__miner_credit_judgment_terminal_v1' USING ERRCODE='55000'; END IF;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'fa255113c389c07157a8ff32f8cb8a40272eccf5e5a33b45350171e3ff4ed4f8' THEN RAISE EXCEPTION 'Supersession readback differs: lab_arena__miner_credit_judgment_terminal_v1' USING ERRCODE='55000'; END IF;
END;
$patch517_8$;

DO $patch517_9$
DECLARE d TEXT; m JSONB; target OID:=to_regprocedure('public.lab_arena__baseline_final_ready_v1(text)');
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF d IS NULL THEN RAISE EXCEPTION 'Supersession function missing: lab_arena__baseline_final_ready_v1' USING ERRCODE='55000'; END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security517$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]$security517$::JSONB THEN RAISE EXCEPTION 'Supersession security differs: lab_arena__baseline_final_ready_v1' USING ERRCODE='42501'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='637c20c4b2a69741e72921797a78c7c94967e9cfcd8f8aae30d75e5dd0add52f' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'acb8d403ce29dc306db08d43dc65f2aaeefe578177dc9580cc171eb251bae294' THEN RAISE EXCEPTION 'Supersession preimage differs: lab_arena__baseline_final_ready_v1' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,$needle517_9_0$AND judge.kind = 'score' AND judge.scored_run_id = v_execution.run_id
$needle517_9_0$,'')))<>length($needle517_9_0$AND judge.kind = 'score' AND judge.scored_run_id = v_execution.run_id
$needle517_9_0$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena__baseline_final_ready_v1/0' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_9_0$AND judge.kind = 'score' AND judge.scored_run_id = v_execution.run_id
$needle517_9_0$,$replace517_9_0$AND judge.kind = 'score' AND judge.scored_run_id = v_execution.run_id
        AND judge.scoring_generation_floor=v_round.scoring_generation_floor
$replace517_9_0$);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'637c20c4b2a69741e72921797a78c7c94967e9cfcd8f8aae30d75e5dd0add52f' THEN RAISE EXCEPTION 'Supersession postimage differs: lab_arena__baseline_final_ready_v1' USING ERRCODE='55000'; END IF;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'637c20c4b2a69741e72921797a78c7c94967e9cfcd8f8aae30d75e5dd0add52f' THEN RAISE EXCEPTION 'Supersession readback differs: lab_arena__baseline_final_ready_v1' USING ERRCODE='55000'; END IF;
END;
$patch517_9$;

DO $patch517_10$
DECLARE d TEXT; m JSONB; target OID:=to_regprocedure('public.lab_arena__all_judgments_ready_v1(text)');
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF d IS NULL THEN RAISE EXCEPTION 'Supersession function missing: lab_arena__all_judgments_ready_v1' USING ERRCODE='55000'; END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security517$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"sql"]$security517$::JSONB THEN RAISE EXCEPTION 'Supersession security differs: lab_arena__all_judgments_ready_v1' USING ERRCODE='42501'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='54e9f92db1a9000d95e8895a5834bc43970fdcbef975c987334b7df41b39a12c' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'8eba1484e39bfb5961f1f8ae8fdaf8c8fc391063b24367b3c446b4aa8ec9f2fb' THEN RAISE EXCEPTION 'Supersession preimage differs: lab_arena__all_judgments_ready_v1' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,$needle517_10_0$WHERE round_row.round_id=p_round_id
$needle517_10_0$,'')))<>length($needle517_10_0$WHERE round_row.round_id=p_round_id
$needle517_10_0$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena__all_judgments_ready_v1/0' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_10_0$WHERE round_row.round_id=p_round_id
$needle517_10_0$,$replace517_10_0$WHERE round_row.round_id=p_round_id
                AND judge.scoring_generation_floor=round_row.scoring_generation_floor
$replace517_10_0$);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'54e9f92db1a9000d95e8895a5834bc43970fdcbef975c987334b7df41b39a12c' THEN RAISE EXCEPTION 'Supersession postimage differs: lab_arena__all_judgments_ready_v1' USING ERRCODE='55000'; END IF;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'54e9f92db1a9000d95e8895a5834bc43970fdcbef975c987334b7df41b39a12c' THEN RAISE EXCEPTION 'Supersession readback differs: lab_arena__all_judgments_ready_v1' USING ERRCODE='55000'; END IF;
END;
$patch517_10$;

DO $patch517_11$
DECLARE d TEXT; m JSONB; target OID:=to_regprocedure('public.lab_arena_miner_credit_failure_v1(text)');
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF d IS NULL THEN RAISE EXCEPTION 'Supersession function missing: lab_arena_miner_credit_failure_v1' USING ERRCODE='55000'; END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security517$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]$security517$::JSONB THEN RAISE EXCEPTION 'Supersession security differs: lab_arena_miner_credit_failure_v1' USING ERRCODE='42501'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='f81fc633f2a8d25f6633b96e1157e6c1165bdc34833bc6e72ac6734868c5e724' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'1dce9f8faf59183ad4771b8fadfe5d252a59229b25038379b1d0fc66c9933d16' THEN RAISE EXCEPTION 'Supersession preimage differs: lab_arena_miner_credit_failure_v1' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,$needle517_11_0$      OR previous.company_judgment_refs IS DISTINCT FROM r.company_judgment_refs$needle517_11_0$,'')))<>length($needle517_11_0$      OR previous.company_judgment_refs IS DISTINCT FROM r.company_judgment_refs$needle517_11_0$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena_miner_credit_failure_v1/0' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_11_0$      OR previous.company_judgment_refs IS DISTINCT FROM r.company_judgment_refs$needle517_11_0$,$replace517_11_0$      OR (previous.company_judgment_refs IS DISTINCT FROM r.company_judgment_refs
        AND NOT EXISTS(SELECT 1 FROM public.lab_arena_rounds current_round
          JOIN public.lab_arena_scoring_supersessions revision
            ON revision.round_id=current_round.round_id
              AND revision.generation_floor=current_round.scoring_generation_floor
          WHERE current_round.round_id=r.round_id
            AND r.scoring_generation_floor=current_round.scoring_generation_floor
            AND previous.scoring_generation_floor<current_round.scoring_generation_floor
            AND revision.new_round_doc->'configuration_doc'=current_round.configuration_doc
            AND jsonb_set(jsonb_set(revision.old_round_doc->'configuration_doc',
                '{scorer_policy,judge_models,role_batch_check}',
                revision.new_round_doc#>'{configuration_doc,scorer_policy,judge_models,role_batch_check}'),
                '{scorer_policy,env_bindings,ARENA_ROLE_BATCH_MODEL}',
                revision.new_round_doc#>'{configuration_doc,scorer_policy,judge_models,role_batch_check}')
              =revision.new_round_doc->'configuration_doc'))$replace517_11_0$);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'f81fc633f2a8d25f6633b96e1157e6c1165bdc34833bc6e72ac6734868c5e724' THEN RAISE EXCEPTION 'Supersession postimage differs: lab_arena_miner_credit_failure_v1' USING ERRCODE='55000'; END IF;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'f81fc633f2a8d25f6633b96e1157e6c1165bdc34833bc6e72ac6734868c5e724' THEN RAISE EXCEPTION 'Supersession readback differs: lab_arena_miner_credit_failure_v1' USING ERRCODE='55000'; END IF;
END;
$patch517_11$;

DO $patch517_12$
DECLARE d TEXT; m JSONB; target OID:=to_regprocedure('public.lab_arena_record_run_scores(text,smallint,jsonb)');
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF d IS NULL THEN RAISE EXCEPTION 'Supersession function missing: lab_arena_record_run_scores' USING ERRCODE='55000'; END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security517$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]$security517$::JSONB THEN RAISE EXCEPTION 'Supersession security differs: lab_arena_record_run_scores' USING ERRCODE='42501'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='678461f0d511beb5714fd05fdadd515b25af0aa5010a6059c6b228242dfda6e8' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'c7a47916a4fc909440c09de30f408259901eb06c968cdb1079e2d773e92500d1' THEN RAISE EXCEPTION 'Supersession preimage differs: lab_arena_record_run_scores' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,$needle517_12_0$    v_value := (v_score ->> 'per_icp_score')::NUMERIC(12, 6);$needle517_12_0$,'')))<>length($needle517_12_0$    v_value := (v_score ->> 'per_icp_score')::NUMERIC(12, 6);$needle517_12_0$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena_record_run_scores/0' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_12_0$    v_value := (v_score ->> 'per_icp_score')::NUMERIC(12, 6);$needle517_12_0$,$replace517_12_0$    IF coalesce(v_score->'scoring_generation_floor','0'::JSONB)
      IS DISTINCT FROM to_jsonb(v_round.scoring_generation_floor) THEN
      RAISE EXCEPTION 'lab_arena_superseded_score_authority' USING ERRCODE='42501';
    END IF;
    v_value := (v_score ->> 'per_icp_score')::NUMERIC(12, 6);$replace517_12_0$);
 IF (length(d)-length(replace(d,$needle517_12_1$    IF v_late_judge AND ($needle517_12_1$,'')))<>length($needle517_12_1$    IF v_late_judge AND ($needle517_12_1$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena_record_run_scores/1' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_12_1$    IF v_late_judge AND ($needle517_12_1$,$replace517_12_1$    IF v_round.scoring_generation_floor>0 AND v_run.status='accepted'
      AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs judge
        WHERE judge.round_id=v_run.round_id AND judge.kind='score'
          AND judge.scored_run_id=v_run.run_id
          AND judge.scoring_generation_floor=v_round.scoring_generation_floor
          AND judge.status='accepted' AND judge.terminal_cause='accepted'
          AND judge.output_ref IS NOT NULL
          AND (judge.submission_id,judge.miner_hotkey,judge.stage,judge.icp_position)
            IS NOT DISTINCT FROM (v_run.submission_id,v_run.miner_hotkey,v_run.stage,v_run.icp_position)
          AND public.lab_arena__company_judgment_refs_v1(v_round,judge.company_judgment_refs))
      AND NOT (v_value=0 AND public.lab_arena__miner_credit_judgment_terminal_v1(v_run.run_id)) THEN
      RAISE EXCEPTION 'lab_arena_superseded_score_authority' USING ERRCODE='42501';
    END IF;
    IF v_late_judge AND v_round.scoring_generation_floor=0 AND ($replace517_12_1$);
 IF (length(d)-length(replace(d,$needle517_12_2$AND judged.kind='score' AND judged.status='accepted' AND judged.terminal_cause='accepted'
$needle517_12_2$,'')))<>length($needle517_12_2$AND judged.kind='score' AND judged.status='accepted' AND judged.terminal_cause='accepted'
$needle517_12_2$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena_record_run_scores/2' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_12_2$AND judged.kind='score' AND judged.status='accepted' AND judged.terminal_cause='accepted'
$needle517_12_2$,$replace517_12_2$AND judged.kind='score' AND judged.status='accepted' AND judged.terminal_cause='accepted'
          AND judged.scoring_generation_floor=v_round.scoring_generation_floor
$replace517_12_2$);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'678461f0d511beb5714fd05fdadd515b25af0aa5010a6059c6b228242dfda6e8' THEN RAISE EXCEPTION 'Supersession postimage differs: lab_arena_record_run_scores' USING ERRCODE='55000'; END IF;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'678461f0d511beb5714fd05fdadd515b25af0aa5010a6059c6b228242dfda6e8' THEN RAISE EXCEPTION 'Supersession readback differs: lab_arena_record_run_scores' USING ERRCODE='55000'; END IF;
END;
$patch517_12$;

DO $patch517_13$
DECLARE d TEXT; m JSONB; target OID:=to_regprocedure('public.lab_arena__publication_execution_incomplete_v1(text,text)');
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF d IS NULL THEN RAISE EXCEPTION 'Supersession function missing: lab_arena__publication_execution_incomplete_v1' USING ERRCODE='55000'; END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM $security517$["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]$security517$::JSONB THEN RAISE EXCEPTION 'Supersession security differs: lab_arena__publication_execution_incomplete_v1' USING ERRCODE='42501'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='5a276ceaafe5f9de1da40d4aae39df36f5f5e17c0e9157375216e4b88c13ac72' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'ac255ffd6bbcd2c28d8c0c9421b9c13cdaa87e477d55a34c5f48032a1b8a4458' THEN RAISE EXCEPTION 'Supersession preimage differs: lab_arena__publication_execution_incomplete_v1' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,$needle517_13_0$AND judge.kind = 'score'
        AND judge.round_id$needle517_13_0$,'')))<>length($needle517_13_0$AND judge.kind = 'score'
        AND judge.round_id$needle517_13_0$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena__publication_execution_incomplete_v1/0' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_13_0$AND judge.kind = 'score'
        AND judge.round_id$needle517_13_0$,$replace517_13_0$AND judge.kind = 'score'
        AND judge.scoring_generation_floor=v_round.scoring_generation_floor
        AND judge.round_id$replace517_13_0$);
 IF (length(d)-length(replace(d,$needle517_13_1$AND judge.kind = 'score'
            AND judge.scored_run_id$needle517_13_1$,'')))<>length($needle517_13_1$AND judge.kind = 'score'
            AND judge.scored_run_id$needle517_13_1$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena__publication_execution_incomplete_v1/1' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_13_1$AND judge.kind = 'score'
            AND judge.scored_run_id$needle517_13_1$,$replace517_13_1$AND judge.kind = 'score'
            AND judge.scoring_generation_floor=v_round.scoring_generation_floor
            AND judge.scored_run_id$replace517_13_1$);
 IF (length(d)-length(replace(d,$needle517_13_2$AND active.kind = 'score'
$needle517_13_2$,'')))<>length($needle517_13_2$AND active.kind = 'score'
$needle517_13_2$) THEN RAISE EXCEPTION 'Supersession anchor differs: lab_arena__publication_execution_incomplete_v1/2' USING ERRCODE='55000'; END IF;
 d:=replace(d,$needle517_13_2$AND active.kind = 'score'
$needle517_13_2$,$replace517_13_2$AND active.kind = 'score'
              AND active.scoring_generation_floor=v_round.scoring_generation_floor
$replace517_13_2$);
 IF encode(extensions.digest(d,'sha256'),'hex')<>'5a276ceaafe5f9de1da40d4aae39df36f5f5e17c0e9157375216e4b88c13ac72' THEN RAISE EXCEPTION 'Supersession postimage differs: lab_arena__publication_execution_incomplete_v1' USING ERRCODE='55000'; END IF;
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'5a276ceaafe5f9de1da40d4aae39df36f5f5e17c0e9157375216e4b88c13ac72' THEN RAISE EXCEPTION 'Supersession readback differs: lab_arena__publication_execution_incomplete_v1' USING ERRCODE='55000'; END IF;
END;
$patch517_13$;

DO $installed517$
DECLARE t OID:=to_regclass('public.lab_arena_scoring_supersessions');
 f OID:=to_regprocedure('public.lab_arena_supersede_role_model_v1(text,jsonb,bigint,bigint,text,jsonb,text,text)');
 reader OID:=to_regprocedure('public.lab_arena_scoring_policy_history_v1(text)');
 hold_fn OID:=to_regprocedure('public.lab_arena_scoring_operator_hold_v1(text,jsonb,bigint,bigint,bigint,text,text,boolean)');
BEGIN
 IF t IS NOT NULL AND EXISTS(SELECT 1 FROM pg_class c WHERE c.oid=t AND (
   pg_get_userbyid(c.relowner)<>'lab_arena_owner' OR NOT c.relrowsecurity
   OR c.relforcerowsecurity OR c.reloptions IS NOT NULL
   OR EXISTS(SELECT 1 FROM pg_policy policy WHERE policy.polrelid=c.oid)
   OR NOT EXISTS(SELECT 1 FROM pg_trigger trigger WHERE trigger.tgrelid=c.oid AND NOT trigger.tgisinternal
     AND trigger.tgname='lab_arena_scoring_supersessions_append_only'
     AND trigger.tgfoid=to_regprocedure('public.lab_arena_append_only_v1()')
     AND trigger.tgenabled='O' AND trigger.tgtype=27)
   OR EXISTS(SELECT 1 FROM pg_trigger trigger WHERE trigger.tgrelid=c.oid AND NOT trigger.tgisinternal
     AND trigger.tgname<>'lab_arena_scoring_supersessions_append_only')
   OR EXISTS(SELECT 1 FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner))) a
     WHERE a.grantee<>c.relowner AND NOT (a.grantee=to_regrole('lab_arena_service')
       AND a.privilege_type='SELECT' AND NOT a.is_grantable)))) THEN
  RAISE EXCEPTION 'Supersession history security differs' USING ERRCODE='42501';
 END IF;
 IF f IS NOT NULL AND EXISTS(SELECT 1 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
   WHERE p.oid=f AND (pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef
    OR p.provolatile<>'v' OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']
    OR l.lanname<>'plpgsql'
    OR EXISTS(SELECT 1 FROM aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a
      WHERE a.grantee<>p.proowner))) THEN
  RAISE EXCEPTION 'Supersession operator security differs' USING ERRCODE='42501';
 END IF;
 IF reader IS NOT NULL AND (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,
   p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
   WHERE p.oid=reader) IS DISTINCT FROM
   '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"sql"]'::JSONB THEN
  RAISE EXCEPTION 'Supersession policy reader security differs' USING ERRCODE='42501';
 END IF;
 IF hold_fn IS NOT NULL AND (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,
   p.prosecdef,p.provolatile,p.proconfig,l.lanname) FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
   WHERE p.oid=hold_fn) IS DISTINCT FROM
   '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
  RAISE EXCEPTION 'Supersession operator hold security differs' USING ERRCODE='42501';
 END IF;
END;
$installed517$;
-- The public results view keeps its existing columns and appends one public
-- authority boundary; no private configuration or historical payload is added.
DO $view517$
DECLARE d TEXT;
BEGIN
 SELECT pg_get_viewdef('public.lab_arena_published_results_v1'::regclass,TRUE) INTO d;
 IF NOT EXISTS(SELECT 1 FROM pg_attribute WHERE attrelid='public.lab_arena_published_results_v1'::regclass
   AND attname='scoring_generation_floor' AND NOT attisdropped) THEN
  d:=rtrim(rtrim(d),';');
  EXECUTE 'CREATE OR REPLACE VIEW public.lab_arena_published_results_v1 WITH (security_invoker=true) AS SELECT prior.*, current_round.scoring_generation_floor FROM ('||d||') prior JOIN public.lab_arena_rounds current_round USING(round_id)';
 END IF;
END;
$view517$;
DO $security517$
BEGIN
 IF (SELECT jsonb_build_array(pg_get_userbyid(c.relowner),(SELECT jsonb_agg(jsonb_build_array(pg_get_userbyid(a.grantee),a.privilege_type,a.is_grantable,pg_get_userbyid(a.grantor)) ORDER BY pg_get_userbyid(a.grantee),a.privilege_type) FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner)))a WHERE a.grantee<>c.relowner),c.relrowsecurity,c.relforcerowsecurity,c.reloptions)
   FROM pg_class c WHERE c.oid=to_regclass('public.lab_arena_published_results_v1'))
   IS DISTINCT FROM $relation517$["lab_arena_owner",[["lab_arena_service","SELECT",false,"lab_arena_owner"],["leadpoet_parity_reader","SELECT",false,"lab_arena_owner"]],false,false,["security_invoker=true"]]$relation517$::JSONB THEN
  RAISE EXCEPTION 'Supersession relation security differs: lab_arena_published_results_v1' USING ERRCODE='42501';
 END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(c.relowner),(SELECT jsonb_agg(jsonb_build_array(pg_get_userbyid(a.grantee),a.privilege_type,a.is_grantable,pg_get_userbyid(a.grantor)) ORDER BY pg_get_userbyid(a.grantee),a.privilege_type) FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner)))a WHERE a.grantee<>c.relowner),c.relrowsecurity,c.relforcerowsecurity,c.reloptions)
   FROM pg_class c WHERE c.oid=to_regclass('public.lab_arena_restart_claim_control'))
   IS DISTINCT FROM $relation517$["lab_arena_owner",[["leadpoet_parity_reader","SELECT",false,"lab_arena_owner"]],true,false,null]$relation517$::JSONB THEN
  RAISE EXCEPTION 'Supersession relation security differs: lab_arena_restart_claim_control' USING ERRCODE='42501';
 END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(c.relowner),(SELECT jsonb_agg(jsonb_build_array(pg_get_userbyid(a.grantee),a.privilege_type,a.is_grantable,pg_get_userbyid(a.grantor)) ORDER BY pg_get_userbyid(a.grantee),a.privilege_type) FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner)))a WHERE a.grantee<>c.relowner),c.relrowsecurity,c.relforcerowsecurity,c.reloptions)
   FROM pg_class c WHERE c.oid=to_regclass('public.lab_arena_rounds'))
   IS DISTINCT FROM $relation517$["lab_arena_owner",[["lab_arena_service","SELECT",false,"lab_arena_owner"],["leadpoet_parity_reader","SELECT",false,"lab_arena_owner"]],true,false,null]$relation517$::JSONB THEN
  RAISE EXCEPTION 'Supersession relation security differs: lab_arena_rounds' USING ERRCODE='42501';
 END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(c.relowner),(SELECT jsonb_agg(jsonb_build_array(pg_get_userbyid(a.grantee),a.privilege_type,a.is_grantable,pg_get_userbyid(a.grantor)) ORDER BY pg_get_userbyid(a.grantee),a.privilege_type) FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner)))a WHERE a.grantee<>c.relowner),c.relrowsecurity,c.relforcerowsecurity,c.reloptions)
   FROM pg_class c WHERE c.oid=to_regclass('public.lab_arena_runs'))
   IS DISTINCT FROM $relation517$["lab_arena_owner",[["lab_arena_service","SELECT",false,"lab_arena_owner"],["leadpoet_parity_reader","SELECT",false,"lab_arena_owner"]],true,false,null]$relation517$::JSONB THEN
  RAISE EXCEPTION 'Supersession relation security differs: lab_arena_runs' USING ERRCODE='42501';
 END IF;
END;
$security517$;
NOTIFY pgrst,'reload schema';
COMMIT;
