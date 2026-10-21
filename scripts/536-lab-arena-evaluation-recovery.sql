-- Owner-controlled fair continuation. Original sourcing, judgments and bills stay put.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='120s';
DO $schema_create536$
DECLARE before_doc JSONB;
BEGIN
 SELECT jsonb_build_object('oid',oid,'owner',nspowner,'acl',to_jsonb(nspacl),
   'owner_had_create',has_schema_privilege('lab_arena_owner',oid,'CREATE'))
 INTO before_doc FROM pg_namespace WHERE nspname='public';
 IF before_doc IS NULL THEN
  RAISE EXCEPTION 'Supersession public schema missing' USING ERRCODE='55000';
 END IF;
 PERFORM set_config('leadpoet.arena536_public_schema_before',before_doc::TEXT,TRUE);
 IF NOT (before_doc->>'owner_had_create')::BOOLEAN THEN
  GRANT CREATE ON SCHEMA public TO lab_arena_owner;
 END IF;
END;
$schema_create536$;

DO $recovery_relations536$
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
$recovery_relations536$;
DO $recovery_history536$
DECLARE t OID:='public.lab_arena_scoring_supersessions'::regclass;
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
END;
$recovery_history536$;
ALTER TABLE public.lab_arena_scoring_supersessions ADD COLUMN IF NOT EXISTS recovery_doc JSONB;

DO $security536$
DECLARE wanted RECORD; actual RECORD;
BEGIN
 FOR wanted IN SELECT * FROM (VALUES
  ('public.lab_arena__queue_recovery_judges_v1(public.lab_arena_rounds,jsonb)','0c9fc6da3e617f2ba0e4b5b58a7c2a0e645c1063b2ed408fca6bd7d2f258323c','v','{lab_arena_owner=X/lab_arena_owner}'),
  ('public.lab_arena_recover_evaluation_v1(text,jsonb,bigint,bigint,jsonb,jsonb,jsonb,text,text)','e0227ef3a287f815fe80cb4b46d72bef9fdaab2d93b062d40def5ea28184b65f','v','{lab_arena_owner=X/lab_arena_owner}'),
  ('public.lab_arena_scoring_policy_history_v1(text)','28339cee61add9965be96443569ff9f9aa6d7ebdfedbb6f17c023326bd1226c7','s','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}')
 ) v(signature,source_hash,volatility,acl) LOOP
  SELECT pg_get_userbyid(p.proowner) AS owner,p.proacl::TEXT AS acl,p.prosecdef,p.provolatile,
    p.proconfig,l.lanname,encode(extensions.digest(p.prosrc,'sha256'),'hex') AS source_hash
    INTO actual FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
    WHERE p.oid=to_regprocedure(wanted.signature);
  IF NOT FOUND THEN CONTINUE; END IF;
  IF actual.owner<>'lab_arena_owner' OR actual.acl IS DISTINCT FROM wanted.acl
    OR actual.prosecdef IS DISTINCT FROM TRUE OR actual.provolatile::TEXT<>wanted.volatility
    OR actual.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
    OR actual.lanname<>(CASE WHEN wanted.signature='public.lab_arena_scoring_policy_history_v1(text)' THEN 'sql' ELSE 'plpgsql' END)
    OR (actual.source_hash<>wanted.source_hash AND NOT (
      wanted.signature='public.lab_arena_scoring_policy_history_v1(text)'
      AND encode(extensions.digest(pg_get_functiondef(to_regprocedure(wanted.signature)),'sha256'),'hex')=
        '3429c3cecbbc51053a8716ac5db62919d288f611fede8fc828eacaf734ece527')) THEN
    RAISE EXCEPTION 'recovery536 function security or source differs: %',wanted.signature USING ERRCODE='55000';
  END IF;
 END LOOP;
END;
$security536$;

-- This queue is internal to the existing initializer and controlled recovery.
CREATE OR REPLACE FUNCTION public.lab_arena__queue_recovery_judges_v1(
 p_round public.lab_arena_rounds,p_work_items JSONB)
RETURNS INTEGER LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $queue$
DECLARE item JSONB; source public.lab_arena_runs; existing public.lab_arena_runs;
 next_attempt INTEGER; queued INTEGER:=0;
BEGIN
 IF jsonb_typeof(p_work_items) IS DISTINCT FROM 'array'
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
    WHERE h.round_id=p_round.round_id AND h.generation_floor=p_round.scoring_generation_floor
      AND h.recovery_doc->>'operation'='cancelled_evaluation_recovery_v1'
      AND h.new_round_doc->'configuration_doc'=p_round.configuration_doc) THEN
  RAISE EXCEPTION 'recovery judge authority missing' USING ERRCODE='42501'; END IF;
 FOR item IN SELECT value FROM jsonb_array_elements(p_work_items) LOOP
  IF jsonb_typeof(item) IS DISTINCT FROM 'object'
   OR (SELECT array_agg(k ORDER BY k) FROM jsonb_object_keys(item)k) IS DISTINCT FROM
      ARRAY['company_judgment_refs','icp_position','output_ref','scored_run_id','submission_id']::TEXT[]
   OR NOT public.lab_arena__company_judgment_refs_v1(p_round,item->'company_judgment_refs') THEN
   RAISE EXCEPTION 'recovery judge input invalid' USING ERRCODE='22023'; END IF;
  SELECT * INTO source FROM public.lab_arena_runs a WHERE a.run_id=item->>'scored_run_id'
    AND a.round_id=p_round.round_id AND a.kind='execute' AND a.status='accepted'
    AND a.terminal_cause='accepted' AND a.output_ref=item->>'output_ref'
    AND a.submission_id=item->>'submission_id' AND a.icp_position=(item->>'icp_position')::INTEGER;
  IF NOT FOUND OR NOT EXISTS(SELECT 1 FROM jsonb_array_elements(p_round.participants)p
    WHERE p->>'submission_id'=source.submission_id AND p->>'miner_hotkey'=source.miner_hotkey) THEN
   RAISE EXCEPTION 'recovery source invalid' USING ERRCODE='22023'; END IF;
  -- During execution only the preserved baseline is judged. The ordinary
  -- close_stage transition closes pending miner judgments, so v3 queues those
  -- after its normal stage2_closed boundary from the saved validated inputs.
  IF p_round.status='stage2' AND source.stage=2 THEN CONTINUE; END IF;
  SELECT * INTO existing FROM public.lab_arena_runs a
   WHERE a.assignment_id=source.assignment_id||':score'
     AND a.scoring_generation_floor=p_round.scoring_generation_floor ORDER BY attempt DESC LIMIT 1;
  IF FOUND THEN
   IF existing.scored_run_id IS DISTINCT FROM source.run_id
    OR existing.company_judgment_refs IS DISTINCT FROM item->'company_judgment_refs' THEN
    RAISE EXCEPTION 'recovery judge input conflict' USING ERRCODE='22023'; END IF;
   CONTINUE;
  END IF;
  SELECT coalesce(max(attempt),0)+1 INTO next_attempt FROM public.lab_arena_runs
   WHERE assignment_id=source.assignment_id||':score';
  INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,
    stage,icp_position,attempt,status,stage_generation,kind,scored_run_id,company_judgment_refs)
  VALUES(source.assignment_id||':score:'||next_attempt,source.assignment_id||':score',
    source.round_id,source.submission_id,source.miner_hotkey,source.stage,source.icp_position,
    next_attempt,'pending',p_round.stage_generation,'score',source.run_id,item->'company_judgment_refs');
  queued:=queued+1;
 END LOOP;
 RETURN queued;
END;
$queue$;
ALTER FUNCTION public.lab_arena__queue_recovery_judges_v1(public.lab_arena_rounds,JSONB) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__queue_recovery_judges_v1(public.lab_arena_rounds,JSONB)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_recover_evaluation_v1(
 p_round_id TEXT,p_expected_configuration JSONB,p_expected_status_generation BIGINT,
 p_expected_stage_generation BIGINT,p_new_configuration JSONB,p_work_items JSONB,
 p_excluded_participants JSONB,p_request_id TEXT,p_actor_ref TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $recover$
DECLARE r public.lab_arena_rounds; n public.lab_arena_rounds; c public.lab_arena_restart_claim_control;
 h public.lab_arena_scoring_supersessions; e JSONB; participant JSONB; projections JSONB;
 source public.lab_arena_runs; issuance public.lab_arena_ledger; terminal public.lab_arena_ledger;
 body JSONB; excluded JSONB:='[]'; keep JSONB; request_hash TEXT; execute_count INTEGER:=0;
 queued INTEGER; count_sources INTEGER; start_at TIMESTAMPTZ:=clock_timestamp();
BEGIN
 IF jsonb_typeof(p_expected_configuration) IS DISTINCT FROM 'object'
  OR jsonb_typeof(p_new_configuration) IS DISTINCT FROM 'object'
  OR jsonb_typeof(p_work_items) IS DISTINCT FROM 'array'
  OR jsonb_typeof(p_excluded_participants) IS DISTINCT FROM 'array'
  OR char_length(coalesce(p_request_id,'')) NOT BETWEEN 16 AND 128
  OR char_length(coalesce(p_actor_ref,'')) NOT BETWEEN 1 AND 128 THEN
  RAISE EXCEPTION 'recovery input invalid' USING ERRCODE='22023'; END IF;
 request_hash:='sha256:'||encode(extensions.digest(jsonb_build_object(
  'configuration',p_expected_configuration,'status_generation',p_expected_status_generation,
  'stage_generation',p_expected_stage_generation,'new_configuration',p_new_configuration,
  'work_items',p_work_items,'excluded_participants',p_excluded_participants,'actor_ref',p_actor_ref)::TEXT,'sha256'),'hex');
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO r FROM public.lab_arena_rounds WHERE round_id=p_round_id FOR UPDATE;
 IF NOT FOUND THEN RAISE EXCEPTION 'recovery round missing' USING ERRCODE='P0002'; END IF;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions WHERE round_id=p_round_id AND request_id=p_request_id;
 IF FOUND THEN
  IF h.request_hash IS DISTINCT FROM request_hash THEN
   RAISE EXCEPTION 'recovery request conflict' USING ERRCODE='22023'; END IF;
  RETURN jsonb_build_object('status','existing','generation_floor',h.generation_floor);
 END IF;
 SELECT * INTO c FROM public.lab_arena_restart_claim_control WHERE singleton FOR SHARE;
 IF c.singleton IS NULL OR NOT c.operator_paused OR c.actor_ref IS DISTINCT FROM p_actor_ref
  OR coalesce(c.guard_commitment,'')<>'' OR coalesce(c.owner_commitment,'')<>''
  OR c.guard_expires_at IS NOT NULL OR coalesce(c.candidate_commit,'')<>''
  OR coalesce(c.restart_scope,'')<>'' OR coalesce(c.restart_phase,'')<>''
  OR coalesce(c.captured_leases,'[]'::JSONB)<>'[]'::JSONB
  OR r.status NOT IN ('cancelled','stage2_scoring','stage2_judged','scored')
  OR (r.status='cancelled' AND r.cancel_reason IS DISTINCT FROM 'scoring_incomplete')
  OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL
  OR r.reward_activated_at IS NOT NULL OR r.reward_basis_hash IS NOT NULL
  OR r.reward_basis_doc IS NOT NULL OR r.effective_reward_epoch IS NOT NULL
  OR r.promotion_doc IS NOT NULL OR r.baseline_promoted_at IS NOT NULL
  OR r.configuration_doc IS DISTINCT FROM p_expected_configuration
  OR r.status_generation IS DISTINCT FROM p_expected_status_generation
  OR r.stage_generation IS DISTINCT FROM p_expected_stage_generation
  OR r.benchmark_ref IS NULL OR r.stage1_scoring_plan_doc IS NULL
  OR r.stage2_scoring_plan_doc IS NULL
  OR r.configuration_doc->>'execution_sequence_policy' IS DISTINCT FROM 'baseline_scored_first_v1'
  OR public.lab_arena__company_judgment_policy_v1(r) IS NOT TRUE
  OR public.lab_arena_judge_continuation_enabled_v1(r.round_id) IS NOT TRUE THEN
  RAISE EXCEPTION 'recovery precondition failed' USING ERRCODE='55000'; END IF;
 IF (p_new_configuration-ARRAY['scorer_image_digest','scorer_image_reference','schedule'])
    IS DISTINCT FROM (r.configuration_doc-ARRAY['scorer_image_digest','scorer_image_reference','schedule'])
  OR ((p_new_configuration->'schedule')-ARRAY['stage_2_close','final_scoring_close','publication_deadline'])
    IS DISTINCT FROM ((r.configuration_doc->'schedule')-ARRAY['stage_2_close','final_scoring_close','publication_deadline'])
  OR coalesce(p_new_configuration->>'scorer_image_digest','') !~ '^sha256:[0-9a-f]{64}$'
  OR p_new_configuration->>'scorer_image_digest'=r.configuration_doc->>'scorer_image_digest'
  OR p_new_configuration->>'scorer_image_reference' IS DISTINCT FROM
    regexp_replace(r.configuration_doc->>'scorer_image_reference','sha256:[0-9a-f]{64}$',p_new_configuration->>'scorer_image_digest')
  OR (start_at<(p_new_configuration#>>'{schedule,stage_2_close}')::TIMESTAMPTZ
    AND (p_new_configuration#>>'{schedule,stage_2_close}')::TIMESTAMPTZ<(p_new_configuration#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ
    AND (p_new_configuration#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ<(p_new_configuration#>>'{schedule,publication_deadline}')::TIMESTAMPTZ
    AND (p_new_configuration#>>'{schedule,publication_deadline}')::TIMESTAMPTZ<=start_at+interval '24 hours') IS NOT TRUE THEN
  RAISE EXCEPTION 'recovery configuration invalid' USING ERRCODE='22023'; END IF;
 IF EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=p_round_id AND a.status IN('pending','leased','submitted'))
  OR EXISTS(SELECT 1 FROM (SELECT DISTINCT assignment_id FROM public.lab_arena_runs WHERE round_id=p_round_id)a
    WHERE NOT public.lab_arena__judge_calls_clear_v1(a.assignment_id))
  OR EXISTS(SELECT 1 FROM public.lab_arena_ledger l WHERE l.round_id=p_round_id AND l.call_identity IS NOT NULL
    AND l.entry_kind IN('reservation','dispatch') AND NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger later
      WHERE later.call_identity=l.call_identity AND later.entry_id>l.entry_id)) THEN
  RAISE EXCEPTION 'recovery not drained' USING ERRCODE='55000'; END IF;
 FOR e IN SELECT value FROM jsonb_array_elements(p_excluded_participants) LOOP
  SELECT value INTO participant FROM jsonb_array_elements(r.participants)p(value)
   WHERE value->>'submission_id'=e->>'submission_id' AND NOT coalesce((value->>'is_king')::BOOLEAN,FALSE)
     AND NOT coalesce((value->>'is_baseline')::BOOLEAN,FALSE);
  IF participant IS NULL OR EXISTS(SELECT 1 FROM jsonb_array_elements(excluded)x
     WHERE x->>'submission_id'=e->>'submission_id') OR e->'accepted_empty' IS DISTINCT FROM 'true'::JSONB
    OR e->'current_route_unavailable' IS DISTINCT FROM 'true'::JSONB
    OR coalesce(e->>'output_sha256','') !~ '^sha256:[0-9a-f]{64}$'
    OR coalesce(e->>'model','') !~ '^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._:-]{0,119}$' THEN
   RAISE EXCEPTION 'recovery exclusion invalid' USING ERRCODE='22023'; END IF;
  SELECT * INTO source FROM public.lab_arena_runs a WHERE a.run_id=e->>'run_id' AND a.round_id=p_round_id
    AND a.submission_id=e->>'submission_id' AND a.kind='execute' AND a.status='accepted'
    AND a.terminal_cause='accepted' AND a.output_ref=e->>'output_ref';
  IF NOT FOUND THEN RAISE EXCEPTION 'recovery exclusion source invalid' USING ERRCODE='22023'; END IF;
  SELECT * INTO issuance FROM public.lab_arena_ledger l WHERE l.run_id=source.run_id
   AND l.provider='openrouter' AND l.entry_kind='reservation'
   AND l.operation_id IN('openrouter.chat','openrouter.responses')
   ORDER BY (l.entry_doc->>'action_sequence')::BIGINT DESC,l.entry_id DESC LIMIT 1;
  SELECT * INTO terminal FROM public.lab_arena_ledger l WHERE l.entry_id=(e->>'settlement_entry_id')::BIGINT
    AND l.run_id=source.run_id AND l.round_id=p_round_id AND l.call_identity=issuance.call_identity
    AND l.provider='openrouter' AND l.operation_id=issuance.operation_id
    AND l.funding_source=issuance.funding_source AND l.entry_kind='settlement';
  IF NOT FOUND OR issuance.call_identity IS NULL OR terminal.terminal_response IS DISTINCT FROM e->'terminal_response'
    OR terminal.terminal_response->'status' IS DISTINCT FROM '404'::JSONB
    OR terminal.terminal_response->'call_succeeded' IS DISTINCT FROM 'false'::JSONB
    OR (SELECT count(*) FROM public.lab_arena_ledger l WHERE l.call_identity=issuance.call_identity)<>3
    OR NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger l WHERE l.call_identity=issuance.call_identity
      AND l.run_id=source.run_id AND l.round_id=p_round_id AND l.entry_kind='dispatch'
      AND l.provider='openrouter' AND l.operation_id=issuance.operation_id AND l.funding_source=issuance.funding_source
      AND issuance.entry_id<l.entry_id AND l.entry_id<terminal.entry_id) THEN
   RAISE EXCEPTION 'recovery exclusion receipt invalid' USING ERRCODE='22023'; END IF;
  body:=convert_from(decode(terminal.terminal_response->>'body_b64','base64'),'UTF8')::JSONB;
  IF body#>'{error,code}' IS DISTINCT FROM '404'::JSONB OR coalesce(body#>>'{error,message}','') NOT IN(
    'No endpoints found matching your data policy.',
    'No endpoints found matching your data policy (Zero data retention).',
    'No endpoints found matching your data policy. Configure: https://openrouter.ai/settings/privacy',
    'No endpoints found matching your data policy (Zero data retention). Configure: https://openrouter.ai/settings/privacy') THEN
   RAISE EXCEPTION 'recovery exclusion is not a native privacy route failure' USING ERRCODE='22023'; END IF;
  excluded:=excluded||jsonb_build_array(participant||jsonb_build_object('status','provider_blocked',
    'evaluation','not_evaluated','reason','openrouter_zdr_route_unavailable'));
 END LOOP;
 SELECT jsonb_agg(p.value ORDER BY p.ordinal) INTO keep FROM jsonb_array_elements(r.participants) WITH ORDINALITY p(value,ordinal)
  WHERE NOT EXISTS(SELECT 1 FROM jsonb_array_elements(excluded)removed WHERE removed->>'submission_id'=p.value->>'submission_id');
 IF (SELECT count(*) FROM jsonb_array_elements(keep)p WHERE coalesce((p->>'is_king')::BOOLEAN,FALSE))<>1
   OR NOT EXISTS(SELECT 1 FROM jsonb_array_elements(keep)p WHERE NOT coalesce((p->>'is_king')::BOOLEAN,FALSE)) THEN
  RAISE EXCEPTION 'recovery has no retained miners' USING ERRCODE='22023'; END IF;
 n:=r; n.configuration_doc:=p_new_configuration; n.participants:=keep; n.status:='stage2'; n.cancel_reason:=NULL;
 n.stage_generation:=r.stage_generation+1; n.status_generation:=r.status_generation+1;
 n.scoring_generation_floor:=n.stage_generation; n.stage2_scoring_plan_doc:=NULL;
 SELECT coalesce(jsonb_agg(p->>'submission_id' ORDER BY p->>'submission_id'),'[]') INTO n.finalists
  FROM jsonb_array_elements(keep)p WHERE NOT coalesce((p->>'is_king')::BOOLEAN,FALSE);
 SELECT count(*) INTO count_sources FROM public.lab_arena_runs a WHERE a.round_id=p_round_id AND a.kind='execute'
  AND a.status='accepted' AND EXISTS(SELECT 1 FROM jsonb_array_elements(keep)p WHERE p->>'submission_id'=a.submission_id);
 IF count_sources<>jsonb_array_length(p_work_items) OR count_sources<>(SELECT count(DISTINCT x->>'scored_run_id') FROM jsonb_array_elements(p_work_items)x) THEN
  RAISE EXCEPTION 'recovery work incomplete' USING ERRCODE='22023'; END IF;
 SELECT coalesce(jsonb_object_agg(a.run_id,jsonb_build_object('per_icp_score',a.per_icp_score,
  'qualification_doc',a.qualification_doc)),'{}') INTO projections FROM public.lab_arena_runs a WHERE a.round_id=p_round_id AND a.kind='execute';
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,
  transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(p_round_id,n.scoring_generation_floor,p_request_id,request_hash,p_actor_ref,txid_current(),pg_backend_pid(),
  to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at',projections,jsonb_build_object('operation','cancelled_evaluation_recovery_v1',
   'excluded_participants',excluded,'exclusion_proofs',p_excluded_participants));
 UPDATE public.lab_arena_rounds SET configuration_doc=n.configuration_doc,participants=n.participants,status=n.status,
  cancel_reason=n.cancel_reason,status_generation=n.status_generation,stage_generation=n.stage_generation,
  scoring_generation_floor=n.scoring_generation_floor,stage2_scoring_plan_doc=NULL,finalists=n.finalists WHERE round_id=p_round_id;
 UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL WHERE round_id=p_round_id AND kind='execute'
  AND EXISTS(SELECT 1 FROM jsonb_array_elements(keep)p WHERE p->>'submission_id'=lab_arena_runs.submission_id)
  AND (per_icp_score IS NOT NULL OR qualification_doc IS NOT NULL);
 queued:=public.lab_arena__queue_recovery_judges_v1(n,p_work_items);
 FOR source IN SELECT a.* FROM public.lab_arena_runs a WHERE a.round_id=p_round_id AND a.kind='execute' AND a.stage=2
   AND a.status='failed' AND a.terminal_cause='stage_closed' AND a.terminal_doc->>'previous_status'='pending'
   AND a.terminal_doc->'infrastructure_incomplete'='true'::JSONB AND a.attempt=1 AND a.lease_generation=0
   AND a.runner_hotkey IS NULL AND a.claim_request_id IS NULL AND a.lease_token_hash IS NULL
   AND a.result_doc IS NULL AND a.output_ref IS NULL AND NOT a.champion_restart_required
   AND EXISTS(SELECT 1 FROM jsonb_array_elements(keep)p WHERE p->>'submission_id'=a.submission_id)
   AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs later WHERE later.assignment_id=a.assignment_id AND later.attempt>a.attempt)
 LOOP
  INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,
   attempt,status,stage_generation,kind)
  VALUES(source.assignment_id||':2',source.assignment_id,p_round_id,source.submission_id,source.miner_hotkey,
   source.stage,source.icp_position,2,'pending',n.stage_generation,'execute');
  execute_count:=execute_count+1;
 END LOOP;
 RETURN jsonb_build_object('status','recovered','generation_floor',n.scoring_generation_floor,
   'queued_judges',queued,'queued_executions',execute_count,'excluded_participants',excluded,'request_hash',request_hash);
END;
$recover$;
ALTER FUNCTION public.lab_arena_recover_evaluation_v1(TEXT,JSONB,BIGINT,BIGINT,JSONB,JSONB,JSONB,TEXT,TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_recover_evaluation_v1(TEXT,JSONB,BIGINT,BIGINT,JSONB,JSONB,JSONB,TEXT,TEXT)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- Existing API: expose only public participant identities and the fixed reason.
CREATE OR REPLACE FUNCTION public.lab_arena_scoring_policy_history_v1(p_round_id TEXT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $history$
 SELECT jsonb_build_object('round_id',p_round_id,'revisions',coalesce(jsonb_agg(jsonb_build_object(
  'previous_generation_floor',coalesce((h.old_round_doc->>'scoring_generation_floor')::BIGINT,0),
  'generation_floor',h.generation_floor,'old_scorer_policy',h.old_round_doc#>'{configuration_doc,scorer_policy}',
  'new_scorer_policy',h.new_round_doc#>'{configuration_doc,scorer_policy}',
  'recovery',CASE WHEN h.recovery_doc IS NOT NULL THEN jsonb_build_object('operation',h.recovery_doc->>'operation',
    'excluded_participants',coalesce((SELECT jsonb_agg(jsonb_build_object('submission_id',p->>'submission_id',
      'miner_hotkey',p->>'miner_hotkey','status','provider_blocked','evaluation','not_evaluated',
      'reason','openrouter_zdr_route_unavailable')) FROM jsonb_array_elements(h.recovery_doc->'excluded_participants')p),'[]')) END
 ) ORDER BY h.generation_floor),'[]')) FROM public.lab_arena_scoring_supersessions h WHERE h.round_id=p_round_id;
$history$;
ALTER FUNCTION public.lab_arena_scoring_policy_history_v1(TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_scoring_policy_history_v1(TEXT) FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_scoring_policy_history_v1(TEXT) TO lab_arena_service;
DO $patch535_hold$
DECLARE d TEXT; before_metadata JSONB; target OID:='public.lab_arena_scoring_operator_hold_v1(text,jsonb,bigint,bigint,bigint,text,text,boolean)'::regprocedure;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,before_metadata FROM pg_proc p WHERE p.oid=target;
 IF pg_get_userbyid((SELECT proowner FROM pg_proc WHERE oid=target))<>'lab_arena_owner'
  OR (SELECT proacl::TEXT FROM pg_proc WHERE oid=target) IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}' THEN
  RAISE EXCEPTION 'recovery535 function owner differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='b459393e67c712a288e11acacbea1b3f8f946ab347eafaf894c5227612daaea2' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'7b62ff0a6ea6e3d157d446bfe9b4102a7e7fbd5012d308c00d33830dc58271c0' THEN
  RAISE EXCEPTION 'recovery535 hold preimage differs'; END IF;
 EXECUTE replace(d,$old535$r.status NOT IN ('stage2_scoring','stage2_judged','scored')$old535$,$new535$r.status NOT IN ('stage2','stage2_scoring','stage2_judged','scored','cancelled')$new535$);
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM before_metadata
   OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'b459393e67c712a288e11acacbea1b3f8f946ab347eafaf894c5227612daaea2' THEN
  RAISE EXCEPTION 'recovery535 hold postimage differs'; END IF;
END;
$patch535_hold$;

DO $patch535_ready$
DECLARE d TEXT; before_metadata JSONB; target OID:='public.lab_arena__all_judgments_ready_v1(text)'::regprocedure;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,before_metadata FROM pg_proc p WHERE p.oid=target;
 IF pg_get_userbyid((SELECT proowner FROM pg_proc WHERE oid=target))<>'lab_arena_owner'
  OR (SELECT proacl::TEXT FROM pg_proc WHERE oid=target) IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}' THEN
  RAISE EXCEPTION 'recovery535 function owner differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='b873cbbadd22825ddf4515b2c10248d558c92405d3112418452562d7ff2b728a' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'54e9f92db1a9000d95e8895a5834bc43970fdcbef975c987334b7df41b39a12c' THEN
  RAISE EXCEPTION 'recovery535 ready preimage differs'; END IF;
 EXECUTE replace(d,$old535$WHERE source.round_id=p_round_id AND source.kind='execute'$old535$,$new535$WHERE source.round_id=p_round_id AND source.kind='execute'
      AND EXISTS(SELECT 1 FROM public.lab_arena_rounds admitted_round,
        jsonb_array_elements(admitted_round.participants) admitted
        WHERE admitted_round.round_id=p_round_id AND admitted->>'submission_id'=source.submission_id)$new535$);
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM before_metadata
   OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'b873cbbadd22825ddf4515b2c10248d558c92405d3112418452562d7ff2b728a' THEN
  RAISE EXCEPTION 'recovery535 ready postimage differs'; END IF;
END;
$patch535_ready$;

DO $patch535_open$
DECLARE d TEXT; before_metadata JSONB; target OID:='public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::regprocedure;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,before_metadata FROM pg_proc p WHERE p.oid=target;
 IF pg_get_userbyid((SELECT proowner FROM pg_proc WHERE oid=target))<>'lab_arena_owner'
  OR (SELECT proacl::TEXT FROM pg_proc WHERE oid=target) IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}' THEN
  RAISE EXCEPTION 'recovery535 function owner differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='fc5402e96541655ae190e5b7827450724474eb9c0ee88748ee43ece5fe9838d7' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'e6d08167d9cad6df8444320972f6f6c26b97bbddccedb48473bd270d06d90b34' THEN
  RAISE EXCEPTION 'recovery535 open preimage differs'; END IF;
 EXECUTE replace(d,$old535$  v_generation := v_round.stage_generation + 1;$old535$,$new535$  -- cancelled_recovery535: current-floor roots keep their ordinary identity.
  IF EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      WHERE h.round_id=p_round_id AND h.generation_floor=v_round.scoring_generation_floor
        AND h.recovery_doc->>'operation'='cancelled_evaluation_recovery_v1'
        AND h.new_round_doc->'configuration_doc'=v_round.configuration_doc) THEN
    IF jsonb_array_length(p_work_items)<>(SELECT count(*) FROM public.lab_arena_runs a
        WHERE a.round_id=p_round_id AND a.stage=p_stage AND a.kind='execute' AND a.status='accepted'
          AND EXISTS(SELECT 1 FROM jsonb_array_elements(v_round.participants)p WHERE p->>'submission_id'=a.submission_id))
      OR jsonb_array_length(p_work_items)<>(SELECT count(DISTINCT x->>'scored_run_id') FROM jsonb_array_elements(p_work_items)x)
      OR EXISTS(SELECT 1 FROM jsonb_array_elements(p_work_items)x LEFT JOIN public.lab_arena_runs a
        ON a.run_id=x->>'scored_run_id' WHERE a.stage IS DISTINCT FROM p_stage) THEN
      RAISE EXCEPTION 'recovery scoring cohort incomplete' USING ERRCODE='22023'; END IF;
    v_generation := v_round.stage_generation + 1;
    v_round.stage_generation := v_generation;
    v_created := public.lab_arena__queue_recovery_judges_v1(v_round,p_work_items);
    UPDATE public.lab_arena_rounds SET status=v_next,status_generation=status_generation+1,
      stage_generation=v_generation WHERE round_id=p_round_id;
    RETURN jsonb_build_object('status','ok','round_status',v_next,
      'stage_generation',v_generation,'assignments',v_created,'reused',0);
  END IF;
  v_generation := v_round.stage_generation + 1;$new535$);
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM before_metadata
   OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'fc5402e96541655ae190e5b7827450724474eb9c0ee88748ee43ece5fe9838d7' THEN
  RAISE EXCEPTION 'recovery535 open postimage differs'; END IF;
END;
$patch535_open$;

DO $patch535_claim$
DECLARE d TEXT; before_metadata JSONB; target OID:='public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::regprocedure;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,before_metadata FROM pg_proc p WHERE p.oid=target;
 IF pg_get_userbyid((SELECT proowner FROM pg_proc WHERE oid=target))<>'lab_arena_owner'
  OR (SELECT proacl::TEXT FROM pg_proc WHERE oid=target) IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}' THEN
  RAISE EXCEPTION 'recovery535 function owner differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='0b86688d858e6a9a17b011e812541a53041099414df50b3e886a1ede07c35c01' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'ed57dcfcf6ded84f9c8ae44ee5dc390e89afc8ed9bc142e5a5ce4d3b21b7dadf' THEN
  RAISE EXCEPTION 'recovery535 claim preimage differs'; END IF;
 EXECUTE replace(replace(d,$old535$    AND runs.status = 'pending'
    -- Avoid credit-history proof$old535$,$new535$    AND runs.status = 'pending'
    -- Recovery preserves sourcing but restores the baseline-first judge barrier.
    AND (runs.kind<>'execute' OR runs.stage<>2 OR NOT EXISTS (
      SELECT 1 FROM public.lab_arena_scoring_supersessions recovery
      WHERE recovery.round_id=p_round_id
        AND recovery.generation_floor=v_round.scoring_generation_floor
        AND recovery.recovery_doc->>'operation'='cancelled_evaluation_recovery_v1'
        AND recovery.new_round_doc->'configuration_doc'=v_round.configuration_doc
    ) OR NOT EXISTS (
      SELECT 1 FROM public.lab_arena_runs baseline
      WHERE baseline.round_id=p_round_id AND baseline.kind='execute'
        AND baseline.status='accepted' AND baseline.stage=1
        AND EXISTS(SELECT 1 FROM jsonb_array_elements(v_round.participants) p
          WHERE p->>'submission_id'=baseline.submission_id AND p->'is_king'='true'::JSONB)
        AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs judge
          WHERE judge.round_id=p_round_id AND judge.kind='score'
            AND judge.scored_run_id=baseline.run_id
            AND judge.scoring_generation_floor=v_round.scoring_generation_floor
            AND judge.status='accepted' AND judge.terminal_cause='accepted'
            AND judge.output_ref IS NOT NULL AND judge.result_doc IS NOT NULL)
    ))
    -- Avoid credit-history proof$new535$),$old_response536$'stage_generation', v_round.stage_generation,$old_response536$,$new_response536$'stage_generation', CASE WHEN EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions recovery
      WHERE recovery.round_id=p_round_id AND recovery.generation_floor=v_round.scoring_generation_floor
        AND recovery.recovery_doc->>'operation'='cancelled_evaluation_recovery_v1'
        AND recovery.new_round_doc->'configuration_doc'=v_round.configuration_doc)
      THEN v_run.stage_generation ELSE v_round.stage_generation END,$new_response536$);
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM before_metadata
   OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'0b86688d858e6a9a17b011e812541a53041099414df50b3e886a1ede07c35c01' THEN
  RAISE EXCEPTION 'recovery535 claim postimage differs'; END IF;
END;
$patch535_claim$;

DO $patch535_open3$
DECLARE d TEXT; before_metadata JSONB; target OID:='public.lab_arena_open_scoring_v3(text,smallint,jsonb)'::regprocedure;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,before_metadata FROM pg_proc p WHERE p.oid=target;
 IF pg_get_userbyid((SELECT proowner FROM pg_proc WHERE oid=target))<>'lab_arena_owner'
  OR (SELECT proacl::TEXT FROM pg_proc WHERE oid=target) IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}' THEN
  RAISE EXCEPTION 'recovery535 function owner differs'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='51743119194087238e0b30e1f7803b1b5699950573d3215064b40c1844c9c42b' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'6ab7f8918fdb54b2a5042437c9f584a789ed9a4bafa0cd0dd4a6bc7c657aa142' THEN
  RAISE EXCEPTION 'recovery535 open3 preimage differs'; END IF;
 EXECUTE replace(d,$old535$  v_generation := v_round.stage_generation + 1;$old535$,$new535$  -- cancelled_recovery535: current-floor roots keep their ordinary identity.
  IF EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      WHERE h.round_id=p_round_id AND h.generation_floor=v_round.scoring_generation_floor
        AND h.recovery_doc->>'operation'='cancelled_evaluation_recovery_v1'
        AND h.new_round_doc->'configuration_doc'=v_round.configuration_doc) THEN
    IF jsonb_array_length(p_work_items)<>(SELECT count(*) FROM public.lab_arena_runs a
        WHERE a.round_id=p_round_id AND a.stage=p_stage AND a.kind='execute' AND a.status='accepted'
          AND EXISTS(SELECT 1 FROM jsonb_array_elements(v_round.participants)p WHERE p->>'submission_id'=a.submission_id))
      OR jsonb_array_length(p_work_items)<>(SELECT count(DISTINCT x->>'scored_run_id') FROM jsonb_array_elements(p_work_items)x)
      OR EXISTS(SELECT 1 FROM jsonb_array_elements(p_work_items)x LEFT JOIN public.lab_arena_runs a
        ON a.run_id=x->>'scored_run_id' WHERE a.stage IS DISTINCT FROM p_stage) THEN
      RAISE EXCEPTION 'recovery scoring cohort incomplete' USING ERRCODE='22023'; END IF;
    v_generation := v_round.stage_generation + 1;
    v_round.stage_generation := v_generation;
    v_created := public.lab_arena__queue_recovery_judges_v1(v_round,p_work_items);
    UPDATE public.lab_arena_rounds SET status=v_next,status_generation=status_generation+1,
      stage_generation=v_generation WHERE round_id=p_round_id;
    RETURN jsonb_build_object('status','ok','round_status',v_next,
      'stage_generation',v_generation,'assignments',v_created,'reused',0);
  END IF;
  v_generation := v_round.stage_generation + 1;$new535$);
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM before_metadata
   OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'51743119194087238e0b30e1f7803b1b5699950573d3215064b40c1844c9c42b' THEN
  RAISE EXCEPTION 'recovery535 open3 postimage differs'; END IF;
END;
$patch535_open3$;

DO $schema_restore536$
DECLARE before_doc JSONB:=current_setting('leadpoet.arena536_public_schema_before',TRUE)::JSONB;
 actual_doc JSONB;
BEGIN
 IF before_doc IS NULL THEN
  RAISE EXCEPTION 'Supersession schema permission snapshot missing' USING ERRCODE='55000';
 END IF;
 IF NOT (before_doc->>'owner_had_create')::BOOLEAN THEN
  REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
 END IF;
 SELECT jsonb_build_object('oid',oid,'owner',nspowner,'acl',to_jsonb(nspacl))
 INTO actual_doc FROM pg_namespace WHERE nspname='public';
 IF actual_doc IS DISTINCT FROM before_doc-'owner_had_create' THEN
  RAISE EXCEPTION 'Supersession schema permissions changed' USING ERRCODE='42501';
 END IF;
END;
$schema_restore536$;
NOTIFY pgrst,'reload schema';
COMMIT;
