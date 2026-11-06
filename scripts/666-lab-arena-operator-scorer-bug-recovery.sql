-- Explicit owner-only, finite recovery after a verified scorer correction.
-- Existing audit records, frozen grading/input, completed checkpoints and all
-- cumulative accounting remain authoritative. This creates no automatic retry.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena666_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $installed666$
DECLARE expected RECORD; actual RECORD;
BEGIN
 FOR expected IN SELECT * FROM (VALUES
 ('public.lab_arena__operator_scorer_bug_child_v1(public.lab_arena_runs)','52470b9f7bc12e4e7789c47513ab09e26bd4be49cbd684d0863f31172b7fc4de','s','sql'),
 ('public.lab_arena__operator_scorer_bug_host_v1(public.lab_arena_runs)','7cdaf78e4d0bc2a88eacf32e0142f2d4859a6f9d7bc06db438cb041205ebb749','s','sql'),
 ('public.lab_arena__operator_scorer_bug_claim_v1(public.lab_arena_rounds,public.lab_arena_runs)','f311303db711af3fd8ef0d0b85e34882429928eb67e933dd2d0fe29c8165296d','s','sql'),
 ('public.lab_arena_grant_scorer_bug_recovery_v1(text,text,text,text,text,text,boolean)','fa770960743e6136d9ce5461253c4f942607f261711a41ce1042376ccd4a4d95','v','plpgsql')
 ) functions(signature,source_sha256,volatility,language) LOOP
  SELECT p.*,l.lanname INTO actual FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
   WHERE p.oid=to_regprocedure(expected.signature);
  IF FOUND AND (encode(extensions.digest(actual.prosrc,'sha256'),'hex')<>expected.source_sha256
   OR pg_get_userbyid(actual.proowner)<>'lab_arena_owner' OR NOT actual.prosecdef
   OR actual.provolatile<>expected.volatility OR actual.lanname<>expected.language
   OR actual.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
   OR actual.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
   OR actual.proisstrict OR actual.proleakproof OR actual.proparallel<>'u') THEN
   RAISE EXCEPTION 'Scorer recovery666 installed function differs: %',expected.signature; END IF;
 END LOOP;
END;
$installed666$;

-- Immutable child identity: usable after completion for payer reconciliation.
CREATE OR REPLACE FUNCTION public.lab_arena__operator_scorer_bug_child_v1(p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $child666$
 SELECT coalesce(p_run.kind='score' AND EXISTS(
  SELECT 1 FROM public.lab_arena_scoring_supersessions h
  WHERE h.round_id=p_run.round_id
   AND h.recovery_doc->>'operation'='round_operational_update_v1'
   AND h.recovery_doc->>'recovery_kind'='operator_scorer_bug_recovery_v1'
   AND h.recovery_doc->'child_identity'=jsonb_build_object('run_id',p_run.run_id,'assignment_id',p_run.assignment_id,'round_id',p_run.round_id,'submission_id',p_run.submission_id,'miner_hotkey',p_run.miner_hotkey,'stage',p_run.stage,'icp_position',p_run.icp_position,'attempt',p_run.attempt,'scored_run_id',p_run.scored_run_id,'stage_generation',p_run.stage_generation,'scoring_generation_floor',p_run.scoring_generation_floor,'judgment_cache_key',p_run.judgment_cache_key,'judgment_input_hash',p_run.judgment_input_hash,'judgment_scope_doc',p_run.judgment_scope_doc,'company_judgment_refs',p_run.company_judgment_refs)),FALSE);
$child666$;
ALTER FUNCTION public.lab_arena__operator_scorer_bug_child_v1(public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__operator_scorer_bug_child_v1(public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena__operator_scorer_bug_host_v1(p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $host666$
 SELECT coalesce(public.lab_arena__operator_scorer_bug_child_v1(p_run) AND EXISTS(
  SELECT 1 FROM public.lab_arena_scoring_supersessions h
  JOIN public.lab_arena_runs parent ON parent.run_id=h.recovery_doc->>'parent_run_id'
  WHERE h.round_id=p_run.round_id
   AND h.recovery_doc->>'recovery_kind'='operator_scorer_bug_recovery_v1'
   AND h.recovery_doc->'child_identity'=jsonb_build_object('run_id',p_run.run_id,'assignment_id',p_run.assignment_id,'round_id',p_run.round_id,'submission_id',p_run.submission_id,'miner_hotkey',p_run.miner_hotkey,'stage',p_run.stage,'icp_position',p_run.icp_position,'attempt',p_run.attempt,'scored_run_id',p_run.scored_run_id,'stage_generation',p_run.stage_generation,'scoring_generation_floor',p_run.scoring_generation_floor,'judgment_cache_key',p_run.judgment_cache_key,'judgment_input_hash',p_run.judgment_input_hash,'judgment_scope_doc',p_run.judgment_scope_doc,'company_judgment_refs',p_run.company_judgment_refs)
   AND h.recovery_doc->'organizer_openrouter_authorized'='true'::JSONB
   AND public.lab_arena__nov06_organizer_openrouter665_v1(parent)),FALSE);
$host666$;
ALTER FUNCTION public.lab_arena__operator_scorer_bug_host_v1(public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__operator_scorer_bug_host_v1(public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- Only claim authority is live-state dependent; payer identity above is durable.
CREATE OR REPLACE FUNCTION public.lab_arena__operator_scorer_bug_claim_v1(
 p_round public.lab_arena_rounds,p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $claim666$
 SELECT coalesce(public.lab_arena__operator_scorer_bug_child_v1(p_run)
  AND p_run.scoring_generation_floor=p_round.scoring_generation_floor
  AND p_round.status NOT IN('published','cancelled') AND p_round.published_at IS NULL
  AND p_round.publication_doc IS NULL AND p_round.reward_activated_at IS NULL
  AND clock_timestamp()<(p_round.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ
  AND clock_timestamp()<(p_round.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs done
    WHERE done.assignment_id=p_run.assignment_id AND done.status='accepted')
  AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
   JOIN public.lab_arena_reviewer_runtimes m ON
    m.grading_image_digest=p_round.configuration_doc->>'scorer_image_digest'
    AND m.grading_image_reference=p_round.configuration_doc->>'scorer_image_reference'
    AND m.runtime_image_digest=h.recovery_doc->>'runtime_image_digest'
    AND m.runtime_image_reference=h.recovery_doc->>'runtime_image_reference'
   WHERE h.round_id=p_run.round_id
    AND h.recovery_doc->>'recovery_kind'='operator_scorer_bug_recovery_v1'
    AND h.recovery_doc->'child_identity'=jsonb_build_object('run_id',p_run.run_id,'assignment_id',p_run.assignment_id,'round_id',p_run.round_id,'submission_id',p_run.submission_id,'miner_hotkey',p_run.miner_hotkey,'stage',p_run.stage,'icp_position',p_run.icp_position,'attempt',p_run.attempt,'scored_run_id',p_run.scored_run_id,'stage_generation',p_run.stage_generation,'scoring_generation_floor',p_run.scoring_generation_floor,'judgment_cache_key',p_run.judgment_cache_key,'judgment_input_hash',p_run.judgment_input_hash,'judgment_scope_doc',p_run.judgment_scope_doc,'company_judgment_refs',p_run.company_judgment_refs)),FALSE);
$claim666$;
ALTER FUNCTION public.lab_arena__operator_scorer_bug_claim_v1(public.lab_arena_rounds,public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__operator_scorer_bug_claim_v1(public.lab_arena_rounds,public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_grant_scorer_bug_recovery_v1(
 p_parent_run_id TEXT,p_parent_sha256 TEXT,p_runtime_image_digest TEXT,
 p_request_id TEXT,p_actor_ref TEXT,p_correction_ref TEXT,p_dry_run BOOLEAN DEFAULT TRUE)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $grant666$
DECLARE parent public.lab_arena_runs; child public.lab_arena_runs; r public.lab_arena_rounds;
 mapping public.lab_arena_reviewer_runtimes;
 prior public.lab_arena_scoring_supersessions; revision BIGINT; proof JSONB; request_hash TEXT;
 checkpoints JSONB; signal_checkpoints JSONB; parent_hash TEXT; response JSONB; organizer BOOLEAN;
BEGIN
 IF p_parent_run_id IS NULL OR p_parent_sha256 !~ '^sha256:[0-9a-f]{64}$'
  OR p_runtime_image_digest !~ '^sha256:[0-9a-f]{64}$'
  OR p_request_id IS NULL OR length(p_request_id) NOT BETWEEN 1 AND 160
  OR p_actor_ref IS NULL OR length(p_actor_ref) NOT BETWEEN 1 AND 200
  OR p_correction_ref IS NULL OR length(p_correction_ref) NOT BETWEEN 1 AND 500
  OR p_dry_run IS NULL THEN RAISE EXCEPTION 'scorer bug recovery request invalid'; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT parent FROM public.lab_arena_runs WHERE run_id=p_parent_run_id;
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=parent.round_id FOR UPDATE;
 SELECT * INTO prior FROM public.lab_arena_scoring_supersessions
  WHERE round_id=r.round_id AND request_id=p_request_id;
 IF FOUND THEN
  IF prior.recovery_doc->>'recovery_kind'<>'operator_scorer_bug_recovery_v1'
   OR prior.recovery_doc->>'parent_run_id' IS DISTINCT FROM p_parent_run_id
   OR prior.recovery_doc->>'parent_sha256' IS DISTINCT FROM p_parent_sha256
   OR prior.recovery_doc->>'runtime_image_digest' IS DISTINCT FROM p_runtime_image_digest
   OR prior.actor_ref IS DISTINCT FROM p_actor_ref
   OR prior.recovery_doc->>'correction_ref' IS DISTINCT FROM p_correction_ref THEN
   RAISE EXCEPTION 'scorer bug recovery replay differs'; END IF;
  SELECT * INTO STRICT child FROM public.lab_arena_runs WHERE run_id=prior.recovery_doc#>>'{child_identity,run_id}';
  IF NOT public.lab_arena__operator_scorer_bug_child_v1(child) THEN
   RAISE EXCEPTION 'scorer bug recovery replay child differs'; END IF;
  RETURN jsonb_build_object('status','granted','idempotent',TRUE,'request_id',p_request_id,
   'child_identity',prior.recovery_doc->'child_identity','checkpoints',prior.recovery_doc->'checkpoints',
   'signal_checkpoints',prior.recovery_doc->'signal_checkpoints',
   'runtime_image_digest',p_runtime_image_digest,'organizer_openrouter_authorized',
   prior.recovery_doc->'organizer_openrouter_authorized');
 END IF;
 -- Use the normal operational lock order. The failed parent is immutable;
 -- no active lease, round mutation, drain or restart is needed for this append.
 IF current_setting('session_replication_role')<>'origin' THEN
  RAISE EXCEPTION 'scorer bug recovery requires ordinary protected writes'; END IF;
 SELECT * INTO STRICT parent FROM public.lab_arena_runs WHERE run_id=p_parent_run_id FOR UPDATE;
 parent_hash:='sha256:'||encode(extensions.digest(to_jsonb(parent)::TEXT,'sha256'),'hex');
 IF parent_hash IS DISTINCT FROM p_parent_sha256 OR parent.kind<>'score' OR parent.status<>'failed'
  OR parent.terminal_cause<>'judge_error' OR parent.result_doc->>'terminal_status' IS DISTINCT FROM 'judge_error'
  OR parent.result_doc#>>'{failure_diagnostic,stage}' IS DISTINCT FROM 'scorer'
  OR parent.result_doc#>>'{failure_diagnostic,error_class}' IS DISTINCT FROM 'judge_error'
  OR parent.scoring_generation_floor<>r.scoring_generation_floor
  OR public.lab_arena__operator_scorer_bug_child_v1(parent)
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.assignment_id=parent.assignment_id
     AND (a.attempt>parent.attempt OR a.status='accepted'))
  OR NOT public.lab_arena__judge_calls_clear_v1(parent.assignment_id) THEN
  RAISE EXCEPTION 'scorer bug recovery failed parent, latest state or clear calls differs'; END IF;
 IF (r.published_at IS NULL AND r.publication_doc IS NULL AND r.reward_activated_at IS NULL
  AND r.status NOT IN('published','cancelled')
  AND clock_timestamp()<(r.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ
  AND clock_timestamp()<(r.configuration_doc#>>'{schedule,publication_deadline}')::TIMESTAMPTZ) IS NOT TRUE THEN
  RAISE EXCEPTION 'scorer bug recovery round or deadline differs'; END IF;
 SELECT * INTO STRICT mapping FROM public.lab_arena_reviewer_runtimes
  WHERE grading_image_digest=r.configuration_doc->>'scorer_image_digest'
   AND grading_image_reference=r.configuration_doc->>'scorer_image_reference' FOR SHARE;
 IF mapping.runtime_image_digest IS DISTINCT FROM p_runtime_image_digest
  OR mapping.runtime_image_digest=coalesce(parent.reviewer_runtime_doc->>'runtime_image_digest',
       r.configuration_doc->>'scorer_image_digest') THEN
  RAISE EXCEPTION 'scorer bug recovery corrected selected runtime differs'; END IF;
 child:=parent; child.run_id:=parent.assignment_id||':'||(parent.attempt+1)::TEXT;
 child.attempt:=parent.attempt+1; child.stage_generation:=r.stage_generation;
 IF NOT public.lab_arena__judge_continuation_v1(r,child) THEN
  RAISE EXCEPTION 'scorer bug recovery frozen continuation binding differs'; END IF;
 organizer:=public.lab_arena__nov06_organizer_openrouter665_v1(parent);
 SELECT coalesce(jsonb_object_agg(j.cache_key||':'||j.authority_slot::TEXT,encode(extensions.digest(to_jsonb(j)::TEXT,'sha256'),'hex')),'{}')
  INTO checkpoints FROM public.lab_arena_company_judgments j
  WHERE EXISTS(SELECT 1 FROM jsonb_array_elements(coalesce(parent.company_judgment_refs,'[]')) ref
    WHERE ref->>'cache_key'=j.cache_key);
 SELECT coalesce(jsonb_object_agg(j.cache_key||':'||j.authority_slot::TEXT,encode(extensions.digest(to_jsonb(j)::TEXT,'sha256'),'hex')),'{}')
  INTO signal_checkpoints FROM public.lab_arena_signal_judgments j
  WHERE j.source_scored_run_id=parent.scored_run_id;
 proof:=jsonb_build_object('operation','round_operational_update_v1',
  'recovery_kind','operator_scorer_bug_recovery_v1','parent_run_id',parent.run_id,'parent_sha256',parent_hash,
  'correction_ref',p_correction_ref,'scoring_generation_floor',r.scoring_generation_floor,
  'runtime_image_digest',mapping.runtime_image_digest,'runtime_image_reference',mapping.runtime_image_reference,
  'organizer_openrouter_authorized',organizer,'child_identity',jsonb_build_object('run_id',child.run_id,'assignment_id',child.assignment_id,'round_id',child.round_id,'submission_id',child.submission_id,'miner_hotkey',child.miner_hotkey,'stage',child.stage,'icp_position',child.icp_position,'attempt',child.attempt,'scored_run_id',child.scored_run_id,'stage_generation',child.stage_generation,'scoring_generation_floor',child.scoring_generation_floor,'judgment_cache_key',child.judgment_cache_key,'judgment_input_hash',child.judgment_input_hash,'judgment_scope_doc',child.judgment_scope_doc,'company_judgment_refs',child.company_judgment_refs),
  'checkpoints',checkpoints,'signal_checkpoints',signal_checkpoints);
 response:=jsonb_build_object('status',CASE WHEN p_dry_run THEN 'dry_run' ELSE 'granted' END,
  'idempotent',FALSE,'request_id',p_request_id,'child_identity',proof->'child_identity',
  'checkpoints',checkpoints,'signal_checkpoints',signal_checkpoints,
  'runtime_image_digest',mapping.runtime_image_digest,'organizer_openrouter_authorized',organizer);
 IF p_dry_run THEN RETURN response; END IF;
 request_hash:='sha256:'||encode(extensions.digest(proof::TEXT,'sha256'),'hex');
 SELECT coalesce(max(generation_floor),0)+1 INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,
  transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(r.round_id,revision,p_request_id,request_hash,p_actor_ref,txid_current(),pg_backend_pid(),
  to_jsonb(r)-'updated_at',to_jsonb(r)-'updated_at','{}',proof);
 INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,icp_position,attempt,
  status,stage_generation,scoring_generation_floor,kind,scored_run_id,previous_runner_hotkey,judgment_cache_key,
  judgment_input_hash,judgment_scope_doc,judgment_group_leader,judgment_group_miner_hotkeys,company_judgment_refs)
 VALUES(child.run_id,parent.assignment_id,parent.round_id,parent.submission_id,parent.miner_hotkey,parent.stage,
  parent.icp_position,child.attempt,'pending',r.stage_generation,r.scoring_generation_floor,'score',parent.scored_run_id,
  parent.runner_hotkey,parent.judgment_cache_key,parent.judgment_input_hash,parent.judgment_scope_doc,
  parent.judgment_group_leader,parent.judgment_group_miner_hotkeys,parent.company_judgment_refs);
 RETURN response;
END;
$grant666$;
ALTER FUNCTION public.lab_arena_grant_scorer_bug_recovery_v1(TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,BOOLEAN) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_grant_scorer_bug_recovery_v1(TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,BOOLEAN)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- Shared function patches follow; exact665 source/security is required.
DO $patch666$
DECLARE target RECORD; patch RECORD; function_oid OID; d TEXT; updated TEXT; metadata JSONB; identity JSONB;
BEGIN
 FOR target IN SELECT * FROM (VALUES
 ('public.lab_arena_provider_funding(text,text)','3fc9adbb3808515dd61674e89669821b18798d93543c936db79a7ac0de099d1c','6a8070de71d9d9d9a523e4e78f32cbd7c6c742188629d40e543c505f8bd02c7c','s'),
 ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)','819ca4df91b5648c055525c34eb40ed0cf488e6c74ab32ab87417661f693c7d4','5e13436bc113afb45e501a7f8e35f24309ee8a0e24af4779f0233a7bcd042529','v'),
 ('public.lab_arena_complete_attempt(text,text,jsonb,text,text)','d052adb5936f51d965aae574f61a567e7779417765aa197eac56efbb76f64f65','00562ddcea9dc11ecf70a116174f26adcf031cd301ea9126cc80a0b2123ab0a3','v'),
 ('public.lab_arena_expire_leases(text)','13418dc69e3f20877334cbdfc1fbafcd99db7d4697be92ba42dc78ae197edd13','b752c7a25489d2d438a2203bd8658f8b3c8b0c75e2325ee701dd8c817418129d','v')
 ) expected(signature,preimage,postimage,volatility) LOOP
  function_oid:=to_regprocedure(target.signature);
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',jsonb_build_array(
   pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
  INTO d,metadata,identity FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=function_oid;
  IF d IS NULL OR identity IS DISTINCT FROM jsonb_build_array('lab_arena_owner',
   '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,
   target.volatility,ARRAY['search_path=pg_catalog, public']::TEXT[],'plpgsql') THEN
   RAISE EXCEPTION 'Scorer recovery666 function security differs: %',target.signature; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')=target.postimage THEN CONTINUE; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>target.preimage THEN
   RAISE EXCEPTION 'Scorer recovery666 function source differs: %',target.signature; END IF;
  updated:=d;
  FOR patch IN SELECT * FROM (VALUES
 ('public.lab_arena_provider_funding(text,text)',$old666$  v_source := CASE$old666$,$new666$  v_source := CASE
    WHEN p_provider='openrouter'
      AND public.lab_arena__operator_scorer_bug_host_v1(v_run) THEN 'host'$new666$),
 ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)',$old666$      OR public.lab_arena__nov06_organizer_openrouter665_v1(runs))$old666$,$new666$      OR public.lab_arena__nov06_organizer_openrouter665_v1(runs)
      OR public.lab_arena__operator_scorer_bug_claim_v1(v_round,runs))
    AND (NOT public.lab_arena__operator_scorer_bug_child_v1(runs)
      OR public.lab_arena__operator_scorer_bug_claim_v1(v_round,runs))$new666$),
 ('public.lab_arena_complete_attempt(text,text,jsonb,text,text)',$old666$     AND NOT public.lab_arena__nov06_organizer_openrouter665_v1(v_run)$old666$,$new666$     AND NOT public.lab_arena__nov06_organizer_openrouter665_v1(v_run)
     AND NOT public.lab_arena__operator_scorer_bug_child_v1(v_run)$new666$),
 ('public.lab_arena_expire_leases(text)',$old666$    IF NOT public.lab_arena__nov06_organizer_openrouter665_v1(v_run)$old666$,$new666$    IF NOT public.lab_arena__nov06_organizer_openrouter665_v1(v_run)
      AND NOT public.lab_arena__operator_scorer_bug_child_v1(v_run)$new666$),
 ('public.lab_arena_expire_leases(text)',$old666$      AND NOT public.lab_arena__nov06_organizer_openrouter665_v1(candidate)$old666$,$new666$      AND NOT public.lab_arena__nov06_organizer_openrouter665_v1(candidate)
      AND NOT public.lab_arena__operator_scorer_bug_child_v1(candidate)$new666$)
  ) changes(signature,old_text,new_text) WHERE signature=target.signature LOOP
   IF length(updated)-length(replace(updated,patch.old_text,''))<>length(patch.old_text) THEN
    RAISE EXCEPTION 'Scorer recovery666 function anchor differs: %',target.signature; END IF;
   updated:=replace(updated,patch.old_text,patch.new_text);
  END LOOP;
  IF encode(extensions.digest(updated,'sha256'),'hex')<>target.postimage THEN
   RAISE EXCEPTION 'Scorer recovery666 function postimage differs: %',target.signature; END IF;
  EXECUTE updated;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=function_oid) IS DISTINCT FROM metadata
   OR encode(extensions.digest(pg_get_functiondef(function_oid),'sha256'),'hex')<>target.postimage THEN
   RAISE EXCEPTION 'Scorer recovery666 function readback differs: %',target.signature; END IF;
 END LOOP;
END;
$patch666$;
DO $restore666$
BEGIN
 IF NOT (SELECT had_create FROM pg_temp.arena666_schema_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.arena666_schema_acl) THEN
  RAISE EXCEPTION 'Scorer recovery666 schema ACL differs'; END IF;
END;
$restore666$;
NOTIFY pgrst,'reload schema';
COMMIT;
